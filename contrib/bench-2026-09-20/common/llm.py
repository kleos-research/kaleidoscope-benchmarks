"""The one door to the Azure gateway: a spend ledger and a hard cap in front of every call.

The gateway returns no cost, so a run with no price reports itself as free --
the same defect as a counter reading zero because nothing is wired to it. Prices
here are ASSUMED (GPT-5-class list prices, deliberately on the high side) and are
printed with every total so nobody mistakes the estimate for a bill. Override
with BENCH_PRICE_IN / BENCH_PRICE_OUT (USD per million tokens).

The cap is checked BEFORE each call under a file lock, so N concurrent workers
can overshoot by at most N calls. BENCH_SCOPE tags every row; BENCH_SCOPE_CAP_USD
bounds one scope inside the global BENCH_CAP_USD.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
ENV_FILE = Path("/path/to/kaleidoscope/.env")
LEDGER = ROOT / "spend" / "ledger.jsonl"
LOCK = ROOT / "spend" / ".lock"

DEPLOYMENT = os.environ.get("BENCH_DEPLOYMENT", "gpt-5.6-luna")
# Checked online 2026-09-20. Microsoft's Foundry announcement lists GPT-5.6 Luna, Global Standard,
# short context, at $0.20 input / $0.02 cached input / $1.20 output per Mtok -- the same as OpenAI's
# list price after the 80% cut of 2026-07-30. A Microsoft Q&A thread reports Data Zone deployments
# still BILLED at $1.10 / $6.60 in August 2026. Which applies depends on the deployment type, which
# this machine cannot see. So the CAP is enforced at the worst reported rate (it protects real
# money), and every report shows the list-price figure beside it.
LIST_PRICE = (0.20, 1.20)
PRICE_IN = float(os.environ.get("BENCH_PRICE_IN", "1.10"))
PRICE_OUT = float(os.environ.get("BENCH_PRICE_OUT", "6.60"))
# Owner's instruction 2026-09-20: a safety cap of $200 PER BENCHMARK ("put a safety cap of two
# hundred for each"), so the global ceiling is the two scopes together and each scope defaults to
# 200 unless BENCH_SCOPE_CAP_USD says otherwise. It was 15 while the harness was unproven.
CAP_USD = float(os.environ.get("BENCH_CAP_USD", "400"))
DEFAULT_SCOPE_CAP_USD = "200"

_client = None
_temperature_ok = True


class BudgetExceeded(RuntimeError):
    """Raised instead of making a call. Never caught-and-continued by a harness."""


def _load_env() -> None:
    # setdefault: a real export wins. Values are never printed or logged.
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _load_env()
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        _client = OpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            base_url=f"{endpoint}/openai/v1/",
            # **The 300s "stall" was this number.** Measured 2026-09-20: requests
            # carrying `tools=` intermittently sat at ~300s or a multiple of it
            # (17% of arena agent steps), while 20 concurrent tool-free calls on
            # the same key finished in 1.5-39s. Both a 304.9s and a 605.4s call
            # RETURNED SUCCESSFULLY on the SDK's own retry -- so it is the client
            # giving up and re-asking, not a server hang, and a shorter deadline
            # buys a faster recovery rather than a lost call.
            #
            # 90 rather than the 30-60 the diagnosis suggested, because a
            # legitimate tool call was observed completing at **211s**; cutting
            # below that would convert real work into retries. It is an env knob
            # because the right value is a measurement nobody has taken yet:
            # sweep it against the new `degraded_steps` counter and require zero.
            timeout=float(os.environ.get("BENCH_LLM_TIMEOUT", "90")),
            max_retries=4,
        )
    return _client


@contextmanager
def _locked():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def totals() -> dict:
    out = {"all": 0.0, "calls": 0, "in": 0, "out": 0, "reasoning": 0}
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            cost = row["in"] * PRICE_IN / 1e6 + row["out"] * PRICE_OUT / 1e6
            out["all"] += cost
            out["calls"] += 1
            out["in"] += row["in"]
            out["out"] += row["out"]
            out["reasoning"] += row.get("reasoning") or 0
            out[row["scope"]] = out.get(row["scope"], 0.0) + cost
    return out


def _guard(scope: str) -> None:
    spent = totals()
    if spent["all"] >= CAP_USD:
        raise BudgetExceeded(f"global cap ${CAP_USD:.2f} reached (est ${spent['all']:.2f})")
    scope_cap = os.environ.get("BENCH_SCOPE_CAP_USD", DEFAULT_SCOPE_CAP_USD)
    if scope_cap and spent.get(scope, 0.0) >= float(scope_cap):
        raise BudgetExceeded(f"scope '{scope}' cap ${float(scope_cap):.2f} reached (est ${spent[scope]:.2f})")


def chat(messages, *, role: str, effort: str = "high", max_completion_tokens: int = 8000,
         temperature: float | None = None, tag: str = "") -> tuple[str, dict]:
    """One completion. Returns (text, usage). Raises BudgetExceeded before spending past a cap."""
    global _temperature_ok
    scope = os.environ.get("BENCH_SCOPE", "unscoped")
    with _locked():
        _guard(scope)
    request = {
        "model": DEPLOYMENT,
        "messages": messages,
        "reasoning_effort": effort,
        # A reasoning model counts its thinking against this field, and Azure
        # refuses the old `max_tokens` spelling outright.
        "max_completion_tokens": max_completion_tokens,
    }
    if temperature is not None and _temperature_ok:
        request["temperature"] = temperature
    started = time.time()
    try:
        response = client().chat.completions.create(**request)
    except Exception as exc:  # one retry, only for a refusal that names temperature
        if "temperature" in request and "temperature" in str(exc).lower():
            _temperature_ok = False
            request.pop("temperature")
            response = client().chat.completions.create(**request)
        else:
            raise
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    row = {
        "ts": round(time.time(), 3), "scope": scope, "role": role, "tag": tag,
        "model_asked": DEPLOYMENT, "model_served": getattr(response, "model", None),
        "effort": effort, "temperature_sent": "temperature" in request,
        "in": usage.prompt_tokens, "out": usage.completion_tokens, "reasoning": reasoning,
        "finish": response.choices[0].finish_reason,
        "ms": int((time.time() - started) * 1000),
        "est_usd": usage.prompt_tokens * PRICE_IN / 1e6 + usage.completion_tokens * PRICE_OUT / 1e6,
        "price_assumed": [PRICE_IN, PRICE_OUT],
    }
    with _locked():
        with open(LEDGER, "a") as handle:
            handle.write(json.dumps(row) + "\n")
    return (response.choices[0].message.content or ""), row


def create_raw(request: dict, *, role: str, tag: str = ""):
    """A guarded pass-through for callers that need the full API (tools, response_format, ...).

    The harness keeps its own request shape; this door only (a) checks the cap,
    (b) pins the deployment, (c) renames `max_tokens`, which Azure refuses for a
    reasoning model, (d) drops `temperature` if the deployment rejects it, and
    (e) writes the ledger row. Returns the raw SDK response.
    """
    global _temperature_ok
    scope = os.environ.get("BENCH_SCOPE", "unscoped")
    with _locked():
        _guard(scope)
    request = dict(request)
    request["model"] = DEPLOYMENT
    if "max_tokens" in request:
        request.setdefault("max_completion_tokens", request.pop("max_tokens"))
    request.setdefault("reasoning_effort", os.environ.get("BENCH_EFFORT", "high"))
    if not _temperature_ok:
        request.pop("temperature", None)
    started = time.time()
    try:
        response = client().chat.completions.create(**request)
    except Exception as exc:
        if "temperature" in request and "temperature" in str(exc).lower():
            _temperature_ok = False
            request.pop("temperature")
            response = client().chat.completions.create(**request)
        else:
            raise
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    row = {
        "ts": round(time.time(), 3), "scope": scope, "role": role, "tag": tag,
        "model_asked": DEPLOYMENT, "model_served": getattr(response, "model", None),
        "effort": request.get("reasoning_effort"), "temperature_sent": "temperature" in request,
        "in": usage.prompt_tokens, "out": usage.completion_tokens,
        "reasoning": getattr(details, "reasoning_tokens", None) if details else None,
        "finish": response.choices[0].finish_reason, "ms": int((time.time() - started) * 1000),
        "est_usd": usage.prompt_tokens * PRICE_IN / 1e6 + usage.completion_tokens * PRICE_OUT / 1e6,
        "price_assumed": [PRICE_IN, PRICE_OUT],
    }
    with _locked():
        with open(LEDGER, "a") as handle:
            handle.write(json.dumps(row) + "\n")
    return response


def report() -> str:
    t = totals()
    scopes = {k: round(v, 3) for k, v in t.items() if k not in ("all", "calls", "in", "out", "reasoning")}
    at_list = t["in"] * LIST_PRICE[0] / 1e6 + t["out"] * LIST_PRICE[1] / 1e6
    return (f"calls={t['calls']} in={t['in']:,} out={t['out']:,} (reasoning={t['reasoning']:,}) | "
            f"at Azure list price ${LIST_PRICE[0]}/${LIST_PRICE[1]} per Mtok: ${at_list:.3f} | "
            f"at the worst reported Azure billing ${PRICE_IN}/${PRICE_OUT} (what the ${CAP_USD:.0f} cap is checked against): "
            f"${t['all']:.3f}; by scope {scopes}")
