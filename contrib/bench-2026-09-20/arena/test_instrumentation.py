"""Prove the new instrumentation can go red, without spending anything.

Three things must hold, and each is checked by making the thing happen:
  1. a raising call writes a ledger row AND a failures-log row, and re-raises unchanged;
  2. the exception still reaches `agent/math.py`'s handler, which flags the step and counts it;
  3. a BudgetStop is NOT recorded as a gateway failure (it never reached the gateway).

No LLM call: `common.llm.create_raw` is monkeypatched to raise. `common/*` is not modified.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
os.environ["BENCH_SCOPE"] = "instrument-selftest"
os.environ["BENCH_TAG"] = "instrument-selftest/tools-stall"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "MemoryArena"))
sys.path.insert(0, str(BASE / "MemoryArena" / "env" / "env_systems"))

from common import llm as bench_llm  # noqa: E402
from formal_reasoning_env import llm_backend  # noqa: E402

LEDGER = BASE / "spend" / "ledger.jsonl"
FAILURES = BASE / "logs" / "arena_llm_failures.jsonl"


def tail(path: Path) -> dict | None:
    if not path.exists():
        return None
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


class FakeTimeout(Exception):
    """Stands in for openai.APITimeoutError: same shape, no network."""


def main() -> int:
    ok = True
    before_ledger = len([l for l in LEDGER.read_text().splitlines() if l.strip()])
    before_fail = len(FAILURES.read_text().splitlines()) if FAILURES.exists() else 0

    # ---- 1. a raising call is recorded and re-raised -------------------------------------
    def boom(request, *, role, tag=""):
        raise FakeTimeout("Request timed out.")

    real = bench_llm.create_raw
    bench_llm.create_raw = boom
    client = llm_backend.GuardedClient(role="agent")
    raised = None
    try:
        client.chat.completions.create(model="x", messages=[{"role": "user", "content": "hi"}],
                                       tools=[{"type": "function"}], temperature=0.0)
    except FakeTimeout as exc:
        raised = exc
    finally:
        bench_llm.create_raw = real

    row = tail(LEDGER)
    frow = tail(FAILURES)
    checks = [
        ("exception propagated unchanged", isinstance(raised, FakeTimeout)),
        ("one ledger row added", len([l for l in LEDGER.read_text().splitlines() if l.strip()]) == before_ledger + 1),
        ("one failures row added", (len(FAILURES.read_text().splitlines()) if FAILURES.exists() else 0) == before_fail + 1),
        ("ledger row marked failed", bool(row and row.get("failed") is True)),
        ("ledger row names the error", bool(row and row.get("finish") == "error:FakeTimeout")),
        ("ledger row records tools=True", bool(row and row.get("tools") is True)),
        ("ledger row costs nothing", bool(row and row["in"] == 0 and row["out"] == 0 and row["est_usd"] == 0.0)),
        ("ledger row keeps every key old readers need",
         bool(row and all(k in row for k in ("ts", "scope", "role", "tag", "in", "out", "est_usd", "finish", "ms", "reasoning")))),
        ("failures row is the same row", bool(frow and frow.get("ts") == (row or {}).get("ts"))),
    ]

    # ---- 2. the agent flags and counts the degraded step ---------------------------------
    from agent.math import MathAgent

    agent = MathAgent.__new__(MathAgent)          # no backend, no network
    agent.backend_name = "openai"
    agent.degraded_steps = 0
    agent.last_tool_path = None
    agent.model_name, agent.temperature, agent.max_tokens = "x", 0.0, 10
    agent._act_with_tools = lambda prompt: (_ for _ in ()).throw(FakeTimeout("Request timed out."))
    agent._reasoning_with_errors = lambda prompt: ("42", [])
    action = agent.act("PROBLEM: what is 6*7?")
    checks += [
        ("agent still returned an answer", bool(action.get("answer") == "42")),
        ("agent counted the degraded step", agent.degraded_steps == 1),
        ("agent flagged the step", (agent.last_tool_path or {}).get("status") == "degraded_no_tools"),
        ("agent kept the error text", "FakeTimeout" in str((agent.last_tool_path or {}).get("error"))),
        ("agent recorded how long the stall took", isinstance((agent.last_tool_path or {}).get("seconds"), float)),
    ]

    # ---- 3. a budget stop is NOT a gateway failure ---------------------------------------
    n_before = len([l for l in LEDGER.read_text().splitlines() if l.strip()])

    def capped(request, *, role, tag=""):
        raise bench_llm.BudgetExceeded("scope cap reached")

    bench_llm.create_raw = capped
    try:
        client.chat.completions.create(model="x", messages=[], tools=[{"type": "function"}])
    except llm_backend.BudgetStop:
        pass
    finally:
        bench_llm.create_raw = real
    checks.append(("budget stop wrote no failure row",
                   len([l for l in LEDGER.read_text().splitlines() if l.strip()]) == n_before))

    for name, passed in checks:
        print(("  PASS  " if passed else "  FAIL  ") + name)
        ok &= bool(passed)
    print(("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED") + f"  ({sum(p for _, p in checks)}/{len(checks)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
