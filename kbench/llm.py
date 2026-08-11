"""One OpenAI client, one retry policy, one spend ledger.

Every model call in this repo goes through `complete()`. That is deliberate:
cost, latency and failure handling are properties of the harness, not of
whichever module happens to be calling.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .config import settings

# Per million tokens. Override with KBENCH_PRICE_IN / KBENCH_PRICE_OUT if you are
# running against a different model or endpoint — the ledger is a convenience,
# not a billing record, and it says so in the report.
DEFAULT_PRICE_IN = 2.00
DEFAULT_PRICE_OUT = 8.00

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class Completion:
    text: str
    usage: Usage
    latency_ms: float
    model: str
    error: str | None = None
    attempts: int = 1


@dataclass
class Spend:
    """Token and cost totals, split by stage so a report can attribute them."""

    by_stage: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    tokens_in: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tokens_out: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, stage: str, usage: Usage, price_in: float, price_out: float) -> float:
        cost = (usage.prompt_tokens * price_in + usage.completion_tokens * price_out) / 1e6
        # Locked because every stage of this harness is concurrent and `+=` on a
        # float is a load, an add and a store.
        with self._lock:
            self.by_stage[stage] += cost
            self.tokens_in[stage] += usage.prompt_tokens
            self.tokens_out[stage] += usage.completion_tokens
        return cost

    @property
    def total(self) -> float:
        with self._lock:
            return sum(self.by_stage.values())

    def line(self) -> str:
        parts = ", ".join(f"{k} ${v:.3f}" for k, v in sorted(self.by_stage.items()))
        return f"${self.total:.3f} ({parts})" if parts else "$0.000"


_client: OpenAI | None = None
_client_lock = threading.Lock()


def client() -> OpenAI:
    """Built once, under a lock, on first use.

    Not at import: constructing a client validates the key, and a harness that
    reaches the network merely to be imported cannot report "this run spent
    nothing" honestly.
    """
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(
                api_key=settings.require_api_key(),
                base_url=settings.base_url,
                timeout=300.0,
                max_retries=0,  # retries are ours, so they are visible in `attempts`
            )
    return _client


def complete(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    stage: str = "unspecified",
    spend: Spend | None = None,
    response_format: dict[str, Any] | None = None,
) -> Completion:
    """One chat completion, with bounded retries and a recorded cost.

    Transport failures retry with exponential backoff; a refusal does not. The
    attempt count travels on the result so a run can report how much of its
    latency was retrying.
    """
    started = time.time()
    last_error: str | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if response_format is not None:
                kwargs["response_format"] = response_format

            response = client().chat.completions.create(**kwargs)
            raw = response.usage
            details = getattr(raw, "prompt_tokens_details", None)
            usage = Usage(
                prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
                cached_tokens=getattr(details, "cached_tokens", 0) or 0 if details else 0,
            )
            if spend is not None:
                spend.add(stage, usage, DEFAULT_PRICE_IN, DEFAULT_PRICE_OUT)
            return Completion(
                text=(response.choices[0].message.content or "").strip(),
                usage=usage,
                latency_ms=(time.time() - started) * 1000.0,
                model=model,
                attempts=attempt,
            )
        except Exception as exc:  # noqa: BLE001 — the SDK raises a wide family
            last_error = f"{type(exc).__name__}: {exc}"
            status = getattr(exc, "status_code", None)
            retryable = status in RETRYABLE_STATUS or status is None
            if attempt == MAX_ATTEMPTS or not retryable:
                break
            time.sleep(min(2 ** attempt, 16))

    return Completion(
        text="",
        usage=Usage(),
        latency_ms=(time.time() - started) * 1000.0,
        model=model,
        error=last_error,
        attempts=MAX_ATTEMPTS,
    )


def parse_json_object(text: str) -> dict:
    """Parse a model's JSON reply, tolerating a fenced block around it.

    Deliberately narrow: it strips one leading and trailing fence and parses. It
    does not attempt to repair malformed JSON, because a repaired extraction is
    an invented one and this harness would rather record a failure.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 2:
            body = lines[1:]
            if body and body[-1].strip().startswith("```"):
                body = body[:-1]
            candidate = "\n".join(body).strip()
    return json.loads(candidate)
