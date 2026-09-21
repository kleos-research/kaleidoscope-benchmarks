#!/usr/bin/env python
"""Descending ladder to find the reader deployment's real INPUT limit.

The gateway publishes no context window (probe_context_limit.py: `models.retrieve` carries a
`capabilities` block that names only which APIs the model serves). A 1.5M-token request came back
429, not 400 -- the rate limiter answers before the length check -- so the ladder starts below the
per-minute token budget and walks down.

Reading the outcomes:
  * 400 with `context_length_exceeded` (or a message naming a maximum): the limit, stated by the
    server, and the request is NOT billed.
  * 200: the prompt fitted, so the limit is at or above that size. This one IS billed for its input.
  * 429: the rate limiter, not the model. Retried after the backoff the response asks for; the
    x-ratelimit-* headers are recorded because the concurrency plan needs them anyway.

Every call goes through common/llm.py, so the ledger and the cap see it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from common import llm  # noqa: E402

OUT = BASE / "logs" / "mab_context_ladder.json"
RATE_HEADERS = ("x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
                "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
                "retry-after", "retry-after-ms", "x-ratelimit-reset-tokens")


def headers_of(exc) -> dict:
    response = getattr(exc, "response", None)
    raw = getattr(response, "headers", None)
    if not raw:
        return {}
    return {k: raw.get(k) for k in RATE_HEADERS if raw.get(k) is not None}


def probe(text: str, encoder, tag: str) -> dict:
    sent = len(encoder.encode(text, disallowed_special=()))
    for attempt in range(1, 7):
        started = time.time()
        try:
            _, usage = llm.chat([{"role": "user", "content": text}], role="probe", effort="low",
                                max_completion_tokens=16, tag=tag)
            return {"prompt_tokens_local": sent, "accepted": True, "usage_in": usage["in"],
                    "finish": usage["finish"], "seconds": round(time.time() - started, 1), "attempt": attempt}
        except llm.BudgetExceeded:
            raise
        except Exception as exc:
            message = str(exc)
            status = getattr(exc, "status_code", None)
            record = {"prompt_tokens_local": sent, "accepted": False, "status": status,
                      "error": message[:1500], "headers": headers_of(exc),
                      "seconds": round(time.time() - started, 1), "attempt": attempt}
            if status == 429 and attempt < 6:
                wait = 20 * attempt
                print(f"    429 at {sent:,} tokens, waiting {wait}s (headers {record['headers']})")
                time.sleep(wait)
                continue
            return record
    return {"prompt_tokens_local": sent, "accepted": False, "status": "gave_up"}


def main() -> int:
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
    sizes = [int(x) for x in (sys.argv[1:] or ["500000", "400000", "330000", "300000", "280000"])]
    findings = {"ts": time.time(), "deployment": llm.DEPLOYMENT, "probes": []}

    words = [f"w{i:07d}" for i in range(max(sizes))]
    joined = " ".join(words)
    per_word = len(encoder.encode(joined, disallowed_special=())) / len(words)
    print(f"filler: {per_word:.2f} tokens per word")

    for size in sizes:
        take = max(1, int(size / per_word))
        text = " ".join(words[:take]) + "\n\nReply with the word ok."
        print(f"probe ~{size:,} tokens ...", flush=True)
        row = probe(text, encoder, tag=f"fullctx:ladder:{size}")
        row["asked"] = size
        findings["probes"].append(row)
        verdict = "ACCEPTED" if row.get("accepted") else f"refused {row.get('status')}"
        print(f"  {row['prompt_tokens_local']:,} tokens -> {verdict}")
        if not row.get("accepted"):
            print("   ", (row.get("error") or "")[:700].replace("\n", " "))
        if row.get("headers"):
            print("    headers:", row["headers"])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(findings, indent=1, default=str))
        if row.get("accepted"):
            print("  -> the limit is at or above this size; ladder stops")
            break

    print(f"\nwritten {OUT}")
    print(llm.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
