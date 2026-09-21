#!/usr/bin/env python
"""Establish the reader deployment's REAL context window, rather than assuming one.

Three questions, in the order that costs least:

  1. Does the gateway publish it?  `GET /openai/v1/models` and `/models/<deployment>` are free.
     Azure's model objects sometimes carry a `capabilities` block naming the input limit.
  2. Does an over-long prompt name the limit in its refusal?  A request the server rejects for
     length is NOT billed and its error text usually reads
     "maximum context length is N tokens, however you requested M". That single free call is the
     cheapest direct measurement there is.
  3. If neither answers, the caller bisects (probe_accept.py), which IS billed for input.

Nothing here is written down as a constant: the answer is whatever this prints.
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

OUT = BASE / "logs" / "mab_context_limit_probe.json"


def filler(tokens: int, encoder) -> str:
    """A string of about `tokens` tokens. Distinct words, so nothing collapses in a tokenizer and
    nothing reads as an instruction."""
    words = [f"w{i:07d}" for i in range(tokens)]
    text = " ".join(words)
    n = len(encoder.encode(text, disallowed_special=()))
    if n > tokens:
        text = " ".join(words[: max(1, int(len(words) * tokens / n))])
    return text


def main() -> int:
    findings = {"ts": time.time(), "deployment": llm.DEPLOYMENT, "steps": []}

    # -- 1. what the gateway publishes ---------------------------------------------------------
    try:
        rows = []
        for model in llm.client().models.list():
            rows.append(model.model_dump() if hasattr(model, "model_dump") else dict(model))
        matched = [r for r in rows if llm.DEPLOYMENT in json.dumps(r, default=str)]
        findings["steps"].append({"step": "models.list", "ok": True, "n": len(rows), "rows": matched[:5]})
        print(f"models.list: {len(rows)} entries, {len(matched)} naming the deployment")
        for row in matched[:5]:
            print("  match:", json.dumps(row, default=str)[:700])
    except Exception as exc:
        findings["steps"].append({"step": "models.list", "ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]})
        print(f"models.list failed: {type(exc).__name__}: {str(exc)[:200]}")

    try:
        model = llm.client().models.retrieve(llm.DEPLOYMENT)
        row = model.model_dump() if hasattr(model, "model_dump") else dict(model)
        findings["steps"].append({"step": "models.retrieve", "ok": True, "row": row})
        print("models.retrieve:", json.dumps(row, default=str)[:900])
    except Exception as exc:
        findings["steps"].append({"step": "models.retrieve", "ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]})
        print(f"models.retrieve failed: {type(exc).__name__}: {str(exc)[:200]}")

    # -- 2. an over-long prompt, whose refusal is free and usually names the limit --------------
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
    for probe_tokens in (1_500_000,):
        text = filler(probe_tokens, encoder)
        sent = len(encoder.encode(text, disallowed_special=()))
        started = time.time()
        try:
            _, usage = llm.chat([{"role": "user", "content": text + "\n\nReply with the word ok."}],
                                role="probe", max_completion_tokens=16, tag="fullctx:overlong")
            findings["steps"].append({"step": "overlong", "probe_tokens": sent, "accepted": True,
                                      "usage_in": usage["in"], "seconds": round(time.time() - started, 1)})
            print(f"overlong probe {sent:,} tokens was ACCEPTED (in={usage['in']:,}) -- limit is above this")
        except llm.BudgetExceeded:
            raise
        except Exception as exc:
            message = str(exc)
            findings["steps"].append({"step": "overlong", "probe_tokens": sent, "accepted": False,
                                      "error": f"{type(exc).__name__}: {message}"[:2000],
                                      "seconds": round(time.time() - started, 1)})
            print(f"overlong probe {sent:,} tokens REFUSED after {time.time() - started:.0f}s:")
            print("   ", message[:1200].replace("\n", " "))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(findings, indent=1, default=str))
    print(f"\nwritten {OUT}")
    print(llm.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
