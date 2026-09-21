"""Smallest reproducer for the stall that made MemoryArena's smoke uninterpretable.

Observed: an agent request carrying `tools=` sat for 320s and 454s while, on the same API key
at the same moment, 20 plain chat completions from another run returned in 1.5-39s. This asks
what actually distinguishes the two, with the smallest request that can show it.

Four arms, same tiny prompt, interleaved so a gateway-wide slowdown cannot masquerade as a
tools effect:
  plain_none   no tools, reasoning_effort "none"     <- control
  plain_high   no tools, reasoning_effort "high"     <- control for the effort field
  tools_none   tools,    reasoning_effort "none"     <- what the harness actually sends
  tools_high   tools,    reasoning_effort "high"     <- the combination the gateway documents as refused

Everything goes through common/llm.py (the only door) and through the instrumented
GuardedClient, so a stall here also demonstrates the new failure row end to end.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
os.environ.setdefault("BENCH_SCOPE", "arena-probe")
os.environ.setdefault("BENCH_SCOPE_CAP_USD", "0.50")
os.environ["BENCH_TAG"] = "toolstall-probe"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "MemoryArena"))
sys.path.insert(0, str(BASE / "MemoryArena" / "env" / "env_systems"))

from formal_reasoning_env import llm_backend  # noqa: E402

OUT = BASE / "logs" / "arena_toolstall_probe.jsonl"
PROMPT = [{"role": "user", "content": "What is 6*7? Answer with the number only."}]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "reasoning",
        "description": "Step-by-step reasoning.",
        "parameters": {"type": "object",
                       "properties": {"request": {"type": "string"}},
                       "required": ["request"]},
    },
}]
ARMS = {
    "plain_none": {"effort": "none", "tools": False},
    "plain_high": {"effort": "high", "tools": False},
    "tools_none": {"effort": "none", "tools": True},
    "tools_high": {"effort": "high", "tools": True},
}


def one(arm: str, spec: dict, trial: int) -> dict:
    request = {"model": "ignored-the-door-pins-it", "messages": PROMPT,
               "max_completion_tokens": 64, "reasoning_effort": spec["effort"]}
    if spec["tools"]:
        request["tools"] = TOOLS
        request["tool_choice"] = "auto"
    client = llm_backend.GuardedClient(role="probe")
    started = time.time()
    row = {"arm": arm, "trial": trial, "effort": spec["effort"], "tools": spec["tools"],
           "started": round(started, 3)}
    try:
        response = client.chat.completions.create(**request)
        row.update(ok=True, seconds=round(time.time() - started, 1),
                   finish=response.choices[0].finish_reason,
                   out_tokens=response.usage.completion_tokens,
                   text=(response.choices[0].message.content or "")[:60])
    except BaseException as exc:
        row.update(ok=False, seconds=round(time.time() - started, 1),
                   error_class=type(exc).__name__, error=str(exc)[:300])
    print(f"  {arm:<11} trial {trial}  {row['seconds']:7.1f}s  "
          f"{'ok finish=' + str(row.get('finish')) if row['ok'] else 'FAILED ' + row['error_class']}",
          flush=True)
    with open(OUT, "a") as handle:
        handle.write(json.dumps(row) + "\n")
    return row


def main() -> int:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rows = []
    print(f"interleaved, {trials} trials per arm; client timeout is 300s (common/llm.py)")
    for trial in range(1, trials + 1):
        for arm, spec in ARMS.items():
            rows.append(one(arm, spec, trial))

    print("\n" + "=" * 78)
    print(f"{'arm':<12}{'n':<4}{'ok':<4}{'failed':<8}{'median s':<11}{'max s':<9}{'errors'}")
    print("-" * 78)
    for arm in ARMS:
        sel = [r for r in rows if r["arm"] == arm]
        good = [r for r in sel if r["ok"]]
        bad = [r for r in sel if not r["ok"]]
        secs = [r["seconds"] for r in sel]
        errs = sorted({r["error_class"] for r in bad})
        print(f"{arm:<12}{len(sel):<4}{len(good):<4}{len(bad):<8}"
              f"{statistics.median(secs):<11.1f}{max(secs):<9.1f}{','.join(errs) or '-'}")
    print(f"\nrows written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
