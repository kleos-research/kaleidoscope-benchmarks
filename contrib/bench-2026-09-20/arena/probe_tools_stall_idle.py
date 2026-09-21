"""Third reproducer. The payload is ruled out: the harness's exact request replays in 1.9s,
2 of 2. So what is left is WHEN the call is made.

In the harness every agent call follows a long gap -- a writer call (15-90s), a kscope search,
a judge call -- so the pooled HTTPS connection has been idle for a minute or more. In both
probes so far the calls were 2 seconds apart and the connection never went cold. Every observed
stall was ~300s, which is exactly `common/llm.py`'s client timeout, and it always hit the FIRST
agent call of a step.

Hypothesis: a keep-alive connection the gateway has already dropped is reused, and the client
waits out the full read timeout instead of failing fast.

This sends the harness's own captured request after increasing idle gaps and times each one.
`openai` is configured with max_retries=4, so a stall that the SDK retries would show up as a
long-but-successful call; one that raises shows up as a failure with its exception class.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
os.environ.setdefault("BENCH_SCOPE", "arena-probe")
os.environ.setdefault("BENCH_SCOPE_CAP_USD", "1.00")
os.environ["BENCH_TAG"] = "idlestall-probe"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "MemoryArena"))
sys.path.insert(0, str(BASE / "MemoryArena" / "env" / "env_systems"))

from formal_reasoning_env import llm_backend  # noqa: E402

OUT = BASE / "logs" / "arena_idlestall_probe.jsonl"
GAPS = [0, 0, 60, 150, 300]   # seconds of idleness BEFORE each call


class Captured(Exception):
    pass


def capture() -> dict:
    from agent.math import MathAgent
    import agent.math as math_mod

    holder = {}

    class Stub:
        def __init__(self, role):
            self.chat = type("C", (), {"completions": self})()

        def create(self, **request):
            holder["request"] = copy.deepcopy(request)
            raise Captured()

    real = math_mod.GuardedClient
    math_mod.GuardedClient = Stub
    try:
        agent = MathAgent(model_name="gpt-5-mini", temperature=0.0, max_tokens=8192, backend="openai")
        row = json.loads(open(BASE / "data/memoryarena__formal_reasoning_math__data.jsonl").readline())
        prompt = agent.build_prompt(task=row["questions"][0], background=row["backgrounds"][0])
        try:
            agent.act("<memory_context>\nNone\n</memory_context>\nUser: " + prompt)
        except Captured:
            pass
    finally:
        math_mod.GuardedClient = real
    return holder["request"]


def main() -> int:
    request = capture()
    client = llm_backend.GuardedClient(role="probe")
    print("the harness's captured tool request, sent after increasing idle gaps")
    print(f"{'idle before':<14}{'latency':<12}{'outcome'}")
    print("-" * 60)
    rows = []
    for gap in GAPS:
        if gap:
            time.sleep(gap)      # inside the script: the connection goes cold, nothing else runs
        started = time.time()
        row = {"idle_before_s": gap, "started": round(started, 3)}
        try:
            response = client.chat.completions.create(**dict(request))
            row.update(ok=True, seconds=round(time.time() - started, 1),
                       finish=response.choices[0].finish_reason)
        except BaseException as exc:
            row.update(ok=False, seconds=round(time.time() - started, 1),
                       error_class=type(exc).__name__, error=str(exc)[:200])
        rows.append(row)
        print(f"{gap:<14}{row['seconds']:<12.1f}"
              + (f"ok finish={row['finish']}" if row["ok"] else "FAILED " + row["error_class"]),
              flush=True)
        with open(OUT, "a") as handle:
            handle.write(json.dumps(row) + "\n")

    warm = [r["seconds"] for r in rows if r["idle_before_s"] == 0]
    cold = [(r["idle_before_s"], r["seconds"], r["ok"]) for r in rows if r["idle_before_s"] > 0]
    print("\nwarm (back-to-back):", warm)
    print("after an idle gap   :", cold)
    slow = [c for c in cold if c[1] > 60]
    print("\nVERDICT:", "idle gap reproduces the stall" if slow
          else "idle gap did NOT reproduce it in this many trials -- the stall stays intermittent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
