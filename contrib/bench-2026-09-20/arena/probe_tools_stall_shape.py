"""Second reproducer: a bare `tools=` request returns in 1.4s, so `tools=` alone is not it.

This captures the EXACT request `agent/math.py:_act_with_tools` builds -- by letting the real
agent build it against a real dataset row and intercepting it before it is sent -- then replays
it and three one-change variants of it. Whatever single field turns a 1.4s call into a 300s
timeout is the difference between these arms.

  harness_exact      what the harness sends, replayed verbatim
  plus_max_tokens    + max_completion_tokens=8192 (the harness sends NO output cap at all)
  no_temperature     - temperature (the harness sends temperature=0.0)
  tiny_prompt        harness shape, but the 20-token prompt that already returns in 1.4s
"""
from __future__ import annotations

import copy
import json
import os
import statistics
import sys
import time
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
os.environ.setdefault("BENCH_SCOPE", "arena-probe")
os.environ.setdefault("BENCH_SCOPE_CAP_USD", "1.00")
os.environ["BENCH_TAG"] = "toolstall-probe2"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "MemoryArena"))
sys.path.insert(0, str(BASE / "MemoryArena" / "env" / "env_systems"))

from formal_reasoning_env import llm_backend  # noqa: E402

OUT = BASE / "logs" / "arena_toolstall_probe2.jsonl"


class Captured(Exception):
    pass


def capture_harness_request() -> dict:
    """Let the real MathAgent build its real first tool call; steal the request, send nothing."""
    from agent.math import MathAgent

    holder = {}

    class Stub:
        def __init__(self, role):
            self.chat = type("C", (), {"completions": self})()

        def create(self, **request):
            holder["request"] = copy.deepcopy(request)
            raise Captured("captured")

    real = llm_backend.GuardedClient
    import agent.math as math_mod
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


def send(arm: str, request: dict, trial: int) -> dict:
    client = llm_backend.GuardedClient(role="probe")
    started = time.time()
    row = {"arm": arm, "trial": trial, "started": round(started, 3),
           "has_tools": bool(request.get("tools")),
           "has_max_completion_tokens": "max_completion_tokens" in request,
           "has_temperature": "temperature" in request,
           "prompt_chars": sum(len(str(m.get("content") or "")) for m in request.get("messages") or [])}
    try:
        response = client.chat.completions.create(**request)
        row.update(ok=True, seconds=round(time.time() - started, 1),
                   finish=response.choices[0].finish_reason,
                   in_tokens=response.usage.prompt_tokens, out_tokens=response.usage.completion_tokens)
    except BaseException as exc:
        row.update(ok=False, seconds=round(time.time() - started, 1),
                   error_class=type(exc).__name__, error=str(exc)[:200])
    print(f"  {arm:<17} trial {trial}  {row['seconds']:7.1f}s  "
          + (f"ok finish={row.get('finish')} out={row.get('out_tokens')}" if row["ok"]
             else "FAILED " + row["error_class"]), flush=True)
    with open(OUT, "a") as handle:
        handle.write(json.dumps(row) + "\n")
    return row


def main() -> int:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    base = capture_harness_request()
    print("captured harness request:")
    print("  keys              :", sorted(base))
    print("  messages          :", len(base.get("messages") or []),
          "totalling", sum(len(str(m.get('content') or '')) for m in base['messages']), "chars")
    print("  tools             :", bool(base.get("tools")), "| tool_choice:", base.get("tool_choice"))
    print("  temperature       :", base.get("temperature"))
    print("  reasoning_effort  :", base.get("reasoning_effort", "(unset -> door default 'high')"))
    print("  output cap        :", base.get("max_completion_tokens", "NONE -- the harness sets no cap"))
    print()

    tiny = dict(base)
    tiny["messages"] = [{"role": "user", "content": "What is 6*7? Answer with the number only."}]
    capped = dict(base)
    capped["max_completion_tokens"] = 8192
    notemp = {k: v for k, v in base.items() if k != "temperature"}
    arms = {"harness_exact": base, "plus_max_tokens": capped,
            "no_temperature": notemp, "tiny_prompt": tiny}

    rows = []
    for trial in range(1, trials + 1):
        for arm, req in arms.items():
            rows.append(send(arm, dict(req), trial))

    print("\n" + "=" * 86)
    print(f"{'arm':<18}{'n':<4}{'ok':<4}{'failed':<8}{'median s':<11}{'max s':<9}{'errors'}")
    print("-" * 86)
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm]
        secs = [r["seconds"] for r in sel]
        errs = sorted({r["error_class"] for r in sel if not r["ok"]})
        print(f"{arm:<18}{len(sel):<4}{sum(r['ok'] for r in sel):<4}{sum(not r['ok'] for r in sel):<8}"
              f"{statistics.median(secs):<11.1f}{max(secs):<9.1f}{','.join(errs) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
