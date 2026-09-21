"""Read-only analysis of an arena smoke run: scores, served counts, memory fate, cost.

Reads only what the run wrote: the adapter log, the driver's task_map, the harness's own
result.jsonl files and the spend ledger. Makes no LLM call and opens no vault.

MemoryArena has no per-question gold memory the way MemoryAgentBench does, so "was the needed
memory served" is not defined identically. What IS defined, and is what this prints: at each
search the vault holds exactly the records of the earlier subtasks of the same paper, and every
one of them has one of three fates -- served, omitted by a named kscope reason, or never a
candidate at all (neither served nor omitted).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
LIST_IN, LIST_OUT = 0.20, 1.20      # Azure Global Standard list, checked 2026-09-20
PESS_IN, PESS_OUT = 1.10, 6.60      # worst reported Azure Data Zone billing


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main(run_id: str) -> None:
    out = BASE / "results" / "arena" / run_id
    tasks = jsonl(out / "task_map.jsonl")
    adapter = jsonl(BASE / "logs" / f"arena_adapter-{run_id}.jsonl")
    ledger = [r for r in jsonl(BASE / "spend" / "ledger.jsonl")
              if str(r.get("tag", "")).startswith(run_id + "/")]
    by_user = defaultdict(list)
    for r in adapter:
        by_user[r["user_id"]].append(r)

    print("=" * 100)
    print(f"RUN {run_id}: {len(tasks)} tasks")
    print("=" * 100)

    arm_scores = defaultdict(list)
    print(f"\n{'arm':<12}{'task':<10}{'paper':<14}{'status':<12}{'q':<4}{'correct':<10}{'served per q'}")
    print("-" * 100)
    for t in sorted(tasks, key=lambda x: (x["arm"], x["domain"], x["id"])):
        res = jsonl(out / t["domain"] / t["arm"] / t["paper_name"] / "result.jsonl")
        correct = [bool(r.get("is_correct")) for r in res]
        arm_scores[t["arm"]] += correct
        user = (t.get("new_vaults") or [None])[0]
        searches = [r for r in sorted(by_user.get(user, []), key=lambda r: r["ts"])
                    if r["op"] == "wrap_user_prompt"]
        served = [s.get("served") for s in searches]
        print(f"{t['arm']:<12}{t['domain'] + ':' + str(t['id']):<10}{t['paper_name']:<14}"
              f"{t['status']:<12}{t['subtasks']:<4}{str(sum(correct)) + '/' + str(len(correct)):<10}{served}")

    print(f"\n{'arm':<14}{'score':<14}{'accuracy'}")
    print("-" * 44)
    for arm, sc in sorted(arm_scores.items()):
        print(f"{arm:<14}{str(sum(sc)) + '/' + str(len(sc)):<14}{(sum(sc) / len(sc) if sc else 0):.3f}")

    # ---------------------------------------------------------------- memory fate
    print("\n" + "=" * 100)
    print("MEMORY FATE, per question, kscope arm (corpus = records written by earlier steps)")
    print("=" * 100)
    print(f"{'task':<10}{'q':<4}{'corpus':<8}{'served':<8}{'from_q':<14}{'cut':<5}{'reasons':<34}{'never_cand':<11}{'floor'}")
    print("-" * 100)
    fate = Counter()
    for t in sorted(tasks, key=lambda x: (x["domain"], x["id"])):
        if t["arm"] != "kscope":
            continue
        user = (t.get("new_vaults") or [None])[0]
        rows = sorted(by_user.get(user, []), key=lambda r: r["ts"])
        written = {r["memory_id"]: r["chunk_index"] for r in rows
                   if r["op"] == "add_chunk" and r.get("memory_id")}
        q = -1
        for r in rows:
            if r["op"] != "wrap_user_prompt":
                continue
            q += 1
            corpus = q + 1                      # chunk 0 (seed) .. chunk q
            srv = r.get("served") or 0
            cut = r.get("omitted_hits") or 0
            never = corpus - srv - cut
            idxs = [written.get(m) for m in (r.get("served_memory_ids") or [])]
            # chunk 0 is the harness's seed; chunk k>=1 is subtask k-1's record
            from_q = [i - 1 for i in idxs if i is not None and i >= 1]
            fate["served"] += srv
            fate["cut"] += cut
            fate["never_candidate"] += max(never, 0)
            floor = (r.get("served_floor") or {}).get("similarity_floor")
            print(f"{t['domain'] + ':' + str(t['id']):<10}{q:<4}{corpus:<8}{srv:<8}{str(from_q):<14}{cut:<5}"
                  f"{str(r.get('omission_reasons') or {})[:33]:<34}{never:<11}{floor}")
    total = sum(fate.values())
    print("-" * 100)
    print(f"TOTAL memory-slots across all kscope searches: {total} | " +
          " | ".join(f"{k}={v} ({v / total:.1%})" for k, v in fate.items()) if total else "no searches")

    # ---------------------------------------------------------------- cost
    print("\n" + "=" * 100)
    print("COST (gateway returns no cost; every dollar below is tokens x an ASSUMED rate)")
    print("=" * 100)
    by_arm = defaultdict(lambda: Counter())
    for r in ledger:
        arm = r["tag"].split("/")[1]
        by_arm[arm]["calls"] += 1
        by_arm[arm][r["role"]] += 1
        by_arm[arm]["in"] += r["in"]
        by_arm[arm]["out"] += r["out"]
        by_arm[arm]["reasoning"] += r.get("reasoning") or 0
    n_tasks = Counter(t["arm"] for t in tasks)
    n_q = Counter()
    for t in tasks:
        n_q[t["arm"]] += t["subtasks"]
    print(f"{'arm':<13}{'tasks':<7}{'q':<5}{'calls':<7}{'/task':<7}{'/q':<6}{'in_tok':<10}{'out_tok':<10}"
          f"{'list $':<9}{'pess $':<9}{'roles'}")
    print("-" * 110)
    tot = Counter()
    for arm, c in sorted(by_arm.items()):
        key = arm if arm in n_tasks else ("kscope_none" if arm == "kscope_none" else arm)
        nt, nq = n_tasks.get(key, 0), n_q.get(key, 0)
        li = c["in"] * LIST_IN / 1e6 + c["out"] * LIST_OUT / 1e6
        pe = c["in"] * PESS_IN / 1e6 + c["out"] * PESS_OUT / 1e6
        roles = {k: v for k, v in c.items() if k in ("writer", "agent", "judge")}
        print(f"{arm:<13}{nt:<7}{nq:<5}{c['calls']:<7}{(c['calls'] / nt if nt else 0):<7.1f}"
              f"{(c['calls'] / nq if nq else 0):<6.1f}{c['in']:<10,}{c['out']:<10,}{li:<9.4f}{pe:<9.4f}{roles}")
        tot.update(c)
    li = tot["in"] * LIST_IN / 1e6 + tot["out"] * LIST_OUT / 1e6
    pe = tot["in"] * PESS_IN / 1e6 + tot["out"] * PESS_OUT / 1e6
    print("-" * 110)
    print(f"{'RUN TOTAL':<13}{sum(n_tasks.values()):<7}{sum(n_q.values()):<5}{tot['calls']:<7}"
          f"{'':<7}{'':<6}{tot['in']:<10,}{tot['out']:<10,}{li:<9.4f}{pe:<9.4f}"
          f"reasoning={tot['reasoning']:,}")

    slow = sorted(ledger, key=lambda r: -(r.get("ms") or 0))[:5]
    print("\nslowest calls in this run:")
    for r in slow:
        print(f"  {r['ms'] / 1000:8.1f}s  role={r['role']:<7} tag={r['tag']:<32} in={r['in']:<6} "
              f"out={r['out']:<6} finish={r['finish']}")

    # ------------------------------------------------- silent timeout / degraded agent path
    # A failed LLM call writes NO ledger row: common/llm.py appends only after a response
    # returns. agent/math.py:296 wraps the tool-calling path in `except Exception` and falls
    # back to a plain no-tool reasoning call. So a gateway stall shows up ONLY as a gap
    # between the search that ends a step and the first agent row that follows it.
    print("\n" + "=" * 100)
    print("SILENT GATEWAY STALLS (a failed call leaves no ledger row; the agent then drops its tool path)")
    print("=" * 100)
    agent_rows = sorted([r for r in ledger if r["role"] == "agent"], key=lambda r: r["ts"])
    stalls, steps = [], 0
    for t in sorted(tasks, key=lambda x: (x["arm"], x["domain"], x["id"])):
        user = (t.get("new_vaults") or [None])[0]
        for r in sorted(by_user.get(user, []), key=lambda x: x["ts"]):
            if r["op"] != "wrap_user_prompt":
                continue
            steps += 1
            nxt = next((a for a in agent_rows if a["ts"] - a["ms"] / 1000 >= r["ts"] - 1), None)
            if nxt is None:
                continue
            gap = (nxt["ts"] - nxt["ms"] / 1000) - r["ts"]
            if gap > 120:
                stalls.append((t["arm"], f"{t['domain']}:{t['id']}", round(gap, 1)))
    if stalls:
        for arm, task, gap in stalls:
            print(f"  {arm:<13}{task:<10}{gap:8.1f}s of dead time before the first agent call "
                  f"-> that step's tool-calling path timed out and was silently replaced")
    print(f"  {len(stalls)} of {steps} agent steps stalled >120s with no ledger row to show for it")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smoke-20260920c")
