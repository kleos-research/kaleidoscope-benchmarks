"""The two repaired clauses of check_smoke.py must be able to go red. This feeds check_smoke.main
the real files of an old run through a patched `jsonl` loader that plants one contamination at
a time in the adapter log, and requires the matching assertion to fail. No LLM call, no vault.

usage: test_check_smoke_can_go_red.py <run_id>   (a completed run with a kscope task)"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
sys.path.insert(0, str(BASE / "arena"))
import check_smoke  # noqa: E402

real_jsonl = check_smoke.jsonl


def run(run_id: str, mutate) -> dict:
    def patched(path: Path):
        rows = real_jsonl(path)
        if path.name == f"arena_adapter-{run_id}.jsonl":
            rows = mutate(copy.deepcopy(rows))
        return rows
    check_smoke.jsonl = patched
    try:
        return check_smoke.main(run_id)["assertions"]
    finally:
        check_smoke.jsonl = real_jsonl


def main(run_id: str) -> int:
    base = run(run_id, lambda rows: rows)
    kscope_users = sorted({r["user_id"] for r in base and real_jsonl(BASE / "logs" / f"arena_adapter-{run_id}.jsonl") if r["writer"] == "kscope"})
    u = kscope_users[0]

    def foreign_answer(rows):  # chunk 1 of the first kscope task carries another answer under the right question
        for r in rows:
            if r["user_id"] == u and r["op"] == "add_chunk" and r["chunk_index"] == 1:
                r["chunk"] = r["chunk"].replace("## solution: ", "## solution: SOMEBODY ELSE'S ANSWER ", 1)
        return rows

    def judge_line(rows):  # what judge_result_in_memory=True would have written
        for r in rows:
            if r["user_id"] == u and r["op"] == "add_chunk" and r["chunk_index"] == 1:
                r["chunk"] = r["chunk"] + "## Judge: CORRECT\n"
        return rows

    def gold_appended(rows):  # the gold answer of subtask 0 pasted into the chunk
        gold = None
        for t in check_smoke.jsonl(BASE / "results" / "arena" / run_id / "task_map.jsonl"):
            if (t.get("new_vaults") or [None])[0] == u:
                gold = str(check_smoke.jsonl(check_smoke.DATA[t["domain"]])[t["id"]]["answers"][0])
        for r in rows:
            if r["user_id"] == u and r["op"] == "add_chunk" and r["chunk_index"] == 1:
                r["chunk"] = r["chunk"] + "## hint: " + gold + "\n"
        return rows

    checks = [
        ("baseline: assertion 4 passes on the real files", base["4_no_leaks"]["pass"]),
        ("foreign answer under the right question -> assertion 1's resolved-ownership clause red",
         any(v["unresolved_indexes"] for v in run(run_id, foreign_answer)["1_isolation"]["each_vault_resolved_by_position_and_answer"].values())),
        ("baseline: resolved-ownership clause green on the real files",
         not any(v["unresolved_indexes"] for v in base["1_isolation"]["each_vault_resolved_by_position_and_answer"].values())),
        ("foreign answer under the right question -> assertion 4 red (structural)", not run(run_id, foreign_answer)["4_no_leaks"]["pass"]),
        ("judge verdict line in a chunk -> assertion 4 red", not run(run_id, judge_line)["4_no_leaks"]["pass"]),
        ("gold answer pasted into a chunk -> assertion 4 red", not run(run_id, gold_appended)["4_no_leaks"]["pass"]),
    ]
    ok = True
    for name, passed in checks:
        print(("  PASS  " if passed else "  FAIL  ") + name)
        ok &= bool(passed)
    check_smoke.main(run_id)  # leave the run's check.json as the real one
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
