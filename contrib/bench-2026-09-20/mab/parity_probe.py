#!/usr/bin/env python
"""Diagnostic, no LLM: replay the smoke's 10 searches against a copy-on-write CLONE of the writing
arm's vault with the two read-path cuts switched off (`parity` profile), to see where the memory
holding each fact ranks when the similarity floor is not applied.

Two traps this handles: any search writes an exposure row, so it runs on a clone and never on the
vault the smoke measured; and an identical request is REPLAYED from the ledger with the old bytes,
profile notwithstanding, so the byte budget is 32767 instead of 32768 (it never binds: the largest
served context in the smoke was ~15 KB) and the response's floor is checked to read 0.0.

Usage: BENCH_SCOPE=mab-probe python mab/parity_probe.py <run_id> <clone_root>
"""
import json
import sys
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "mab"))
sys.path.insert(0, str(BASE / "MemoryAgentBench"))

from common import kscope_io  # noqa: E402
import smoke_check  # noqa: E402


def main() -> int:
    run_id, clone = sys.argv[1], Path(sys.argv[2]).resolve()
    assert "diagnostic" in str(clone), "this probe only ever runs on a diagnostic clone"
    identity = json.loads(Path(str(clone) + "-identity.json").read_text())
    kscope_io.check_binary()
    vault = kscope_io.Vault(clone, identity["workspace_id"], identity["principal_id"], identity["journal"], "parity")

    rows = smoke_check.jsonl(BASE / "logs" / f"mab_adapter-{run_id}.jsonl")
    krows = [r for r in rows if r["writer"] == "kscope"]
    questions_log = {r["query_id"]: r for r in krows if r["event"] == "question"}
    units = [r for r in krows if r["event"] == "unit"]
    mem_of_unit = {}
    for b in (r for r in krows if r["event"] == "remember_batch"):
        for res in b.get("results", []):
            if res["status"] == "created":
                mem_of_unit[(b["chunk_index"], res["unit_index"])] = res["memory_id"]
    for r in (r for r in krows if r["event"] == "fallback_resend" and r["stored"]):
        mem_of_unit[(r["chunk_index"], r["unit_index"])] = r["memory_id"]

    context, questions, answers = smoke_check.load_context()
    facts = smoke_check.parse_facts(context)
    print("| q | fact | shipped: fate | parity (floor off): rank of its memory | parity served |")
    print("|---|---|---|---|---|")
    for qi in sorted(questions_log):
        q = questions_log[qi]
        rc, body, err = vault.search(q["sent_query"], top_k=10, maximum_context_bytes=32767)
        assert rc == 0 and body["served_floor"]["similarity_floor"] == 0.0, (rc, body.get("served_floor"), err)
        parity_rank = {h["memory_id"]: i + 1 for i, h in enumerate(body["selected_hits"])}
        shipped_rank = {h["memory_id"]: h["rank"] for h in q["served_hits"]}
        shipped_cut = {o["memory_id"]: o["reason"] for o in (q["omissions"] or [])}
        stem, family = smoke_check.fact_family(questions[qi], answers[qi], facts)
        for position, (serial, text) in enumerate(family):
            role = "CURRENT" if position == len(family) - 1 else "stale"
            unit = next(u for u in units if text in u["unit_text"])
            memory_id = mem_of_unit[(unit["chunk_index"], unit["unit_index"])]
            shipped = (f"served rank {shipped_rank[memory_id]}" if memory_id in shipped_rank
                       else f"cut: {shipped_cut[memory_id]}" if memory_id in shipped_cut else "never a candidate")
            parity = f"rank {parity_rank[memory_id]}" if memory_id in parity_rank else "not in the 10 served"
            print(f"| {qi} | {role} {serial}: {text.rstrip('.')[len(stem):].strip()} | {shipped} | {parity} | {len(body['selected_hits'])} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
