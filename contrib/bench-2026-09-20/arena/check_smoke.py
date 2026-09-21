"""Independent checker for an arena smoke run: builds the seven SHARED_RULES assertions from
the logs, the vaults on disk, the harness's own result files and the spend ledger.

It reads what the producer wrote; it imports nothing from the adapter. Each assertion carries
the numbers it was decided on, and the string search used for the leak check is first run on a
known positive (the question text, which every chunk carries) so a blind instrument is caught.

usage: check_smoke.py <run_id> [also-mine-prefix ...]  -> prints a summary, writes results/arena/<run_id>/check.json

Two clauses were repaired on 2026-09-20 BEFORE the parallel smoke ran, from facts the dataset
alone establishes (zero LLM cost), and the raw numbers they replaced are still printed:
  * assertion 1 resolved a chunk's owner by "## Task: <q>" text. 28 question strings are shared
    verbatim by 2-3 math tasks (papers split into _part_1/_part_2: 8/20, 9/17, 26/35, 2/14/34,
    6/23/38), so that clause reads red for 12 of 40 math tasks with perfect isolation. Ownership
    is now settled by the chunk's position AND the agent's own answer from this task's result
    file; `chunk_owner_papers` (the raw text map) is still reported.
  * assertion 4 counted every gold string inside the chunks. Nine phys golds are under 15 chars
    ("1", "2", "4", "$1$", ...), so that count is >0 for phys:3/19/1 by construction. The leak
    test is now structural: each chunk must equal, byte for byte, what build_memory_entry makes
    from (question, agent answer, agent tool trace) with reward=None -- i.e. it contains nothing
    but the question and the agent's own output. The raw gold counts are still reported, split
    into hits that lie inside the question/agent text and hits that do not (the leaks).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
sys.path.insert(0, str(BASE))
from common import kscope_io, llm  # noqa: E402

WRITER = {"kscope": "kscope", "kscope_none": "none", "kscope_full": "full_history"}
# Other roots this scope legitimately touched, named on the command line: my probe vaults
# (prefix `probe-`) and any aborted attempt. Anything else under scope `arena` fails assertion 1.
ALSO_MINE = ["probe-"]
DATA = {"math": BASE / "data/memoryarena__formal_reasoning_math__data.jsonl",
        "phys": BASE / "data/memoryarena__formal_reasoning_phys__data.jsonl"}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def squash(text: str) -> str:
    return " ".join(str(text).split())


def main(run_id: str) -> dict:
    out_root = BASE / "results" / "arena" / run_id
    task_map = jsonl(out_root / "task_map.jsonl")
    adapter = jsonl(BASE / "logs" / f"arena_adapter-{run_id}.jsonl")
    calls = jsonl(BASE / "logs" / "kscope_calls-arena.jsonl")
    ledger = [r for r in jsonl(BASE / "spend" / "ledger.jsonl") if r.get("scope") == "arena"]
    data = {d: jsonl(p) for d, p in DATA.items()}
    by_user = defaultdict(list)
    for row in adapter:
        by_user[row["user_id"]].append(row)
    report: dict = {"run_id": run_id, "tasks": [], "assertions": {}}

    # ------------------------------------------------------------------ per task
    for t in task_map:
        gold_row = data[t["domain"]][t["id"]]
        assert gold_row["paper_name"] == t["paper_name"]
        user = (t.get("new_vaults") or [None])[0]
        rows = by_user.get(user, [])
        adds = [r for r in rows if r["op"] == "add_chunk"]
        wraps = [r for r in rows if r["op"] == "wrap_user_prompt"]
        final = rows[-1]["counters"] if rows else {}
        result_file = out_root / t["domain"] / t["arm"] / t["paper_name"] / "result.jsonl"
        results = jsonl(result_file)
        root = Path(t["vault_root"]) if t.get("vault_root") else None
        disk_memories = sorted(root.glob("workspaces/*/records/memory/mem_*")) if root else []
        disk_texts = [p.read_text() for m in disk_memories for p in m.glob("versions/*/content.md")]
        log_calls = [c for c in calls if root and c["root"] == str(root)]

        # which paper do this user's chunks belong to? (question text -> paper)
        owners = Counter()
        for a in adds:
            for d, rws in data.items():
                for r in rws:
                    if any(("## Task: " + q) in a["chunk"] for q in r["questions"]):
                        owners[(d, r["id"])] += 1
        # leak check, on chunks handed to the write door AND on what is on disk
        chunks = [a["chunk"] for a in adds]
        leak = []
        for i, gold in enumerate(gold_row["answers"]):
            gold = str(gold)
            leak.append({"subtask": i, "gold_chars": len(gold),
                         "in_chunks_exact": sum(c.count(gold) for c in chunks),
                         "in_chunks_whitespace_insensitive": sum(squash(c).count(squash(gold)) for c in chunks),
                         "on_disk_exact": sum(x.count(gold) for x in disk_texts)})
        positive_control = [sum(c.count(q) for c in chunks) for q in gold_row["questions"]]
        judge_marks = sum(c.count("## Judge:") for c in chunks)
        # ownership settled by position and by the agent's own answer (see the header)
        resolved, unresolved, by_order_only = 0, [], 0
        for a in adds:
            k = a["chunk_index"]
            if k == 0:
                ok = a["chunk"] == "Initial result: Empty\n"
            else:
                q = gold_row["questions"][k - 1] if k - 1 < len(gold_row["questions"]) else None
                ok = q is not None and a["chunk"].startswith("## Task: " + q + "\n")
                if ok and k - 1 < len(results):
                    ok = ("## solution: " + str(results[k - 1].get("response")) + "\n") in a["chunk"]
                elif ok:
                    by_order_only += 1
            resolved += bool(ok)
            if not ok:
                unresolved.append(k)
        shared_questions = sum(1 for q in gold_row["questions"]
                               if sum(1 for d, rws in data.items() for r in rws if q in r["questions"]) > 1)
        # structural leak test: chunk k == build_memory_entry(question, agent answer, tool trace, reward=None)
        structural = []
        for a in adds:
            k = a["chunk_index"]
            if k == 0:
                structural.append(a["chunk"] == "Initial result: Empty\n")
            elif k - 1 < len(results):
                r = results[k - 1]
                rebuilt = f"## Task: {r['query']}\n## solution: {r['response']}\n"
                tool_info = (r.get("observation") or {}).get("tool_info")
                if tool_info is not None and len(tool_info) != 0:
                    rebuilt += f"## Tool Calls Info: {str(tool_info)}\n"
                structural.append(a["chunk"] == rebuilt)
            else:
                structural.append(None)  # no result row to rebuild from (the task did not complete)
        for l in leak:
            inside = 0
            k = 0
            for a in adds:
                kk = a["chunk_index"]
                if kk >= 1 and kk - 1 < len(results):
                    r = results[kk - 1]
                    agent_text = str(r.get("query", "")) + str(r.get("response", "")) + str((r.get("observation") or {}).get("tool_info", ""))
                    inside += agent_text.count(str(gold_row["answers"][l["subtask"]]))
            l["inside_question_or_agent_text"] = inside
            l["unexplained_exact"] = max(0, l["in_chunks_exact"] - inside)

        # propagation: a later search served a memory an EARLIER add_chunk created
        written_at = {a["memory_id"]: a["chunk_index"] for a in adds if a.get("memory_id")}
        timeline, served_earlier = [], []
        events = sorted(rows, key=lambda r: r["ts"])
        step = -1
        for r in events:
            if r["op"] != "wrap_user_prompt":
                continue
            step += 1
            hits = [(mid, written_at.get(mid)) for mid in (r.get("served_memory_ids") or [])]
            timeline.append({"subtask": step, "outcome": r.get("outcome"), "served": r.get("served"),
                             "served_chunk_indexes": [idx for _, idx in hits],
                             "omission_reasons": r.get("omission_reasons"), "omitted_hits": r.get("omitted_hits"),
                             "stop_reason": r.get("stop_reason"), "abstained": r.get("abstained"),
                             "query_truncated": r.get("query_truncated"), "query_bytes": r.get("query_bytes"),
                             "query_bytes_dropped": r.get("query_bytes_dropped"),
                             "served_bytes": r.get("served_bytes"), "context_tag_in_served": r.get("served_holding_context_tag"),
                             "failure": r.get("failure"), "similarity_floor": (r.get("served_floor") or {}).get("similarity_floor")})
            # chunk_index 0 is the harness's constant seed; chunk k (k>=1) is subtask k-1's record
            served_earlier += [{"subtask": step, "served_record_of_subtask": idx - 1} for _, idx in hits if idx and idx >= 1]

        report["tasks"].append({
            "arm": t["arm"], "domain": t["domain"], "id": t["id"], "paper_name": t["paper_name"], "status": t["status"],
            "error": t.get("error"), "user_id": user, "vault_root": str(root) if root else None,
            "new_vaults_seen_by_driver": t.get("new_vaults"), "subtasks": t["subtasks"],
            "is_correct": [bool(r.get("is_correct")) for r in results],
            "empty_answers": sum(1 for r in results if not str(r.get("response") or "").strip()),
            "chunk_owner_papers": {f"{d}:{i}": n for (d, i), n in owners.items()},
            "chunks_resolved_to_this_task": resolved, "chunks_unresolved": unresolved,
            "chunks_resolved_by_question_order_only": by_order_only,
            "questions_shared_with_other_papers_in_dataset": shared_questions,
            "structural_chunk_matches": structural,
            "counters": final,
            "chunks_in": final.get("chunks_in"), "accepted": final.get("accepted"),
            "memories_on_disk": len(disk_memories),
            "kscope_log": {"remember_accepted": sum(c.get("accepted") or 0 for c in log_calls if c["op"] == "remember"),
                           "remember_refused_calls": sum(1 for c in log_calls if c["op"] == "remember" and c["rc"] != 0),
                           "searches_ok": sum(1 for c in log_calls if c["op"] == "search" and c["rc"] == 0),
                           "searches_refused": sum(1 for c in log_calls if c["op"] == "search" and c["rc"] != 0),
                           "inits": sum(1 for c in log_calls if c["op"] == "init")},
            "writer_prompt_shas": sorted({(a.get("extract") or {}).get("prompt_sha") for a in adds if a.get("extract")} - {None}),
            "extract": [{"chunk_index": a["chunk_index"], "bytes": a["chunk_bytes"], "outcome": a["outcome"],
                         "fallback": a.get("fallback"), "attempts": (a.get("extract") or {}).get("attempts"),
                         "parse_failures": (a.get("extract") or {}).get("parse_failures"),
                         "extract_error": (a.get("extract") or {}).get("error"), "refusal": a.get("refusal"),
                         "delta": a.get("delta"), "ms": a.get("ms")} for a in adds],
            "searches": timeline, "served_earlier": served_earlier,
            "leak": leak, "leak_positive_control_question_counts": positive_control, "judge_marks_in_chunks": judge_marks,
            "memory_context_chars": [len(r.get("memory_context") or "") for r in results],
        })

    tasks = report["tasks"]
    k_tasks = [t for t in tasks if t["arm"] == "kscope"]
    # ---------------------------------------------------------------- assertions
    expected_roots = {t["vault_root"] for t in tasks if t["vault_root"]}
    disk_roots = set()
    for w in sorted(set(WRITER.values())):   # every arm's writer dir, not a hardcoded two
        d = kscope_io.VAULTS / "arena" / run_id / w
        if d.is_dir():
            disk_roots |= {str(p) for p in d.iterdir() if p.is_dir()}
    touched = kscope_io.roots_touched("arena")
    run_prefix = str(kscope_io.VAULTS / "arena" / run_id) + "/"
    touched_run = {r for r in touched if r.startswith(run_prefix)}
    touched_other = sorted(touched - touched_run)
    report["assertions"]["1_isolation"] = {
        "pass": (all(len(t["new_vaults_seen_by_driver"] or []) == 1 for t in tasks)
                 and expected_roots == disk_roots == touched_run
                 and len(expected_roots) == len(tasks)
                 and all(not t["chunks_unresolved"] and t["chunks_resolved_to_this_task"] == len(t["extract"]) for t in tasks)
                 and all(any(r.startswith(str(kscope_io.VAULTS / "arena" / mine) + "/") or r.startswith(str(kscope_io.VAULTS / "arena" / mine))
                             for mine in ALSO_MINE) for r in touched_other)),
        "tasks": len(tasks), "vault_roots": sorted(expected_roots), "roots_on_disk_for_run": len(disk_roots),
        "roots_touched_for_run": len(touched_run),
        "roots_touched_outside_run": touched_other, "declared_prefixes_for_those": list(ALSO_MINE),
        "each_vault_holds_one_papers_chunks_raw_text_map": {t["vault_root"]: t["chunk_owner_papers"] for t in tasks},
        "each_vault_resolved_by_position_and_answer": {t["vault_root"]: {"resolved": t["chunks_resolved_to_this_task"], "of": len(t["extract"]),
                                                                        "unresolved_indexes": t["chunks_unresolved"],
                                                                        "by_question_order_only": t["chunks_resolved_by_question_order_only"],
                                                                        "questions_shared_in_dataset": t["questions_shared_with_other_papers_in_dataset"]}
                                                       for t in tasks},
    }
    personal = str(kscope_io.PERSONAL_VAULT)
    naming_personal = [c for c in calls if c["root"] == personal or c["root"].startswith(personal + "/")]
    outside = [c["root"] for c in calls if not c["root"].startswith(str(kscope_io.VAULTS) + "/")]
    try:
        kscope_io.assert_root(kscope_io.PERSONAL_VAULT)
        guard = "DID NOT REFUSE"
    except kscope_io.IsolationError as exc:
        guard = f"refused: {exc}"
    report["assertions"]["2_personal_vault_untouched"] = {
        "pass": not naming_personal and not outside and guard.startswith("refused"),
        "calls_in_log": len(calls), "calls_naming_personal_vault": len(naming_personal),
        "calls_outside_vaults_dir": len(outside), "guard_on_personal_vault": guard}
    report["assertions"]["3_propagation"] = {
        "pass": bool(k_tasks) and all(t["chunks_in"] == t["accepted"] == t["memories_on_disk"] == t["kscope_log"]["remember_accepted"]
                                      == t["subtasks"] + 1 for t in k_tasks if t["status"] == "completed")
                and any(t["served_earlier"] for t in k_tasks),
        "per_task": [{"task": f"{t['domain']}:{t['id']}", "chunks_in": t["chunks_in"], "accepted": t["accepted"],
                      "memories_on_disk": t["memories_on_disk"], "kscope_log_accepted": t["kscope_log"]["remember_accepted"],
                      "expected_chunks": t["subtasks"] + 1, "served_earlier": t["served_earlier"]} for t in k_tasks],
        "none_arm": [{"task": f"{t['domain']}:{t['id']}", "chunks_in": t["chunks_in"], "accepted": t["accepted"],
                      "memories_on_disk": t["memories_on_disk"], "kscope_calls_besides_init": sum(v for k, v in t["kscope_log"].items() if k != "inits")}
                     for t in tasks if t["arm"] != "kscope"],
    }
    report["assertions"]["4_no_leaks"] = {
        "pass": bool(tasks)
                and all(t["structural_chunk_matches"] and all(m is True for m in t["structural_chunk_matches"]) for t in tasks)
                and all(l["unexplained_exact"] == 0 for t in tasks for l in t["leak"])
                and all(t["judge_marks_in_chunks"] == 0 for t in tasks)
                and all(min(t["leak_positive_control_question_counts"] or [0]) >= 1 for t in tasks if t["status"] == "completed"),
        "raw_gold_hits_in_chunks_exact": sum(l["in_chunks_exact"] for t in tasks for l in t["leak"]),
        "raw_gold_hits_inside_question_or_agent_text": sum(l["inside_question_or_agent_text"] for t in tasks for l in t["leak"]),
        "gold_hits_unexplained": sum(l["unexplained_exact"] for t in tasks for l in t["leak"]),
        "chunks_structurally_equal_to_rebuild": sum(1 for t in tasks for m in t["structural_chunk_matches"] if m is True),
        "chunks_structurally_different": sum(1 for t in tasks for m in t["structural_chunk_matches"] if m is False),
        "chunks_unverifiable_no_result_row": sum(1 for t in tasks for m in t["structural_chunk_matches"] if m is None),
        "per_task": [{"task": f"{t['arm']}/{t['domain']}:{t['id']}", "gold_hits": t["leak"],
                      "structural_chunk_matches": t["structural_chunk_matches"],
                      "judge_marks_in_chunks": t["judge_marks_in_chunks"],
                      "positive_control_question_hits": t["leak_positive_control_question_counts"]} for t in tasks],
    }
    run_ledger = [r for r in ledger if str(r.get("tag", "")).startswith(run_id)]
    finish = Counter((r["role"], r["finish"]) for r in run_ledger)
    report["assertions"]["5_failures"] = {
        "extraction_failures": sum((t["counters"] or {}).get("extraction_failures", 0) for t in tasks),
        "refused_items": sum((t["counters"] or {}).get("refused", 0) for t in tasks),
        "refusal_reasons": [{"task": f"{t['domain']}:{t['id']}", "chunk_index": e["chunk_index"], "refusal": e["refusal"]}
                            for t in k_tasks for e in t["extract"] if e.get("refusal")],
        "stored_under_fallback_delta": sum((t["counters"] or {}).get("fallback_stored", 0) for t in tasks),
        "chunks_lost": sum((t["counters"] or {}).get("lost", 0) for t in tasks),
        "search_failures": sum((t["counters"] or {}).get("search_failures", 0) for t in tasks),
        "harness_exceptions": [{"task": f"{t['arm']}/{t['domain']}:{t['id']}", "status": t["status"], "error": t["error"]}
                               for t in tasks if t["status"] != "completed"],
        "empty_agent_answers": {f"{t['arm']}/{t['domain']}:{t['id']}": t["empty_answers"] for t in tasks},
        "llm_finish_reasons_by_role": {f"{role}/{fin}": n for (role, fin), n in sorted(finish.items())},
        "writer_prompt_shas": sorted({s for t in k_tasks for s in t["writer_prompt_shas"]}),
    }
    report["assertions"]["5_failures"]["pass"] = (
        report["assertions"]["5_failures"]["chunks_lost"] == 0
        and report["assertions"]["5_failures"]["search_failures"] == 0
        and not report["assertions"]["5_failures"]["harness_exceptions"]
        and report["assertions"]["5_failures"]["extraction_failures"] == 0
        and report["assertions"]["5_failures"]["refused_items"] == 0
        and len(report["assertions"]["5_failures"]["writer_prompt_shas"]) <= 1)
    report["assertions"]["6_served_set_shape"] = {
        "per_search": [{"task": f"{t['domain']}:{t['id']}", **s} for t in k_tasks for s in t["searches"]],
        "kscope_search_refusals_in_log": [c for c in calls if c["op"] == "search" and c["rc"] != 0 and c["root"].startswith(run_prefix)].__len__(),
    }
    by_role = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "reasoning": 0, "est_usd": 0.0})
    for r in run_ledger:
        key = f"{r['tag'].split('/')[1] if '/' in r['tag'] else r['tag']}/{r['role']}"
        agg = by_role[key]
        agg["calls"] += 1
        agg["in"] += r["in"]
        agg["out"] += r["out"]
        agg["reasoning"] += r.get("reasoning") or 0
        agg["est_usd"] = round(agg["est_usd"] + r["est_usd"], 4)
    report["assertions"]["7_spend"] = {
        "llm_report": llm.report(), "this_run_by_arm_and_role": dict(by_role),
        "this_run_est_usd": round(sum(r["est_usd"] for r in run_ledger), 4),
        "arena_scope_est_usd_including_probes": round(sum(r["est_usd"] for r in ledger), 4),
        "effort_by_role": {f"{role}/{eff}": n for (role, eff), n in sorted(Counter((r["role"], str(r.get("effort"))) for r in run_ledger).items())},
        "temperature_sent_by_role": {f"{role}/{sent}": n for (role, sent), n in sorted(Counter((r["role"], bool(r.get("temperature_sent"))) for r in run_ledger).items())},
    }
    (out_root / "check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


if __name__ == "__main__":
    ALSO_MINE += sys.argv[2:]
    rep = main(sys.argv[1])
    for name, body in rep["assertions"].items():
        print(f"{name}: {'PASS' if body.get('pass') else ('FAIL' if 'pass' in body else 'see body')}")
    for t in rep["tasks"]:
        print(f"  {t['arm']:12s} {t['domain']}:{t['id']:<3d} status={t['status']} correct={t['is_correct']} chunks_in={t['chunks_in']} "
              f"accepted={t['accepted']} on_disk={t['memories_on_disk']} served={[s['served'] for s in t['searches']]} "
              f"served_earlier={t['served_earlier']}")
    print(rep["assertions"]["7_spend"]["llm_report"])
