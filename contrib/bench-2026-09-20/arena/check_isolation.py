"""The isolation proof for a PARALLEL arena run: every vault holds exactly its own paper's chunks.

Reads the disk (vault trees), the harness's own result files, the driver's task_map, the adapter
log and the kscope call log. Imports nothing from the adapter. Every check below is decided on
numbers it prints, and `--selftest` plants four contaminations into copies of the loaded state
and requires each one to go red -- an instrument that cannot fail is not evidence.

What "its own chunks" means here. The harness writes, per paper, one constant seed chunk
("Initial result: Empty\\n") and then, after subtask i, `MathAgent.build_memory_entry`:

    "## Task: " + question_i + "\\n" + "## solution: " + agent_answer_i + "\\n"
    + ("## Tool Calls Info: " + str(tool_info_i) + "\\n"  if tool_info_i else "")

with reward=None (so no "## Judge:" line). Every field of that string is in the paper's own
result.jsonl (`query`, `response`, `observation.tool_info`), written by the harness at the end
of the paper. So the expected content of a vault is REBUILT from the result file alone, and the
disk must match it byte for byte (kscope stores the chunk under a "# <title>\\n\\n" heading the
runtime supplies). A chunk from any other paper -- even one asking the identical question --
carries that paper's own generated answer and cannot match.

Why question text alone is not enough: the dataset splits some papers into `_part_1` /
`_part_2` tasks that share questions verbatim (28 question strings appear in 2-3 math tasks:
8/20, 9/17, 26/35, 2/14/34, 6/23/38). "## Task: <q> appears in exactly one vault" is therefore
false BY CONSTRUCTION for those questions in a full run; the check below expects such a
question in exactly the vaults of the tasks that hold it, and settles ownership by the answer.

usage: check_isolation.py <run_id> [--selftest] [--personal-snapshot <file> <start_epoch> <end_epoch>]
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
VAULTS = BASE / "vaults" / "arena"
PERSONAL = Path("/path/to/kaleidoscope/.kaleidoscope")
DATA = {"math": BASE / "data/memoryarena__formal_reasoning_math__data.jsonl",
        "phys": BASE / "data/memoryarena__formal_reasoning_phys__data.jsonl"}
WRITER = {"kscope": "kscope", "kscope_none": "none", "kscope_full": "full_history"}
SEED = "Initial result: Empty\n"


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def rebuild_chunk(row: dict) -> str:
    """`MathAgent.build_memory_entry(task, action, observation, reward=None)` from the result row."""
    text = f"## Task: {row['query']}\n## solution: {row['response']}\n"
    tool_info = (row.get("observation") or {}).get("tool_info")
    if tool_info is not None and len(tool_info) != 0:
        text += f"## Tool Calls Info: {str(tool_info)}\n"
    return text


def norm(text: str) -> str:
    """The store's own normalisation: trailing spaces/tabs stripped from every line."""
    return "\n".join(line.rstrip(" \t") for line in text.split("\n"))


def disk_chunks(root: Path) -> list[str]:
    """Every stored memory's body, with the runtime's '# <title>\\n\\n' heading removed."""
    out = []
    for content in sorted(root.glob("workspaces/*/records/memory/mem_*/versions/*/content.md")):
        text = content.read_text()
        if not text.startswith("# ") or "\n\n" not in text:
            out.append(text)  # not the expected shape; it will fail to match and be listed
            continue
        out.append(text[text.index("\n\n") + 2:])
    return out


def load(run_id: str) -> dict:
    out_root = BASE / "results" / "arena" / run_id
    data = {d: jsonl(p) for d, p in DATA.items()}
    tasks = []
    for t in jsonl(out_root / "task_map.jsonl"):
        result = out_root / t["domain"] / t["arm"] / str(t.get("paper_name")) / "result.jsonl"
        rows = jsonl(result) if t.get("paper_name") else []
        root = Path(t["vault_root"]) if t.get("vault_root") else None
        tasks.append({
            "arm": t["arm"], "domain": t["domain"], "id": t["id"], "paper_name": t.get("paper_name"),
            "status": t["status"], "subtasks": t["subtasks"], "user_id": (t.get("new_vaults") or [None])[0],
            "vault_root": str(root) if root else None, "result_rows": len(rows),
            "expected_chunks": ([SEED] + [rebuild_chunk(r) for r in rows]) if rows else None,
            "disk_chunks": disk_chunks(root) if root and root.is_dir() else None,
            "questions": data[t["domain"]][t["id"]]["questions"],
        })
    adapter = jsonl(BASE / "logs" / f"arena_adapter-{run_id}.jsonl")
    calls = jsonl(BASE / "logs" / "kscope_calls-arena.jsonl")
    on_disk = {}
    for w in WRITER.values():
        d = VAULTS / run_id / w
        if d.is_dir():
            on_disk[w] = sorted(str(p) for p in d.iterdir() if p.is_dir())
    # which tasks in the dataset hold each question text (the duplicated-question map)
    holders = defaultdict(set)
    for d, rows in data.items():
        for r in rows:
            for q in r["questions"]:
                holders[q].add((d, r["id"]))
    return {"run_id": run_id, "tasks": tasks, "adapter": adapter, "calls": calls, "on_disk": on_disk,
            "holders": holders, "run_prefix": str(VAULTS / run_id) + "/"}


def evaluate(st: dict) -> dict:
    tasks, rep = st["tasks"], {"run_id": st["run_id"], "checks": {}}
    completed = [t for t in tasks if t["status"] == "completed"]

    # A. one row per vault, one vault per row, nothing else on disk, nothing else touched
    rows_roots = [t["vault_root"] for t in tasks if t["vault_root"]]
    disk_roots = [r for roots in st["on_disk"].values() for r in roots]
    touched = sorted({c["root"] for c in st["calls"] if c["root"].startswith(st["run_prefix"])})
    dup_rows = [r for r, n in Counter(rows_roots).items() if n > 1]
    rep["checks"]["A_vault_roots"] = {
        "pass": (len(rows_roots) == len(tasks) and not dup_rows and set(rows_roots) == set(disk_roots) == set(touched)
                 and all(t["user_id"] and t["vault_root"].endswith("/" + t["user_id"]) for t in tasks)
                 and all(("/" + WRITER[t["arm"]] + "/") in t["vault_root"] for t in tasks)),
        "task_rows": len(tasks), "rows_with_a_vault": len(rows_roots), "rows_sharing_a_vault": dup_rows,
        "vault_dirs_on_disk": len(disk_roots), "orphan_dirs_no_row": sorted(set(disk_roots) - set(rows_roots)),
        "rows_without_dir": sorted(set(rows_roots) - set(disk_roots)),
        "roots_in_kscope_call_log_for_run": len(touched), "touched_not_in_rows": sorted(set(touched) - set(rows_roots)),
        "rows_not_touched": sorted(set(rows_roots) - set(touched)),
    }

    # B. the disk holds exactly the rebuilt chunks (kscope arm); nothing at all (none arm).
    #    kscope stores Markdown with trailing spaces stripped from each line (measured on
    #    smoke-20260920b: "## solution: \n" came back "## solution:\n"), so bodies are compared
    #    after that one normalisation, and the number of chunks that needed it is reported.
    per_task, bad, needed_norm = [], [], 0
    for t in completed:
        exp, disk = t["expected_chunks"], t["disk_chunks"]
        if t["arm"] == "kscope":
            ne, nd = [norm(c) for c in (exp or [])], [norm(c) for c in (disk or [])]
            ok = (disk is not None and exp is not None and len(disk) == len(exp) == t["subtasks"] + 1
                  and Counter(nd) == Counter(ne))
            needed_norm += sum(1 for c in (disk or []) if c not in (exp or []) and norm(c) in ne)
            extra = [c[:120] for c in nd if c not in ne]
            missing = [c[:120] for c in ne if c not in nd]
        else:
            ok = disk is not None and len(disk) == 0
            extra, missing = [c[:120] for c in (disk or [])], []
        per_task.append({"task": f"{t['arm']}/{t['domain']}:{t['id']}", "expected": len(exp or []),
                         "on_disk": len(disk) if disk is not None else None, "match": ok})
        if not ok:
            bad.append({"task": f"{t['arm']}/{t['domain']}:{t['id']}", "on_disk_not_expected": extra, "expected_not_on_disk": missing})
    rep["checks"]["B_disk_equals_own_result_file"] = {
        "pass": bool(completed) and all(p["match"] for p in per_task) and not bad,
        "tasks_checked": len(per_task), "chunks_equal_only_after_trailing_space_strip": needed_norm,
        "mismatches": bad, "per_task": per_task}

    # C. every rebuilt chunk (except the seed) is unique to one task in the run
    owner = defaultdict(set)
    for t in completed:
        for c in (t["expected_chunks"] or [])[1:]:
            owner[c].add(f"{t['arm']}/{t['domain']}:{t['id']}")
    shared = {c[:100]: sorted(o) for c, o in owner.items() if len(o) > 1}
    rep["checks"]["C_chunk_fingerprints_unique"] = {
        "pass": bool(owner) and not shared, "distinct_chunks": len(owner), "chunks_shared_by_tasks": shared}

    # D. '## Task: <q>' appears in exactly the vaults of the tasks that hold q (per arm), decided on disk
    #    for kscope and on the adapter log for both arms
    by_user = defaultdict(list)
    for r in st["adapter"]:
        by_user[r["user_id"]].append(r)
    in_run = {(t["domain"], t["id"]) for t in completed}
    findings, shared_q, checked = [], 0, 0
    for arm in sorted({t["arm"] for t in completed}):
        arm_tasks = [t for t in completed if t["arm"] == arm]
        texts = {}
        for t in arm_tasks:
            logged = [r["chunk"] for r in by_user.get(t["user_id"], []) if r["op"] == "add_chunk"]
            texts[(t["domain"], t["id"])] = "\n".join(logged) + ("\n".join(t["disk_chunks"] or []) if arm == "kscope" else "")
        for t in arm_tasks:
            for q in t["questions"]:
                checked += 1
                expected = {k for k in st["holders"][q] if k in in_run and any((x["domain"], x["id"]) == k for x in arm_tasks)}
                if len(st["holders"][q]) > 1:
                    shared_q += 1
                found = {k for k, text in texts.items() if ("## Task: " + q) in text}
                if found != expected:
                    findings.append({"arm": arm, "task": f"{t['domain']}:{t['id']}", "question": q[:80],
                                     "expected_in": sorted(map(str, expected)), "found_in": sorted(map(str, found))})
    rep["checks"]["D_task_text_in_exactly_own_vaults"] = {
        "pass": checked > 0 and not findings, "questions_checked": checked,
        "questions_shared_across_papers_in_dataset": shared_q, "violations": findings}

    # E. the adapter log: every user_id is a task of this run; chunk sequence contiguous and equal to the rebuild
    known = {t["user_id"]: t for t in tasks if t["user_id"]}
    unknown_users = sorted(u for u in by_user if u not in known)
    seq_bad, content_bad = [], []
    for u, rows in by_user.items():
        t = known.get(u)
        if not t:
            continue
        adds = sorted((r for r in rows if r["op"] == "add_chunk"), key=lambda r: r["chunk_index"])
        idx = [r["chunk_index"] for r in adds]
        inits = sum(1 for r in rows if r["op"] == "init")
        if inits != 1 or (t["status"] == "completed" and idx != list(range(t["subtasks"] + 1))):
            seq_bad.append({"task": f"{t['arm']}/{t['domain']}:{t['id']}", "inits": inits, "chunk_indexes": idx})
        if t["status"] == "completed" and t["expected_chunks"] is not None:
            got = [r["chunk"] for r in adds]
            if got != t["expected_chunks"]:
                content_bad.append({"task": f"{t['arm']}/{t['domain']}:{t['id']}",
                                    "first_diff_index": next((i for i, (a, b) in enumerate(zip(got, t["expected_chunks"])) if a != b), None),
                                    "logged": len(got), "expected": len(t["expected_chunks"])})
    rep["checks"]["E_adapter_log_per_user"] = {
        "pass": bool(by_user) and not unknown_users and not seq_bad and not content_bad,
        "user_ids_in_log": len(by_user), "user_ids_not_in_task_map": unknown_users,
        "sequence_violations": seq_bad, "content_mismatches": content_bad}

    # F. kscope call log per root: kscope arm = 1 init, subtasks+1 accepted remembers, >= subtasks searches; none arm = init only
    per_root = defaultdict(Counter)
    for c in st["calls"]:
        if c["root"].startswith(st["run_prefix"]):
            per_root[c["root"]][c["op"]] += 1
            if c["op"] == "remember":
                per_root[c["root"]]["remember_accepted"] += c.get("accepted") or 0
    f_bad = []
    for t in completed:
        ops = per_root.get(t["vault_root"], Counter())
        if t["arm"] == "kscope":
            ok = ops["init"] == 1 and ops["remember_accepted"] == t["subtasks"] + 1 and ops["search"] >= t["subtasks"]
        else:
            ok = ops["init"] == 1 and ops["remember"] == 0 and ops["search"] == 0
        if not ok:
            f_bad.append({"task": f"{t['arm']}/{t['domain']}:{t['id']}", "ops": dict(ops)})
    rep["checks"]["F_kscope_calls_per_root"] = {"pass": bool(completed) and not f_bad, "roots": len(per_root), "violations": f_bad}

    rep["pass"] = all(c["pass"] for c in rep["checks"].values())
    return rep


def personal_vault_text_check(st: dict) -> dict:
    """grep the whole personal vault for text that exists only because of this run."""
    markers = [st["run_id"]]
    for t in st["tasks"]:
        if t["user_id"]:
            markers.append(t["user_id"])
        for c in (t["expected_chunks"] or [])[1:]:
            sol = c.split("## solution: ", 1)[1][:80] if "## solution: " in c else ""
            if len(sol.strip()) >= 20:
                markers.append(sol)
    markers = sorted(set(markers))
    pattern_file = Path("/tmp") / f"arena-markers-{st['run_id']}.txt"
    pattern_file.write_text("\n".join(m.replace("\n", " ") for m in markers) + "\n")
    proc = subprocess.run(["grep", "-rlF", "-f", str(pattern_file), str(PERSONAL)], capture_output=True, text=True)
    hits = [line for line in proc.stdout.splitlines() if line.strip()]
    pattern_file.unlink(missing_ok=True)
    # positive control: the same grep over this run's own vault tree must hit
    ctl = subprocess.run(["grep", "-rlF", "-e", markers[0], str(VAULTS / st["run_id"])], capture_output=True, text=True)
    return {"pass": proc.returncode == 1 and not hits and bool(ctl.stdout.strip()),
            "markers": len(markers), "files_in_personal_vault_with_a_marker": hits[:20],
            "grep_rc(1=no match)": proc.returncode, "positive_control_files_in_run_tree": len(ctl.stdout.splitlines())}


def selftest(st: dict) -> dict:
    """Plant four contaminations in copies of the loaded state; each must turn its check red."""
    comp = [t for t in st["tasks"] if t["status"] == "completed" and t["arm"] == "kscope" and t["disk_chunks"]]
    if len(comp) < 2:
        return {"skipped": "needs two completed kscope tasks"}
    results = {}
    # 1. a foreign chunk on disk (task 2's second chunk copied into task 1's vault)
    s = copy.deepcopy(st)
    a, b = [t for t in s["tasks"] if t["status"] == "completed" and t["arm"] == "kscope"][:2]
    a["disk_chunks"].append(b["expected_chunks"][1])
    results["foreign_chunk_on_disk -> B red"] = not evaluate(s)["checks"]["B_disk_equals_own_result_file"]["pass"]
    # 2. two task rows pointing at one vault
    s = copy.deepcopy(st)
    a, b = [t for t in s["tasks"] if t["arm"] == "kscope"][:2]
    b["vault_root"], b["user_id"] = a["vault_root"], a["user_id"]
    results["two_rows_one_vault -> A red"] = not evaluate(s)["checks"]["A_vault_roots"]["pass"]
    # 3. the adapter routed one chunk to the wrong user (task 2's chunk logged under task 1's user)
    s = copy.deepcopy(st)
    a, b = [t for t in s["tasks"] if t["status"] == "completed" and t["arm"] == "kscope"][:2]
    stray = copy.deepcopy(next(r for r in s["adapter"] if r["user_id"] == b["user_id"] and r["op"] == "add_chunk" and r["chunk_index"] == 1))
    stray["user_id"] = a["user_id"]
    stray["chunk_index"] = a["subtasks"] + 1
    s["adapter"].append(stray)
    ev = evaluate(s)
    results["stray_adapter_row -> D or E red"] = not (ev["checks"]["D_task_text_in_exactly_own_vaults"]["pass"] and ev["checks"]["E_adapter_log_per_user"]["pass"])
    # 4. a stored chunk carrying the judge's verdict (what judge_result_in_memory=True would write)
    s = copy.deepcopy(st)
    a = next(t for t in s["tasks"] if t["status"] == "completed" and t["arm"] == "kscope")
    a["disk_chunks"][1] = a["disk_chunks"][1] + "## Judge: CORRECT\n"
    results["judge_line_in_stored_chunk -> B red"] = not evaluate(s)["checks"]["B_disk_equals_own_result_file"]["pass"]
    results["all_planted_contaminations_detected"] = all(results.values())
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--personal-snapshot", nargs=3, metavar=("FILE", "START", "END"))
    args = parser.parse_args()
    st = load(args.run_id)
    rep = evaluate(st)
    rep["G_personal_vault_text"] = personal_vault_text_check(st)
    if args.personal_snapshot:
        proc = subprocess.run([sys.executable, str(BASE / "arena" / "check_personal_vault.py"), *args.personal_snapshot],
                              capture_output=True, text=True)
        rep["H_personal_vault_mtime_window"] = {"stdout": proc.stdout[-4000:], "rc": proc.returncode}
    if args.selftest:
        rep["selftest"] = selftest(st)
    out = BASE / "results" / "arena" / args.run_id / "isolation.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    for name, body in rep["checks"].items():
        print(f"{name}: {'PASS' if body['pass'] else 'FAIL'}  " + json.dumps({k: v for k, v in body.items() if k not in ('pass', 'per_task')}, default=str)[:400])
    g = rep["G_personal_vault_text"]
    print(f"G_personal_vault_text: {'PASS' if g['pass'] else 'FAIL'}  {json.dumps(g)[:400]}")
    if "H_personal_vault_mtime_window" in rep:
        print("H_personal_vault_mtime_window:\n" + rep["H_personal_vault_mtime_window"]["stdout"])
    if "selftest" in rep:
        print("selftest:", json.dumps(rep["selftest"]))
    print("ALL ISOLATION CHECKS:", "PASS" if rep["pass"] and g["pass"] else "FAIL", "->", out)
    return 0 if rep["pass"] and g["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
