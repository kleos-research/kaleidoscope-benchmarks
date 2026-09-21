"""Every number the arena report needs, from what a run wrote. Read-only; no LLM call; no vault.

Sources: task_map.jsonl + records (driver), result.jsonl (harness), all_results.json (the
harness's own eval.py, produced by run_eval.sh), the adapter log, the spend ledger, the failures
log and the SDK retry lines in the driver/server logs. Prints a Markdown-ish summary and writes
results/arena/<run_id>/report.json.

usage: report_run.py <run_id> [--smoke-tasks math:8 ...] [--timeout 90]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
LIST_IN, LIST_OUT = 0.20, 1.20
PESS_IN, PESS_OUT = 1.10, 6.60


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def paired_bootstrap(diffs: list[float], n: int = 20000, seed: int = 20260920) -> tuple[float, float]:
    if not diffs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(mean(rng.choice(diffs) for _ in diffs) for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def cluster_bootstrap_rate_diff(pairs: list[tuple[list[bool], list[bool]]], n: int = 20000, seed: int = 20260920):
    """pairs: per task ([kscope subtask outcomes], [none subtask outcomes]); resample tasks."""
    if not pairs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        sample = [rng.choice(pairs) for _ in pairs]
        k = sum(sum(a) for a, _ in sample) / max(1, sum(len(a) for a, _ in sample))
        z = sum(sum(b) for _, b in sample) / max(1, sum(len(b) for _, b in sample))
        out.append(k - z)
    out.sort()
    return (out[int(0.025 * n)], out[int(0.975 * n) - 1])


def scores(tasks: list[dict], out: Path) -> dict:
    per = {}
    for t in tasks:
        if t["status"] != "completed":
            continue
        rows = jsonl(out / t["domain"] / t["arm"] / t["paper_name"] / "result.jsonl")
        per[(t["arm"], t["domain"], t["id"])] = {
            "correct": [bool(r.get("is_correct")) for r in rows],
            "empty": sum(1 for r in rows if not str(r.get("response") or "").strip()),
            "time": [r.get("time") or 0 for r in rows],
            "memory_context_chars": [len(r.get("memory_context") or "") for r in rows],
            "tool_path": [(r.get("agent_tool_path") or {}).get("status") for r in rows],
            "tool_seconds": [(r.get("agent_tool_path") or {}).get("seconds") for r in rows],
            "tool_errors": [(r.get("agent_tool_path") or {}).get("error") for r in rows
                            if (r.get("agent_tool_path") or {}).get("status") == "degraded_no_tools"],
        }
    return per


def arm_summary(per: dict, arm: str, keys=None) -> dict:
    sel = {k: v for k, v in per.items() if k[0] == arm and (keys is None or k[1:] in keys)}
    subtasks = [c for v in sel.values() for c in v["correct"]]
    progress = [mean(v["correct"]) for v in sel.values() if v["correct"]]
    final = [v["correct"][-1] for v in sel.values() if v["correct"]]
    return {"tasks": len(sel), "subtasks": len(subtasks), "subtask_pass_rate": round(mean(subtasks), 4) if subtasks else None,
            "subtasks_correct": sum(subtasks),
            "avg_progress_score": round(mean(progress), 4) if progress else None,
            "overall_average_passrate_last_subtask": round(mean(final), 4) if final else None,
            "empty_answers": sum(v["empty"] for v in sel.values()),
            "degraded_steps": sum(1 for v in sel.values() for s in v["tool_path"] if s == "degraded_no_tools"),
            "steps_with_tool_path_recorded": sum(1 for v in sel.values() for s in v["tool_path"] if s),
            "mean_subtask_wall_s": round(mean(x for v in sel.values() for x in v["time"]), 1) if subtasks else None,
            "mean_memory_context_chars": round(mean(x for v in sel.values() for x in v["memory_context_chars"]), 0) if subtasks else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--smoke-tasks", nargs="*", default=[])
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()
    run_id = args.run_id
    out = BASE / "results" / "arena" / run_id
    tasks = jsonl(out / "task_map.jsonl")
    per = scores(tasks, out)
    rep = {"run_id": run_id, "tasks_by_status": {f"{a}/{st}": n for (a, st), n in Counter((t["arm"], t["status"]) for t in tasks).items()}}
    smoke = {tuple([s.split(":")[0], int(s.split(":")[1])]) for s in args.smoke_tasks}

    # ------------------------------------------------------------------ scores
    rep["scores"] = {}
    for arm in ("kscope", "kscope_none", "kscope_full"):
        rep["scores"][arm] = {"all": arm_summary(per, arm),
                              "math": arm_summary(per, arm, {k[1:] for k in per if k[1] == "math"}),
                              "phys": arm_summary(per, arm, {k[1:] for k in per if k[1] == "phys"})}
        if smoke:
            rep["scores"][arm]["smoke_tasks"] = arm_summary(per, arm, smoke)
            rep["scores"][arm]["non_smoke_tasks"] = arm_summary(per, arm, {k[1:] for k in per} - smoke)
    # the harness's own eval.py numbers, if run_eval.sh was run
    rep["eval_py"] = {}
    for f in sorted(glob.glob(str(out / "*" / "*" / "all_results.json"))):
        p = Path(f)
        rep["eval_py"][f"{p.parent.parent.name}/{p.parent.name}"] = {k: json.load(open(f))[k] for k in
                                                                    ("overall_average_passrate", "avg_progress_score", "average_memory_length", "memory_length", "average_session_time", "min_k")}
    # paired differences
    baseline = os.environ.get("BENCH_BASELINE_ARM", "kscope_full")
    if not any(k[0] == baseline for k in per):
        baseline = "kscope_none"
    rep["paired_baseline_arm"] = baseline
    both = sorted({k[1:] for k in per if k[0] == "kscope"} & {k[1:] for k in per if k[0] == baseline})
    diffs = [mean(per[("kscope",) + k]["correct"]) - mean(per[(baseline,) + k]["correct"]) for k in both]
    pairs = [(per[("kscope",) + k]["correct"], per[(baseline,) + k]["correct"]) for k in both]
    rep["paired"] = {
        "tasks_with_both_arms": len(both),
        "mean_progress_diff_kscope_minus_none": round(mean(diffs), 4) if diffs else None,
        "progress_diff_95ci_paired_bootstrap": [round(x, 4) for x in paired_bootstrap(diffs)],
        "tasks_kscope_better": sum(1 for d in diffs if d > 0), "tasks_none_better": sum(1 for d in diffs if d < 0),
        "tasks_tied": sum(1 for d in diffs if d == 0),
        "subtask_rate_diff_95ci_cluster_bootstrap": [round(x, 4) for x in cluster_bootstrap_rate_diff(pairs)],
        "per_task": [{"task": f"{k[0]}:{k[1]}", "kscope": f"{sum(per[('kscope',)+k]['correct'])}/{len(per[('kscope',)+k]['correct'])}",
                      "none": f"{sum(per[(baseline,)+k]['correct'])}/{len(per[(baseline,)+k]['correct'])}", "diff": round(d, 3),
                      "smoke": k in smoke} for k, d in zip(both, diffs)],
    }

    # A degraded step was answered by a path with NO TOOLS, because the provider refused the
    # prompt (a content filter on LaTeX from RL papers) or the call timed out. It is not a
    # memory effect. The rate is not equal across arms, so dropping it per-arm would compare
    # two different subtask sets. Dropped PAIRWISE instead: if either arm's subtask k of task
    # T was degraded, subtask k is dropped from BOTH arms, and the count is published.
    clean_diffs, clean_pairs, dropped, kept = [], [], 0, 0
    for k in both:
        a, b = per[("kscope",) + k], per[(baseline,) + k]
        ta, tb = a.get("tool_path") or [], b.get("tool_path") or []
        ca, cb = [], []
        for i in range(min(len(a["correct"]), len(b["correct"]))):
            bad = (i < len(ta) and ta[i] == "degraded_no_tools") or (i < len(tb) and tb[i] == "degraded_no_tools")
            if bad:
                dropped += 1
                continue
            ca.append(a["correct"][i]); cb.append(b["correct"][i]); kept += 1
        if ca:
            clean_diffs.append(mean(ca) - mean(cb))
            clean_pairs.append((ca, cb))
    rep["paired_degraded_dropped_pairwise"] = {
        "subtasks_dropped": dropped, "subtasks_kept": kept,
        "tasks_with_any_clean_subtask": len(clean_diffs),
        "mean_progress_diff": round(mean(clean_diffs), 4) if clean_diffs else None,
        "progress_diff_95ci_paired_bootstrap": [round(x, 4) for x in paired_bootstrap(clean_diffs)] if clean_diffs else None,
        "subtask_rate_diff_95ci_cluster_bootstrap": [round(x, 4) for x in cluster_bootstrap_rate_diff(clean_pairs)] if clean_pairs else None,
        "kscope_subtask_rate": round(mean([c for ca, _ in clean_pairs for c in ca]), 4) if clean_pairs else None,
        "baseline_subtask_rate": round(mean([c for _, cb in clean_pairs for c in cb]), 4) if clean_pairs else None,
    }

    # ------------------------------------------------------------------ memory: served, preceding record, refusals
    adapter = jsonl(BASE / "logs" / f"arena_adapter-{run_id}.jsonl")
    by_user = defaultdict(list)
    for r in adapter:
        by_user[r["user_id"]].append(r)
    served_counts, prev_served, any_earlier, abstained, searches_after_first = [], 0, 0, 0, 0
    served_bytes = []
    omission = Counter()
    coerce = Counter()
    counters_final = Counter()
    for t in tasks:
        if t["arm"] != "kscope" or t["status"] != "completed":
            continue
        rows = sorted(by_user.get((t.get("new_vaults") or [None])[0], []), key=lambda r: r["ts"])
        written = {r["memory_id"]: r["chunk_index"] for r in rows if r["op"] == "add_chunk" and r.get("memory_id")}
        q = -1
        for r in rows:
            if r["op"] != "wrap_user_prompt":
                continue
            q += 1
            served_counts.append(r.get("served") or 0)
            served_bytes.append(r.get("served_bytes") or 0)
            omission.update(r.get("omission_reasons") or {})
            abstained += bool(r.get("abstained"))
            idxs = {written.get(m) for m in (r.get("served_memory_ids") or [])}
            if q >= 1:
                searches_after_first += 1
                prev_served += (q in idxs)               # chunk q is subtask q-1's record
                any_earlier += any(i is not None and i >= 1 for i in idxs)
        if rows:
            for k, v in rows[-1]["counters"].items():
                if isinstance(v, int):
                    counters_final[k] += v
    rep["memory"] = {
        "searches": len(served_counts), "mean_served_count": round(mean(served_counts), 3) if served_counts else None,
        "served_count_distribution": dict(Counter(served_counts)), "mean_served_bytes": round(mean(served_bytes), 0) if served_bytes else None,
        "searches_after_first_subtask": searches_after_first,
        "preceding_subtask_record_served": prev_served,
        "preceding_subtask_record_served_rate": round(prev_served / searches_after_first, 3) if searches_after_first else None,
        "any_earlier_record_served_rate": round(any_earlier / searches_after_first, 3) if searches_after_first else None,
        "abstained": abstained, "omission_reasons": dict(omission), "adapter_counters_summed": dict(counters_final),
    }

    # ------------------------------------------------------------------ LLM: calls, tokens, cost, timeouts
    ledger = [r for r in jsonl(BASE / "spend" / "ledger.jsonl") if str(r.get("tag", "")).startswith(run_id + "/")]
    arm_of = lambda r: r["tag"].split("/")[1] if "/" in r["tag"] else r["tag"]
    llm = {}
    n_q = Counter()
    for t in tasks:
        if t["status"] == "completed":
            n_q[t["arm"]] += t["subtasks"]
    to_ms = args.timeout * 1000
    for arm in ("kscope", "kscope_none", "kscope_full"):
        sel = [r for r in ledger if arm_of(r) == arm]
        ok = [r for r in sel if not r.get("failed")]
        failed = [r for r in sel if r.get("failed")]
        tin, tout = sum(r["in"] for r in ok), sum(r["out"] for r in ok)
        rescued = [r for r in ok if r["ms"] > to_ms]
        llm[arm] = {
            "calls": len(ok), "calls_per_question": round(len(ok) / n_q[arm], 2) if n_q[arm] else None,
            "calls_by_role": dict(Counter(r["role"] for r in ok)),
            "tokens_in": tin, "tokens_out": tout, "reasoning_tokens": sum(r.get("reasoning") or 0 for r in ok),
            # the efficiency axis: what does the MODEL actually read per subtask in this arm?
            "agent_calls": sum(1 for r in ok if r["role"] == "agent"),
            "mean_prompt_tokens_per_agent_call": round(
                sum(r["in"] for r in ok if r["role"] == "agent") / max(1, sum(1 for r in ok if r["role"] == "agent")), 1),
            "agent_prompt_tokens_per_subtask": round(
                sum(r["in"] for r in ok if r["role"] == "agent") / n_q[arm], 1) if n_q[arm] else None,
            "total_prompt_tokens_per_subtask": round(tin / n_q[arm], 1) if n_q[arm] else None,
            "usd_list_0.20_1.20": round((tin * LIST_IN + tout * LIST_OUT) / 1e6, 3),
            "usd_pessimistic_1.10_6.60": round((tin * PESS_IN + tout * PESS_OUT) / 1e6, 3),
            "usd_list_per_question": round((tin * LIST_IN + tout * LIST_OUT) / 1e6 / n_q[arm], 4) if n_q[arm] else None,
            "finish_reasons_by_role": dict(Counter(f"{r['role']}/{r['finish']}" for r in ok)),
            "temperature_sent_by_role": dict(Counter(f"{r['role']}/{r['temperature_sent']}" for r in ok)),
            "effort_by_role": dict(Counter(f"{r['role']}/{r['effort']}" for r in ok)),
            # The harness caps agent output at max_tokens=8192 (its own config) and the deployment
            # is a reasoning model: a call that spends the whole budget thinking returns
            # finish="length" with NO content, and `_reasoning_with_errors` then retries it up to
            # three times. Same config in both arms, but it is the run's dominant token cost and
            # the source of every empty answer, so it is counted rather than left in the total.
            "calls_finishing_on_length": sum(1 for r in ok if r["finish"] == "length"),
            "calls_finishing_on_length_by_role": dict(Counter(r["role"] for r in ok if r["finish"] == "length")),
            "output_tokens_burned_on_length": sum(r["out"] for r in ok if r["finish"] == "length"),
            "share_of_output_tokens_burned_on_length": round(sum(r["out"] for r in ok if r["finish"] == "length") / max(1, tout), 3),
            "calls_over_timeout_i.e._rescued_by_sdk_retry": len(rescued),
            "rescued_by_role": dict(Counter(r["role"] for r in rescued)),
            "rescued_ms": sorted(round(r["ms"] / 1000, 1) for r in rescued)[:40],
            "failed_calls": len(failed), "failed_by_role_and_error": dict(Counter(f"{r['role']}/{r['finish']}" for r in failed)),
            "agent_high_effort_ms_p50_p90_max": None,
        }
        hi = sorted(r["ms"] / 1000 for r in ok if r["role"] == "agent" and r.get("effort") == "high")
        if hi:
            llm[arm]["agent_high_effort_ms_p50_p90_max"] = [round(statistics.median(hi), 1), round(hi[int(len(hi) * 0.9)], 1), round(hi[-1], 1)]
        wr = sorted(r["ms"] / 1000 for r in ok if r["role"] == "writer")
        if wr:
            llm[arm]["writer_s_p50_p90_max"] = [round(statistics.median(wr), 1), round(wr[int(len(wr) * 0.9)], 1), round(wr[-1], 1)]
    rep["llm"] = llm
    # SDK retry lines from every process log of this run
    retry = Counter()
    for f in glob.glob(str(BASE / "logs" / f"arena_*-{run_id}-*.log")):
        role = "agent" if "driver" in f else ("writer" if "memoryserver" in f else ("judge" if "envserver" in f else "?"))
        for line in open(f, errors="replace"):
            m = re.search(r"\| (Encountered a timeout exception: \w+|Retrying request in [\d.]+ seconds \(retry \d+ of \d+\)|Encountered an HTTP status error: \d+|Raising timeout error|Raising connection error|Encountered exception: \w+)", line)
            if m:
                msg = m.group(1)
                msg = re.sub(r"in [\d.]+ seconds", "in N seconds", msg)
                retry[f"{role}: {msg}"] += 1
    rep["sdk_retry_lines_by_role"] = dict(sorted(retry.items()))
    # Why each degraded step happened. A timeout is the failure the 90s deadline was chosen
    # against; a content-policy 400 is a different animal entirely and no timeout changes it.
    def classify(err: str) -> str:
        e = (err or "").lower()
        if "timeout" in e or "timed out" in e:
            return "timeout"
        if "usage policy" in e or "flagged as potentially violating" in e:
            return "content_policy_400"
        if "badrequest" in e:
            return "other_400"
        if "connection" in e:
            return "connection"
        return "other:" + (err or "")[:60]
    rep["degraded_steps_by_arm"] = {arm: rep["scores"][arm]["all"]["degraded_steps"] for arm in ("kscope", "kscope_none", "kscope_full")}
    rep["degraded_step_causes"] = {arm: dict(Counter(classify(e) for k, v in per.items() if k[0] == arm for e in v["tool_errors"]))
                                   for arm in ("kscope", "kscope_none", "kscope_full")}
    rep["degraded_steps_per_subtask_by_arm"] = {
        arm: round(rep["scores"][arm]["all"]["degraded_steps"] / max(1, rep["scores"][arm]["all"]["subtasks"]), 4)
        for arm in ("kscope", "kscope_none", "kscope_full")}

    # ------------------------------------------------------------------ wall clock
    comp = [t for t in tasks if t["status"] == "completed"]
    rep["wall"] = {
        "run_started": min((t["started"] for t in tasks), default=None), "run_ended": max((t["ended"] for t in tasks), default=None),
        "run_wall_h": round((max(t["ended"] for t in tasks) - min(t["started"] for t in tasks)) / 3600, 2) if tasks else None,
        "mean_task_wall_s_by_arm": {arm: round(mean(t["ended"] - t["started"] for t in comp if t["arm"] == arm), 1) for arm in ("kscope", "kscope_none", "kscope_full") if any(t["arm"] == arm for t in comp)},
        "mean_question_wall_s_by_arm": {arm: round(sum(t["ended"] - t["started"] for t in comp if t["arm"] == arm) / max(1, sum(t["subtasks"] for t in comp if t["arm"] == arm)), 1) for arm in ("kscope", "kscope_none", "kscope_full") if any(t["arm"] == arm for t in comp)},
        "sequential_equivalent_h": round(sum(t["ended"] - t["started"] for t in comp) / 3600, 2) if comp else None,
    }
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "paired"}, indent=1, default=str))
    print("paired:", json.dumps({k: v for k, v in rep["paired"].items() if k != "per_task"}))
    for row in rep["paired"]["per_task"]:
        print("  ", row)
    print("->", out / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
