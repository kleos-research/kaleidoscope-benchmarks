"""How often kscope's query bound truncates a MemoryArena prompt, and what it costs.

Read-only; no LLM call, no vault. Joins the adapter log's per-search rows to the harness's
own per-subtask verdicts: the search for subtask k is the k-th `wrap_user_prompt` row of that
paper, and `is_correct[k]` is that subtask's verdict in the paper's result.jsonl.

usage: query_truncation.py <run_id>
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def main(run_id: str) -> int:
    out = BASE / "results" / "arena" / run_id
    rows = jsonl(BASE / "logs" / f"arena_adapter-{run_id}.jsonl")
    tasks = jsonl(out / "task_map.jsonl")
    by_user = {}
    for t in tasks:
        for u in (t.get("user_ids_captured") or []):
            by_user[u] = t

    # per-arm search accounting
    per_arm = {}
    joined = []          # (truncated, correct, served, abstained, dropped)
    for user, task in by_user.items():
        arm = task["arm"]
        searches = [r for r in rows if r["user_id"] == user and r["op"] == "wrap_user_prompt"]
        verdicts = task.get("is_correct") or []
        a = per_arm.setdefault(arm, {"searches": 0, "truncated": 0, "dropped": [], "none_by_design": 0,
                                     "truncated_fields_present": 0})
        for k, r in enumerate(searches):
            a["searches"] += 1
            if r.get("outcome") == "none_by_design":
                a["none_by_design"] += 1
                continue
            if "query_truncated" in r:
                a["truncated_fields_present"] += 1
            trunc = bool(r.get("query_truncated"))
            a["truncated"] += trunc
            if trunc:
                a["dropped"].append(r.get("query_bytes_dropped") or 0)
            if k < len(verdicts):
                joined.append({"arm": arm, "truncated": trunc, "correct": bool(verdicts[k]),
                               "served": r.get("served") or 0, "abstained": bool(r.get("abstained")),
                               "dropped": r.get("query_bytes_dropped") or 0,
                               "query_bytes": r.get("query_bytes"),
                               "cleaned": r.get("cleaned_query_bytes")})

    report = {"run_id": run_id, "per_arm": {}, "kscope_arm": {}}
    print(f"\n{'='*74}\nQUERY TRUNCATION  run {run_id}\n{'='*74}")
    for arm, a in sorted(per_arm.items()):
        n = a["searches"]
        eligible = n - a["none_by_design"]
        rate = round(a["truncated"] / eligible, 4) if eligible else None
        report["per_arm"][arm] = {"searches_logged": n, "none_by_design": a["none_by_design"],
                                  "searches_that_ran": eligible, "truncated": a["truncated"],
                                  "truncation_rate": rate,
                                  "rows_carrying_a_query_truncated_field": a["truncated_fields_present"]}
        print(f"\n{arm}")
        print(f"  wrap_user_prompt rows        {n}")
        print(f"  of those, none_by_design     {a['none_by_design']}   (no search ran: the arm does not search)")
        print(f"  searches that actually ran   {eligible}")
        print(f"  query_truncated: true        {a['truncated']}  rate={rate}")
        if a["dropped"]:
            d = sorted(a["dropped"])
            report["per_arm"][arm]["dropped_bytes"] = {
                "median": statistics.median(d), "max": max(d), "min": min(d),
                "p25": d[len(d) // 4], "p75": d[3 * len(d) // 4]}
            print(f"  query_bytes_dropped          median={statistics.median(d):,.0f}  "
                  f"min={min(d):,}  max={max(d):,}")

    k = [j for j in joined if j["arm"] == "kscope"]
    tr = [j for j in k if j["truncated"]]
    un = [j for j in k if not j["truncated"]]
    print(f"\n{'-'*74}\nWHAT IT COSTS (kscope arm, subtasks joined to their own search)\n{'-'*74}")
    for label, group in (("truncated", tr), ("untruncated", un)):
        if not group:
            print(f"  {label:<12} n=0")
            continue
        correct = sum(g["correct"] for g in group)
        served0 = sum(1 for g in group if g["served"] == 0)
        abst = sum(g["abstained"] for g in group)
        lo, hi = wilson(correct, len(group))
        report["kscope_arm"][label] = {
            "subtasks": len(group), "correct": correct, "pass_rate": round(correct / len(group), 4),
            "pass_rate_95ci_wilson": [lo, hi],
            "mean_served": round(sum(g["served"] for g in group) / len(group), 3),
            "served_zero": served0, "served_zero_rate": round(served0 / len(group), 4),
            "abstained": abst, "abstention_rate": round(abst / len(group), 4)}
        print(f"  {label:<12} n={len(group):<4} pass={correct}/{len(group)}={correct/len(group):.4f} "
              f"95%CI[{lo}, {hi}]  mean_served={sum(g['served'] for g in group)/len(group):.2f}  "
              f"served_0={served0} ({served0/len(group):.1%})  abstained={abst} ({abst/len(group):.1%})")
    if tr and un:
        diff = (sum(g["correct"] for g in tr) / len(tr)) - (sum(g["correct"] for g in un) / len(un))
        report["kscope_arm"]["pass_rate_diff_truncated_minus_untruncated"] = round(diff, 4)
        print(f"\n  pass-rate difference (truncated - untruncated) = {diff:+.4f}")
        print("  NOTE: not a causal estimate. A long prompt is also a harder prompt; this is")
        print("        the association, and the two cannot be separated without an ablation.")

    # ---- the two controls the association needs ---------------------------------------
    # (a) does truncation co-occur with the similarity floor, as the coordinator suspected?
    # (b) is truncation just a proxy for EARLY subtasks, where the vault is nearly empty and
    #     abstention is expected for a reason that has nothing to do with the query?
    floor_t = Counter(); floor_u = Counter()
    for j in k:
        (floor_t if j["truncated"] else floor_u).update({"n": 1})
    idx_t, idx_u = [], []
    corpus_t, corpus_u = [], []
    for user, task in by_user.items():
        if task["arm"] != "kscope":
            continue
        searches = [r for r in rows if r["user_id"] == user and r["op"] == "wrap_user_prompt"]
        for i, r in enumerate(searches):
            (idx_t if r.get("query_truncated") else idx_u).append(i)
            (corpus_t if r.get("query_truncated") else corpus_u).append(i + 1)  # memories available
            reasons = r.get("omission_reasons") or {}
            (floor_t if r.get("query_truncated") else floor_u).update(
                {"below_similarity_floor": reasons.get("below_similarity_floor", 0),
                 "redundant_with_selected": reasons.get("redundant_with_selected", 0),
                 "context_byte_budget": reasons.get("context_byte_budget", 0),
                 "omitted_hits": r.get("omitted_hits") or 0})
    report["similarity_floor_interaction"] = {
        "truncated": dict(floor_t), "untruncated": dict(floor_u),
        "below_floor_per_search_truncated": round(floor_t["below_similarity_floor"] / max(1, floor_t["n"]), 3),
        "below_floor_per_search_untruncated": round(floor_u["below_similarity_floor"] / max(1, floor_u["n"]), 3)}
    report["subtask_index_control"] = {
        "truncated_subtask_index_median": statistics.median(idx_t) if idx_t else None,
        "untruncated_subtask_index_median": statistics.median(idx_u) if idx_u else None,
        "truncated_at_subtask_0": sum(1 for i in idx_t if i == 0),
        "untruncated_at_subtask_0": sum(1 for i in idx_u if i == 0),
        "memories_available_median_truncated": statistics.median(corpus_t) if corpus_t else None,
        "memories_available_median_untruncated": statistics.median(corpus_u) if corpus_u else None}
    print(f"\n{'-'*74}\nCONTROLS\n{'-'*74}")
    print(f"  below_similarity_floor omissions per search: "
          f"truncated={report['similarity_floor_interaction']['below_floor_per_search_truncated']}  "
          f"untruncated={report['similarity_floor_interaction']['below_floor_per_search_untruncated']}")
    sc = report["subtask_index_control"]
    print(f"  subtask index (median): truncated={sc['truncated_subtask_index_median']}  "
          f"untruncated={sc['untruncated_subtask_index_median']}")
    print(f"  searches at subtask 0 (vault holds 1 seed memory): truncated={sc['truncated_at_subtask_0']}  "
          f"untruncated={sc['untruncated_at_subtask_0']}")
    print(f"  memories available (median): truncated={sc['memories_available_median_truncated']}  "
          f"untruncated={sc['memories_available_median_untruncated']}")

    # ---- the comparison that actually isolates truncation ------------------------------
    # Subtask 0 is BOTH the longest prompt (whole paper background, nothing yet summarised)
    # and the emptiest vault (one seed memory, "Initial result: Empty"). It is truncated and
    # it abstains for two independent reasons. Drop it and ask the question again.
    k_idx = []
    for user, task in by_user.items():
        if task["arm"] != "kscope":
            continue
        searches = [r for r in rows if r["user_id"] == user and r["op"] == "wrap_user_prompt"]
        verdicts = task.get("is_correct") or []
        for i, r in enumerate(searches):
            if i < len(verdicts):
                k_idx.append({"i": i, "truncated": bool(r.get("query_truncated")),
                              "correct": bool(verdicts[i]), "served": r.get("served") or 0,
                              "abstained": bool(r.get("abstained"))})
    later = [j for j in k_idx if j["i"] >= 1]
    lt = [j for j in later if j["truncated"]]
    lu = [j for j in later if not j["truncated"]]
    report["excluding_subtask_0"] = {}
    print(f"\n{'-'*74}\nTRUNCATION WITH SUBTASK 0 EXCLUDED (the confound removed)\n{'-'*74}")
    for label, group in (("truncated", lt), ("untruncated", lu)):
        if not group:
            print(f"  {label:<12} n=0"); continue
        c = sum(g["correct"] for g in group)
        lo, hi = wilson(c, len(group))
        report["excluding_subtask_0"][label] = {
            "subtasks": len(group), "correct": c, "pass_rate": round(c / len(group), 4),
            "pass_rate_95ci_wilson": [lo, hi],
            "mean_served": round(sum(g["served"] for g in group) / len(group), 3),
            "served_zero": sum(1 for g in group if g["served"] == 0),
            "abstained": sum(g["abstained"] for g in group)}
        print(f"  {label:<12} n={len(group):<4} pass={c}/{len(group)}={c/len(group):.4f} 95%CI[{lo}, {hi}]  "
              f"mean_served={sum(g['served'] for g in group)/len(group):.2f}  "
              f"served_0={sum(1 for g in group if g['served']==0)}  "
              f"abstained={sum(g['abstained'] for g in group)}")
    if lt and lu:
        d = (sum(g["correct"] for g in lt)/len(lt)) - (sum(g["correct"] for g in lu)/len(lu))
        report["excluding_subtask_0"]["pass_rate_diff"] = round(d, 4)
        print(f"\n  pass-rate difference (truncated - untruncated), subtask>=1 = {d:+.4f}")

    report["query_keep"] = {
        "value": sorted({r.get("query_keep") for r in rows if r.get("query_keep")}),
        "deliberate": "yes -- bound_query's docstring: the harness builds BACKGROUND then PROBLEM, "
                      "so the END holds the question and the head is LaTeX preamble. Selectable via "
                      "KSCOPE_ARENA_QUERY_KEEP so a run states which half it kept. Not changed mid-run.",
    }
    bound = [r.get("bound_learned_from_refusal") for r in rows if r.get("bound_learned_from_refusal")]
    report["query_bound_learned_from_the_binary"] = sorted(set(bound))
    print(f"\n  query_keep = {report['query_keep']['value']}  (deliberate, see bound_query docstring)")
    print(f"  query bound learned from the binary's own refusal: {sorted(set(bound))} bytes")

    path = out / "query_truncation.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
