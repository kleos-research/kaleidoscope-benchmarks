"""The owner's report shape, assembled from what check_smoke.py and report_run.py wrote.
Read-only; no LLM call, no vault. usage: summarise.py <run_id>"""
from __future__ import annotations
import json, sys
from pathlib import Path
BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")

def main(run_id: str) -> int:
    out = BASE / "results" / "arena" / run_id
    chk = json.loads((out / "check.json").read_text())
    rep = json.loads((out / "report.json").read_text())
    a = chk["assertions"]

    print(f"\n{'='*78}\nRUN {run_id}\n{'='*78}")
    print("\nASSERTIONS")
    for name in ("1_isolation", "2_personal_vault_untouched", "3_propagation", "4_no_leaks", "5_failures"):
        print(f"  {'GREEN' if a[name].get('pass') else 'RED  '}  {name}")

    k = [t for t in a["3_propagation"]["per_task"]]
    dup = sum(t["memories_on_disk"] - t["accepted"] for t in k)
    print(f"\nPROPAGATION  tasks={len(k)}  duplicates(on_disk - accepted)={dup}")
    bad = [t for t in k if not (t["chunks_in"] == t["accepted"] == t["memories_on_disk"]
                                == t["kscope_log_accepted"] == t["expected_chunks"])]
    print(f"  tasks where chunks_in==accepted==on_disk==log_accepted==subtasks+1: {len(k)-len(bad)}/{len(k)}")
    for t in bad:
        print(f"    MISMATCH {t['task']}: in={t['chunks_in']} acc={t['accepted']} disk={t['memories_on_disk']} "
              f"log={t['kscope_log_accepted']} expected={t['expected_chunks']}")

    f = a["5_failures"]
    print(f"\nFAILURES  refused={f['refused_items']} extraction={f['extraction_failures']} lost={f['chunks_lost']} "
          f"search={f['search_failures']} harness_exceptions={len(f['harness_exceptions'])} "
          f"fallback_stored={f['stored_under_fallback_delta']} prompt_shas={f['writer_prompt_shas']}")
    for r in f["refusal_reasons"]:
        print(f"    REFUSAL {r['task']} chunk {r['chunk_index']}: {r['refusal'][:150]}")
    c = rep["memory"]["adapter_counters_summed"]
    print(f"  adapter repairs: handles_shortened={c.get('handles_shortened')} "
          f"surfaces_dropped_on_collision={c.get('surfaces_dropped_on_collision')} "
          f"facts_dropped_with_surface={c.get('facts_dropped_with_surface')}")
    print(f"                   reserved_relation_items={c.get('reserved_relation_items')} "
          f"facts_dropped={c.get('reserved_relation_facts_dropped')} retry_accepted={c.get('reserved_retry_accepted')}")
    print(f"                   duplicate_surface_items={c.get('duplicate_surface_items')} "
          f"declarations_dropped={c.get('duplicate_surface_declarations_dropped')} "
          f"retry_accepted={c.get('duplicate_surface_retry_accepted')}")
    print(f"  full-history arm: chunks={c.get('history_chunks')} served_chunks={c.get('history_served_chunks')} "
          f"JUDGE MARKS IN SERVED={c.get('judge_marks_in_served')} (must be 0)")
    print(f"                   stored_with_degraded_delta={c.get('stored_with_degraded_delta')}  "
          f"(a kept-but-unfolded write; must be 0)")

    print("\nSCORES (subtask level)")
    for arm in ("kscope", "kscope_none", "kscope_full"):
        for split in ("all", "math", "phys"):
            s = rep["scores"][arm][split]
            if not s["tasks"]:
                continue
            print(f"  {arm:<12} {split:<5} tasks={s['tasks']:<3} subtasks={s['subtasks']:<4} "
                  f"pass={s['subtask_pass_rate']} ({s['subtasks_correct']}/{s['subtasks']})  "
                  f"progress={s['avg_progress_score']}  empty={s['empty_answers']}  degraded={s['degraded_steps']}")
    print("\nBENCHMARK'S OWN eval.py")
    for key, v in rep["eval_py"].items():
        print(f"  {key:<22} passrate={round(v['overall_average_passrate'],4)} progress={round(v['avg_progress_score'],4)} "
              f"avg_memory_length={round(v['average_memory_length'],1)}")

    p = rep["paired"]
    print(f"\nPAIRED (kscope - {rep.get('paired_baseline_arm','?')}), n={p['tasks_with_both_arms']} tasks")
    print(f"  mean per-task progress diff = {p['mean_progress_diff_kscope_minus_none']}  "
          f"95% CI {p['progress_diff_95ci_paired_bootstrap']}")
    print(f"  subtask-rate diff 95% CI (cluster bootstrap) {p['subtask_rate_diff_95ci_cluster_bootstrap']}")
    print(f"  kscope better {p['tasks_kscope_better']} | none better {p['tasks_none_better']} | tied {p['tasks_tied']}")

    d = rep.get("paired_degraded_dropped_pairwise") or {}
    if d.get("mean_progress_diff") is not None:
        print(f"\nPAIRED, degraded subtasks dropped PAIRWISE from both arms")
        print(f"  dropped {d['subtasks_dropped']} subtasks, kept {d['subtasks_kept']}, "
              f"{d['tasks_with_any_clean_subtask']} tasks retain a clean subtask")
        print(f"  mean per-task progress diff = {d['mean_progress_diff']}  "
              f"95% CI {d['progress_diff_95ci_paired_bootstrap']}")
        print(f"  subtask rates: kscope {d['kscope_subtask_rate']} vs baseline {d['baseline_subtask_rate']}  "
              f"rate-diff CI {d['subtask_rate_diff_95ci_cluster_bootstrap']}")

    m = rep["memory"]
    print(f"\nMEMORY  searches={m['searches']} mean_served={m['mean_served_count']} "
          f"mean_served_bytes={m['mean_served_bytes']} abstained={m['abstained']}")
    print(f"  preceding subtask's record served: {m['preceding_subtask_record_served']}/"
          f"{m['searches_after_first_subtask']} ({m['preceding_subtask_record_served_rate']})")
    print(f"  any earlier record served rate: {m['any_earlier_record_served_rate']}")
    print(f"  omission reasons: {m['omission_reasons']}")

    print(f"\nDEGRADED STEPS  {rep['degraded_steps_by_arm']}  per subtask {rep['degraded_steps_per_subtask_by_arm']}")
    for arm, causes in rep["degraded_step_causes"].items():
        print(f"  {arm}: {causes}")

    print("\nLLM / COST")
    tot_l = tot_p = 0.0
    for arm in ("kscope", "kscope_none", "kscope_full"):
        l = rep["llm"][arm]
        tot_l += l["usd_list_0.20_1.20"]; tot_p += l["usd_pessimistic_1.10_6.60"]
        print(f"    prompt tokens: {l.get('mean_prompt_tokens_per_agent_call')}/agent call, "
              f"{l.get('agent_prompt_tokens_per_subtask')}/subtask (agent role only)")
        print(f"  {arm:<12} calls={l['calls']:<5} in={l['tokens_in']:>9,} out={l['tokens_out']:>9,} "
              f"length_finishes={l['calls_finishing_on_length']:<4} "
              f"out_burned_on_length={l['share_of_output_tokens_burned_on_length']:.1%}  "
              f"${l['usd_list_0.20_1.20']:.2f} list / ${l['usd_pessimistic_1.10_6.60']:.2f} pessimistic")
        print(f"               failed={l['failed_calls']} {l['failed_by_role_and_error']}")
    print(f"  RUN TOTAL: ${tot_l:.2f} at Azure list ($0.20/$1.20) | ${tot_p:.2f} at the worst reported rate ($1.10/$6.60)")
    w = rep["wall"]
    print(f"\nWALL  {w['run_wall_h']} h wall, {w['sequential_equivalent_h']} h sequential-equivalent "
          f"({round(w['sequential_equivalent_h']/w['run_wall_h'],1)}x)")
    print(f"  spend ledger: {a['7_spend']['llm_report']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
