"""Reproduce every number on docs/continual-learning-bench/README.md.

The runs are public in the benchmark's own format, in this pull request's
branch:
https://github.com/pgasawa/continual-learning-bench/pull/23

    git clone -b kaleidoscope-gpt-5.6-luna-results \
        https://github.com/parthpahwa1/continual-learning-bench
    cd continual-learning-bench && uv sync
    uv run python /path/to/kaleidoscope-benchmarks/docs/continual-learning-bench/analyze.py \
        --bench . all

Everything here reads recorded fields: scores, rewards, the action each step
took, and whether an instance was solved. Nothing is inferred from the model's
text.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

KALEIDOSCOPE = "kaleidoscope-gpt-5.6-luna"
ICL = "icl-gpt-5.6-luna"
REFERENCE = "icl-gpt-5.4"
TASKS = {
    "blind_spectrum_monitoring": "radio mapping",
    "codebase_adaptation": "bug fixing",
    "cohort_studies": "medical cohorts",
    "database_exploration": "database questions",
    "exploitable_poker": "poker",
    "sales_prediction": "sales forecasting",
}
OTHER_DATABASE_RUNS = Path(__file__).with_name("other-database-runs.json")
T_975_DF4 = 2.7764451051977987  # two-sided 95% t quantile, 4 degrees of freedom


def payload(bench: Path) -> dict:
    """The benchmark's own analysis, from its own scorer."""
    sys.path.insert(0, str(bench))
    sys.path.insert(0, str(bench / "scripts"))
    import analyze_final_results  # noqa: PLC0415

    return analyze_final_results.build_analysis_payload(
        results_dir=bench / "final_results" / "runs", runs=[KALEIDOSCOPE, ICL, REFERENCE]
    )


def rollouts(bench: Path, run: str, task: str) -> list[dict]:
    """Each rollout's trace: instance outcomes and every step."""
    path = bench / "final_results" / "runs" / run / "tasks" / f"{task}.json.gz"
    artifact = json.loads(gzip.open(path).read())
    traces = [item["trace"] for item in artifact["run_traces"]]
    return sorted(traces, key=lambda t: (t.get("execution") or {}).get("run_index", 0))


def outcomes(trace: dict) -> list[dict]:
    return trace.get("instance_outcomes") or (trace.get("result") or {}).get("instance_outcomes") or []


def mean_interval(values: list[float]) -> tuple[float, float, float]:
    m = statistics.mean(values)
    half = T_975_DF4 * statistics.stdev(values) / math.sqrt(len(values))
    return m, m - half, m + half


def cmd_scores(bench: Path) -> None:
    p = payload(bench)
    rows = {(r["run_name"], r["task"]): r for r in p["task_summary"]}
    print("Score (normalized reward) and gain (normalized gain), x100")
    print(f"{'task':20} {'score K':>8} {'score ICL':>9} {'gain K':>7} {'gain ICL':>8}")
    for task, name in TASKS.items():
        k, i = rows[(KALEIDOSCOPE, task)], rows[(ICL, task)]
        print(f"{name:20} {100 * k['normalized_reward_mean']:8.1f} {100 * i['normalized_reward_mean']:9.1f} "
              f"{100 * k['normalized_gain_mean']:7.1f} {100 * i['normalized_gain_mean']:8.1f}")
    board = {r["run_name"]: r for r in p["leaderboard"]}
    for run in (KALEIDOSCOPE, ICL):
        print(f"average {run}: score {100 * board[run]['normalized_reward_mean']:.1f}, "
              f"gain {100 * board[run]['normalized_gain_mean']:.1f}")
    points = [x for x in p["run_points"] if x["run_name"] == ICL and x["task"] == "codebase_adaptation"]
    per_rollout = [round(100 * x["normalized_reward"], 1) for x in sorted(points, key=lambda x: x["run_index"])]
    print(f"ICL bug fixing, per rollout: {per_rollout}")


def cmd_database(bench: Path) -> None:
    task = "database_exploration"
    for run in (KALEIDOSCOPE, ICL):
        solved = queries_on_solved = 0
        last_ten = 0
        for trace in rollouts(bench, run, task):
            outs = outcomes(trace)
            solved_ids = {o["instance_id"] for o in outs if o.get("success")}
            solved += len(solved_ids)
            last_ten += sum(1 for o in outs[-10:] if o.get("success"))
            for step in trace.get("interactions") or []:
                action = str(((step.get("response") or {}).get("action") or {}).get("action", "")).upper()
                if action == "QUERY" and (step.get("query") or {}).get("instance_id") in solved_ids:
                    queries_on_solved += 1
        print(f"{run}: solved {solved} of 200; {queries_on_solved / solved:.1f} exploratory queries per "
              f"solved question; last ten questions of each rollout: {last_ten} of 50 solved")


def cmd_poker(bench: Path) -> None:
    task = "exploitable_poker"
    p = payload(bench)
    points = {(x["run_name"], x["run_index"]): x["normalized_reward"] for x in p["run_points"] if x["task"] == task}
    diffs = [100 * (points[(KALEIDOSCOPE, i)] - points[(ICL, i)]) for i in range(5)]
    m, lo, hi = mean_interval(diffs)
    print(f"Kaleidoscope minus ICL, per rollout: {[round(d, 1) for d in diffs]}; "
          f"mean {m:.1f}, 95% interval {lo:.1f} to {hi:.1f}")
    # Rewards per hand, summed over the five rollouts. Rollout i of each setup plays the same hands.
    totals: dict[str, dict[str, float]] = {}
    for run in (KALEIDOSCOPE, ICL):
        by_hand: dict[str, float] = {}
        for trace in rollouts(bench, run, task):
            for o in outcomes(trace):
                by_hand[o["instance_id"]] = by_hand.get(o["instance_id"], 0.0) + o["reward"]
        totals[run] = by_hand
    hands = sorted(totals[ICL])
    gap = {h: totals[ICL][h] - totals[KALEIDOSCOPE][h] for h in hands}
    whole = sum(gap.values())
    top = sorted(hands, key=lambda h: gap[h], reverse=True)[:3]
    rest = [h for h in hands if h not in top]
    print(f"ICL's lead over all {len(hands)} hands, five rollouts: {whole:.0f} big blinds; "
          f"on the three hands with the largest gap: {sum(gap[h] for h in top):.0f}")
    for run in (KALEIDOSCOPE, ICL):
        per_hand = sum(totals[run][h] for h in rest) / (5 * len(rest))
        print(f"{run}: {per_hand:+.2f} big blinds per hand on the other {len(rest)} hands")


def cmd_other_database_runs(bench: Path) -> None:
    p = payload(bench)
    base = p["reward_normalization_baselines"]["database_exploration"]
    data = json.loads(OTHER_DATABASE_RUNS.read_text())
    for name, run in data["runs"].items():
        scores = [(sum(r) - base) / (len(r) - base) for r in run["rollout_rewards"]]
        print(f"{name}: {100 * statistics.mean(scores):.1f} ({run['description']})")


COMMANDS = {
    "scores": cmd_scores,
    "database": cmd_database,
    "poker": cmd_poker,
    "other-database-runs": cmd_other_database_runs,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, required=True, help="a checkout of the PR branch above")
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    args = parser.parse_args()
    for name, fn in COMMANDS.items():
        if args.command in (name, "all"):
            print(f"== {name}")
            fn(args.bench.resolve())


if __name__ == "__main__":
    main()
