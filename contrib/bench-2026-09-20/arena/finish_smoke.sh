#!/bin/bash
# Wait for the smoke orchestrator to exit, then run the whole verification chain.
set -uo pipefail
BASE=/path/to/kaleidoscope/experiments/bench-2026-09-20
RUN=$1
PID=$2
PY=$BASE/venv-arena/bin/python
# The benchmark owns its instrument. /opt/homebrew/bin/kscope was replaced mid-run on
# 2026-09-20 at 23:33:37 by another session; five vault roots of smoke-20260920t ran against
# the wrong bytes before it was caught. Never resolve the binary through the installed path.
export BENCH_KSCOPE=/path/to/kaleidoscope/experiments/binaries/kscope-361f5975d522083d
cd "$BASE"
while kill -0 "$PID" 2>/dev/null; do sleep 15; done
echo "ORCHESTRATOR EXITED"
tail -4 logs/arena_orch-$RUN.out
date +%s > results/arena/$RUN/end_epoch
echo "### eval.py"
BENCH_SCOPE=arena BENCH_SCOPE_CAP_USD=200 bash arena/run_eval.sh "$RUN" > logs/arena_eval-$RUN.out 2>&1 && echo "eval ok" || echo "EVAL FAILED rc=$?"
echo "### check_smoke"
BENCH_SCOPE=arena $PY arena/check_smoke.py "$RUN" probe- smoke-20260920a smoke-20260920b smoke-20260920c instrument-check preflight-20260920 smoke-20260920p smoke-20260920q smoke-20260920r smoke-20260920s smoke-20260920t smoke-20260920u adapter-repair > logs/arena_check-$RUN.out 2>&1 && echo "check ok" || echo "CHECK rc=$?"
echo "### check_isolation"
BENCH_SCOPE=arena $PY arena/check_isolation.py "$RUN" --selftest > logs/arena_isolation-$RUN.out 2>&1 && echo "isolation ok" || echo "ISOLATION rc=$?"
echo "### report_run"
BENCH_SCOPE=arena BENCH_BASELINE_ARM=kscope_full $PY arena/report_run.py "$RUN" --smoke-tasks math:8 math:15 math:17 math:22 math:26 math:28 math:37 phys:0 phys:3 phys:14 > logs/arena_report-$RUN.out 2>&1 && echo "report ok" || echo "REPORT rc=$?"
echo "### query_truncation"
BENCH_SCOPE=arena $PY arena/query_truncation.py "$RUN" > logs/arena_qtrunc-$RUN.out 2>&1 && echo "qtrunc ok" || echo "QTRUNC rc=$?"
echo "ALL DONE $RUN"
