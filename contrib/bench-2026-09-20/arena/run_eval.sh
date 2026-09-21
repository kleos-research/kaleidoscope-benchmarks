#!/bin/bash
# usage: run_eval.sh <run_id>
# Runs the harness's OWN evaluator (env/env_systems/formal_reasoning_env/eval.py, unmodified) --
# run_math.py has its auto-eval call commented out. Three passes: math, phys, and a combined
# directory holding copies of every paper's result.jsonl so each arm gets one number over all tasks.
set -euo pipefail
BASE=/path/to/kaleidoscope/experiments/bench-2026-09-20
RUN_ID=$1
OUT=$BASE/results/arena/$RUN_ID
PY=$BASE/venv-arena/bin/python
export TIKTOKEN_CACHE_DIR=$BASE/cache/tiktoken PYTHONDONTWRITEBYTECODE=1
rm -rf "$OUT/all"
for domain in math phys; do
  for arm_dir in "$OUT/$domain"/*/; do
    arm=$(basename "$arm_dir")
    for paper_dir in "$arm_dir"*/; do
      [ -f "$paper_dir/result.jsonl" ] || continue
      mkdir -p "$OUT/all/$arm/$(basename "$paper_dir")"
      cp "$paper_dir/result.jsonl" "$OUT/all/$arm/$(basename "$paper_dir")/result.jsonl"
    done
  done
done
printf '{"output": {"json_output_dir": "%s"}}\n' "$OUT/all" > "$OUT/configs/all_eval.json"
cd $BASE/MemoryArena
for cfg in "$OUT/configs/math_kscope.json" "$OUT/configs/phys_kscope.json" "$OUT/configs/all_eval.json"; do
  echo "##### eval.py $cfg"
  $PY env/env_systems/formal_reasoning_env/eval.py "$cfg"
done
