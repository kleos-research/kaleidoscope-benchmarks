# Environment for every MemoryAgentBench process started from this directory.
# No secrets live here: common/llm.py reads the key from the repo .env itself.
# Source it:  . mab/env.sh
BENCH_BASE=/path/to/kaleidoscope/experiments/bench-2026-09-20

# Spend accounting: every LLM call made under this scope counts against this cap.
export BENCH_SCOPE=mab
export BENCH_SCOPE_CAP_USD=4

# Every cache under the base dir, so nothing lands in ~/.cache, ~/nltk_data or $TMPDIR.
export HF_HOME=$BENCH_BASE/cache/mab/hf_home
export NLTK_DATA=$BENCH_BASE/cache/mab/nltk_data
export TIKTOKEN_CACHE_DIR=$BENCH_BASE/cache/mab/tiktoken
