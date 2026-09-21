# Environment for the FULL MemoryAgentBench run (2026-09-20). Source AFTER mab/env.sh:
#   . mab/env.sh && . mab/env_full.sh
# No secrets live here: common/llm.py reads the key from the repo .env itself.

# Owner's instruction 2026-09-20: a safety cap of $200 per benchmark scope. env.sh still says 4
# (the cap while the harness was unproven); common/llm.py enforces whatever this says, at the
# pessimistic $1.10/$6.60 rate, before every call.
export BENCH_SCOPE=mab
export BENCH_SCOPE_CAP_USD=200

# The dataset is in the HF cache under this directory (cache/mab/hf_home); 25 harness launches must
# not each ask the hub whether revision "main" moved.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# The writer's client timeout. common/llm.py defaults BENCH_LLM_TIMEOUT to 90 s, chosen for the
# arena's tool-bearing calls. Writer calls at the 12-sentence unit rule measured p50 27 s, p90
# 41-52 s, max 62 s over 78 calls (ledger, runs gate/gatefix); a 90 s deadline would cut the tail
# of ~2,200 calls into SDK retries whose abandoned generations are billed but never reach the
# ledger. 180 s is 3x the measured max. Reader calls (p99 3 s) are unaffected by either value.
export BENCH_LLM_TIMEOUT=180
