# Environment for the FULL-CONTEXT arm of MemoryAgentBench. Source AFTER mab/env.sh:
#   . mab/env.sh && . mab/env_fullctx.sh
# No secrets live here: common/llm.py reads the key from the repo .env itself.

# Its own spend scope, for two reasons. First, the assertion "this arm touched no memory system"
# is then a file that must NOT exist (logs/kscope_calls-mab-fullctx.jsonl), rather than a
# timestamp comparison inside the log the other arms already filled. Second, the arm's cost is the
# headline of the arm, and a scope is how the ledger separates it.
# The cap keeps the MemoryAgentBench benchmark inside ONE $200 envelope: scope `mab` stood at
# $58.00 pessimistic when this was written, so 140 leaves the two together under 200.
export BENCH_SCOPE=mab-fullctx
export BENCH_SCOPE_CAP_USD=140

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# A 296,153-token prompt with reasoning effort `high` is not a 3-second reader call. The default
# (90 s) and the full run's 180 s would both cut it, and a cut call is billed but never reaches the
# ledger -- it would look free and be counted as a failure.
export BENCH_LLM_TIMEOUT=600

# The gateway's published ceiling, read from x-ratelimit-limit-tokens on a deliberate 429
# (logs/mab_context_ladder.json): 500,000 tokens/min, 500 requests/min. One prompt of this arm is
# ~300k of that minute, so two cannot share a minute and the arm's floor is one question per
# minute. The target fraction leaves room for the OTHER benchmark running on the same key; the
# pacer reads its consumption out of the shared ledger rather than assuming it is idle.
export BENCH_TPM_LIMIT=500000
export BENCH_TPM_TARGET_FRACTION=0.90
export BENCH_PACE_MAX_WAIT=300
