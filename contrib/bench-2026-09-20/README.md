# `bench-2026-09-20` — MemoryAgentBench and MemoryArena harness

The harness that produced
[`docs/reports/memoryagentbench-2026-09.md`](../../docs/reports/memoryagentbench-2026-09.md).

**This is contributed code and it does not yet conform to this repository's
`kbench` shape.** It is published because the results are published and a result
without its harness is an assertion. Read it as evidence, not as an example to
copy.

## What is here

| directory | what it does |
| --- | --- |
| `common/` | binary resolution with a hash pin, vault isolation guards, the LLM client with spend caps, the memory writer |
| `mab/` | MemoryAgentBench drivers, the master-corpus ingest, and an independent checker that re-derives every chain from the dataset parquet |
| `arena/` | MemoryArena orchestrator, the memory and environment servers, task drivers, and isolation checkers |
| `patches/` | the two upstream benchmarks' integration points |
| `notes/` | the rules both harnesses share |

## Before you run it

**Paths are not portable.** Every `BASE`, `ROOT` and vault path in this tree was
an absolute path on the author's machine and has been rewritten to
`/path/to/kaleidoscope`. Nothing runs until you replace them. Making these
configurable is the first thing to fix.

**`kscope` must be installed and carry an embedding model.** Verify with
`kscope model` reporting `"status": "bundled"` — a model-less build is a
different system rather than a slower one, and it fails quietly.

**Any ranked search writes.** Every retrieval records an exposure row, so a
measurement writes into the store it measures. The harness clones each vault
copy-on-write before querying it; do not route around that.

## What it does differently from `kbench`, and why that matters

- **It is not four-phase.** `ingest -> answer -> judge -> report` are not
  cleanly decoupled here, so a judge experiment can cost answer calls. This is
  the main thing that would need restructuring.
- **It pins the binary by hash per call.** Worth keeping in any port. A check
  that memoises its result cannot see a binary replaced mid-run, which is a
  failure this campaign experienced and did not catch.
- **It drops provider-degraded questions pairwise** across the arms being
  compared, rather than per-arm, so a provider refusing one arm's prompts more
  often than another's cannot masquerade as a memory difference.
- **The checker is independent of the producer.** It re-derives chains from the
  dataset rather than trusting the harness's own bookkeeping. Keep that property
  in any port: independence is what makes a checker evidence.
