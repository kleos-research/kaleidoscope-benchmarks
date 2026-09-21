# MemoryAgentBench

**Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions** —
Yuanzhe Hu, Yu Wang and Julian McAuley, ICLR 2026.
[Paper](https://arxiv.org/abs/2507.05257) ·
[Dataset](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) ·
[Code](https://github.com/HUST-AI-HYZ/MemoryAgentBench)

MemoryAgentBench feeds an agent a long history in chunks and then asks questions
about it. This harness runs its **FactConsolidation** task: the history is a
list of facts, some of which are later overwritten, and the agent must answer
with the current value. Single-hop questions ask about one fact; multi-hop
questions chain two to four.

## Results

On FactConsolidation at 262K tokens, with GPT-5.6 Luna as the reader in every
arm:

| | single-hop accuracy | reader tokens per question |
| --- | ---: | ---: |
| entire history in the prompt | 85 | 291,746 |
| **Kaleidoscope** | **90** | **1,195** |
| BM25 keyword search | 78 | 2,066 |
| no memory | 13 | 189 |

It loses to BM25 on multi-hop questions at every length. Every length, the
token comparison, the losses and what the numbers can and cannot support:
[docs/memoryagentbench](../../../docs/memoryagentbench/).

## Setup

This harness plugs a Kaleidoscope adapter into MemoryAgentBench's own code
rather than reimplementing it. From the root of this repository:

```bash
# 1. MemoryAgentBench, at the commit the adapter was written against
git clone https://github.com/HUST-AI-HYZ/MemoryAgentBench results/memoryagentbench/MemoryAgentBench
git -C results/memoryagentbench/MemoryAgentBench checkout fe1735d
git -C results/memoryagentbench/MemoryAgentBench apply \
    "$PWD/kbench/benchmarks/memoryagentbench/upstream.patch"
pip install -r results/memoryagentbench/MemoryAgentBench/requirements.txt

# 2. The FactConsolidation data
huggingface-cli download --repo-type dataset ai-hyz/MemoryAgentBench \
    data/Conflict_Resolution-00000-of-00001.parquet --local-dir results/memoryagentbench
```

`kscope` must be installed and carry its embedding model — check that
`kscope model` reports `"status": "bundled"`. Without the model, semantic
retrieval is off and the harness refuses to run.

Everything the run writes goes under `results/memoryagentbench/`. Set
`MAB_WORKDIR` to put it somewhere else.

## Running

Each run needs a spend cap for the model calls it makes:

```bash
export BENCH_SCOPE=memoryagentbench BENCH_SCOPE_CAP_USD=200

# every arm at every length
python -m kbench.benchmarks.memoryagentbench.run --run_id first --lengths 6k,32k,64k,262k

# the full-context arm, reading the history that run ingested
python -m kbench.benchmarks.memoryagentbench.run_full_context --run_id first-full --ingest_run_id first

# scores, recall and the tables above
python -m kbench.benchmarks.memoryagentbench.check first --out report.md
```

The full 800-question run cost about $10 at list price. The full-context arm
sends roughly 290,000 tokens per question, so it costs far more than every other
arm combined. Try a handful of questions first with `--max_queries 5`.

## What's in this directory

| file | what it does |
| --- | --- |
| `run.py` | runs every arm at every length |
| `run_arm.py` | runs one arm on one history, through MemoryAgentBench's own entry point |
| `run_full_context.py` | the whole-history control |
| `ingest.py` | writes a history into a fresh Kaleidoscope store, once per length |
| `check.py` | scores a run and builds the tables, independently of the code that ran it |
| `score_partial.py` | scores a run that has not finished |
| `kscope_io.py` | runs the `kscope` binary, and refuses to touch your own memory store |
| `writer.py` | turns a chunk of history into a memory Kaleidoscope can store |
| `llm_client.py` | the model client, with the spend cap |
| `upstream.patch` | the Kaleidoscope adapter and its four arm configurations |

`check.py` re-derives every question's answer chain from the dataset instead of
trusting the harness's own bookkeeping. A checker that shares code with the
thing it checks can share its mistakes.

## Status

Unlike BEAM, this harness does not yet use this repository's candidate and
contract binding (`--candidate`, `--public-contract` and their digests). It
resolves `kscope` from your `PATH`, or from `BENCH_KSCOPE`, and pins it by hash
only if you set `BENCH_KSCOPE_SHA`. It re-verifies the binary whenever the file
changes, so a binary replaced mid-run is caught rather than trusted.

The published numbers were produced by this code before it moved into this
repository, and have not yet been re-run from here. See
[docs/memoryagentbench](../../../docs/memoryagentbench/) for that and the other
caveats.
