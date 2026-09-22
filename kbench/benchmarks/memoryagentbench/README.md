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

On FactConsolidation at 262K tokens, over 100 single-hop and 100 multi-hop
questions, with GPT-5.6 Luna as the reader in every setup:

| | single-hop | multi-hop | total tokens | cost |
| --- | ---: | ---: | ---: | ---: |
| entire history in the prompt | 85 | **25** | 58.4M | $11.71 |
| **Kaleidoscope** (writing + reading) | **90** | 10 | 9.3M | $6.34 |
| BM25 keyword search | 78 | 11 | 0.46M | **$0.14** |

Tokens and cost cover all 200 questions, at list price: $0.20 per million input
tokens and $1.20 per million output tokens. Most of Kaleidoscope's cost is
writing memory, which it pays once per history. For reading alone, it sends
1,195 tokens per single-hop question against 291,746 for the whole history.
BM25 writes nothing, so it is far cheaper than both.

The Kaleidoscope and BM25 figures come from `check.py` (see Running, below), and
the whole-history figures from `score_partial.py`. Dollars are tokens times the
list price.

Kaleidoscope loses on multi-hop questions: to BM25 at every length, and to the
whole history at 262K. Every length, the costs, the losses and the command
behind each number: [docs/memoryagentbench](../../../docs/memoryagentbench/).

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
pip install huggingface_hub     # provides the `hf` command
hf download ai-hyz/MemoryAgentBench data/Conflict_Resolution-00000-of-00001.parquet \
    --repo-type dataset --local-dir results/memoryagentbench

# 3. The model endpoint
cp .env.example .env            # then set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env
```

Every model call goes to GPT-5.6 Luna on Azure OpenAI, through a deployment
named `gpt-5.6-luna`. The harness reads the endpoint and key from `.env` in the
root of this repository, and that file must exist.

`kscope` must be installed with its built-in model: check that `kscope model`
reports `"status": "bundled"`. The harness refuses to run without it.

## Running

The harness needs a spend cap, and three cache directories inside its working
directory:

```bash
export BENCH_SCOPE=mab BENCH_SCOPE_CAP_USD=200
export HF_HOME="$PWD/results/memoryagentbench/cache/hf_home"
export NLTK_DATA="$PWD/results/memoryagentbench/cache/nltk_data"
export TIKTOKEN_CACHE_DIR="$PWD/results/memoryagentbench/cache/tiktoken"
```

Try one short history and a few questions first:

```bash
python -m kbench.benchmarks.memoryagentbench.run --run_id smoke --lengths 6k --max_queries 5
```

Then the full run:

```bash
# every setup at every length
python -m kbench.benchmarks.memoryagentbench.run --run_id first --lengths 6k,32k,64k,262k

# the whole-history setup, reading the history that run stored
python -m kbench.benchmarks.memoryagentbench.run_full_context --run_id first-full --ingest_run_id first

# scores, retrieval, tokens and cost
python -m kbench.benchmarks.memoryagentbench.check first --out report.md
```

Everything a run writes goes under `results/memoryagentbench/`. To put it
somewhere else, set `MAB_WORKDIR`, and move the three cache directories under it
too.

The published run cost $9.91 at list price for all four lengths, including
Kaleidoscope's memory writing. The whole-history setup cost another $11.71 for
its 200 questions, because each question sends it about 292,000 tokens. The
spend cap is checked against a deliberately high price, $1.10 per million input
tokens and $6.60 per million output tokens, so it stops a run early rather than
late.

## What's in this directory

| file | what it does |
| --- | --- |
| `run.py` | runs every setup at every length |
| `run_arm.py` | runs one setup on one history, through MemoryAgentBench's own entry point |
| `run_full_context.py` | the whole-history setup |
| `ingest.py` | writes a history into a fresh Kaleidoscope store, once per length |
| `check.py` | scores a run and builds the tables, independently of the code that ran it |
| `score_partial.py` | scores the whole-history setup from its log, even before it finishes |
| `kscope_io.py` | runs the `kscope` binary, and refuses to touch your own memory store |
| `writer.py` | turns a piece of history into a memory Kaleidoscope can store |
| `llm_client.py` | the model client, with the spend cap |
| `upstream.patch` | the Kaleidoscope adapter and its four setups |

`check.py` re-derives every question's answer chain from the dataset instead of
trusting the harness's own bookkeeping. A checker that shares code with the
thing it checks can share its mistakes.

## Status

Unlike BEAM, this harness does not pin a `kscope` build with a public contract.
It uses the `kscope` on your `PATH`, or the one `BENCH_KSCOPE` names, and pins
it by hash only if you set `BENCH_KSCOPE_SHA`. It checks the binary again
whenever the file changes, so a binary replaced mid-run is caught.

The published numbers were produced by this code before it moved into this
repository, and have not yet been re-run from here. See
[docs/memoryagentbench](../../../docs/memoryagentbench/) for that and the other
caveats.
