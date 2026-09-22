# BEAM

**Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs** —
Mohammad Tavakoli, Alireza Salemi, Carrie Ye, Mohamed Abdalla, Hamed Zamani and
J. Ross Mitchell, ICLR 2026.
[Paper](https://arxiv.org/abs/2510.27246) ·
[Dataset](https://huggingface.co/datasets/Mohammadta/BEAM) ·
[Code](https://github.com/mohammadtavakoli78/BEAM)

BEAM tests ten memory abilities over long conversations, from 100K to 10M
tokens. This harness writes each conversation into Kaleidoscope, asks BEAM's
questions, and scores the answers. Published results are in
[docs/beam](../../../docs/beam/).

## What you need

- **A `kscope` build and its public contract.** The contract is a JSON file
  generated for each build. It records the build's SHA-256 and the tools the
  build offers. The npm package doesn't include it yet, so for now you can run
  this harness only against a build that came with its contract.
- **An API key for an OpenAI-compatible endpoint**, for the models that write
  memory, answer and judge.
- **The BEAM data**, fetched as below.

## Getting the data

BEAM is published by its authors under CC BY-SA 4.0 and is not redistributed
here. Fetch the tier you want and save it where the harness looks for it,
`data/beam-<tier>.parquet`:

```bash
pip install huggingface_hub     # provides the `hf` command
hf download Mohammadta/BEAM data/100K-00000-of-00001.parquet \
    --repo-type dataset --local-dir data
mv data/data/100K-00000-of-00001.parquet data/beam-100K.parquet
```

The same dataset has `500K` and `1M` files. The 10M tier is published
separately, as [Mohammadta/BEAM-10M](https://huggingface.co/datasets/Mohammadta/BEAM-10M),
in two files that you would need to combine into `data/beam-10M.parquet`.

## Pinning the build

Every run is tied to one `kscope` build, so a score always names the build that
produced it. You give the harness four things:

| flag | what it is |
| --- | --- |
| `--candidate` | the `kscope` binary to measure |
| `--candidate-sha256` | that binary's SHA-256 |
| `--public-contract` | the build's public contract, a JSON file |
| `--public-contract-sha256` | the contract's SHA-256 |

`shasum -a 256 <file>` prints a file's SHA-256. Before it creates anything, the
harness checks that both files match their digests, that the contract describes
this exact binary, and that it lists exactly the two agent tools, `remember` and
`search`. It also refuses a build that is missing its built-in model. If any
check fails, it stops.

Pinning proves which build was measured. It does not prove that the build is a
signed, official release.

## Running

```bash
cp .env.example .env          # then set OPENAI_API_KEY in .env

python -m kbench.benchmarks.beam.run all --tier 100K \
  --candidate /path/to/kscope \
  --candidate-sha256 <sha256 of kscope> \
  --public-contract /path/to/kaleidoscope-public-contract.json \
  --public-contract-sha256 <sha256 of the contract>
```

A run has four phases, and you can run each one on its own. `ingest` and
`answer` need the four pinning flags, with the same values each time. `judge`
and `report` need no flags: they refuse inputs that changed after they were
written, and `report` refuses phases that came from different builds.

```bash
python -m kbench.benchmarks.beam.run ingest --tier 100K <pinning flags>
python -m kbench.benchmarks.beam.run answer --tier 100K <pinning flags>
python -m kbench.benchmarks.beam.run judge  --tier 100K
python -m kbench.benchmarks.beam.run report --tier 100K
```

`python -m kbench.benchmarks.beam.run --help` lists every option.

## How a run works

```
ingest  ──►  answer  ──►  judge  ──►  report
  │            │            │
  │            │            └─ scores.jsonl   rubric scores, and Kendall tau
  │            └────────────── answers.jsonl  retrieved context and answer
  └─────────────────────────── ingest.json    memories written, and spend
```

Each phase writes its output under `results/beam/<tier>/`, and nothing
downstream reruns anything upstream. Changing the judge costs judge calls only,
and a crash while judging never loses the answers.

1. **Ingest** walks each conversation from start to end. A model reads two
   messages at a time (`--chunk-size`) and states what they establish as a
   structured memory, which the harness writes to Kaleidoscope. The writer
   supplies the meaning; Kaleidoscope does not infer it. The format is in
   [AGENTS.md](../../../AGENTS.md).
2. **Answer** runs one search per question and gives the reader exactly the
   context Kaleidoscope returns. That context carries more than the text of
   the memories, such as which ones contradict each other and when each fact
   held, so the harness passes it on unchanged.
3. **Judge** scores each answer against BEAM's rubric, one model call per
   rubric item. `event_ordering` is scored by normalised Kendall tau instead,
   as BEAM does.
4. **Report** prints the tables and writes `report.md`.

Extraction is cached by prompt, so editing the extraction prompt pays again
only for what changed.

Each conversation gets its own Kaleidoscope profile and memory store, so
conversations run in parallel, and so do the questions within one. Messages
within a conversation are written in order, because a later fact can correct an
earlier one. Restarting between phases reopens the same stores. Stores and
caches are keyed to the pinned build, so a different build never reuses them.
Result files record the build's digests, but no local paths.

## Configuration

Set these in `.env` or in the environment.

| variable | default | what it controls |
| --- | --- | --- |
| `OPENAI_API_KEY` | none | required |
| `OPENAI_BASE_URL` | OpenAI | any OpenAI-compatible endpoint |
| `KBENCH_EXTRACTOR_MODEL` | `gpt-4.1` | the model that writes memory |
| `KBENCH_READER_MODEL` | `gpt-4.1` | the model that answers from the retrieved context |
| `KBENCH_JUDGE_MODEL` | `gpt-4.1` | the model that scores answers against the rubric |
| `KBENCH_TOP_K` | `8` | memories retrieved per question (`--top-k`) |
| `KBENCH_MAXIMUM_CONTEXT_BYTES` | `32768` | size limit of the retrieved context (`--maximum-context-bytes`) |
| `KBENCH_CONVERSATION_WORKERS` | `4` | conversations in flight |
| `KBENCH_QUESTION_WORKERS` | `4` | questions in flight per conversation |
| `KBENCH_JUDGE_WORKERS` | `8` | judge calls in flight |
| `KBENCH_DATA_DIR` | `data/` | the data, memory stores and caches |
| `KBENCH_RESULTS_DIR` | `results/` | where results are written |

At most `conversation workers × question workers` questions run at once. Raise
both until the endpoint pushes back.

To compare two runs fairly:

- **Use the same reader.** A difference in the reader looks exactly like a
  difference in memory in the final score.
- **Use the same `--top-k` and `--maximum-context-bytes`.** They decide how much
  the reader sees, so runs are comparable only at the same values. The harness
  records both with every run.
- **Keep the judge independent of the setup being graded.** If the judge
  followed the setup, judge quality and setup quality would be mixed together.
  That is why the judge is configured separately from the reader.

## Reading the output

The report gives two scores and never averages them:

| score | what it measures |
| --- | --- |
| **evidence recall** | whether retrieval found the messages BEAM labels as evidence (its `source_chat_ids`). No model is involved, so it is free and deterministic. |
| **BEAM score** | the judged rubric mean, comparable to published results |

A retrieval change can move one and not the other. If evidence recall rises and
the BEAM score does not, retrieval was not the bottleneck. Use evidence recall
to iterate, because it costs nothing and no judge can skew it. Use the BEAM
score to compare with published work.

Before comparing a BEAM score with anything:

- **The headline is the mean of the ten ability means**, as BEAM reports it,
  not the mean over questions. Abstention is one of the ten, and a system that
  answers nothing scores 1.000 on it, so read the per-ability table first.
- **Scores move with every model and setting**: the extractor, the reader, the
  judge, the retrieval depth and the chunk size. Every report prints its
  configuration, and every row records the models that produced it.
- **Writing bounds everything.** If the extractor declines to record an
  exchange, no retrieval setting can recover it. The ingest report counts the
  exchanges that were judged not worth keeping.

## Two properties of the dataset

**Questions are asked after the conversation, not during it.** In the 100K
tier, 252 of 400 questions have evidence spanning more than one message, the
widest span is 262 messages, and some questions need evidence from as many as
five sessions. So the harness builds the whole memory before it asks anything.

**Message ids are numbered per conversation.** In the 100K tier, 392 distinct
ids cover 5,732 messages, so message 14 of one conversation and message 14 of
another are unrelated. Mixing them up would score evidence against the wrong
conversation and inflate recall, with no error. The loader checks that every
conversation is kept separate, and stops if one is not.

## Checking a build without a BEAM run

Two checks that need a build and its contract, but no BEAM data and no API
key. Neither produces a score.

**A smoke test of the real binary.** The ordinary tests use a stand-in for
`kscope`. This one runs the build itself: it creates a temporary profile and
memory store, writes one memory, and finds it again by search and by id.

```bash
export KBENCH_LIVE_CANDIDATE=/absolute/path/to/kscope
export KBENCH_LIVE_CANDIDATE_SHA256=<sha256 of kscope>
export KBENCH_LIVE_PUBLIC_CONTRACT=/absolute/path/to/kaleidoscope-public-contract.json
export KBENCH_LIVE_PUBLIC_CONTRACT_SHA256=<sha256 of the contract>
python -m pytest tests/test_live_candidate_smoke.py -q
```

**Every phase on a tiny synthetic corpus.** This runs write, search, answer,
judge and report on three made-up questions, with no model: the answers and
the judging follow fixed rules. It works only with the one build it was written
for, whose digests are in [`fixture.py`](fixture.py), and refuses any other.
Its output directory must be new and outside this repository.

```bash
python -m kbench.benchmarks.beam.fixture \
  --run-root /tmp/kbench-fixture-run \
  --candidate /absolute/path/to/kscope \
  --candidate-sha256 <sha256 of kscope> \
  --public-contract /absolute/path/to/kaleidoscope-public-contract.json \
  --public-contract-sha256 <sha256 of the contract>
```

It writes a record that ties every output file to the build, the contract and
the corpus. Passing shows that the harness and the build work together; it
says nothing about how well Kaleidoscope does on BEAM.
