# Kaleidoscope Benchmarks

Open-source evaluation suite for [Kaleidoscope](https://github.com/kleos-research/kaleidoscope),
an offline, filesystem-native memory runtime for AI agents.

The harness measures the memory system. It does not reimplement it — `kscope` is
assumed installed, and every operation shells out to the real binary, so what
gets measured is what ships.

## Benchmarks

| Suite | What it measures | Status |
| --- | --- | --- |
| **BEAM** | Ten memory abilities over conversations from 100K to 10M tokens | Supported |
| LongMemEval | Long-horizon question answering | Planned |
| LoCoMo | Long conversational memory | Planned |

## Quick start

Kaleidoscope must be installed and `kscope` on your `PATH`. Verify the build
carries an embedding model — without one the semantic retrieval channel is off
and the harness will refuse to run:

```bash
kscope model     # expects "status": "bundled"
```

Then:

```bash
git clone https://github.com/kleos-research/kaleidoscope-benchmarks
cd kaleidoscope-benchmarks
pip install -e .

cp .env.example .env      # add your OPENAI_API_KEY

# Fetch the BEAM tier you want into data/ — see benchmarks/beam/README.md
python -m kbench.benchmarks.beam.run all --tier 100K
```

Any OpenAI-compatible endpoint works — set `OPENAI_BASE_URL`.

## How it works

Four phases. Each writes its output to disk, and **nothing downstream reruns
anything upstream**.

```
ingest  ──►  answer  ──►  judge  ──►  report
  │            │            │
  │            │            └─ scores.jsonl   rubric + Kendall tau
  │            └────────────── answers.jsonl  retrieved context + hypothesis
  └─────────────────────────── ingest.json    memories written, edges, spend
```

**1. Ingest** — walk each conversation front to back, extract what each exchange
establishes, write it through `remember`. The extractor supplies the semantics;
Kaleidoscope never infers them. See [AGENTS.md](AGENTS.md).

**2. Answer** — for each question, `compile` returns the bounded context
Kaleidoscope itself assembled, and the reader answers from exactly that. The
harness does not re-render the hits: `compile`'s context carries graph paths,
contradiction flags and validity windows that re-rendering would discard.

**3. Judge** — one call per rubric item, plus normalised Kendall tau for
`event_ordering`, which BEAM does not score with a rubric.

**4. Report** — tables.

Run a phase on its own whenever you only want that phase:

```bash
python -m kbench.benchmarks.beam.run judge --judge-model gpt-4.1-mini
```

Re-judging costs judge calls only. Extraction is cached by prompt hash, so
editing the prompt re-pays for what changed and nothing else.

## Results

Measured numbers for 100K and 1M, per question and per ability, with the
configuration that produced each row: [docs/beam](docs/beam). Read that
README's first section before quoting anything from it — the comparison against
published work is not controlled for the reader model.

## Two scores, never averaged

| | |
| --- | --- |
| **evidence recall** | Model-free, from BEAM's own `source_chat_ids`. Retrieval quality alone — no reader, no judge, deterministic, free. |
| **BEAM score** | The judged rubric mean, comparable to published numbers. |

They answer different questions. A retrieval change can move one and not the
other, and if evidence recall rises while the BEAM score does not, retrieval was
not the bottleneck. Collapsing them into one column hides exactly that.

Use evidence recall to iterate — it costs nothing and cannot be perturbed by a
judge defect. Use the BEAM score to compare against published work.

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | required |
| `OPENAI_BASE_URL` | OpenAI | any compatible endpoint |
| `KBENCH_EXTRACTOR_MODEL` | `gpt-4.1` | writes memory |
| `KBENCH_READER_MODEL` | `gpt-4.1` | answers from retrieved context |
| `KBENCH_JUDGE_MODEL` | `gpt-4.1` | scores against the rubric |
| `KBENCH_COMPILE_LIMIT` | `100` | ceiling on memories per `compile` |
| `KBENCH_CONVERSATION_WORKERS` | `4` | conversations in flight |
| `KBENCH_QUESTION_WORKERS` | `4` | questions per conversation |
| `KSCOPE_BINARY` | `kscope` | path to the binary |

**Keep the reader identical across arms you intend to compare.** A reader
difference is indistinguishable from a memory difference in the final score, and
it is the easiest way to publish a number that means nothing.

**And keep the depth identical.** `KBENCH_COMPILE_LIMIT` is the same hazard one
step along. The default was `5` while BEAM's published comparisons read their
store at `top_50` and `top_200` — so a run asked Kaleidoscope for five memories,
asked everything else for a hundred, and reported the scores in one table. The
default is now `100`, inside the range published work reports rather than above
it. `compile` returns a bounded exposure, not a top-k slice: the limit is a
ceiling on what may be exposed, and it stops early when nothing further earns
its place, so raising it does not mechanically enlarge the context.

The judge is deliberately not tied to the reader. If it tracked the arm being
graded, judge quality and arm quality would be confounded.

## Parallelism

Conversations are independent — separate stores, and BEAM's evidence never
crosses conversations — so they run concurrently, as do the questions within
them. Total in-flight work is `conversation_workers x question_workers`.

Ingestion is the exception: chunks within a conversation are **ordered**, because
a `supersedes` can only name a memory already written. Writing turn 40 before
turn 12 loses the revision silently.

## A note on benchmark scores

Benchmark scores are not absolute numbers. They move with the extractor, the
reader, the judge, the retrieval depth, and the chunking. A number is only
meaningful beside the configuration that produced it, which is why every report
prints that configuration and every row carries the models that produced it.

Two specifics worth knowing before comparing anything:

- **The headline is the mean of the ten ability means**, as BEAM reports it — not
  the mean over questions. Abstention is one ability of ten, so a system that
  answers nothing scores 1.000 there and near zero elsewhere.
- **The write path bounds everything.** If the extractor declines to record an
  exchange, no retrieval configuration can recover it. The ingest report prints
  how many exchanges were judged not durable for exactly this reason.

## Project structure

```
kbench/
├── config.py                    environment, models, concurrency
├── llm.py                       one client, one retry policy, one spend ledger
├── kaleidoscope.py              kscope CLI wrapper, one vault per conversation
└── benchmarks/beam/
    ├── dataset.py               loading, and per-conversation scoping
    ├── extract.py               exchange -> semantic delta
    ├── ingest.py                phase 1
    ├── answer.py                phase 2
    ├── judge.py                 phase 3
    ├── metrics.py               model-free evidence recall
    ├── report.py                phase 4
    ├── run.py                   CLI
    └── prompts/                 extraction, reader, judge, tau alignment
AGENTS.md                        the memory-writing contract
CLAUDE.md                        how to work in this repository
```

## License

Apache 2.0.

BEAM is published by its own authors under its own terms and is not redistributed
here — the harness fetches it. See `kbench/benchmarks/beam/README.md`.
