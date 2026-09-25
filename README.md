# Kaleidoscope Benchmarks

Benchmark harnesses for [Kaleidoscope](https://memory.kleosresearch.xyz/), a
local memory store for AI agents, and the results they produced.

An AI agent forgets what it learned when its session ends, so the next session
repeats the same work and the same mistakes. Kaleidoscope keeps that memory as
files on the user's own disk. This repository measures how well it works: it
runs Kaleidoscope through published academic benchmarks and compares it with
simpler options, such as keyword search or pasting the whole history into the
prompt. Every run calls the real `kscope` binary rather than a reimplementation,
so the numbers describe what ships, and the losses are reported beside the wins.

## Results

### MemoryAgentBench

[MemoryAgentBench](https://arxiv.org/abs/2507.05257) gives an agent a long
history of facts, some of them later overwritten, and then asks for the current
values. Single-hop questions ask about one fact; multi-hop questions chain two
to four. At the longest history, 262K tokens, with the same answering model
(GPT-5.6 Luna) in every setup:

| setup | single-hop | multi-hop | total tokens | cost |
| --- | ---: | ---: | ---: | ---: |
| entire history in the prompt | 85 | **25** | 58.4M | $11.71 |
| **Kaleidoscope** (writing + reading) | **90** | 10 | 9.3M | $6.34 |
| BM25 keyword search | 78 | 11 | 0.46M | **$0.14** |

Scores are out of 100. Tokens and cost cover all 200 questions, 100 of each
type, at list price: $0.20 per million input tokens and $1.20 per million
output tokens.

- **Kaleidoscope is the most accurate on single-hop questions.** On multi-hop
  questions the whole history leads with 25, far ahead of Kaleidoscope (10) and
  BM25 (11).
- **Against the whole history, Kaleidoscope uses 6.3 times fewer tokens and
  costs 46% less.** Most of its cost is writing memory: about 9.1M tokens
  ($6.25) to store this history, paid once. Each question after that is cheap,
  so the writing is repaid after about 31 questions in tokens, or about 107 in
  dollars.
- **BM25 is far cheaper than both, because it writes nothing.**
- **Reading alone**, Kaleidoscope sends the answering model 1,195 tokens per
  single-hop question, against 291,746 for the whole history. That comparison
  leaves out the writing.

Where the numbers come from: `python -m kbench.benchmarks.memoryagentbench.check <run_id>`
reports the scores and, in its "Tokens and cost" section, the writing and
reading tokens for every history length. The whole-history numbers come from
`python -m kbench.benchmarks.memoryagentbench.score_partial` on that setup's
log: its score, and its prompt tokens per question, times 100 questions of each
type, plus the 36,042 tokens of its answers. Dollars are tokens times the list
price. Every length, the losses, and the full commands:
[docs/memoryagentbench](docs/memoryagentbench/).

### Continual Learning Bench

[Continual Learning Bench](https://arxiv.org/abs/2606.05661) runs an agent
through six tasks where it should improve with experience: radio mapping, bug
fixing, medical cohort studies, database questions, poker and sales
forecasting. Both setups below use the same model, **GPT-5.6 Luna** at
reasoning effort `high`:

| setup | model | radio | bug fixing | medical | database | poker | sales | average |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Kaleidoscope** | GPT-5.6 Luna | **14.0** | **52.1** | **−1.7** | 41.2 | −6.5 | **66.2** | **27.6** |
| in-context learning (whole history in the prompt) | GPT-5.6 Luna | 9.0 | 31.0 | −8.1 | **61.9** | **−1.6** | 61.4 | 25.6 |

For reference, the best entry on the benchmark's
[public leaderboard](https://continual-learning-bench.com) (last updated 18
July 2026) is in-context learning with Claude Sonnet 4.6, averaging 19.6. That
is a different model, so the comparison is not controlled.

- **Kaleidoscope leads on four of six tasks, most on bug fixing**, by 21
  points.
- **It trails on database questions.** Its model keeps re-checking a
  database's layout instead of trusting what memory recalls, and every extra
  query costs reward.
- **Poker is a tie within noise.** Three all-in hands decide the gap.
- **One run per setup:** the average difference is within noise.

Per-task gains, how Kaleidoscope was used, and where it loses are in
[docs/continual-learning-bench](docs/continual-learning-bench/).

### BEAM

[BEAM](https://arxiv.org/abs/2510.27246) tests ten memory abilities, such as
updating a fact or resolving a contradiction, over conversations of 100K to 10M
tokens. Results at 100K and 1M, per question and per ability, are in
[docs/beam](docs/beam/). Read that page's first section before quoting it: it
compares Kaleidoscope with published mem0 results that used a different
answering model.

## Quickstart

You need macOS or Linux, Python 3.10 or newer, and Node.js 18 or newer.

**1. Install Kaleidoscope** and check that its built-in model is present:

```bash
npm install -g @kleos-research/kaleidoscope
kscope model        # prints JSON that includes "status": "bundled"
```

**2. Install this repository:**

```bash
git clone https://github.com/kleos-research/kaleidoscope-benchmarks
cd kaleidoscope-benchmarks
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

**3. Run the tests.** They use a stand-in for `kscope`, so they are free and
need no API key:

```bash
python -m pytest tests/ -q
```

**4. Run a benchmark.** Both benchmarks call a language model, so a run costs
money.

- **MemoryAgentBench** runs against the `kscope` you just installed. It needs
  an Azure OpenAI deployment of GPT-5.6 Luna. The published run, all four
  history lengths, cost $9.91 at list price, plus $11.71 for the whole-history
  setup. See [its README](kbench/benchmarks/memoryagentbench/).
- **BEAM** works with any OpenAI-compatible API, but runs against one exact
  build of `kscope`. You pin that build by passing the binary and its *public
  contract*, a JSON file generated for each build, together with the SHA-256
  of each. The npm package doesn't include the contract yet, so for now you can
  run BEAM only against a build that came with one. See
  [its README](kbench/benchmarks/beam/).

A BEAM run, once you have the data and a build with its contract, looks like
this:

```bash
cp .env.example .env          # then set OPENAI_API_KEY in .env
shasum -a 256 /path/to/kscope /path/to/kaleidoscope-public-contract.json

python -m kbench.benchmarks.beam.run all --tier 100K \
  --candidate /path/to/kscope \
  --candidate-sha256 <sha256 of kscope> \
  --public-contract /path/to/kaleidoscope-public-contract.json \
  --public-contract-sha256 <sha256 of the contract>
```

The harness refuses to start if either file does not match its SHA-256, so a
run always measures the build you meant.

## Benchmarks

| benchmark | what it measures | status |
| --- | --- | --- |
| [BEAM](kbench/benchmarks/beam/) | ten memory abilities over conversations of 100K to 10M tokens | supported |
| [MemoryAgentBench](kbench/benchmarks/memoryagentbench/) | answering from a long history whose facts are later overwritten, single- and multi-hop, 6K to 262K tokens | supported |
| [Continual Learning Bench](docs/continual-learning-bench/) | getting better over a sequence of related tasks, across six domains | results published; it runs as a plug-in inside the benchmark's own repository |
| LongMemEval | long-horizon question answering | planned |
| LoCoMo | long conversational memory | planned |

Each benchmark lives in two places: the harness and how to run it under
`kbench/benchmarks/<name>/`, and what it measured under `docs/<name>/`.

## Repository layout

```
kbench/
├── config.py               settings, read from the environment and .env
├── llm.py                  the model client and spend tracking
├── kaleidoscope.py         runs kscope, one memory store per conversation
└── benchmarks/
    ├── beam/               the BEAM harness: ingest, answer, judge, report
    └── memoryagentbench/   an adapter for MemoryAgentBench's own harness, and an independent checker
docs/
├── beam/                   BEAM results: every question, and per-setup summaries
└── memoryagentbench/       MemoryAgentBench results, with a PDF
tests/                      unit tests; free, no model calls
```

## Contributing

Before you open a pull request, run the tests and the linter:

```bash
python -m pytest tests/ -q
ruff check kbench/
```

Every number in a README or a report should come with the command that
reproduces it. AI agents working in this repository: read
[CLAUDE.md](CLAUDE.md) and [AGENTS.md](AGENTS.md) first.

## License

Apache 2.0. BEAM and MemoryAgentBench, code and data, belong to their authors
and come under their own licences; this repository does not redistribute them.
