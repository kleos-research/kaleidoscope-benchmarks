# The memory-writing contract

What an agent — human-written or model-driven — supplies when writing into
Kaleidoscope. This file is the source `kbench/benchmarks/beam/prompts/extraction.md`
is written against, so changing it changes what gets written.

Tools that read repository instructions pick this file up automatically.
`CLAUDE.md` points here rather than duplicating it.

---

## The one rule everything follows from

**Kaleidoscope never infers what a memory means. The caller supplies the
semantics.**

`remember` requires a `semantic_delta` carrying a title and at least one fact on
every create. There is no path that accepts prose and derives structure from it.
The operation that used to do that was retired precisely because a memory
written with no title, no facts and a placeholder type is retrievable but
participates in nothing — no merge veto, no contradiction check, no graph edge.

So the extractor's job is not "summarise this exchange". It is **"state what this
exchange establishes, in a form a later reader can match against a question"**.

---

## The shape

```json
{
  "memory_type": "decision",
  "title": "Transactions table gains category and notes columns",
  "content_md": "The user decided to add two columns to the transactions table: category and notes.",
  "facts": [
    {"subject": "transactions table", "predicate": "gains_column", "object": "category"},
    {"subject": "transactions table", "predicate": "gains_column", "object": "notes"}
  ],
  "entities": ["transactions table"],
  "supersedes": null,
  "contradicts": []
}
```

### `memory_type` — a closed vocabulary

`architecture`, `constraint`, `correction`, `decision`, `note`, `outcome`,
`preference`, `procedure`.

An unaccepted type is **refused at the service boundary**, and the refusal costs
the extraction call that produced it. A workspace can extend the vocabulary
through the ontology; nothing in this benchmark does. Keep this list and the
service's `ACCEPTED_MEMORY_TYPES` in sync — the test suite checks that they match.

### `facts` — SPO triples, and no confidence

At least one, capped at 32. `predicate` must be a lowercase bounded identifier
(`ends_on`, `gains_column`) — spaces are refused.

**The model is not asked for a confidence.** The service's schema carries one and
it feeds an admission term, so the harness sends a constant. But a language
model has no calibrated distinction between 0.49 and 0.5; asking for one returns
a number that looks precise, is noise, and then flows into an admission
decision. If you are unsure what the exchange states, do not write the fact.
The field exists for callers with a real calibrated source — a classifier, a
vote, a measurement — and an extractor is not one.

Note that facts are **not independently retrievable**: the lexical document is
built from title and content only, so a term appearing solely in a fact is not
matchable. Put anything a question might key on into the title or the content
too.

### `title` — the handle a revision will use

Name the specific thing. "Sprint one end date", not "Update". A vague title
cannot be targeted, so a later correction writes a duplicate instead of
replacing its predecessor.

### There is no `worth_remembering` flag

There used to be, and it was redundant with `facts: []`. An extraction with no
facts writes nothing, because `remember` requires at least one — so the flag was
a *second* judgement about the same question, and a second judgement can only
lose information.

It is removed for being redundant, and that is the whole argument. How much a
gate costs depends entirely on how the prompt is written, so no fixed figure
belongs here — two extractions over the same corpus with different prompts
dropped 745 exchanges and 83 respectively. Measure it on your own prompt with
your own benchmark's metrics if you want a number.

---

## `supersedes` and `contradicts` — how an agent actually resolves them

These are the two mechanisms for change over time, and the reason a memory
system beats a transcript at all.

- **`supersedes`** — this memory *replaces* an earlier one. A changed date, a
  reversed decision, a corrected number. The predecessor is retired from
  serving; its bytes remain.
- **`contradicts`** — this memory *disputes* an earlier one without cleanly
  replacing it. Both stay live and the conflict is recorded as an edge.

**The agent resolves them from a numbered list.** Before each extraction it is
shown the memories already written for this conversation:

```
PRIOR MEMORIES from this conversation, numbered:
1. Sprint one end date — Sprint one ends on 2024-03-29.
2. Transactions table gains category and notes columns — ...
3. Deployment target — The service deploys to staging first.
```

and it returns `"supersedes": 1`, not a title.

**Numbers, not titles, and that is deliberate.** Matching by exact title string
is brittle in the obvious way: a near-miss is silently a no-op that writes a
duplicate instead of a retirement, and nothing reports it. An integer either
resolves or does not, and one that does not is counted as
`unresolved_references` so a run can see the extractor inventing references.
mem0's update prompt remaps existing memories to integers for the same reason.

**The list is bounded, and that bound has a cliff.** `PRIOR_WINDOW` is 40 by
recency. Measured on BEAM 100K, a window of 40 reaches about 80% of candidate
revisions; 80 reaches about 93%. A revision pointing further back than the
window simply cannot see its target. Selecting by *similarity* instead of
recency removes the cliff — that is a known improvement, not yet the default.

---

## What the extractor is shown, and what it is not

Three things: **the exchange**, **the date it was said on**, and **the numbered
prior memories**.

It is **never** shown the benchmark's questions. That would leak the evaluation
into the write path and make every score meaningless.

---

## Ordering

Ingestion walks a conversation front to back and **must**. A `supersedes` can
only name a memory that already exists, so writing turn 40 before turn 12 loses
the revision silently.

Conversations are fully independent and run concurrently.

---

## Batching — `remember.items`

`remember` accepts an `items` array of up to **20** creates in one call, given
instead of the top-level `content_md`/`semantic_delta`.

**A batch is a cheaper way to deliver declared semantics, never a cheaper way to
avoid declaring them.** Every item carries its own `content_md`, its own title
and its own facts. There is no batch-level delta, nothing is inherited between
items, and an item with no facts is refused — taking the whole batch with it,
before anything reaches disk, rather than leaving half of it there.

What it actually saves is the derived work. Kaleidoscope re-derives the graph
and activates the lexical index **once per call**, so a per-memory write pays
that per memory. Over 500 creates through the CLI:

| | single | batched (20) |
| --- | --- | --- |
| calls | 500 | 25 |
| seconds | 60.03 | 16.37 |
| graph rebuilds | 500 | 25 |
| index activations | 500 | 25 |
| index mutations | 500 | 500 |

The index still receives all 500 documents. Only the number of activations
changes.

### Order is load-bearing inside a batch

Items apply **in sequence**, each against a projection the previous one updated
— so a later item can consolidate against, or supersede, an earlier one in the
same batch. This is why a batch is not the unordered fan-out that a `mem0`-style
`add` pools, and why it cannot be reordered for throughput.

### The one thing batching gives up

A buffered item has **no `memory_id` yet**. So an extraction that supersedes a
memory still sitting in the buffer has nothing to point at, and the harness
flushes the buffer before writing it. It does not try to predict the id: keys
are derived by the service, and a harness that guesses at identity writes edges
to whichever memory happened to land at that index.

`ingest.json` reports `remember_calls` beside `written`, so the amortisation a
run actually achieved is a number in the output rather than an assumption. A
conversation dense in revisions flushes often and shows a ratio near 1.
