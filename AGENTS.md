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

Measured on BEAM 100K, the gate itself was close to harmless: it dropped 83 of
2,866 exchanges, only 12 of which carried evidence — a 1.6% false-negative rate,
and it never once dropped a question's sole evidence. So it is removed for being
redundant rather than for being destructive.

**The number worth knowing is a different one.** Only 69% of evidence messages
are reachable in the written memories at all, and the gate accounts for 1.6
points of that gap. The rest is abstraction: the instruction to state the
durable fact rather than copy verbatim paraphrases away the distinctive terms a
later question keys on. Rule 3 exists for that reason — put anything a question
might match into the title or the content, not only into a fact.

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
