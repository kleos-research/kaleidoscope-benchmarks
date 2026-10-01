# The memory-writing contract

This is the source contract for
`kbench/benchmarks/beam/prompts/extraction.md`. Changing it changes what the
benchmark writes.

## The rule

Kaleidoscope never infers what a memory means. The caller supplies the
semantics. Every `remember` create carries Markdown beginning with `# ` and a
`semantic_delta` with a title and at least one fact. An extraction with no
facts writes nothing.

The extractor states what one exchange establishes in a form a later question
can match. It is never shown the benchmark questions.

## Runtime-owned vocabulary

Do not transcribe accepted `memory_type` values into code, prompts, tests or
documentation. Before extracting for a conversation, the controller reads:

```text
kscope call --profile <profile> ontology
```

with `{"mode":"read"}` and takes the values from
`declarable.memory_types`. `ontology` is an operator call, not a model-facing
tool. The resulting list is inserted into the extraction prompt and its cache
fingerprint. An extractor output outside that list is refused rather than
silently mapped to a hand-picked fallback.

## One exchange may establish more than one thing

The contract is **per memory**, not per exchange. An exchange that settles two
unrelated things should produce two memories, each complete in itself — its own
title, its own content, its own facts. Several memories go in one `remember`
call, up to the limit in "Ordering and batching".

**The BEAM extractor does not do this yet.** `prompts/extraction.md` still asks
for one object per exchange, and `extract.py` and `ingest.py` write one memory
per exchange. This section is the contract they are moving to. Until they move,
it describes the target, not what the benchmark writes.

This is not a style preference, and the reason is a property of the retriever.
A fact can be found by search on its own, by the names and the relation it
states, but a match only ever brings back the memory the fact belongs to, and
that memory takes one place in the results. Its title and its text are about
whatever the memory is about. Bundle two unrelated claims, and a question about
the second gets back a memory titled and written about the first, with both
claims sharing one place in the results.

> **User:** Sprint one now ends March 29th, not the 22nd. And put Priya on auth.
>
> Two memories — *"Sprint one end date"* and *"Auth work ownership"* — not one
> memory about both.

A small probe on BEAM 100K put the two contracts side by side on the same
exchanges. Asked for several memories, the extractor wrote about as many facts
per exchange as when asked for one, spread over nearly twice as many memories;
on a larger sample, the several-memory prompt also drew out more facts. A probe
is not a score: whether more memories per exchange raise the benchmark score has
not been measured.

What does **not** change is the window. One extraction still reads **one
exchange**: two messages, the default of `--chunk-size`, the same granularity as
mem0's own BEAM runner and LIGHT's `pair_chunk`. Widening it is a different
change and a bad one: an earlier research harness packed about ten messages per
call, and the extractor merged, dropped and generalised, which was then misread
as a property of the memory system rather than of the packing. Output arity is
the knob. Input size is not.

More memories cost no extra extraction calls: there is still one per exchange.
They do add `remember` items, and each call carries a bounded batch, so
`remember` calls grow with the number of memories.

## Current write shape

```json
{
  "mode": "create",
  "items": [
    {
      "content_md": "# Transactions table columns\n\nThe transactions table gains category and notes columns.\n",
      "semantic_delta": {
        "memory_type": "<one value returned by this profile's ontology>",
        "title": "Transactions table columns",
        "entities": [
          {"n": "transactions table", "kind": "artifact", "is": "the table that stores transactions"},
          {"n": "category", "kind": "concept", "is": "a transaction classification column"},
          {"n": "notes", "kind": "concept", "is": "a free-text transaction annotation column"}
        ],
        "facts": [
          {"subject": "transactions table", "predicate": "gains_column", "object": "category", "mode": "decision", "basis": "stated", "confidence": 1.0},
          {"subject": "transactions table", "predicate": "gains_column", "object": "notes", "mode": "decision", "basis": "stated", "confidence": 1.0}
        ],
        "evidence": [
          {"kind": "conversation_turn", "reference": "2026-03-15T00:00:00Z"}
        ],
        "occurred_at": {"t": "2026-03-15T00:00:00Z", "grain": "instant"}
      }
    }
  ]
}
```

Every fact endpoint must be declared in `entities` with exact `n`, a `kind`,
and a required identifying `is` gloss. A date is never an entity. Predicates
are lowercase bounded identifiers. Anything a question may key on belongs in
the title or Markdown too: a fact is matched only by the names and relation it
states, while the title and Markdown are what the memory itself is matched by.

The model is not asked for numeric confidence. This harness supplies a constant
for facts it accepts. If the exchange does not support a fact, do not write it.

## Corrections and disputes

`supersedes` is not an input field. Neither are `relations` or caller-authored
retry keys. Current graph semantics derive replacement from facts. A correction
may use `corrections` as `{handle, says}` objects, while an unresolved dispute
uses `contradicts` with active memory IDs.

The extraction prompt sees a bounded, numbered list of prior memories and may
return numbers under `contradicts`. The controller resolves only numbers it
actually supplied; invented references increment `unresolved_references` and
are never guessed by title.

## Ordering and batching

Ingestion walks a conversation front to back. Conversations are independent
and run concurrently; chunks inside one conversation do not, because later
facts and disputes can depend on earlier writes.

`remember.items` accepts the batch limit published by the digest-bound public
contract. Each item carries its own Markdown, title, entities and facts. Items
apply in order. Per-item validation failures are reported at their indexes;
fields outside the operation schema refuse the request before any item is
read. Repair and resend only a refused item when the response proves the
request itself was accepted.

A buffered item has no memory ID. When a later extraction disputes it, flush
the buffer first; never predict an ID. The adapter owns retry identity, so the
benchmark sends no `idempotency_key`.

## Read contract

The only agent read tool is `search`:

- ranked: `query`, `top_k`, and `maximum_context_bytes`; the result carries
  ranked memories under `selected_hits` and bounded model text in
  `context_text`; and
- addressed: `memory_id` only; the current memory is returned at top level.

A BEAM question performs exactly one ranked acquisition search and gives the
reader the engine-rendered `context_text`. It does not re-render hits or follow
them with addressed reads. “Evidence recall” is the benchmark metric noun; it
does not name a tool.

## Candidate and privacy boundary

Every engine phase requires an executable path, its expected SHA-256, a
generated public-contract path and its expected SHA-256. The contract must bind
that executable and publish exactly `remember` and `search`. Absent or
mismatched inputs refuse before profile or vault work.

Each conversation owns one deterministic native profile and one vault. Public
configuration and result metadata contain no root, workspace, principal or
journal coordinates. The benchmark records candidate and contract digests but
does not claim signature verification or release evidence until a signed
DX-06A candidate is supplied and independently verified.
