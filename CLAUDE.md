# Working in this repository

**The memory-writing contract is [AGENTS.md](AGENTS.md). Read it before touching
anything under `kbench/benchmarks/*/extract.py` or `prompts/extraction.md`.** It
is not duplicated here — one copy, so the two cannot drift.

## What this repository is

An evaluation harness for [Kaleidoscope](https://memory.kleosresearch.xyz/).
It measures the memory system; it is not the memory system. `kscope` is assumed
installed and every call shells out to it, so what gets measured is the shipped
binary rather than a reimplementation.

## The shape of a benchmark

Four phases, decoupled, each writing to disk:

    ingest -> answer -> judge -> report

Nothing downstream reruns anything upstream. A judge experiment costs judge
calls only; a crash in phase 3 never loses phase 2's answers. Keep it that way —
the temptation to fuse phases for convenience is how a benchmark ends up
re-spending its expensive half to change its cheap one.

## Rules that are not stylistic

**Never infer structure from model or corpus text with pattern matching.** No
regex, no `startswith` / `endswith` / `contains` used to decide what something
*means*. Reading a JSON field is fine. This is the same rule Kaleidoscope
enforces on itself, and the harness has no business being looser than the system
it measures.

**Scope evidence per conversation.** BEAM message ids are per-conversation
indices — 392 distinct ids cover 5,732 messages in the 100K tier. A global
`id -> message` map scores evidence against the wrong conversation and silently
*inflates* recall. `dataset.assert_conversations_are_isolated()` checks this at
load; do not route around it.

**Keep the reader identical across arms you intend to compare.** A reader
difference is indistinguishable from a memory difference in the final score.
Stamp the model that produced each row onto the row, not the constant currently
in the config — otherwise changing a default silently relabels every historical
result.

**Report failures as failures.** A partial run presented as complete is worse
than no run. If a phase is skipped, say which and why.

## Parallelism

Conversations are independent stores and run concurrently. Questions within a
conversation issue independent ranked searches and run concurrently; the
native vault owns exposure-write locking. **Chunks within a conversation do
not** — later facts and contradictions can depend on earlier writes.

If you add a nested pool, size the inner one for the *product*. A conversation
worker submits its questions and then blocks on the results; with an inner pool
sized for one conversation, the waiters occupy every slot and starve the work
they are waiting on. That presents as a hang, not an error.

## Before you commit

```bash
python -m pytest tests/ -q
ruff check kbench/
```

Numbers in a README or a report should be reproducible by a command in that same
document. If you cannot give the command, do not give the number.
