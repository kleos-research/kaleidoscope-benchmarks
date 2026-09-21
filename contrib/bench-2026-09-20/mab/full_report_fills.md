@@ headline

**The result in one table** — `substring_exact_match` out of 100, single-hop / multi-hop:

| length | facts in the pool | `none` (reader alone) | `bm25_units` | **`kscope` (shipped)** |
| --- | --- | --- | --- | --- |
| 6k | 455 | 19 / 0 | 97 / 33 | 95 / 15 |
| 32k | 2,310 | 26 / 0 | 91 / 13 | 95 / 8 |
| 64k | 4,580 | 24 / 0 | 85 / 13 | **96** / 10 |
| 262k | 18,332 | 13 / 2 | 78 / 11 | **90** / 10 |

Three things this run establishes, each with its mechanism measured rather than inferred:

1. **On single-hop, kscope holds as the corpus grows 40x and lexical retrieval does not.** kscope goes 95 → 95 → 96 → 90 while BM25 over the *same units, same reader* goes 97 → 91 → 85 → 78. The reason is upstream of the score: the answer-bearing memory is among kscope's ten candidates in **100.0%** of resolved single-hop questions at 6k, 32k and 64k and 98.0% at 262k, against BM25's 96.9% → 93.8% → 86.9% → 83.8%. Every single-hop loss kscope suffers is a memory it *had already retrieved* and then dropped at the shipped similarity floor (6, 2, 1 and 4 questions).
2. **On multi-hop, both retrievers fail for the same structural reason, and it is not a ranking problem.** A chain's second hop is keyed on an entity that appears only *inside the first hop's text* and never in the question, so one retrieval per question cannot reach it. Hop 1 is served for 77–93% of chains; hop 2 is **never a candidate** for 56–89 of ~98 chains, for kscope and BM25 alike. This is the same wall every published memory system hits (all ≤ 7.0 multi-hop).
3. **When kscope serves the whole chain, the reader is right almost every time** — 13/13, 6/6, 4/4 and 1/2 across the four lengths. The store's content and the reader are not the bottleneck; retrieval coverage is.

kscope reaches those single-hop numbers on **roughly half the reader input tokens** BM25 needs (854–1,195 vs 1,879–2,066 per question), because it serves 2.8–5.1 memories where BM25 serves a fixed 10.

@@ results

## What the numbers mean

**Single-hop is the headline and it is a scaling result.** The interesting column is not any single cell but the slope. BM25's quality decays monotonically as the pool grows (97 → 91 → 85 → 78) because a lexical match over 18,332 near-identical sentences increasingly ranks the wrong one; kscope's hybrid retrieval does not (95 → 95 → 96 → 90). At 262k the gap is 12 points on the identical corpus with the identical reader.

The retrieval-level table shows this is a genuine recall difference, not a reader effect. Counting only hops the checker could map to a stored unit:

| length | kscope candidate recall@10 | BM25 candidate recall@10 | kscope questions lost to the similarity floor |
| --- | --- | --- | --- |
| 6k | **100.0%** (98/98) | 96.9% | 6 |
| 32k | **100.0%** (96/96) | 93.8% | 2 |
| 64k | **100.0%** (99/99) | 86.9% | 1 |
| 262k | **98.0%** (97/99) | 83.8% | 4 |

Read the last column as the cost of a shipped default, not a defect: the 0.45 similarity floor removed a memory kscope had already ranked in its top ten, on 13 single-hop questions across the whole run. Those are the only single-hop questions where kscope had the answer in hand and did not serve it.

**Multi-hop is a wall both systems hit, and the run locates it precisely.** Failure by hop position, counting every chain the checker resolved:

| cell | hop 1 | hop 2 | hop 3 | hop 4 |
| --- | --- | --- | --- | --- |
| 6k kscope | 77 served / 17 cut / 2 never | 18 / 22 / 56 | 4 / 11 / 22 | 0 / 2 / 12 |
| 6k bm25 | 93 / 0 / 3 | 43 / 0 / 53 | 20 / 0 / 17 | 9 / 0 / 5 |
| 32k kscope | 80 / 13 / 5 | 8 / 5 / 85 | 1 / 1 / 39 | 0 / 1 / 15 |
| 32k bm25 | 83 / 0 / 15 | 9 / 0 / 89 | 1 / 0 / 40 | 2 / 0 / 14 |

Hop 1 — the fact whose subject the question names — is found. Hop 2 is *never a candidate*, for both systems, because nothing in the query mentions its subject. No ranking change fixes that; it needs a second retrieval keyed on what the first one returned, or a traversal inside the store. kscope has an entity-and-claim graph that could in principle make that hop, and the served response here is a flat ranked list of memories with no traversal in it — consistent with the standing finding that the graph walk has no consumer on the served read path.

Where kscope loses to BM25 on multi-hop (15 vs 33 at 6k) the cause is volume, not quality: ten whole units at ~2,000 tokens happen to contain more chain facts than three at ~700, even though they were not retrieved *for* those facts. That is an argument for serving more on multi-hop, not for lexical retrieval being better at it — and it costs 2.8x the reader tokens to get there.

**The blind floor is doing its job.** `none` scores 13–26 on single-hop (the reader's own world knowledge, which FactConsolidation deliberately contradicts) and 0–2 on multi-hop. Every memory number above is against that floor, not against zero.

@@ wrong

## What went wrong, plainly

**Two reader calls were refused by Azure's content filter, and both cost kscope a question.** At 32k single-hop q10 and 64k multi-hop q40 the gateway returned a 400 (`The response was filtered due to the prompt triggering Azure OpenAI's content management policy`) on all six attempts — deterministic, not transient. The adapter recorded every attempt in the spend ledger and in `logs/mab_llm_failures.jsonl`, returned an empty answer, and flagged the question `reader_error`, so the harness scored it 0 and the checker reports it apart from an ordinary wrong answer. **Assertion D is RED for those two cells and I have left it red.** In both cases kscope had served exactly one memory and that memory's text was not in BM25's top ten, so the other two arms answered the same question normally (both got it right). kscope's 32k single-hop and 64k multi-hop scores are therefore each up to one point lower than the retrieval alone would give.

**Five answers were empty because the reader spent its whole budget on reasoning.** Three at 32k multi-hop and one at 262k multi-hop in `bm25_units`, one at 262k multi-hop in `kscope`: `finish_reason: length` after 4,000 completion tokens with no visible text. Scored 0. The budget is identical for every arm, but it binds more often on the arm with the larger context, which is worth knowing when reading the multi-hop numbers.

**Three units were refused by `remember` at 262k (0.19% of 1,581), all recovered.** Two for `missing field 'means'` inside a `propose` entry and one for declaring the surface `blizzard entertainment` as two entities. Each was resent with the minimal fallback delta and stored, so no unit was lost and no text left the corpus — but those three units carry no triples, only their verbatim text. The refusal rate is far under the 2% stop condition. Zero refusals at 6k, 32k and 64k.

**Twenty-four `propose` clamps (0 / 0 / 5 / 19 by length), none reader-visible.** Every clamp is the 8-entry `propose` cap; no unit came near the 32-fact or 32-entity caps (the largest was 12 facts / 24 entities). Per the standing decision these do not block the run, and the fold is byte-identical downstream.

**The unit rule separates a serial number from its fact in 2.5–4.6% of units.** A unit that ends with a bare marker (`… Apple Inc.. 35.`) leaves fact 35's text to start the next unit, so that memory shows the fact without the serial the task tells the reader to compare. 73 of 1,581 units at 262k, 1 of 39 at 6k. The cause is the sentence tokenizer not splitting after an abbreviation's period, so the marker trails the previous sentence instead of being glued forward. The unit rule is frozen for this run and I did not touch it; it is reported as an observed property. It also costs the checker its fact→unit mapping for 0.22–0.39% of facts, which is why one 262k single-hop hop is unaccounted in the fate table.

**I found and fixed two defects in my own checker, both of which inflated or misdirected a number before anyone read it.** Writer tags (`mab:<run>:c<chunk>:u<unit>`) and reader tags (`mab:<run>:<arm>:q<id>`) repeat across lengths, and reader tags also repeat across the sh/mh rows that run concurrently. The first version therefore (a) reported 371 reader calls for 200 questions, double-counting the overlap of two time windows, and (b) compared each length's "last writer call" against a *different* length's ingestion that was still running, turning the write-before-question ordering check red for a reason that did not exist. Calls are now matched by tag **and** a time window, and the report carries an attribution check that goes red if any ledger row inside a checked window is unattributed — it now reads 4,631 of 4,631 attributed, 0 left over.

**The 262k multi-hop search latency (mean 1.22 s, p90 4.01 s) is contention, not a store property, and I am not quoting it as one.** The single-hop run against a same-sized clone of the same vault in the same window measured p50 0.199 s / p90 0.260 s / max 0.518 s. The 27 slow multi-hop searches are the *last* queries of the run (ids 82–95), and the maximum degree of any corpus entity named in them is 1–2, the same as the fast ones — so it tracks when the query ran, not what it asked. Five other harness processes were still running against the same disk. The defensible read-latency figure at 1,581 memories is the single-hop one: **p50 0.199 s, p90 0.260 s**.

**Write cost grows with vault size, visibly.** `remember` for a batch of up to 20 units took ~300 ms at 6k and, within the 262k ingestion alone, rose 586 → 775 → 832 → 1,067 ms across the four quarters (first batch 316 ms, last 5,153 ms). Per unit that is 20 ms at 6k against 70 ms at 262k — a 3.5x increase for a 40x corpus. Ingestion wall clock was 2.7 / 12.1 / 30.1 / 95.7 minutes for 39 / 200 / 397 / 1,581 units, dominated throughout by the writer LLM calls (mean 22–25 s each, 8 concurrent), not by kscope. The 262k vault is 215 MB on disk.
