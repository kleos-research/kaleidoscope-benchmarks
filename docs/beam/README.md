# BEAM results — 100K and 1M

Per-question rows and per-arm aggregates for Kaleidoscope on the BEAM benchmark,
at two conversation tiers and two retrieval depths.

## Read this before the numbers

**The reader and judge are not the same model mem0 published with.** These runs
use `openai-gpt-56-luna` at `reasoning_effort=xhigh`; mem0's published BEAM
results use `gpt-5` for both. A reader difference is indistinguishable from a
memory difference in the final score — [the top-level
README](../../README.md#configuration) says so about arms within a suite, and it
is just as true across published runs.

So the comparison below is **not** controlled for the reader. It is reported
because the underlying numbers are public on both sides and someone will make
the comparison anyway; it is qualified here rather than left to be discovered.
A same-reader run has not been done.

Two further gaps, both stated rather than adjusted for:

* **Nine abilities, not ten.** `event_ordering` was not run, so no Kendall tau
  was computed. The mem0 columns below are recomputed over the same nine from
  their published per-ability figures, so both sides exclude it — but neither
  number is a ten-ability BEAM headline.
* **Depths do not line up.** mem0 publishes `top_50` and `top_200`. These runs
  are k=50 and k=100. The k=100 column is compared against their `top_200`,
  which reads four times as many memories.

## BEAM 1M, rubric mean over the nine shared abilities

| ability | mem0 `top_50` | mem0 `top_200` | kscope k=50 | kscope k=100 |
| --- | ---: | ---: | ---: | ---: |
| abstention | **0.539** | 0.525 | 0.511 | 0.407 |
| contradiction_resolution | 0.373 | 0.357 | 0.705 | **0.734** |
| information_extraction | 0.663 | 0.700 | 0.739 | **0.764** |
| instruction_following | 0.763 | 0.852 | 0.855 | **0.895** |
| knowledge_update | 0.593 | 0.650 | **0.836** | 0.743 |
| multi_session_reasoning | 0.625 | 0.652 | 0.714 | **0.755** |
| preference_following | 0.843 | **0.883** | 0.861 | 0.867 |
| summarization | 0.543 | 0.635 | 0.668 | **0.727** |
| temporal_reasoning | 0.589 | 0.618 | 0.707 | **0.708** |
| **mean of ability means** | **0.614** | **0.652** | **0.733** | **0.733** |

mem0's figures are the `avg_score` fields of
[`beam_1m_top50_results.json`](https://github.com/mem0ai/memory-benchmarks/blob/main/results/platform/beam_1m_top50_results.json)
and `beam_1m_results.json`, restricted to the nine abilities and averaged the
same way.

The largest gaps are `contradiction_resolution` (0.734 against 0.357) and
`knowledge_update` — abilities that turn on a later fact displacing an earlier
one rather than on recall. The one column where mem0 leads at both depths is
`abstention`, and kscope's own k=100 is worse than its k=50 there, which is at
least consistent: a reader shown more memories finds something to say when it
should decline.

## Depth buys nothing, and costs tokens

| tier | arm | depth | headline | context tokens | response chars |
| --- | --- | ---: | ---: | ---: | ---: |
| 1M | B | 50 | 0.733 | 6,015 | 2,452 |
| 1M | B | 100 | 0.733 | 11,254 | 2,748 |
| 100K | A control | 50 | 0.721 | 6,504 | 1,851 |
| 100K | A control | 100 | 0.706 | 12,793 | 2,012 |
| 100K | B | 50 | 0.717 | 6,209 | 1,892 |
| 100K | B | 100 | 0.767 | 11,904 | 2,107 |
| 100K | C | 50 | 0.719 | 6,654 | 1,941 |
| 100K | C | 100 | 0.738 | 13,192 | 2,193 |

At 1M, k=50 and k=100 are the same score to three decimals — **+0.0006 paired
on 623 identical questions**, `t = +0.07`, 95% CI `[-0.016, +0.017]` — for
**1.9x the context**. The 100K tier says the same thing for the control arm
(0.721 to 0.706).

Read beside the table above, k=50 at 6,015 context tokens scores above mem0's
`top_200`.

## Twelve times the corpus, the same score

| | 100K | 1M |
| --- | ---: | ---: |
| conversations | 20 | 35 |
| memories seeded | 6,169 | **74,658** |
| memories per vault | ~300 | ~2,100 |
| questions | 360 | 630 |
| arm B headline, k=50 | 0.717 | 0.733 |

The corpus grew 12x and the score did not move.

## Configuration

Identical across every arm here unless the column says otherwise.

| | |
| --- | --- |
| reader | `openai-gpt-56-luna`, `reasoning_effort=xhigh`, temperature 0 |
| judge | `openai-gpt-56-luna`, `reasoning_effort=xhigh`, one call per rubric item |
| protocol | mem0's answer prompt and nugget judge, so the scores are on their scale |
| abilities | nine; `event_ordering` not run |
| arm A | shipped retrieval, no switches |
| arm B | `csls` + `projected_dedup` + `fan_in` + `order_key=sequence` |
| arm C | `prf` + `projected_dedup` + `fan_in` + `order_key=sequence` |

**Arms B and C are bundles of four switches, all of which ship off.** A
difference between B and A is not attributable to CSLS alone. At 100K the four
configurations at two depths span 0.706-0.767, and only one contrast in that
grid survived a paired test.

`context_tokens` is the harness's own accounting at BEAM's 3.5 chars/token, not
a tokenizer's count. It is consistent across arms, which is what a comparison
needs, and is not what an endpoint bills.

## Files

These sit under `docs/` and not `results/`, because `results/` is gitignored —
it is where a local run writes, and published reference numbers must not collide
with a reader's own output.


| file | |
| --- | --- |
| `beam-100k-questions.csv` | 2,160 rows — every question of every 100K arm |
| `beam-1m-questions.csv` | 1,260 rows — every question of every 1M arm |
| `beam-summary.json` | per-arm aggregates, computed from exactly those rows |

One row per question per arm: `tier`, `arm`, `depth`, `conversation_id`,
`question_index`, `ability`, `score`, `context_chars`, `context_tokens`,
`n_retrieved`, `response_chars`, `prompt_tokens`, `completion_tokens`, the
reader and judge with their efforts, and `config_id`.

`config_id` is a hash of the retrieval configuration and the binary's
fingerprint. Two rows carrying the same id were produced by the same
configuration of the same build; two rows differing in depth carry different
ids.

The aggregates are computed from the published rows rather than copied from a
report, so the CSV and the JSON cannot disagree. `headline` is the **mean of the
ability means**, as BEAM reports it; `mean_over_questions` is beside it because
the two differ whenever abilities carry unequal counts.

## Provenance, and what cannot be reproduced from this repo yet

These rows were produced by Kaleidoscope's internal BEAM harness, not by
`kbench`. The two implement the same benchmark and are converging — `kbench` is
the one that shells out to a real `kscope` and is the one to build on — but they
are not the same code, and a `kbench` run at these settings has not been done.
Treat the rows as the record of what was measured, not as something this repo
currently regenerates.

Seeding wrote 74,658 of 75,039 corpus memories at 1M. 82 were refused at write,
58 of those because a fact named a surface no entity declared; **299 are
unaccounted for** and were most likely dropped in conversion, which is not
counted. That is 0.4% of the corpus and a gap in the accounting rather than a
threat to the numbers.

Empty answers: 0 of 598 at k=100, 0 of 528 at k=50. One question of 630 at
k=100 received no context and scored 0.
