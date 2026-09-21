# Kaleidoscope on MemoryAgentBench FactConsolidation — full run

Run id `full-20260920T1905`, 2026-09-20. Eight dataset rows (`factconsolidation_{sh,mh}_{6k,32k,64k,262k}`), exactly 100 questions each, three setups through one reader: **800 questions × 3 = 2,400 reader calls**, plus one writer call per unit for the kscope arm. Everything below is measured; the checker (`mab/full_check.py`) re-derives every count from the artefacts the run left behind and its generated sections are pasted verbatim under the headings marked *(generated)*.

<<FILL:headline>>

## What was run

| setup | memory | writes | reads | reader |
| --- | --- | --- | --- | --- |
| `kscope` | Kaleidoscope, installed binary `/opt/homebrew/bin/kscope` (sha256 `361f5975d522083d…`, `kscope model` → bundled), **`shipped` profile, no settings changed** | one gpt-5.6-luna writer call per unit (12 sentences / 350 tokens, whichever first) → structured delta; unit stored verbatim as `content_md` | one `search`, `top_k=10`, `maximum_context_bytes=32768` | gpt-5.6-luna, reasoning effort high, visible answer capped at 10 tokens |
| `none` | nothing | none | an empty vault (served 0, asserted) | the same |
| `bm25_units` | the same units, `rank_bm25.BM25Okapi` over whitespace tokens — the harness's own `Simple_rag_bm25` retriever (langchain `BM25Retriever` default) — over units instead of 4096-token chunks | none | top-10 units | the same |

The reader prompt scaffold is the harness's `rag_agent` template for `factconsolidation` (`Memory i:` blocks + the question), identical across setups; the reader receives whatever the retrieval served and nothing else.

**Ingest once, clone per row.** The single-hop and multi-hop rows of a length carry a byte-identical context (sha256-equal, verified again by the ingestion script), so each length was ingested exactly once into a sealed master vault (`mab/ingest_master.py`; seal = sha256 over every file of the vault). Every question run clones the master copy-on-write (`cp -Rc`), verifies the clone carries the seal, and the adapter then only checks that every chunk and unit the harness hands it is byte-identical to the manifest. No master was ever searched (a search would append an exposure row and break the seal; the checker re-hashes every master at check time). The `bm25_units` arm verifies its units against the same manifest, so all three setups see the same text.

**Frozen:** `common/*` (LLM door, kscope door, writer), the unit rule, the writer prompt (sha `7450226fa497afc7` on every writer call of every length). The one env knob set for the writer processes: `BENCH_LLM_TIMEOUT=180`, raised from the default 90 because writer calls in the pre-run gate measured max 62 s and a call the client cuts is billed but never reaches the ledger. The run itself went past that estimate: writer latency stayed at p50 20–26 s and p90 32–37 s at every length, but **11 of 2,221 calls exceeded the 180 s deadline** (3 at 64k, 8 at 262k, longest 226 s). Those were retried inside the SDK and returned successfully, so no unit was lost; had the deadline stayed at 90 s, roughly 30–40 calls would have been cut and re-asked instead.

## Published context — NOT like-for-like

arXiv 2507.05257v4 Table 3 (0–100, GPT-4o-mini reader, length unstated, probably 262k): single-hop / multi-hop — BM25 48.0 / 3.0, HippoRAG-v2 54.0 / 5.0, Mem0 18.0 / 2.0, Zep 7.0 / 3.0, MemGPT 28.0 / 3.0, Cognee 28.0 / 3.0; GPT-5-mini long-context 78.0 / 28.0. Every memory system scores ≤ 7.0 on multi-hop. Our reader is gpt-5.6-luna with reasoning; our `none` row is that reader's own floor (what it "knows" unaided), and `bm25_units` is the only same-reader anchor. Placing kscope against the published rows compares readers as much as memories.

## Two ceilings the dataset imposes

1. **The gold contradicts the benchmark's own rule for some questions.** The task prompt says the newest serial wins. Parsing every numbered fact with the corpus's 37 sentence frames, the gold answer of some questions is reachable only through a fact that a *higher* serial in the same (subject, relation) family overrides — e.g. *What is the capital of Papal States?* is scored against Rome (serial 1291) while the context also states `2016. The capital of Papal States is Watertown`. Counts (of 100): single-hop 6k 0 / 32k 2 / 64k 2 / 262k 3; multi-hop 6k 4 / 32k 13 / 64k 10 / 262k 34. A reader that applies the rule perfectly cannot score those, so the achievable maximum is below 100 — sharply so for multi-hop at 262k.
2. **Multi-hop needs 2–4 memories in one served set** (hop counts per cell are in the generated table). A retrieval that serves the first hop's memory and cuts the rest cannot answer, however good the reader.

<<FILL:results>>

## Scores *(generated)*

<<SECTION: 1. Scores>>

## Retrieval-level analysis *(generated)*

<<SECTION: 2. Retrieval-level analysis (kscope and bm25_units)>>

## Ingestion *(generated)*

<<SECTION: 3. Ingestion (one master vault per length)>>

## Assertions *(generated)*

<<SECTION: 4. Assertions (each can go red)>>

## Tokens and cost *(generated)*

<<SECTION: 5. Tokens and cost>>

<<FILL:wrong>>

## Files

- run scripts: `mab/ingest_master.py`, `mab/run_mab.py`, `mab/drive_full.py`, `mab/env_full.sh`; adapter `MemoryAgentBench/methods/kscope_agent.py` (diff in `patches/memoryagentbench.diff`); checker `mab/full_check.py`
- masters and clones: `vaults/mab/full-20260920T1905/{master,kscope,none}/…`, manifests beside each master (`*-manifest.json`)
- logs: `logs/mab_adapter-full-20260920T1905-*.jsonl`, `logs/mab_harness-full-20260920T1905-*.log`, `logs/mab_ingest-full-20260920T1905-*.log`, `logs/mab_driver-full-20260920T1905-*.log`, `logs/kscope_calls-mab.jsonl`, `spend/ledger.jsonl`, `logs/mab_llm_failures.jsonl`
- results: `results/mab/full-20260920T1905/<arm>/<row>/Conflict_Resolution/*_results.json`
- generated report: `reports/mab_full_full-20260920T1905_generated.md`; this document is that file stitched into `mab/full_report_narrative.md` by `mab/build_full_report.py`
- the adapter and launcher the SMOKE runs used, kept because they exist nowhere else: `mab/adapter-smoke-version/` (on no run path)
