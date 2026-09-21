# MemoryAgentBench x kscope: adapter and 10-question smoke test

Run id `smoke-20260920T160348`, 2026-09-20. Sub-dataset `factconsolidation_sh_6k` (Conflict_Resolution split), first 10 of 100 questions, two arms, through the harness's own `main.py`.

**Read this first.** Ten questions prove plumbing and nothing statistical. Nothing was tuned to move a score. Every number below is either copied from the harness's results file or recomputed by an independent checker (`mab/smoke_check.py`) from the logs the run left behind; the tables in this file are that checker's output, pasted by script, not retyped.

## 1. Goal, and what came out

The goal was to find out whether kscope can sit behind MemoryAgentBench as a retrieve-then-read memory without any hack, and whether information travels end to end: harness chunk -> writer -> vault -> search -> reader -> harness score.

It does. Both arms ran to completion with exit code 0, no exception, no lost unit, and the spend cap was never approached.

| arm | what it is | substring_exact_match (the split's metric) | exact_match |
|---|---|---|---|
| `none` | nothing written, empty vault, same reader and prompt | 2 / 10 | 1 / 10 |
| `kscope` | one writer LLM call per ~350-token unit, stored in kscope, top-10 search | 7 / 10 | 6 / 10 |

The score is not the finding. The finding is at the retrieval level (section 4):

1. In all 10 searches the memory holding the CURRENT fact was among kscope's 10 candidates. It was served in 6 and **cut in 4, every time by one named mechanism: `below_similarity_floor`** (the shipped 0.45 floor). With the floor switched off on a clone of the same vault, that memory ranks 3, 4, 4 and 7 in those four searches.
2. Which side of the floor a memory lands on tracked one thing in this sample: whether the writer's structured delta carried the asked-about fact as a triple. Current fact in the delta: 6 of 6 served. Current fact only in the verbatim `content_md`: 0 of 4 served.
3. The writer left a quarter of the facts out of its deltas, and not by accident: `common/writer.py` caps a memory at 32 entities, this corpus has two fresh entities per fact, so a 24-fact unit cannot declare more than ~16 facts' worth. 344 of 454 facts were extracted; after two kscope refusals only 309 of 454 (68%) sit in the vault as triples.
4. When the served set held only the STALE fact (2 questions) the reader answered the stale value, both times. When it held both (2 questions) the reader picked the higher serial number, both times. kscope served stale and current side by side with no marking, and resolution was the reader's: re-checked on a clone under the shipped profile, a served hit carries `content_md`, `facts`, `memory_id`, `version_id`, `memory_type`, `sequence`, `created_on` and nothing else, even for q0 and q6 where the two facts share subject AND predicate (`goaltender | associated_with | ice hockey` and `goaltender | associated_with | pesäpallo`). That agrees with what was already established: the supersession fold does not change what is served.

## 2. What ran

- **Harness**: `MemoryAgentBench/` at commit `fe1735d`, plus the patch in `patches/memoryagentbench.diff` (section 7). Command, per arm (from `runs/mab/<run_id>/<arm>/command.txt`):
  `python main.py --agent_config runs/mab/<run_id>/<arm>/agent.yaml --dataset_config configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml --max_test_queries_ablation 10`
  launched by `mab/run_mab.py`, which only rewrites `output_dir` in a per-run copy of the agent YAML and sets `MAB_RUN_ID`.
- **Data**: `ai-hyz/MemoryAgentBench`, split `Conflict_Resolution`, revision `main` = `7ea066982b140a19337e17e60d45d4076e042faf`, loaded by the harness's own `load_dataset` call into `HF_HOME` under this directory. One context of 5,988 tokens (455 numbered facts, one per line), which the harness's chunker (`chunk_size: 4096`, nltk punkt + tiktoken) turned into 2 chunks of 4,319 and 2,119 tokens. The checker re-chunks the parquet copy in `data/` and gets the same two sha256 as the adapter logged.
- **kscope**: `/opt/homebrew/bin/kscope` 0.0.6, sha256 `361f5975d522083d...` (pinned by `common/kscope_io.py`), `kscope model` = `"status":"bundled"` (potion-base-8M). Profile `shipped`. `search` with `top_k=10`, `maximum_context_bytes=32768`.
- **LLM**: deployment `gpt-5.6-luna` (served as `gpt-5.6-luna-2026-07-09` on all 42 calls), reasoning effort `high`, every call through `common.llm.chat`. **No temperature is sent on any call** (shipped agent configs say 0.7; this is a reasoning deployment, and not sending one keeps runs as repeatable as the endpoint allows). The harness's own OpenAI/Azure client is never constructed.
- **Environment**: `BENCH_SCOPE=mab`, `BENCH_SCOPE_CAP_USD=4`, `HF_HOME`, `NLTK_DATA`, `TIKTOKEN_CACHE_DIR` all under this directory (`mab/env.sh`). Venv `venv-mab/` on Python 3.10.18 with openai 3.16.2, tiktoken 0.14.0, datasets 5.0.1, nltk 3.10.3, rouge_score 0.1.2, editdistance 0.8.1, numpy, pyyaml, python-dotenv, tqdm. No torch, transformers or langchain. Disk: venv 333 MB, caches 264 MB (of which the HF cache is 197 MB: `load_dataset` downloads all four splits, 75 MB of parquet, to read one), vaults 10 MB.
- **Order**: the `none` arm ran first (24 s), then the `kscope` arm (448 s, of which 428 s is ingestion: 22 writer calls, 4 at a time).

### The two arms

Both arms are one class (`MemoryAgentBench/methods/kscope_agent.py`) with a `kscope_writer` switch in the agent YAML. Both create one vault per adapter instance in the constructor, at `vaults/mab/<run_id>/<writer>/<sub_dataset>-<uuid4>`; the harness builds a fresh `AgentWrapper` per context, so one instance is one context, and nothing is keyed on a counter. Both run the identical answer path:

1. the harness's own `agent._extract_retrieval_query(message)` (the regex every retrieve-then-read baseline uses),
2. one `Vault.search`,
3. the BM25 baseline's assembly, byte for byte: each served hit's `content_md` plus a newline, labelled `Memory i:`, joined, then `"\n" + message`, system message from `get_template(sub_dataset, "system", agent_name)`, wrapped by the harness's `format_chat`,
4. one shared reader function, `read_answer`, which calls `common.llm.chat`.

- `none`: `ingest` stores nothing. The search runs against the arm's own empty vault and must serve 0 hits (the adapter raises if it ever serves one, which would be an isolation breach). The reader gets the same scaffold with an empty retrieved block.
- `kscope`: on each chunk, during ingestion, the adapter cuts units (below), calls `common.writer.extract(unit)` once per unit, builds the item with `common.writer.to_item(unit, delta)` so `content_md` is the unit verbatim, and writes with `Vault.remember_items` in batches of <= 20 before `ingest` returns. `ingest` raises if it is ever called after a question has been seen. The writer's only input is the unit text.

**Agent names.** `Simple_rag_kscope` and `Simple_rag_kscope_none`. "rag" makes `utils/templates.py normalize_agent_name` return `rag_agent`, the same prompt template the shipped `Simple_rag_bm25` gets (checked by calling the function on all three names), and selects the rag-shaped save-folder path in `initialization.py`. "kscope" selects the adapter, in a branch placed ahead of the generic "rag" branch. Neither name contains any earlier dispatch key (letta, mem0, cognee, zep, knowl, agentmemory, Long_context_agent).

**Unit rule (domain-independent).** Sentences from the same nltk punkt tokenizer the harness chunks with, packed consecutively up to 350 tokens (harness tokenizer); each unit is a verbatim slice of the chunk; a sentence over budget becomes its own unit. One addition, decided before any score existed and stated here so it can be vetoed: punkt splits a bare list marker such as `12.` into a "sentence" of its own, so a segment containing no letter is kept with the sentence that follows it instead of being allowed to end a unit. Without it, roughly every other unit boundary would separate a serial number from its fact, and the serial number is this benchmark's only ordering signal. It glued 412 markers across the two chunks and is not specific to this corpus (any enumerated list). Result: 13 + 7 = 20 units; the 18 full-size ones are 335 to 350 tokens and hold 23 to 27 facts each, the two chunk tails are 201 and 45 tokens.

**The output cap, applied identically to both arms.** Shipped readers pass `max_tokens=generation_max_length` only when the model name contains "gpt-4" (`agent.py:896`); `gpt-5.6-luna` does not, so the 10-token cap of every Conflict_Resolution config would silently vanish, and under substring match a verbose answer reciting both values would score 1.0. An API cap of 10 cannot be used either, because reasoning tokens count against `max_completion_tokens`. So the reader is called with `max_completion_tokens=4000` and the VISIBLE answer is cut to its first 10 tokens (the harness's `gpt-4o-mini` tiktoken encoding) before it is returned as `output`. **Answers actually truncated: 1 of 20** (`none` q1: `'Creating the manga series *Rurouni Kenshin*.'` -> `'Creating the manga series *Rurouni Kens'`, wrong either way). No score in this smoke depended on the cap. No reader call ended on `length`; no answer was empty.

**Timing fields.** `input_len`/`output_len` are the API's `prompt_tokens`/`completion_tokens` (the latter includes reasoning). `query_time_len` is the reader call. `memory_construction_time` is the search time, plus, on the first answered question only, the ingestion wall time measured during ingestion (427.6 s) -- the same convention the harness's own memory agents follow, which is why the harness's per-question average reads 42.8 s.

## 3. Per question

<<SECTION: Per-question outputs and scores>>

Three of the ten questions (q3, q4, q9) have a single fact in the corpus and nothing to resolve; their gold answers are also true in the real world, and the `none` arm got q3 and q4 from what the reader already knew and declined q9. The other seven gold answers are counterfactual edits that contradict an earlier, real-world fact in the same context -- which is why the blind arm answers `Hockey`, `England`, `Football`.

## 4. The retrieval-level view

<<SECTION: kscope arm: did the served set contain the CURRENT value, the STALE value, both, or neither?>>

How the verdict lined up with the score, 10 questions, so an anecdote and not a rate: **both** 2 -> 2 correct (the reader chose the higher serial number); **current only** 4 -> 4 correct, but two of those (q4, q9) have no stale fact to serve; **stale only** 2 -> 0 correct, and both wrong answers ARE the stale value (`Pierre Beaumarchais`, `American football`); **neither** 2 -> 1 correct (q3, a single-fact family answered from world knowledge with nothing served; q2 answered "The knowledge pool does not specify.").

Restricted to the seven families that actually conflict: both 2, current only 2 (q1, q8), stale only 2 (q5, q7), neither 1 (q2). The split between "current only" and "stale only" is not recency at work. In q1 and q8 the floor happened to cut the stale fact's memory; in q5 and q7 it cut the current one. The next table shows what decided it.

<<SECTION: kscope arm: what happened to the memory holding each fact>>

Read down the last two columns. Counting both current and stale facts: in the delta -> 10 of 11 served (the exception is q8's stale fact, which one unit wrote as `Japan | uses | Japanese` while another wrote `Japan | has_official_language | Swedish`); not in the delta -> 0 of 6 served. No fact's memory was ever "not a candidate": retrieval found all 17, and the floor then cut 7.

### Diagnostic: the same 10 searches with the floor off (no LLM, no score)

To check that the floor, and not ranking, is what removed those memories, the writing arm's vault was cloned copy-on-write to `vaults/mab/<run_id>/diagnostic-parity/` and the 10 logged queries were replayed with `common/kscope_io.py`'s `parity` profile (similarity floor 0.0, redundancy cut off). `mab/parity_probe.py`; scope `mab-probe`; no reader was called, so there is no score for this and none should be inferred.

<<FILE: reports/mab_smoke_parity_generated.md>>

With the floor off every search serves 10 of the 20 memories, and the current fact's memory is in all 10 served sets. At 6k "in the top 10" is weak, since 10 is half the corpus; the ranks are the informative part: 1 or 2 wherever the fact was in a delta, 3 to 7 where it was not.

### What I think is happening, and what that rests on

The binary's own response names the cut (`omissions[].reason = below_similarity_floor`, `served_floor.similarity_floor = 0.45`). **[measured]** On branch `claude/served-score-floor` (commit `94c9cd7`, `crates/kaleidoscope-service/src/read_path.rs`) the floor reads `served_similarity = max(the memory's own cosine, the best cosine any of its projected facts or entities reached)`. **[source-read, and of a branch: the installed binary is pinned by sha256, not by commit, and the worktree I was given predates the floor, so I cannot show this is the code that ran]**. A 350-token unit of 24 unrelated facts has a blurred vector of its own, so against a one-fact question it clears 0.45 only through a projected fact -- which exists only if the writer put that fact in the delta. **[inference. In this sample a triple was necessary and not sufficient: 0 of 6 facts without one were served, 10 of 11 with one were. Not tested by intervention]**. The intervention that would test it costs one re-ingest: the same corpus at a unit size where the 32-entity cap does not bind (about 150 tokens, ~11 facts), same floor. I did not run it; changing the unit size after seeing a score is exactly the tuning the brief forbids, so it is the owner's call.

Two consequences worth stating plainly. (a) The two units stored with the fallback delta (0.0 and 0.8, 48 source facts) were not a candidate in any of the 10 searches (served 0, cut 0): a memory with no real triples was invisible here. None of the 10 questions asked about them, so this is an observation, not a measured loss. (b) 63 of the 100 candidate slots in the writing arm were cut by the floor and one search (q3) served nothing and abstained.

## 5. Served-set shape per search

<<SECTION: Served-set shape per search (assertion 6)>>

The retrieval query is what the harness extracts, e.g. `Based on the provided Knowledge Pool, Which sport is goaltender associated with? \nAnswer:` -- 9 of its 18 tokens are template boilerplate (`Based on the provided Knowledge Pool,` and `Answer:`), identical for every question and every baseline. It is passed as extracted except that whitespace runs become single spaces (section 8, first item). The `none` arm's `context_bytes` of 740 with zero hits is kscope's fixed context header; the adapter never reads `context_text`, only each hit's `content_md`.

## 6. The seven assertions

Each of these can go red, and one did while I was building the checker: assertion 4's prompt-size check failed on its first run because two writer calls carried 32 more input tokens than the rest. They were `common/writer.py`'s own retry message on the two calls that ended on `length`; the check now compares first attempts with first attempts and retries with retries.

### Assertion 1 — isolation

<<SECTION: Assertion 1 — isolation>>

Also under assertion 1: six more roots were touched under a separate scope, `mab-probe`, for work that calls no LLM, so that `roots_touched('mab')` stays the record of what the harness runs touched: four scratch vaults under `vaults/mab-probe/` (response-shape probes), one under `vaults/mab/probe-20260920T160327/` (the adapter's ingest path driven with a canned writer, to exercise batching, refusal and the guards for free), and the parity clone of section 4. All ten roots are under `vaults/`. The harness never created `MemoryAgentBench/agents/` or `MemoryAgentBench/outputs/` (checked after the run: neither exists). The adapter refuses to start if its agent save folder exists, and refuses to answer with zero ingested chunks; both guards were exercised in a no-LLM probe and both raised.

### Assertion 2 — personal vault untouched

<<SECTION: Assertion 2 — personal vault untouched>>

### Assertion 3 — propagation

<<SECTION: Assertion 3 — propagation>>

### Assertion 4 — no leaks

<<SECTION: Assertion 4 — no leaks>>

### Assertion 5 — failures, all counted

<<SECTION: Assertion 5 — failures, all counted>>

### Assertion 6 — served-set shape

The table in section 5: served count, omission reasons, `stop_reason` and `abstained` for all 20 searches. The same counters are appended per call to `logs/mab_adapter-<run_id>.jsonl`, with query length and whether the answer was truncated.

### Assertion 7 — spend

<<SECTION: Assertion 7 — spend>>

Scope `mab` in total: 44 calls, $1.331 at the cap rate, $0.242 at list price, against a $4 scope cap. 42 of those calls are this smoke run ($1.280); the other 2 are development (one `none`-arm reader call through `main.py`, one single-unit writer call). `llm.report()` above is the whole ledger and includes other agents' scopes. Both price pairs are `common/llm.py`'s: the cap is checked at the worst reported Azure billing rate ($1.10 / $6.60 per Mtok); list price is $0.20 / $1.20. Neither is a bill.

### What the writer kept

<<SECTION: What the writer kept per unit>>

Units 0.0 and 0.8 are the two kscope refused: the counts shown for them are what the writer produced, not what is stored (each is stored with the one-fact fallback delta). Fifteen units stopped at 15 to 20 facts to stay inside 32 entities; the two short units (0.12, 1.6) fit under the cap as they were. Three (0.5, 0.9, 1.2) kept every fact, declared 45 to 48 entities, and had `writer.py` clamp the list to 32, so 13 to 16 of their entities went in undeclared ("probed bare ... counted as degraded" per the contract) -- accepted. Across the 20 independent calls the writer coined 55 predicate names for 344 triples, and one relation got two names (`associated_with` x30, `associated_with_sport` x12), because no call sees what another coined. Since the slot key carries the predicate, two names for one relation cannot collide, which bounds what any write-time supersession could do on this corpus; I did not measure that.

## 7. Every harness patch, with its reason

`patches/memoryagentbench.diff` (`git -C MemoryAgentBench diff`, with the three new files marked intent-to-add so they appear; it reverse-applies cleanly against the tree that ran). Scoring, prompt templates, data loading and chunking are untouched.

| # | file | change | reason |
|---|---|---|---|
| 1 | `agent.py` top | `import torch` wrapped in `try/except ImportError` | imported at module top, used by none of the path we exercise; guarded rather than installing torch |
| 2 | `agent.py` top | `langchain_core.documents.Document` and four `transformers` names wrapped the same way | same; `Document` is used only inside the graph-rag, BM25 and self-rag handlers. With the packages present, behaviour is unchanged |
| 3 | `agent.py` `_initialize_agent_by_type` | +3 lines: a `kscope` branch ahead of `rag` | register the adapter; it must precede `rag` because our names contain "rag" on purpose |
| 4 | `agent.py` `send_message` | +3 lines: same branch, same position | route both the memorizing and the answering call to the adapter |
| 5 | `methods/kscope_agent.py` | new file | the adapter |
| 6 | `configs/agent_conf/RAG_Agents/gpt-5.6-luna/Kscope_gpt-5.6-luna.yaml`, `...-none.yaml` | new files | the two arms; every shared value copied from `Simple_rag_bm25`, no `temperature` key |

Outside the harness: `mab/run_mab.py` (launcher), `mab/env.sh`, `mab/smoke_check.py` (checker), `mab/parity_probe.py` (diagnostic), `mab/build_report.py`.

## 8. What surprised me, where the brief and the code disagree, and what I could not do

1. **kscope refuses the harness's retrieval query as extracted.** It contains a newline before `Answer:`, and `search` answers `invalid_arguments: Text rejects '\n' at byte 72` (exit 2). The adapter collapses whitespace runs to single spaces before searching and logs both strings; no word changes. Without this every search is a refusal. The adapter raises on any search failure rather than answering from an empty block.
2. **The harness resumes.** `initialization.load_existing_results` reloads an existing results file and `main.py` then skips every already-answered question -- after re-ingesting. A second run into the same `output_dir` would spend the whole ingestion budget and answer nothing. `--force` does not cover it. The launcher gives every run a fresh `results/mab/<run_id>/<arm>` and refuses an existing one.
3. **`chunk_size: 4096` is not 4096.** The chunker sums per-sentence token counts but joins with spaces, and punkt turns every `N.` into its own sentence, so the first chunk is 4,319 tokens. It also cuts between a serial number and its fact: chunk 0 ends `... 306. Thomas Kyd was born in the city of Leeds. 307.` and chunk 1 begins `Jesus Christ worked in the city of Galilee. 308. ...`. Fact 307 reaches every agent in this harness without its number. None of the 10 questions touched it.
4. **Brief vs code.** (a) `methods/knowl.py` has no `_extract_retrieval_query` of its own in this clone; it calls `agent._extract_retrieval_query` (`agent.py:911`), and so does ours. (b) SHARED_RULES assertion 4 as written ("gold answer occurrences in the ingested chunks, expected 0") cannot hold for FactConsolidation, where the answer is a fact the context states; the counts are reported and the check that can go red is on the question text, the writer-call timestamps, the call order in the kscope log and the writer prompt sizes. (c) Everything else I re-checked matched: the `max_test_queries` YAML key is written by `_apply_ablation_parameters` and read by nothing; the ingestion-skip trap is as described; `_create_standard_response`'s five keys are all consumed.
5. **`main.py` calls `dotenv.load_dotenv()`**, which walks up from the harness directory and loads the repository `.env` -- the Azure endpoint and key, and several unrelated secrets -- into the harness process. Nothing on our path prints the environment or constructs the harness's client, and I printed only the variable names while checking. Worth knowing before anyone adds a baseline that does use the harness client: with `AZURE_OPENAI_ENDPOINT` present it would talk to the real gateway outside the spend cap.
6. **The writer is nearly all of the cost and most of it is thinking.** 22 calls, 178,657 output tokens of which 149,839 reasoning; mean 64 s per call; two calls spent all 12,000 tokens reasoning and returned nothing (`finish=length`), and `writer.py`'s retry recovered both. Readers cost $0.026 for 20 answers.
7. **kscope refused 2 of 20 writer deltas**: `invalid type: string "person", expected a sequence` -- the writer wrote `propose[].from` as a string where the contract wants a list of kinds. The contract text says "The declared domain kinds" without showing the shape. Following the rule for extraction failures, the adapter re-sent each refused unit once with a minimal fallback delta (verbatim content, one placeholder fact) rather than letting it vanish; both were accepted. Those units' 35 extracted facts are not in the vault as triples. This is `common/writer.py`'s prompt, which is not mine to edit.
8. **Both `remember` batches came back `minted_over_budget: true`** (budget 8 entities and 3 predicates per call; a batch here mints hundreds). It is a report flag; the writes were accepted.
9. **An identical search is replayed, profile notwithstanding.** Replaying a logged query on the clone under `parity` returned the shipped result with the original `exposure_id`. The diagnostic uses a byte budget of 32767 to make the request distinct and asserts the response's floor reads 0.0.
10. **Could not**: build a Python 3.13 venv (`editdistance` has no cp313 wheel and this machine's C++ standard headers are not found by either clang), hence 3.10; use the file-writing tool for anything here (it refuses paths outside the session's git worktree, while the rules forbid writing inside one), so every file under this directory was written from the shell. nltk 3.10 needs `punkt_tab`, which the harness does not download; it was fetched once into `NLTK_DATA`.
11. **Not done, deliberately**: no BM25 arm. The shipped BM25 path calls the harness's own client, which is forbidden here; it would need its own thin adapter through the shared reader. At 6k it would also serve the entire context (2 chunks, `retrieve_num: 10`).

## 9. Before the full run

Stated as an estimate with its inputs, because the owner's standing rule is to see per-arm LLM calls before a matrix. Measured here: 20 units for a 5,988-token context (one unit per ~300 context tokens), $0.0627 per unit at the cap rate ($0.0114 at list) retries included, 64 s per writer call, 21 s per unit of wall time at 4 concurrent calls. The Conflict_Resolution split is 8 rows: four context sizes (26,157 / 136,565 / 273,473 / 1,118,123 characters; about 6k / 31k / 63k / 256k tokens at this corpus's 4.37 characters per token), each used twice -- **the `mh` and `sh` rows have byte-identical contexts at every size** (checked), but one vault per context per arm means ingesting each twice. That is about 712k tokens, about 2,400 writer calls, **about $149 at the cap rate or $27 at list price for the writing arm's ingestion alone**, and about 14 hours at 4 concurrent calls. The two 262k rows are 72% of it. The global cap is $15. The three `sh` rows below 262k alone are about 330 units, roughly $21 at the cap rate ($3.80 at list). Writer reasoning effort is the largest lever (84% of writer output tokens are reasoning) and it lives in `common/writer.py`.

The second thing to settle first is section 4: at this unit size a quarter to a third of the facts never become triples, and in this sample a fact that is not a triple was never served under the shipped floor. A full run at these settings would mostly measure that.

## 10. Reproduce

```
cd experiments/bench-2026-09-20 && . mab/env.sh
RUN_ID=smoke-$(date +%Y%m%dT%H%M%S)
./venv-mab/bin/python mab/run_mab.py --run_id $RUN_ID --arm none   --max_queries 10
./venv-mab/bin/python mab/run_mab.py --run_id $RUN_ID --arm kscope --max_queries 10   # ~8 min, ~$1.3 at the cap rate
./venv-mab/bin/python mab/smoke_check.py $RUN_ID                                       # exit 1 if any assertion is red
```

Artefacts of this run: `results/mab/smoke-20260920T160348/{none,kscope}/Conflict_Resolution/*_results.json` (harness output), `logs/mab_adapter-smoke-20260920T160348.jsonl` (per-call counters, unit texts, served hits), `logs/mab_harness-smoke-20260920T160348-{none,kscope}.log`, `logs/kscope_calls-mab.jsonl`, `spend/ledger.jsonl`, vaults under `vaults/mab/smoke-20260920T160348/`. No server was started; nothing of mine is left running.
