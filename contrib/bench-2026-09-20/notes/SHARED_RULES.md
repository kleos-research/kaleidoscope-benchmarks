# Shared rules for every benchmark adapter in this directory

Owner's hard requirements (verbatim): "We don't want hacks." / "All experiments
are gonna have their individual vaults. It won't go into the global one." /
"first run it on five and ten examples, just see everything works end to end...
information is propagating as expected. No leaks, no failures, and then we
actually run the full set."

## What is already built and tested — USE IT, DO NOT EDIT IT
`common/` (import with `sys.path.insert(0, "<this dir>")`):
- `common/llm.py` — the ONLY door to the LLM. `chat(messages, role=, effort=, max_completion_tokens=, tag=)`
  returns `(text, usage_row)`; `create_raw(request_dict, role=, tag=)` is a guarded pass-through for
  requests that need `tools=` etc. and returns the raw SDK response. Both check a spend cap BEFORE each
  call and raise `BudgetExceeded` — never catch-and-continue that. Deployment is `gpt-5.6-luna` on the
  Azure gateway, reasoning effort `high`. It loads the key from the repo `.env` itself.
  **NEVER print, log, echo or write the API key or endpoint anywhere.**
  `llm.report()` gives the running spend. Prices are ASSUMED ($1.25/$10 per Mtok), say so when quoting.
- `common/kscope_io.py` — the ONLY door to kscope. `Vault.create(root, profile="shipped")`,
  `.remember_items(items)` (<=20 per call), `.search(query, top_k, maximum_context_bytes)`.
  It refuses any root outside `vaults/`, refuses the personal vault, refuses to reuse an existing root,
  pins the binary by sha256, and logs every call to `logs/kscope_calls-<scope>.jsonl`.
  `roots_touched(scope)` is the isolation proof.
- `common/writer.py` — `extract(unit_text, tag=)` -> `(delta|None, stats)`; `to_item(unit, delta)` builds the
  `remember` item with `content_md` = the unit VERBATIM. Prompt is domain-independent; `prompt_sha()`
  must be identical for every call in a run. If `extract` returns None, store the unit with the minimal
  fallback delta and COUNT it as an extraction failure — never skip it silently.
If you need something these lack, write a wrapper in your own module. Do not modify `common/*`
(another agent is using it concurrently).

## Environment
- Set `BENCH_SCOPE` and `BENCH_SCOPE_CAP_USD` (given in your brief) in the environment of every process
  that can make an LLM call, including servers you start.
- Make your own venv under this directory (`python3 -m venv`), install only what the harness path you
  exercise needs. Put `HF_HOME` under this directory. Never install into the system/conda python.
- All outputs, vaults, caches go under this directory. Never write inside a git worktree of the
  kaleidoscope repo, never under the harness's own `agents/` or `outputs/` trees.
- Do not use the kaleidoscope MCP tools (`mcp__kaleidoscope__*`) — they write to the owner's personal vault.
- Kill every server process you start before you finish.

## What counts as a hack (forbidden)
- Any code path where the memory system sees the question, the gold answer, or a judge verdict at write time.
- Buffering writes and performing them at query time with the query in scope.
- Normalising entity names, hand-writing predicates, or tuning the writer prompt to the benchmark.
- Serving text from a Python dict instead of from a kscope `search` response.
- Reusing a vault across tasks/contexts, or addressing a vault by `--profile`, `KSCOPE_ROOT` or cwd.
- Changing harness scoring, prompts or data. Patches to the harness must be the minimum needed to
  register the adapter and route LLM calls through `common/llm.py`; save them as
  `patches/<benchmark>.diff` (`git -C <clone> diff`).

## Smoke-pass assertions every report must carry (each must be able to go red)
1. isolation: one vault per task/context per arm; list them; `roots_touched` contains nothing else.
2. personal vault untouched by the harness: no call in the log names it (the guard makes this structural).
3. propagation: per task, chunks handed to the write door == memories accepted by kscope; and at least
   one later search actually served content written by an earlier step (show one concrete example).
4. no leaks: count occurrences of each gold answer string inside the ingested chunks; expected 0.
5. failures: extraction parse failures, refused items, harness exceptions — all counted, all listed.
6. served-set shape per search: served count, omission reasons, stop_reason, abstained.
7. spend: `llm.report()` at the end.
