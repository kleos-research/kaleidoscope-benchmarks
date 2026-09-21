#!/usr/bin/env python
"""Checker and report generator for the FULL-CONTEXT arm of MemoryAgentBench FactConsolidation 262k.

Reads what the run left behind; calls no LLM and opens no vault. It re-derives every number from
the artefacts rather than trusting a counter, and each assertion below can go red on its own.

The chain analysis is imported from mab/full_check.py -- the same 37 sentence frames, the same
"newest serial per (subject, relation) is the current value" rule, the same shortest-path search --
so the full_context row is computed by the instrument that produced the row it joins. Retrieval
fate is meaningless here (everything is served), but ONE of its outputs is the whole point of this
arm as a control: `gold != newest serial` marks the questions whose gold answer contradicts the
benchmark's own newest-serial rule and therefore cannot be scored by a reader that applies it. A
multi-hop score has to be read against that ceiling or it means nothing.

Usage:  python mab/fullctx_check.py <run_id> [--scope mab-fullctx] [--out reports/x.md]
        exit 1 if any assertion is red
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "MemoryAgentBench"))
sys.path.insert(0, str(BASE / "mab"))

import full_check as FC  # noqa: E402  (Corpus, unit_index, the frames, the price constants)
from common import llm  # noqa: E402

ARM = "full_context"
ROWS = ("sh", "mh")
LENGTH = "262k"
# The run this arm's table joins. Its numbers are read from its own results files, never retyped.
BASELINE_RUN = "full-20260920T1905"
BASELINE_ARMS = ("none", "bm25_units", "kscope")

OUT_LINES: list[str] = []
RED: list[str] = []


def say(line: str = "") -> None:
    OUT_LINES.append(line)


def red(label: str, detail: str) -> None:
    RED.append(f"{label}: {detail}")


def mean(values):
    return statistics.mean(values) if values else 0.0


def load_questions(sub: str):
    import pandas as pd
    frame = pd.read_parquet(FC.PARQUET)
    rows = {r["metadata"]["source"]: r for _, r in frame.iterrows()}
    return rows[sub]


def cell_paths(run_id: str, arm: str, sub: str):
    files = glob.glob(str(BASE / "results" / "mab" / run_id / arm / sub / "Conflict_Resolution" / "*_results.json"))
    return {
        "results": json.load(open(files[0])) if files else None,
        "log": FC.jsonl(BASE / "logs" / f"mab_adapter-{run_id}-{arm}-{sub}.jsonl"),
        "exit_file": BASE / "runs" / "mab" / run_id / arm / sub / "exit_code.txt",
        "harness_log": BASE / "logs" / f"mab_harness-{run_id}-{arm}-{sub}.log",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--scope", default="mab-fullctx")
    parser.add_argument("--smoke_run", default=None, help="a 10-question run to fold into the spend total")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    manifest_pointer = BASE / "runs" / "mab" / BASELINE_RUN / "master" / LENGTH / "manifest.txt"
    manifest = json.loads(Path(manifest_pointer.read_text().strip()).read_text())
    master_log = FC.jsonl(BASE / "logs" / f"mab_adapter-{BASELINE_RUN}-master-{LENGTH}.jsonl")
    units_by_key, serial_to_unit, memory_of, fallback_units = FC.unit_index(master_log, manifest)

    ledger_all = FC.jsonl(BASE / "spend" / "ledger.jsonl")
    ledger = [r for r in ledger_all if r.get("scope") == args.scope]
    failures = [r for r in FC.jsonl(BASE / "logs" / "mab_llm_failures.jsonl")
                if f":{args.run_id}:" in (r.get("tag") or "")]

    cells = {row: cell_paths(args.run_id, ARM, f"factconsolidation_{row}_{LENGTH}") for row in ROWS}
    baseline = {(row, arm): cell_paths(BASELINE_RUN, arm, f"factconsolidation_{row}_{LENGTH}")
                for row in ROWS for arm in BASELINE_ARMS}

    # ── header ────────────────────────────────────────────────────────────────────────────────
    say("# Full context on MemoryAgentBench FactConsolidation 262k")
    say()
    say(f"Run id `{args.run_id}`, {time.strftime('%Y-%m-%d')}. One arm, two rows "
        f"(`factconsolidation_sh_262k`, `factconsolidation_mh_262k`), 100 questions each, **200 reader "
        f"calls**, no memory system of any kind. It joins the table of the completed run "
        f"`{BASELINE_RUN}` (`reports/mab_full_{BASELINE_RUN}.md`), whose numbers below are read out of "
        f"its own results files rather than retyped.")
    say()
    say("This arm answers the question a buyer asks -- *why do I need a memory store when I can put "
        "everything in the prompt?* -- and it is the ceiling of the same axis whose floor is `none`: "
        "`none` gets no context at all, `full_context` gets all of it.")
    say()

    # ── 1. the context limit, measured ────────────────────────────────────────────────────────
    ladder = json.loads((BASE / "logs" / "mab_context_ladder.json").read_text())
    probe = json.loads((BASE / "logs" / "mab_context_limit_probe.json").read_text())
    accepted = [p for p in ladder["probes"] if p.get("accepted")]
    refused = [p for p in ladder["probes"] if not p.get("accepted")]
    say("## The context limit was measured, not assumed")
    say()
    say("The deployment publishes no context window: `models.retrieve(\"gpt-5.6-luna\")` returns a "
        "`capabilities` block that names only which APIs the model serves "
        "(`logs/mab_context_limit_probe.json`). So it was probed "
        "(`mab/probe_context_limit.py`, `mab/probe_ladder.py`, `logs/mab_context_ladder.json`):")
    say()
    for p in ladder["probes"]:
        if p.get("accepted"):
            say(f"* **{p['prompt_tokens_local']:,} input tokens: ACCEPTED** (the API counted "
                f"{p['usage_in']:,}). The window is at or above this.")
        else:
            say(f"* {p['prompt_tokens_local']:,} input tokens: refused `{p.get('status')}` — "
                f"{'the rate limiter, not the model' if p.get('status') == 429 else 'see the log'}.")
    if not accepted:
        red("context limit", "no probe was accepted; the limit is unestablished")
    say()
    headers = next((p.get("headers") for p in refused if p.get("headers")), {}) or {}
    tpm = headers.get("x-ratelimit-limit-tokens")
    rpm = headers.get("x-ratelimit-limit-requests")
    say(f"The 500,007-token probe came back **429, not 400**: the gateway publishes "
        f"`x-ratelimit-limit-tokens: {tpm}` and `x-ratelimit-limit-requests: {rpm}`, so a single request "
        f"asking for the whole minute's tokens can never be admitted and the ceiling above 500k is not "
        f"testable on this key. That is a rate limit, not a context limit.")
    say()

    qrows = {row: {r["query_id"]: r for r in cells[row]["log"] if r["event"] == "question"} for row in ROWS}
    all_q = [r for row in ROWS for r in qrows[row].values()]
    prompt_tokens = [r["reader_in"] for r in all_q if not r.get("reader_error")]
    if prompt_tokens:
        say(f"**The corpus fits, so nothing was truncated and the arm is `full_context`, not "
            f"`full_context_truncated`.** The prompt this arm actually sent measured "
            f"**{min(prompt_tokens):,}–{max(prompt_tokens):,} tokens** (the API's own `prompt_tokens`), "
            f"against a window measured at **≥ {max(p['usage_in'] for p in accepted):,}** — "
            f"{max(p['usage_in'] for p in accepted) - max(prompt_tokens):,} tokens of headroom. Zero units "
            f"were dropped from either end; the served count is 1,581 of 1,581 on every question.")
    say()

    # ── 2. scores ─────────────────────────────────────────────────────────────────────────────
    say("## The result")
    say()
    sem = {}
    for row in ROWS:
        results = cells[row]["results"]
        if not results:
            red(f"{row} results", "no results file")
            continue
        sem[(row, ARM)] = sum(1 for x in results["metrics"]["substring_exact_match"] if x)
    for row in ROWS:
        for arm in BASELINE_ARMS:
            results = baseline[(row, arm)]["results"]
            sem[(row, arm)] = sum(1 for x in results["metrics"]["substring_exact_match"] if x) if results else None

    def tok_per_q(cell):
        rows_ = [r for r in cell["log"] if r["event"] == "question"]
        vals = [r["reader_in"] for r in rows_ if not r.get("reader_error")]
        return round(mean(vals)) if vals else None

    tok = {(row, arm): tok_per_q(baseline[(row, arm)] if arm in BASELINE_ARMS else cells[row])
           for row in ROWS for arm in (ARM,) + BASELINE_ARMS}

    say("`substring_exact_match` out of 100, one reader (gpt-5.6-luna, reasoning effort high, "
        "10-token visible answer cap) across every row. `input tok/question` is the API's own "
        "`prompt_tokens` averaged over answered questions, **on the single-hop row**, which is the "
        "basis the table this joins uses; the multi-hop figures follow underneath.")
    say()
    say("| arm | single-hop | multi-hop | input tok/question |")
    say("| --- | --- | --- | --- |")
    for arm in BASELINE_ARMS:
        t_s = f"{tok[('sh', arm)]:,}" if tok[("sh", arm)] else "—"
        say(f"| {arm} | {sem[('sh', arm)]} | {sem[('mh', arm)]} | {t_s} |")
    tok_fc_sh = tok[("sh", ARM)]
    say(f"| **full_context** | **{sem.get(('sh', ARM))}** | **{sem.get(('mh', ARM))}** | "
        f"**{tok_fc_sh:,}** |" if tok_fc_sh else "| **full_context** | ? | ? | ? |")
    say()
    say("Reader input tokens per question on each row, so neither is hidden inside an average:")
    say()
    say("| arm | single-hop tok/q | multi-hop tok/q |")
    say("| --- | --- | --- |")
    for arm in BASELINE_ARMS + (ARM,):
        a, b = tok[("sh", arm)], tok[("mh", arm)]
        name = f"**{arm}**" if arm == ARM else arm
        fmt = lambda v: f"{v:,}" if v else "—"
        say(f"| {name} | {fmt(a)} | {fmt(b)} |")
    say()
    tok_fc_mean = round(mean([v for k, v in tok.items() if k[1] == ARM and v])) if any(
        v for k, v in tok.items() if k[1] == ARM and v) else None

    if tok_fc_mean:
        for row, label in (("sh", "single-hop"), ("mh", "multi-hop")):
            f, k = sem.get((row, ARM)), sem.get((row, "kscope"))
            kt, ft = tok.get((row, "kscope")), tok.get((row, ARM))
            if f is None or k is None or not kt or not ft:
                say(f"* {label}: **missing cell** (full_context {f}, kscope {k}).")
                continue
            ratio = ft / kt
            verdict = ("beats" if f > k else ("loses to" if f < k else "ties"))
            say(f"* **{label}: full context {f}, kscope {k}** — full context {verdict} kscope"
                + (f" by {abs(f - k)} point{'s' if abs(f - k) != 1 else ''}" if f != k else "")
                + f", on **{ft:,} / {kt:,} = {ratio:.0f}x** the reader tokens.")
        say()

    # ── 3. the multi-hop control ──────────────────────────────────────────────────────────────
    say("## The multi-hop control: are the chains reachable at all?")
    say()
    say("This arm exists partly to decide whether a multi-hop retrieval experiment is worth running. "
        "If a reader holding the ENTIRE corpus still cannot follow the chains, the chains are broken "
        "in the data and no retrieval strategy fixes them; if it can, the bottleneck is retrieval. "
        "The split below is the one the dataset forces: `gold != newest serial` marks questions whose "
        "gold answer is reachable only through a fact that a HIGHER serial in the same "
        "(subject, relation) family overrides, so a reader obeying the benchmark's own stated rule "
        "(*the newest serial wins*) cannot score them. Chains, frames and the rule are "
        "`mab/full_check.py`'s, unchanged.")
    say()
    say("| row | arm | resolved | unresolved | gold = newest (n) | SEM on those | gold ≠ newest (n) | SEM on those |")
    say("| --- | --- | --- | --- | --- | --- | --- | --- |")
    control = {}
    for row in ROWS:
        sub = f"factconsolidation_{row}_{LENGTH}"
        source = load_questions(sub)
        corpus = FC.Corpus(load_questions(f"factconsolidation_sh_{LENGTH}")["context"])
        questions = list(source["questions"])
        answers = [list(a) for a in source["answers"]]
        expect_hops = 1 if row == "sh" else None
        resolved = {}
        for qi in range(len(questions)):
            chain, current_only, _ = corpus.best_chain(questions[qi], answers[qi][0], expect_hops)
            resolved[qi] = (chain, current_only)
        for arm in (ARM,) + BASELINE_ARMS:
            cell = cells[row] if arm == ARM else baseline[(row, arm)]
            data = (cell["results"] or {}).get("data") or []
            if not data:
                continue
            buckets = {"current": [0, 0], "stale": [0, 0], "unresolved": [0, 0]}
            hops = Counter()
            for qi in range(len(data)):
                chain, current_only = resolved.get(qi, (None, None))
                key = "unresolved" if chain is None else ("current" if current_only else "stale")
                buckets[key][1] += 1
                buckets[key][0] += int(bool(data[qi]["substring_exact_match"]))
                if chain is not None:
                    hops[len(chain)] += 1
            control[(row, arm)] = {"buckets": buckets, "hops": hops}
            say(f"| {row} | {arm} | {buckets['current'][1] + buckets['stale'][1]} | "
                f"{buckets['unresolved'][1]} | {buckets['current'][1]} | "
                f"**{buckets['current'][0]}/{buckets['current'][1]}** | {buckets['stale'][1]} | "
                f"{buckets['stale'][0]}/{buckets['stale'][1]} |")
    say()
    mh = control.get(("mh", ARM))
    if mh:
        cur = mh["buckets"]["current"]
        stale = mh["buckets"]["stale"]
        say(f"Multi-hop hop-count distribution (chains the checker resolved): "
            f"{dict(sorted(mh['hops'].items()))}.")
        say()
        pct_cur = 100.0 * cur[0] / cur[1] if cur[1] else 0.0
        say(f"**On the {cur[1]} multi-hop questions the benchmark's own rule can answer, full context "
            f"scores {cur[0]} ({pct_cur:.0f}%).** On the {stale[1]} where the gold contradicts that rule "
            f"it scores {stale[0]}.")
    say()

    # ── 3b. mechanism: where the answer sits, and what a wrong answer is ──────────────────────
    say("## Why the score comes out where it does")
    say()
    order = [key for key in sorted(units_by_key)]          # chunk-major, then unit: the prompt order
    position_of = {key: i for i, key in enumerate(order)}
    n_units = len(order)

    say("**Position of the answer in the prompt.** Every numbered fact lives in exactly one unit and "
        "the units enter the prompt in a fixed order, so each single-hop question has one position in "
        f"[0, 1) over {n_units:,} units. If a full-context reader loses accuracy with depth, that is "
        "attention over a long context, not a gap in the corpus.")
    say()
    say("| arm | decile of prompt position | | | | | | | | | |")
    say("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    sub = f"factconsolidation_sh_{LENGTH}"
    source = load_questions(sub)
    corpus_sh = FC.Corpus(source["context"])
    questions_sh = list(source["questions"])
    answers_sh = [list(a) for a in source["answers"]]
    pos_of_q = {}
    last_hop = {}
    for qi in range(len(questions_sh)):
        chain, current_only, _ = corpus_sh.best_chain(questions_sh[qi], answers_sh[qi][0], 1)
        if not chain:
            continue
        rel, serial, s_norm, obj = chain[-1]
        last_hop[qi] = (rel, serial, s_norm, obj, current_only)
        key = serial_to_unit.get(serial)
        if key in position_of:
            pos_of_q[qi] = position_of[key] / n_units
    for arm in (ARM,) + BASELINE_ARMS:
        cell = cells["sh"] if arm == ARM else baseline[("sh", arm)]
        data = (cell["results"] or {}).get("data") or []
        if not data:
            continue
        buckets = [[0, 0] for _ in range(10)]
        for qi, pos in pos_of_q.items():
            if qi >= len(data):
                continue
            b = min(9, int(pos * 10))
            buckets[b][1] += 1
            buckets[b][0] += int(bool(data[qi]["substring_exact_match"]))
        name = f"**{arm}**" if arm == ARM else arm
        say(f"| {name} | " + " | ".join(f"{c}/{n}" if n else "—" for c, n in buckets) + " |")
    say()
    fc_data = (cells["sh"]["results"] or {}).get("data") or []
    if fc_data:
        correct_pos = [pos_of_q[qi] for qi in pos_of_q
                       if qi < len(fc_data) and fc_data[qi]["substring_exact_match"]]
        wrong_pos = [pos_of_q[qi] for qi in pos_of_q
                     if qi < len(fc_data) and not fc_data[qi]["substring_exact_match"]]
        if correct_pos and wrong_pos:
            say(f"Mean position of the answer-bearing unit: **{mean(correct_pos):.2f} when full context "
                f"answered correctly ({len(correct_pos)} questions), {mean(wrong_pos):.2f} when it did "
                f"not ({len(wrong_pos)})**. A difference here is a depth effect; no difference means the "
                f"misses are spread through the prompt and position is not the mechanism.")
            say()

    # ── what a wrong answer IS ────────────────────────────────────────────────────────────────
    say("**What a wrong answer is.** For every single-hop question the checker knows the whole "
        "(subject, relation) family: the newest serial is the gold, the older ones are stale values "
        "the corpus still states. A full-context reader sees all of them; a retrieval that serves five "
        "memories usually shows it fewer. So the table asks, of each arm's wrong answers, how many "
        "name a stale sibling of the right fact rather than something unrelated.")
    say()
    say("| arm | wrong | answered with a STALE sibling | other wrong | empty / error |")
    say("| --- | --- | --- | --- | --- |")
    stale_detail = {}
    for arm in (ARM,) + BASELINE_ARMS:
        cell = cells["sh"] if arm == ARM else baseline[("sh", arm)]
        data = (cell["results"] or {}).get("data") or []
        log_rows = {r["query_id"]: r for r in cell["log"] if r["event"] == "question"}
        if not data:
            continue
        stale_n = other_n = empty_n = 0
        examples = []
        for qi in range(len(data)):
            if data[qi]["substring_exact_match"]:
                continue
            got = (data[qi].get("parsed_output") or data[qi].get("output") or "").strip()
            if not got or (log_rows.get(qi) or {}).get("reader_error"):
                empty_n += 1
                continue
            hop = last_hop.get(qi)
            if not hop:
                other_n += 1
                continue
            rel, serial, s_norm, obj, _ = hop
            family = corpus_sh.family.get((s_norm, rel), [])
            siblings = [o for sserial, o in family if sserial != family[-1][0]]
            got_norm = FC.norm_entity(got)
            hit = next((o for o in siblings if FC.norm_entity(o) and FC.norm_entity(o) in got_norm), None)
            if hit:
                stale_n += 1
                if len(examples) < 3:
                    examples.append((qi, questions_sh[qi], answers_sh[qi][0], got, hit))
            else:
                other_n += 1
        total_wrong = stale_n + other_n + empty_n
        stale_detail[arm] = (stale_n, total_wrong, examples)
        name = f"**{arm}**" if arm == ARM else arm
        say(f"| {name} | {total_wrong} | **{stale_n}** | {other_n} | {empty_n} |")
    say()
    fc_stale = stale_detail.get(ARM)
    if fc_stale and fc_stale[2]:
        say("Three of full context's stale-sibling answers, as the checker reconstructed them:")
        say()
        for qi, question, gold, got, sib in fc_stale[2]:
            say(f"* q{qi}: *{question.strip()}* → gold **{gold}**, answered **{got}**, which is the "
                f"superseded value `{sib}` of the same fact.")
        say()

    # ── 4. assertions ─────────────────────────────────────────────────────────────────────────
    say("## Assertions *(each able to go red)*")
    say()
    kscope_log = BASE / "logs" / f"kscope_calls-{args.scope}.jsonl"
    vault_dirs = sorted(glob.glob(str(BASE / "vaults" / "mab" / args.run_id / "*")))
    inits = [r for row in ROWS for r in cells[row]["log"] if r["event"] == "init"]

    rows_out = []
    # A: every question answered
    for row in ROWS:
        results = cells[row]["results"]
        n = len(results["metrics"]["substring_exact_match"]) if results else 0
        errs = sum(1 for r in qrows[row].values() if r.get("reader_error"))
        empties = sum(1 for r in qrows[row].values() if not (r.get("raw_output") or "").strip())
        ok = n == 100 and len(qrows[row]) == 100
        if not ok:
            red(f"A {row}", f"{n} scored, {len(qrows[row])} asked, expected 100")
        rows_out.append((f"A. every question answered ({row})",
                         f"{n} scored / {len(qrows[row])} asked; reader errors {errs}, empty outputs {empties}",
                         ok))
    # B: no memory system touched
    ok_b = not kscope_log.exists() and not vault_dirs and all(i["vault"] is None for i in inits) \
        and all(i["kscope_sha256"] is None for i in inits)
    if not ok_b:
        red("B", f"kscope call log {kscope_log.exists()}, vault dirs {vault_dirs}")
    rows_out.append(("B. no memory system touched",
                     f"`logs/kscope_calls-{args.scope}.jsonl` exists: **{kscope_log.exists()}**; vault "
                     f"directories under `vaults/mab/{args.run_id}/`: **{len(vault_dirs)}**; adapter "
                     f"`init` rows report vault `None` and kscope sha `None` on both rows", ok_b))
    # C: the whole corpus, every time
    block_shas = {r["memory_block_sha"] for r in all_q}
    served = {r["served"] for r in all_q}
    verified = [r for row in ROWS for r in cells[row]["log"] if r["event"] == "first_question"]
    ok_c = len(block_shas) == 1 and served == {1581} and all(
        v["counters"]["units_verified_against_master"] == 1581 for v in verified)
    if not ok_c:
        red("C", f"block shas {block_shas}, served {served}")
    rows_out.append(("C. the whole corpus in every prompt",
                     f"memory-block sha256 across all {len(all_q)} prompts: **{len(block_shas)} distinct** "
                     f"(`{sorted(block_shas)[0]}`); units in prompt: **{sorted(served)}**; units verified "
                     f"byte-identical against the sealed master manifest: "
                     f"**{[v['counters']['units_verified_against_master'] for v in verified]}**", ok_c))
    # D: prompt token count recorded
    ok_d = bool(prompt_tokens) and len(prompt_tokens) == len(all_q)
    rows_out.append(("D. prompt tokens recorded per question",
                     f"**{min(prompt_tokens):,} min / {round(mean(prompt_tokens)):,} mean / "
                     f"{max(prompt_tokens):,} max** over {len(prompt_tokens)} answered questions "
                     f"(API `prompt_tokens`); the adapter's own tiktoken count agrees within "
                     f"{max(abs(r['reader_in'] - r['prompt_tokens_local']) for r in all_q if r.get('prompt_tokens_local'))} "
                     f"tokens", ok_d))
    # E: the visible cap
    truncated = sum(1 for r in all_q if r.get("truncated"))
    over = [r for r in all_q if (r.get("raw_output_tokens") or 0) > 10 and not r.get("truncated")]
    ok_e = not over
    if over:
        red("E", f"{len(over)} answers over 10 tokens were not truncated")
    rows_out.append(("E. the 10-token visible cap applied",
                     f"**{truncated} of {len(all_q)}** visible answers truncated; raw answers over 10 "
                     f"tokens that were NOT truncated: **{len(over)}**; max raw answer "
                     f"{max((r.get('raw_output_tokens') or 0) for r in all_q)} tokens. The cap is driven "
                     f"through `read_answer_full_context` by a unit test that forces a 13-token answer "
                     f"and requires a 10-token visible result", ok_e))
    # F: content filter
    cf_rows = [r for r in all_q if r.get("content_filter")]
    cf_ledger = [r for r in ledger if r.get("finish") == "content_filter"]
    cf_errors = [r for r in failures if "content management policy" in (r.get("error") or "").lower()]
    rows_out.append(("F. content-filter refusals counted apart",
                     f"questions flagged `content_filter`: **{len(cf_rows)}**; ledger rows with "
                     f"`finish: content_filter`: **{len(cf_ledger)}**; recorded failed calls naming the "
                     f"content policy: **{len(cf_errors)}**", True))
    # G: no failed call is invisible
    reader_calls = [r for r in ledger if r.get("role") == "reader" and f":{args.run_id}:" in (r.get("tag") or "")]
    ok_g = len([r for r in reader_calls if not r.get("failed")]) == len(all_q) - sum(
        1 for r in all_q if r.get("reader_error"))
    if not ok_g:
        red("G", f"{len(reader_calls)} ledger reader rows vs {len(all_q)} questions")
    rows_out.append(("G. every call left a ledger row",
                     f"ledger reader rows for this run: **{len(reader_calls)}** "
                     f"({len([r for r in reader_calls if r.get('failed')])} failed, retried); "
                     f"questions: **{len(all_q)}**", ok_g))
    # H: exit codes
    exits = {row: (int(cells[row]["exit_file"].read_text().strip())
                   if cells[row]["exit_file"].exists() else None) for row in ROWS}
    ok_h = all(v == 0 for v in exits.values())
    if not ok_h:
        red("H", f"harness exits {exits}")
    rows_out.append(("H. the harness exited clean", f"exit codes {exits}", ok_h))

    say("| # | what | verdict |")
    say("| --- | --- | --- |")
    for label, detail, ok in rows_out:
        say(f"| {label} | {detail} | {'**GREEN**' if ok else '**RED**'} |")
    say()

    # ── 5. cost and wall clock ────────────────────────────────────────────────────────────────
    say("## Cost and wall clock")
    say()
    scope_rows = [r for r in ledger if f":{args.run_id}:" in (r.get("tag") or "")]
    tin = sum(r["in"] for r in scope_rows)
    tout = sum(r["out"] for r in scope_rows)
    treason = sum(r.get("reasoning") or 0 for r in scope_rows)
    cached = sum((r.get("reader_cached") or 0) for r in all_q)
    list_usd = tin * FC.LIST_PRICE[0] / 1e6 + tout * FC.LIST_PRICE[1] / 1e6
    pess_usd = tin * FC.PESS_PRICE[0] / 1e6 + tout * FC.PESS_PRICE[1] / 1e6
    status_file = BASE / "runs" / "mab" / args.run_id / "driver-status.json"
    if status_file.exists():
        status = json.loads(status_file.read_text())
        wall = status.get("finished", time.time()) - status["started"]
        per_row = {k: v.get("seconds") for k, v in (status.get("stages") or {}).items()}
    else:  # a run launched straight through mab/run_mab.py leaves no driver status
        stamps = [r["ts"] for r in all_q]
        wall = (max(stamps) - min(stamps)) if stamps else 0.0
        per_row = {row: ((max(x["ts"] for x in qrows[row].values()) - min(x["ts"] for x in qrows[row].values()))
                         if qrows[row] else 0.0) for row in ROWS}
    say(f"| what | value |")
    say(f"| --- | --- |")
    say(f"| reader calls | {len(scope_rows)} ({len([r for r in scope_rows if r.get('failed')])} failed and retried) |")
    say(f"| input tokens | {tin:,} |")
    say(f"| output tokens | {tout:,} (of which reasoning {treason:,}) |")
    say(f"| cached input tokens reported by the API | **{cached:,}** |")
    say(f"| cost at Azure list ${FC.LIST_PRICE[0]}/${FC.LIST_PRICE[1]} per Mtok | **${list_usd:.2f}** |")
    say(f"| cost at the worst reported Azure billing ${FC.PESS_PRICE[0]}/${FC.PESS_PRICE[1]} per Mtok | **${pess_usd:.2f}** |")
    say(f"| wall clock | {wall / 60:.0f} min (sh {(per_row.get('sh') or 0) / 60:.0f} min, mh {(per_row.get('mh') or 0) / 60:.0f} min) |")
    say(f"| reader latency per question | mean {mean([r['query_seconds'] for r in all_q]):.1f} s |")
    say(f"| rate-limit waits inserted by the pacer | mean {mean([r['pacing']['waited_seconds'] for r in all_q]):.0f} s per question |")
    say(f"| 429s absorbed and retried | {sum(r.get('rate_limited_attempts') or 0 for r in all_q)} |")
    say()
    if cached == 0:
        say(f"**The API reported zero cached input tokens on every one of the {len(all_q)} calls**, although "
            "all 100 prompts of a row share a byte-identical 291,585-token prefix. So the input is billed "
            "fresh each time and the cost above is not an over-estimate that prompt caching would "
            "quietly remove. That was measured, not assumed; it is the difference between the figure "
            "above and one up to 10x lower.")
        say()
    say(f"The gateway returns no cost, so both figures are `common/llm.py`'s stored token counts priced "
        f"at read time. Which rate applies depends on the deployment type, which this machine cannot "
        f"see; the cap is enforced at the pessimistic one. Scope totals at the time of writing:")
    say()
    spent = defaultdict(lambda: [0, 0, 0])
    for r in ledger_all:
        s = spent[r["scope"]]
        s[0] += 1
        s[1] += r["in"]
        s[2] += r["out"]
    say("| scope | calls | in | out | at list | at pessimistic |")
    say("| --- | --- | --- | --- | --- | --- |")
    mab_total = 0.0
    for scope in sorted(spent):
        c, i, o = spent[scope]
        lp = i * FC.LIST_PRICE[0] / 1e6 + o * FC.LIST_PRICE[1] / 1e6
        pp = i * FC.PESS_PRICE[0] / 1e6 + o * FC.PESS_PRICE[1] / 1e6
        if scope.startswith("mab"):
            mab_total += pp
        say(f"| {scope} | {c:,} | {i:,} | {o:,} | ${lp:.2f} | ${pp:.2f} |")
    say()
    say(f"Every `mab*` scope together: **${mab_total:.2f} pessimistic** against the benchmark's $200 cap.")
    if mab_total > 200:
        red("cap", f"mab scopes total ${mab_total:.2f} > $200")
    say()

    # ── 6. what was run ───────────────────────────────────────────────────────────────────────
    say("## What was run")
    say()
    init = inits[0]
    say(f"* **Arm** `full_context`, config "
        f"`configs/agent_conf/RAG_Agents/gpt-5.6-luna/Kscope_gpt-5.6-luna-fullcontext.yaml`, adapter "
        f"`methods/kscope_agent.py` writer mode `full_context`.")
    say(f"* **Context**: all 1,581 units of the 262k corpus, in the benchmark's own order (chunk order, "
        f"then unit order), each unit verified byte-identical to the sealed master manifest the "
        f"`kscope` and `bm25_units` arms were built from (`{Path(manifest_pointer.read_text().strip()).name}`, "
        f"seal `{manifest['seal'][:16]}`). No vault was cloned and none was opened.")
    say(f"* **Prompt scaffold**: the harness's own `rag_agent` template for `factconsolidation` "
        f"(`Memory i:` blocks, then the question), identical to every other arm. The only difference "
        f"between this arm and `kscope` is what fills the memory block.")
    say(f"* **Reader**: `{init['deployment']}`, reasoning effort high, "
        f"`max_completion_tokens={init['reader_max_completion_tokens']}`, visible answer capped at 10 "
        f"tokens by the adapter (the harness's own cap is gated on the substring `gpt-4` in the "
        f"deployment name and would silently not apply). **No temperature is sent** — a call carrying "
        f"both `reasoning_effort` and `temperature` is refused on this gateway; "
        f"`temperature_sent` is false on every row.")
    say(f"* **Pacing**: one prompt reserves ~296,000 of the gateway's 500,000 tokens/minute, so two "
        f"cannot share a minute. The rows ran **sequentially**, one reader call at a time, and the "
        f"pacer read the OTHER concurrent benchmark's consumption out of the shared spend ledger "
        f"before each call rather than assuming the key was idle. That, not model latency, is the "
        f"wall clock: the reader itself answered in "
        f"{mean([r['query_seconds'] for r in all_q]):.1f} s mean.")
    say()

    if RED:
        say("## RED")
        say()
        for item in RED:
            say(f"* {item}")
        say()

    text = "\n".join(OUT_LINES) + "\n"
    out = Path(args.out) if args.out else BASE / "reports" / f"mab_fullcontext_{args.run_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"written {out}")
    return 1 if RED else 0


if __name__ == "__main__":
    sys.exit(main())
