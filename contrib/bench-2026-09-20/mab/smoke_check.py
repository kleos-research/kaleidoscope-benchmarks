#!/usr/bin/env python
"""Checker for a MemoryAgentBench smoke run. Reads what the run left behind; calls no LLM and no vault.

Independent of the adapter on purpose: it re-derives the chunks from the dataset with the harness's
own chunker, re-reads the harness's results files, the adapter log, the kscope call log and the spend
ledger, and recomputes every count rather than trusting the adapter's counters.

Usage:  python mab/smoke_check.py <run_id>     (prints markdown; exit 1 if any assertion is red)
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "MemoryAgentBench"))

from common import kscope_io, llm  # noqa: E402

SUB_DATASET = "factconsolidation_sh_6k"
PARQUET = BASE / "data" / "MemoryAgentBench__data__Conflict_Resolution-00000-of-00001.parquet"
SCOPE = "mab"


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def md_cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", "⏎")


def load_context():
    import pandas as pd
    frame = pd.read_parquet(PARQUET)
    row = next(r for _, r in frame.iterrows() if r["metadata"]["source"] == SUB_DATASET)
    return row["context"], list(row["questions"]), [list(a) for a in row["answers"]]


def parse_facts(context: str) -> list[tuple[int, str]]:
    facts = []
    for line in context.splitlines():
        match = re.match(r"^(\d+)\. (.*)$", line.strip())
        if match:
            facts.append((int(match.group(1)), match.group(2).strip()))
    return facts


def fact_family(question: str, golds: list[str], facts):
    """The facts that answer this question: same sentence stem, different object. Current = highest serial."""
    best = None
    q_words = set(re.findall(r"\w+", question.lower()))
    for serial, text in facts:
        for gold in golds:
            if text.rstrip(".").endswith(gold):
                stem = text.rstrip(".")[: -len(gold)]
                family = [(s, t) for s, t in facts if t.startswith(stem)]
                overlap = len(q_words & set(re.findall(r"\w+", stem.lower())))
                key = (overlap, len(family))
                if best is None or key > best[0]:
                    best = (key, stem, family)
    if best is None:
        return None, []
    return best[1], sorted(best[2])


def main() -> int:
    run_id = sys.argv[1]
    red: list[str] = []
    out: list[str] = []
    say = out.append

    context, questions, answers = load_context()
    facts = parse_facts(context)
    from utils.eval_other_utils import chunk_text_into_sentences, normalize_answer
    chunks = chunk_text_into_sentences(context, chunk_size=4096)
    import hashlib
    chunk_shas = [hashlib.sha256(c.encode()).hexdigest()[:16] for c in chunks]

    adapter = [r for r in jsonl(BASE / "logs" / f"mab_adapter-{run_id}.jsonl")]
    arms = {}
    for arm in ("none", "kscope"):
        rows = [r for r in adapter if r["writer"] == arm]
        files = glob.glob(str(BASE / "results" / "mab" / run_id / arm / "Conflict_Resolution" / "*_results.json"))
        arms[arm] = {"rows": rows, "results": json.load(open(files[0])) if files else None, "results_file": files[0] if files else None}
    kcalls = jsonl(BASE / "logs" / f"kscope_calls-{SCOPE}.jsonl")
    ledger = [r for r in jsonl(BASE / "spend" / "ledger.jsonl") if r["scope"] == SCOPE and f":{run_id}:" in r.get("tag", "")]

    n_questions = len(arms["kscope"]["results"]["data"]) if arms["kscope"]["results"] else 0

    # ── per-question table ───────────────────────────────────────────────────────────────────
    say("### Per-question outputs and scores\n")
    say("Score = the harness's own `substring_exact_match` (max over raw and parsed output), as written to its results file. "
        "`T` marks an answer the adapter truncated to 10 tokens.\n")
    say("| q | question | gold | none: output | none | kscope: output | kscope |")
    say("|---|---|---|---|---|---|---|")
    qrows = {arm: {r["query_id"]: r for r in arms[arm]["rows"] if r["event"] == "question"} for arm in arms}
    for qi in range(n_questions):
        cells = [str(qi), md_cell(questions[qi]), md_cell(" / ".join(answers[qi]))]
        for arm in ("none", "kscope"):
            data = arms[arm]["results"]["data"][qi]
            assert data["query_id"] == qi and data["answer"] == answers[qi], "results file does not line up with the dataset"
            mark = " `T`" if qrows[arm][qi]["truncated"] else ""
            cells += [md_cell(repr(data["output"])) + mark, str(int(bool(data["substring_exact_match"])))]
        say("| " + " | ".join(cells) + " |")
    say("")
    for arm in ("none", "kscope"):
        data = arms[arm]["results"]["data"]
        say(f"- **{arm}**: substring_exact_match {sum(bool(d['substring_exact_match']) for d in data)}/{len(data)}, "
            f"exact_match {sum(bool(d['exact_match']) for d in data)}/{len(data)}; truncated answers "
            f"{sum(r['truncated'] for r in qrows[arm].values())}/{len(data)}; empty answers "
            f"{sum(not r['raw_output'].strip() for r in qrows[arm].values())}; reader finish reasons "
            f"{dict(Counter(r['reader_finish'] for r in qrows[arm].values()))}")
    say("")

    # ── current vs stale in the served set ──────────────────────────────────────────────────
    say("### kscope arm: did the served set contain the CURRENT value, the STALE value, both, or neither?\n")
    say("For each question the checker finds the fact family in the source context (same sentence stem, different object). "
        "The highest serial number is the current fact; its object must equal the gold answer. A fact counts as served when "
        "its sentence text appears in some served hit's `content_md`; `#` shows whether its serial number is still attached to it there.\n")
    say("| q | fact family (serial: object) | current served | stale served | verdict | served | kscope score |")
    say("|---|---|---|---|---|---|---|")
    verdicts = Counter()
    families = {}
    for qi in range(n_questions):
        stem, family = fact_family(questions[qi], answers[qi], facts)
        families[qi] = (stem, family)
        served = [h["content_md"] for h in qrows["kscope"][qi]["served_hits"]]
        blob = "\n".join(served)
        if not family:
            say(f"| {qi} | (no family found) | ? | ? | ? | {len(served)} | |")
            red.append(f"q{qi}: checker found no fact family")
            continue
        current = family[-1]
        stale = family[:-1]
        if not any(current[1].rstrip(".").endswith(g) for g in answers[qi]):
            red.append(f"q{qi}: highest-serial fact does not carry the gold answer")

        def shown(serial, text):
            if text not in blob:
                return None
            return "#" if f"{serial}. {text}" in blob else "no #"
        cur = shown(*current)
        stl = [(s, shown(s, t)) for s, t in stale]
        stale_served = [f"{s} ({m})" for s, m in stl if m]
        verdict = ("both" if cur and stale_served else "current only" if cur else "stale only" if stale_served else "neither")
        verdicts[verdict] += 1
        fam = "; ".join(f"{s}: {t.rstrip('.')[len(stem):]}" for s, t in family)
        score = int(bool(arms["kscope"]["results"]["data"][qi]["substring_exact_match"]))
        say(f"| {qi} | `{md_cell(stem.strip())}` → {md_cell(fam)} | {('yes (' + cur + ')') if cur else 'NO'} | "
            f"{', '.join(stale_served) if stale_served else ('none' if stale else 'n/a (no stale fact)')} | **{verdict}** | {len(served)} | {score} |")
    say("")
    say(f"Verdict counts over {n_questions} questions: {dict(verdicts)}.\n")

    say("### kscope arm: what happened to the memory holding each fact\n")
    say("Each fact lives in exactly one stored memory (one unit). For each question: was that memory served (and at what rank), was it a candidate the read path cut "
        "(with kscope's own omission reason), or was it never among the 10 candidates? `in delta` says whether the writer's structured delta for that unit "
        "carries the fact as a subject/predicate/object triple, or whether it survives only in the verbatim `content_md`.\n")
    say("| q | fact | unit | in delta | fate of its memory |")
    say("|---|---|---|---|---|")
    krows_all = arms["kscope"]["rows"]
    unit_rows = [r for r in krows_all if r["event"] == "unit"]
    mem_of_unit = {}
    for b in (r for r in krows_all if r["event"] == "remember_batch"):
        for res in b.get("results", []):
            if res["status"] == "created":
                mem_of_unit[(b["chunk_index"], res["unit_index"])] = res["memory_id"]
    fallback_units = set()
    for r in (r for r in krows_all if r["event"] == "fallback_resend" and r["stored"]):
        mem_of_unit[(r["chunk_index"], r["unit_index"])] = r["memory_id"]
        fallback_units.add((r["chunk_index"], r["unit_index"]))
    fates = Counter()
    for qi in range(n_questions):
        stem, family = families[qi]
        q = qrows["kscope"][qi]
        rank_of = {h["memory_id"]: h["rank"] for h in q["served_hits"]}
        reason_of = {o.get("memory_id"): o.get("reason") for o in (q["omissions"] or [])}
        stem_words = set(re.findall(r"\w+", stem.lower())) - {"the", "of", "is", "in", "was", "with", "a"}
        for position, (serial, text) in enumerate(family):
            role = "CURRENT" if position == len(family) - 1 else "stale"
            unit = next(u for u in unit_rows if text in u["unit_text"])
            key = (unit["chunk_index"], unit["unit_index"])
            obj = text.rstrip(".")[len(stem):].strip().lower()
            if key in fallback_units:
                in_delta = "no (unit stored with the fallback delta)"
            else:
                hit = [f for f in unit["delta_facts"] if obj in f"{f[0]} {f[2]}".lower()
                       and stem_words & set(re.findall(r"\w+", f"{f[0]} {f[1]} {f[2]}".lower()))]
                in_delta = ("yes: `" + md_cell(" | ".join(map(str, hit[0]))) + "`") if hit else "NO (content_md only)"
            memory_id = mem_of_unit[key]
            if memory_id in rank_of:
                fate = f"served, rank {rank_of[memory_id]} of {q['served']}"
                fates[(role, "served")] += 1
            elif memory_id in reason_of:
                fate = f"candidate, CUT: `{reason_of[memory_id]}`"
                fates[(role, "cut: " + str(reason_of[memory_id]))] += 1
            else:
                fate = "never a candidate (not among the 10 the read path considered)"
                fates[(role, "never a candidate")] += 1
            say(f"| {qi} | {role} {serial}: {md_cell(text)} | {key[0]}.{key[1]} | {in_delta} | {fate} |")
    say("")
    say("Fate counts: " + "; ".join(f"{role} {fate}: {n}" for (role, fate), n in sorted(fates.items())) + ".\n")

    # ── served-set shape ─────────────────────────────────────────────────────────────────────
    say("### Served-set shape per search (assertion 6)\n")
    say("| arm | q | query chars / tokens | served | omitted | omission reasons | stop_reason | abstained | context_bytes | similarity floor: scored / dropped |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ("none", "kscope"):
        for qi in range(n_questions):
            r = qrows[arm][qi]
            reasons = dict(Counter(o.get("reason") for o in (r["omissions"] or [])))
            floor = r["served_floor"] or {}
            say(f"| {arm} | {qi} | {r['query_chars']} / {r['query_tokens']} | {r['served']} | {r['omitted_hits']} | {reasons or '-'} | "
                f"{r['stop_reason']} | {r['abstained']} | {r['context_bytes']} | {floor.get('similarity_scored')} / {floor.get('similarity_dropped')} |")
    say("")

    # ── assertion 1: isolation ───────────────────────────────────────────────────────────────
    say("### Assertion 1 — isolation\n")
    inits = {arm: [r for r in arms[arm]["rows"] if r["event"] == "init"] for arm in arms}
    touched = sorted(kscope_io.roots_touched(SCOPE))
    this_run = [r for r in touched if f"/vaults/mab/{run_id}/" in r]
    other = [r for r in touched if r not in this_run]
    expect = sorted(i["vault"] for arm in arms for i in inits[arm])
    ok = (all(len(inits[arm]) == 1 for arm in arms) and sorted(this_run) == expect
          and all(r.startswith(str(BASE / "vaults" / "mab") + "/") for r in touched))
    none_root = inits["none"][0]["vault"] if inits["none"] else None
    none_writes = [c for c in kcalls if c["root"] == none_root and c["op"] == "remember"]
    ok = ok and not none_writes
    say(f"- one context was run per arm, so one vault per arm is expected. Vaults created by this run ({len(this_run)}):")
    for root in this_run:
        calls = Counter(c["op"] for c in kcalls if c["root"] == root)
        say(f"  - `{root}` — calls: {dict(calls)}")
    say(f"- `roots_touched('{SCOPE}')` has {len(touched)} roots in total: the {len(this_run)} above plus {len(other)} from my earlier dev runs under the same scope:")
    for root in other:
        say(f"  - `{root}`")
    say(f"- every root is under `vaults/mab/`: {all(r.startswith(str(BASE / 'vaults' / 'mab') + '/') for r in touched)}; "
        f"`remember` calls against the none arm's vault: {len(none_writes)}")
    say(f"- **{'GREEN' if ok else 'RED'}**\n")
    if not ok:
        red.append("assertion 1 isolation")

    # ── assertion 2: personal vault ──────────────────────────────────────────────────────────
    say("### Assertion 2 — personal vault untouched\n")
    personal = str(kscope_io.PERSONAL_VAULT)
    all_logs = sorted(glob.glob(str(BASE / "logs" / "kscope_calls-mab*.jsonl")))
    naming = [(p, c) for p in all_logs for c in jsonl(p) if personal in json.dumps(c) or not str(c.get("root", "")).startswith(str(BASE / "vaults") + "/")]
    say(f"- rows in {[Path(p).name for p in all_logs]} that name `{personal}` or any root outside `vaults/`: {len(naming)} "
        f"of {sum(len(jsonl(p)) for p in all_logs)} rows")
    say(f"- **{'GREEN' if not naming else 'RED'}**\n")
    if naming:
        red.append("assertion 2 personal vault")

    # ── assertion 3: propagation ─────────────────────────────────────────────────────────────
    say("### Assertion 3 — propagation\n")
    krows = arms["kscope"]["rows"]
    ingests = [r for r in krows if r["event"] == "ingest_chunk"]
    units = [r for r in krows if r["event"] == "unit"]
    batches = [r for r in krows if r["event"] == "remember_batch"]
    resends = [r for r in krows if r["event"] == "fallback_resend"]
    kroot = inits["kscope"][0]["vault"]
    accepted_by_kscope = sum((c.get("accepted") or 0) for c in kcalls if c["root"] == kroot and c["op"] == "remember")
    shas_ok = [r["chunk_sha"] for r in ingests] == chunk_shas
    rejoined = all("".join("".join(u["unit_text"] for u in units if u["chunk_index"] == ci).split()) == "".join(chunk.split())
                   for ci, chunk in enumerate(chunks))
    created = {res["memory_id"]: (b["chunk_index"], res["unit_index"]) for b in batches for res in b.get("results", []) if res["status"] == "created"}
    created.update({r["memory_id"]: (r["chunk_index"], r["unit_index"]) for r in resends if r["stored"]})
    say(f"- chunks the harness handed to the adapter: {len(ingests)}; their sha256 equal the checker's own re-chunking of the dataset: {shas_ok}")
    say(f"- units cut from those chunks: {len(units)}; the units re-join to exactly the chunks' non-space characters: {rejoined}")
    say(f"- memories accepted, counted from the kscope call log (not the adapter): {accepted_by_kscope}; distinct memory ids: {len(created)}")
    prop_ok = shas_ok and rejoined and len(units) == accepted_by_kscope == len(created) and len(units) > 0
    # a concrete served-after-written example
    example = None
    for qi in range(n_questions):
        for hit in qrows["kscope"][qi]["served_hits"]:
            if hit["memory_id"] in created:
                ci, ui = created[hit["memory_id"]]
                unit = next(u for u in units if u["chunk_index"] == ci and u["unit_index"] == ui)
                if unit["unit_text"] in hit["content_md"]:
                    example = (qi, hit, unit, ci, ui)
                    break
        if example:
            break
    served_ids = {h["memory_id"] for qi in range(n_questions) for h in qrows["kscope"][qi]["served_hits"]}
    foreign = served_ids - set(created)
    say(f"- served memory ids that were NOT created by this run's writes: {len(foreign)} (expected 0); distinct memories served across the {n_questions} searches: {len(served_ids)} of {len(created)}")
    if example:
        qi, hit, unit, ci, ui = example
        write_ts = next(b["ts"] for b in batches if any(res["memory_id"] == hit["memory_id"] for res in b.get("results", [])))
        say(f"- concrete example: question {qi} was served `{hit['memory_id']}` at rank {hit['rank']}; that id was returned by the `remember` call for chunk {ci} unit {ui} "
            f"at ts {write_ts}, {qrows['kscope'][qi]['ts'] - write_ts:.1f}s before the question was logged; the unit text (sha `{unit['unit_sha']}`, {unit['unit_tokens']} tokens) is contained verbatim in the served `content_md`, "
            f"which begins `{md_cell(hit['content_md'][:70])}…`")
    else:
        say("- no search served anything written earlier")
    prop_ok = prop_ok and example is not None and not foreign
    say(f"- **{'GREEN' if prop_ok else 'RED'}**\n")
    if not prop_ok:
        red.append("assertion 3 propagation")

    # ── assertion 4: no leaks ────────────────────────────────────────────────────────────────
    say("### Assertion 4 — no leaks\n")
    all_text = "\n".join(chunks)
    unit_blob = "\n".join(u["unit_text"] for u in units)
    q_hits = {qi: all_text.count(questions[qi]) + unit_blob.count(questions[qi]) for qi in range(n_questions)}
    gold_hits = {qi: sum(all_text.count(g) for g in answers[qi]) for qi in range(n_questions)}
    writer_calls = [r for r in ledger if r["role"] == "writer"]
    reader_calls = [r for r in ledger if r["role"] == "reader" and f":{run_id}:kscope:" in r["tag"]]
    first_q = next(r["ts"] for r in krows if r["event"] == "first_question")
    last_writer = max(r["ts"] for r in writer_calls)
    first_reader = min(r["ts"] for r in reader_calls)
    kc = [c for c in kcalls if c["root"] == kroot]
    last_remember = max(c["ts"] for c in kc if c["op"] == "remember")
    first_search = min(c["ts"] for c in kc if c["op"] == "search")
    unit_tokens = {u["tag"]: u["unit_tokens"] for u in units}
    seen_tags: set[str] = set()
    overhead = {"first attempt": set(), "retry": set()}
    for r in sorted(writer_calls, key=lambda row: row["ts"] - row["ms"] / 1000):
        overhead["retry" if r["tag"] in seen_tags else "first attempt"].add(r["in"] - unit_tokens[r["tag"]])
        seen_tags.add(r["tag"])
    overhead = {k: sorted(v) for k, v in overhead.items()}
    import tiktoken
    retry_message = "That was not one valid JSON object with a non-empty `facts` list. Return only the JSON object."
    retry_tokens = len(tiktoken.encoding_for_model("gpt-4o-mini").encode(retry_message))
    sub = all(u["unit_text"] in chunks[u["chunk_index"]] for u in units)
    say(f"- question text inside the ingested chunks or the writer's inputs (expected 0): {sum(q_hits.values())} occurrences over {n_questions} questions")
    say(f"- every writer input is a verbatim slice of an ingested chunk: {sub}")
    say(f"- writer calls for this run: {len(writer_calls)}; last writer call finished at ts {last_writer:.3f}; the adapter saw its first question at ts {first_q:.3f} "
        f"({first_q - last_writer:.1f}s later); first reader call finished at {first_reader:.3f}")
    say(f"- kscope call log for the writing vault: last `remember` started {last_remember:.3f}, first `search` started {first_search:.3f} ({first_search - last_remember:.1f}s later); "
        f"no `remember` after a `search`: {last_remember < first_search}")
    say(f"- writer prompt size check: API-reported input tokens minus the unit's own token count takes the values {overhead} across the {len(writer_calls)} writer calls "
        f"({len(seen_tags)} first attempts, {len(writer_calls) - len(seen_tags)} retries). A retry carries writer.py's fixed retry message ({retry_tokens} tokens of text plus two message frames), "
        "which is the whole difference between the two classes. A question appended to any one writer input would add ~20 tokens to that call alone and split a class.")
    say(f"- writer prompt sha over the run: {sorted({u['prompt_sha'] for u in units})}")
    say(f"- gold answer strings inside the ingested chunks: {gold_hits}. **This is non-zero by construction and is not a leak**: FactConsolidation asks about facts "
        "the context states, so the answer value is in the corpus the harness ingests for every agent. The 'expected 0' form of this check fits a benchmark whose answers are "
        "not in the ingested text; here the leak question is whether the QUESTION (or the gold as an answer to it) reached the writer, which the four lines above answer.")
    leak_ok = (sum(q_hits.values()) == 0 and sub and last_writer < first_q and last_writer < first_reader
               and last_remember < first_search and len(overhead["first attempt"]) == 1 and len(overhead["retry"]) <= 1)
    say(f"- **{'GREEN' if leak_ok else 'RED'}**\n")
    if not leak_ok:
        red.append("assertion 4 leaks")

    # ── assertion 5: failures ────────────────────────────────────────────────────────────────
    say("### Assertion 5 — failures, all counted\n")
    parse_fail = sum(u["parse_failures"] for u in units)
    extraction_failed = [u for u in units if u["extraction_failed"]]
    clamped = [u for u in units if u["clamped"]]
    refused = [(b["chunk_index"], res["unit_index"], res["reason"]) for b in batches for res in b.get("results", []) if res["status"] != "created"]
    whole = [b for b in batches if b.get("whole_call_failure")]
    lost = [r for r in resends if not r["stored"]]
    over_budget = [b for b in batches if b.get("minted_over_budget")]
    tracebacks = {}
    for arm in arms:
        log = (BASE / "logs" / f"mab_harness-{run_id}-{arm}.log").read_text(errors="replace")
        tracebacks[arm] = log.count("Traceback (most recent call last)")
    finish = Counter(r["finish"] for r in ledger)
    say(f"- writer calls {len(writer_calls)} for {len(units)} units (attempts {sum(u['writer_attempts'] for u in units)}); parse failures: {parse_fail}; "
        f"extraction failures stored with the fallback delta: {len(extraction_failed)} {[(u['chunk_index'], u['unit_index']) for u in extraction_failed]}")
    say(f"- units where writer.py clamped a list over its cap: {len(clamped)} {[(u['chunk_index'], u['unit_index'], u['clamped']) for u in clamped]}")
    say(f"- items kscope refused on first try: {len(refused)}" + ("".join(f"\n  - chunk {c} unit {u}: `{md_cell(reason)}`" for c, u, reason in refused)))
    say(f"- fallback resends: {len(resends)}; units lost after a second refusal: {len(lost)}; whole-call `remember` failures: {len(whole)}")
    say(f"- `remember` batches flagged `minted_over_budget` (a report flag; the write is still accepted): {len(over_budget)} of {len(batches)}")
    say(f"- harness exceptions (tracebacks in the harness logs): {tracebacks}; launcher exit codes are in the run section")
    say(f"- LLM finish reasons across this run's {len(ledger)} calls: {dict(finish)}; empty reader answers: "
        f"{sum(not r['raw_output'].strip() for arm in arms for r in qrows[arm].values())}")
    fail_ok = not lost and not any(tracebacks.values())
    say(f"- **{'GREEN' if fail_ok else 'RED'}** (green = nothing lost and no exception; the counts above are the finding)\n")
    if not fail_ok:
        red.append("assertion 5 failures")

    # ── what the writer kept ─────────────────────────────────────────────────────────────────
    say("### What the writer kept per unit\n")
    say("| chunk.unit | tokens | numbered facts in the unit | facts in the delta | entities | proposals | memory_type | title |")
    say("|---|---|---|---|---|---|---|---|")
    tot_src = tot_kept = 0
    for u in units:
        n_src = len(re.findall(r"(?:(?<=\s)|^)\d+\.(?=\s)", u["unit_text"]))
        tot_src += n_src
        tot_kept += u["facts"]
        say(f"| {u['chunk_index']}.{u['unit_index']} | {u['unit_tokens']} | {n_src} | {u['facts']} | {u['entities']} | {u['proposals']} | {u['memory_type']} | {md_cell(u['title'])} |")
    say(f"\nTotal: {tot_kept} facts in deltas for {tot_src} numbered facts in the units ({len(facts)} in the source context).\n")

    # ── spend ────────────────────────────────────────────────────────────────────────────────
    say("### Assertion 7 — spend\n")
    by_role = {}
    for r in ledger:
        key = (r["tag"].split(":")[2] if r["role"] == "reader" else "kscope", r["role"])
        agg = by_role.setdefault(key, {"calls": 0, "in": 0, "out": 0, "reasoning": 0, "ms": 0})
        agg["calls"] += 1
        agg["in"] += r["in"]
        agg["out"] += r["out"]
        agg["reasoning"] += r.get("reasoning") or 0
        agg["ms"] += r["ms"]
    say("| arm | role | calls | input tokens | output tokens (of which reasoning) | summed call time | cost at cap rate | cost at list rate |")
    say("|---|---|---|---|---|---|---|---|")
    for (arm, role), a in sorted(by_role.items()):
        cap = a["in"] * llm.PRICE_IN / 1e6 + a["out"] * llm.PRICE_OUT / 1e6
        lst = a["in"] * llm.LIST_PRICE[0] / 1e6 + a["out"] * llm.LIST_PRICE[1] / 1e6
        say(f"| {arm} | {role} | {a['calls']} | {a['in']:,} | {a['out']:,} ({a['reasoning']:,}) | {a['ms'] / 1000:.0f}s | ${cap:.3f} | ${lst:.3f} |")
    say(f"\n`llm.report()` at check time: `{llm.report()}`\n")
    served_models = Counter(r["model_served"] for r in ledger)
    say(f"Model served: {dict(served_models)}; temperature sent on any call: {any(r['temperature_sent'] for r in ledger)}.\n")

    say("### Checker verdict\n")
    say("RED: " + "; ".join(red) if red else "All checked assertions GREEN.")
    print("\n".join(out))
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
