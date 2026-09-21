#!/usr/bin/env python
"""The 6k gate: does the writer stay inside `remember`'s caps at the unit size that ran?

Three conditions, fixed before the run that they judge (HANDOFF.md section 4):
  A  zero units reaching the 32-entity or 32-fact cap
  B  zero clamps
  C  zero unrecovered refusals

Plus the quantity the unit rule exists to move: the share of the corpus's numbered facts that
reach the vault as subject/predicate/object triples, rather than surviving only as prose inside
`content_md` where the read path's similarity floor cuts them.

Reads only what a run left behind -- the adapter log, the kscope call log, the spend ledger.
Calls no LLM and opens no vault, so it is free and can be run against any past run_id. Running it
against a run made under the OLD unit rule is the demonstration that it can go red.

Usage:  python mab/gate_check.py <run_id> [<run_id> ...]    (exit 1 if any run is RED)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
sys.path.insert(0, str(BASE))

from common import llm  # noqa: E402

CAPS = {"facts": 32, "entities": 32}  # common/writer.py CAPS; `remember` refuses above these
SCOPE = "mab"


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def covered_as_triple(sentence: str, triples) -> bool:
    """A numbered source fact counts as reaching the delta when some triple of the SAME unit has
    both its subject and its object present in that fact's own sentence. Endpoint-only on purpose:
    the predicate is the writer's coinage and never appears verbatim in the corpus."""
    haystack = norm(sentence)
    for triple in triples:
        if len(triple) < 3:
            continue
        subject, obj = norm(triple[0]), norm(triple[2])
        if len(subject) >= 2 and len(obj) >= 2 and subject in haystack and obj in haystack:
            return True
    return False


def numbered_facts(unit_text: str):
    """The corpus states one fact per numbered line. Returns [(serial, sentence)]."""
    return [(int(m.group(1)), m.group(2).strip())
            for m in re.finditer(r"(?:(?<=\s)|^)(\d+)\.\s+(.*?)(?=(?:\s\d+\.\s)|\Z)", unit_text, re.S)]


def check(run_id: str, out: list) -> bool:
    say = out.append
    adapter_log = BASE / "logs" / f"mab_adapter-{run_id}.jsonl"
    if not adapter_log.exists():
        say(f"## {run_id}\n\nNo adapter log at `{adapter_log}` — nothing to judge.\n")
        return False
    rows = jsonl(adapter_log)
    kw = [r for r in rows if r.get("writer") == "kscope"]
    units = [r for r in kw if r["event"] == "unit"]
    batches = [r for r in kw if r["event"] == "remember_batch"]
    resends = [r for r in kw if r["event"] == "fallback_resend"]
    inits = [r for r in kw if r["event"] == "init"]
    ingests = [r for r in kw if r["event"] == "ingest_chunk" and r.get("stored")]

    say(f"## Gate: `{run_id}`\n")
    if not units:
        say("No writing arm in this run: the gate judges the writer and has nothing to judge.\n")
        return False

    # ── proof that the rule under test is the rule that ran ─────────────────────────────────
    init = inits[0] if inits else {}
    per_unit_sentences = [u.get("unit_sentences") for u in units]
    known = [s for s in per_unit_sentences if s is not None]
    say("**The rule that ran** (from the adapter's own `init` and `ingest_chunk` rows, not from the config on disk):\n")
    say(f"- `kscope_unit_sentences` = {init.get('unit_sentences')}, `kscope_unit_tokens` = {init.get('unit_tokens')}, "
        f"profile `{init.get('profile')}`, writer prompt sha `{init.get('writer_prompt_sha')}`")
    say(f"- chunks ingested {len(ingests)}, units {len(units)}, "
        f"`max_sentences` recorded by the splitter: {sorted({i.get('max_sentences') for i in ingests})}")
    if known:
        say(f"- sentences per unit: min {min(known)}, median {sorted(known)[len(known) // 2]}, max {max(known)}")
    else:
        say("- sentences per unit: **not recorded** — this run predates the sentence cap")
    toks = [u["unit_tokens"] for u in units]
    say(f"- tokens per unit: min {min(toks)}, median {sorted(toks)[len(toks) // 2]}, max {max(toks)}\n")

    # ── condition A: the caps ────────────────────────────────────────────────────────────────
    at_cap = {key: [u for u in units if u.get(key, 0) >= cap] for key, cap in CAPS.items()}
    n_at_cap = len({(u["chunk_index"], u["unit_index"]) for key in CAPS for u in at_cap[key]})
    facts_per = [u["facts"] for u in units]
    ents_per = [u["entities"] for u in units]
    a_ok = n_at_cap == 0

    # ── condition B: clamps ──────────────────────────────────────────────────────────────────
    clamped = [u for u in units if u.get("clamped")]
    b_ok = not clamped

    # ── condition C: unrecovered refusals ────────────────────────────────────────────────────
    refused = [(b["chunk_index"], res["unit_index"], res.get("reason"))
               for b in batches for res in b.get("results", []) if res.get("status") != "created"]
    whole_call_failures = [b for b in batches if b.get("whole_call_failure")]
    lost = [r for r in resends if not r.get("stored")]
    c_ok = not lost

    say("**The three conditions**\n")
    say("| # | condition | measured | pass |")
    say("|---|---|---|---|")
    say(f"| A | zero units reaching the 32-entity or 32-fact cap | {n_at_cap} units at a cap "
        f"(max facts/unit {max(facts_per)}, max entities/unit {max(ents_per)}; headroom "
        f"{CAPS['facts'] - max(facts_per)} facts / {CAPS['entities'] - max(ents_per)} entities) | "
        f"**{'PASS' if a_ok else 'FAIL'}** |")
    say(f"| B | zero clamps | {len(clamped)} units clamped"
        f"{' ' + str([(u['chunk_index'], u['unit_index'], u['clamped']) for u in clamped]) if clamped else ''} | "
        f"**{'PASS' if b_ok else 'FAIL'}** |")
    say(f"| C | zero unrecovered refusals | {len(lost)} units lost "
        f"({len(refused)} items refused on first try, {len(resends)} fallback resends, "
        f"{len(whole_call_failures)} whole-call failures) | **{'PASS' if c_ok else 'FAIL'}** |")
    say("")
    if refused:
        say("Refusals, each with kscope's own reason:\n")
        for chunk_index, unit_index, reason in refused:
            say(f"- chunk {chunk_index} unit {unit_index}: `{str(reason)[:200]}`")
        say("")

    # ── the quantity the change exists to move ───────────────────────────────────────────────
    fallback_units = {(r["chunk_index"], r["unit_index"]) for r in resends if r.get("stored")}
    fallback_units |= {(u["chunk_index"], u["unit_index"]) for u in units if u.get("extraction_failed")}
    total_src = triple_covered = 0
    delta_facts_stored = 0
    per_unit = []
    for u in units:
        key = (u["chunk_index"], u["unit_index"])
        src = numbered_facts(u["unit_text"])
        stored_triples = [] if key in fallback_units else (u.get("delta_facts") or [])
        covered = sum(1 for _, sentence in src if covered_as_triple(sentence, stored_triples))
        total_src += len(src)
        triple_covered += covered
        delta_facts_stored += 0 if key in fallback_units else u["facts"]
        per_unit.append((key, u.get("unit_sentences"), u["unit_tokens"], len(src), u["facts"],
                         u["entities"], covered, key in fallback_units))
    prose_only = total_src - triple_covered
    say("**Facts as triples vs facts surviving only as prose** — a numbered source fact counts as a "
        "triple when a triple of its own unit carries both its subject and its object; a unit stored "
        "with the fallback delta contributes none.\n")
    say(f"- numbered source facts in the units: **{total_src}**")
    say(f"- reached the vault as a triple: **{triple_covered} ({triple_covered / total_src:.1%})**")
    say(f"- survive only as prose in `content_md`: **{prose_only} ({prose_only / total_src:.1%})**")
    say(f"- (the coarser count the smoke report used — delta facts actually stored, ignoring which source "
        f"fact each one is: {delta_facts_stored} for {total_src} source facts, {delta_facts_stored / total_src:.1%})")
    say(f"- units stored with the fallback delta (no real triples at all): {len(fallback_units)} {sorted(fallback_units)}\n")

    say("**Per unit** — `src` numbered facts in the unit, `delta` facts the writer returned, "
        "`ent` entities, `triples` source facts covered by a stored triple. `cap` marks a unit at a cap.\n")
    say("| chunk.unit | sent | tokens | src | delta | ent | triples | note |")
    say("|---|---|---|---|---|---|---|---|")
    for (ci, ui), sent, tok, src, delta, ent, cov, fb in per_unit:
        note = []
        if delta >= CAPS["facts"] or ent >= CAPS["entities"]:
            note.append("**cap**")
        if fb:
            note.append("fallback delta")
        say(f"| {ci}.{ui} | {sent if sent is not None else '-'} | {tok} | {src} | {delta} | {ent} | {cov} | "
            f"{', '.join(note) or ''} |")
    say("")

    # ── spend for this run ───────────────────────────────────────────────────────────────────
    ledger = [r for r in jsonl(BASE / "spend" / "ledger.jsonl")
              if r["scope"] == SCOPE and f":{run_id}:" in r.get("tag", "")]
    if ledger:
        by_role = {}
        for r in ledger:
            agg = by_role.setdefault(r["role"], {"calls": 0, "in": 0, "out": 0, "reasoning": 0})
            agg["calls"] += 1
            agg["in"] += r["in"]
            agg["out"] += r["out"]
            agg["reasoning"] += r.get("reasoning") or 0
        say("**Spend for this run** — the gateway returns no cost, so both figures are tokens times an "
            "assumed rate, neither is a bill.\n")
        say("| role | calls | input tokens | output tokens (of which reasoning) | at list $0.20/$1.20 | at pessimistic $1.10/$6.60 |")
        say("|---|---|---|---|---|---|")
        for role, a in sorted(by_role.items()):
            lst = a["in"] * llm.LIST_PRICE[0] / 1e6 + a["out"] * llm.LIST_PRICE[1] / 1e6
            cap = a["in"] * llm.PRICE_IN / 1e6 + a["out"] * llm.PRICE_OUT / 1e6
            say(f"| {role} | {a['calls']} | {a['in']:,} | {a['out']:,} ({a['reasoning']:,}) | ${lst:.3f} | ${cap:.3f} |")
        tot_in = sum(a["in"] for a in by_role.values())
        tot_out = sum(a["out"] for a in by_role.values())
        say(f"| **total** | {sum(a['calls'] for a in by_role.values())} | {tot_in:,} | {tot_out:,} | "
            f"${tot_in * llm.LIST_PRICE[0] / 1e6 + tot_out * llm.LIST_PRICE[1] / 1e6:.3f} | "
            f"${tot_in * llm.PRICE_IN / 1e6 + tot_out * llm.PRICE_OUT / 1e6:.3f} |")
        say(f"\nfinish reasons {dict(Counter(r['finish'] for r in ledger))}\n")

    green = a_ok and b_ok and c_ok
    say(f"### Verdict: {'GREEN' if green else 'RED'}"
        + ("" if green else " — " + ", ".join(
            name for name, ok in (("A caps", a_ok), ("B clamps", b_ok), ("C unrecovered refusals", c_ok)) if not ok))
        + "\n")
    return green


def main() -> int:
    out: list[str] = []
    results = {run_id: check(run_id, out) for run_id in sys.argv[1:]}
    out.append("---\n")
    out.append("Runs: " + ", ".join(f"`{k}` {'GREEN' if v else 'RED'}" for k, v in results.items()))
    print("\n".join(out))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
