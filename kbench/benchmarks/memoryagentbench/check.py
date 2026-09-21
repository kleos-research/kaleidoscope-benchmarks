"""Checker and report generator for the FULL MemoryAgentBench run. Reads what the run left behind;
calls no LLM and opens no vault (it hashes the master vaults, which is a read).

Independent of the adapter on purpose: it re-derives the chunks from the dataset parquet with the
harness's own chunker, re-cuts the units with the adapter's rule and compares shas, re-reads the
harness's results files, the adapter logs, the master manifests, the kscope call log and the spend
ledger, and recomputes every count rather than trusting a counter.

The multi-hop analysis rests on the corpus's own sentence templates (37 relation frames, induced
from the text and covering every numbered fact; the checker refuses to run if one is left over):
each fact becomes (subject, relation, object), the newest serial per (subject, relation) is the
CURRENT value, and a question's chain is the shortest path of current facts from an entity named
in the question to the gold answer. Single-hop questions are the depth-1 case of the same search.

Usage:  python -m kbench.benchmarks.memoryagentbench.check <run_id> [--lengths 6k,32k,64k,262k] [--out report.md]
        exit 1 if any assertion is red
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("MAB_WORKDIR", HERE.parents[2] / "results" / "memoryagentbench"))
os.environ.setdefault("MAB_WORKDIR", str(BASE))
os.environ.setdefault("MAB_HARNESS_DIR", str(HERE))
HARNESS = BASE / "MemoryAgentBench"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HARNESS))

import kscope_io
import llm_client as llm
import writer

PARQUET = BASE / "data" / "Conflict_Resolution-00000-of-00001.parquet"
SCOPE = "mab"
ARMS = ("none", "bm25_units", "kscope")
ROWS = ("sh", "mh")
LENGTHS = ("6k", "32k", "64k", "262k")
PIN = os.environ.get("BENCH_KSCOPE_SHA", "")  # optional content pin; empty = unpinned
LIST_PRICE = llm.LIST_PRICE
PESS_PRICE = (llm.PRICE_IN, llm.PRICE_OUT)

# The corpus's sentence frames (MQuAKE cloze templates as this dataset renders them, typos included).
# `s` and `o` are the subject and object slots. Order matters only for the last, catch-all frame.
TEMPLATES = [
    ("citizen_of", r"^(?P<s>.+?) is a citizen of (?P<o>.+)$", ["citizen", "citizenship", "country"]),
    ("religion", r"^(?P<s>.+?) is affiliated with the religion of (?P<o>.+)$", ["religion", "religious"]),
    ("sport", r"^(?P<s>.+?) is associated with the sport of (?P<o>.+)$", ["sport", "sports", "play", "plays", "played"]),
    ("author", r"^The author of (?P<s>.+) is (?P<o>.+)$", ["author", "wrote", "written", "writer"]),
    ("speaks", r"^(?P<s>.+?) speaks the language of (?P<o>.+)$", ["language", "speak", "speaks", "spoken"]),
    ("created_by", r"^(?P<s>.+?) was created by (?P<o>.+)$", ["created", "creator", "create"]),
    ("spouse", r"^(?P<s>.+?) is married to (?P<o>.+)$", ["spouse", "married", "partner", "wife", "husband"]),
    ("born_city", r"^(?P<s>.+?) was born in the city of (?P<o>.+)$", ["born", "birthplace", "birth"]),
    ("founded_by", r"^(?P<s>.+?) was founded by (?P<o>.+)$", ["founded", "founder", "established"]),
    ("developed_by", r"^(?P<s>.+?) was developed by (?P<o>.+)$", ["developed", "developer"]),
    ("died_city", r"^(?P<s>.+?) died in the city of (?P<o>.+)$", ["died", "death", "die", "pass", "passed"]),
    ("performed_by", r"^(?P<s>.+?) was performed by (?P<o>.+)$", ["performed", "performer", "performs"]),
    ("producer", r"^The company that produced (?P<s>.+) is (?P<o>.+)$", ["produced", "producer", "company"]),
    ("hq_city", r"^The headquarters of (?P<s>.+) is located in the city of (?P<o>.+)$", ["headquarters", "headquartered"]),
    ("continent", r"^(?P<s>.+?) is located in the continent of (?P<o>.+)$", ["continent"]),
    ("broadcaster", r"^The origianl broadcaster of (?P<s>.+) is (?P<o>.+)$", ["broadcaster", "aired", "broadcast"]),
    ("created_country", r"^(?P<s>.+?) was created in the country of (?P<o>.+)$", ["created", "origin", "country", "birthplace"]),
    ("capital", r"^The capital of (?P<s>.+) is (?P<o>.+)$", ["capital"]),
    ("work_city", r"^(?P<s>.+?) worked in the city of (?P<o>.+)$", ["worked", "work", "works", "city"]),
    ("educated", r"^The univeristy where (?P<s>.+) was educated is (?P<o>.+)$", ["educated", "university", "educational", "institution", "school", "college"]),
    ("official_language", r"^The official language of (?P<s>.+) is (?P<o>.+)$", ["official", "language"]),
    ("employer", r"^(?P<s>.+?) is employed by (?P<o>.+)$", ["employed", "employer", "works", "work"]),
    ("famous_for", r"^(?P<s>.+?) is famous for (?P<o>.+)$", ["famous", "known", "notable"]),
    ("head_of_government", r"^The name of the current head of the (?P<s>.+) government is (?P<o>.+)$", ["head", "government"]),
    ("written_language", r"^(?P<s>.+?) was written in the language of (?P<o>.+)$", ["written", "language", "wrote"]),
    ("ceo", r"^The chief executive officer of (?P<s>.+) is (?P<o>.+)$", ["chief", "executive", "ceo", "officer"]),
    ("founded_city", r"^(?P<s>.+?) was founded in the city of (?P<o>.+)$", ["founded", "city"]),
    ("chairperson", r"^The chairperson of (?P<s>.+) is (?P<o>.+)$", ["chairperson", "chair", "chairman"]),
    ("head_of_state", r"^The name of the current head of state in (?P<s>.+) is (?P<o>.+)$", ["head", "state"]),
    ("music_type", r"^The type of music that (?P<s>.+) plays is (?P<o>.+)$", ["music", "genre", "type"]),
    ("head_coach", r"^The head coach of (?P<s>.+) is (?P<o>.+)$", ["coach"]),
    ("child", r"^(?P<s>.+?)'s child is (?P<o>.+)$", ["child", "son", "daughter"]),
    ("position", r"^(?P<s>.+?) plays the position of (?P<o>.+)$", ["position"]),
    ("field", r"^(?P<s>.+?) works in the field of (?P<o>.+)$", ["field", "occupation", "profession"]),
    ("director", r"^The director of (?P<s>.+) is (?P<o>.+)$", ["director", "directed"]),
    ("original_language", r"^The original language of (?P<s>.+) is (?P<o>.+)$", ["original", "language"]),
    ("office_holder", r"^The (?P<s>.+) is (?P<o>.+)$", ["who", "holds", "office"]),  # "The Prime Minister of Sweden is X"; last on purpose
]
COMPILED = [(name, re.compile(pattern), set(hints)) for name, pattern, hints in TEMPLATES]


# ── small helpers ────────────────────────────────────────────────────────────────────────────

def jsonl(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def md(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", "⏎")


def norm_entity(text: str) -> str:
    return " ".join(text.strip().strip('"').lower().split())


def pct(n, d) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def p90(values) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(0.9 * len(values)))] if values else 0.0


# ── corpus: facts, templates, chains ─────────────────────────────────────────────────────────

class Corpus:
    def __init__(self, context: str):
        self.facts = []  # (serial, text_without_period)
        for line in context.splitlines():
            m = re.match(r"^(\d+)\. (.*)$", line.strip())
            if m:
                self.facts.append((int(m.group(1)), m.group(2).rstrip(".")))
        self.parsed = {}      # serial -> (rel, s, o)
        self.family = defaultdict(list)   # (s_norm, rel) -> [(serial, o)]
        self.unparsed = []
        for serial, text in self.facts:
            hit = None
            for name, pattern, _ in COMPILED:
                m = pattern.match(text)
                if m:
                    hit = (name, m.group("s").strip(), m.group("o").strip())
                    break
            if hit is None:
                self.unparsed.append((serial, text))
                continue
            self.parsed[serial] = hit
            self.family[(norm_entity(hit[1]), hit[0])].append((serial, hit[2]))
        for key in self.family:
            self.family[key].sort()
        # forward index: subject -> [(rel, serial, o)] over CURRENT facts only, and over all facts
        self.current = {key: fam[-1] for key, fam in self.family.items()}
        self.out_current = defaultdict(list)
        self.out_all = defaultdict(list)
        for (s_norm, rel), fam in self.family.items():
            serial, o = fam[-1]
            self.out_current[s_norm].append((rel, serial, o))
            for serial, o in fam:
                self.out_all[s_norm].append((rel, serial, o))
        self.entity_names = sorted({norm_entity(s) for _, s, _ in self.parsed.values()}, key=len, reverse=True)
        self.text_of = {serial: text for serial, text in self.facts}

    def starts_in(self, question: str) -> list[str]:
        # possessives ("Walter Chrysler's child", "Pirlo’s"), quotes and punctuation around a name are
        # not part of the name; strip them from the question before matching whole entity strings
        q = norm_entity(question)
        q = re.sub(r"(?:'|’)s\b", " ", q)
        q = re.sub(r"[?\"'’,.!]", " ", q)
        q = " " + " ".join(q.split()) + " "
        found = []
        for name in self.entity_names:
            if len(name) < 3:
                continue
            probe = " " + " ".join(re.sub(r"[?\"'’,.!]", " ", name).split()) + " "
            if probe in q and not any(name in f for f in found):
                found.append(name)
        return found

    def chains(self, question: str, gold: str, max_depth: int = 4, current_only: bool = True):
        """All paths of length <= max_depth from an entity named in the question to the gold, over
        current facts (or all facts). A path is a list of (rel, serial, subject_norm, object)."""
        goal = norm_entity(gold)
        index = self.out_current if current_only else self.out_all
        paths = []
        for start in self.starts_in(question):
            frontier = [(start, [])]
            for depth in range(max_depth):
                nxt = []
                for node, path in frontier:
                    for rel, serial, o in index.get(node, []):
                        step = path + [(rel, serial, node, o)]
                        if norm_entity(o) == goal:
                            paths.append(step)
                        elif depth + 1 < max_depth:
                            nxt.append((norm_entity(o), step))
                frontier = nxt
        return paths

    def best_chain(self, question: str, gold: str, expect_hops: int | None = None):
        q_words = set(re.findall(r"\w+", question.lower()))
        for current_only in (True, False):
            paths = self.chains(question, gold, current_only=current_only)
            if not paths:
                continue

            def score(path):
                hop_ok = 0 if expect_hops is None else abs(len(path) - expect_hops)
                hints = sum(len(q_words & next(h for n, _, h in COMPILED if n == rel)) for rel, _, _, _ in path)
                # prefer the expected hop count, then more hint overlap, then shorter
                return (hop_ok, -hints, len(path))
            best = min(paths, key=score)
            return best, current_only, len(paths)
        return None, None, 0


# ── the run's artefacts ──────────────────────────────────────────────────────────────────────

class Run:
    def __init__(self, run_id: str, lengths: list[str]):
        self.run_id = run_id
        self.lengths = lengths
        import pandas as pd
        frame = pd.read_parquet(PARQUET)
        self.rows = {r["metadata"]["source"]: r for _, r in frame.iterrows()}
        self.kcalls = [r for r in jsonl(BASE / "logs" / f"kscope_calls-{SCOPE}.jsonl") if f"/vaults/mab/{run_id}/" in r.get("root", "")]
        self.ledger = [r for r in jsonl(BASE / "spend" / "ledger.jsonl") if r.get("scope") == SCOPE and f":{run_id}:" in r.get("tag", "")]
        for i, r in enumerate(self.ledger):
            r["_i"] = i
        self.failures = [r for r in jsonl(BASE / "logs" / "mab_llm_failures.jsonl") if f":{run_id}:" in r.get("tag", "")]
        self.manifests = {}
        for length in lengths:
            pointer = BASE / "runs" / "mab" / run_id / "master" / length / "manifest.txt"
            self.manifests[length] = json.loads(Path(pointer.read_text().strip()).read_text()) if pointer.exists() else None
        self.master_logs = {length: jsonl(BASE / "logs" / f"mab_adapter-{run_id}-master-{length}.jsonl") for length in lengths}
        self.cells = {}
        for length in lengths:
            for row in ROWS:
                sub = f"factconsolidation_{row}_{length}"
                for arm in ARMS:
                    files = glob.glob(str(BASE / "results" / "mab" / run_id / arm / sub / "Conflict_Resolution" / "*_results.json"))
                    exit_file = BASE / "runs" / "mab" / run_id / arm / sub / "exit_code.txt"
                    self.cells[(length, row, arm)] = {
                        "sub": sub,
                        "log": jsonl(BASE / "logs" / f"mab_adapter-{run_id}-{arm}-{sub}.jsonl"),
                        "results": json.loads(Path(files[0]).read_text()) if files else None,
                        "harness_log": BASE / "logs" / f"mab_harness-{run_id}-{arm}-{sub}.log",
                        "exit": int(exit_file.read_text().strip()) if exit_file.exists() else None,
                    }
        self.corpora = {length: Corpus(self.rows[f"factconsolidation_sh_{length}"]["context"]) for length in lengths}
        # A writer tag is `mab:<run>:c<chunk>:u<unit>` and a reader tag `mab:<run>:<arm>:q<id>`, so tags
        # REPEAT across lengths (and, for readers, across the sh/mh rows that run concurrently). Calls are
        # therefore attributed by tag AND a time window, and `unattributed` below must come out empty.
        self.ingest_window = {}
        for length in lengths:
            m, mlog = self.manifests.get(length), self.master_logs.get(length) or []
            init = next((r for r in mlog if r["event"] == "init"), None)
            if m and init:
                self.ingest_window[length] = (init["ts"], m["created_ts"])
        self.reader_window = {}
        for length in lengths:
            for arm in ARMS:
                stamps = []
                for row in ROWS:
                    for q in self.cells[(length, row, arm)]["log"]:
                        if q["event"] == "question":
                            stamps.append((q["ts"] - (q.get("reader_ms") or 0) / 1000 - 5, q["ts"] + 1))
                if stamps:
                    self.reader_window[(length, arm)] = (min(a for a, _ in stamps), max(b for _, b in stamps))

    def writer_calls(self, length: str) -> list:
        """The writer calls of ONE length's master ingestion: its unit tags inside its own ingest window."""
        tags = {u["tag"] for u in (self.master_logs.get(length) or []) if u["event"] == "unit"}
        lo, hi = self.ingest_window.get(length, (0, 0))
        return [r for r in self.ledger if r["role"] == "writer" and r["tag"] in tags and lo <= r["ts"] <= hi]

    def reader_calls(self, length: str, arm: str) -> list:
        lo, hi = self.reader_window.get((length, arm), (0, 0))
        return [r for r in self.ledger if r["role"] == "reader" and f":{self.run_id}:{arm}:q" in r["tag"] and lo <= r["ts"] <= hi]


def unit_index(master_log: list, manifest: dict):
    """serial -> (chunk, unit); (chunk, unit) -> memory id / unit row."""
    units = {(r["chunk_index"], r["unit_index"]): r for r in master_log if r["event"] == "unit"}
    serial_to_unit = {}
    for key, u in units.items():
        for m in re.finditer(r"(?:(?<=\s)|^)(\d+)\.\s", u["unit_text"]):
            serial_to_unit.setdefault(int(m.group(1)), key)
    memory_of = {}
    fallback = set()
    for chunk in manifest["chunks"]:
        for u in chunk["units"]:
            memory_of[(chunk["index"], u["index"])] = u.get("memory_id")
            if u.get("fallback"):
                fallback.add((chunk["index"], u["index"]))
    return units, serial_to_unit, memory_of, fallback


# ── the checker ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--lengths", default=",".join(LENGTHS))
    parser.add_argument("--out", default=None)
    parser.add_argument("--max_queries", type=int, default=100)
    args = parser.parse_args()
    lengths = [x for x in args.lengths.split(",") if x]
    run = Run(args.run_id, lengths)
    out: list[str] = []
    say = out.append
    red: list[str] = []

    say(f"# MemoryAgentBench full run `{args.run_id}` — generated by check.py at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── binary now ───────────────────────────────────────────────────────────────────────────
    binary_now = None
    try:
        binary_now = kscope_io.check_binary()
    except Exception as exc:  # noqa: BLE001
        red.append(f"binary check at check time: {exc}")
    say(f"kscope binary at check time: sha256 `{(binary_now or {}).get('sha256', '?')[:16]}` (pinned `{PIN or 'none'}`), "
        f"model `{(binary_now or {}).get('model', 'NOT BUNDLED')}`.\n")

    # ── template coverage per length ─────────────────────────────────────────────────────────
    for length in lengths:
        corpus = run.corpora[length]
        if corpus.unparsed:
            red.append(f"{length}: {len(corpus.unparsed)} facts matched no template, e.g. {corpus.unparsed[:3]}")

    # ── 1. scores ────────────────────────────────────────────────────────────────────────────
    say("## 1. Scores\n")
    say("`substring_exact_match` (SEM) and `exact_match` (EM) are the harness's own metrics as written to its results "
        "files, out of 100 questions per cell. `err` = reader calls that failed after every retry (answered as an empty "
        "string, scored 0, counted here and NOT as an ordinary wrong answer); `empty` = empty model outputs; `T` = visible "
        "answers truncated to 10 tokens. Every cell is the SAME reader (gpt-5.6-luna, effort high, 10-token visible cap).\n")
    say("| length | row | setup | SEM /100 | EM /100 | err | empty | T | mean served | reader in tok/q | search s/q |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    scores = {}
    from utils.eval_other_utils import normalize_answer, parse_output
    for length in lengths:
        for row in ROWS:
            sub = f"factconsolidation_{row}_{length}"
            questions = list(run.rows[sub]["questions"])
            answers = [list(a) for a in run.rows[sub]["answers"]]
            for arm in ARMS:
                cell = run.cells[(length, row, arm)]
                qrows = {r["query_id"]: r for r in cell["log"] if r["event"] == "question"}
                data = (cell["results"] or {}).get("data") or []
                n = len(data)
                if n == 0:
                    say(f"| {length} | {row} | {arm} | (no results) | | | | | | | |")
                    red.append(f"{length}/{row}/{arm}: no results file")
                    continue
                sem = sum(bool(d["substring_exact_match"]) for d in data)
                em = sum(bool(d["exact_match"]) for d in data)
                # independent recomputation of SEM with the harness's own scorer
                recomputed = 0
                aligned = True
                for i, d in enumerate(data):
                    if d.get("query_id") != i or d.get("answer") != answers[i] or d.get("query", "").find(questions[i]) < 0:
                        aligned = False
                    gold_ok = any(normalize_answer(g) in normalize_answer(d["output"]) for g in answers[i])
                    parsed = parse_output(d["output"])
                    parsed_ok = parsed is not None and any(normalize_answer(g) in normalize_answer(parsed) for g in answers[i])
                    recomputed += int(gold_ok or parsed_ok)
                errs = sum(1 for q in qrows.values() if q.get("reader_error"))
                empties = sum(1 for q in qrows.values() if not (q.get("raw_output") or "").strip())
                trunc = sum(1 for q in qrows.values() if q.get("truncated"))
                served = mean(q["served"] for q in qrows.values()) if qrows else 0
                rin = mean(q["reader_in"] for q in qrows.values()) if qrows else 0
                ssec = mean(q["search_seconds"] for q in qrows.values()) if qrows else 0
                scores[(length, row, arm)] = {"sem": sem, "em": em, "n": n, "err": errs, "empty": empties, "trunc": trunc,
                                              "served": served, "reader_in": rin, "search_s": ssec}
                flag = ""
                if recomputed != sem:
                    flag = f" (recomputed {recomputed}!)"
                    red.append(f"{length}/{row}/{arm}: results SEM {sem} != recomputed {recomputed}")
                if not aligned:
                    red.append(f"{length}/{row}/{arm}: results rows do not line up with the dataset")
                if n != args.max_queries:
                    red.append(f"{length}/{row}/{arm}: {n} questions answered, expected {args.max_queries}")
                if len(qrows) != n:
                    red.append(f"{length}/{row}/{arm}: adapter logged {len(qrows)} questions, results hold {n}")
                say(f"| {length} | {row} | {arm} | {sem}{flag} | {em} | {errs} | {empties} | {trunc} | {served:.2f} | {rin:.0f} | {ssec:.3f} |")
    say("")

    # compact matrix
    say("Compact SEM matrix (single-hop / multi-hop):\n")
    say("| length | none | bm25_units | kscope |")
    say("|---|---|---|---|")
    for length in lengths:
        cells = []
        for arm in ARMS:
            a = scores.get((length, "sh", arm), {}).get("sem", "-")
            b = scores.get((length, "mh", arm), {}).get("sem", "-")
            cells.append(f"{a} / {b}")
        say(f"| {length} | " + " | ".join(cells) + " |")
    say("")

    # ── 2. retrieval-level analysis: kscope and bm25_units ───────────────────────────────────
    say("## 2. Retrieval-level analysis (kscope and bm25_units)\n")
    say("For every question the checker resolves the chain of facts that answers it: each numbered fact is parsed with "
        "the corpus's own sentence frames into (subject, relation, object); the newest serial per (subject, relation) is "
        "the CURRENT value and older ones are STALE; the chain is the shortest path of current facts from an entity "
        "named in the question to the gold answer (single-hop = one hop). Each fact lives in exactly one stored unit "
        "(memory). For each hop's memory: **served** (rank), **cut** (a candidate the read path dropped, with kscope's "
        "own omission reason), or **never a candidate** (not among the top_k=10 the read path considered). A verdict per "
        "question: **all hops served** / **partial** / **none**, plus whether any STALE sibling of a hop was served. The "
        "same analysis runs on the bm25_units arm at the unit level (cut/never collapse to 'not in top-10').\n\n"
        "`gold ≠ newest serial` counts questions whose gold answer is reachable only through a hop that is NOT the newest "
        "serial of its (subject, relation) family -- e.g. *What is the capital of Papal States?* is scored against Rome "
        "(serial 1291) although the corpus also states `2016. The capital of Papal States is Watertown`. By the benchmark's "
        "own rule (newest serial wins) those questions cannot be answered correctly, so they bound the achievable score "
        "from above; for them the chain shown is the one the gold actually follows. `unresolved` = the checker found no "
        "path at depth <= 4 from an entity named in the question to the gold (reported, not analysed).\n")
    retrieval = {}
    for length in lengths:
        manifest = run.manifests[length]
        corpus = run.corpora[length]
        if not manifest:
            continue
        units, serial_to_unit, memory_of, fallback_units = unit_index(run.master_logs[length], manifest)
        for row in ROWS:
            sub = f"factconsolidation_{row}_{length}"
            questions = list(run.rows[sub]["questions"])
            answers = [list(a) for a in run.rows[sub]["answers"]]
            expect_hops = 1 if row == "sh" else None
            resolved = {}
            for qi in range(len(questions)):
                chain, current_only, n_paths = corpus.best_chain(questions[qi], answers[qi][0], expect_hops)
                resolved[qi] = (chain, current_only, n_paths)
            for arm in ("kscope", "bm25_units"):
                cell = run.cells[(length, row, arm)]
                qrows = {r["query_id"]: r for r in cell["log"] if r["event"] == "question"}
                data = (cell["results"] or {}).get("data") or []
                stats = {"questions": len(qrows), "unresolved": 0, "stale_path": 0, "hops": Counter(),
                         "fate": Counter(), "verdict": Counter(), "stale_served_q": 0, "cut_reasons": Counter(),
                         "sem_by_verdict": defaultdict(lambda: [0, 0]), "current_in_delta": Counter(),
                         "examples": []}
                for qi, q in qrows.items():
                    chain, current_only, _ = resolved.get(qi, (None, None, 0))
                    if chain is None:
                        stats["unresolved"] += 1
                        continue
                    if not current_only:
                        stats["stale_path"] += 1
                    stats["hops"][len(chain)] += 1
                    if arm == "kscope":
                        rank_of = {h["memory_id"]: h["rank"] for h in q["served_hits"]}
                        reason_of = {o.get("memory_id"): o.get("reason") for o in (q.get("omissions") or [])}
                    else:
                        rank_of = {f"unit:{h['unit_key'][0]}.{h['unit_key'][1]}": h["rank"] for h in q["served_hits"]}
                        reason_of = {}
                    served_hops = 0
                    stale_served = False
                    for rel, serial, s_norm, o in chain:
                        key = serial_to_unit.get(serial)
                        if key is None:
                            stats["fate"]["fact not found in any unit"] += 1
                            continue
                        mid = memory_of.get(key) if arm == "kscope" else f"unit:{key[0]}.{key[1]}"
                        if mid in rank_of:
                            stats["fate"]["served"] += 1
                            served_hops += 1
                        elif mid in reason_of:
                            stats["fate"]["cut"] += 1
                            stats["cut_reasons"][reason_of[mid]] += 1
                        else:
                            stats["fate"]["never a candidate" if arm == "kscope" else "not in top-10"] += 1
                        # is the current fact carried as a triple in the unit's delta?
                        if arm == "kscope":
                            u = units.get(key)
                            if key in fallback_units or not u:
                                stats["current_in_delta"]["no (fallback delta)"] += 1
                            else:
                                obj = norm_entity(o)
                                subj_words = set(re.findall(r"\w+", s_norm)) - {"the", "of", "a", "and"}
                                hit = any(obj in norm_entity(f"{f[0]} {f[2]}") and subj_words & set(re.findall(r"\w+", norm_entity(f"{f[0]} {f[1]} {f[2]}")))
                                          for f in (u.get("delta_facts") or []) if len(f) >= 3 and all(isinstance(x, str) for x in f))
                                stats["current_in_delta"]["yes" if hit else "no (prose only)"] += 1
                        # stale siblings of this hop
                        for st_serial, _ in corpus.family[(s_norm, rel)][:-1]:
                            st_key = serial_to_unit.get(st_serial)
                            st_mid = (memory_of.get(st_key) if arm == "kscope" else f"unit:{st_key[0]}.{st_key[1]}") if st_key else None
                            if st_mid in rank_of:
                                stale_served = True
                    verdict = "all hops served" if served_hops == len(chain) else ("partial" if served_hops else "none served")
                    stats["verdict"][verdict] += 1
                    stats["stale_served_q"] += int(stale_served)
                    if qi < len(data):
                        stats["sem_by_verdict"][verdict][0] += int(bool(data[qi]["substring_exact_match"]))
                        stats["sem_by_verdict"][verdict][1] += 1
                    if len(stats["examples"]) < 3 and row == "mh":
                        stats["examples"].append((qi, questions[qi], answers[qi][0], [(rel, serial, corpus.text_of[serial]) for rel, serial, _, _ in chain], verdict))
                retrieval[(length, row, arm)] = stats

    say("| length | row | setup | resolved q | unresolved | gold ≠ newest serial | hops (n: count) | hop memories served / cut / never | cut reasons | verdict all / partial / none | stale sibling served (q) | SEM when all served | SEM otherwise |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (length, row, arm), s in sorted(retrieval.items(), key=lambda kv: (LENGTHS.index(kv[0][0]), kv[0][1], kv[0][2])):
        f = s["fate"]
        never = f.get("never a candidate", 0) + f.get("not in top-10", 0)
        v = s["verdict"]
        sbv = s["sem_by_verdict"]
        all_served = sbv.get("all hops served", [0, 0])
        other = [sum(sbv[k][0] for k in sbv if k != "all hops served"), sum(sbv[k][1] for k in sbv if k != "all hops served")]
        say(f"| {length} | {row} | {arm} | {s['questions'] - s['unresolved']} | {s['unresolved']} | {s['stale_path']} | "
            f"{dict(sorted(s['hops'].items()))} | {f.get('served', 0)} / {f.get('cut', 0)} / {never} | {dict(s['cut_reasons']) or '-'} | "
            f"{v.get('all hops served', 0)} / {v.get('partial', 0)} / {v.get('none served', 0)} | {s['stale_served_q']} | "
            f"{all_served[0]}/{all_served[1]} | {other[0]}/{other[1]} |")
    say("")
    say("kscope: is the current fact carried as a subject/predicate/object triple in its unit's delta, or does it survive only as prose in `content_md`?\n")
    say("| length | row | in delta: yes | no (prose only) | no (fallback delta) |")
    say("|---|---|---|---|---|")
    for (length, row, arm), s in sorted(retrieval.items(), key=lambda kv: (LENGTHS.index(kv[0][0]), kv[0][1])):
        if arm == "kscope":
            c = s["current_in_delta"]
            say(f"| {length} | {row} | {c.get('yes', 0)} | {c.get('no (prose only)', 0)} | {c.get('no (fallback delta)', 0)} |")
    say("")
    say("Three resolved multi-hop chains per cell (kscope), as the checker reconstructed them:\n")
    for (length, row, arm), s in sorted(retrieval.items(), key=lambda kv: (LENGTHS.index(kv[0][0]), kv[0][1])):
        if arm == "kscope" and row == "mh":
            for qi, question, gold, chain, verdict in s["examples"]:
                say(f"- {length} q{qi}: *{md(question)}* → **{md(gold)}**; chain: " + " → ".join(f"`{serial}. {md(text)}`" for _, serial, text in chain) + f" — {verdict}")
    say("")

    # served-set shape summary for kscope
    say("Served-set shape, kscope (per cell: served count distribution, omission reasons over all questions, stop reasons, abstentions):\n")
    say("| length | row | served: min / mean / max | served==0 | omission reasons | stop_reason | abstained | context bytes mean | search s: mean / p90 |")
    say("|---|---|---|---|---|---|---|---|---|")
    for length in lengths:
        for row in ROWS:
            cell = run.cells[(length, row, "kscope")]
            qrows = [r for r in cell["log"] if r["event"] == "question"]
            if not qrows:
                continue
            served = [q["served"] for q in qrows]
            reasons = Counter(o.get("reason") for q in qrows for o in (q.get("omissions") or []))
            stops = Counter(q.get("stop_reason") for q in qrows)
            abst = sum(1 for q in qrows if q.get("abstained"))
            cb = mean(q.get("context_bytes") or 0 for q in qrows)
            ss = [q["search_seconds"] for q in qrows]
            say(f"| {length} | {row} | {min(served)} / {mean(served):.2f} / {max(served)} | {sum(1 for s in served if s == 0)} | {dict(reasons)} | {dict(stops)} | {abst} | {cb:.0f} | {mean(ss):.3f} / {p90(ss):.3f} |")
    say("")

    # ── 3. ingestion per length ──────────────────────────────────────────────────────────────
    say("## 3. Ingestion (one master vault per length)\n")
    say("| length | chunks | units | accepted | refused (1st try) | fallback resends | lost | extraction failures (writer errors) | clamps (units: fields) | propose coerced units (dropped entries) | over-budget batches | facts-lost rate | wall clock | writer call s: mean / p90 / max | writer calls (retries) | kscope `remember` ms per unit: mean / max batch | vault size |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for length in lengths:
        m = run.manifests[length]
        if not m:
            say(f"| {length} | (no master) | | | | | | | | | | | | | | | |")
            red.append(f"{length}: no sealed master")
            continue
        mlog = run.master_logs[length]
        units = [r for r in mlog if r["event"] == "unit"]
        batches = [r for r in mlog if r["event"] == "remember_batch"]
        c = m["counters"]
        clamped = [(u["chunk_index"], u["unit_index"], u["clamped"]) for u in units if u.get("clamped")]
        clamp_fields = Counter(k for _, _, cl in clamped for k in cl)
        wcalls = [r for r in run.writer_calls(length) if not r.get("failed")]
        wms = [r["ms"] / 1000 for r in wcalls]
        root = m["vault"]["root"]
        rem = [r for r in run.kcalls if r["root"] == root and r["op"] == "remember"]
        rem_ms = sum(r["ms"] for r in rem)
        size = sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file()) / 1e6
        rate = m["gate"]["facts_lost_rate"]
        say(f"| {length} | {len(m['chunks'])} | {c['units']} | {c['accepted']} | {c['refused_first_try']} | {c['fallback_resends']} | {c['lost_units']} | "
            f"{c['extraction_failures']} ({c['writer_errors']}) | {len(clamped)}: {dict(clamp_fields) or '-'} | {c['propose_coerced_units']} ({c['propose_dropped_entries']}) | "
            f"{c['minted_over_budget_batches']} of {len(batches)} | {rate:.2%} | {m['timing']['ingest_wall_seconds'] / 60:.1f} min | "
            f"{mean(wms):.0f} / {p90(wms):.0f} / {max(wms) if wms else 0:.0f} | {len(wcalls)} ({len(wcalls) - len(units)}) | "
            f"{rem_ms / max(1, c['units']):.0f} / {max((r['ms'] for r in rem), default=0)} | {size:.0f} MB |")
        if c["lost_units"] or rate > 0.02:
            red.append(f"{length}: facts-lost rate {rate:.2%} or lost units {c['lost_units']}")
    say("")
    say("Clamps are writer.py's own list caps (32 facts, 32 entities, 8 propose): a clamp on `propose` has no reader-visible "
        "effect (the fold is byte-identical downstream, HANDOFF §3); a clamp on `facts` or `entities` would drop triples and is "
        "listed per unit below if any occurred. A refusal costs the unit every triple it had (it is stored with the one-fact "
        "fallback delta); the facts-lost rate counts refusals plus extraction failures over units, threshold 2%.\n")
    for length in lengths:
        mlog = run.master_logs[length]
        for u in (r for r in mlog if r["event"] == "unit" and r.get("clamped")):
            say(f"- {length} chunk {u['chunk_index']} unit {u['unit_index']}: clamped {u['clamped']} (facts {u['facts']}, entities {u['entities']}, proposals {u['proposals']})")
        for r in (r for r in mlog if r["event"] == "fallback_resend"):
            say(f"- {length} chunk {r['chunk_index']} unit {r['unit_index']}: REFUSED first try: `{md(r['first_refusal'])[:200]}` → fallback stored: {r['stored']}")
        for u in (r for r in mlog if r["event"] == "unit" and r.get("extraction_failed")):
            say(f"- {length} chunk {u['chunk_index']} unit {u['unit_index']}: extraction failed (writer error: {md(u.get('writer_error'))}), stored with the fallback delta")
    say("")
    # per-length write-time trend inside the largest vault
    for length in lengths:
        m = run.manifests[length]
        if not m:
            continue
        root = m["vault"]["root"]
        rem = sorted((r for r in run.kcalls if r["root"] == root and r["op"] == "remember"), key=lambda r: r["ts"])
        if len(rem) >= 8:
            k = len(rem) // 4
            quart = [mean(r["ms"] for r in rem[i * k:(i + 1) * k]) for i in range(4)]
            say(f"- {length}: `remember` ms per batch (of up to 20 units) by quarter of the ingestion, in order: "
                + " → ".join(f"{q:.0f}" for q in quart) + f" (first batch {rem[0]['ms']} ms, last batch {rem[-1]['ms']} ms, {len(rem)} batches)")
    say("")

    # ── 4. assertions ────────────────────────────────────────────────────────────────────────
    say("## 4. Assertions (each can go red)\n")
    import tiktoken
    from methods.kscope_agent import split_units, tree_sha
    from utils.eval_other_utils import chunk_text_into_sentences
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    say("| length | setup | A binary pinned + bundled | B accepted == sent | C refusals / clamps | D every question answered | E served recorded | F nothing outside vaults/ | G failed calls in ledger | H no exceptions, exit 0 | verdict |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    failed_rows_all = [r for r in run.ledger if r.get("failed")]
    for length in lengths:
        m = run.manifests[length]
        if not m:
            continue
        corpus_context = run.rows[f"factconsolidation_sh_{length}"]["context"]
        chunks = chunk_text_into_sentences(corpus_context, chunk_size=4096)
        chunk_shas = [sha16(c) for c in chunks]
        manifest_chunk_shas = [c["sha"] for c in m["chunks"]]
        rechunk_ok = chunk_shas == manifest_chunk_shas
        recut_ok = True
        for ci, chunk in enumerate(chunks):
            texts, _ = split_units(chunk, tokenizer, m["unit_rule"]["unit_tokens"], m["unit_rule"]["unit_sentences"])
            if [sha16(t) for t in texts] != [u["sha"] for u in m["chunks"][ci]["units"]]:
                recut_ok = False
        n_units = sum(len(c["units"]) for c in m["chunks"])
        master_root = m["vault"]["root"]
        master_calls = [r for r in run.kcalls if r["root"] == master_root]
        accepted_by_kscope = sum((r.get("accepted") or 0) for r in master_calls if r["op"] == "remember")
        seal_now = tree_sha(Path(master_root))
        master_searched = sum(1 for r in master_calls if r["op"] == "search")
        mlog = run.master_logs[length]
        units_logged = [r for r in mlog if r["event"] == "unit"]
        # master row
        a_ok = m["kscope"]["sha256"].startswith(PIN) and m["kscope"]["model"] == "bundled" and binary_now and binary_now["sha256"] == m["kscope"]["sha256"]
        b_ok = rechunk_ok and recut_ok and n_units == len(units_logged) == m["counters"]["units"] == accepted_by_kscope + m["counters"]["lost_units"] == m["counters"]["accepted"] + m["counters"]["lost_units"] and m["counters"]["lost_units"] == 0
        c_txt = f"refused {m['counters']['refused_first_try']}, clamped {m['counters']['clamped_units']}, lost {m['counters']['lost_units']}, rate {m['gate']['facts_lost_rate']:.2%}"
        c_ok = m["gate"]["pass"]
        f_ok = seal_now == m["seal"] and master_searched == 0 and master_root.startswith(str(BASE / "vaults" / "mab" / args.run_id) + "/")
        writer_failed = [r for r in run.writer_calls(length) if r.get("failed")]
        writer_call_failures = sum(u.get("call_failures", 0) for u in units_logged)
        g_ok = len(writer_failed) == writer_call_failures
        ingest_log = BASE / "logs" / f"mab_ingest-{args.run_id}-{length}.log"
        h_ok = ingest_log.exists() and "Traceback" not in ingest_log.read_text(errors="replace")
        verdict = all([a_ok, b_ok, c_ok, f_ok, g_ok, h_ok])
        if not verdict:
            red.append(f"{length}/master: " + ", ".join(n for n, ok in (("A", a_ok), ("B", b_ok), ("C", c_ok), ("F", f_ok), ("G", g_ok), ("H", h_ok)) if not ok))
        say(f"| {length} | master ingest | {'ok' if a_ok else 'RED'} `{m['kscope']['sha256'][:16]}` | {'ok' if b_ok else 'RED'}: rechunk {rechunk_ok}, recut {recut_ok}, units {n_units} = logged {len(units_logged)} = accepted-by-kscope {accepted_by_kscope} | "
            f"{c_txt} {'ok' if c_ok else 'RED'} | n/a | n/a | {'ok' if f_ok else 'RED'}: seal {'intact' if seal_now == m['seal'] else 'MOVED'}, searches on master {master_searched} | "
            f"{'ok' if g_ok else 'RED'}: {len(writer_failed)} failed rows = {writer_call_failures} counted | {'ok' if h_ok else 'RED'} | **{'GREEN' if verdict else 'RED'}** |")
        for row in ROWS:
            for arm in ARMS:
                cell = run.cells[(length, row, arm)]
                log = cell["log"]
                init = next((r for r in log if r["event"] == "init"), None)
                fq = next((r for r in log if r["event"] == "first_question"), None)
                qrows = [r for r in log if r["event"] == "question"]
                data = (cell["results"] or {}).get("data") or []
                if not init or not fq:
                    say(f"| {length} | {row}/{arm} | (no adapter log) | | | | | | | | **RED** |")
                    red.append(f"{length}/{row}/{arm}: no adapter log")
                    continue
                cnt = fq["counters"]
                if arm == "bm25_units":
                    a_txt, a_ok = "n/a (no kscope)", True
                else:
                    a_ok = bool(init.get("kscope_sha256", "").startswith(PIN)) and bool(binary_now and binary_now["sha256"] == init.get("kscope_sha256"))
                    a_txt = f"{'ok' if a_ok else 'RED'} `{(init.get('kscope_sha256') or '?')[:16]}`"
                if arm == "none":
                    b_ok = cnt["chunks_in"] == len(m["chunks"])
                    b_txt = f"{'ok' if b_ok else 'RED'}: {cnt['chunks_in']} chunks seen, nothing stored"
                else:
                    b_ok = (cnt["chunks_verified_against_master"] == len(m["chunks"]) and cnt["units_verified_against_master"] == n_units
                            and init.get("master_seal") == m["seal"] and (arm != "kscope" or cnt["accepted"] == n_units))
                    b_txt = f"{'ok' if b_ok else 'RED'}: {cnt['chunks_verified_against_master']}/{len(m['chunks'])} chunks, {cnt['units_verified_against_master']}/{n_units} units verified vs master seal"
                c_txt = "n/a (no writes)"
                errs = sum(1 for q in qrows if q.get("reader_error"))
                d_ok = len(data) == args.max_queries and len(qrows) == len(data) and errs == 0 and all(d.get("query_id") == i for i, d in enumerate(data))
                d_txt = f"{'ok' if d_ok else 'RED'}: {len(data)} answered, {errs} reader errors, {sum(1 for q in qrows if not (q.get('raw_output') or '').strip())} empty"
                e_ok = all(isinstance(q.get("served"), int) for q in qrows) and (arm != "kscope" or all(q.get("search_rc") == 0 for q in qrows))
                e_txt = f"{'ok' if e_ok else 'RED'}: served recorded on {sum(1 for q in qrows if isinstance(q.get('served'), int))}/{len(qrows)}"
                vault = init.get("vault")
                if arm == "bm25_units":
                    f_ok, f_txt = True, "n/a (no vault)"
                else:
                    calls = [r for r in run.kcalls if r["root"] == vault]
                    ops = Counter(r["op"] for r in calls)
                    inside = bool(vault) and vault.startswith(str(BASE / "vaults" / "mab" / args.run_id) + "/")
                    f_ok = inside and ops.get("remember", 0) == 0 and ops.get("search", 0) == len(qrows) and (arm != "none" or all(q["served"] == 0 for q in qrows))
                    f_txt = f"{'ok' if f_ok else 'RED'}: {dict(ops)} on its own root, under vaults/mab/{args.run_id}: {inside}"
                # reader tags carry no row name, so failures are attributed by the adapter's own count per cell
                counted = sum(q.get("reader_call_failures", 0) for q in qrows)
                g_ok = True  # verified in aggregate below (tags do not name the row)
                g_txt = f"{counted} failed calls counted by the adapter"
                hl = cell["harness_log"].read_text(errors="replace") if cell["harness_log"].exists() else ""
                tb = hl.count("Traceback (most recent call last)")
                h_ok = tb == 0 and cell["exit"] == 0
                h_txt = f"{'ok' if h_ok else 'RED'}: {tb} tracebacks, exit {cell['exit']}"
                verdict = all([a_ok, b_ok, d_ok, e_ok, f_ok, g_ok, h_ok])
                if not verdict:
                    red.append(f"{length}/{row}/{arm}: " + ", ".join(n for n, ok in (("A", a_ok), ("B", b_ok), ("D", d_ok), ("E", e_ok), ("F", f_ok), ("H", h_ok)) if not ok))
                say(f"| {length} | {row}/{arm} | {a_txt} | {b_txt} | {c_txt} | {d_txt} | {e_txt} | {f_txt} | {g_txt} | {h_txt} | **{'GREEN' if verdict else 'RED'}** |")
    say("")
    # G in aggregate: failed rows in the ledger vs failures counted by adapters
    counted_total = 0
    for length in lengths:
        counted_total += sum(u.get("call_failures", 0) for u in run.master_logs[length] if u["event"] == "unit")
        for row in ROWS:
            for arm in ARMS:
                counted_total += sum(q.get("reader_call_failures", 0) for q in run.cells[(length, row, arm)]["log"] if q["event"] == "question")
    g_all = len(failed_rows_all) == counted_total == len(run.failures)
    say(f"G in aggregate: failed-call rows in the spend ledger for this run: {len(failed_rows_all)}; rows in logs/mab_llm_failures.jsonl: {len(run.failures)}; "
        f"failures counted by the adapters (writer + reader): {counted_total} → {'ok' if g_all else 'RED'}. "
        + ("; ".join(f"{r['role']} {r['tag']} {r['finish']} attempt {r.get('attempt')}" for r in failed_rows_all[:20]) if failed_rows_all else "no LLM call failed."))
    if not g_all:
        red.append("G: failed-call accounting disagrees")
    # global isolation: every root this run touched is under vaults/mab/<run_id>/, none is the personal vault
    roots = sorted({r["root"] for r in run.kcalls})
    outside = [r for r in roots if not r.startswith(str(BASE / "vaults" / "mab" / args.run_id) + "/")]
    personal = [r for r in jsonl(BASE / "logs" / f"kscope_calls-{SCOPE}.jsonl") if str(kscope_io.PERSONAL_VAULT) in json.dumps(r)]
    agents_dir = (HARNESS / "agents").exists()
    outputs_dir = (HARNESS / "outputs").exists()
    iso_ok = not outside and not personal and not agents_dir and not outputs_dir
    say(f"\nIsolation in aggregate: {len(roots)} roots touched under this run id ({Counter(r.split('/')[-2] for r in roots)}), "
        f"{len(outside)} outside vaults/mab/{args.run_id}/; rows naming the personal vault in the whole mab call log: {len(personal)}; "
        f"harness agents/ dir exists: {agents_dir}; outputs/ dir exists: {outputs_dir} → {'ok' if iso_ok else 'RED'}")
    if not iso_ok:
        red.append("isolation in aggregate")
    # leak structure: every writer call of a length finished before the first question of any run on that length
    for length in lengths:
        units_logged = [r for r in run.master_logs[length] if r["event"] == "unit"]
        wcalls = run.writer_calls(length)
        last_writer = max((r["ts"] for r in wcalls), default=0)
        first_q = min((r["ts"] for row in ROWS for arm in ARMS for r in run.cells[(length, row, arm)]["log"] if r["event"] == "first_question"), default=float("inf"))
        shas = {u["prompt_sha"] for u in units_logged}
        ok = last_writer < first_q and len(shas) == 1
        say(f"- {length}: last writer call finished at {last_writer:.0f}, first question of any arm at {first_q:.0f} ({(first_q - last_writer) / 60:.1f} min later); writer prompt sha {sorted(shas)} → {'ok' if ok else 'RED'}")
        if not ok:
            red.append(f"{length}: writer/question order or prompt sha")
        # coercion idempotency on every unit: the writer's second pass must change nothing
        changed = 0
        for u in units_logged:
            prop = u.get("delta_propose")
            if prop:
                again, stats = writer.coerce_propose({"propose": prop})
                if again.get("propose") != prop or any(stats.values()):
                    changed += 1
        say(f"- {length}: units whose logged (adapter-coerced) `propose` block the writer's second coercion pass would change: {changed} of {sum(1 for u in units_logged if u.get('delta_propose'))} with proposals → {'ok' if changed == 0 else 'RED'}")
        if changed:
            red.append(f"{length}: coercion not idempotent on {changed} units")
    say("")

    # ── 5. spend ─────────────────────────────────────────────────────────────────────────────
    say("## 5. Tokens and cost\n")
    say(f"The gateway returns no cost. Both figures are tokens times an assumed rate: **list** = Azure Global Standard list price "
        f"${LIST_PRICE[0]}/${LIST_PRICE[1]} per Mtok (in/out); **pessimistic** = the worst reported Azure billing ${PESS_PRICE[0]}/${PESS_PRICE[1]}, "
        f"which is what the ${llm.CAP_USD:.0f} cap is checked against. Reasoning tokens are inside output tokens.\n")
    say("| length | role / arm | calls | failed | input tokens | output tokens (reasoning) | list $ | pessimistic $ |")
    say("|---|---|---|---|---|---|---|---|")
    tot = Counter()
    attributed = set()
    for length in lengths:
        groups = {"writer (master ingest)": run.writer_calls(length)}
        for arm in ARMS:
            groups[f"reader / {arm}"] = run.reader_calls(length, arm)
        for name, calls in groups.items():
            attributed.update(r["_i"] for r in calls)
            ok_calls = [r for r in calls if not r.get("failed")]
            i = sum(r["in"] for r in ok_calls)
            o = sum(r["out"] for r in ok_calls)
            rs = sum(r.get("reasoning") or 0 for r in ok_calls)
            lst = i * LIST_PRICE[0] / 1e6 + o * LIST_PRICE[1] / 1e6
            pes = i * PESS_PRICE[0] / 1e6 + o * PESS_PRICE[1] / 1e6
            tot["calls"] += len(ok_calls)
            tot["failed"] += len(calls) - len(ok_calls)
            tot["in"] += i
            tot["out"] += o
            tot["reasoning"] += rs
            say(f"| {length} | {name} | {len(ok_calls)} | {len(calls) - len(ok_calls)} | {i:,} | {o:,} ({rs:,}) | ${lst:.3f} | ${pes:.3f} |")
    lst = tot["in"] * LIST_PRICE[0] / 1e6 + tot["out"] * LIST_PRICE[1] / 1e6
    pes = tot["in"] * PESS_PRICE[0] / 1e6 + tot["out"] * PESS_PRICE[1] / 1e6
    say(f"| **all** | | {tot['calls']} | {tot['failed']} | {tot['in']:,} | {tot['out']:,} ({tot['reasoning']:,}) | **${lst:.3f}** | **${pes:.3f}** |")
    # A row outside every checked length's window belongs to a length this invocation was not asked
    # about (e.g. an ingestion still running), so only rows INSIDE a checked window may be left over.
    spans = [w for length in lengths for w in (run.ingest_window.get(length),) if w]
    spans += [w for length in lengths for arm in ARMS for w in (run.reader_window.get((length, arm)),) if w]
    def inside(row):
        return any(lo <= row["ts"] <= hi for lo, hi in spans)
    leftover = [r for r in run.ledger if r["_i"] not in attributed and inside(r)]
    outside = sum(1 for r in run.ledger if r["_i"] not in attributed and not inside(r))
    questions_logged = sum(1 for length in lengths for row in ROWS for arm in ARMS
                           for q in run.cells[(length, row, arm)]["log"] if q["event"] == "question")
    reader_errors = sum(1 for length in lengths for row in ROWS for arm in ARMS
                        for q in run.cells[(length, row, arm)]["log"] if q["event"] == "question" and q.get("reader_error"))
    reader_attributed = sum(1 for r in run.ledger if r["_i"] in attributed and r["role"] == "reader" and not r.get("failed"))
    attr_ok = not leftover and reader_attributed == questions_logged - reader_errors
    say(f"\nAttribution check (tags repeat across lengths and rows, so calls are matched by tag AND time window): "
        f"ledger rows for this run {len(run.ledger)}, attributed {len(attributed)}, left over inside a checked window {len(leftover)}, "
        f"outside every checked window {outside} (a length not checked here, or one still running); successful reader calls attributed "
        f"{reader_attributed} vs questions logged {questions_logged} minus {reader_errors} whose reader call never succeeded → {'ok' if attr_ok else 'RED'}"
        + ("" if not leftover else " — " + "; ".join(f"{r['role']} {r['tag']}" for r in leftover[:10])))
    if not attr_ok:
        red.append("spend attribution incomplete")
    served_models = Counter(r.get("model_served") for r in run.ledger if not r.get("failed"))
    finish = Counter(r.get("finish") for r in run.ledger)
    say(f"\nModel served: {dict(served_models)}; finish reasons: {dict(finish)}; temperature sent on any call: {any(r.get('temperature_sent') for r in run.ledger)}.")
    say(f"\n`llm.report()` at check time: `{llm.report()}`\n")

    say("## Checker verdict\n")
    say("RED: " + "; ".join(red) if red else "All checked assertions GREEN.")
    text = "\n".join(out)
    out_path = Path(args.out) if args.out else BASE / "reports" / f"mab_full_{args.run_id}_generated.md"
    out_path.write_text(text)
    print(text)
    print(f"\n[written to {out_path}]")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
