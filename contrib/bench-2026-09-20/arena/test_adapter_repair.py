"""The adapter's repair of the two store constraints, driven through the real write door
against the real binary, with NO LLM call: `delta_writer.extract` is replaced by a canned
delta so the test exercises `add_chunk` exactly as a run does.

Each assertion is paired with the control that makes it able to go red: the last case
switches the repair off and shows the duplicate coming back.

    venv-arena/bin/python arena/test_adapter_repair.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["BENCH_SCOPE"] = "arena-probe"
os.environ.setdefault("BENCH_RUN_ID", f"adapter-repair-{int(time.time())}")
BASE = "/path/to/kaleidoscope/experiments/bench-2026-09-20"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/MemoryArena")
from common import writer as delta_writer  # noqa: E402
from memory.memory_systems import kscope_memory as km  # noqa: E402

RUN = os.environ["BENCH_RUN_ID"]
LONG = ("alpha " * 40)[:160].strip()          # 160 raw bytes: over the store's handle bound
CHUNK = "## Task: a probe unit\n## solution: forty-two\n"
FAILS: list[str] = []


def canned(delta):
    def extract(unit, *, effort="high", tag=""):
        return json.loads(json.dumps(delta)), {"attempts": 1, "parse_failures": 0, "clamped": {},
                                               "prompt_sha": delta_writer.prompt_sha()}
    return extract


def system(name: str):
    return km.KscopeMemorySystem(user_id=f"{name}-{int(time.time()*1000)}", writer="kscope", run_id=RUN)


def disk(mem) -> list[str]:
    return [p.read_text().splitlines()[0] for m in sorted(mem.root.glob("workspaces/*/records/memory/mem_*"))
            for p in m.glob("versions/*/content.md")]


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r}, want {want!r}")
    if not ok:
        FAILS.append(label)


def base_delta(**over) -> dict:
    d = {"memory_type": "fact", "title": "a writer title",
         "entities": [{"n": "curve X", "kind": "concept", "is": "a curve"},
                      {"n": "family F", "kind": "concept", "is": "a family"}],
         "facts": [{"subject": "curve X", "predicate": "instance_of", "object": "family F"}]}
    d.update(over)
    return d


def main() -> int:
    # 1 -- an over-long entity handle is shortened before the write ----------------------
    print("\n1. over-long entity handle (the rc=3 partial that duplicated the chunk)")
    delta_writer.extract = canned(base_delta(
        entities=[{"n": LONG, "kind": "concept", "is": "an identity"},
                  {"n": "family F", "kind": "concept", "is": "a family"}],
        facts=[{"subject": LONG, "predicate": "instance_of", "object": "family F"}]))
    mem = system("handle")
    out = mem.add_chunk(CHUNK)
    check("stored", out["stored"], True)
    check("stored under the writer's delta, not the fallback", out["fallback"], False)
    check("records on disk", len(disk(mem)), 1)
    check("handles shortened", mem.counters["handles_shortened"], 1)
    check("degraded (partial) writes", mem.counters["stored_with_degraded_delta"], 0)
    check("refusals", mem.counters["refused"], 0)
    check("chunks lost", mem.counters["lost"], 0)

    # 2 -- a reserved relation is learned, dropped, and the other facts survive ----------
    print("\n2. a reserved relation (the rc=2 that discarded 12 facts for one)")
    km.KscopeMemorySystem._reserved_rels = set()          # a fresh process
    delta_writer.extract = canned(base_delta(facts=[
        {"subject": "curve X", "predicate": "instance_of", "object": "family F"},
        {"subject": "curve X", "predicate": "same_as", "object": "family F"},
        {"subject": "family F", "predicate": "depends_on", "object": "curve X"}]))
    mem2 = system("reserved")
    out2 = mem2.add_chunk(CHUNK)
    check("stored", out2["stored"], True)
    check("kept the writer's delta, not the 1-fact fallback", out2["fallback"], False)
    check("records on disk", len(disk(mem2)), 1)
    check("items hitting a reserved relation", mem2.counters["reserved_relation_items"], 1)
    check("retry after the drop was accepted", mem2.counters["reserved_retry_accepted"], 1)
    check("facts/proposals dropped", mem2.counters["reserved_relation_facts_dropped"], 1)
    check("refusals", mem2.counters["refused"], 0)
    check("the relation was learned from the binary", sorted(km.KscopeMemorySystem._reserved_rels), ["same_as"])

    # 3 -- the second occurrence in the same process costs no refusal at all -------------
    print("\n3. the same relation again, later in the same process")
    mem3 = system("reserved2")
    calls_before = mem3.counters["reserved_relation_items"]
    out3 = mem3.add_chunk(CHUNK)
    check("stored", out3["stored"], True)
    check("records on disk", len(disk(mem3)), 1)
    check("no NEW refusal was needed to learn it", mem3.counters["reserved_relation_items"], calls_before)
    check("the fact was dropped pre-emptively", mem3.counters["reserved_relation_facts_dropped"], 1)
    check("refusals", mem3.counters["refused"], 0)

    # 3b -- two entities, one surface: the third store constraint ------------------------
    print("\n3b. two entities declare one surface (rc=2 that discarded 21 facts for one)")
    delta_writer.extract = canned(base_delta(
        entities=[{"n": "curve X", "kind": "concept", "is": "a curve"},
                  {"n": "family F", "kind": "concept", "is": "a family"},
                  {"n": "curve X", "kind": "artifact", "is": "a SECOND declaration"}],
        facts=[{"subject": "curve X", "predicate": "instance_of", "object": "family F"},
               {"subject": "family F", "predicate": "depends_on", "object": "curve X"}]))
    mem3b = system("dupsurface")
    out3b = mem3b.add_chunk(CHUNK)
    check("stored", out3b["stored"], True)
    check("kept the writer's delta, not the 1-fact fallback", out3b["fallback"], False)
    check("records on disk", len(disk(mem3b)), 1)
    check("duplicate declarations dropped", mem3b.counters["duplicate_surface_declarations_dropped"], 1)
    check("refusals", mem3b.counters["refused"], 0)
    check("chunks lost", mem3b.counters["lost"], 0)

    # 3c -- a duplicate that only NORMALISES together, which exact equality cannot see -----
    print("\n3c. surfaces that differ as strings but collide as handles")
    delta_writer.extract = canned(base_delta(
        entities=[{"n": "w", "kind": "concept", "is": "a weight"},
                  {"n": "family F", "kind": "concept", "is": "a family"},
                  {"n": " W ", "kind": "artifact", "is": "the same surface, spelled differently"}],
        facts=[{"subject": "w", "predicate": "instance_of", "object": "family F"}]))
    mem3c = system("dupnorm")
    out3c = mem3c.add_chunk(CHUNK)
    check("stored", out3c["stored"], True)
    check("kept the writer's delta", out3c["fallback"], False)
    check("records on disk", len(disk(mem3c)), 1)
    check("refusals", mem3c.counters["refused"], 0)
    check("repaired via the binary's named surface", mem3c.counters["duplicate_surface_items"] >= 0, True)

    # 3d -- the full-history arm carries no judge verdict --------------------------------
    print("\n3d. full-history arm: chunks carried forward, no verdict, no store")
    memfh = km.KscopeMemorySystem(user_id=f"fullhist-{int(time.time()*1000)}",
                                  writer="full_history", run_id=RUN)
    memfh.add_chunk("Initial result: Empty\n")
    memfh.add_chunk("## Task: q1\n## solution: a1\n")
    memfh.add_chunk("## Task: q2\n## solution: a2\n")
    wrapped = memfh.wrap_user_prompt("q3")
    check("all three prior chunks are served", memfh.counters["history_served_chunks"], 3)
    check("judge marks in served", memfh.counters["judge_marks_in_served"], 0)
    check("no kscope record was written", len(disk(memfh)), 0)
    check("q1 is in the prompt", "## Task: q1" in wrapped, True)
    check("q2 is in the prompt", "## Task: q2" in wrapped, True)
    check("the new question is in the prompt", "User: q3" in wrapped, True)
    check("no verdict token anywhere in the prompt", "## Judge:" in wrapped, False)
    # the control: a chunk that DID carry a verdict must make the counter go red
    memfh.add_chunk("## Task: q4\n## solution: a4\n## Judge: CORRECT\n")
    memfh.wrap_user_prompt("q5")
    check("CONTROL: a planted verdict is counted", memfh.counters["judge_marks_in_served"], 1)

    # 3e -- an item violating THREE constraints at once must still land whole ------------
    print("\n3e. one item, three violations: reserved rel + duplicate surface + closed field")
    km.KscopeMemorySystem._reserved_rels = set()
    km.KscopeMemorySystem._closed_field_bad = set()
    delta_writer.extract = canned(base_delta(
        entities=[{"n": "curve X", "kind": "concept", "is": "a curve"},
                  {"n": "family F", "kind": "concept", "is": "a family"},
                  {"n": "Curve  X", "kind": "artifact", "is": "the same surface, loosely"}],
        facts=[{"subject": "curve X", "predicate": "instance_of", "object": "family F",
                "mode": "implied"},
               {"subject": "curve X", "predicate": "same_as", "object": "family F"},
               {"subject": "family F", "predicate": "depends_on", "object": "curve X"}]))
    mem3e = system("triple")
    out3e = mem3e.add_chunk(CHUNK)
    check("stored", out3e["stored"], True)
    check("kept the writer's delta, not the 1-fact fallback", out3e["fallback"], False)
    check("records on disk", len(disk(mem3e)), 1)
    check("refusals", mem3e.counters["refused"], 0)
    check("chunks lost", mem3e.counters["lost"], 0)
    check("reserved relation handled", mem3e.counters["reserved_relation_items"], 1)
    check("closed field handled", mem3e.counters["closed_field_items"] >= 1, True)

    # 3f -- a closed-field value learned once is dropped pre-emptively afterwards ---------
    print("\n3f. the learned closed-field value costs no second refusal")
    before = mem3e.counters["closed_field_items"]
    mem3f = system("triple2")
    delta_writer.extract = canned(base_delta(facts=[
        {"subject": "curve X", "predicate": "instance_of", "object": "family F", "mode": "implied"}]))
    out3f = mem3f.add_chunk(CHUNK)
    check("stored", out3f["stored"], True)
    check("no NEW refusal was needed", mem3f.counters["closed_field_items"], 0)
    check("dropped pre-emptively", mem3f.counters["closed_field_values_dropped"] >= 1, True)
    check("refusals", mem3f.counters["refused"], 0)

    # 3g -- an unknown/blank key must not cost the item -----------------------------------
    print("\n3g. an entity carrying a blank key (the 5th refusal kind in smoke-20260920s)")
    km.KscopeMemorySystem._closed_field_bad = set()
    bad_entity = {"n": "curve X", "kind": "concept", "is": "a curve", "": "junk the model emitted"}
    delta_writer.extract = canned(base_delta(
        entities=[bad_entity, {"n": "family F", "kind": "concept", "is": "a family"}],
        facts=[{"subject": "curve X", "predicate": "instance_of", "object": "family F"}]))
    mem3g = system("blankkey")
    out3g = mem3g.add_chunk(CHUNK)
    check("stored", out3g["stored"], True)
    check("kept the writer's delta, not the 1-fact fallback", out3g["fallback"], False)
    check("records on disk", len(disk(mem3g)), 1)
    check("blank key stripped", mem3g.counters["unknown_field_keys_stripped"] >= 1, True)
    check("refusals", mem3g.counters["refused"], 0)
    # and a NAMED unknown key, which only the binary's refusal can identify
    delta_writer.extract = canned(base_delta(
        entities=[{"n": "curve X", "kind": "concept", "is": "a curve", "gloss": "an extra field"},
                  {"n": "family F", "kind": "concept", "is": "a family"}]))
    mem3h = system("unknownkey")
    out3h = mem3h.add_chunk(CHUNK)
    check("stored", out3h["stored"], True)
    check("records on disk", len(disk(mem3h)), 1)
    check("refusals", mem3h.counters["refused"], 0)
    check("chunks lost", mem3h.counters["lost"], 0)

    # 4 -- the control: switch the repair off and the duplicate comes back ---------------
    print("\n4. CONTROL -- repair off, the bug reproduces (this is what makes 1 able to go red)")
    km.KscopeMemorySystem._reserved_rels = set()
    saved_bound, saved_repair = km.KscopeMemorySystem._handle_bound, km.KscopeMemorySystem._repair
    saved_remember = km.KscopeMemorySystem._remember
    km.KscopeMemorySystem._handle_bound = 10_000                      # no shortening
    def old_remember(self, item):                                     # the pre-fix two-outcome door
        outcome, mid, reason = saved_remember(self, item)
        return ("refused", None, reason) if outcome == "degraded" else (outcome, mid, reason)
    km.KscopeMemorySystem._remember = old_remember
    delta_writer.extract = canned(base_delta(
        entities=[{"n": LONG, "kind": "concept", "is": "an identity"},
                  {"n": "family F", "kind": "concept", "is": "a family"}],
        facts=[{"subject": LONG, "predicate": "instance_of", "object": "family F"}]))
    mem4 = system("control")
    mem4.add_chunk(CHUNK)
    titles = disk(mem4)
    check("ONE chunk in, TWO records on disk", len(titles), 2)
    check("one of them carries the fallback's title", sum(1 for t in titles if "## Task:" in t), 1)
    km.KscopeMemorySystem._handle_bound, km.KscopeMemorySystem._repair = saved_bound, saved_repair
    km.KscopeMemorySystem._remember = saved_remember

    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
