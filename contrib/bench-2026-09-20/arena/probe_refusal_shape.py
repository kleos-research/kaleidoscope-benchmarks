"""Zero-LLM-cost falsifier for the four refusals the interrupted arena smoke recorded.

The adapter logged these kscope refusals of the writer's delta:
  rc=2 invalid type: string "element", expected a sequence
  rc=2 invalid type: string "concept", expected a sequence
  rc=2 invalid type: string "false",   expected a boolean
  rc=2 two entities declare the surface "f"

Hypothesis: the first three are `propose[]` shape errors -- `from`/`to` are sequences of kinds
and `many` is a boolean, and the writer prompt's inline example shows `"propose": []` so the
model invents the shape from the contract's prose, which never says "array". The fourth is the
one-surface-one-entity rule.

This probe sends each shape directly to the binary. It makes NO LLM call. It runs under its own
BENCH_SCOPE so it cannot touch the arena run's call log or its roots_touched proof.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["BENCH_SCOPE"] = "arena-probe"
BASE = "/path/to/kaleidoscope/experiments/bench-2026-09-20"
sys.path.insert(0, BASE)
from common import kscope_io  # noqa: E402

BASE_FACT = {"subject": "curve X", "predicate": "part_of", "object": "family F"}
ENTS = [{"n": "curve X", "kind": "concept", "is": "a regular proper curve"},
        {"n": "family F", "kind": "concept", "is": "a family of curves"}]
PROPOSE_OK = {"rel": "fibers_over", "means": "the subject admits a morphism onto the object",
              "many": False, "over_time": "state", "from": ["concept"], "to": ["concept"]}


def case(name: str, delta: dict) -> dict:
    root = kscope_io.VAULTS / "arena-probe" / f"shape-{int(time.time()*1000)}"
    v = kscope_io.Vault.create(root)
    rc, body, err = v.remember_items([{"content_md": "A probe unit.", "semantic_delta": delta}])
    body = body or {}
    results = body.get("results") or []
    reason = (results[0] if results else {}).get("reason") or body.get("message") or (err or "").strip()
    return {"case": name, "rc": rc, "accepted": body.get("accepted_items"),
            "status": body.get("status"), "reason": str(reason)[:220]}


def main() -> None:
    def delta(**over):
        d = {"memory_type": "fact", "title": "probe", "facts": [dict(BASE_FACT)], "entities": [dict(e) for e in ENTS]}
        d.update(over)
        return d

    cases = [
        ("control: no propose", delta()),
        ("propose well formed (from/to lists, many bool)", delta(propose=[dict(PROPOSE_OK)])),
        ("propose.from as a STRING", delta(propose=[{**PROPOSE_OK, "from": "concept"}])),
        ("propose.to as a STRING", delta(propose=[{**PROPOSE_OK, "to": "element"}])),
        ("propose.many as the STRING 'false'", delta(propose=[{**PROPOSE_OK, "many": "false"}])),
        ("two entities declaring one surface", delta(entities=ENTS + [{"n": "curve X", "kind": "artifact", "is": "a duplicate declaration"}])),
    ]
    out = [case(n, d) for n, d in cases]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
