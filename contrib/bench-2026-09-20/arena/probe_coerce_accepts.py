"""Zero-LLM-cost check that `common/writer.py`'s `coerce_propose` (via `to_item`, the function the
arena adapter calls) turns the three refused `propose` shapes of the interrupted smoke into
items kscope ACCEPTS, and that the fourth refusal (two entities, one surface) is still refused
-- coerce_propose does not touch entities, and this probe must not quietly claim it does.

Runs under BENCH_SCOPE=arena-probe so it cannot touch the arena call log."""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["BENCH_SCOPE"] = "arena-probe"
BASE = "/path/to/kaleidoscope/experiments/bench-2026-09-20"
sys.path.insert(0, BASE)
from common import kscope_io  # noqa: E402
from common import writer as delta_writer  # noqa: E402

ENTS = [{"n": "curve X", "kind": "concept", "is": "a regular proper curve"},
        {"n": "family F", "kind": "concept", "is": "a family of curves"}]
FACT = {"subject": "curve X", "predicate": "fibers_over", "object": "family F"}
PROPOSE_OK = {"rel": "fibers_over", "means": "the subject admits a morphism onto the object",
              "many": False, "over_time": "state", "from": ["concept"], "to": ["concept"]}


def send(name: str, delta: dict) -> dict:
    root = kscope_io.VAULTS / "arena-probe" / f"coerce-{int(time.time() * 1000)}"
    vault = kscope_io.Vault.create(root)
    item = delta_writer.to_item("A probe unit.\n", delta)          # the adapter's exact path
    _, stats = delta_writer.coerce_propose(delta)
    rc, body, err = vault.remember_items([item])
    body = body or {}
    results = body.get("results") or []
    reason = (results[0] if results else {}).get("reason") or body.get("message") or (err or "").strip()
    return {"case": name, "coerce_stats": stats, "rc": rc, "accepted": body.get("accepted_items"),
            "reason": str(reason)[:160]}


def main() -> int:
    def delta(**over):
        d = {"memory_type": "fact", "title": "probe", "facts": [dict(FACT)], "entities": [dict(e) for e in ENTS]}
        d.update(over)
        return d

    cases = [
        ("propose.from as a STRING", delta(propose=[{**PROPOSE_OK, "from": "concept"}])),
        ("propose.to as a STRING", delta(propose=[{**PROPOSE_OK, "to": "element"}])),
        ("propose.many as the STRING 'false'", delta(propose=[{**PROPOSE_OK, "many": "false"}])),
        ("propose.many unreadable ('sometimes') -> proposal dropped, fact kept", delta(propose=[{**PROPOSE_OK, "many": "sometimes"}])),
        ("two entities declaring one surface (NOT a propose defect)", delta(entities=ENTS + [{"n": "curve X", "kind": "artifact", "is": "dup"}])),
    ]
    out = [send(n, d) for n, d in cases]
    print(json.dumps(out, indent=1))
    ok = all(r["accepted"] == 1 for r in out[:4]) and out[4]["accepted"] != 1
    print("EXPECTED PATTERN (first four accepted, fifth refused):", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
