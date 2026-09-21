"""Zero-LLM-cost falsifier for the two store constraints the smoke hit, and for the claim
that an `rc=3 status=partial` remember HAS ALREADY WRITTEN the memory record.

The smoke's assertion 3 went red because two kscope vaults held one memory MORE than the
adapter had chunks. Both tasks had exactly one refusal that came back `rc=3 status=partial`.
The adapter's `_remember` treats anything but `(rc=0, accepted_items=1)` as "nothing was
written" and re-stores the chunk under the fallback delta.

Disk evidence from the smoke (vault 5285961d, phys:14): two records carry the SAME body and
differ only in the `# <title>` heading kscope renders on top -- one titled with the writer's
title, one titled with the fallback's first line. Only the second is in the adapter log. So
`partial` wrote the first and the adapter wrote the second.

This probe asks the binary directly. NO LLM call. Own BENCH_SCOPE, so it cannot touch the
arena run's call log or its roots_touched proof.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

os.environ["BENCH_SCOPE"] = "arena-probe"
BASE = "/path/to/kaleidoscope/experiments/bench-2026-09-20"
sys.path.insert(0, BASE)
from common import kscope_io  # noqa: E402

SHORT_A, SHORT_B = "curve X", "family F"
BOUND_RE = re.compile(r"Handle is (\d+) bytes, over the (\d+)-byte bound")


def records(root) -> list[str]:
    return sorted(p.name for p in root.glob("workspaces/*/records/memory/mem_*"))


def titles(root) -> list[str]:
    out = []
    for m in sorted(root.glob("workspaces/*/records/memory/mem_*")):
        for p in m.glob("versions/*/content.md"):
            out.append(p.read_text().splitlines()[0][:70])
    return out


def send(name: str, delta: dict, vault=None, root=None) -> dict:
    if vault is None:
        root = kscope_io.VAULTS / "arena-probe" / f"partial-{int(time.time() * 1000)}"
        vault = kscope_io.Vault.create(root)
    before = records(root)
    rc, body, err = vault.remember_items([{"content_md": "A probe unit.\n", "semantic_delta": delta}])
    body = body or {}
    after = records(root)
    results = body.get("results") or []
    first = results[0] if results and isinstance(results[0], dict) else {}
    reason = first.get("reason") or first.get("message") or body.get("message") or (err or "").strip()
    return {"case": name, "rc": rc, "status": body.get("status"), "accepted": body.get("accepted_items"),
            "refused": body.get("refused_items"), "rejected": body.get("rejected_items"),
            "review": body.get("review_items"), "unfolded": body.get("unfolded_items"),
            "memory_kept": first.get("memory_kept"), "item_status": first.get("status"),
            "graph_fold_total": body.get("graph_fold_total"),
            "records_before": len(before), "records_after": len(after),
            "wrote_a_record": len(after) > len(before), "titles": titles(root),
            "reason": str(reason)[:260], "_vault": vault, "_root": root}


def strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def main() -> int:
    def delta(**over):
        d = {"memory_type": "fact", "title": "probe title",
             "facts": [{"subject": SHORT_A, "predicate": "instance_of", "object": SHORT_B}],
             "entities": [{"n": SHORT_A, "kind": "concept", "is": "a regular proper curve"},
                          {"n": SHORT_B, "kind": "concept", "is": "a family of curves"}]}
        d.update(over)
        return d

    out = []

    # ---- 1. where is the handle bound, measured on the raw surface ------------------
    print("# bisect: raw entity-surface bytes -> accepted?", flush=True)
    bisect = []
    for n in (60, 80, 100, 110, 120, 124, 126, 130, 140, 160):
        surface = "alpha " * 40
        surface = surface[:n].strip()
        r = send(f"surface {len(surface.encode())}B", delta(
            entities=[{"n": surface, "kind": "concept", "is": "x"},
                      {"n": SHORT_B, "kind": "concept", "is": "y"}],
            facts=[{"subject": surface, "predicate": "instance_of", "object": SHORT_B}]))
        m = BOUND_RE.search(r["reason"] or "")
        bisect.append({"raw_bytes": len(surface.encode()), "rc": r["rc"], "status": r["status"],
                       "accepted": r["accepted"], "wrote": r["wrote_a_record"],
                       "handle_bytes": int(m.group(1)) if m else None,
                       "limit": int(m.group(2)) if m else None})
        print("   ", bisect[-1], flush=True)
    out.append({"bisect_surface_bytes": bisect})

    # ---- 2. a reproduced partial: is the memory kept? -------------------------------
    over = ("alpha " * 40)[:160].strip()
    r = send("PARTIAL: over-long entity handle", delta(
        entities=[{"n": over, "kind": "concept", "is": "x"}, {"n": SHORT_B, "kind": "concept", "is": "y"}],
        facts=[{"subject": over, "predicate": "instance_of", "object": SHORT_B},
               {"subject": SHORT_A, "predicate": "instance_of", "object": SHORT_B}]))
    out.append(strip(r))
    print("\n# PARTIAL:", json.dumps(strip(r), indent=1), flush=True)

    # ---- 3. reserved relation in facts: rc=2, nothing written, and the repair -------
    bad = delta(facts=[{"subject": SHORT_A, "predicate": "instance_of", "object": SHORT_B},
                       {"subject": SHORT_A, "predicate": "same_as", "object": SHORT_B},
                       {"subject": SHORT_B, "predicate": "depends_on", "object": SHORT_A}])
    r2 = send("RESERVED: same_as in facts", bad)
    out.append(strip(r2))
    print("\n# RESERVED:", json.dumps(strip(r2), indent=1), flush=True)
    # retry, in the SAME vault, with the reserved fact dropped -- the writer's other facts survive
    repaired = dict(bad)
    repaired["facts"] = [f for f in bad["facts"] if f["predicate"] != "same_as"]
    r3 = send("RESERVED repaired in the same vault", repaired, vault=r2["_vault"], root=r2["_root"])
    out.append(strip(r3))
    print("\n# RESERVED REPAIRED:", json.dumps(strip(r3), indent=1), flush=True)

    # ---- 4. can a reserved relation name be read out of the refusal? ----------------
    named = re.search(r'rel: "([^"]+)" is reserved', r2["reason"] or "")
    out.append({"reserved_relation_named_by_the_binary": named.group(1) if named else None,
                "full_reason": r2["reason"]})
    print("\n# relation named by the binary:", named.group(1) if named else None, flush=True)

    path = kscope_io.ROOT / "results" / "arena" / "probe_partial_writes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
