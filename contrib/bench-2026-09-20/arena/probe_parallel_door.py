"""Zero-LLM-cost concurrency probe of the kscope door: N processes each create a vault, remember
K chunks carrying a per-process marker (fallback-shaped delta, no writer), and search for their
own marker AND for every other process's marker. A vault that serves a foreign marker, refuses a
write, or fails to serve its own marker is a red result.

The point is the binary and `common/kscope_io.py` under simultaneous invocation on DISTINCT
roots (locks are per root; the call log and the ledger lock are shared). Scope arena-probe."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ["BENCH_SCOPE"] = "arena-probe"
BASE = "/path/to/kaleidoscope/experiments/bench-2026-09-20"
sys.path.insert(0, BASE)


def one(i: int, n: int, k: int, stamp: int) -> dict:
    from common import kscope_io
    root = kscope_io.VAULTS / "arena-probe" / f"parallel-{stamp}" / f"w{i}"
    vault = kscope_io.Vault.create(root)
    marker = f"zebra{stamp}w{i}"
    accepted = 0
    for j in range(k):
        text = f"## Task: worker {i} record {j} about the {marker} lemma\n## solution: the {marker} bound holds with constant {j}\n"
        rc, body, _ = vault.remember_items([{"content_md": text, "semantic_delta": {
            "memory_type": "note", "title": f"worker {i} record {j}",
            "facts": [{"subject": "agent record", "predicate": "instance_of", "object": "unstructured record"}]}}])
        accepted += (body or {}).get("accepted_items") or 0
    served_own, served_foreign, refused_searches = 0, 0, 0
    for other in range(n):
        m = f"zebra{stamp}w{other}"
        rc, body, _ = vault.search(f"the {m} lemma bound", 10, 32768)
        if rc != 0 or not isinstance(body, dict):
            refused_searches += 1
            continue
        texts = [h.get("content_md") or "" for h in body.get("selected_hits") or []]
        if other == i:
            served_own += sum(1 for t in texts if m in t)
        # foreign = a served text carrying ANY other worker's marker, whatever was asked for
        served_foreign += sum(1 for t in texts if any(f"zebra{stamp}w{o}" in t for o in range(n) if o != i))
    return {"worker": i, "root": str(root), "accepted": accepted, "expected": k, "served_own": served_own,
            "served_foreign": served_foreign, "refused_searches": refused_searches}


def main() -> int:
    n, k = int(sys.argv[1]) if len(sys.argv) > 1 else 8, int(sys.argv[2]) if len(sys.argv) > 2 else 4
    stamp = int(time.time())
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n) as pool:
        rows = list(pool.map(one, range(n), [n] * n, [k] * n, [stamp] * n))
    print(json.dumps(rows, indent=1))
    ok = all(r["accepted"] == k and r["served_own"] >= 1 and r["served_foreign"] == 0 and r["refused_searches"] == 0 for r in rows)
    print(f"{n} workers x {k} writes + {n} searches each in {time.time() - t0:.1f}s ->", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
