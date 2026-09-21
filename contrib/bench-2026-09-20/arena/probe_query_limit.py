"""Probe: does `kscope search` refuse a long query? No LLM call is made here.

Uses a hand-written delta (this is a probe vault, not a benchmark vault) and the
real MemoryArena prompts, which are question + background and run to ~13 KB.
"""
import json
import sys
import time
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
sys.path.insert(0, str(BASE))
from common import kscope_io  # noqa: E402

rows = [json.loads(l) for l in open(BASE / "data/memoryarena__formal_reasoning_math__data.jsonl")]
task = rows[39]
root = kscope_io.VAULTS / "arena" / f"probe-{int(time.time())}" / "qlen"
vault = kscope_io.Vault.create(root)
item = {"content_md": "## Task: probe\n## solution: a localization of a C-infinity ring is unique up to unique isomorphism\n",
        "semantic_delta": {"memory_type": "note", "title": "probe record",
                           "facts": [{"subject": "probe record", "predicate": "instance_of", "object": "unstructured record"}]}}
rc, body, err = vault.remember_items([item])
print("remember rc", rc, "accepted", body and body.get("accepted_items"), "refused", body and body.get("refused_items"), "status", body and body.get("status"))
if body and body.get("refused_items"):
    print(json.dumps(body, indent=1)[:1500])

full = f"""
            ### BACKGROUND:
            {task['backgrounds'][0]}
            ### PROBLEM:
            {task['questions'][0]}"""
print("full prompt bytes", len(full.encode()), "chars", len(full))
for n in [len(full), 8192, 4096, 2048, 1024, 512, 256]:
    q = full[:n]
    rc, body, err = vault.search(q, top_k=10, maximum_context_bytes=32768)
    msg = ""
    if rc != 0 or not body or "selected_hits" not in body:
        msg = (json.dumps(body)[:400] if body else "") + " | stderr: " + (err or "")[:400]
    print(f"query chars={len(q):6d} bytes={len(q.encode()):6d} rc={rc} served={len((body or {}).get('selected_hits') or [])} "
          f"stop={(body or {}).get('stop_reason')} {msg}")
print("root", root)
