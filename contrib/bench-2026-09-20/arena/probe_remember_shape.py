"""Probe: what `remember` returns per item for an accepted and a refused item (no LLM call)."""
import json, sys, time
from pathlib import Path
BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
sys.path.insert(0, str(BASE))
from common import kscope_io

root = kscope_io.VAULTS / "arena" / f"probe-{int(time.time())}" / "remember-shape"
vault = kscope_io.Vault.create(root)
good = {"content_md": "Initial result: Empty\n",
        "semantic_delta": {"memory_type": "note", "title": "Initial result: Empty",
                           "facts": [{"subject": "agent record", "predicate": "instance_of", "object": "unstructured record"}]}}
bad_predicate = {"content_md": "refusal probe",
                 "semantic_delta": {"memory_type": "note", "title": "refusal probe",
                                    "facts": [{"subject": "a", "predicate": "same_as", "object": "b"}]}}
long_entity = {"content_md": "long entity probe",
               "semantic_delta": {"memory_type": "note", "title": "long entity probe",
                                  "facts": [{"subject": "x" * 1500, "predicate": "uses", "object": "b"}]}}
for name, item in [("good", good), ("bad_predicate", bad_predicate), ("long_entity", long_entity)]:
    rc, body, err = vault.remember_items([item])
    res = ((body or {}).get("results") or [{}])[0]
    slim = {k: v for k, v in res.items() if k not in ("graph_fold",)}
    print(name, "rc", rc, "accepted", (body or {}).get("accepted_items"), "refused", (body or {}).get("refused_items"),
          "status", (body or {}).get("status"))
    print("   result[0] (minus graph_fold):", json.dumps(slim)[:700])
    if err.strip():
        print("   stderr:", err.strip()[:300])
    if not (body or {}).get("results"):
        print("   body:", json.dumps(body)[:700])
