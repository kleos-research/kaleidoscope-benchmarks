"""Probe: the shape of a search response and of a remember response (no LLM call)."""
import json, sys, time
from pathlib import Path
BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
sys.path.insert(0, str(BASE))
from common import kscope_io

root = kscope_io.VAULTS / "arena" / f"probe-{int(time.time())}" / "shape"
vault = kscope_io.Vault.create(root)
big = "## Task: big content probe\n## solution: " + ("lorem ipsum dolor sit amet \\frac{a}{b} " * 3000) + "\n"
items = [
    {"content_md": "## Task: What is a localization of a C-infinity ring?\n## solution: unique up to unique isomorphism\n",
     "semantic_delta": {"memory_type": "note", "title": "localization of a C-infinity ring",
                        "facts": [{"subject": "agent record", "predicate": "instance_of", "object": "unstructured record"}]}},
    {"content_md": big,
     "semantic_delta": {"memory_type": "note", "title": "big content probe",
                        "facts": [{"subject": "agent record", "predicate": "instance_of", "object": "unstructured record"}]}},
    {"content_md": "control char in a fact",
     "semantic_delta": {"memory_type": "note", "title": "control char probe",
                        "facts": [{"subject": "angle \theta", "predicate": "instance_of", "object": "unstructured record"}]}},
]
print("big content bytes", len(big.encode()))
rc, body, err = vault.remember_items(items)
print("remember rc", rc)
print(json.dumps(body, indent=1)[:3000])
print("stderr:", err[:500])
rc, body, err = vault.search("What is a localization of a C-infinity ring", top_k=10, maximum_context_bytes=32768)
print("search rc", rc, "keys", sorted(body.keys()) if body else None)
slim = json.loads(json.dumps(body))
for hit in slim.get("selected_hits") or []:
    for k, v in list(hit.items()):
        if isinstance(v, str) and len(v) > 200:
            hit[k] = v[:200] + f"...<{len(v)} chars>"
print(json.dumps(slim, indent=1)[:6000])
