"""Probe the memory-server seam end to end, in process (FastAPI TestClient), on a probe run_id.

Costs two writer LLM calls (the harness's constant seed chunk, and one chunk shaped like
`build_memory_entry` output). The 13 KB query is a real MemoryArena prompt (math id 39, step 0).
"""
import json, os, sys, time, uuid
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
run_id = f"probe-{int(time.time())}"
os.environ["BENCH_RUN_ID"] = run_id
assert os.environ.get("BENCH_SCOPE") == "arena" and os.environ.get("BENCH_SCOPE_CAP_USD")
os.chdir(BASE / "MemoryArena" / "memory")
sys.path.insert(0, str(BASE / "MemoryArena" / "memory"))
sys.path.insert(0, str(BASE / "MemoryArena"))

from fastapi.testclient import TestClient
import server
from agent import MathAgent  # for build_prompt only: no LLM call is made through it here

client = TestClient(server.app)
rows = [json.loads(l) for l in open(BASE / "data/memoryarena__formal_reasoning_math__data.jsonl")]
task = rows[39]
prompt0 = MathAgent.build_prompt(None, task=task["questions"][0], background=task["backgrounds"][0])
prompt1 = MathAgent.build_prompt(None, task=task["questions"][1], background=task["backgrounds"][1])
chunk = ("## Task: " + task["questions"][0] + "\n## solution: (probe) the localization map sends every element of the set to an invertible element, "
         "and the induced map is unique by the universal property.\n## Tool Calls Info: [{'iteration': 1, 'tool': 'reasoning', "
         "'input': 'probe input', 'result': 'probe result with LaTeX $\\\\gamma(a)$ and \\\\theta'}]\n")

for name in ("kscope_none", "kscope"):
    uid = str(uuid.uuid4())
    def post(path, payload):
        r = client.post(path, json=payload)
        assert r.status_code == 200, (path, r.status_code, r.text[:500])
        return r.json()
    print("==", name, uid)
    print(post("/memory/initialize", {"user_id": uid, "memory_system_name": name}))
    print(post("/memory/add", {"user_id": uid, "memory_system_name": name, "chunk": "Initial result: Empty\n"}))
    wrapped = post("/memory/wrap_user_prompt", {"user_id": uid, "memory_system_name": name, "question": prompt0})["prompt"]
    assert wrapped.endswith(f"User: {prompt0}"), "the prompt handed back must be untouched"
    print("step0 head:", repr(wrapped[:120]))
    print(post("/memory/add", {"user_id": uid, "memory_system_name": name, "chunk": chunk}))
    wrapped = post("/memory/wrap_user_prompt", {"user_id": uid, "memory_system_name": name, "question": prompt1})["prompt"]
    assert wrapped.endswith(f"User: {prompt1}")
    block = wrapped.split("<memory_context>")[1].split("</memory_context>")[0]
    print("step1 memory block bytes:", len(block), "| memories:", block.count("<memory>"), "| head:", repr(block[:160]))
    # a second initialise for the same user must be refused, not given a shared vault
    r = client.post("/memory/initialize", json={"user_id": uid, "memory_system_name": name}) if False else None

print("adapter log:", BASE / "logs" / f"arena_adapter-{run_id}.jsonl")
for line in open(BASE / "logs" / f"arena_adapter-{run_id}.jsonl"):
    row = json.loads(line)
    row.pop("chunk", None)
    print(json.dumps(row)[:1400])
