"""Run ONE MemoryArena paper through the harness's own loop, in its own process.

This is run_smoke.py's per-paper body, made safe to run beside other papers:

  * run_smoke.py proved "one paper -> one new vault" by diffing the vault directory around the
    call. Under concurrency that diff also sees every OTHER worker's new vault, so it cannot be
    the proof any more. Here the paper's `user_id` is captured where the harness mints it --
    `run_math.MemoryClient(task_id, ...)` is replaced by a subclass that records the id it was
    constructed with and changes nothing else -- and the vault is then required to exist at
    `vaults/arena/<run_id>/<writer>/<user_id>` afterwards and NOT before. The independent proof
    that the vault holds only this paper's chunks is arena/check_isolation.py, which reads the
    disk and the harness's result file and never trusts this record.
  * the record is written to its own file (one writer per file); the orchestrator appends it
    to task_map.jsonl, so concurrent appends never interleave.
  * BENCH_TAG carries the task (`<run_id>/<arm>/<domain>:<id>`), so every agent-role ledger
    row is attributable to its paper. Writer rows carry the user_id (adapter); judge rows carry
    the arm (env server).

usage: run_task.py --run-id R --arm kscope|kscope_none --task math:8 --mem-url U --env-url U
exit codes: 0 completed, 2 harness exception, 3 vault accounting wrong, 4 spend cap
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
ARENA = BASE / "MemoryArena"
HF_CONFIG = {"math": "formal_reasoning_math", "phys": "formal_reasoning_phys"}
WRITER = {"kscope": "kscope", "kscope_none": "none", "kscope_full": "full_history"}
sys.path.insert(0, str(BASE / "arena"))
from serve import Redact, SdkRetryLinesOnly  # noqa: E402  (same filters as the servers)


def secrets() -> list[str]:
    from common import llm
    llm._load_env()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    return [s for s in (endpoint, endpoint.split("://", 1)[-1], os.environ.get("AZURE_OPENAI_API_KEY", "")) if s]


def redact(text: str, hidden: list[str]) -> str:
    for secret in hidden:
        text = text.replace(secret, "<redacted>")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", required=True, choices=sorted(WRITER))
    parser.add_argument("--task", required=True, help="domain:id, e.g. math:37")
    parser.add_argument("--mem-url", required=True)
    parser.add_argument("--env-url", required=True)
    args = parser.parse_args()
    domain, raw_id = args.task.split(":")
    task_id = int(raw_id)
    for name in ("BENCH_SCOPE", "BENCH_SCOPE_CAP_USD", "BENCH_RUN_ID"):
        if not os.environ.get(name):
            raise SystemExit(f"{name} must be set in the environment of every process that can call the LLM")
    assert os.environ["BENCH_RUN_ID"] == args.run_id, (os.environ["BENCH_RUN_ID"], args.run_id)
    os.environ.setdefault("BENCH_TAG", f"{args.run_id}/{args.arm}/{domain}:{task_id}")

    os.chdir(ARENA)
    sys.path.insert(0, str(ARENA))
    sys.path.insert(0, str(BASE))
    out_root = BASE / "results" / "arena" / args.run_id
    (out_root / "configs").mkdir(parents=True, exist_ok=True)
    (out_root / "records").mkdir(parents=True, exist_ok=True)
    hidden = secrets()
    handler = logging.FileHandler(BASE / "logs" / f"arena_driver-{args.run_id}-{args.arm}-{domain}-{task_id}.log")
    handler.setFormatter(logging.Formatter("%(created).3f | %(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    handler.addFilter(Redact(hidden))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "datasets", "fsspec", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sdk = logging.getLogger("openai._base_client")
    sdk.setLevel(logging.DEBUG)
    sdk.addFilter(SdkRetryLinesOnly())
    logging.getLogger("openai").setLevel(logging.DEBUG)

    import run_math  # the harness; imported, not copied
    from datasets import load_dataset
    from common import llm, kscope_io
    from formal_reasoning_env.llm_backend import BudgetStop  # same module object agent/math.py uses

    captured: list[str] = []

    class RecordingMemoryClient(run_math.MemoryClient):
        """Identical to the harness's MemoryClient; also remembers the task_id it was given."""

        def __init__(self, user_id, *a, **kw):
            captured.append(str(user_id))
            super().__init__(user_id, *a, **kw)

    run_math.MemoryClient = RecordingMemoryClient

    cfg = json.load(open(ARENA / "configs" / "formal_reasoning_configs" / f"{domain}_{args.arm}.json"))
    cfg["memory"]["base_url"] = args.mem_url
    cfg["env"]["base_url"] = args.env_url
    cfg["output"]["json_output_dir"] = str(out_root / domain)
    assert cfg["memory"]["memory_system_name"] == args.arm
    assert cfg["task_specific"]["dataset"]["hf_config"] == HF_CONFIG[domain]
    # Every task of an (arm, domain) writes the same bytes, but two of them can be in flight at
    # once: a truncating write racing a reader leaves run_eval.sh a short file. Write-then-rename.
    cfg_path = out_root / "configs" / f"{domain}_{args.arm}.json"
    tmp = cfg_path.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    os.replace(tmp, cfg_path)

    ds = load_dataset(cfg["task_specific"]["dataset"]["hf_dataset"], cfg["task_specific"]["dataset"]["hf_config"],
                      split=cfg["task_specific"]["dataset"]["hf_split"])
    row = ds[task_id]
    assert row["id"] == task_id, (row["id"], task_id)
    paper = row["paper_name"]
    tasks = [(row["questions"][i], row["answers"][i], row["backgrounds"][i]) for i in range(len(row["questions"]))]

    result_file = Path(run_math._get_paper_result_file(cfg, paper))
    if result_file.exists():
        raise SystemExit(f"refusing to re-run {args.task}: {result_file} exists (the harness would skip it silently)")

    vault_dir = BASE / "vaults" / "arena" / args.run_id / WRITER[args.arm]
    record = {"run_id": args.run_id, "arm": args.arm, "domain": domain, "id": task_id, "paper_name": paper,
              "subtasks": len(tasks), "started": round(time.time(), 3), "pid": os.getpid(),
              "tag": os.environ["BENCH_TAG"], "bench_llm_timeout": os.environ.get("BENCH_LLM_TIMEOUT", "(default)"),
              "kscope_sha256_16": hashlib.sha256(kscope_io.KSCOPE.read_bytes()).hexdigest()[:16]}
    stop = False
    logs: list = []
    try:
        logs = run_math.run_task_with_memory_and_env(cfg=cfg, tasks=tasks, paper_key=paper)
        record.update(status="completed", logged_subtasks=len(logs),
                      is_correct=[bool(log.get("is_correct")) for log in logs],
                      tool_path=[(log.get("agent_tool_path") or {}).get("status") for log in logs],
                      tool_path_seconds=[(log.get("agent_tool_path") or {}).get("seconds") for log in logs],
                      degraded_steps=(logs[-1].get("agent_degraded_steps_so_far") if logs else None),
                      subtask_seconds=[round(log.get("time") or 0, 1) for log in logs])
    except BudgetStop as exc:
        record.update(status="budget_stop", error=str(exc))
        stop = True
    except Exception as exc:  # counted and listed, never retried
        record.update(status="harness_exception", error=redact(f"{type(exc).__name__}: {str(exc)[:500]}", hidden),
                      traceback=redact(traceback.format_exc()[-3000:], hidden))
        try:  # a cap raised inside a SERVER arrives here as an HTTP 500
            with llm._locked():
                llm._guard(os.environ["BENCH_SCOPE"])
        except llm.BudgetExceeded as cap:
            record.update(status="budget_stop", cap=str(cap))
            stop = True
    # Vault accounting: exactly one MemoryClient was built, and its vault exists now.
    roots = [vault_dir / u for u in captured]
    record.update(ended=round(time.time(), 3), user_ids_captured=list(captured),
                  new_vaults=[u for u in captured if (vault_dir / u).is_dir()],
                  vault_root=str(roots[0]) if len(roots) == 1 and roots[0].is_dir() else None)
    (out_root / "records" / f"{args.arm}__{domain}__{task_id}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "traceback"}), flush=True)
    exit_code = 0
    if len(captured) != 1 or record["vault_root"] is None:
        print(f"!! expected exactly one memory client and one vault for {args.task}, captured {captured}", flush=True)
        exit_code = 3
    if record["status"] != "completed":
        exit_code = exit_code or 2
    if stop:
        print("!! spend cap reached: stopping", flush=True)
        return 4
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
