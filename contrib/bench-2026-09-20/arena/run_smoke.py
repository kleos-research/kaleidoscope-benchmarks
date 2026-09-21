"""Drive MemoryArena's own per-paper loop for a chosen set of papers, exactly once each.

Why this exists instead of `python run_math.py -c <config>`:
  * `run_math.main()` walks the WHOLE dataset (40 math papers); there is no task filter.
  * `__main__` calls `main()` twice and `main()` swallows per-paper exceptions, so a paper that
    failed in the first pass is silently re-run in the second -- into a second vault, for a
    second bill. A spend cap raised inside a paper would be swallowed the same way.
This driver imports run_math and calls `run_task_with_memory_and_env` -- the harness's own
loop, unmodified -- once per selected paper. It loads the dataset with the harness's own
`load_dataset` call, never skips silently, and proves "one paper -> one new vault" by
diffing the vault directory around each call.
"""
from __future__ import annotations

import argparse
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
WRITER = {"kscope": "kscope", "kscope_none": "none"}


def secrets() -> list[str]:
    """The gateway host and key, read from the environment common/llm.py fills. Never printed."""
    from common import llm
    llm._load_env()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    host = endpoint.split("://", 1)[-1]
    return [s for s in (endpoint, host, os.environ.get("AZURE_OPENAI_API_KEY", "")) if s]


def redact(text: str, hidden: list[str]) -> str:
    for secret in hidden:
        text = text.replace(secret, "<redacted>")
    return text


class Redact(logging.Filter):
    """httpx logs every request URL at INFO, and the harness logs at INFO. The URL is the gateway."""

    def __init__(self, hidden: list[str]):
        super().__init__()
        self.hidden = hidden

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = redact(record.getMessage(), self.hidden), ()
        return True


def vault_dirs(run_id: str, arm: str) -> set[str]:
    root = BASE / "vaults" / "arena" / run_id / WRITER[arm]
    return {p.name for p in root.iterdir() if p.is_dir()} if root.is_dir() else set()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", required=True, choices=sorted(WRITER))
    parser.add_argument("--mem-url", required=True)
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--tasks", nargs="+", required=True, help="domain:id, e.g. math:37 phys:0")
    args = parser.parse_args()

    for name in ("BENCH_SCOPE", "BENCH_SCOPE_CAP_USD"):
        if not os.environ.get(name):
            raise SystemExit(f"{name} must be set in the environment of every process that can call the LLM")

    os.chdir(ARENA)
    sys.path.insert(0, str(ARENA))
    sys.path.insert(0, str(BASE))
    out_root = BASE / "results" / "arena" / args.run_id
    (out_root / "configs").mkdir(parents=True, exist_ok=True)
    hidden = secrets()
    handler = logging.FileHandler(BASE / "logs" / f"arena_driver-{args.run_id}-{args.arm}.log")
    handler.addFilter(Redact(hidden))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        handlers=[handler])
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "urllib3", "huggingface_hub", "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import run_math  # the harness; imported, not copied
    from datasets import load_dataset
    from common import llm
    from formal_reasoning_env.llm_backend import BudgetStop  # same module object agent/math.py uses

    task_map = out_root / "task_map.jsonl"
    exit_code = 0
    for spec in args.tasks:
        domain, raw_id = spec.split(":")
        task_id = int(raw_id)
        cfg = json.load(open(ARENA / "configs" / "formal_reasoning_configs" / f"{domain}_{args.arm}.json"))
        # Run-time overrides: where the two servers listen, and a per-run output directory.
        cfg["memory"]["base_url"] = args.mem_url
        cfg["env"]["base_url"] = args.env_url
        cfg["output"]["json_output_dir"] = str(out_root / domain)
        assert cfg["memory"]["memory_system_name"] == args.arm
        assert cfg["task_specific"]["dataset"]["hf_config"] == HF_CONFIG[domain]
        json.dump(cfg, open(out_root / "configs" / f"{domain}_{args.arm}.json", "w"), indent=2)

        ds = load_dataset(cfg["task_specific"]["dataset"]["hf_dataset"],
                          cfg["task_specific"]["dataset"]["hf_config"],
                          split=cfg["task_specific"]["dataset"]["hf_split"])
        row = ds[task_id]
        assert row["id"] == task_id, (row["id"], task_id)
        paper = row["paper_name"]
        # Built exactly as run_math.main() builds it.
        tasks = [(row["questions"][i], row["answers"][i], row["backgrounds"][i]) for i in range(len(row["questions"]))]

        result_file = Path(run_math._get_paper_result_file(cfg, paper))
        if result_file.exists():
            raise SystemExit(f"refusing to re-run {spec}: {result_file} exists (the harness would skip it silently)")

        record = {"run_id": args.run_id, "arm": args.arm, "domain": domain, "id": task_id, "paper_name": paper,
                  "subtasks": len(tasks), "started": round(time.time(), 3)}
        before = vault_dirs(args.run_id, args.arm)
        stop = False
        try:
            logs = run_math.run_task_with_memory_and_env(cfg=cfg, tasks=tasks, paper_key=paper)
            record.update(status="completed", logged_subtasks=len(logs),
                          is_correct=[bool(log.get("is_correct")) for log in logs])
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
        new = sorted(vault_dirs(args.run_id, args.arm) - before)
        record.update(ended=round(time.time(), 3), new_vaults=new,
                      vault_root=str(BASE / "vaults" / "arena" / args.run_id / WRITER[args.arm] / new[0]) if len(new) == 1 else None)
        with open(task_map, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "traceback"}), flush=True)
        if len(new) != 1:
            print(f"!! expected exactly one new vault for {spec}, found {len(new)}", flush=True)
            exit_code = 3
        if record["status"] != "completed":
            exit_code = exit_code or 2
        if stop:
            print("!! spend cap reached: stopping", flush=True)
            return 4
    print(llm.report(), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
