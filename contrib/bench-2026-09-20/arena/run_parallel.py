"""Run MemoryArena papers in parallel: one server pair per arm, N worker slots, one
`run_task.py` PROCESS per (arm, paper).

Why tasks and not subtasks: a paper's subtasks are a chain -- each one's prompt is wrapped with
memory written by the ones before it -- so they stay sequential inside their process. Papers
are independent (one vault each), so they run side by side.

Why one process per paper: a paper's `user_id` is captured in its own process (run_task.py),
BENCH_TAG is per process (so agent ledger rows name their paper), a stuck paper can be killed
without touching the others, and a spend cap raised in one paper is seen by the orchestrator,
which then drains the queue instead of letting the other workers keep spending.

Why one server pair per ARM and not per worker: memory/server.py keys its instances by
user_id and env/env_server.py by task_id; both are plain dict lookups, and the FastAPI
endpoints are sync `def`s that uvicorn runs on a thread pool (40 threads), so N concurrent
papers use N threads. The remaining process-wide state -- the OpenAI client, the
`_temperature_ok` flag, the writer's cached schema text, the adapter's learned query bound --
is idempotent (every writer of it writes the same value). The server's BENCH_TAG is per arm,
which is what judge-role ledger rows carry. The isolation proof does not rest on this
reasoning: arena/check_isolation.py reads the disk afterwards.

Scheduling: both arms of the same paper are queued back to back, longest papers first, so the
two arms see the same gateway weather and the makespan is bounded by the longest chain.

usage: run_parallel.py --run-id R --workers 8 --tasks math:8 phys:0 ... [--arms kscope kscope_none]
       [--llm-timeout 90] [--task-timeout-s 14400] [--resume]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path
from queue import Empty, Queue

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
PY = BASE / "venv-arena" / "bin" / "python"
ARENA_DIR = BASE / "arena"
DATA = {"math": BASE / "data/memoryarena__formal_reasoning_math__data.jsonl",
        "phys": BASE / "data/memoryarena__formal_reasoning_phys__data.jsonl"}
WRITER = {"kscope": "kscope", "kscope_none": "none", "kscope_full": "full_history"}

children: dict[int, subprocess.Popen] = {}
children_lock = threading.Lock()
stop_event = threading.Event()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def base_env(run_id: str, llm_timeout: str) -> dict:
    env = dict(os.environ)
    env.update({
        "BENCH_SCOPE": os.environ.get("BENCH_SCOPE", "arena"),
        "BENCH_SCOPE_CAP_USD": os.environ.get("BENCH_SCOPE_CAP_USD", "200"),
        "BENCH_RUN_ID": run_id, "BENCH_LLM_TIMEOUT": llm_timeout,
        "HF_HOME": str(BASE / "cache" / "hf"), "HF_HUB_CACHE": str(BASE / "data" / ".hfcache"),
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
        "TIKTOKEN_CACHE_DIR": str(BASE / "cache" / "tiktoken"),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    })
    return env


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_servers(run_id: str, arm: str, env: dict) -> dict:
    mem_port, env_port = free_port(), free_port()
    while env_port == mem_port:
        env_port = free_port()
    procs = {}
    for which, port in (("memory", mem_port), ("env", env_port)):
        log = BASE / "logs" / f"arena_{which}server-{run_id}-{arm}.log"
        out = open(BASE / "logs" / f"arena_{which}server-{run_id}-{arm}.out", "a")
        p = subprocess.Popen([str(PY), str(ARENA_DIR / "serve.py"), which, "--port", str(port), "--log", str(log)],
                             env={**env, "BENCH_TAG": f"{run_id}/{arm}"}, stdout=out, stderr=subprocess.STDOUT)
        procs[which] = p
        with children_lock:
            children[p.pid] = p
    deadline = time.time() + 90
    while time.time() < deadline:
        if http_ok(f"http://127.0.0.1:{mem_port}/openapi.json") and http_ok(f"http://127.0.0.1:{env_port}/env/available"):
            break
        for which, p in procs.items():
            if p.poll() is not None:
                raise SystemExit(f"{which} server for arm {arm} died during start-up (rc={p.returncode})")
        time.sleep(0.5)
    else:
        raise SystemExit(f"servers for arm {arm} did not come up in 90s")
    return {"arm": arm, "mem_port": mem_port, "env_port": env_port,
            "mem_pid": procs["memory"].pid, "env_pid": procs["env"].pid, "procs": procs}


def kill_all() -> None:
    with children_lock:
        procs = list(children.values())
    for p in procs:
        if p.poll() is None:
            try:
                p.terminate()
            except ProcessLookupError:
                pass
    deadline = time.time() + 15
    for p in procs:
        while p.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        if p.poll() is None:
            try:
                p.kill()
            except ProcessLookupError:
                pass
    for p in procs:
        try:
            p.wait(timeout=10)
        except Exception:
            pass


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--arms", nargs="+", default=["kscope", "kscope_none"], choices=sorted(WRITER))
    parser.add_argument("--tasks", nargs="+", required=True, help="domain:id ...")
    parser.add_argument("--llm-timeout", default=os.environ.get("BENCH_LLM_TIMEOUT", "90"))
    parser.add_argument("--task-timeout-s", type=float, default=4 * 3600)
    parser.add_argument("--resume", action="store_true", help="skip (arm, task) pairs that already have a record")
    args = parser.parse_args()
    run_id = args.run_id
    out_root = BASE / "results" / "arena" / run_id
    task_map = out_root / "task_map.jsonl"
    if task_map.exists() and not args.resume:
        raise SystemExit(f"{task_map} exists; a run id is used once (pass --resume to continue this one)")
    out_root.mkdir(parents=True, exist_ok=True)
    (BASE / "logs").mkdir(exist_ok=True)
    (BASE / "cache" / "tiktoken").mkdir(parents=True, exist_ok=True)

    # subtask counts, for longest-first scheduling only (the driver loads the dataset itself)
    sizes = {}
    for domain, path in DATA.items():
        for row in jsonl(path):
            sizes[f"{domain}:{row['id']}"] = len(row["questions"])
    for spec in args.tasks:
        if spec not in sizes:
            raise SystemExit(f"unknown task {spec}")
    done = {(r["arm"], r["domain"], r["id"]) for r in jsonl(task_map)} if args.resume else set()
    order = sorted(dict.fromkeys(args.tasks), key=lambda s: -sizes[s])
    queue: Queue = Queue()
    n_queued = 0
    for spec in order:
        domain, tid = spec.split(":")
        for arm in args.arms:
            if (arm, domain, int(tid)) in done:
                continue
            queue.put((arm, spec))
            n_queued += 1

    env = base_env(run_id, args.llm_timeout)
    plan = {"run_id": run_id, "workers": args.workers, "arms": args.arms, "tasks_in_order": order,
            "queued_pairs": n_queued, "skipped_as_done": len(done), "llm_timeout": args.llm_timeout,
            "task_timeout_s": args.task_timeout_s, "scope": env["BENCH_SCOPE"], "scope_cap_usd": env["BENCH_SCOPE_CAP_USD"],
            "started": round(time.time(), 3), "orchestrator_pid": os.getpid()}
    (out_root / f"plan-{int(plan['started'])}.json").write_text(json.dumps(plan, indent=1))
    print(json.dumps(plan), flush=True)

    def on_signal(signum, frame):
        print(f"!! signal {signum}: stopping workers and servers", flush=True)
        stop_event.set()
        kill_all()
        sys.exit(130)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    servers = {arm: start_servers(run_id, arm, env) for arm in args.arms}
    (out_root / f"servers-{int(plan['started'])}.json").write_text(json.dumps(
        {arm: {k: v for k, v in s.items() if k != "procs"} for arm, s in servers.items()}, indent=1))
    for arm, s in servers.items():
        print(f"arm {arm}: memory :{s['mem_port']} (pid {s['mem_pid']})  env :{s['env_port']} (pid {s['env_pid']})", flush=True)

    map_lock = threading.Lock()
    in_flight: dict[str, float] = {}
    counts: Counter = Counter()

    def status_dump():
        (out_root / "status.json").write_text(json.dumps({
            "ts": round(time.time(), 3), "elapsed_s": round(time.time() - plan["started"], 1),
            "queued_remaining": queue.qsize(), "in_flight": {k: round(time.time() - v, 1) for k, v in in_flight.items()},
            "finished_by_status": dict(counts), "stop": stop_event.is_set()}, indent=1))

    def worker(slot: int):
        while not stop_event.is_set():
            try:
                arm, spec = queue.get_nowait()
            except Empty:
                return
            domain, tid = spec.split(":")
            key = f"{arm}/{spec}"
            tag = f"{run_id}/{arm}/{spec}"
            out_path = BASE / "logs" / f"arena_task-{run_id}-{arm}-{domain}-{tid}.out"
            started = time.time()
            with map_lock:
                in_flight[key] = started
                status_dump()
            with open(out_path, "a") as out:
                p = subprocess.Popen([str(PY), str(ARENA_DIR / "run_task.py"), "--run-id", run_id, "--arm", arm,
                                      "--task", spec, "--mem-url", f"http://127.0.0.1:{servers[arm]['mem_port']}",
                                      "--env-url", f"http://127.0.0.1:{servers[arm]['env_port']}"],
                                     env={**env, "BENCH_TAG": tag}, stdout=out, stderr=subprocess.STDOUT)
            with children_lock:
                children[p.pid] = p
            timed_out = False
            try:
                rc = p.wait(timeout=args.task_timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                rc = p.returncode
            with children_lock:
                children.pop(p.pid, None)
            rec_path = out_root / "records" / f"{arm}__{domain}__{tid}.json"
            if rec_path.exists():
                record = json.loads(rec_path.read_text())
            else:
                record = {"run_id": run_id, "arm": arm, "domain": domain, "id": int(tid), "paper_name": None,
                          "subtasks": sizes[spec], "started": round(started, 3), "ended": round(time.time(), 3),
                          "status": "timeout" if timed_out else "crashed", "error": f"driver exit code {rc}, no record file",
                          "new_vaults": [], "vault_root": None, "tag": tag}
            record["driver_rc"] = rc
            record["slot"] = slot
            record["wall_s"] = round(time.time() - started, 1)
            with map_lock:
                with open(task_map, "a") as handle:
                    handle.write(json.dumps(record) + "\n")
                in_flight.pop(key, None)
                counts[record["status"]] += 1
                status_dump()
            print(f"[{time.strftime('%H:%M:%S')}] slot {slot} {key:<22} {record['status']:<18} wall={record['wall_s']:>7.1f}s "
                  f"correct={record.get('is_correct')} degraded={record.get('degraded_steps')} tool_path={record.get('tool_path')} "
                  f"remaining={queue.qsize()}", flush=True)
            if record["status"] == "budget_stop":
                print("!! spend cap reached in a worker: draining the queue", flush=True)
                stop_event.set()

    threads = [threading.Thread(target=worker, args=(i,), name=f"slot-{i}", daemon=True) for i in range(args.workers)]
    for t in threads:
        t.start()
        time.sleep(2.0)  # stagger the first wave so eight drivers do not import and init at the same instant
    for t in threads:
        t.join()

    sys.path.insert(0, str(BASE))
    from common import llm
    print(llm.report(), flush=True)
    kill_all()
    left = [p.pid for p in children.values() if p.poll() is None]
    rows = jsonl(task_map)
    summary = Counter(r["status"] for r in rows)
    print(f"finished: {dict(summary)}; rows in task_map={len(rows)}; children still alive={left}", flush=True)
    return 0 if (summary.get("completed", 0) == len(rows) and not left and not stop_event.is_set()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
