#!/usr/bin/env python
"""Drive the full MemoryAgentBench run, one context length at a time, smallest first.

Per length:
  1. if no sealed master exists for (run_id, length) under runs/mab/<run_id>/master/<length>/,
     run mab/ingest_master.py (blocking). Its exit code is the lost-facts gate: non-zero stops the
     whole drive -- nothing is adjusted, the report says why.
  2. launch every (arm, row) as its own harness process through mab/run_mab.py, all in parallel
     (each process makes one reader call at a time, so this is at most len(arms) x 2 concurrent
     reader calls, all short), and wait for all of them. A non-zero exit stops the drive.
Writes runs/mab/<run_id>/driver-status.json after every step, so progress is a file.

Usage:  . mab/env.sh && . mab/env_full.sh && python mab/drive_full.py --run_id <id> --lengths 32k,64k,262k
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
PY = sys.executable
ARMS_DEFAULT = ("none", "bm25_units", "kscope")
ROWS = ("sh", "mh")


def log(status_path: Path, status: dict, line: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {line}", flush=True)
    status["log"].append(f"[{stamp}] {line}")
    status["updated"] = time.time()
    status_path.write_text(json.dumps(status, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--lengths", default="32k,64k,262k")
    parser.add_argument("--arms", default=",".join(ARMS_DEFAULT))
    parser.add_argument("--writer_workers", type=int, default=8)
    parser.add_argument("--max_queries", type=int, default=0, help="0 = every question")
    args = parser.parse_args()
    for name in ("BENCH_SCOPE", "BENCH_SCOPE_CAP_USD"):
        if not os.environ.get(name):
            sys.exit(f"{name} missing: source mab/env.sh and mab/env_full.sh")
    lengths = [x.strip() for x in args.lengths.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    run_root = BASE / "runs" / "mab" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "driver-status.json"
    status = {"run_id": args.run_id, "lengths": lengths, "arms": arms, "started": time.time(),
              "stages": {}, "log": [], "stopped": None}
    log(status_path, status, f"drive {args.run_id}: lengths {lengths}, arms {arms}, writer workers {args.writer_workers}")

    for length in lengths:
        stage = status["stages"].setdefault(length, {})
        pointer = run_root / "master" / length / "manifest.txt"
        if not pointer.exists():
            ingest_log = BASE / "logs" / f"mab_ingest-{args.run_id}-{length}.log"
            log(status_path, status, f"{length}: ingesting master (log {ingest_log.name})")
            started = time.time()
            with open(ingest_log, "w") as handle:
                proc = subprocess.run([PY, str(BASE / "mab" / "ingest_master.py"), "--run_id", args.run_id,
                                       "--length", length, "--writer_workers", str(args.writer_workers)],
                                      stdout=handle, stderr=subprocess.STDOUT, cwd=BASE)
            stage["ingest_exit"] = proc.returncode
            stage["ingest_seconds"] = round(time.time() - started, 1)
            log(status_path, status, f"{length}: ingest exit {proc.returncode} after {stage['ingest_seconds'] / 60:.1f} min")
            if proc.returncode != 0 or not pointer.exists():
                status["stopped"] = f"{length}: master ingestion exit {proc.returncode} (gate or failure); see {ingest_log}"
                log(status_path, status, "STOP: " + status["stopped"])
                return 3
        manifest = pointer.read_text().strip()
        stage["manifest"] = manifest
        gate = json.loads(Path(manifest).read_text())["gate"]
        stage["gate"] = gate
        if not gate["pass"]:
            status["stopped"] = f"{length}: master gate failed {gate}"
            log(status_path, status, "STOP: " + status["stopped"])
            return 3
        log(status_path, status, f"{length}: master sealed, units {gate['units']}, refused {gate['refused_first_try']}, "
                                 f"facts-lost rate {gate['facts_lost_rate']:.2%}, clamped {gate['clamped_units']}")

        procs = {}
        for arm in arms:
            for row in ROWS:
                dataset = f"configs/data_conf/Conflict_Resolution/Factconsolidation_{row}_{length}.yaml"
                command = [PY, str(BASE / "mab" / "run_mab.py"), "--run_id", args.run_id, "--arm", arm,
                           "--dataset_config", dataset, "--max_queries", str(args.max_queries)]
                if arm in ("kscope", "bm25_units"):
                    command += ["--master_manifest", manifest]
                out = BASE / "logs" / f"mab_launcher-{args.run_id}-{arm}-{row}_{length}.log"
                procs[(arm, row)] = (subprocess.Popen(command, stdout=open(out, "w"), stderr=subprocess.STDOUT, cwd=BASE), time.time())
        log(status_path, status, f"{length}: launched {len(procs)} harness processes")
        exits = {}
        for (arm, row), (proc, started) in procs.items():
            proc.wait()
            exits[f"{arm}/{row}"] = {"exit": proc.returncode, "seconds": round(time.time() - started, 1)}
            log(status_path, status, f"{length}: {arm}/{row} exit {proc.returncode} after {(time.time() - started) / 60:.1f} min")
        stage["arms"] = exits
        if any(e["exit"] != 0 for e in exits.values()):
            status["stopped"] = f"{length}: a harness process failed: {exits}"
            log(status_path, status, "STOP: " + status["stopped"])
            return 4
    status["finished"] = time.time()
    log(status_path, status, "drive complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
