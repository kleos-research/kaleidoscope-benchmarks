#!/usr/bin/env python
"""Drive the full_context arm: both 262k rows, 100 questions each, ONE AT A TIME.

Sequential on purpose. One prompt of this arm reserves ~296,000 of the gateway's 500,000
tokens/minute, so two of them cannot share a minute however many processes ask; running the rows
concurrently would buy nothing and would blind each process's pacer to the other (the pacer reads
OTHER scopes out of the shared ledger, and two rows of this arm are the same scope).

Usage:  . mab/env.sh && . mab/env_fullctx.sh && python mab/drive_fullctx.py --run_id <id>
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
MANIFEST_POINTER = BASE / "runs" / "mab" / "full-20260920T1905" / "master" / "262k" / "manifest.txt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--rows", default="sh,mh")
    parser.add_argument("--max_queries", type=int, default=0, help="0 = every question")
    args = parser.parse_args()
    for name in ("BENCH_SCOPE", "BENCH_SCOPE_CAP_USD"):
        if not os.environ.get(name):
            sys.exit(f"{name} missing: source mab/env.sh and mab/env_fullctx.sh")
    manifest = MANIFEST_POINTER.read_text().strip()
    if not Path(manifest).is_file():
        sys.exit(f"no master manifest at {manifest}")

    run_root = BASE / "runs" / "mab" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "driver-status.json"
    status = {"run_id": args.run_id, "arm": "full_context", "rows": args.rows.split(","),
              "manifest": manifest, "scope": os.environ["BENCH_SCOPE"],
              "started": time.time(), "stages": {}, "log": []}

    def log(line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {line}", flush=True)
        status["log"].append(f"[{stamp}] {line}")
        status["updated"] = time.time()
        status_path.write_text(json.dumps(status, indent=1))

    log(f"full_context drive {args.run_id}: rows {args.rows}, sequential, manifest {Path(manifest).name}")
    for row in [r.strip() for r in args.rows.split(",") if r.strip()]:
        dataset = f"configs/data_conf/Conflict_Resolution/Factconsolidation_{row}_262k.yaml"
        command = [PY, str(BASE / "mab" / "run_mab.py"), "--run_id", args.run_id,
                   "--arm", "full_context", "--dataset_config", dataset,
                   "--max_queries", str(args.max_queries), "--master_manifest", manifest]
        out = BASE / "logs" / f"mab_launcher-{args.run_id}-full_context-{row}_262k.log"
        started = time.time()
        log(f"{row}: launching (launcher log {out.name})")
        with open(out, "w") as handle:
            proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=BASE)
        seconds = round(time.time() - started, 1)
        status["stages"][row] = {"exit": proc.returncode, "seconds": seconds}
        log(f"{row}: exit {proc.returncode} after {seconds / 60:.1f} min")
        if proc.returncode != 0:
            status["stopped"] = f"{row}: harness exit {proc.returncode}; see {out}"
            log("STOP: " + status["stopped"])
            return 4
    status["finished"] = time.time()
    log("drive complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
