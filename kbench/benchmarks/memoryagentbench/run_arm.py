"""Launch ONE arm of MemoryAgentBench, on ONE dataset row, through the harness's own main.py.

What this adds to a bare `python main.py ...`, and why:
  * a per-run copy of the agent YAML whose only changes are `output_dir` and, for the full run,
    `kscope_master_manifest`. The harness resumes from an existing results file
    (initialization.load_existing_results): a second run into the same directory would re-ingest --
    spending money -- and then skip every question already answered. So every (run, arm, row) gets
    a fresh results directory under results/mab/<run_id>/<arm>/<sub_dataset>, never under the
    harness's own outputs/ tree, and an existing run directory is refused.
  * MAB_RUN_ID, which the adapter needs to name its vault and its log, and MAB_LOG_SUFFIX so that
    arms running concurrently under one run id each append to their own adapter log.
  * a refusal to start unless the spend scope and cap are in the environment (see the README).
  * the harness's stdout/stderr captured to logs/mab_harness-<run_id>-<arm>-<sub_dataset>.log.
The only question cap the harness reads is --max_test_queries_ablation; this passes it through.

Arms `kscope` and `bm25_units` take --master_manifest (from ingest.py): the kscope arm
clones the sealed master instead of ingesting, the bm25_units arm verifies its units against it.
--no_master keeps the smoke-run behaviour (in-process ingestion) for the kscope arm.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("MAB_WORKDIR", HERE.parents[2] / "results" / "memoryagentbench"))
os.environ.setdefault("MAB_WORKDIR", str(BASE))
os.environ.setdefault("MAB_HARNESS_DIR", str(HERE))
HARNESS = BASE / "MemoryAgentBench"
ARMS = {
    "none": "configs/agent_conf/RAG_Agents/gpt-5.6-luna/Kscope_gpt-5.6-luna-none.yaml",
    "kscope": "configs/agent_conf/RAG_Agents/gpt-5.6-luna/Kscope_gpt-5.6-luna.yaml",
    "bm25_units": "configs/agent_conf/RAG_Agents/gpt-5.6-luna/Kscope_gpt-5.6-luna-bm25units.yaml",
    "full_context": "configs/agent_conf/RAG_Agents/gpt-5.6-luna/Kscope_gpt-5.6-luna-fullcontext.yaml",
}
REQUIRED_ENV = ("BENCH_SCOPE", "BENCH_SCOPE_CAP_USD", "HF_HOME", "NLTK_DATA", "TIKTOKEN_CACHE_DIR")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--dataset_config", default="configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml")
    parser.add_argument("--max_queries", type=int, required=True, help="0 = every question")
    parser.add_argument("--master_manifest", default=None, help="sealed master vault manifest (kscope, bm25_units)")
    parser.add_argument("--no_master", action="store_true", help="kscope arm ingests in-process (smoke behaviour)")
    args = parser.parse_args()

    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        sys.exit(f"missing environment: {missing}. Set them; see the Running section of this benchmark's README.")
    for name in ("HF_HOME", "NLTK_DATA", "TIKTOKEN_CACHE_DIR"):
        if BASE not in Path(os.environ[name]).resolve().parents:
            sys.exit(f"{name} must live under {BASE}")
    if args.arm in ("kscope", "bm25_units", "full_context") and not args.master_manifest and not args.no_master:
        sys.exit(f"arm {args.arm} needs --master_manifest (or --no_master for a smoke run)")
    if args.master_manifest and not Path(args.master_manifest).is_file():
        sys.exit(f"no manifest at {args.master_manifest}")

    dataset_config = yaml.safe_load((HARNESS / args.dataset_config).read_text())
    sub_dataset = dataset_config["sub_dataset"]
    run_dir = BASE / "runs" / "mab" / args.run_id / args.arm / sub_dataset
    results_dir = BASE / "results" / "mab" / args.run_id / args.arm / sub_dataset
    if run_dir.exists() or results_dir.exists():
        sys.exit(f"refusing to reuse {run_dir} / {results_dir}: the harness would resume from old results")
    run_dir.mkdir(parents=True)

    config = yaml.safe_load((HARNESS / ARMS[args.arm]).read_text())
    config["output_dir"] = str(results_dir)
    if args.master_manifest:
        config["kscope_master_manifest"] = str(Path(args.master_manifest).resolve())
    agent_yaml = run_dir / "agent.yaml"
    agent_yaml.write_text(yaml.safe_dump(config, sort_keys=False))

    env = dict(os.environ)
    env["MAB_RUN_ID"] = args.run_id
    env["MAB_LOG_SUFFIX"] = f"-{args.arm}-{sub_dataset}"
    env["PYTHONUNBUFFERED"] = "1"
    env["MAB_WORKDIR"] = str(BASE)
    env["MAB_HARNESS_DIR"] = str(HERE)
    command = [sys.executable, "main.py", "--agent_config", str(agent_yaml),
               "--dataset_config", args.dataset_config,
               "--max_test_queries_ablation", str(args.max_queries)]
    (run_dir / "command.txt").write_text(" ".join(command) + "\n")
    log_path = BASE / "logs" / f"mab_harness-{args.run_id}-{args.arm}-{sub_dataset}.log"
    with open(log_path, "w") as log:
        proc = subprocess.run(command, check=False, cwd=HARNESS, env=env, stdout=log, stderr=subprocess.STDOUT)
    (run_dir / "exit_code.txt").write_text(f"{proc.returncode}\n")
    print(f"arm={args.arm} row={sub_dataset} run_id={args.run_id} exit={proc.returncode} log={log_path}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
