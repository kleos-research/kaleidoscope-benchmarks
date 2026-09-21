#!/usr/bin/env python
"""Ingest ONE context length of FactConsolidation, once, into a sealed MASTER vault.

The single-hop and multi-hop rows of a length carry a byte-identical context (sha256-equal), so
the text is written once here and every question run clones the result (`cp -Rc`, verified by
seal) instead of ingesting it again. The master is never searched: any search writes an exposure
row and would break the seal, and the adapter refuses to clone a master whose seal has moved.

What runs is the adapter's own `ingest`, chunk by chunk, in the order the harness would hand the
chunks over -- the chunks come from the harness's ConversationCreator, so they are the chunks
`main.py` feeds every agent -- with the same unit rule, the same writer, the same coercion and the
same batch/fallback writes the smoke runs exercised. Only the writer concurrency is a parameter.

Writes <root>-manifest.json beside the vault: every chunk sha, every unit sha and its memory id,
the counters, the seal, the binary sha and the unit rule. The question runs verify the chunks and
units they receive against it.

Usage:  . mab/env.sh && . mab/env_full.sh && python mab/ingest_master.py --run_id <id> --length 6k [--writer_workers 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import tiktoken
import yaml

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
HARNESS = BASE / "MemoryAgentBench"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HARNESS))
os.chdir(HARNESS)  # the harness resolves its own data paths relative to its root

from conversation_creator import ConversationCreator  # noqa: E402
from methods.kscope_agent import KscopeAdapter, tree_sha  # noqa: E402
from common import kscope_io, llm, writer  # noqa: E402

AGENT_YAML = HARNESS / "configs/agent_conf/RAG_Agents/gpt-5.6-luna/Kscope_gpt-5.6-luna.yaml"
REQUIRED_ENV = ("BENCH_SCOPE", "BENCH_SCOPE_CAP_USD", "HF_HOME", "NLTK_DATA", "TIKTOKEN_CACHE_DIR")
LOST_FACTS_THRESHOLD = 0.02  # owner: stop if refusals that lose a unit's facts exceed 2% of units


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--length", required=True, choices=("6k", "32k", "64k", "262k"))
    parser.add_argument("--writer_workers", type=int, default=8)
    args = parser.parse_args()

    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        sys.exit(f"missing environment: {missing}. Run `. mab/env.sh && . mab/env_full.sh` first.")
    if os.environ["BENCH_SCOPE"] != "mab":
        sys.exit("BENCH_SCOPE must be 'mab' for the full run")
    for name in ("HF_HOME", "NLTK_DATA", "TIKTOKEN_CACHE_DIR"):
        if BASE not in Path(os.environ[name]).resolve().parents:
            sys.exit(f"{name} must live under {BASE}")

    run_dir = BASE / "runs" / "mab" / args.run_id / "master" / args.length
    if run_dir.exists():
        sys.exit(f"refusing to reuse {run_dir}: a master for this length was already started under this run id")
    run_dir.mkdir(parents=True)
    os.environ["MAB_LOG_SUFFIX"] = f"-master-{args.length}"

    agent_config = yaml.safe_load(AGENT_YAML.read_text())
    if agent_config["kscope_writer"] != "kscope" or agent_config["model"] != llm.DEPLOYMENT:
        sys.exit("the agent YAML is not the kscope writing arm on the pinned deployment")

    # The chunks exactly as the harness hands them to an agent, for BOTH rows of this length.
    rows = {}
    for kind in ("sh", "mh"):
        dataset_config = yaml.safe_load(
            (HARNESS / f"configs/data_conf/Conflict_Resolution/Factconsolidation_{kind}_{args.length}.yaml").read_text())
        creator = ConversationCreator(agent_config, dataset_config)
        chunks = creator.get_chunks()
        qa = creator.get_query_and_answers()
        assert len(chunks) == 1 and len(qa) == 1, "one context per row expected"
        rows[kind] = {"sub_dataset": dataset_config["sub_dataset"], "chunks": chunks[0],
                      "questions": len(qa[0]), "context": creator.contexts[0],
                      "chunk_size": dataset_config["chunk_size"],
                      "generation_max_length": dataset_config["generation_max_length"]}
    sh, mh = rows["sh"], rows["mh"]
    if [sha16(c) for c in sh["chunks"]] != [sha16(c) for c in mh["chunks"]] or sh["context"] != mh["context"]:
        sys.exit("single-hop and multi-hop contexts differ at this length: one master cannot serve both")
    chunks = sh["chunks"]
    print(f"length {args.length}: {len(chunks)} chunks, context {len(sh['context']):,} chars, "
          f"sha256 {hashlib.sha256(sh['context'].encode()).hexdigest()[:16]}, "
          f"{sh['questions']} sh + {mh['questions']} mh questions", flush=True)

    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")  # AgentWrapper's choice for a non-gpt-4o model name
    adapter = KscopeAdapter(
        writer_mode="kscope", sub_dataset=f"factconsolidation_{args.length}", run_id=args.run_id,
        top_k=agent_config["retrieve_num"], maximum_context_bytes=agent_config["kscope_maximum_context_bytes"],
        unit_tokens=agent_config["kscope_unit_tokens"], unit_sentences=agent_config["kscope_unit_sentences"],
        reader_max_completion_tokens=agent_config["kscope_reader_max_completion_tokens"],
        generation_max_length=sh["generation_max_length"], profile=agent_config["kscope_profile"],
        agent_save_folder=str(run_dir / "no-agent-save-folder"), tokenizer=tokenizer,
        writer_workers=args.writer_workers, vault_role="master")
    root = adapter.vault.root
    (run_dir / "vault.txt").write_text(str(root) + "\n")
    print(f"master vault {root}\nwriter workers {adapter.writer_workers}, prompt sha {writer.prompt_sha()}, "
          f"llm timeout {os.environ.get('BENCH_LLM_TIMEOUT')}", flush=True)

    started = time.time()
    per_chunk = []
    for index, chunk in enumerate(chunks):
        chunk_started = time.time()
        adapter.ingest(chunk)
        per_chunk.append(round(time.time() - chunk_started, 3))
        c = adapter.counters
        done = index + 1
        eta = (time.time() - started) / done * (len(chunks) - done)
        print(f"chunk {done}/{len(chunks)} in {per_chunk[-1]:.0f}s | units {c['units']} accepted {c['accepted']} "
              f"refused {c['refused_first_try']} fallback {c['fallback_resends']} lost {c['lost_units']} "
              f"extraction_failures {c['extraction_failures']} writer_errors {c['writer_errors']} "
              f"clamped {c['clamped_units']} call_failures {c['writer_call_failures']} | "
              f"elapsed {(time.time() - started) / 60:.1f} min, eta {eta / 60:.1f} min", flush=True)
    wall = time.time() - started

    # Per-unit facts from the adapter's own log (the source the checker reads too).
    units_logged = {}
    resends = {}
    for line in adapter.log_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("instance") != adapter.instance:
            continue
        if row["event"] == "unit":
            units_logged[(row["chunk_index"], row["unit_index"])] = row
        elif row["event"] == "fallback_resend":
            resends[(row["chunk_index"], row["unit_index"])] = row
    chunk_entries = []
    for index, chunk in enumerate(chunks):
        entries = []
        ui = 0
        while (index, ui) in units_logged:
            u = units_logged[(index, ui)]
            entries.append({"index": ui, "sha": u["unit_sha"], "tokens": u["unit_tokens"], "sentences": u["unit_sentences"],
                            "memory_id": adapter.unit_memory.get((index, ui)),
                            "fallback": bool(u["extraction_failed"] or (index, ui) in resends),
                            "refused_first_try": (index, ui) in resends,
                            "facts": u["facts"], "entities": u["entities"], "clamped": u["clamped"]})
            ui += 1
        chunk_entries.append({"index": index, "sha": sha16(chunk),
                              "tokens": len(tokenizer.encode(chunk, disallowed_special=())), "units": entries})
    n_units = sum(len(c["units"]) for c in chunk_entries)
    if n_units != adapter.counters["units"]:
        sys.exit(f"manifest lists {n_units} units, the adapter counted {adapter.counters['units']}")

    c = adapter.counters
    facts_lost_units = c["extraction_failures"] + c["refused_first_try"]
    gate = {"units": c["units"], "refused_first_try": c["refused_first_try"], "extraction_failures": c["extraction_failures"],
            "writer_errors": c["writer_errors"], "lost_units": c["lost_units"], "clamped_units": c["clamped_units"],
            "facts_lost_units": facts_lost_units, "refusal_rate": c["refused_first_try"] / max(1, c["units"]),
            "facts_lost_rate": facts_lost_units / max(1, c["units"]), "threshold": LOST_FACTS_THRESHOLD,
            "pass": facts_lost_units / max(1, c["units"]) <= LOST_FACTS_THRESHOLD and c["lost_units"] == 0}

    seal = tree_sha(root)
    identity = json.loads((root.parent / f"{root.name}-identity.json").read_text())
    manifest = {
        "schema": "mab-master-manifest.v1", "run_id": args.run_id, "length": args.length,
        "sub_datasets": [sh["sub_dataset"], mh["sub_dataset"]],
        "context_sha256": hashlib.sha256(sh["context"].encode()).hexdigest(), "context_chars": len(sh["context"]),
        "chunk_size": sh["chunk_size"], "chunks": chunk_entries,
        "unit_rule": {"unit_tokens": agent_config["kscope_unit_tokens"], "unit_sentences": agent_config["kscope_unit_sentences"]},
        "writer": {"prompt_sha": writer.prompt_sha(), "workers": adapter.writer_workers, "deployment": llm.DEPLOYMENT,
                   "llm_timeout": os.environ.get("BENCH_LLM_TIMEOUT")},
        "kscope": {**kscope_io.check_binary(), "profile": agent_config["kscope_profile"], "binary": str(kscope_io.KSCOPE)},
        "vault": {"root": str(root), "identity": identity},
        "seal": seal, "counters": dict(c),
        "timing": {"ingest_wall_seconds": round(wall, 3), "adapter_ingest_seconds": round(adapter.ingest_seconds, 3),
                   "per_chunk_seconds": per_chunk},
        "gate": gate, "adapter_log": str(adapter.log_path), "instance": adapter.instance,
        "created_ts": round(time.time(), 3),
    }
    manifest_path = root.parent / f"{root.name}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    (run_dir / "manifest.txt").write_text(str(manifest_path) + "\n")
    print(f"\nsealed {root}\n  seal {seal[:16]}  units {c['units']} accepted {c['accepted']} "
          f"| refused {c['refused_first_try']} extraction_failures {c['extraction_failures']} lost {c['lost_units']} "
          f"clamped {c['clamped_units']} | facts-lost rate {gate['facts_lost_rate']:.2%} "
          f"({'PASS' if gate['pass'] else 'FAIL'} at {LOST_FACTS_THRESHOLD:.0%})\n  wall {wall / 60:.1f} min\n"
          f"  manifest {manifest_path}\n  {llm.report()}", flush=True)
    return 0 if gate["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
