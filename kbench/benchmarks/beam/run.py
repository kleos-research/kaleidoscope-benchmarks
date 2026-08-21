"""BEAM benchmark CLI.

Four phases, each runnable on its own and each writing its output to disk:

    ingest    build the memory            -> results/beam/ingest.json
    answer    retrieve and answer         -> results/beam/answers.jsonl
    judge     score against the rubric    -> results/beam/scores.jsonl
    report    tables                      -> stdout / results/beam/report.md

`all` runs them in order. Nothing downstream reruns anything upstream, so a
judge experiment costs judge calls only, and a crash in phase 3 never loses
phase 2's answers.

    python -m kbench.benchmarks.beam.run all --tier 100K
    python -m kbench.benchmarks.beam.run judge --judge-model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ...config import settings
from ...kaleidoscope import KaleidoscopeError, ReleaseCandidate, sha256_file
from . import answer as answer_phase
from . import ingest as ingest_phase
from . import judge as judge_phase
from . import metrics, report
from .dataset import (
    Conversation,
    assert_conversations_are_isolated,
    load_tier,
    render,
)


def _paths(tier: str) -> dict[str, Path]:
    out = settings.results_dir / "beam" / tier
    return {
        "out": out,
        "vaults": settings.data_dir / "vaults" / tier,
        "cache": settings.data_dir / "extraction-cache" / tier,
        "ingest": out / "ingest.json",
        "answers": out / "answers.jsonl",
        "answers_meta": out / "answers.jsonl.meta.json",
        "scores": out / "scores.jsonl",
        "scores_meta": out / "scores.jsonl.meta.json",
        "report": out / "report.md",
    }


def _load(tier: str) -> list[Conversation]:
    conversations = load_tier(tier, settings.data_dir)
    facts = assert_conversations_are_isolated(conversations)
    print(
        f"BEAM {tier}: {facts['conversations']} conversations, {facts['messages']} messages, "
        f"{facts['questions']} questions ({facts['questions_with_evidence']} labelled)"
    )
    if facts["ids_reused_across_conversations"]:
        print(
            f"  note: {facts['ids_reused_across_conversations']} message ids are reused across "
            f"conversations — evidence scoring is per-conversation, as it must be"
        )
    return conversations


def _candidate(args) -> ReleaseCandidate:
    try:
        return ReleaseCandidate.load(
            executable=args.candidate,
            executable_sha256=args.candidate_sha256,
            public_contract=args.public_contract,
            public_contract_sha256=args.public_contract_sha256,
        )
    except (KaleidoscopeError, FileNotFoundError) as exc:
        raise SystemExit(f"release candidate refused: {exc}") from exc


def _profile_prefix(args, candidate: ReleaseCandidate) -> str:
    if args.profile_prefix:
        return args.profile_prefix
    tier = "".join(character.lower() for character in args.tier if character.isalnum())
    return f"beam-{tier}-{candidate.executable_sha256[:8]}"


def _require_matching_candidate(path: Path, candidate: ReleaseCandidate) -> None:
    if not path.exists():
        raise SystemExit(f"{path} not found — run the preceding candidate-bound phase first")
    record = json.loads(path.read_text())
    evidence = record.get("candidate") or {}
    for key in ("executable_sha256", "public_contract_sha256"):
        if evidence.get(key) != candidate.evidence[key]:
            raise SystemExit(f"{path.name} is bound to a different {key}")


def cmd_ingest(args) -> int:
    candidate = _candidate(args)
    candidate.require_bundled_model()
    conversations = _load(args.tier)
    paths = _paths(args.tier)
    reports, spend = ingest_phase.run(
        conversations,
        vault_root=paths["vaults"] / candidate.executable_sha256[:16],
        cache_root=paths["cache"] / candidate.public_contract_sha256[:16],
        candidate=candidate,
        profile_prefix=_profile_prefix(args, candidate),
        chunk_size=args.chunk_size,
        workers=args.conversation_workers,
    )
    ingest_phase.write_report(reports, spend, paths["ingest"], candidate=candidate)
    written = sum(r.written for r in reports)
    print(f"\n{written} memories written. {spend.line()}")
    print(f"  -> {paths['ingest']}")
    return 0


def cmd_answer(args) -> int:
    candidate = _candidate(args)
    model = candidate.require_bundled_model()
    print(f"encoder: {model['model']['name']} ({model['model']['dtype']})")
    conversations = _load(args.tier)
    paths = _paths(args.tier)
    _require_matching_candidate(paths["ingest"], candidate)
    answers, spend = answer_phase.run(
        conversations,
        vault_root=paths["vaults"] / candidate.executable_sha256[:16],
        out_path=paths["answers"],
        candidate=candidate,
        profile_prefix=_profile_prefix(args, candidate),
        top_k=args.top_k,
        maximum_context_bytes=args.maximum_context_bytes,
        conversation_workers=args.conversation_workers,
        question_workers=args.question_workers,
    )

    evidence = {
        c.conversation_id: metrics.ConversationEvidence(c.messages, render) for c in conversations
    }
    summary = metrics.summarise([a.as_dict() for a in answers], evidence)
    print()
    print(summary.render())
    print(f"\n{spend.line()}")
    print(f"  -> {paths['answers']}")
    return 0


def cmd_judge(args) -> int:
    conversations = _load(args.tier)
    paths = _paths(args.tier)
    if not paths["answers"].exists():
        raise SystemExit(f"{paths['answers']} not found — run the answer phase first")
    if not paths["answers_meta"].exists():
        raise SystemExit(
            f"{paths['answers_meta']} not found — answers are not candidate-bound"
        )
    answer_inputs = json.loads(paths["answers_meta"].read_text())
    if answer_inputs.get("answers_sha256") != sha256_file(paths["answers"]):
        raise SystemExit("answers changed after their input manifest was written")

    rubrics = {(c.conversation_id, q.index): q.rubric for c in conversations for q in c.questions}
    references = {
        (c.conversation_id, q.index): list(q.raw.get("reasoning_steps") or [])
        for c in conversations
        for q in c.questions
    }
    scores, spend = judge_phase.run(
        paths["answers"],
        rubrics=rubrics,
        references=references,
        out_path=paths["scores"],
        workers=args.judge_workers,
    )
    graded = [s for s in scores if s.score is not None]
    paths["scores_meta"].write_text(
        json.dumps(
            {
                "schema_version": "kbench.judge-inputs.v1",
                "candidate": answer_inputs.get("candidate"),
                "answers_sha256": sha256_file(paths["answers"]),
                "scores_sha256": sha256_file(paths["scores"]),
                "judge_model": settings.models.judge,
                "release_evidence_claimed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\n{len(graded)}/{len(scores)} scored. {spend.line()}")
    print(f"  -> {paths['scores']}")
    return 0


def cmd_report(args) -> int:
    conversations = _load(args.tier)
    paths = _paths(args.tier)
    if not paths["answers_meta"].exists():
        raise SystemExit(
            f"{paths['answers_meta']} not found — answers are not candidate-bound"
        )
    answer_inputs = json.loads(paths["answers_meta"].read_text())
    if answer_inputs.get("answers_sha256") != sha256_file(paths["answers"]):
        raise SystemExit("answers changed after their input manifest was written")
    if paths["ingest"].exists():
        ingest_inputs = json.loads(paths["ingest"].read_text())
        for key in ("executable_sha256", "public_contract_sha256"):
            if (answer_inputs.get("candidate") or {}).get(key) != (
                ingest_inputs.get("candidate") or {}
            ).get(key):
                raise SystemExit(f"ingest and answer use different {key}")
    if paths["scores"].exists():
        if not paths["scores_meta"].exists():
            raise SystemExit(f"{paths['scores_meta']} not found — scores are not candidate-bound")
        score_inputs = json.loads(paths["scores_meta"].read_text())
        for key in ("executable_sha256", "public_contract_sha256"):
            if (score_inputs.get("candidate") or {}).get(key) != (
                answer_inputs.get("candidate") or {}
            ).get(key):
                raise SystemExit(f"answers and scores use different {key}")
        if score_inputs.get("answers_sha256") != sha256_file(paths["answers"]):
            raise SystemExit("scores were produced from different answer bytes")
        if score_inputs.get("scores_sha256") != sha256_file(paths["scores"]):
            raise SystemExit("scores changed after their input manifest was written")
    text = report.build(
        conversations,
        answers_path=paths["answers"],
        scores_path=paths["scores"] if paths["scores"].exists() else None,
        ingest_path=paths["ingest"] if paths["ingest"].exists() else None,
    )
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(text)
    print(text)
    print(f"\n  -> {paths['report']}")
    return 0


def cmd_all(args) -> int:
    for step in (cmd_ingest, cmd_answer, cmd_judge, cmd_report):
        code = step(args)
        if code != 0:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kbench.benchmarks.beam.run",
        description="Run the BEAM benchmark against Kaleidoscope.",
    )
    parser.add_argument("phase", choices=["ingest", "answer", "judge", "report", "all"])
    parser.add_argument("--tier", default="100K", help="BEAM tier: 100K, 500K, 1M, 10M")
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.retrieval.top_k,
        help="explicit ranked-search depth; compare only runs with the same value",
    )
    parser.add_argument(
        "--maximum-context-bytes",
        type=int,
        default=settings.retrieval.maximum_context_bytes,
        help="explicit ranked-search context budget",
    )
    parser.add_argument("--candidate", help="path to the immutable kscope candidate")
    parser.add_argument("--candidate-sha256", help="expected candidate SHA-256")
    parser.add_argument("--public-contract", help="generated public-contract JSON")
    parser.add_argument("--public-contract-sha256", help="expected public-contract SHA-256")
    parser.add_argument(
        "--profile-prefix",
        default=None,
        help="portable native profile prefix; default binds tier and candidate digest",
    )
    parser.add_argument("--chunk-size", type=int, default=2, help="messages per extraction")
    parser.add_argument(
        "--conversation-workers",
        type=int,
        default=settings.concurrency.conversations,
        help="conversations in flight; they are independent stores",
    )
    parser.add_argument(
        "--question-workers",
        type=int,
        default=settings.concurrency.questions,
        help="questions in flight per conversation",
    )
    parser.add_argument("--judge-workers", type=int, default=settings.retrieval.judging)
    parser.add_argument("--judge-model", default=None, help="override the judge for this run")
    args = parser.parse_args(argv)

    if args.judge_model:
        # Overriding here rather than in the environment keeps the override
        # visible in the command that produced the scores.
        object.__setattr__(settings.models, "judge", args.judge_model)

    return {
        "ingest": cmd_ingest,
        "answer": cmd_answer,
        "judge": cmd_judge,
        "report": cmd_report,
        "all": cmd_all,
    }[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
