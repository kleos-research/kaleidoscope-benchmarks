"""Credential-free DX-09 fixture pipeline against a real engine candidate.

This lane proves the benchmark plumbing without pretending that a three-question
synthetic corpus is BEAM.  It uses explicit semantic records instead of an LLM
extractor, one real ranked search per question, a label-driven deterministic
reader, and a boolean local judge.  It deliberately emits no performance score.

All mutable state must live outside the repository.  A new isolated platform
home keeps native profiles out of the developer's normal configuration, while a
fresh run root makes reuse impossible by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...config import REPO_ROOT
from ...kaleidoscope import (
    CALL_TIMEOUT_SECONDS,
    SAFE_ENVIRONMENT_KEYS,
    KaleidoscopeError,
    ReleaseCandidate,
    VaultPool,
    sha256_bytes,
    sha256_file,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "dx09-credential-free.json"
EVIDENCE_SCHEMA = "kbench.dx09-fixture-evidence.v1"
EXPECTED_CANDIDATE_SHA256 = "988192ac9677d5dd55a3642b2da493a0806bb860b5b3c0f509b37ddadee08825"
EXPECTED_PUBLIC_CONTRACT_SHA256 = "a2357ed6c00e3e143d08581590571447e31d24fd0e7d2466d28a211a0515c75e"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_canonical_json(value))


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(_canonical_json(value) for value in values))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _fresh_external_root(raw: Path | str) -> Path:
    path = Path(raw).expanduser().resolve()
    repository = REPO_ROOT.resolve()
    if _is_within(path, repository):
        raise KaleidoscopeError("fixture run root must be outside the repository")
    if path.exists():
        raise KaleidoscopeError("fixture run root already exists; supply a fresh path")
    path.mkdir(mode=0o700, parents=True)
    path.chmod(0o700)
    return path


def _isolated_runner(root: Path) -> Callable[[Path, list[str], str | None], tuple[int, str, str]]:
    home = root / "platform-home"
    xdg = root / "xdg"
    appdata = root / "appdata"
    temporary = root / "tmp"
    for directory in (home, xdg, appdata, temporary):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)

    environment = {
        key: os.environ[key]
        for key in SAFE_ENVIRONMENT_KEYS
        if key in os.environ
        and key not in {"APPDATA", "HOME", "LOCALAPPDATA", "TEMP", "TMP", "TMPDIR", "USERPROFILE"}
    }
    environment.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CONFIG_HOME": str(xdg),
            "APPDATA": str(appdata),
            "LOCALAPPDATA": str(appdata),
            "TEMP": str(temporary),
            "TMP": str(temporary),
            "TMPDIR": str(temporary),
        }
    )

    def run(executable: Path, args: list[str], stdin: str | None) -> tuple[int, str, str]:
        completed = subprocess.run(
            [str(executable), *args],
            input=stdin,
            capture_output=True,
            text=True,
            env=environment,
            timeout=CALL_TIMEOUT_SECONDS,
            check=False,
        )
        return completed.returncode, completed.stdout, completed.stderr

    return run


def _load_fixture() -> tuple[dict, str]:
    raw = FIXTURE_PATH.read_bytes()
    fixture = json.loads(raw)
    if fixture.get("schema_version") != "kbench.dx09-fixture-corpus.v1":
        raise KaleidoscopeError("unsupported DX-09 fixture corpus schema")
    conversations = fixture.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise KaleidoscopeError("DX-09 fixture corpus has no conversations")
    return fixture, sha256_bytes(raw)


def _candidate(
    *,
    executable: Path | str,
    executable_sha256: str,
    public_contract: Path | str,
    public_contract_sha256: str,
    runner,
) -> ReleaseCandidate:
    if executable_sha256 != EXPECTED_CANDIDATE_SHA256:
        raise KaleidoscopeError("DX-09 fixture requires the frozen 988192 candidate digest")
    if public_contract_sha256 != EXPECTED_PUBLIC_CONTRACT_SHA256:
        raise KaleidoscopeError("DX-09 fixture requires the frozen a2357 public-contract digest")
    return ReleaseCandidate.load(
        executable=executable,
        executable_sha256=executable_sha256,
        public_contract=public_contract,
        public_contract_sha256=public_contract_sha256,
        runner=runner,
    )


def _memory_item(record: dict, memory_type: str) -> dict:
    return {
        "content_md": record["content_md"],
        "semantic_delta": {
            "memory_type": memory_type,
            "title": record["title"],
            "entities": record["entities"],
            "facts": record["facts"],
            "evidence": [
                {
                    "kind": "synthetic_fixture",
                    "reference": record["fixture_id"],
                }
            ],
        },
    }


def _stable_hits(raw_hits: list[dict], runtime_to_fixture: dict[str, str]) -> list[dict]:
    hits = []
    for hit in raw_hits:
        memory_id = hit.get("memory_id")
        fixture_id = runtime_to_fixture.get(memory_id)
        if fixture_id is None:
            raise KaleidoscopeError("fixture search returned an unknown memory id")
        hits.append(
            {
                "fixture_id": fixture_id,
                "rank": hit.get("rank"),
                "score": hit.get("score"),
            }
        )
    return hits


def _stable_context(hits: list[dict], records: dict[str, dict]) -> str:
    sections = []
    for hit in hits:
        fixture_id = hit["fixture_id"]
        record = records[fixture_id]
        sections.extend(
            [
                f"Memory {hit['rank']} | fixture={fixture_id}",
                record["content_md"].rstrip(),
            ]
        )
    return "\n\n".join(sections) + "\n"


def _artifact_hashes(root: Path) -> dict[str, str]:
    names = ("ingest.json", "answers.jsonl", "judgements.jsonl", "report.md")
    return {name: sha256_file(root / name) for name in names}


def _assert_public_artifacts_are_path_free(
    root: Path,
    *,
    executable: Path,
    public_contract: Path,
) -> None:
    poisons = {
        str(root),
        str(REPO_ROOT.resolve()),
        str(executable.resolve()),
        str(public_contract.resolve()),
    }
    for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY"):
        value = os.environ.get(variable)
        if value:
            poisons.add(value)
    for path in root.iterdir():
        if path.name in {"platform-home", "xdg", "appdata", "tmp", "vaults"} or not path.is_file():
            continue
        text = path.read_text()
        for poison in poisons:
            if poison and poison in text:
                raise KaleidoscopeError(
                    f"public fixture artifact {path.name} contains private data"
                )


def _report(*, candidate: dict, checks: list[dict], counts: dict[str, int]) -> str:
    status = "PASS" if all(check["passed"] for check in checks) else "FAIL"
    lines = [
        "# DX-09 credential-free fixture pipeline",
        "",
        f"Functional status: **{status}**",
        "",
        "> This is a local, synthetic plumbing check. It is not a BEAM run, does not",
        "> emit a performance score, is not comparable to production, and is not release evidence.",
        "",
        "## Bound candidate",
        "",
        f"- executable SHA-256: `{candidate['executable_sha256']}`",
        f"- public-contract SHA-256: `{candidate['public_contract_sha256']}`",
        f"- signature verified: `{candidate['signature_verified']}`",
        "",
        "## Pipeline",
        "",
        f"- conversations: {counts['conversations']}",
        f"- memories written and addressed: {counts['memories']}",
        f"- ranked retrievals, deterministic answers, and local judgements: {counts['questions']}",
        "",
        "| check | result |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| `{check['question_id']}` | {'pass' if check['passed'] else 'fail'} |"
        for check in checks
    )
    lines.extend(
        [
            "",
            "## Promotion boundary",
            "",
            "Production signing, the licensed BEAM corpus, provider credentials, evaluator",
            "credentials, and a preregistered controlled run remain required before any",
            "performance or release claim.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(
    *,
    run_root: Path | str,
    executable: Path | str,
    executable_sha256: str,
    public_contract: Path | str,
    public_contract_sha256: str,
    top_k: int = 5,
    maximum_context_bytes: int = 32 * 1024,
) -> dict:
    """Run all credential-free fixture stages and return path-free evidence."""
    if top_k < 1 or maximum_context_bytes < 1:
        raise KaleidoscopeError("fixture search bounds must be positive")
    root = _fresh_external_root(run_root)
    executable_path = Path(executable).expanduser().resolve(strict=True)
    contract_path = Path(public_contract).expanduser().resolve(strict=True)
    candidate = _candidate(
        executable=executable_path,
        executable_sha256=executable_sha256,
        public_contract=contract_path,
        public_contract_sha256=public_contract_sha256,
        runner=_isolated_runner(root),
    )
    candidate.require_bundled_model()
    fixture, fixture_sha256 = _load_fixture()
    pool = VaultPool(
        root / "vaults" / candidate.acquisition_key,
        fixture["created_at"],
        candidate=candidate,
        profile_prefix=(
            f"dx09-fixture-{candidate.executable_sha256[:8]}-{candidate.public_contract_sha256[:8]}"
        ),
    )

    ingest_rows: list[dict] = []
    answer_rows: list[dict] = []
    judgement_rows: list[dict] = []
    profiles: set[str] = set()

    for conversation in fixture["conversations"]:
        vault = pool.for_conversation(conversation["conversation_id"])
        profiles.add(vault.profile)
        memory_types = vault.memory_types()
        memory_type = memory_types[0]
        memories = conversation["memories"]
        response = vault.call(
            "remember",
            {
                "mode": "create",
                "items": [_memory_item(record, memory_type) for record in memories],
            },
        )
        results = response.get("results") or []
        if len(results) != len(memories):
            raise KaleidoscopeError("fixture remember returned the wrong result count")
        logical_to_runtime: dict[str, str] = {}
        records_by_fixture = {record["fixture_id"]: record for record in memories}
        for record, result in zip(memories, results):
            if result.get("status") in {"refused", "rejected"}:
                raise KaleidoscopeError("fixture memory was refused")
            memory_id = result.get("memory_id")
            if not isinstance(memory_id, str) or not memory_id.startswith("mem_"):
                raise KaleidoscopeError("fixture remember returned no memory id")
            addressed = vault.addressed_search(memory_id)
            addressed_verified = addressed.get("content_md") == record["content_md"]
            if not addressed_verified:
                raise KaleidoscopeError("fixture addressed read did not match the write")
            logical_to_runtime[record["fixture_id"]] = memory_id
            ingest_rows.append(
                {
                    "conversation_id": conversation["conversation_id"],
                    "fixture_id": record["fixture_id"],
                    "memory_type": memory_type,
                    "addressed_verified": True,
                }
            )

        runtime_to_logical = {runtime: logical for logical, runtime in logical_to_runtime.items()}
        for question in conversation["questions"]:
            searched = vault.ranked_search(
                question["query"],
                top_k=top_k,
                maximum_context_bytes=maximum_context_bytes,
            )
            hits = _stable_hits(searched["selected_hits"], runtime_to_logical)
            retrieved_ids = {hit["fixture_id"] for hit in hits}
            required_ids = question["required_memories"]
            retrieval_satisfied = set(required_ids).issubset(retrieved_ids)
            hypothesis = question["expected_answer"] if retrieval_satisfied else ""
            answer = {
                "conversation_id": conversation["conversation_id"],
                "question_id": question["question_id"],
                "query": question["query"],
                "required_fixture_ids": required_ids,
                "selected_hits": hits,
                "context_md": _stable_context(hits, records_by_fixture),
                "retrieval_satisfied": retrieval_satisfied,
                "hypothesis": hypothesis,
                "reader": "deterministic_fixture_labels",
            }
            answer_rows.append(answer)
            checks = {
                "required_memories_retrieved": retrieval_satisfied,
                "context_returned": bool(answer["context_md"]),
                "answer_exact": hypothesis == question["expected_answer"],
                "single_ranked_search": True,
            }
            judgement_rows.append(
                {
                    "conversation_id": conversation["conversation_id"],
                    "question_id": question["question_id"],
                    "method": "deterministic_fixture_boolean_checks",
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )

    if len(profiles) != len(fixture["conversations"]):
        raise KaleidoscopeError("fixture conversations did not receive isolated profiles")

    _write_json(
        root / "ingest.json",
        {
            "schema_version": "kbench.dx09-fixture-ingest.v1",
            "candidate": candidate.evidence,
            "fixture_corpus_sha256": fixture_sha256,
            "records": ingest_rows,
            "release_evidence_claimed": False,
        },
    )
    _write_jsonl(root / "answers.jsonl", answer_rows)
    _write_jsonl(root / "judgements.jsonl", judgement_rows)
    counts = {
        "conversations": len(fixture["conversations"]),
        "memories": len(ingest_rows),
        "questions": len(answer_rows),
    }
    (root / "report.md").write_text(
        _report(candidate=candidate.evidence, checks=judgement_rows, counts=counts)
    )
    _assert_public_artifacts_are_path_free(
        root,
        executable=executable_path,
        public_contract=contract_path,
    )

    all_passed = all(record["passed"] for record in judgement_rows)
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "status": "passed" if all_passed else "failed",
        "mode": "local_synthetic_credential_free",
        "candidate": candidate.evidence,
        "fixture_corpus_sha256": fixture_sha256,
        "search": {
            "top_k": top_k,
            "maximum_context_bytes": maximum_context_bytes,
            "calls_per_question": 1,
        },
        "counts": counts,
        "checks": {
            "fresh_external_run_root": True,
            "isolated_profile_per_conversation": True,
            "all_writes_addressed": all(row["addressed_verified"] for row in ingest_rows),
            "all_fixture_judgements_passed": all_passed,
            "public_artifacts_path_and_secret_scan": True,
        },
        "artifacts": _artifact_hashes(root),
        "signature_verified": False,
        "release_evidence_claimed": False,
        "performance_claimed": False,
        "production_comparable": False,
        "requires_for_release": [
            "production signing and independent signature verification",
            "licensed BEAM corpus",
            "provider and evaluator credentials",
            "preregistered controlled BEAM run",
        ],
    }
    _write_json(root / "evidence.json", evidence)
    _assert_public_artifacts_are_path_free(
        root,
        executable=executable_path,
        public_contract=contract_path,
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kbench-dx09-fixture",
        description="Run the credential-free DX-09 fixture pipeline against a frozen candidate.",
    )
    parser.add_argument(
        "--run-root", required=True, help="fresh output/vault root outside the repo"
    )
    parser.add_argument("--candidate", required=True, help="path to the immutable kscope candidate")
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--public-contract", required=True)
    parser.add_argument("--public-contract-sha256", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--maximum-context-bytes", type=int, default=32 * 1024)
    args = parser.parse_args(argv)
    try:
        evidence = execute(
            run_root=args.run_root,
            executable=args.candidate,
            executable_sha256=args.candidate_sha256,
            public_contract=args.public_contract,
            public_contract_sha256=args.public_contract_sha256,
            top_k=args.top_k,
            maximum_context_bytes=args.maximum_context_bytes,
        )
    except (KaleidoscopeError, FileNotFoundError) as exc:
        print(f"DX-09 fixture refused: {exc}", file=sys.stderr)
        return 2
    print(_canonical_json(evidence), end="")
    return 0 if evidence["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
