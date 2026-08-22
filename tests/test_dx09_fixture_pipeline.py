"""Deterministic tests for the credential-free DX-09 fixture lane."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from kbench.benchmarks.beam import fixture
from kbench.config import REPO_ROOT
from kbench.kaleidoscope import KaleidoscopeError, sha256_bytes, sha256_file


@dataclass
class FixtureEngine:
    profiles: dict[str, dict] = field(default_factory=dict)
    memories: dict[str, list[dict]] = field(default_factory=dict)
    ranked_calls: int = 0

    def run(self, _executable: Path, args: list[str], stdin: str | None) -> tuple[int, str, str]:
        payload = json.loads(stdin) if stdin else None
        if args == ["model"]:
            return (
                0,
                json.dumps(
                    {
                        "status": "bundled",
                        "model": {"name": "fixture-model", "dtype": "f32"},
                    }
                ),
                "",
            )
        if args == ["profile", "list"]:
            return 0, json.dumps({"profiles": sorted(self.profiles)}), ""
        if args[:2] == ["profile", "show"]:
            return 0, json.dumps(self.profiles[args[2]]), ""
        if args and args[0] == "init-profile":
            profile, root = args[1], args[2]
            self.profiles[profile] = {"name": profile, "root": root}
            self.memories[profile] = []
            return 0, json.dumps({"status": "initialized"}), ""
        if args[:2] != ["call", "--profile"]:
            return 1, "", "unsupported"

        profile, operation = args[2], args[3]
        if operation == "ontology":
            return (
                0,
                json.dumps({"declarable": {"memory_types": ["runtime_fixture"]}}),
                "",
            )
        if operation == "remember":
            results = []
            for item in payload["items"]:
                number = len(self.memories[profile]) + 1
                memory_id = f"mem_{profile[-4:]}_{number:02d}"
                stored = {
                    "memory_id": memory_id,
                    "content_md": item["content_md"],
                    "memory_type": item["semantic_delta"]["memory_type"],
                }
                self.memories[profile].append(stored)
                results.append({"status": "created", "memory_id": memory_id})
            return 0, json.dumps({"results": results}), ""
        if operation == "search" and "memory_id" in payload:
            record = next(
                item for item in self.memories[profile] if item["memory_id"] == payload["memory_id"]
            )
            return 0, json.dumps(record), ""
        if operation == "search" and "query" in payload:
            self.ranked_calls += 1
            selected = self.memories[profile][: payload["top_k"]]
            return (
                0,
                json.dumps(
                    {
                        "context_text": "\n".join(item["content_md"] for item in selected),
                        "selected_hits": [
                            {
                                "memory_id": item["memory_id"],
                                "rank": index,
                                "score": 1.0 / index,
                            }
                            for index, item in enumerate(selected, 1)
                        ],
                        "abstention": {"abstained": False},
                    }
                ),
                "",
            )
        return 1, "", "unsupported"


def _candidate_files(root: Path) -> tuple[Path, str, Path, str]:
    executable = root / "candidate"
    executable.write_bytes(b"fixture candidate\n")
    executable_sha256 = sha256_file(executable)
    contract = {
        "schema_version": "kaleidoscope.public-contract.v1",
        "product": {"version": "fixture"},
        "target": {"triple": "fixture-platform"},
        "executable": {"sha256": executable_sha256},
        "limits": {"remember_batch_items": 20},
        "mcp": {"tools": [{"name": "remember"}, {"name": "search"}]},
    }
    contract_path = root / "public-contract.json"
    contract_bytes = fixture._canonical_json(contract).encode()
    contract_path.write_bytes(contract_bytes)
    return executable, executable_sha256, contract_path, sha256_bytes(contract_bytes)


def test_fixture_pipeline_drives_every_phase_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, executable_sha256, contract, contract_sha256 = _candidate_files(tmp_path)
    engine = FixtureEngine()
    monkeypatch.setattr(fixture, "EXPECTED_CANDIDATE_SHA256", executable_sha256)
    monkeypatch.setattr(fixture, "EXPECTED_PUBLIC_CONTRACT_SHA256", contract_sha256)
    monkeypatch.setattr(fixture, "_isolated_runner", lambda _root: engine.run)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-fixture-artifacts")

    run_root = tmp_path / "external-run"
    evidence = fixture.execute(
        run_root=run_root,
        executable=executable,
        executable_sha256=executable_sha256,
        public_contract=contract,
        public_contract_sha256=contract_sha256,
    )

    assert evidence["status"] == "passed"
    assert evidence["mode"] == "local_synthetic_credential_free"
    assert evidence["candidate"]["signature_verified"] is False
    assert evidence["release_evidence_claimed"] is False
    assert evidence["performance_claimed"] is False
    assert evidence["production_comparable"] is False
    assert evidence["counts"] == {"conversations": 2, "memories": 3, "questions": 3}
    assert engine.ranked_calls == 3
    assert len(engine.profiles) == 2

    judgements = [
        json.loads(line) for line in (run_root / "judgements.jsonl").read_text().splitlines()
    ]
    assert all(record["passed"] for record in judgements)
    assert all("score" not in record and "performance_score" not in record for record in judgements)
    assert "not comparable to production" in (run_root / "report.md").read_text()
    for name in ("ingest.json", "answers.jsonl", "judgements.jsonl", "report.md", "evidence.json"):
        text = (run_root / name).read_text()
        assert str(tmp_path) not in text
        assert "must-not-reach-fixture-artifacts" not in text


def test_fixture_requires_fresh_external_root(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(KaleidoscopeError, match="already exists"):
        fixture._fresh_external_root(existing)
    with pytest.raises(KaleidoscopeError, match="outside the repository"):
        fixture._fresh_external_root(REPO_ROOT / "forbidden-fixture-output")


def test_fixture_refuses_any_candidate_other_than_the_frozen_pair(tmp_path: Path) -> None:
    executable, executable_sha256, contract, contract_sha256 = _candidate_files(tmp_path)
    with pytest.raises(KaleidoscopeError, match="frozen 988192"):
        fixture._candidate(
            executable=executable,
            executable_sha256=executable_sha256,
            public_contract=contract,
            public_contract_sha256=contract_sha256,
            runner=FixtureEngine().run,
        )
