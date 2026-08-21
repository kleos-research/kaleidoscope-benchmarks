"""Deterministic release-boundary and profile-first contract tests."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from kbench.benchmarks.beam import answer, extract, ingest, run
from kbench.benchmarks.beam.dataset import Conversation, Question
from kbench.kaleidoscope import (
    KaleidoscopeError,
    ReleaseCandidate,
    VaultPool,
    _subprocess_runner,
    sha256_bytes,
    sha256_file,
)
from kbench.llm import Completion, Spend, Usage


@dataclass
class FakeEngine:
    """A deterministic process boundary with disk-like state across clients."""

    profiles: dict[str, dict] = field(default_factory=dict)
    memories: dict[str, list[dict]] = field(default_factory=dict)
    calls: list[tuple[list[str], dict | None]] = field(default_factory=list)

    def run(self, _executable: Path, args: list[str], stdin: str | None) -> tuple[int, str, str]:
        payload = json.loads(stdin) if stdin else None
        self.calls.append((list(args), payload))
        if args == ["model"]:
            return 0, json.dumps({"status": "bundled", "model": {"name": "fake"}}), ""
        if args == ["profile", "list"]:
            return 0, json.dumps({"version": 1, "profiles": sorted(self.profiles)}), ""
        if args[:2] == ["profile", "show"]:
            profile = args[2]
            if profile not in self.profiles:
                return 1, "", "profile does not exist"
            return 0, json.dumps(self.profiles[profile]), ""
        if args and args[0] == "init-profile":
            profile, root = args[1], args[2]
            if profile in self.profiles:
                return 1, "", "profile already exists"
            self.profiles[profile] = {
                "version": 1,
                "name": profile,
                "root": root,
                "workspace_id": "private-workspace-coordinate",
                "principal_id": "private-principal-coordinate",
                "journal": "private-journal-coordinate",
                "durability": "process-local",
            }
            self.memories[profile] = []
            return 0, json.dumps({"status": "initialized", "version": 1}), ""
        if args[:2] != ["call", "--profile"] or len(args) != 4:
            return 1, "", "unsupported fake command"

        profile, operation = args[2], args[3]
        if profile not in self.profiles:
            return 1, "", "missing profile"
        if operation == "ontology":
            return (
                0,
                json.dumps(
                    {
                        "status": "empty",
                        "declarable": {"memory_types": ["runtime_decision", "runtime_note"]},
                    }
                ),
                "",
            )
        if operation == "remember":
            results = []
            for item in payload["items"]:
                index = len(self.memories[profile]) + 1
                memory_id = f"mem_fake_{index:04d}"
                record = {
                    "memory_id": memory_id,
                    "version_id": f"ver_fake_{index:04d}",
                    "memory_type": item["semantic_delta"]["memory_type"],
                    "content_md": item["content_md"],
                }
                self.memories[profile].append(record)
                results.append({"status": "created", **record})
            return 0, json.dumps({"status": "completed", "results": results}), ""
        if operation == "search" and "memory_id" in payload:
            found = next(
                (
                    item
                    for item in self.memories[profile]
                    if item["memory_id"] == payload["memory_id"]
                ),
                None,
            )
            return (0, json.dumps(found), "") if found else (1, "", "not found")
        if operation == "search" and "query" in payload:
            hits = [
                {"memory_id": item["memory_id"], "rank": rank, "score": 1.0 / rank}
                for rank, item in enumerate(self.memories[profile][: payload["top_k"]], 1)
            ]
            context = "\n".join(item["content_md"] for item in self.memories[profile])
            return (
                0,
                json.dumps(
                    {
                        "status": "compiled",
                        "context_text": context,
                        "selected_hits": hits,
                        "abstention": {"abstained": not bool(hits)},
                    }
                ),
                "",
            )
        return 1, "", "unsupported fake operation"


def candidate_fixture(tmp_path: Path, engine: FakeEngine) -> ReleaseCandidate:
    executable = tmp_path / "kscope"
    executable.write_bytes(b"deterministic fake engine candidate\n")
    executable_sha = sha256_file(executable)
    contract = {
        "schema_version": "kaleidoscope.public-contract.v1",
        "product": {"version": "1.2.3"},
        "target": {"triple": "fake-test-platform"},
        "executable": {"sha256": executable_sha},
        "limits": {"remember_batch_items": 20},
        "retired_operations": {"agent_tools": [], "non_product": []},
        "mcp": {"tools": [{"name": "remember"}, {"name": "search"}]},
    }
    contract_path = tmp_path / "kaleidoscope-public-contract.json"
    contract_bytes = (json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n").encode()
    contract_path.write_bytes(contract_bytes)
    return ReleaseCandidate.load(
        executable=executable,
        executable_sha256=executable_sha,
        public_contract=contract_path,
        public_contract_sha256=sha256_bytes(contract_bytes),
        runner=engine.run,
    )


def test_candidate_and_contract_digests_are_mandatory_and_exact(tmp_path: Path):
    engine = FakeEngine()
    candidate = candidate_fixture(tmp_path, engine)
    assert candidate.evidence["signature_verified"] is False
    assert candidate.acquisition_key == (
        f"{candidate.executable_sha256[:12]}-{candidate.public_contract_sha256[:12]}"
    )

    with pytest.raises(KaleidoscopeError, match="candidate digest"):
        ReleaseCandidate.load(
            executable=candidate.executable,
            executable_sha256=None,
            public_contract=tmp_path / "kaleidoscope-public-contract.json",
            public_contract_sha256=candidate.public_contract_sha256,
            runner=engine.run,
        )
    with pytest.raises(KaleidoscopeError, match="candidate digest mismatch"):
        ReleaseCandidate.load(
            executable=candidate.executable,
            executable_sha256="0" * 64,
            public_contract=tmp_path / "kaleidoscope-public-contract.json",
            public_contract_sha256=candidate.public_contract_sha256,
            runner=engine.run,
        )
    with pytest.raises(KaleidoscopeError, match="public-contract digest mismatch"):
        ReleaseCandidate.load(
            executable=candidate.executable,
            executable_sha256=candidate.executable_sha256,
            public_contract=tmp_path / "kaleidoscope-public-contract.json",
            public_contract_sha256="0" * 64,
            runner=engine.run,
        )
    contract_path = tmp_path / "kaleidoscope-public-contract.json"
    wrong_contract = json.loads(contract_path.read_text())
    wrong_contract["executable"]["sha256"] = "f" * 64
    wrong_bytes = (
        json.dumps(wrong_contract, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    contract_path.write_bytes(wrong_bytes)
    with pytest.raises(KaleidoscopeError, match="different candidate"):
        ReleaseCandidate.load(
            executable=candidate.executable,
            executable_sha256=candidate.executable_sha256,
            public_contract=contract_path,
            public_contract_sha256=sha256_bytes(wrong_bytes),
            runner=engine.run,
        )
    candidate.executable.write_bytes(b"changed after verification\n")
    with pytest.raises(KaleidoscopeError, match="changed after verification"):
        candidate.require_bundled_model()


def test_candidate_process_receives_only_closed_non_secret_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, str] = {}

    def fake_run(*_args, **kwargs):
        captured.update(kwargs["env"])
        return subprocess.CompletedProcess([], 0, "{}", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("HOME", "/safe/config-home")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret-canary")
    monkeypatch.setenv("KSCOPE_ROOT", "/private/vault-coordinate-canary")
    monkeypatch.setenv("KSCOPE_WORKSPACE", "private-workspace-canary")

    assert _subprocess_runner(Path("/absolute/kscope"), ["model"], None) == (0, "{}", "")
    assert captured["HOME"] == "/safe/config-home"
    assert "OPENAI_API_KEY" not in captured
    assert "KSCOPE_ROOT" not in captured
    assert "KSCOPE_WORKSPACE" not in captured


def test_engine_phase_refuses_absent_candidate_before_dataset_or_vault_work():
    with pytest.raises(SystemExit, match="candidate executable is required"):
        run.main(["ingest", "--tier", "100K"])


def test_phase_inputs_refuse_a_different_candidate_digest(tmp_path: Path):
    engine = FakeEngine()
    candidate = candidate_fixture(tmp_path, engine)
    report = tmp_path / "ingest.json"
    report.write_text(json.dumps({"candidate": candidate.evidence}))
    run._require_matching_candidate(report, candidate)

    mismatched = json.loads(report.read_text())
    mismatched["candidate"]["executable_sha256"] = "0" * 64
    report.write_text(json.dumps(mismatched))
    with pytest.raises(SystemExit, match="different executable_sha256"):
        run._require_matching_candidate(report, candidate)


def test_one_profile_per_conversation_and_restart_persistence(tmp_path: Path):
    engine = FakeEngine()
    candidate = candidate_fixture(tmp_path, engine)
    vault_root = tmp_path / "private-vaults"
    first = VaultPool(
        vault_root,
        "2026-01-01T00:00:00Z",
        candidate=candidate,
        profile_prefix="beam-test",
    )
    conversation = first.for_conversation("conversation-a")
    assert first.for_conversation("conversation-a") is conversation
    other = first.for_conversation("conversation-b")
    assert other.profile != conversation.profile
    assert len(engine.profiles) == 2

    created = conversation.call(
        "remember",
        {
            "mode": "create",
            "items": [
                {
                    "content_md": "# Persisted memory\n\nIt survives a client restart.\n",
                    "semantic_delta": {
                        "memory_type": "runtime_note",
                        "title": "Persisted memory",
                        "entities": [
                            {
                                "n": "client restart",
                                "kind": "event",
                                "is": "a new benchmark controller process",
                            },
                            {
                                "n": "persisted memory",
                                "kind": "artifact",
                                "is": "memory stored by the fake engine",
                            },
                        ],
                        "facts": [
                            {
                                "subject": "persisted memory",
                                "predicate": "survives",
                                "object": "client restart",
                                "mode": "fact",
                                "basis": "stated",
                                "confidence": 1.0,
                            }
                        ],
                    },
                }
            ],
        },
    )
    memory_id = created["results"][0]["memory_id"]

    restarted = VaultPool(
        vault_root,
        "2026-01-01T00:00:00Z",
        candidate=candidate,
        profile_prefix="beam-test",
    )
    reopened = restarted.for_conversation("conversation-a")
    assert len(engine.profiles) == 2
    assert reopened.addressed_search(memory_id)["content_md"].startswith("# Persisted")

    profile_calls = [args for args, _ in engine.calls if args[:2] == ["call", "--profile"]]
    assert profile_calls
    assert all(len(args) == 4 for args in profile_calls)


def test_ranked_acquisition_searches_once_and_public_metadata_has_no_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    engine = FakeEngine()
    candidate = candidate_fixture(tmp_path, engine)
    pools = VaultPool(
        tmp_path / "private-vaults",
        "2026-01-01T00:00:00Z",
        candidate=candidate,
        profile_prefix="beam-test",
    )
    vault = pools.for_conversation("7")
    vault.call(
        "remember",
        {
            "mode": "create",
            "items": [
                {
                    "content_md": "# Answer\n\nThe launch color is indigo.\n",
                    "semantic_delta": {
                        "memory_type": "runtime_note",
                        "title": "Answer",
                        "entities": [
                            {
                                "n": "launch",
                                "kind": "event",
                                "is": "the launch asked about by the test question",
                            },
                            {"n": "indigo", "kind": "concept", "is": "the launch color"},
                        ],
                        "facts": [
                            {
                                "subject": "launch",
                                "predicate": "uses_color",
                                "object": "indigo",
                                "mode": "fact",
                                "basis": "stated",
                                "confidence": 1.0,
                            }
                        ],
                    },
                }
            ],
        },
    )

    def fake_provider(**_kwargs) -> Completion:
        return Completion(
            text="indigo",
            usage=Usage(),
            latency_ms=0.0,
            model="fake-reader",
        )

    monkeypatch.setattr(answer, "complete", fake_provider)
    question = Question(
        ability="information_extraction",
        text="What is the launch color?",
        ideal="indigo",
        rubric=[],
        evidence_ids=[1],
        index=0,
    )
    conversation = Conversation(conversation_id="7", sessions=[], questions=[question])
    out_path = tmp_path / "answers.jsonl"
    records, _spend = answer.run(
        [conversation],
        vault_root=tmp_path / "private-vaults",
        out_path=out_path,
        candidate=candidate,
        profile_prefix="beam-test",
        top_k=8,
        maximum_context_bytes=32768,
        conversation_workers=1,
        question_workers=1,
    )
    record = records[0]
    assert record.hypothesis == "indigo"

    ranked = [
        payload
        for args, payload in engine.calls
        if args[-1:] == ["search"] and payload and "query" in payload
    ]
    addressed = [
        payload
        for args, payload in engine.calls
        if args[-1:] == ["search"] and payload and "memory_id" in payload
    ]
    assert ranked == [
        {
            "query": "What is the launch color?",
            "top_k": 8,
            "maximum_context_bytes": 32768,
        }
    ]
    assert addressed == []

    metadata_path = out_path.with_suffix(".jsonl.meta.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["answers_sha256"] == sha256_file(out_path)
    public_record = out_path.read_text() + metadata_path.read_text()
    for private_name in (
        "private-vaults",
        "workspace_id",
        "principal_id",
        "journal",
        "KSCOPE_ROOT",
    ):
        assert private_name not in public_record


def test_declarable_vocabulary_is_operator_read_and_cached(tmp_path: Path):
    engine = FakeEngine()
    candidate = candidate_fixture(tmp_path, engine)
    vault = VaultPool(
        tmp_path / "private-vaults",
        "2026-01-01T00:00:00Z",
        candidate=candidate,
        profile_prefix="beam-test",
    ).for_conversation("9")
    assert vault.memory_types() == ("runtime_decision", "runtime_note")
    assert vault.memory_types() == ("runtime_decision", "runtime_note")
    ontology_calls = [args for args, _ in engine.calls if args[-1:] == ["ontology"]]
    assert len(ontology_calls) == 1
    assert ontology_calls[0][:3] == ["call", "--profile", vault.profile]


def test_ingest_uses_runtime_vocabulary_and_current_remember_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    engine = FakeEngine()
    candidate = candidate_fixture(tmp_path, engine)
    pools = VaultPool(
        tmp_path / "private-vaults",
        "2026-01-01T00:00:00Z",
        candidate=candidate,
        profile_prefix="beam-test",
    )

    def fake_extractor(**_kwargs) -> Completion:
        return Completion(
            text=json.dumps(
                {
                    "memory_type": "runtime_note",
                    "title": "Launch color",
                    "content_md": "The launch color is indigo.",
                    "facts": [
                        {
                            "subject": "launch",
                            "predicate": "uses_color",
                            "object": "indigo",
                            "mode": "fact",
                        }
                    ],
                    "entities": [
                        {
                            "n": "launch",
                            "kind": "event",
                            "is": "the launch discussed by the conversation",
                        },
                        {"n": "indigo", "kind": "concept", "is": "the launch color"},
                    ],
                    "contradicts": [],
                }
            ),
            usage=Usage(),
            latency_ms=0.0,
            model="fake-extractor",
        )

    monkeypatch.setattr(extract, "complete", fake_extractor)
    conversation = Conversation(
        conversation_id="11",
        sessions=[
            [
                {
                    "id": 1,
                    "role": "user",
                    "content": "The launch color is indigo.",
                    "time_anchor": "2026-08-22",
                }
            ]
        ],
        questions=[],
    )
    report = ingest.ingest_conversation(
        conversation,
        vaults=pools,
        cache=extract.ExtractionCache(tmp_path / "cache"),
        spend=Spend(),
        batch_items=20,
    )
    assert report.written == 1
    remember_payloads = [payload for args, payload in engine.calls if args[-1:] == ["remember"]]
    assert len(remember_payloads) == 1
    payload = remember_payloads[0]
    assert payload["mode"] == "create"
    assert "idempotency_key" not in json.dumps(payload)
    delta = payload["items"][0]["semantic_delta"]
    assert delta["memory_type"] == "runtime_note"
    assert delta["occurred_at"] == {"t": "2026-08-22T00:00:00Z", "grain": "instant"}
    assert {entity["n"] for entity in delta["entities"]} == {"launch", "indigo"}
