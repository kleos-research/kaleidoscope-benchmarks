"""Opt-in smoke proof against one digest-bound native release candidate.

Ordinary tests use a deterministic fake engine. This lane is deliberately
separate: when all four ``KBENCH_LIVE_*`` inputs are supplied it creates a
fresh temporary profile and vault through the real candidate, writes one
memory, then exercises ranked and addressed search after reopening through the
profile contract. It never turns an unsigned candidate into release evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kbench.kaleidoscope import PUBLIC_TOOLS, ReleaseCandidate, VaultPool

LIVE_INPUTS = {
    "candidate": "KBENCH_LIVE_CANDIDATE",
    "candidate_sha256": "KBENCH_LIVE_CANDIDATE_SHA256",
    "public_contract": "KBENCH_LIVE_PUBLIC_CONTRACT",
    "public_contract_sha256": "KBENCH_LIVE_PUBLIC_CONTRACT_SHA256",
}

pytestmark = pytest.mark.skipif(
    not any(os.environ.get(name) for name in LIVE_INPUTS.values()),
    reason="set the four KBENCH_LIVE_* candidate inputs to run the native smoke lane",
)


def _private_platform_homes(root: Path) -> dict[str, str]:
    """Create owner-only profile parents for each supported platform resolver."""
    home = root / "home"
    xdg = root / "xdg"
    appdata = root / "appdata"
    for directory in (
        home,
        xdg,
        appdata,
        home / "Library" / "Application Support" / "kaleidoscope" / "profiles",
        xdg / "kaleidoscope" / "profiles",
        appdata / "kaleidoscope" / "profiles",
    ):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
    return {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(xdg),
        "APPDATA": str(appdata),
        "LOCALAPPDATA": str(appdata),
    }


def test_live_candidate_profile_write_ranked_and_addressed_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = [name for name in LIVE_INPUTS.values() if not os.environ.get(name)]
    assert not missing, f"incomplete live candidate inputs: {', '.join(missing)}"

    for key, value in _private_platform_homes(tmp_path).items():
        monkeypatch.setenv(key, value)
    for key in (
        "KSCOPE_ROOT",
        "KSCOPE_WORKSPACE",
        "KSCOPE_PRINCIPAL",
        "KSCOPE_JOURNAL",
        "KSCOPE_PROFILE_HOME",
    ):
        monkeypatch.delenv(key, raising=False)
    canary = "dx09-live-secret-canary"
    monkeypatch.setenv("OPENAI_API_KEY", canary)
    monkeypatch.setenv("ANTHROPIC_API_KEY", canary)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", canary)

    candidate = ReleaseCandidate.load(
        executable=os.environ[LIVE_INPUTS["candidate"]],
        executable_sha256=os.environ[LIVE_INPUTS["candidate_sha256"]],
        public_contract=os.environ[LIVE_INPUTS["public_contract"]],
        public_contract_sha256=os.environ[LIVE_INPUTS["public_contract_sha256"]],
    )
    model = candidate.require_bundled_model()
    assert model["status"] == "bundled"
    assert candidate.evidence["mcp_tools"] == list(PUBLIC_TOOLS)
    assert candidate.evidence["signature_verified"] is False
    evidence_text = json.dumps(candidate.evidence, sort_keys=True)
    assert str(candidate.executable) not in evidence_text
    assert str(tmp_path) not in evidence_text

    pool = VaultPool(
        tmp_path / "vaults" / candidate.acquisition_key,
        "2026-08-22T00:00:00Z",
        candidate=candidate,
        profile_prefix=f"dx09-live-{candidate.executable_sha256[:8]}",
    )
    vault = pool.for_conversation("candidate-smoke")
    memory_types = vault.memory_types()
    memory_type = "note" if "note" in memory_types else memory_types[0]
    remember = vault.call(
        "remember",
        {
            "mode": "create",
            "items": [
                {
                    "content_md": (
                        "# DX09 live candidate marker\n\n"
                        "The candidate-bound smoke marker is indigo.\n"
                    ),
                    "semantic_delta": {
                        "title": "DX09 live candidate marker",
                        "memory_type": memory_type,
                        "entities": [
                            {
                                "n": "dx09 live candidate marker",
                                "kind": "artifact",
                                "is": "the marker written by the native DX09 smoke test",
                            },
                            {
                                "n": "indigo",
                                "kind": "concept",
                                "is": "the unique color value used by the smoke marker",
                            },
                        ],
                        "facts": [
                            {
                                "subject": "dx09 live candidate marker",
                                "predicate": "uses",
                                "object": "indigo",
                                "mode": "fact",
                                "basis": "stated",
                                "confidence": 1.0,
                            }
                        ],
                        "evidence": [
                            {"kind": "test", "reference": "candidate-bound smoke"}
                        ],
                    },
                }
            ],
        },
    )
    results = remember.get("results") or []
    assert len(results) == 1
    assert results[0].get("status") not in {"refused", "rejected"}
    memory_id = results[0].get("memory_id")
    assert isinstance(memory_id, str) and memory_id.startswith("mem_")

    ranked = vault.ranked_search(
        "DX09 live candidate marker indigo",
        top_k=5,
        maximum_context_bytes=32 * 1024,
    )
    assert any(hit.get("memory_id") == memory_id for hit in ranked["selected_hits"])
    assert "indigo" in ranked.get("context_text", "").lower()

    addressed = vault.addressed_search(memory_id)
    assert addressed["memory_id"] == memory_id
    assert "indigo" in addressed.get("content_md", "").lower()

    descriptor = candidate.invoke(["profile", "launch", vault.profile])
    # The generated contract's tool-definition inventory is sorted
    # remember/search, while the native launch descriptor deliberately uses the
    # agent-facing discovery order search/remember. Both are exact contracts.
    assert descriptor["tools"] == ["search", "remember"]
    assert descriptor["environment"] == {}
    assert canary not in json.dumps(
        {"remember": remember, "ranked": ranked, "addressed": addressed, "descriptor": descriptor}
    )
