"""Digest-bound, profile-first access to the shipped ``kscope`` executable.

The benchmark is a controller around a release candidate, not another memory
implementation. Every engine call is made through one executable whose bytes
and generated public contract were pinned by the caller. A run refuses before
vault work when either digest is absent or disagrees.

Conversation isolation is expressed with native version-1 profiles. The
benchmark never persists or reports root/workspace/principal/journal tuples;
after ``init-profile`` every operation is ``call --profile NAME``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

CALL_TIMEOUT_SECONDS = 600
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PUBLIC_TOOLS = ("remember", "search")
SAFE_ENVIRONMENT_KEYS = (
    "APPDATA",
    "HOME",
    "LANG",
    "LC_ALL",
    "LOCALAPPDATA",
    "SystemRoot",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "WINDIR",
    "XDG_CONFIG_HOME",
)

CommandRunner = Callable[[Path, list[str], str | None], tuple[int, str, str]]


class KaleidoscopeError(RuntimeError):
    """A candidate, profile, or engine operation refused."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _subprocess_runner(
    executable: Path, args: list[str], stdin: str | None
) -> tuple[int, str, str]:
    # The candidate is an absolute executable and profile calls are explicit.
    # It has no reason to receive API keys, vault-coordinate overrides, or the
    # rest of the controller process environment. Keep only platform paths and
    # locale settings needed to locate its non-secret profile configuration.
    environment = {key: os.environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in os.environ}
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


def _json_object(text: str, label: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise KaleidoscopeError(f"{label} did not return valid JSON") from exc
    if not isinstance(value, dict):
        raise KaleidoscopeError(f"{label} did not return a JSON object")
    return value


def _required_digest(value: str | None, label: str) -> str:
    if not value or SHA256.fullmatch(value) is None:
        raise KaleidoscopeError(f"{label} must be an explicit lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class ReleaseCandidate:
    """One executable and the generated public contract for exactly its bytes."""

    executable: Path
    executable_sha256: str
    public_contract_sha256: str
    public_contract: dict
    stat_identity: tuple[int, int, int, int] = field(repr=False)
    runner: CommandRunner = field(default=_subprocess_runner, repr=False, compare=False)

    @classmethod
    def load(
        cls,
        *,
        executable: Path | str | None,
        executable_sha256: str | None,
        public_contract: Path | str | None,
        public_contract_sha256: str | None,
        runner: CommandRunner = _subprocess_runner,
    ) -> ReleaseCandidate:
        """Load and verify immutable inputs; no best-effort discovery is allowed."""
        if executable is None:
            raise KaleidoscopeError("candidate executable is required")
        if public_contract is None:
            raise KaleidoscopeError("generated public contract is required")
        expected_executable = _required_digest(executable_sha256, "candidate digest")
        expected_contract = _required_digest(public_contract_sha256, "public-contract digest")

        executable_path = Path(executable).expanduser().resolve(strict=True)
        contract_path = Path(public_contract).expanduser().resolve(strict=True)
        if not executable_path.is_file():
            raise KaleidoscopeError("candidate executable is not a regular file")
        if not contract_path.is_file():
            raise KaleidoscopeError("public contract is not a regular file")

        observed_executable = sha256_file(executable_path)
        if observed_executable != expected_executable:
            raise KaleidoscopeError("candidate digest mismatch")
        contract_bytes = contract_path.read_bytes()
        if sha256_bytes(contract_bytes) != expected_contract:
            raise KaleidoscopeError("public-contract digest mismatch")
        contract = _json_object(contract_bytes.decode("utf-8"), "public contract")

        if contract.get("schema_version") != "kaleidoscope.public-contract.v1":
            raise KaleidoscopeError("unsupported public-contract schema")
        if (contract.get("executable") or {}).get("sha256") != expected_executable:
            raise KaleidoscopeError("public contract is bound to a different candidate")
        tools = tuple(
            tool.get("name")
            for tool in ((contract.get("mcp") or {}).get("tools") or [])
            if isinstance(tool, dict)
        )
        if tools != PUBLIC_TOOLS:
            raise KaleidoscopeError("public contract does not expose exactly remember and search")

        stat = executable_path.stat()
        return cls(
            executable=executable_path,
            executable_sha256=expected_executable,
            public_contract_sha256=expected_contract,
            public_contract=contract,
            stat_identity=(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
            runner=runner,
        )

    @property
    def evidence(self) -> dict:
        """Path-free identity safe to put in benchmark output."""
        return {
            "schema_version": self.public_contract["schema_version"],
            "product_version": (self.public_contract.get("product") or {}).get("version"),
            "target": (self.public_contract.get("target") or {}).get("triple"),
            "executable_sha256": self.executable_sha256,
            "public_contract_sha256": self.public_contract_sha256,
            "mcp_tools": list(PUBLIC_TOOLS),
            "signature_verified": False,
        }

    @property
    def acquisition_key(self) -> str:
        """Filesystem/profile namespace bound to both immutable inputs."""
        return f"{self.executable_sha256[:12]}-{self.public_contract_sha256[:12]}"

    def invoke(self, args: list[str], payload: dict | None = None) -> dict:
        """Execute only while the candidate still has the pinned digest."""
        stat = self.executable.stat()
        observed_identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if observed_identity != self.stat_identity:
            raise KaleidoscopeError("candidate changed after verification")
        stdin = (
            None if payload is None else json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        code, stdout, _stderr = self.runner(self.executable, args, stdin)
        if code != 0:
            # Candidate stderr can contain local paths. Public benchmark records
            # receive only the first command category.
            raise KaleidoscopeError(f"candidate refused {args[0]}")
        return _json_object(stdout, args[0])

    def require_bundled_model(self) -> dict:
        report = self.invoke(["model"])
        if report.get("status") != "bundled":
            raise KaleidoscopeError("candidate has no bundled embedding model")
        return report


class Vault:
    """One profile-addressed conversation vault."""

    def __init__(self, candidate: ReleaseCandidate, profile: str) -> None:
        self.candidate = candidate
        self.profile = profile
        self._memory_types: tuple[str, ...] | None = None
        self._vocabulary_lock = threading.Lock()

    def call(self, operation: str, payload: dict) -> dict:
        return self.candidate.invoke(["call", "--profile", self.profile, operation], payload)

    def ranked_search(
        self,
        query: str,
        *,
        top_k: int,
        maximum_context_bytes: int,
    ) -> dict:
        if not query.strip() or top_k < 1 or maximum_context_bytes < 1:
            raise KaleidoscopeError("ranked search requires query and positive bounds")
        result = self.call(
            "search",
            {
                "query": query,
                "top_k": top_k,
                "maximum_context_bytes": maximum_context_bytes,
            },
        )
        if not isinstance(result.get("selected_hits"), list):
            raise KaleidoscopeError("ranked search returned no selected_hits collection")
        return result

    def addressed_search(self, memory_id: str) -> dict:
        if not memory_id.startswith("mem_"):
            raise KaleidoscopeError("addressed search requires a memory_id")
        result = self.call("search", {"memory_id": memory_id})
        if result.get("memory_id") != memory_id:
            raise KaleidoscopeError("addressed search returned a different memory")
        if "selected_hits" in result:
            raise KaleidoscopeError("addressed search returned ranked shape")
        return result

    def memory_types(self) -> tuple[str, ...]:
        """Read declarable types through the operator surface, never an agent tool."""
        with self._vocabulary_lock:
            if self._memory_types is None:
                response = self.call("ontology", {"mode": "read"})
                values = (response.get("declarable") or {}).get("memory_types")
                if not isinstance(values, list) or not values:
                    raise KaleidoscopeError("ontology returned no declarable memory types")
                cleaned = tuple(value for value in values if isinstance(value, str) and value)
                if len(cleaned) != len(values) or len(set(cleaned)) != len(cleaned):
                    raise KaleidoscopeError("ontology returned invalid declarable memory types")
                self._memory_types = cleaned
            return self._memory_types


class VaultPool:
    """Create or reopen exactly one native profile per conversation."""

    def __init__(
        self,
        root: Path,
        created_at: str,
        *,
        candidate: ReleaseCandidate,
        profile_prefix: str,
    ) -> None:
        if PROFILE_NAME.fullmatch(profile_prefix) is None or len(profile_prefix) > 40:
            raise KaleidoscopeError("profile prefix is not portable")
        self.root = Path(root).expanduser().resolve()
        self.created_at = created_at
        self.candidate = candidate
        self.profile_prefix = profile_prefix
        self._vaults: dict[str, Vault] = {}
        self._lock = threading.Lock()

    def _profile_name(self, conversation_id: str) -> str:
        suffix = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:16]
        return f"{self.profile_prefix}-{suffix}"

    def _open(self, conversation_id: str) -> Vault:
        profile = self._profile_name(conversation_id)
        suffix = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:16]
        root = (self.root / f"conv-{suffix}").resolve()
        listing = self.candidate.invoke(["profile", "list"])
        profiles = listing.get("profiles")
        if not isinstance(profiles, list) or any(not isinstance(value, str) for value in profiles):
            raise KaleidoscopeError("profile list returned invalid shape")
        if profile in profiles:
            record = self.candidate.invoke(["profile", "show", profile])
            # The tuple remains inside this process. Only the non-secret profile
            # name crosses later calls or enters a benchmark record.
            if record.get("name") != profile or Path(record.get("root", "")).resolve() != root:
                raise KaleidoscopeError("existing benchmark profile points at another vault")
        else:
            root.parent.mkdir(parents=True, exist_ok=True)
            if root.exists():
                self.candidate.invoke(["profile", "import", profile, str(root), "process-local"])
            else:
                self.candidate.invoke(
                    ["init-profile", profile, str(root), self.created_at, "process-local"]
                )
        return Vault(self.candidate, profile)

    def for_conversation(self, conversation_id: str) -> Vault:
        with self._lock:
            if conversation_id not in self._vaults:
                self._vaults[conversation_id] = self._open(conversation_id)
            return self._vaults[conversation_id]
