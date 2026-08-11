"""Thin wrapper around the `kscope` CLI.

Kaleidoscope is assumed installed and on `PATH` (or pointed at by
`KSCOPE_BINARY`). Nothing here reimplements the runtime — every call shells out,
so what the benchmark measures is the shipped binary rather than a copy of it.

One vault per conversation. That is not an optimisation; it is the isolation the
benchmark depends on. A shared vault would let one conversation's questions
retrieve another conversation's memories and quietly inflate every score.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BINARY = os.environ.get("KSCOPE_BINARY", "kscope")
CALL_TIMEOUT_SECONDS = 600


class KaleidoscopeError(RuntimeError):
    """A `kscope` invocation refused or failed."""


@dataclass(frozen=True)
class Identity:
    """What `kscope init` mints. Every `call` needs all four."""

    root: Path
    workspace_id: str
    principal_id: str
    journal: str

    @classmethod
    def from_json(cls, root: Path, payload: str) -> "Identity":
        record = json.loads(payload)
        return cls(
            root=root,
            workspace_id=record["workspace_id"],
            principal_id=record["principal_id"],
            journal=record["journal"],
        )


def _run(args: list[str], stdin: str | None = None, trace: bool = False) -> tuple[int, str, str]:
    env = dict(os.environ)
    if trace:
        # Turns on `counters::emit` and `trajectory::emit`, both of which write
        # one JSON line to stderr on the way out.
        env["KALEIDOSCOPE_TRACE_JSON"] = "1"
    completed = subprocess.run(
        [DEFAULT_BINARY, *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        timeout=CALL_TIMEOUT_SECONDS,
    )
    return completed.returncode, completed.stdout, completed.stderr


def first_error_line(stdout: str, stderr: str) -> str:
    """The message, not the trace.

    With tracing on, the counters and trajectory emitters write JSON to stderr on
    every exit including failures, so the actual error is one non-JSON line among
    them. Slicing `stderr[:200]` shows the counters and hides the cause.
    """
    for line in stderr.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("{"):
            return stripped
    return (stdout or stderr).strip()[:300]


def trace_records(stderr: str) -> list[dict]:
    """The JSON trace lines, which share stderr with human-readable warnings."""
    records = []
    for line in stderr.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return records


def model() -> dict:
    """What encoder this build carries, if any.

    Worth calling before a run and worth failing on. A build without the bundled
    model has no semantic channel at all, so retrieval is lexical only — it does
    not error, it just quietly measures a different system.
    """
    code, out, err = _run(["model"])
    if code != 0:
        raise KaleidoscopeError(f"kscope model failed: {first_error_line(out, err)}")
    return json.loads(out) if out.strip() else {}


def require_bundled_model() -> dict:
    """Refuse to benchmark a build whose semantic channel is switched off."""
    report = model()
    if report.get("status") != "bundled":
        raise KaleidoscopeError(
            "this kscope build carries no embedding model, so the semantic "
            "retrieval channel is off and retrieval is lexical only. Install a "
            "build with the bundled model, or set KSCOPE_BINARY to one. "
            f"Reported: {report!r}"
        )
    return report


class Vault:
    """One conversation's memory. Thread-safe; instances are cheap."""

    def __init__(self, identity: Identity, trace: bool = False) -> None:
        self.identity = identity
        self.trace = trace

    @classmethod
    def open(cls, root: Path, created_at: str, trace: bool = False) -> "Vault":
        """Open, initialising on first use.

        The identity is persisted beside the vault because `init` mints it and
        nothing else can reproduce it — inventing one earns "identity has invalid
        length or encoding", which the trace lines on the same stream then hide.
        """
        root = Path(root)
        identity_path = root.parent / f"{root.name}.identity.json"
        if not identity_path.exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            code, out, err = _run(["init", str(root), created_at, "process-local"])
            if code != 0:
                raise KaleidoscopeError(f"init failed: {first_error_line(out, err)}")
            identity_path.write_text(out)
        return cls(Identity.from_json(root, identity_path.read_text()), trace=trace)

    def call(self, operation: str, payload: dict) -> dict:
        """One operation. Raises on refusal rather than returning an error shape."""
        code, out, err = self.raw_call(operation, payload)
        if code != 0:
            raise KaleidoscopeError(f"{operation} refused: {first_error_line(out, err)}")
        return json.loads(out) if out.strip() else {}

    def raw_call(self, operation: str, payload: dict) -> tuple[int, str, str]:
        """The unwrapped form, for callers that want the trace or the exit code."""
        identity = self.identity
        return _run(
            [
                "call",
                str(identity.root),
                identity.workspace_id,
                identity.principal_id,
                identity.journal,
                operation,
            ],
            stdin=json.dumps(payload),
            trace=self.trace,
        )


class VaultPool:
    """One vault per conversation, opened once, shared across threads.

    The lock is load-bearing under `--conversation-workers > 1`: a plain
    check-then-insert would let two threads `open` the same root and hold two
    handles onto one vault. `compile` is a write, so that is two writers.
    """

    def __init__(self, root: Path, created_at: str, trace: bool = False) -> None:
        self.root = Path(root)
        self.created_at = created_at
        self.trace = trace
        self._vaults: dict[str, Vault] = {}
        self._lock = threading.Lock()

    def for_conversation(self, conversation_id: str) -> Vault:
        with self._lock:
            if conversation_id not in self._vaults:
                self._vaults[conversation_id] = Vault.open(
                    self.root / f"conv-{conversation_id}",
                    self.created_at,
                    trace=self.trace,
                )
            return self._vaults[conversation_id]
