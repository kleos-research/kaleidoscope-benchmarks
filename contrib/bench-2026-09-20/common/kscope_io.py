"""The one door to the kscope binary, and the isolation guarantee in code.

Every vault this module will touch is an absolute path under
`experiments/bench-2026-09-20/vaults/`. The personal vault is refused by name and
by prefix, and no call ever uses `--profile`, `KSCOPE_ROOT` or the working
directory -- the three routes by which a call could reach it. Every call is
appended to a log, so "which roots did this run touch" is a file, not a belief.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
VAULTS = ROOT / "vaults"
PERSONAL_VAULT = Path("/path/to/kaleidoscope/.kaleidoscope")
KSCOPE = Path(os.environ.get("BENCH_KSCOPE", "/opt/homebrew/bin/kscope"))
# Pinned by content: `--version` says 0.0.6 for builds that differ by whole PRs.
EXPECTED_SHA256_PREFIX = os.environ.get("BENCH_KSCOPE_SHA", "361f5975d522083d")
CREATED_AT = "2026-09-20T00:00:00Z"

# `shipped` is what a user gets. `parity` switches off the two read-path cuts so
# served == min(top_k, corpus); it is a diagnostic nobody installs.
PROFILES = {
    "shipped": {},
    "parity": {"KALEIDOSCOPE_SERVED_SIMILARITY_FLOOR": "0.0",
               "KALEIDOSCOPE_MAX_CANDIDATE_REDUNDANCY": "1.0"},
}

_checked = False


class IsolationError(RuntimeError):
    pass


def check_binary() -> dict:
    global _checked
    digest = hashlib.sha256(KSCOPE.read_bytes()).hexdigest()
    if not digest.startswith(EXPECTED_SHA256_PREFIX):
        raise RuntimeError(f"kscope binary changed: sha256 {digest[:16]} != pinned {EXPECTED_SHA256_PREFIX}")
    model = subprocess.run([str(KSCOPE), "model"], capture_output=True, text=True).stdout
    if '"status":"bundled"' not in model.replace(" ", ""):
        raise RuntimeError("kscope model is not bundled: the vector channel would abstain silently")
    _checked = True
    return {"sha256": digest, "model": "bundled"}


def assert_root(root: Path) -> Path:
    root = Path(root)
    if not root.is_absolute():
        raise IsolationError(f"vault root must be absolute: {root}")
    resolved = root.resolve()
    if resolved == PERSONAL_VAULT or PERSONAL_VAULT in resolved.parents or resolved in PERSONAL_VAULT.parents:
        raise IsolationError(f"refusing the personal vault or an ancestor of it: {resolved}")
    if VAULTS.resolve() not in resolved.parents:
        raise IsolationError(f"vault root must live under {VAULTS}: {resolved}")
    return resolved


def _env(profile: str) -> dict:
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/path/to/home"),
           "KSCOPE_NO_RESIDENT": "1"}
    env.update(PROFILES[profile])
    return env


def _log(row: dict) -> None:
    scope = os.environ.get("BENCH_SCOPE", "unscoped")
    path = ROOT / "logs" / f"kscope_calls-{scope}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(row) + "\n")


@dataclass
class Vault:
    root: Path
    workspace_id: str
    principal_id: str
    journal: str
    profile: str = "shipped"

    @classmethod
    def create(cls, root: Path, profile: str = "shipped") -> "Vault":
        if not _checked:
            check_binary()
        root = assert_root(root)
        if root.exists():
            raise IsolationError(f"vault root already exists, refusing to reuse it: {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run([str(KSCOPE), "init", str(root), CREATED_AT, "process-local"],
                              capture_output=True, text=True, env=_env(profile))
        if proc.returncode != 0:
            raise RuntimeError(f"kscope init rc={proc.returncode}: {(proc.stderr or proc.stdout)[:300]}")
        identity = json.loads(proc.stdout.strip().splitlines()[-1])
        (root.parent / f"{root.name}-identity.json").write_text(json.dumps(identity))
        _log({"ts": time.time(), "op": "init", "root": str(root), "rc": 0})
        return cls(root, identity["workspace_id"], identity["principal_id"], identity["journal"], profile)

    def call(self, operation: str, payload: dict) -> tuple[int, dict | None, str]:
        assert_root(self.root)
        started = time.time()
        proc = subprocess.run(
            [str(KSCOPE), "call", str(self.root), self.workspace_id, self.principal_id,
             self.journal, operation, "--json"],
            input=json.dumps(payload), capture_output=True, text=True, env=_env(self.profile), timeout=600)
        body = None
        text = proc.stdout.strip()
        if text:
            try:
                body = json.loads(text.splitlines()[-1])
            except ValueError:
                try:
                    body = json.loads(text)
                except ValueError:
                    body = None
        row = {"ts": started, "op": operation, "root": str(self.root), "rc": proc.returncode,
               "ms": int((time.time() - started) * 1000), "profile": self.profile}
        if operation == "search" and body:
            row.update(served=len(body.get("selected_hits") or []), omitted=body.get("omitted_hits"),
                       stop=body.get("stop_reason"), abstained=(body.get("abstention") or {}).get("abstained"),
                       exposure_id=body.get("exposure_id"), floor=body.get("served_floor"))
        if operation == "remember" and body:
            row.update(accepted=body.get("accepted_items"), refused=body.get("refused_items"),
                       status=body.get("status"))
        _log(row)
        return proc.returncode, body, proc.stderr

    def remember_items(self, items: list[dict]) -> tuple[int, dict | None, str]:
        """Batch create, at most 20 per call (the shipped limit; overflow is a refusal)."""
        assert 1 <= len(items) <= 20
        return self.call("remember", {"mode": "create", "items": items})

    def search(self, query: str, top_k: int = 10, maximum_context_bytes: int = 32768):
        return self.call("search", {"query": query, "top_k": top_k,
                                    "maximum_context_bytes": maximum_context_bytes})


def roots_touched(scope: str) -> set[str]:
    path = ROOT / "logs" / f"kscope_calls-{scope}.jsonl"
    return {json.loads(line)["root"] for line in path.read_text().splitlines()} if path.exists() else set()
