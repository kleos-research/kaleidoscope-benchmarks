"""Start one MemoryArena server (memory or env) on a loopback port, with two things run_arm.sh
did not have:

  1. every log line is redacted (the gateway host and key never reach a file), and
  2. the OpenAI SDK's own retry machinery is made visible: `openai._base_client` logs
     "Encountered a timeout exception", "Retrying request in ..." and "Encountered an HTTP status
     error" at DEBUG, and a filter keeps ONLY those lines (the SDK's other DEBUG lines are
     dropped unread). Without this a timed-out attempt that the SDK retried leaves no trace
     except a long `ms` on the ledger row.

The harness is not modified: this imports `server:app` / `env.env_server:app` exactly as
run_arm.sh's `uvicorn` command lines did (same cwd, same sys.path).

usage: serve.py memory|env --port P --log <file>
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
ARENA = BASE / "MemoryArena"

SDK_KEEP = (
    "Retrying request in", "Encountered a timeout exception", "Encountered an HTTP status error",
    "Retrying due to status code", "Raising timeout error", "Raising connection error",
    "Encountered exception", "Not retrying", "Re-raising status error", "1 retry left",
    "%i retries left", "Retrying as header", "Not retrying as header",
)


class Redact(logging.Filter):
    def __init__(self, hidden: list[str]):
        super().__init__()
        self.hidden = hidden

    def filter(self, record: logging.LogRecord) -> bool:
        text = record.getMessage()
        for secret in self.hidden:
            text = text.replace(secret, "<redacted>")
        record.msg, record.args = text, ()
        return True


class SdkRetryLinesOnly(logging.Filter):
    """Keep the SDK's retry/timeout narrative, drop everything else it says at DEBUG."""

    def filter(self, record: logging.LogRecord) -> bool:
        return isinstance(record.msg, str) and record.msg.startswith(SDK_KEEP)


def configure_logging(log_path: Path, hidden: list[str]) -> None:
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(created).3f | %(asctime)s | %(levelname)s | %(name)s | %(threadName)s | %(message)s"))
    handler.addFilter(Redact(hidden))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.WARNING)
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sdk = logging.getLogger("openai._base_client")
    sdk.setLevel(logging.DEBUG)
    sdk.addFilter(SdkRetryLinesOnly())
    logging.getLogger("openai").setLevel(logging.DEBUG)  # the parent must not clamp the child
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["memory", "env"])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    for name in ("BENCH_SCOPE", "BENCH_SCOPE_CAP_USD", "BENCH_RUN_ID", "BENCH_TAG"):
        if not os.environ.get(name):
            raise SystemExit(f"{name} must be set in the environment of every process that can call the LLM")

    sys.path.insert(0, str(BASE))
    from common import llm
    llm._load_env()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    hidden = [s for s in (endpoint, endpoint.split("://", 1)[-1], os.environ.get("AZURE_OPENAI_API_KEY", "")) if s]
    configure_logging(Path(args.log), hidden)

    if args.which == "memory":
        os.chdir(ARENA / "memory")
        sys.path.insert(0, str(ARENA / "memory"))
        from server import app  # the harness's own FastAPI app, as `uvicorn server:app` loads it
    else:
        os.chdir(ARENA)
        sys.path.insert(0, str(ARENA))
        from env.env_server import app  # as `uvicorn env.env_server:app` loads it

    import uvicorn
    logging.getLogger("arena.serve").warning("serving %s on 127.0.0.1:%d pid=%d tag=%s llm_timeout=%s",
                                             args.which, args.port, os.getpid(), os.environ.get("BENCH_TAG"),
                                             os.environ.get("BENCH_LLM_TIMEOUT", "(default)"))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_config=None, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
