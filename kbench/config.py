"""Configuration, read once from the environment.

Every model is an OpenAI model and every key is `OPENAI_API_KEY`. Point
`OPENAI_BASE_URL` at any OpenAI-compatible endpoint to use something else — the
code makes no assumption beyond the API shape.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def _load_dotenv() -> None:
    """Read `.env` without a dependency, and without clobbering a real export."""
    for candidate in (REPO_ROOT / ".env", Path.cwd() / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


@dataclass(frozen=True)
class Models:
    """Which model does what.

    The **extractor** writes memory, the **reader** answers from retrieved
    context, and the **judge** scores the answer against the rubric.

    Keep the reader identical across every arm you intend to compare. A reader
    difference is indistinguishable from a memory difference in the final score,
    and it is the single easiest way to publish a number that means nothing.

    The judge is deliberately not tied to the reader. If the judge tracked the
    arm being graded, judge quality and arm quality would be confounded.
    """

    extractor: str = os.environ.get("KBENCH_EXTRACTOR_MODEL", "gpt-4.1")
    reader: str = os.environ.get("KBENCH_READER_MODEL", "gpt-4.1")
    judge: str = os.environ.get("KBENCH_JUDGE_MODEL", "gpt-4.1")

    extractor_temperature: float = 0.0
    reader_temperature: float = 0.0
    judge_temperature: float = 0.0

    reader_max_tokens: int = 2048
    extractor_max_tokens: int = 4096


@dataclass(frozen=True)
class Concurrency:
    """How much runs at once.

    Conversations are independent — each has its own vault and BEAM's evidence
    never crosses conversations — so they parallelise freely. Questions within a
    conversation parallelise too, because retrieval is read-only.

    Total in-flight LLM work is `conversations * questions`. Raise both until
    the endpoint pushes back, then stop.
    """

    conversations: int = int(os.environ.get("KBENCH_CONVERSATION_WORKERS", "4"))
    questions: int = int(os.environ.get("KBENCH_QUESTION_WORKERS", "4"))


@dataclass(frozen=True)
class Retrieval:
    """How deep each question reads.

    **This is the number a published comparison turns on, so it is declared
    here rather than left as a default in three signatures.**

    `compile` returns a bounded exposure, not a top-k slice: it selects within
    the limit and stops early when nothing further earns its place. The limit is
    a ceiling on what may be exposed, not a quota to fill.

    The default was 5. BEAM's published comparisons read their store at
    `top_50` and `top_200`, so a run at 5 was asking Kaleidoscope for five
    memories and everything else for a hundred, then reporting the scores side
    by side. That is a depth handicap wearing the shape of a result, and it is
    the same failure the README warns about for readers: a difference in how the
    arms were asked is indistinguishable from a difference in what they know.

    100 is chosen to sit inside the range published work reports rather than
    above it. Raise it with `KBENCH_COMPILE_LIMIT` and say so beside any number
    it produced.
    """

    compile_limit: int = int(os.environ.get("KBENCH_COMPILE_LIMIT", "100"))
    extraction: int = int(os.environ.get("KBENCH_EXTRACTION_WORKERS", "16"))
    judging: int = int(os.environ.get("KBENCH_JUDGE_WORKERS", "8"))

    @property
    def question_slots(self) -> int:
        """Sized for the product, not for one conversation.

        A conversation worker submits its questions and then blocks on the
        results. With a shared pool sized for one conversation, every slot can be
        occupied by a waiter and the work they are waiting on never runs — the
        classic nested-pool deadlock, which presents as a hang rather than an
        error.
        """
        return max(1, self.conversations * self.questions)


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    base_url: str | None = os.environ.get("OPENAI_BASE_URL") or None
    models: Models = field(default_factory=Models)
    concurrency: Concurrency = field(default_factory=Concurrency)
    retrieval: Retrieval = field(default_factory=Retrieval)
    data_dir: Path = Path(os.environ.get("KBENCH_DATA_DIR", str(DEFAULT_DATA_DIR)))
    results_dir: Path = Path(os.environ.get("KBENCH_RESULTS_DIR", str(DEFAULT_RESULTS_DIR)))

    def require_api_key(self) -> str:
        if not self.api_key:
            raise SystemExit(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "or export the variable."
            )
        return self.api_key


settings = Settings()
