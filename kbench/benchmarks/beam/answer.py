"""Phase 2 — search once and answer.

Two stages share one per-question unit of work: one ranked ``search`` returns
the bounded context Kaleidoscope assembled, and the reader answers from exactly
that. The acquisition path never follows a ranked hit with a second read.

**The context comes from `search.context_text`, not from re-rendering hits.** That
distinction matters: `render_bounded_context` emits graph paths
(`Connected through: A -[supersedes]-> B`), contradiction flags, validity
windows and the ontology capsule. A harness that takes `selected_hits` and
formats them itself throws all of that away and measures a weaker system than
the one under test.

Both stages parallelise fully. Conversations are independent stores; ranked
searches write exposures, whose concurrency is owned by the native vault.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ...config import settings
from ...kaleidoscope import KaleidoscopeError, ReleaseCandidate, VaultPool, sha256_file
from ...llm import Spend, complete
from .dataset import Conversation, Question

READER_PROMPT_PATH = Path(__file__).parent / "prompts" / "reader.md"


@dataclass
class Answer:
    conversation_id: str
    question_index: int
    ability: str
    question: str
    hypothesis: str = ""
    context: str = ""
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_count: int = 0
    context_chars: int = 0
    retrieval_ms: float = 0.0
    reader_ms: float = 0.0
    evidence_ids: list[int] = field(default_factory=list)
    abstained: bool = False
    mean_channels: float | None = None
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def reader_prompt(question: str, context: str) -> str:
    return (
        READER_PROMPT_PATH.read_text().replace("{context}", context).replace("{question}", question)
    )


def answer_question(
    conversation: Conversation,
    question: Question,
    *,
    vaults: VaultPool,
    spend: Spend,
    top_k: int = settings.retrieval.top_k,
    maximum_context_bytes: int = settings.retrieval.maximum_context_bytes,
) -> Answer:
    record = Answer(
        conversation_id=conversation.conversation_id,
        question_index=question.index,
        ability=question.ability,
        question=question.text,
        evidence_ids=list(question.evidence_ids),
    )
    vault = vaults.for_conversation(conversation.conversation_id)

    started = time.time()
    try:
        searched = vault.ranked_search(
            question.text,
            top_k=top_k,
            maximum_context_bytes=maximum_context_bytes,
        )
    except KaleidoscopeError as exc:
        record.error = str(exc)
        record.retrieval_ms = (time.time() - started) * 1000.0
        return record
    record.retrieval_ms = (time.time() - started) * 1000.0

    # Kaleidoscope's own bounded context, verbatim.
    record.context = searched.get("context_text", "") or ""
    record.context_chars = len(record.context)
    hits = searched.get("selected_hits") or []
    record.retrieved_count = len(hits)
    record.retrieved_ids = [h.get("memory_id", "") for h in hits]

    abstention = searched.get("abstention") or {}
    record.abstained = bool(abstention.get("abstained"))
    record.mean_channels = abstention.get("mean_channels")

    started = time.time()
    result = complete(
        model=settings.models.reader,
        messages=[{"role": "user", "content": reader_prompt(question.text, record.context)}],
        temperature=settings.models.reader_temperature,
        max_tokens=settings.models.reader_max_tokens,
        stage="reader",
        spend=spend,
    )
    record.reader_ms = (time.time() - started) * 1000.0
    record.hypothesis = result.text
    if result.error:
        record.error = result.error
    return record


def run(
    conversations: list[Conversation],
    *,
    vault_root: Path,
    out_path: Path,
    created_at: str = "2026-01-01T00:00:00Z",
    candidate: ReleaseCandidate,
    profile_prefix: str,
    top_k: int = settings.retrieval.top_k,
    maximum_context_bytes: int = settings.retrieval.maximum_context_bytes,
    conversation_workers: int | None = None,
    question_workers: int | None = None,
) -> tuple[list[Answer], Spend]:
    """Phase 2 over every question, concurrently, streaming to disk."""
    concurrency = settings.concurrency
    conversation_workers = conversation_workers or concurrency.conversations
    question_workers = question_workers or concurrency.questions
    question_slots = max(1, conversation_workers * question_workers)

    vaults = VaultPool(
        vault_root,
        created_at,
        candidate=candidate,
        profile_prefix=profile_prefix,
    )
    spend = Spend()
    answers: list[Answer] = []
    write_lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("w")

    def emit(record: Answer) -> None:
        # Streamed as each answer lands, so a crash keeps what it earned and a
        # long run is watchable.
        with write_lock:
            handle.write(json.dumps(record.as_dict()) + "\n")
            handle.flush()
            answers.append(record)

    def one_conversation(conversation: Conversation) -> None:
        for record in question_pool.map(
            lambda q: answer_question(
                conversation,
                q,
                vaults=vaults,
                spend=spend,
                top_k=top_k,
                maximum_context_bytes=maximum_context_bytes,
            ),
            conversation.questions,
        ):
            emit(record)

    total = sum(len(c.questions) for c in conversations)
    print(
        f"answering {total} questions — {conversation_workers} conversations x "
        f"{question_workers} questions ({question_slots} slots)"
    )
    try:
        # The question pool is sized for the PRODUCT. A conversation worker
        # submits its questions and then blocks on the results; with a pool
        # sized for one conversation, the waiters occupy every slot and starve
        # the work they are waiting for.
        with (
            ThreadPoolExecutor(max_workers=question_slots) as question_pool,
            ThreadPoolExecutor(max_workers=conversation_workers) as conversation_pool,
        ):
            futures = [
                conversation_pool.submit(one_conversation, conversation)
                for conversation in conversations
            ]
            for future in as_completed(futures):
                future.result()
    finally:
        handle.close()

    answers.sort(
        key=lambda a: (
            int(a.conversation_id) if a.conversation_id.isdigit() else 0,
            a.question_index,
        )
    )
    metadata_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "kbench.run-inputs.v1",
                "candidate": candidate.evidence,
                "search": {
                    "top_k": top_k,
                    "maximum_context_bytes": maximum_context_bytes,
                    "calls_per_question": 1,
                },
                "answers_sha256": sha256_file(out_path),
                "release_evidence_claimed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return answers, spend
