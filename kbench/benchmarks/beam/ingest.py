"""Phase 1 — build the memory.

Runs to completion before a single question is asked. That is BEAM's protocol,
not a convenience: 252 of the 100K tier's 400 questions have evidence spanning
more than one message, the widest span is 262 messages, and `sessions_required`
reaches 5. A question whose evidence sits at positions 20 and 282 cannot be
asked at position 20 — the answer does not exist yet.

Parallelism has a hard asymmetry here:

* **Conversations run concurrently.** Separate vaults, no shared state.
* **Chunks within a conversation do not.** A `supersedes` can only name a memory
  already written, so writing turn 40 before turn 12 loses the revision
  silently — it becomes a duplicate instead of a retirement.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ...config import settings
from ...kaleidoscope import KaleidoscopeError, VaultPool
from ...llm import Spend
from .dataset import Conversation, pack, render
from .extract import ExtractionCache, extract


@dataclass
class IngestReport:
    conversation_id: str
    chunks: int = 0
    written: int = 0
    skipped_not_durable: int = 0
    failed: int = 0
    supersedes: int = 0
    contradicts: int = 0
    # Numbers the extractor cited that resolve to nothing. Non-zero means it is
    # inventing references, which is the failure numbering was meant to remove.
    unresolved_references: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {**self.__dict__, "errors": self.errors[:10]}


def ingest_conversation(
    conversation: Conversation,
    *,
    vaults: VaultPool,
    cache: ExtractionCache,
    spend: Spend,
    chunk_size: int = 2,
    progress: threading.Lock | None = None,
) -> IngestReport:
    """Walk one conversation front to back, writing as it goes."""
    report = IngestReport(conversation_id=conversation.conversation_id)
    vault = vaults.for_conversation(conversation.conversation_id)
    chunks = pack(conversation.sessions, chunk_size=chunk_size)
    report.chunks = len(chunks)
    started = time.time()

    # The extractor names prior memories by NUMBER, so this maps the number it
    # was shown back to the memory id `remember` requires. Numbers are stable
    # for the conversation and assigned in write order.
    #
    # A number the extractor invents resolves to nothing and is dropped rather
    # than guessed at — the alternative is writing an edge to whichever memory
    # happened to land at that index, which is worse than no edge.
    by_number: dict[int, str] = {}
    prior: list[dict] = []

    for index, chunk in enumerate(chunks):
        extraction = extract(
            chunk,
            conversation_id=conversation.conversation_id,
            index=index,
            prior=prior,
            cache=cache,
            spend=spend,
        )
        if extraction.error:
            report.failed += 1
            report.errors.append(f"chunk {index}: {extraction.error}")
            continue
        if not extraction.writes:
            # No facts, so nothing durable. Not a failure — the extractor said
            # this exchange establishes nothing, which is a valid answer.
            report.skipped_not_durable += 1
            continue

        delta = dict(extraction.delta)
        superseded = by_number.get(extraction.supersedes_index) if extraction.supersedes_index else None
        if superseded:
            delta["supersedes"] = superseded
            report.supersedes += 1
        else:
            if extraction.supersedes_index is not None:
                report.unresolved_references += 1
        contradicts = [
            by_number[number]
            for number in extraction.contradicts_indices
            if number in by_number and by_number[number] != superseded
        ]
        report.unresolved_references += sum(
            1 for number in extraction.contradicts_indices if number not in by_number
        )
        if contradicts:
            delta["contradicts"] = contradicts
            report.contradicts += 1

        payload = {
            "mode": "create",
            "content_md": f"# {extraction.title}\n\n{extraction.content_md}\n",
            "semantic_delta": delta,
            "idempotency_key": f"beam-{conversation.conversation_id}-{index}",
        }
        try:
            response = vault.call("remember", payload)
        except KaleidoscopeError as exc:
            report.failed += 1
            report.errors.append(f"chunk {index}: {exc}")
            continue

        report.written += 1
        number = len(prior) + 1
        memory_id = response.get("memory_id")
        if memory_id:
            by_number[number] = memory_id
        prior.append(
            {
                "number": number,
                "title": extraction.title,
                "summary": extraction.content_md,
            }
        )

    report.seconds = time.time() - started
    if progress is not None:
        with progress:
            print(
                f"  conv {report.conversation_id:>3}: {report.written} written, "
                f"{report.skipped_not_durable} not durable, {report.failed} failed, "
                f"{report.supersedes} supersedes  ({report.seconds:.0f}s)",
                flush=True,
            )
    return report


def run(
    conversations: list[Conversation],
    *,
    vault_root: Path,
    cache_root: Path,
    created_at: str = "2026-01-01T00:00:00Z",
    chunk_size: int = 2,
    workers: int | None = None,
) -> tuple[list[IngestReport], Spend]:
    """Phase 1 over every conversation, concurrently."""
    workers = workers or settings.concurrency.conversations
    vaults = VaultPool(vault_root, created_at)
    cache = ExtractionCache(cache_root)
    spend = Spend()
    progress = threading.Lock()

    print(f"ingesting {len(conversations)} conversations, {workers} at a time")
    reports: list[IngestReport] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                ingest_conversation,
                conversation,
                vaults=vaults,
                cache=cache,
                spend=spend,
                chunk_size=chunk_size,
                progress=progress,
            ): conversation
            for conversation in conversations
        }
        for future in as_completed(futures):
            reports.append(future.result())

    reports.sort(key=lambda r: int(r.conversation_id) if r.conversation_id.isdigit() else 0)
    return reports, spend


def write_report(reports: list[IngestReport], spend: Spend, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "phase": "ingest",
                "extractor": settings.models.extractor,
                "conversations": [r.as_dict() for r in reports],
                "totals": {
                    "chunks": sum(r.chunks for r in reports),
                    "written": sum(r.written for r in reports),
                    "skipped_not_durable": sum(r.skipped_not_durable for r in reports),
                    "failed": sum(r.failed for r in reports),
                    "supersedes": sum(r.supersedes for r in reports),
                    "contradicts": sum(r.contradicts for r in reports),
                },
                "spend_usd": round(spend.total, 4),
                "spend_by_stage": {k: round(v, 4) for k, v in spend.by_stage.items()},
            },
            indent=2,
        )
    )
