"""Phase 1 — build the memory.

Runs to completion before a single question is asked. That is BEAM's protocol,
not a convenience: 252 of the 100K tier's 400 questions have evidence spanning
more than one message, the widest span is 262 messages, and `sessions_required`
reaches 5. A question whose evidence sits at positions 20 and 282 cannot be
asked at position 20 — the answer does not exist yet.

Parallelism has a hard asymmetry here:

* **Conversations run concurrently.** Separate vaults, no shared state.
* **Chunks within a conversation do not.** Later facts and contradictions can
  refer to earlier writes, so write order is semantic rather than cosmetic.

## Writes are batched, and that is not a relaxation of the above

`remember` takes an `items` array up to the bound in the pinned public contract. Items still
apply **in order**, each against a projection the previous one updated, and
each still declares its own title and its own facts — there is no shared delta
and nothing is inherited. What a batch amortises is the *derived* work:
Kaleidoscope re-derives the graph and activates the lexical index once per
call, so a per-chunk write pays that per chunk.

Measured on the runtime's own counters, 500 creates through the CLI:

| | single | batched |
| --- | --- | --- |
| calls | 500 | 25 |
| seconds | 60.03 | 16.37 |
| graph rebuilds | 500 | 25 |
| index activations | 500 | 25 |
| index mutations | 500 | 500 |

The last row is the point: both hand the index the same 500 documents. Nothing
is dropped to get the other rows down.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ...config import settings
from ...kaleidoscope import KaleidoscopeError, ReleaseCandidate, VaultPool
from ...llm import Spend
from .dataset import Conversation, pack
from .extract import ExtractionCache, extract


@dataclass
class IngestReport:
    conversation_id: str
    chunks: int = 0
    written: int = 0
    skipped_not_durable: int = 0
    failed: int = 0
    contradicts: int = 0
    # Numbers the extractor cited that resolve to nothing. Non-zero means it is
    # inventing references, which is the failure numbering was meant to remove.
    unresolved_references: int = 0
    # Write calls actually issued. `written / batches` is the amortisation this
    # phase bought, and it is reported rather than assumed: a conversation full
    # of revisions flushes early and often, and that shows up here.
    batches: int = 0
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
    batch_items: int,
    chunk_size: int = 2,
    progress: threading.Lock | None = None,
) -> IngestReport:
    """Walk one conversation front to back, writing as it goes."""
    report = IngestReport(conversation_id=conversation.conversation_id)
    vault = vaults.for_conversation(conversation.conversation_id)
    memory_types = vault.memory_types()
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

    # Items waiting to go out as one call, as (number, payload).
    #
    # A buffered item has no `memory_id` yet — the service derives it from the
    # retry identity the CLI derives per item — so a later extraction that
    # names a buffered predecessor cannot be given an id to point at. The buffer
    # is flushed first in that case, below. Predicting the id here instead would
    # mean reimplementing the service's key derivation in the harness, and a
    # harness that guesses at identity is how a benchmark ends up writing edges
    # to whatever memory happened to land at that index.
    batch: list[tuple[int, dict]] = []

    def flush() -> None:
        if not batch:
            return
        payload = {
            "mode": "create",
            "items": [item for _, item in batch],
        }
        report.batches += 1
        try:
            response = vault.call("remember", payload)
        except KaleidoscopeError as exc:
            report.failed += len(batch)
            report.errors.append(f"batch at memory {batch[0][0]}: {exc}")
            batch.clear()
            return
        results = response.get("results") or []
        if len(results) != len(batch):
            # Never align a short result list positionally against the items —
            # that silently attributes one item's id to another.
            report.failed += len(batch)
            report.errors.append(
                f"batch at memory {batch[0][0]}: {len(results)} results for {len(batch)} items"
            )
            batch.clear()
            return
        for (number, _), result in zip(batch, results):
            if result.get("status") in {"rejected", "refused"}:
                report.failed += 1
                report.errors.append(f"memory {number}: {result.get('reason') or 'rejected'}")
                continue
            report.written += 1
            memory_id = result.get("memory_id")
            if memory_id:
                by_number[number] = memory_id
        batch.clear()

    for index, chunk in enumerate(chunks):
        extraction = extract(
            chunk,
            conversation_id=conversation.conversation_id,
            index=index,
            prior=prior,
            cache=cache,
            memory_types=memory_types,
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

        # A reference into the buffer has no id to point at yet, so the buffer
        # goes out first. This is the only thing batching gives up, and it gives
        # it up rather than guessing.
        referenced = set(extraction.contradicts_indices)
        if referenced and any(number in referenced for number, _ in batch):
            flush()

        delta = dict(extraction.delta)
        contradicts = [
            by_number[number] for number in extraction.contradicts_indices if number in by_number
        ]
        report.unresolved_references += sum(
            1 for number in extraction.contradicts_indices if number not in by_number
        )
        if contradicts:
            delta["contradicts"] = contradicts
            report.contradicts += 1

        number = len(prior) + 1
        batch.append(
            (
                number,
                {
                    "content_md": f"# {extraction.title}\n\n{extraction.content_md}\n",
                    "semantic_delta": delta,
                },
            )
        )
        # The extractor is shown this the moment the memory is extracted, not
        # when it is flushed. Numbering is assigned in write order either way,
        # and a buffered predecessor forces a flush above, so the numbers it is
        # offered always resolve.
        prior.append(
            {
                "number": number,
                "title": extraction.title,
                "summary": extraction.content_md,
            }
        )
        if len(batch) >= batch_items:
            flush()

    flush()

    report.seconds = time.time() - started
    if progress is not None:
        with progress:
            print(
                f"  conv {report.conversation_id:>3}: {report.written} written "
                f"in {report.batches} call(s), "
                f"{report.skipped_not_durable} not durable, {report.failed} failed, "
                f"{report.contradicts} contradictions  ({report.seconds:.0f}s)",
                flush=True,
            )
    return report


def run(
    conversations: list[Conversation],
    *,
    vault_root: Path,
    cache_root: Path,
    candidate: ReleaseCandidate,
    profile_prefix: str,
    created_at: str = "2026-01-01T00:00:00Z",
    chunk_size: int = 2,
    workers: int | None = None,
) -> tuple[list[IngestReport], Spend]:
    """Phase 1 over every conversation, concurrently."""
    workers = workers or settings.concurrency.conversations
    vaults = VaultPool(
        vault_root,
        created_at,
        candidate=candidate,
        profile_prefix=profile_prefix,
    )
    batch_items = int(candidate.public_contract["limits"]["remember_batch_items"])
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
                batch_items=batch_items,
                chunk_size=chunk_size,
                progress=progress,
            ): conversation
            for conversation in conversations
        }
        for future in as_completed(futures):
            reports.append(future.result())

    reports.sort(key=lambda r: int(r.conversation_id) if r.conversation_id.isdigit() else 0)
    return reports, spend


def write_report(
    reports: list[IngestReport],
    spend: Spend,
    path: Path,
    *,
    candidate: ReleaseCandidate,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "phase": "ingest",
                "candidate": candidate.evidence,
                "release_evidence_claimed": False,
                "extractor": settings.models.extractor,
                "conversations": [r.as_dict() for r in reports],
                "totals": {
                    "chunks": sum(r.chunks for r in reports),
                    "written": sum(r.written for r in reports),
                    "skipped_not_durable": sum(r.skipped_not_durable for r in reports),
                    "failed": sum(r.failed for r in reports),
                    "contradicts": sum(r.contradicts for r in reports),
                    "remember_calls": sum(r.batches for r in reports),
                },
                "spend_usd": round(spend.total, 4),
                "spend_by_stage": {k: round(v, 4) for k, v in spend.by_stage.items()},
            },
            indent=2,
        )
    )
