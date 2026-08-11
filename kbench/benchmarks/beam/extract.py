"""Turn one exchange into a semantic delta.

This is the write path's only model call. The contract it implements is
`AGENTS.md`; the prompt it renders is `prompts/extraction.md`.

Three design decisions here are deliberate departures from the obvious shape,
and each removes something the model cannot actually supply.

## 1. The extractor is not asked for a confidence

`SemanticFactProposal.confidence` is required by the service and feeds an
admission term, so a value must be sent. It is **not** asked of the model. A
language model has no calibrated distinction between 0.49 and 0.5 — asking for
one returns a number that looks precise and is noise, and that noise then flows
into an admission decision.

So the harness sends `EXTRACTED_FACT_CONFIDENCE` (1.0) for every fact and says
why: the extractor asserts what the exchange *states*. If it is unsure what the
exchange states, the fact should not be written at all. The field is there for
callers with a real calibrated source — a classifier, a vote, a measurement —
and this is not one.

## 2. There is no `worth_remembering` flag

It used to exist and it was redundant with `facts: []`. An extraction that
produces no facts writes nothing, because `remember` requires at least one — so
the flag was a *second* judgement about the same question, and a second
judgement can only lose information.

It lost a great deal. Measured on BEAM 100K, the gate discarded 322 of 1,094
evidence-bearing exchanges, capping reachable evidence at 0.706 before retrieval
ran at all. The prompt now asks for facts, and silence is expressed by returning
none.

## 3. Prior memories are numbered, not named

Supersession used to be resolved by exact title match, which is brittle in the
obvious way — the model must reproduce a string precisely, and a near-miss is
silently a no-op that writes a duplicate. Prior memories are now numbered and
the model returns an integer. Same reason mem0 remaps existing memories to
integers in their update prompt: it removes an entire class of hallucination.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ...config import settings
from ...llm import Spend, complete, parse_json_object
from .dataset import render

PROMPT_PATH = Path(__file__).parent / "prompts" / "extraction.md"

# The service's accepted vocabulary. An unaccepted type is refused at the
# boundary and the refusal costs the call that produced it, so the list offered
# to the extractor must match the list the service accepts, exactly.
MEMORY_TYPES: tuple[str, ...] = (
    "architecture",
    "constraint",
    "correction",
    "decision",
    "note",
    "outcome",
    "preference",
    "procedure",
)
FALLBACK_TYPE = "note"

# See the module docstring. Constant, and not asked of the model.
EXTRACTED_FACT_CONFIDENCE = 1.0

MAX_FACTS = 32
MAX_ENTITIES = 8

# How many already-written memories are offered as supersession candidates.
# Bounded because the list rides in every prompt. Selection is by recency, which
# has a cliff: a revision pointing further back than this cannot reach its
# target. Measured on BEAM 100K, a window of 40 covers about 80% of candidate
# revisions and 80 covers about 93%.
PRIOR_WINDOW = 40


@dataclass
class Extraction:
    """What one exchange produced. `delta is None` means nothing is written."""

    delta: dict | None = None
    title: str = ""
    content_md: str = ""
    supersedes_index: int | None = None
    contradicts_indices: list[int] = field(default_factory=list)
    error: str | None = None
    from_cache: bool = False

    @property
    def writes(self) -> bool:
        return self.delta is not None


def prompt_template() -> str:
    return PROMPT_PATH.read_text()


def prompt_fingerprint() -> str:
    """Cache key component, so a prompt edit invalidates only its own results."""
    return hashlib.sha256(prompt_template().encode()).hexdigest()[:12]


def _prior_block(prior: list[dict]) -> str:
    """Numbered, because the model returns a number rather than a title."""
    if not prior:
        return "(none yet)"
    window = prior[-PRIOR_WINDOW:]
    lines = []
    for memory in window:
        summary = (memory.get("summary") or "").replace("\n", " ")[:160]
        lines.append(f"{memory['number']}. {memory['title']} — {summary}")
    return "\n".join(lines)


class ExtractionCache:
    """Disk cache. Writes are atomic so a crash cannot leave a half-file."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str, index: int, fingerprint: str) -> Path:
        directory = self.root / fingerprint / conversation_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{index:05d}.json"

    def get(self, conversation_id: str, index: int, fingerprint: str) -> dict | None:
        path = self._path(conversation_id, index, fingerprint)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def put(self, conversation_id: str, index: int, fingerprint: str, record: dict) -> None:
        path = self._path(conversation_id, index, fingerprint)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record))
        temporary.replace(path)


def _to_extraction(payload: dict, anchor: str | None) -> Extraction:
    facts = [
        {
            "subject": str(fact.get("subject", "")).strip(),
            "predicate": str(fact.get("predicate", "")).strip(),
            "object": str(fact.get("object", "")).strip(),
            "confidence": EXTRACTED_FACT_CONFIDENCE,
            "evidence": ["conversation turn"],
        }
        for fact in (payload.get("facts") or [])
        if str(fact.get("subject", "")).strip() and str(fact.get("predicate", "")).strip()
    ]
    if not facts:
        # No facts, nothing written. This is how the extractor says "this
        # exchange establishes nothing" — there is no separate flag for it.
        return Extraction()

    title = (payload.get("title") or "").strip()
    content = (payload.get("content_md") or "").strip()
    if not title or not content:
        return Extraction(error="extraction produced facts but no title or content")

    memory_type = payload.get("memory_type")
    if memory_type not in MEMORY_TYPES:
        # Fall back to a type the service accepts. Falling back to an unaccepted
        # name turns a recoverable mistake into a paid-for refusal.
        memory_type = FALLBACK_TYPE

    def _index(value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    delta = {
        "memory_type": memory_type,
        "title": title,
        "facts": facts[:MAX_FACTS],
        "entities": [
            {"entity_type": "topic", "name": str(name).strip()}
            for name in (payload.get("entities") or [])
            if str(name).strip()
        ][:MAX_ENTITIES],
        "evidence": [{"kind": "conversation_turn", "reference": anchor or "unanchored"}],
        "temporal": {"valid_from": anchor} if anchor else {},
    }
    return Extraction(
        delta=delta,
        title=title,
        content_md=content,
        supersedes_index=_index(payload.get("supersedes")),
        contradicts_indices=[
            n for n in (_index(v) for v in (payload.get("contradicts") or [])) if n is not None
        ],
    )


def extract(
    chunk: dict,
    *,
    conversation_id: str,
    index: int,
    prior: list[dict],
    cache: ExtractionCache,
    spend: Spend | None = None,
) -> Extraction:
    """One exchange to one delta. Cached before the call, never after a failure."""
    fingerprint = prompt_fingerprint()
    cached = cache.get(conversation_id, index, fingerprint)
    if cached is not None and not cached.get("error"):
        extraction = _to_extraction(cached.get("payload") or {}, chunk.get("anchor"))
        extraction.from_cache = True
        return extraction

    exchange = "\n\n".join(render(message) for message in chunk["messages"])
    rendered = (
        prompt_template()
        .replace("{anchor}", chunk.get("anchor") or "(unknown)")
        .replace("{prior}", _prior_block(prior))
        .replace("{exchange}", exchange)
    )

    result = complete(
        model=settings.models.extractor,
        messages=[{"role": "user", "content": rendered}],
        temperature=settings.models.extractor_temperature,
        max_tokens=settings.models.extractor_max_tokens,
        stage="extract",
        spend=spend,
        response_format={"type": "json_object"},
    )
    if result.error:
        return Extraction(error=result.error)

    try:
        payload = parse_json_object(result.text)
    except json.JSONDecodeError as exc:
        # Not repaired. A repaired extraction is an invented one. The raw text is
        # kept so a later pass can recover it without re-paying for the call.
        cache.put(
            conversation_id,
            index,
            fingerprint,
            {"error": f"unparseable JSON: {exc}", "raw": result.text},
        )
        return Extraction(error=f"unparseable JSON: {exc}")

    cache.put(conversation_id, index, fingerprint, {"payload": payload})
    return _to_extraction(payload, chunk.get("anchor"))
