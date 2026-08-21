"""Turn one exchange into a semantic delta.

This is the write path's only model call. The contract it implements is
`AGENTS.md`; the prompt it renders is `prompts/extraction.md`.

The runtime supplies the declarable ``memory_type`` vocabulary through its
operator-only ontology read. The harness passes that list into the prompt and
cache key; it never freezes the workspace vocabulary in source.

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

It goes for redundancy, and no figure is attached to that. How much such a gate
costs is a property of the prompt, not of the idea: two extractions over the
same corpus with different prompts dropped 745 exchanges and 83. A number from
one prompt would read as a fact about gates and is not one.

## 3. Contradicted memories are numbered, not named

The model returns an integer from a bounded prior-memory list. Exact title
matching is a silent no-op on a near miss. The deleted ``supersedes`` write
field is not reconstructed here; current graph semantics derive replacement
from facts, while unresolved disputes use ``contradicts``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ...config import settings
from ...llm import Spend, complete, parse_json_object
from .dataset import render

PROMPT_PATH = Path(__file__).parent / "prompts" / "extraction.md"

# See the module docstring. Constant, and not asked of the model.
EXTRACTED_FACT_CONFIDENCE = 1.0
FACT_MODES = ("fact", "preference", "decision", "procedure", "outcome", "event")
PREDICATE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

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
    contradicts_indices: list[int] = field(default_factory=list)
    error: str | None = None
    from_cache: bool = False

    @property
    def writes(self) -> bool:
        return self.delta is not None


def prompt_template() -> str:
    return PROMPT_PATH.read_text()


def prompt_fingerprint(memory_types: tuple[str, ...]) -> str:
    """Cache key component, so a prompt edit invalidates only its own results."""
    material = prompt_template() + "\n" + json.dumps(memory_types, separators=(",", ":"))
    return hashlib.sha256(material.encode()).hexdigest()[:12]


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


def _to_extraction(
    payload: dict,
    anchor: str | None,
    memory_types: tuple[str, ...],
) -> Extraction:
    if not memory_types:
        return Extraction(error="runtime ontology returned no memory types")

    facts = []
    for raw in payload.get("facts") or []:
        subject = str(raw.get("subject", "")).strip()
        predicate = str(raw.get("predicate", "")).strip()
        object_ = str(raw.get("object", "")).strip()
        mode = str(raw.get("mode", "fact")).strip()
        if not subject or not object_ or PREDICATE.fullmatch(predicate) is None:
            continue
        if mode not in FACT_MODES:
            return Extraction(error=f"extraction produced unsupported fact mode {mode!r}")
        facts.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "mode": mode,
                "basis": "stated",
                "confidence": EXTRACTED_FACT_CONFIDENCE,
            }
        )
    if not facts:
        # No facts, nothing written. This is how the extractor says "this
        # exchange establishes nothing" — there is no separate flag for it.
        return Extraction()

    title = (payload.get("title") or "").strip()
    content = (payload.get("content_md") or "").strip()
    if not title or not content:
        return Extraction(error="extraction produced facts but no title or content")

    memory_type = str(payload.get("memory_type", ""))
    if memory_type not in memory_types:
        return Extraction(error="extraction produced a memory_type outside runtime ontology")

    entities = []
    declared = set()
    for raw in payload.get("entities") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("n", "")).strip()
        kind = str(raw.get("kind", "")).strip()
        gloss = str(raw.get("is", "")).strip()
        if name and kind and gloss and name not in declared:
            entities.append({"n": name, "kind": kind, "is": gloss})
            declared.add(name)
    endpoints = {fact[side] for fact in facts for side in ("subject", "object")}
    if not endpoints.issubset(declared) or len(entities) > MAX_ENTITIES:
        return Extraction(error="every fact endpoint must be declared once as an entity")

    def _index(value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    delta = {
        "memory_type": memory_type,
        "title": title,
        "facts": facts[:MAX_FACTS],
        "entities": entities,
        "evidence": [{"kind": "conversation_turn", "reference": anchor or "unanchored"}],
    }
    if anchor:
        delta["occurred_at"] = {"t": anchor, "grain": "instant"}
    return Extraction(
        delta=delta,
        title=title,
        content_md=content,
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
    memory_types: tuple[str, ...],
    spend: Spend | None = None,
) -> Extraction:
    """One exchange to one delta. Cached before the call, never after a failure."""
    fingerprint = prompt_fingerprint(memory_types)
    cached = cache.get(conversation_id, index, fingerprint)
    if cached is not None and not cached.get("error"):
        extraction = _to_extraction(cached.get("payload") or {}, chunk.get("anchor"), memory_types)
        extraction.from_cache = True
        return extraction

    exchange = "\n\n".join(render(message) for message in chunk["messages"])
    rendered = (
        prompt_template()
        .replace("{memory_types}", json.dumps(memory_types))
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
    return _to_extraction(payload, chunk.get("anchor"), memory_types)
