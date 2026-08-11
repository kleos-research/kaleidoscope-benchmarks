"""BEAM loading, and the scoping the rest of the benchmark depends on.

BEAM (ICLR 2026) evaluates ten memory abilities over long conversations, in
tiers from 100K to 10M tokens. Each conversation carries probing questions with
a rubric, and on nine of the ten abilities a set of `source_chat_ids` naming the
messages that answer them.

Those labels are what make retrieval scorable **with no model in the path** —
see `metrics.py`. They are the most valuable thing in the dataset and the
easiest to misuse.

## The trap

**Message ids are per-conversation indices, not global identifiers.** In the
100K tier, 392 distinct ids cover 5,732 messages, so conversation 1's message 14
and conversation 5's message 14 are unrelated. Any global `id -> message` map
scores evidence against the wrong conversation's text and **silently inflates
recall** — no error, just a number that is too high.

Everything here is therefore scoped to a conversation, and
`assert_conversations_are_isolated()` checks it rather than trusting it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# The four field names BEAM uses for evidence, depending on ability.
EVIDENCE_FIELDS: tuple[str, ...] = (
    "source_chat_ids",
    "source_chat_id",
    "conversation_references",
    "evidence_chat_ids",
)

ABILITIES: tuple[str, ...] = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)

# `event_ordering` is not scored by the rubric judge. BEAM reports normalised
# Kendall tau for it, and so do we — see `judge.py`.
TAU_ABILITY = "event_ordering"


@dataclass(frozen=True)
class Question:
    ability: str
    text: str
    ideal: str
    rubric: list[str]
    evidence_ids: list[int]
    index: int
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def is_abstention(self) -> bool:
        """No answer exists. BEAM ships these with no evidence, by construction."""
        return self.ability == "abstention"


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    sessions: list[list[dict]]
    questions: list[Question]

    @property
    def messages(self) -> list[dict]:
        return [message for session in self.sessions for message in session]

    @property
    def message_ids(self) -> set[int]:
        return {m["id"] for m in self.messages if m.get("id") is not None}


def _collect_ids(value: Any, into: list[int]) -> None:
    """Walk any nesting of lists and dicts, collecting every numeric leaf.

    **Recursive on purpose.** BEAM nests `source_chat_ids` a second level on some
    `event_ordering` questions to group ids belonging to one step. A loader that
    flattens exactly one level discards the inner lists, which silently removes
    those questions' evidence — and a question that reads as having no evidence
    drops out of every recall denominator computed from the field.

    `bool` is excluded before the numeric test because `isinstance(True, int)` is
    true in Python and a flag would otherwise be collected as message id 1.
    """
    if isinstance(value, dict):
        for item in value.values():
            _collect_ids(item, into)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_ids(item, into)
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        into.append(int(value))


def evidence_ids(raw: dict) -> list[int]:
    found: list[int] = []
    for key in EVIDENCE_FIELDS:
        if key in raw:
            _collect_ids(raw[key], found)
    return sorted(set(found))


def _plain(value: Any) -> Any:
    """Parquet cells arrive as Python-literal strings, not JSON."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            import ast

            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
    return value


def load_tier(tier: str, data_dir: Path) -> list[Conversation]:
    """Load one BEAM tier from `beam-{tier}.parquet`.

    Download instructions are in the benchmark README. The file is not vendored:
    BEAM is not ours to redistribute.
    """
    import pandas as pd

    path = Path(data_dir) / f"beam-{tier}.parquet"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. See benchmarks/beam/README.md for how to fetch the "
            f"tier from Hugging Face."
        )

    frame = pd.read_parquet(path)
    conversations: list[Conversation] = []
    for position, row in frame.iterrows():
        sessions = _plain(row["chat"])
        by_ability = _plain(row["probing_questions"])
        questions: list[Question] = []
        for ability, items in by_ability.items():
            for item in items:
                questions.append(
                    Question(
                        ability=ability,
                        text=item.get("question", ""),
                        ideal=item.get("ideal_response", "") or item.get("answer", ""),
                        rubric=list(item.get("rubric") or []),
                        evidence_ids=evidence_ids(item),
                        index=len(questions),
                        raw=item,
                    )
                )
        conversations.append(
            Conversation(
                conversation_id=str(row.get("conversation_id", position + 1)),
                sessions=sessions,
                questions=questions,
            )
        )
    return conversations


def render(message: dict) -> str:
    """One message as the text every stage sees.

    Role-prefixed once, here, so no downstream stage re-prefixes it. A double
    prefix (`user: user: ...`) is invisible in a diff and changes what the
    extractor and the retriever both see.
    """
    role = message.get("role", "user")
    content = message.get("content", "")
    return f"{role}: {content}".strip()


def time_anchor(message: dict) -> str | None:
    """The session date, as an RFC 3339 instant, or None if unparseable.

    Failing to None rather than raising is deliberate: BEAM's anchors are not
    uniformly formatted, and an unparseable one should cost that memory its
    temporal validity, not kill the ingest.
    """
    raw = message.get("time_anchor")
    if not raw:
        return None
    from datetime import datetime

    for pattern in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d %B %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(str(raw).strip(), pattern).strftime("%Y-%m-%dT00:00:00Z")
        except ValueError:
            continue
    return None


def pack(sessions: Iterable[list[dict]], chunk_size: int = 2) -> list[dict]:
    """Group messages into the windows the extractor sees.

    One turn pair per chunk by default, which is the unit BEAM's own baselines
    and mem0's runner both use. Chunks never span a session boundary, so a
    chunk's `anchor` is unambiguous.
    """
    chunks: list[dict] = []
    for session_index, session in enumerate(sessions):
        for start in range(0, len(session), chunk_size):
            window = session[start : start + chunk_size]
            if not window:
                continue
            anchor = next((time_anchor(m) for m in window if time_anchor(m)), None)
            chunks.append(
                {
                    "messages": window,
                    "session_index": session_index,
                    "anchor": anchor,
                    "message_ids": [m.get("id") for m in window if m.get("id") is not None],
                }
            )
    return chunks


def assert_conversations_are_isolated(conversations: list[Conversation]) -> dict:
    """Check the property every score depends on, and report what it found.

    Raises if any question's evidence points outside its own conversation. That
    would mean the dataset is not what this harness assumes, and every recall
    number computed afterwards would be wrong.
    """
    owner: dict[int, str] = {}
    reused = 0
    for conversation in conversations:
        for message_id in conversation.message_ids:
            if message_id in owner and owner[message_id] != conversation.conversation_id:
                reused += 1
            owner.setdefault(message_id, conversation.conversation_id)

    stray = [
        (conversation.conversation_id, question.index, evidence_id)
        for conversation in conversations
        for question in conversation.questions
        for evidence_id in question.evidence_ids
        if evidence_id not in conversation.message_ids
    ]
    if stray:
        raise SystemExit(
            f"{len(stray)} evidence ids point outside their own conversation, e.g. "
            f"{stray[:3]}. Evidence scoring assumes per-conversation scoping; this "
            f"dataset does not satisfy it."
        )

    labelled = sum(
        1 for c in conversations for q in c.questions if q.evidence_ids
    )
    return {
        "conversations": len(conversations),
        "messages": sum(len(c.messages) for c in conversations),
        "questions": sum(len(c.questions) for c in conversations),
        "questions_with_evidence": labelled,
        "distinct_message_ids": len(owner),
        "ids_reused_across_conversations": reused,
    }
