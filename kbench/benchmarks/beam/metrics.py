"""Model-free retrieval scoring, from BEAM's own labels.

BEAM names the messages that answer a question on nine of its ten abilities. So
retrieval quality is measurable **with no reader and no judge** — which makes it
free, fast, deterministic, and immune to every judge defect.

Use these numbers to iterate on retrieval. Use the judged scores in `report.py`
to compare against published results. They answer different questions and
neither substitutes for the other.

## The scoping rule

Every function here takes a conversation and never a global index. BEAM message
ids are per-conversation, so a global lookup silently scores evidence against
another conversation's text and inflates recall. `dataset.assert_conversations_are_isolated`
checks this at load; these functions are written so a mistake cannot happen
downstream of it either.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from dataclasses import dataclass

MIN_TERM_LENGTH = 3


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens. No regex — unicode categories decide."""
    out: list[str] = []
    current: list[str] = []
    for character in text.lower():
        if unicodedata.category(character)[0] in {"L", "N"}:
            current.append(character)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return {token for token in out if len(token) >= MIN_TERM_LENGTH}


class ConversationEvidence:
    """Per-conversation index: message id -> the terms that distinguish it.

    Retrieval returns memories, not message ids, so an exact id match is not
    available. Instead a message counts as retrieved when its *distinctive*
    terms — those appearing in few of the conversation's messages — show up in
    the retrieved context. That is a proxy and it is documented as one.
    """

    def __init__(self, messages: list[dict], render) -> None:
        self.by_id: dict[int, str] = {}
        documents: list[set[str]] = []
        for message in messages:
            text = render(message)
            message_id = message.get("id")
            if message_id is not None:
                self.by_id[int(message_id)] = text
            documents.append(_tokens(text))

        frequency: Counter[str] = Counter()
        for document in documents:
            frequency.update(document)
        self.document_frequency = frequency
        # Distinctive means rare within this conversation. Two messages, or 5%
        # of the conversation, whichever is larger — so the cut scales with
        # length instead of being a magic constant.
        self.cutoff = max(2, math.ceil(0.05 * max(1, len(documents))))

    def distinctive_terms(self, message_id: int) -> set[str]:
        text = self.by_id.get(int(message_id))
        if text is None:
            return set()
        return {
            term for term in _tokens(text) if self.document_frequency.get(term, 0) <= self.cutoff
        }

    def recall(self, evidence_ids: list[int], context: str, threshold: float = 0.5) -> float | None:
        """Fraction of the evidence messages present in the retrieved context.

        A message counts as present when at least `threshold` of its distinctive
        terms appear. Returns None when the question carries no labels, so the
        caller can exclude it rather than score it zero — an unlabelled question
        is missing data, not a miss.
        """
        if not evidence_ids:
            return None
        context_terms = _tokens(context)
        hits = 0
        counted = 0
        for evidence_id in evidence_ids:
            terms = self.distinctive_terms(evidence_id)
            if not terms:
                continue
            counted += 1
            overlap = len(terms & context_terms) / len(terms)
            if overlap >= threshold:
                hits += 1
        return hits / counted if counted else None


@dataclass
class RetrievalSummary:
    questions_scored: int
    evidence_recall: float
    by_ability: dict[str, float]
    abstention_fired: int
    unlabelled_excluded: int

    def render(self) -> str:
        lines = [
            (
                f"evidence recall over {self.questions_scored} labelled questions: "
                f"{self.evidence_recall:.4f}"
            ),
            f"  ({self.unlabelled_excluded} unlabelled questions excluded, not scored zero)",
            "",
            f"{'ability':28}{'recall':>10}",
        ]
        for ability, value in sorted(self.by_ability.items()):
            lines.append(f"{ability:28}{value:10.4f}")
        if self.abstention_fired:
            lines.append(f"\nabstention fired on {self.abstention_fired} questions")
        return "\n".join(lines)


def summarise(answers: list[dict], evidence_by_conversation: dict) -> RetrievalSummary:
    """Roll per-answer recall up into a report. No model calls."""
    per_ability: dict[str, list[float]] = {}
    scored: list[float] = []
    unlabelled = 0
    abstained = 0

    for record in answers:
        if record.get("abstained"):
            abstained += 1
        index = evidence_by_conversation.get(record["conversation_id"])
        if index is None:
            continue
        value = index.recall(record.get("evidence_ids") or [], record.get("context") or "")
        if value is None:
            unlabelled += 1
            continue
        scored.append(value)
        per_ability.setdefault(record["ability"], []).append(value)

    return RetrievalSummary(
        questions_scored=len(scored),
        evidence_recall=sum(scored) / len(scored) if scored else 0.0,
        by_ability={ability: sum(values) / len(values) for ability, values in per_ability.items()},
        abstention_fired=abstained,
        unlabelled_excluded=unlabelled,
    )
