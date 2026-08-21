"""Tables, from whatever phases have run.

Reports both scores side by side and never averages them, because they measure
different things:

* **evidence recall** — model-free, from BEAM's labels. Retrieval quality alone.
* **BEAM score** — the judged rubric mean, comparable to published numbers.

A retrieval change can move one and not the other. If evidence recall rises and
the BEAM score does not, retrieval was not the bottleneck — that is a finding,
and collapsing the two columns would hide it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ...config import settings
from .dataset import ABILITIES, Conversation, render
from .metrics import ConversationEvidence


def _read_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(
    conversations: list[Conversation],
    *,
    answers_path: Path,
    scores_path: Path | None = None,
    ingest_path: Path | None = None,
) -> str:
    answers = _read_jsonl(answers_path)
    scores = _read_jsonl(scores_path)
    by_key = {(s["conversation_id"], s["question_index"]): s for s in scores}

    evidence = {c.conversation_id: ConversationEvidence(c.messages, render) for c in conversations}

    recall_by_ability: dict[str, list[float]] = defaultdict(list)
    score_by_ability: dict[str, list[float]] = defaultdict(list)
    unlabelled = 0

    for record in answers:
        ability = record["ability"]
        index = evidence.get(record["conversation_id"])
        if index is not None:
            value = index.recall(record.get("evidence_ids") or [], record.get("context") or "")
            if value is None:
                unlabelled += 1
            else:
                recall_by_ability[ability].append(value)
        scored = by_key.get((record["conversation_id"], record["question_index"]))
        if scored and scored.get("score") is not None:
            score_by_ability[ability].append(scored["score"])

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    lines: list[str] = ["# BEAM — Kaleidoscope", ""]

    if ingest_path and ingest_path.exists():
        ingested = json.loads(ingest_path.read_text())
        totals = ingested["totals"]
        candidate = ingested.get("candidate") or {}
        lines += [
            "## Memory",
            "",
            "| | |",
            "| --- | ---: |",
            f"| extractor | `{ingested['extractor']}` |",
            f"| exchanges | {totals['chunks']} |",
            f"| memories written | {totals['written']} |",
            f"| judged not durable | {totals['skipped_not_durable']} |",
            f"| failed | {totals['failed']} |",
            f"| contradicts edges | {totals['contradicts']} |",
            f"| extraction spend | ${ingested['spend_usd']:.2f} |",
            f"| candidate SHA-256 | `{candidate.get('executable_sha256', 'unbound')}` |",
            f"| public-contract SHA-256 | `{candidate.get('public_contract_sha256', 'unbound')}` |",
            f"| signature verified | `{candidate.get('signature_verified', False)}` |",
            f"| release evidence claimed | `{ingested.get('release_evidence_claimed', False)}` |",
            "",
        ]
        if totals["skipped_not_durable"]:
            share = totals["skipped_not_durable"] / max(1, totals["chunks"])
            lines += [
                (
                    f"> {share:.0%} of exchanges were judged not durable and never written. "
                    "That is a write-path decision made before retrieval ever runs, and it "
                    "bounds every number below — see AGENTS.md on `worth_remembering`."
                ),
                "",
            ]

    lines += [
        "## Results",
        "",
        "`evidence recall` is model-free, computed from BEAM's own `source_chat_ids`.",
        "`BEAM score` is the judged rubric mean. They measure different things and are",
        "never averaged together.",
        "",
        "| ability | evidence recall | BEAM score |",
        "| --- | ---: | ---: |",
    ]
    for ability in ABILITIES:
        recall = mean(recall_by_ability.get(ability, []))
        score = mean(score_by_ability.get(ability, []))
        lines.append(
            f"| {ability} | "
            f"{'—' if recall is None else f'{recall:.3f}'} | "
            f"{'—' if score is None else f'{score:.3f}'} |"
        )

    overall_recall = mean([v for values in recall_by_ability.values() for v in values])
    ability_means = [
        m for m in (mean(score_by_ability.get(a, [])) for a in ABILITIES) if m is not None
    ]
    headline = sum(ability_means) / len(ability_means) if ability_means else None
    lines += [
        (
            f"| **overall** | "
            f"**{'—' if overall_recall is None else f'{overall_recall:.3f}'}** | "
            f"**{'—' if headline is None else f'{headline:.3f}'}** |"
        ),
        "",
        "The headline is the mean of the ten ability means, which is how BEAM reports it —",
        "not the mean over questions. Abstention is one ability of ten, so a system that",
        "answers nothing scores 1.000 there and near zero everywhere else.",
        "",
    ]

    if unlabelled:
        lines += [
            f"> {unlabelled} questions carry no evidence labels and are excluded from the",
            "> recall column rather than scored zero. Forty of them are `abstention`, which",
            "> has no evidence by construction.",
            "",
        ]

    lines += [
        "## Configuration",
        "",
        "| | |",
        "| --- | --- |",
        f"| extractor | `{settings.models.extractor}` |",
        f"| reader | `{settings.models.reader}` |",
        f"| judge | `{settings.models.judge}` |",
        "",
        "Keep the reader identical across arms you intend to compare. A reader difference",
        "is indistinguishable from a memory difference in the final score.",
        "",
    ]
    return "\n".join(lines)
