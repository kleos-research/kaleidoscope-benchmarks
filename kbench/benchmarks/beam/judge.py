"""Phase 3 — score the answers.

Decoupled from phase 2 on purpose. Judging is the expensive half and the part
most likely to be re-run: a new judge model, a corrected rubric, a second
opinion. Keeping it separate means none of that costs a retrieval or a reader
call, and a judge outage cannot lose a run's answers.

Two scoring paths, because BEAM has two:

* **Rubric abilities (nine of ten)** — one judge call per rubric item, scored
  present or absent, averaged. That is BEAM's own shape; a single call asked to
  score the whole rubric at once is a different measurement.
* **`event_ordering`** — normalised Kendall tau between the order the answer
  states and the reference order, not a rubric judgement.

Every row carries the model and effort that produced it. A score stamped with
the constant currently in the config rather than the call that ran is how two
judges end up silently averaged into one column.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ...config import settings
from ...llm import Spend, complete, parse_json_object
from .dataset import TAU_ABILITY

JUDGE_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.md"
TAU_PROMPT_PATH = Path(__file__).parent / "prompts" / "tau_align.md"


@dataclass
class Score:
    conversation_id: str
    question_index: int
    ability: str
    score: float | None = None
    rubric_hits: int = 0
    rubric_total: int = 0
    per_item: list[bool] = field(default_factory=list)
    judge_model: str = ""
    method: str = "rubric"
    error: str = ""

    def as_dict(self) -> dict:
        return {**self.__dict__}


def _judge_one_item(question: str, rubric_item: str, hypothesis: str, spend: Spend) -> bool | None:
    prompt = (
        JUDGE_PROMPT_PATH.read_text()
        .replace("{question}", question)
        .replace("{rubric_item}", rubric_item)
        .replace("{response}", hypothesis)
    )
    result = complete(
        model=settings.models.judge,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.models.judge_temperature,
        stage="judge",
        spend=spend,
        response_format={"type": "json_object"},
    )
    if result.error:
        return None
    try:
        payload = parse_json_object(result.text)
    except json.JSONDecodeError:
        return None
    return bool(payload.get("satisfied"))


def score_rubric(record: dict, rubric: list[str], spend: Spend) -> Score:
    score = Score(
        conversation_id=record["conversation_id"],
        question_index=record["question_index"],
        ability=record["ability"],
        judge_model=settings.models.judge,
        rubric_total=len(rubric),
    )
    if not rubric:
        score.error = "no rubric"
        return score
    if not (record.get("hypothesis") or "").strip():
        # An empty answer scores zero rather than being dropped. Dropping it
        # would remove the arm's failures from its own mean.
        score.score = 0.0
        score.per_item = [False] * len(rubric)
        return score

    for item in rubric:
        satisfied = _judge_one_item(record["question"], item, record["hypothesis"], spend)
        if satisfied is None:
            score.error = "judge failed on at least one rubric item"
            return score
        score.per_item.append(satisfied)
    score.rubric_hits = sum(score.per_item)
    score.score = score.rubric_hits / len(rubric)
    return score


def score_tau(record: dict, reference: list[str], spend: Spend) -> Score:
    """Normalised Kendall tau for `event_ordering`.

    The aligner that matches response lines to reference events is itself a
    model call, so switching the judge model changes this ability through a
    different mechanism than it changes the other nine. Worth remembering when
    reading a judge comparison: the rubric abilities and this one can move in
    opposite directions for the same config change.
    """
    score = Score(
        conversation_id=record["conversation_id"],
        question_index=record["question_index"],
        ability=record["ability"],
        judge_model=settings.models.judge,
        method="tau_norm",
    )
    if not reference or not (record.get("hypothesis") or "").strip():
        score.score = 0.0
        return score

    prompt = (
        TAU_PROMPT_PATH.read_text()
        .replace("{reference}", "\n".join(f"{i + 1}. {e}" for i, e in enumerate(reference)))
        .replace("{response}", record["hypothesis"])
    )
    result = complete(
        model=settings.models.judge,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.models.judge_temperature,
        stage="judge_tau",
        spend=spend,
        response_format={"type": "json_object"},
    )
    if result.error:
        score.error = result.error
        return score
    try:
        payload = parse_json_object(result.text)
    except json.JSONDecodeError as exc:
        score.error = f"unparseable alignment: {exc}"
        return score

    positions = [int(p) for p in (payload.get("reference_positions") or []) if p]
    if len(positions) < 2:
        score.score = 0.0
        return score

    concordant = discordant = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if positions[i] < positions[j]:
                concordant += 1
            elif positions[i] > positions[j]:
                discordant += 1
    pairs = concordant + discordant
    tau = (concordant - discordant) / pairs if pairs else 0.0
    # Normalised to [0, 1] and scaled by how much of the reference was matched:
    # a perfectly ordered answer covering two of ten events is not a perfect one.
    coverage = len(positions) / max(1, len(reference))
    score.score = max(0.0, (tau + 1.0) / 2.0) * coverage
    return score


def run(
    answers_path: Path,
    *,
    rubrics: dict[tuple[str, int], list[str]],
    references: dict[tuple[str, int], list[str]],
    out_path: Path,
    workers: int | None = None,
) -> tuple[list[Score], Spend]:
    """Phase 3 over a captured answers file. Never touches the vault."""
    workers = workers or settings.concurrency.judging
    records = [json.loads(line) for line in answers_path.read_text().splitlines() if line.strip()]
    spend = Spend()
    scores: list[Score] = []
    lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("w")

    def one(record: dict) -> Score:
        key = (record["conversation_id"], record["question_index"])
        if record["ability"] == TAU_ABILITY:
            return score_tau(record, references.get(key, []), spend)
        return score_rubric(record, rubrics.get(key, []), spend)

    print(f"judging {len(records)} answers with {settings.models.judge}, {workers} at a time")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, record) for record in records]
            for future in as_completed(futures):
                score = future.result()
                with lock:
                    handle.write(json.dumps(score.as_dict()) + "\n")
                    handle.flush()
                    scores.append(score)
    finally:
        handle.close()
    return scores, spend
