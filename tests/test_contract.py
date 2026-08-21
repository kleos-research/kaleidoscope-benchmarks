"""Checks that would otherwise be assumptions.

Each of these pins a property some part of the harness depends on. They cost
nothing and run without a key, a binary, or the dataset.
"""

from __future__ import annotations

from kbench.benchmarks.beam import extract
from kbench.benchmarks.beam.dataset import EVIDENCE_FIELDS, _collect_ids
from kbench.benchmarks.beam.metrics import ConversationEvidence

MEMORY_TYPES = ("runtime_type_a", "runtime_type_b")


def test_memory_types_are_runtime_input_not_a_source_constant():
    assert not hasattr(extract, "MEMORY_TYPES")
    assert extract.prompt_fingerprint(MEMORY_TYPES) != extract.prompt_fingerprint(
        ("runtime_type_b",)
    )


def test_the_extractor_is_never_asked_for_a_confidence():
    """A model has no calibrated 0.49 vs 0.50. Asking returns noise."""
    prompt = extract.prompt_template()
    assert "confidence" not in prompt.lower()
    assert extract.EXTRACTED_FACT_CONFIDENCE == 1.0


def test_silence_is_expressed_as_no_facts_not_as_a_flag():
    """`worth_remembering` was redundant with `facts: []` and cost 30% of the
    evidence on BEAM 100K. It must not come back."""
    prompt = extract.prompt_template()
    assert "worth_remembering" not in prompt
    result = extract._to_extraction(
        {"facts": [], "title": "x", "content_md": "y"}, None, MEMORY_TYPES
    )
    assert not result.writes


def test_prior_memories_are_referenced_by_number_not_title():
    """Exact title matching is a silent no-op on a near miss."""
    block = extract._prior_block(
        [{"number": 1, "title": "Sprint one end date", "summary": "ends 2024-03-29"}]
    )
    assert block.startswith("1. Sprint one end date")


def test_facts_carry_the_constant_confidence_the_service_requires():
    result = extract._to_extraction(
        {
            "memory_type": "runtime_type_a",
            "title": "T",
            "content_md": "C",
            "facts": [{"subject": "a", "predicate": "does_b", "object": "c", "mode": "fact"}],
            "entities": [
                {"n": "a", "kind": "actor", "is": "test actor"},
                {"n": "c", "kind": "target", "is": "test target"},
            ],
        },
        "2024-01-01T00:00:00Z",
        MEMORY_TYPES,
    )
    assert result.writes
    assert result.delta["facts"][0]["confidence"] == extract.EXTRACTED_FACT_CONFIDENCE


def test_a_memory_type_outside_runtime_ontology_refuses():
    result = extract._to_extraction(
        {
            "memory_type": "not_a_real_type",
            "title": "T",
            "content_md": "C",
            "facts": [{"subject": "a", "predicate": "does_b", "object": "c"}],
            "entities": [
                {"n": "a", "kind": "actor", "is": "test actor"},
                {"n": "c", "kind": "target", "is": "test target"},
            ],
        },
        None,
        MEMORY_TYPES,
    )
    assert result.error == "extraction produced a memory_type outside runtime ontology"


def test_every_fact_endpoint_requires_a_glossed_entity_declaration():
    result = extract._to_extraction(
        {
            "memory_type": "runtime_type_a",
            "title": "T",
            "content_md": "C",
            "facts": [{"subject": "a", "predicate": "does_b", "object": "c"}],
            "entities": [{"n": "a", "kind": "actor", "is": "test actor"}],
        },
        None,
        MEMORY_TYPES,
    )
    assert result.error == "every fact endpoint must be declared once as an entity"


def test_nested_evidence_ids_are_collected_at_any_depth():
    """BEAM nests source_chat_ids a second level on some event_ordering
    questions. A loader that flattens one level drops them, and a question with
    no evidence silently leaves every recall denominator."""
    found: list[int] = []
    _collect_ids([116, [136, 138], {"a": [202]}], found)
    assert sorted(found) == [116, 136, 138, 202]


def test_booleans_are_not_collected_as_message_ids():
    """isinstance(True, int) is true in Python; a flag would become id 1."""
    found: list[int] = []
    _collect_ids([True, False, 7], found)
    assert found == [7]


def test_evidence_index_holds_only_its_own_conversation():
    """Message ids are per-conversation. A global index inflates recall."""
    render = lambda m: m["content"]
    index = ConversationEvidence(
        [{"id": 14, "content": "quokka telemetry threshold is 42"}], render
    )
    assert 14 in index.by_id
    assert index.distinctive_terms(99) == set()


def test_unlabelled_questions_are_excluded_rather_than_scored_zero():
    render = lambda m: m["content"]
    index = ConversationEvidence([{"id": 1, "content": "anything"}], render)
    assert index.recall([], "some context") is None


def test_every_evidence_field_name_is_checked():
    assert "source_chat_ids" in EVIDENCE_FIELDS
