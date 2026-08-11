from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from flameox.application.native_reducer import (
    BinaryChunkPartitioning,
    NativeDdminReducer,
    NativePredicateClassification,
    NativeReductionAttempt,
    NativeReductionCacheStatus,
    NativeReductionDisposition,
    NativeReductionLimits,
    NativeReductionMinimality,
    NativeReductionPartitioner,
    NativeReductionResult,
    StructuredPartitioning,
)


def test_native_reduction_rejects_incoherent_outcome_state() -> None:
    with pytest.raises(ValidationError):
        NativeReductionAttempt(
            attempt_id="attempt",
            candidate_digest="sha256:" + "0" * 64,
            candidate_size_bytes=1,
            classification=NativePredicateClassification.UNRESOLVED,
            cache_status=NativeReductionCacheStatus.MISS,
            duration_ms=0,
            became_best=True,
        )

    with pytest.raises(ValidationError):
        NativeReductionResult(
            disposition=NativeReductionDisposition.SUCCEEDED,
            original_digest="sha256:" + "0" * 64,
            final_digest="sha256:" + "0" * 64,
            original_unit_count=1,
            final_unit_count=1,
            minimality=NativeReductionMinimality.ONE_MINIMAL,
            final_revalidation=NativePredicateClassification.INTERESTING,
        )


def test_binary_partitioning_carries_its_required_chunk_size() -> None:
    reducer = NativeDdminReducer(BinaryChunkPartitioning(chunk_size=2))

    result = reducer.reduce(
        b"KEEPxx",
        lambda payload: (
            NativePredicateClassification.INTERESTING
            if b"KEEP" in payload
            else NativePredicateClassification.NOT_INTERESTING
        ),
    )

    assert result.final_payload == b"KEEP"
    assert result.minimality == "one_minimal"


def test_predicate_results_are_parsed_before_enum_identity_checks() -> None:
    result = NativeDdminReducer(
        StructuredPartitioning(partitioner=NativeReductionPartitioner.TEXT_LINES)
    ).reduce(
        b"KEEP\ndiscard\n",
        lambda payload: "interesting" if b"KEEP" in payload else "not_interesting",
    )

    assert result.final_payload == b"KEEP\n"
    assert result.final_revalidation is NativePredicateClassification.INTERESTING
    assert result.accepted_best_payloads == (b"KEEP\n",)


@settings(max_examples=20, deadline=None)
@given(
    st.lists(
        st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs",),  # type: ignore[arg-type]
                blacklist_characters="\r\n",
            ),
            max_size=12,
        ),
        max_size=7,
    )
)
def test_text_reduction_is_deterministic_and_preserves_interesting_candidates(
    extra_lines: list[str],
) -> None:
    original = ("KEEP\n" + "".join(f"{line}\n" for line in extra_lines)).encode()

    def predicate(payload: bytes) -> NativePredicateClassification:
        return (
            NativePredicateClassification.INTERESTING
            if b"KEEP\n" in payload
            else NativePredicateClassification.NOT_INTERESTING
        )

    reducer = NativeDdminReducer(
        StructuredPartitioning(partitioner=NativeReductionPartitioner.TEXT_LINES),
        limits=NativeReductionLimits(max_attempts=100, wall_time_seconds=5),
    )
    first = reducer.reduce(original, predicate)
    second = reducer.reduce(original, predicate)

    def signature(result: object) -> list[tuple[object, ...]]:
        attempts = result.attempts  # type: ignore[attr-defined]
        return [
            (
                attempt.candidate_digest,
                attempt.requested_unit_ids,
                attempt.removed_unit_ids,
                attempt.classification,
                attempt.became_best,
            )
            for attempt in attempts
        ]

    assert first.final_revalidation == "interesting"
    assert second.final_revalidation == "interesting"
    assert first.final_payload is not None and b"KEEP\n" in first.final_payload
    assert second.final_payload == first.final_payload
    assert first.final_digest == second.final_digest
    assert signature(first) == signature(second)
    assert all(
        attempt.classification != "unresolved" or not attempt.became_best
        for attempt in first.attempts
    )


def test_unresolved_candidates_are_never_accepted() -> None:
    result = NativeDdminReducer(
        StructuredPartitioning(partitioner=NativeReductionPartitioner.TEXT_LINES)
    ).reduce(
        b"KEEP\nother\n",
        lambda _payload: NativePredicateClassification.UNRESOLVED,
    )

    assert result.disposition == "inconclusive"
    assert result.final_revalidation == "unresolved"
    assert not any(attempt.became_best for attempt in result.attempts)


def test_single_unit_reduction_tests_empty_selection() -> None:
    result = NativeDdminReducer(
        StructuredPartitioning(partitioner=NativeReductionPartitioner.TEXT_LINES)
    ).reduce(
        b"KEEP\n",
        lambda _payload: NativePredicateClassification.INTERESTING,
    )

    assert result.final_payload == b""
    assert result.minimality == "one_minimal"
    assert any(attempt.candidate_size_bytes == 0 for attempt in result.attempts)


def test_json_normalization_incompatibility_preserves_original() -> None:
    original = b'{ "keep": true, "discard": false }'
    result = NativeDdminReducer(
        StructuredPartitioning(partitioner=NativeReductionPartitioner.JSON_TOP_LEVEL)
    ).reduce(
        original,
        lambda payload: (
            NativePredicateClassification.INTERESTING
            if payload == original
            else NativePredicateClassification.NOT_INTERESTING
        ),
    )

    assert result.disposition == "inconclusive"
    assert result.minimality == "partitioner_incompatible"
    assert result.final_payload == original
    assert result.final_revalidation == "not_interesting"


def test_chrome_trace_duration_dependencies_are_kept_together() -> None:
    original = json.dumps(
        {
            "traceEvents": [
                {"name": "begin", "ph": "b", "pid": 1, "tid": 1, "id": "x"},
                {"name": "end", "ph": "e", "pid": 1, "tid": 1, "id": "x"},
                {"name": "discard", "ph": "i", "pid": 1, "tid": 1},
            ],
            "displayTimeUnit": "ns",
        },
        separators=(",", ":"),
    ).encode()
    result = NativeDdminReducer(
        StructuredPartitioning(partitioner=NativeReductionPartitioner.CHROME_TRACE_EVENTS)
    ).reduce(
        original,
        lambda payload: (
            NativePredicateClassification.INTERESTING
            if b'"name":"end"' in payload
            else NativePredicateClassification.NOT_INTERESTING
        ),
    )

    assert result.final_revalidation == "interesting"
    assert result.final_payload is not None
    final_events = json.loads(result.final_payload)["traceEvents"]
    assert {event["name"] for event in final_events} >= {"begin", "end"}
    assert result.budget_exhausted is False
    assert result.minimality == "one_minimal"
    assert any(attempt.dependency_added_unit_ids for attempt in result.attempts)
