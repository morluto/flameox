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
    NativeReductionLimits,
    NativeReductionResult,
    StructuredPartitioning,
)


def test_native_reduction_rejects_incoherent_outcome_state() -> None:
    with pytest.raises(ValidationError):
        NativeReductionAttempt(
            attempt_id="attempt",
            candidate_digest="sha256:" + "0" * 64,
            candidate_size_bytes=1,
            classification="unresolved",
            cache_status="miss",
            duration_ms=0,
            became_best=True,
        )

    with pytest.raises(ValidationError):
        NativeReductionResult(
            disposition="succeeded",
            original_digest="sha256:" + "0" * 64,
            final_digest="sha256:" + "0" * 64,
            original_unit_count=1,
            final_unit_count=1,
            minimality="one_minimal",
            final_revalidation="interesting",
        )


def test_binary_partitioning_carries_its_required_chunk_size() -> None:
    reducer = NativeDdminReducer(BinaryChunkPartitioning(chunk_size=2))

    result = reducer.reduce(
        b"KEEPxx", lambda payload: "interesting" if b"KEEP" in payload else "not_interesting"
    )

    assert result.final_payload == b"KEEP"
    assert result.minimality == "one_minimal"


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
        return "interesting" if b"KEEP\n" in payload else "not_interesting"

    reducer = NativeDdminReducer(
        StructuredPartitioning(partitioner="text_lines"),
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
    result = NativeDdminReducer(StructuredPartitioning(partitioner="text_lines")).reduce(
        b"KEEP\nother\n", lambda _payload: "unresolved"
    )

    assert result.disposition == "inconclusive"
    assert result.final_revalidation == "unresolved"
    assert not any(attempt.became_best for attempt in result.attempts)


def test_single_unit_reduction_tests_empty_selection() -> None:
    result = NativeDdminReducer(StructuredPartitioning(partitioner="text_lines")).reduce(
        b"KEEP\n",
        lambda _payload: "interesting",
    )

    assert result.final_payload == b""
    assert result.minimality == "one_minimal"
    assert any(attempt.candidate_size_bytes == 0 for attempt in result.attempts)


def test_json_normalization_incompatibility_preserves_original() -> None:
    original = b'{ "keep": true, "discard": false }'
    result = NativeDdminReducer(StructuredPartitioning(partitioner="json_top_level")).reduce(
        original,
        lambda payload: "interesting" if payload == original else "not_interesting",
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
    result = NativeDdminReducer(StructuredPartitioning(partitioner="chrome_trace_events")).reduce(
        original,
        lambda payload: "interesting" if b'"name":"end"' in payload else "not_interesting",
    )

    assert result.final_revalidation == "interesting"
    assert result.final_payload is not None
    final_events = json.loads(result.final_payload)["traceEvents"]
    assert {event["name"] for event in final_events} >= {"begin", "end"}
    assert result.budget_exhausted is False
    assert result.minimality == "one_minimal"
    assert any(attempt.dependency_added_unit_ids for attempt in result.attempts)
