from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters import (
    AIPerfCorrelationSummary,
    AIPerfInputsIndex,
    AIPerfRecordParser,
    InferenceArtifactExtractor,
)
from flameox.application import EvidenceQueryService, ImportArtifactRequest, ImportService
from flameox.domain import ArtifactKind, DomainError, ErrorCode, Sensitivity
from flameox.domain.identity import new_id
from flameox.domain.models import ArtifactRegistration
from flameox.evidence import InferenceRequestOutcomeKind
from flameox.storage import ArtifactStore, RunStore, Workspace


def test_aiperf_record_parser_normalizes_safe_request_evidence(tmp_path: Path) -> None:
    result = tmp_path / "profile_export.jsonl"
    result.write_text(
        json.dumps(
            {
                "metadata": {
                    "session_num": 7,
                    "x_request_id": "request-7",
                    "conversation_id": "conversation-a",
                    "turn_index": 2,
                    "credit_issued_ns": 100,
                    "request_start_ns": 125,
                    "request_end_ns": 10_000_125,
                    "was_cancelled": False,
                },
                "metrics": {
                    "input_sequence_length": {"value": 20, "unit": "tokens"},
                    "output_sequence_length": {"value": 3, "unit": "tokens"},
                    "time_to_first_token": {"value": 2, "unit": "ms"},
                    "request_latency": {"value": 10, "unit": "ms"},
                    "inter_token_latency": {"value": 4, "unit": "ms"},
                },
                "error": None,
                "raw_prompt": "must never be normalized",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = list(AIPerfRecordParser().iter_rows(result))

    assert len(rows) == 1
    assert rows[0].outcome.kind is InferenceRequestOutcomeKind.SUCCEEDED
    assert {**rows[0].evidence_columns(), "line_index": rows[0].line_index} == {
        "source_request_id": "conversation-a:2",
        "provider_request_id": "request-7",
        "input_tokens": 20,
        "output_tokens": 3,
        "scheduled_ns": 100,
        "observed_started_ns": 125,
        "ttft_ns": 2_000_000,
        "latency_ns": 10_000_000,
        "tpot_ns": 4_000_000,
        "mean_itl_ns": 4_000_000,
        "success": True,
        "cancelled": False,
        "error_type": None,
        "error_code": None,
        "queue_ns": None,
        "prefill_ns": None,
        "decode_ns": None,
        "cache_hit": None,
        "prefix_hash_count": None,
        "evidence_level": "observed",
        "line_index": 0,
    }
    assert "must never be normalized" not in json.dumps(rows[0].evidence_columns())


def test_aiperf_record_parser_keeps_only_safe_error_classification(tmp_path: Path) -> None:
    result = tmp_path / "profile_export.jsonl"
    payload = {
        "metadata": {
            "session_num": 0,
            "request_start_ns": 1,
            "request_end_ns": 2,
            "was_cancelled": True,
        },
        "metrics": {
            "input_sequence_length": {"value": 1, "unit": "tokens"},
            "output_sequence_length": {"value": 0, "unit": "tokens"},
        },
        "error": {"type": "TimeoutError", "code": 408, "message": "secret body"},
    }
    result.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    row = next(AIPerfRecordParser().iter_rows(result))

    assert row.outcome.kind is InferenceRequestOutcomeKind.CANCELLED
    assert row.success is False
    assert row.cancelled is True
    assert row.error_type == "timeout"
    assert row.error_code == "408"
    assert "secret body" not in json.dumps(row.evidence_columns())


def test_aiperf_record_parser_collapses_untrusted_error_identifiers(tmp_path: Path) -> None:
    result = tmp_path / "profile_export.jsonl"
    payload = _aiperf_record(was_cancelled=False)
    payload["error"] = {
        "type": "ApiError credential=sk-super-secret",
        "code": "tenant-secret-code",
        "message": "another secret body",
    }
    result.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    row = next(AIPerfRecordParser().iter_rows(result))

    assert row.outcome.kind is InferenceRequestOutcomeKind.FAILED
    assert row.error_type == "provider_error"
    assert row.error_code == "provider_error"
    serialized = json.dumps(row.evidence_columns())
    assert "super-secret" not in serialized
    assert "tenant-secret" not in serialized
    assert "another secret" not in serialized


def test_aiperf_record_limit_counts_records_not_blank_lines(tmp_path: Path) -> None:
    result = tmp_path / "profile_export.jsonl"
    payload = {
        "metadata": {"session_num": 0, "request_start_ns": 1},
        "metrics": {
            "input_sequence_length": {"value": 1, "unit": "tokens"},
            "output_sequence_length": {"value": 1, "unit": "tokens"},
        },
        "error": None,
    }
    result.write_text("\n\n" + json.dumps(payload) + "\n" + json.dumps(payload) + "\n")
    parser = AIPerfRecordParser(max_rows=1)

    rows = list(parser.iter_rows(result))

    assert len(rows) == 1
    assert parser.truncated is True


def test_aiperf_rejects_unterminated_oversized_line(tmp_path: Path) -> None:
    result = tmp_path / "profile_export.jsonl"
    result.write_bytes(b"{" + (b"x" * (AIPerfRecordParser.max_line_bytes + 1)))

    with pytest.raises(DomainError) as error:
        list(AIPerfRecordParser().iter_rows(result))

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def _write_inputs_json(path: Path, sessions: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"data": sessions}), encoding="utf-8")


def test_inputs_index_builds_session_turn_counts_without_retaining_payloads(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs.json"
    _write_inputs_json(
        inputs,
        [
            {
                "session_id": "session-a",
                "payloads": [
                    {"messages": [{"role": "user", "content": "secret prompt 0"}]},
                    {"messages": [{"role": "user", "content": "secret prompt 1"}]},
                ],
            },
            {
                "session_id": "session-b",
                "payloads": [{"messages": [{"content": "another secret"}]}],
            },
        ],
    )

    index = AIPerfInputsIndex.from_path(inputs)

    assert index.session_count == 2
    assert index.session_turn_counts == {"session-a": 2, "session-b": 1}
    assert index.has_session("session-a") is True
    assert index.has_session("missing") is False
    assert index.has_turn("session-a", 0) is True
    assert index.has_turn("session-a", 1) is True
    assert index.has_turn("session-a", 2) is False
    assert index.has_turn("session-b", 0) is True
    # No payload content is retained on the index object.
    serialized = json.dumps(index.session_turn_counts)
    assert "secret" not in serialized


def test_inputs_index_rejects_missing_data_array(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"not_data": []}), encoding="utf-8")

    with pytest.raises(DomainError) as error:
        AIPerfInputsIndex.from_path(inputs)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_inputs_index_rejects_entry_without_session_id(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    _write_inputs_json(inputs, [{"payloads": []}])

    with pytest.raises(DomainError) as error:
        AIPerfInputsIndex.from_path(inputs)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_inputs_index_rejects_entry_without_payloads_array(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    _write_inputs_json(inputs, [{"session_id": "s"}])

    with pytest.raises(DomainError) as error:
        AIPerfInputsIndex.from_path(inputs)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_inputs_index_rejects_duplicate_session_id(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    _write_inputs_json(
        inputs,
        [{"session_id": "dup", "payloads": [{}]}, {"session_id": "dup", "payloads": [{}]}],
    )

    with pytest.raises(DomainError) as error:
        AIPerfInputsIndex.from_path(inputs)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_inputs_index_rejects_duplicate_session_id_key(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_bytes(b'{"data":[{"session_id":"first","session_id":"second","payloads":[]}]}')

    with pytest.raises(DomainError) as error:
        AIPerfInputsIndex.from_path(inputs)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_inputs_index_rejects_oversized_file(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"data": []}), encoding="utf-8")
    original = AIPerfInputsIndex.max_input_bytes
    AIPerfInputsIndex.max_input_bytes = inputs.stat().st_size - 1
    try:
        with pytest.raises(DomainError) as error:
            AIPerfInputsIndex.from_path(inputs)
    finally:
        AIPerfInputsIndex.max_input_bytes = original

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_inputs_index_rejects_excess_session_count(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    original_max = AIPerfInputsIndex.max_sessions
    AIPerfInputsIndex.max_sessions = 2
    try:
        _write_inputs_json(
            inputs,
            [
                {"session_id": "s0", "payloads": [{}]},
                {"session_id": "s1", "payloads": [{}]},
                {"session_id": "s2", "payloads": [{}]},
            ],
        )
        with pytest.raises(DomainError) as error:
            AIPerfInputsIndex.from_path(inputs)
    finally:
        AIPerfInputsIndex.max_sessions = original_max

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "session limit" in error.value.message


def test_inputs_index_rejects_excess_turns_per_session(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    original_max = AIPerfInputsIndex.max_turns_per_session
    AIPerfInputsIndex.max_turns_per_session = 3
    try:
        _write_inputs_json(
            inputs,
            [{"session_id": "s0", "payloads": [{}, {}, {}, {}]}],
        )
        with pytest.raises(DomainError) as error:
            AIPerfInputsIndex.from_path(inputs)
    finally:
        AIPerfInputsIndex.max_turns_per_session = original_max

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "turns" in error.value.message


def test_inputs_index_streams_large_session_payloads_without_retaining_them(
    tmp_path: Path,
) -> None:
    """Verify the index handles sessions with large payloads incrementally.

    Each payload is ~4 KB of prompt text; 50 turns per session, 3 sessions
    produces ~600 KB. The index retains only turn counts, never payload content.
    """
    inputs = tmp_path / "inputs.json"
    big_prompt = "x" * 4096
    _write_inputs_json(
        inputs,
        [
            {
                "session_id": f"session-{i}",
                "payloads": [{"messages": [{"content": big_prompt}]} for _ in range(50)],
            }
            for i in range(3)
        ],
    )

    index = AIPerfInputsIndex.from_path(inputs)

    assert index.session_count == 3
    assert all(count == 50 for count in index.session_turn_counts.values())
    # No payload content is retained on the index object.
    assert big_prompt not in json.dumps(index.session_turn_counts)


def test_inputs_index_rejects_invalid_json(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(DomainError) as error:
        AIPerfInputsIndex.from_path(inputs)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_inputs_index_rejects_top_level_array(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps([{"session_id": "s", "payloads": []}]), encoding="utf-8")

    with pytest.raises(DomainError) as error:
        AIPerfInputsIndex.from_path(inputs)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_inputs_index_accepts_empty_data_array(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    _write_inputs_json(inputs, [])

    index = AIPerfInputsIndex.from_path(inputs)

    assert index.session_count == 0
    assert index.session_turn_counts == {}


def test_inputs_index_rejects_excess_nesting_depth(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    original = AIPerfInputsIndex.max_nesting_depth
    AIPerfInputsIndex.max_nesting_depth = 4
    try:
        # depth: root(1) > data(2) > session(3) > payloads(4) > payload_obj(5) -> exceeds
        _write_inputs_json(
            inputs,
            [{"session_id": "s", "payloads": [{"messages": [{"content": "deep"}]}]}],
        )
        with pytest.raises(DomainError) as error:
            AIPerfInputsIndex.from_path(inputs)
    finally:
        AIPerfInputsIndex.max_nesting_depth = original

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "nesting" in error.value.message


def test_inputs_index_rejects_oversized_session_id(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    original = AIPerfInputsIndex.max_session_id_length
    AIPerfInputsIndex.max_session_id_length = 4
    try:
        _write_inputs_json(
            inputs,
            [{"session_id": "too-long-id", "payloads": []}],
        )
        with pytest.raises(DomainError) as error:
            AIPerfInputsIndex.from_path(inputs)
    finally:
        AIPerfInputsIndex.max_session_id_length = original

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "session_id" in error.value.message.lower() or "character" in error.value.message


def test_inputs_index_rejects_non_string_session_id(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps({"data": [{"session_id": 123, "payloads": []}]}),
        encoding="utf-8",
    )

    with pytest.raises(DomainError) as error:
        AIPerfInputsIndex.from_path(inputs)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "string session_id" in error.value.message


def test_inputs_index_rejects_non_object_data_item(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps({"data": [123, "str", True]}),
        encoding="utf-8",
    )

    with pytest.raises(DomainError) as error:
        AIPerfInputsIndex.from_path(inputs)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "must be an object" in error.value.message


def test_inputs_index_rejects_session_without_payloads(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps({"data": [{"session_id": "s"}]}),
        encoding="utf-8",
    )

    with pytest.raises(DomainError) as error:
        AIPerfInputsIndex.from_path(inputs)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "payloads" in error.value.message


def test_inputs_index_rejects_non_array_payloads(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps({"data": [{"session_id": "s", "payloads": "notarray"}]}),
        encoding="utf-8",
    )

    with pytest.raises(DomainError) as error:
        AIPerfInputsIndex.from_path(inputs)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert "payloads" in error.value.message


def test_inputs_index_handles_keys_in_any_order(tmp_path: Path) -> None:
    """session_id and payloads may appear in any order within a session object."""
    inputs = tmp_path / "inputs.json"
    # payloads before session_id
    inputs.write_text(
        json.dumps({"data": [{"payloads": [{}, {}], "session_id": "reversed"}]}),
        encoding="utf-8",
    )

    index = AIPerfInputsIndex.from_path(inputs)

    assert index.session_count == 1
    assert index.session_turn_counts == {"reversed": 2}


def test_inputs_index_counts_mixed_payload_types(tmp_path: Path) -> None:
    """Payload items may be objects, arrays, or scalars — all counted as one turn."""
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "session_id": "mixed",
                        "payloads": [
                            {"messages": [{"content": "obj"}]},
                            "scalar-payload",
                            [1, 2, 3],
                            {"text": "another-obj"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    index = AIPerfInputsIndex.from_path(inputs)

    assert index.session_turn_counts == {"mixed": 4}


def _aiperf_record(
    *,
    session_num: int = 0,
    conversation_id: str | None = None,
    turn_index: int | None = None,
    request_start_ns: int = 1,
    was_cancelled: bool = False,
    input_tokens: int = 1,
    output_tokens: int = 1,
    error: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "session_num": session_num,
        "request_start_ns": request_start_ns,
        "was_cancelled": was_cancelled,
    }
    if conversation_id is not None:
        metadata["conversation_id"] = conversation_id
    if turn_index is not None:
        metadata["turn_index"] = turn_index
    return {
        "metadata": metadata,
        "metrics": {
            "input_sequence_length": {"value": input_tokens, "unit": "tokens"},
            "output_sequence_length": {"value": output_tokens, "unit": "tokens"},
        },
        "error": error,
    }


def test_correlation_reports_all_matched_when_every_record_has_input(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    _write_inputs_json(
        inputs,
        [{"session_id": "conv-a", "payloads": [{}, {}]}],
    )
    index = AIPerfInputsIndex.from_path(inputs)

    export = tmp_path / "profile_export.jsonl"
    export.write_text(
        json.dumps(_aiperf_record(conversation_id="conv-a", turn_index=0))
        + "\n"
        + json.dumps(_aiperf_record(conversation_id="conv-a", turn_index=1))
        + "\n",
        encoding="utf-8",
    )
    parser = AIPerfRecordParser()
    list(parser.iter_rows(export, inputs_index=index))

    summary = parser.correlation_summary(index)
    assert summary.matched_count == 2
    assert summary.missing_session_count == 0
    assert summary.turn_out_of_range_count == 0
    assert summary.no_correlation_id_count == 0
    assert summary.limitations == ()


def test_correlation_reports_missing_session(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    _write_inputs_json(inputs, [{"session_id": "conv-a", "payloads": [{}]}])
    index = AIPerfInputsIndex.from_path(inputs)

    export = tmp_path / "profile_export.jsonl"
    export.write_text(
        json.dumps(_aiperf_record(conversation_id="conv-missing", turn_index=0)) + "\n",
        encoding="utf-8",
    )
    parser = AIPerfRecordParser()
    list(parser.iter_rows(export, inputs_index=index))

    summary = parser.correlation_summary(index)
    assert summary.matched_count == 0
    assert summary.missing_session_count == 1
    assert any("no matching session" in lim for lim in summary.limitations)


def test_correlation_reports_turn_out_of_range(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    _write_inputs_json(inputs, [{"session_id": "conv-a", "payloads": [{}]}])
    index = AIPerfInputsIndex.from_path(inputs)

    export = tmp_path / "profile_export.jsonl"
    export.write_text(
        json.dumps(_aiperf_record(conversation_id="conv-a", turn_index=5)) + "\n",
        encoding="utf-8",
    )
    parser = AIPerfRecordParser()
    list(parser.iter_rows(export, inputs_index=index))

    summary = parser.correlation_summary(index)
    assert summary.matched_count == 0
    assert summary.turn_out_of_range_count == 1
    assert any("turn_index outside" in lim for lim in summary.limitations)


def test_correlation_reports_no_conversation_id(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.json"
    _write_inputs_json(inputs, [{"session_id": "conv-a", "payloads": [{}]}])
    index = AIPerfInputsIndex.from_path(inputs)

    export = tmp_path / "profile_export.jsonl"
    export.write_text(
        json.dumps(_aiperf_record(session_num=42)) + "\n",
        encoding="utf-8",
    )
    parser = AIPerfRecordParser()
    list(parser.iter_rows(export, inputs_index=index))

    summary = parser.correlation_summary(index)
    assert summary.matched_count == 0
    assert summary.no_correlation_id_count == 1
    assert any("lacked conversation_id" in lim for lim in summary.limitations)


def test_correlation_without_inputs_index_produces_no_summary_counts(tmp_path: Path) -> None:
    export = tmp_path / "profile_export.jsonl"
    export.write_text(
        json.dumps(_aiperf_record(conversation_id="conv-a", turn_index=0)) + "\n",
        encoding="utf-8",
    )
    parser = AIPerfRecordParser()
    list(parser.iter_rows(export))

    # correlation_summary requires an index; without one, counts stay at zero.
    assert parser._corr_matched == 0


def _add_trace_artifact_to_run(
    workspace: Workspace,
    run_id: str,
    path: Path,
) -> None:
    """Add an INFERENCE_REQUEST_TRACE artifact to an existing run."""
    store = ArtifactStore(workspace)
    stored = store.import_path(
        path,
        allowed_roots=(workspace.project_root, path.absolute().parent),
        max_bytes=workspace.config.capture.max_artifact_bytes,
    )
    runs = RunStore(workspace)
    run = runs.read(run_id)
    registration = ArtifactRegistration(
        registration_id=new_id(),
        run_id=run_id,
        artifact_id=stored.content.artifact_id,
        display_name=path.name,
        media_type="application/json",
        kind=ArtifactKind.INFERENCE_REQUEST_TRACE,
        role="inference_input",
        producer="aiperf",
        sensitivity=Sensitivity.SENSITIVE,
    )
    runs.append(
        run.model_copy(
            update={
                "revision": run.revision + 1,
                "artifacts": (*run.artifacts, registration),
            }
        ),
        expected_revision=run.revision,
    )


def test_extractor_correlates_with_inputs_json_and_surfaces_limitations(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    inputs_path = tmp_path / "inputs.json"
    _write_inputs_json(
        inputs_path,
        [
            {
                "session_id": "conv-a",
                "payloads": [
                    {"messages": [{"content": "secret prompt turn 0"}]},
                    {"messages": [{"content": "secret prompt turn 1"}]},
                ],
            },
        ],
    )

    result_path = tmp_path / "profile_export.jsonl"
    result_path.write_text(
        json.dumps(_aiperf_record(conversation_id="conv-a", turn_index=0, input_tokens=10))
        + "\n"
        + json.dumps(_aiperf_record(conversation_id="conv-missing", turn_index=0))
        + "\n"
        + json.dumps(_aiperf_record(session_num=99))
        + "\n",
        encoding="utf-8",
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=result_path,
            kind=ArtifactKind.INFERENCE_RESULT,
            sensitivity=Sensitivity.SENSITIVE,
            producer="aiperf",
        )
    )
    _add_trace_artifact_to_run(workspace, imported.run.run_id, inputs_path)

    result = InferenceArtifactExtractor(workspace).extract_aiperf_result(imported.run.run_id)

    assert result.request_count == 3
    assert any("no matching session" in lim for lim in result.limitations)
    assert any("lacked conversation_id" in lim for lim in result.limitations)
    # No prompt content from inputs.json leaks into the extraction result.
    assert "secret prompt" not in result.model_dump_json()


def test_extractor_without_inputs_json_skips_correlation(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    result_path = tmp_path / "profile_export.jsonl"
    result_path.write_text(
        json.dumps(_aiperf_record(conversation_id="conv-a", turn_index=0)) + "\n",
        encoding="utf-8",
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=result_path,
            kind=ArtifactKind.INFERENCE_RESULT,
            sensitivity=Sensitivity.SENSITIVE,
            producer="aiperf",
        )
    )

    result = InferenceArtifactExtractor(workspace).extract_aiperf_result(imported.run.run_id)

    assert result.request_count == 1
    assert result.limitations == ()


def test_extractor_rejects_invalid_inputs_json_as_trace_artifact(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text(json.dumps({"not_data": []}), encoding="utf-8")

    result_path = tmp_path / "profile_export.jsonl"
    result_path.write_text(
        json.dumps(_aiperf_record(conversation_id="conv-a", turn_index=0)) + "\n",
        encoding="utf-8",
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=result_path,
            kind=ArtifactKind.INFERENCE_RESULT,
            sensitivity=Sensitivity.SENSITIVE,
            producer="aiperf",
        )
    )
    _add_trace_artifact_to_run(workspace, imported.run.run_id, inputs_path)

    with pytest.raises(DomainError) as error:
        InferenceArtifactExtractor(workspace).extract_aiperf_result(imported.run.run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_correlation_summary_is_typed_contract_model() -> None:
    summary = AIPerfCorrelationSummary(
        inputs_session_count=5,
        matched_count=3,
        missing_session_count=1,
        turn_out_of_range_count=1,
        no_correlation_id_count=0,
    )
    assert summary.schema_version == 1
    assert summary.matched_count == 3


def test_aiperf_extractor_publishes_prompt_free_request_evidence(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    result_path = tmp_path / "profile_export.jsonl"
    payload = {
        "metadata": {
            "session_num": 0,
            "conversation_id": "conversation",
            "turn_index": 0,
            "request_start_ns": 10,
            "was_cancelled": False,
        },
        "metrics": {
            "input_sequence_length": {"value": 4, "unit": "tokens"},
            "output_sequence_length": {"value": 2, "unit": "tokens"},
            "time_to_first_token": {"value": 1, "unit": "ms"},
            "request_latency": {"value": 3, "unit": "ms"},
        },
        "error": None,
        "payload": {"messages": [{"content": "secret"}]},
    }
    result_path.write_text(json.dumps(payload) + "\n")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=result_path,
            kind=ArtifactKind.INFERENCE_RESULT,
            sensitivity=Sensitivity.SENSITIVE,
            producer="aiperf",
        )
    )

    result = InferenceArtifactExtractor(workspace).extract_aiperf_result(imported.run.run_id)
    page = EvidenceQueryService(workspace).inference_requests(run_id=imported.run.run_id, limit=1)

    assert result.request_count == 1
    assert result.measurement_count > 0
    assert page.requests[0].source_request_id == "conversation:0"
    assert page.requests[0].input_tokens == 4
    assert "secret" not in page.model_dump_json()
    measurements = EvidenceQueryService(workspace).measurements(
        run_id=imported.run.run_id, name_prefix="aiperf.", limit=100
    )
    by_name = {
        row.name: row.value.value if row.value is not None else None
        for row in measurements.measurements
    }
    assert by_name["aiperf.ttft.median_ms"] == pytest.approx(1.0)
    assert by_name["aiperf.end_to_end_latency.p95_ms"] == pytest.approx(3.0)


def test_aiperf_extractor_enforces_independent_request_row_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace.initialize(tmp_path)
    result_path = tmp_path / "profile_export.jsonl"
    result_path.write_text(
        json.dumps(_aiperf_record(session_num=0))
        + "\n"
        + json.dumps(_aiperf_record(session_num=1))
        + "\n",
        encoding="utf-8",
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=result_path,
            kind=ArtifactKind.INFERENCE_RESULT,
            producer="aiperf",
            sensitivity=Sensitivity.SENSITIVE,
        )
    )
    monkeypatch.setattr(InferenceArtifactExtractor, "max_request_rows", 1)

    result = InferenceArtifactExtractor(workspace).extract_aiperf_result(imported.run.run_id)

    assert workspace.config.storage.max_rows_per_generation > 1
    assert result.request_count == 1
    assert result.limitations == ("AIPerf request evidence truncated at 1 rows.",)
