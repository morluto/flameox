from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters import (
    AIPerfCorrelationSummary,
    AIPerfInputsIndex,
    AIPerfRecordParser,
    InferenceArtifactExtractor,
    MooncakeRequestRow,
    MooncakeTraceParser,
    SglangResultParser,
    VllmResultParser,
)
from flameox.application import EvidenceQueryService, ImportArtifactRequest, ImportService
from flameox.domain import ArtifactKind, DomainError, ErrorCode, Sensitivity
from flameox.domain.identity import new_id
from flameox.domain.models import ArtifactRegistration
from flameox.evidence import InferenceRequestOutcomeKind
from flameox.storage import ArtifactStore, RunStore, Workspace


def _write_mooncake_trace(path: Path, lines: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n",
        encoding="utf-8",
    )


def _vllm_metrics(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "completed": 313,
        "total_input": 1_000_000,
        "total_output": 50_000,
        "request_throughput": 5.18,
        "request_goodput": 5.18,
        "output_throughput": 827.0,
        "total_token_throughput": 1735.0,
        "mean_ttft_ms": 2187.98,
        "median_ttft_ms": 2100.0,
        "std_ttft_ms": 120.0,
        "percentiles_ttft_ms": [[50.0, 2100.0], [90.0, 2500.0], [99.0, 3000.0]],
        "mean_tpot_ms": 26.59,
        "median_tpot_ms": 25.0,
        "std_tpot_ms": 3.0,
        "percentiles_tpot_ms": [[50.0, 25.0], [90.0, 30.0], [99.0, 35.0]],
        "mean_itl_ms": 26.0,
        "median_itl_ms": 25.0,
        "std_itl_ms": 3.0,
        "percentiles_itl_ms": [[50.0, 25.0], [90.0, 30.0], [99.0, 35.0]],
        "mean_e2el_ms": 5000.0,
        "median_e2el_ms": 4900.0,
        "std_e2el_ms": 200.0,
        "percentiles_e2el_ms": [[50.0, 4900.0], [90.0, 5500.0], [99.0, 6000.0]],
    }
    base.update(overrides)
    return base


def _vllm_result(**overrides: object) -> dict[str, object]:
    metrics = _vllm_metrics(**(overrides.pop("metrics", {})))  # type: ignore[arg-type]
    base: dict[str, object] = {
        "metrics": metrics,
        "successful_requests": 313,
        "failed_requests": 0,
        "total_requests": 313,
        "actual_duration": 60.48,
        "original_time_span": 60.0,
        "time_scale": 1.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Mooncake streaming JSONL validation
# ---------------------------------------------------------------------------


def test_mooncake_parse_preserves_request_shape_and_timing(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {
                "timestamp": 0,
                "input_length": 6755,
                "output_length": 500,
                "hash_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            },
            {
                "timestamp": 3052,
                "input_length": 13538,
                "output_length": 71,
                "hash_ids": [0, 237, 238],
            },
            {
                "timestamp": 6105,
                "input_length": 1048,
                "output_length": 26,
                "hash_ids": [0, 1039, 1040],
            },
        ],
    )

    summary, rows = MooncakeTraceParser().parse(trace)

    assert summary.request_count == 3
    assert summary.max_input_length == 13538
    assert summary.max_output_length == 500
    assert summary.timestamp_span_ms == 6105
    assert summary.prefix_hash_count == 20
    assert summary.limitations == ()
    assert [row.line_index for row in rows] == [0, 1, 2]
    assert [row.timestamp_ms for row in rows] == [0, 3052, 6105]
    assert all(row.evidence_level == "observed" for row in rows)
    assert all(isinstance(row, MooncakeRequestRow) for row in rows)
    assert len({row.request_id for row in rows}) == 3


def test_mooncake_iter_rows_streams_one_at_a_time(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {"timestamp": 0, "input_length": 10, "output_length": 5, "hash_ids": [0]},
            {"timestamp": 100, "input_length": 20, "output_length": 10, "hash_ids": [1, 2]},
        ],
    )

    rows = list(MooncakeTraceParser().iter_rows(trace))

    assert len(rows) == 2
    assert rows[0].prefix_hash_count == 1
    assert rows[1].prefix_hash_count == 2


def test_mooncake_accepts_optional_sensitive_payload_without_normalizing_it(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {
                "timestamp": 0,
                "input_length": 10,
                "output_length": 5,
                "messages": [{"role": "user", "content": "secret prompt"}],
                "tools": [{"name": "private_tool"}],
                "payload": {"private": "body"},
            }
        ],
    )

    _summary, rows = MooncakeTraceParser().parse(trace)

    assert rows[0].prefix_hash_count == 0
    normalized = json.dumps(rows[0].model_dump(mode="json"))
    assert "secret prompt" not in normalized
    assert "private_tool" not in normalized
    assert "body" not in normalized


def test_mooncake_parse_skips_blank_lines(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    trace.write_text(
        json.dumps({"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [0]})
        + "\n\n  \n"
        + json.dumps({"timestamp": 5, "input_length": 2, "output_length": 2, "hash_ids": [1]})
        + "\n",
        encoding="utf-8",
    )

    summary, rows = MooncakeTraceParser().parse(trace)

    assert summary.request_count == 2
    assert [row.line_index for row in rows] == [0, 3]


def test_mooncake_parse_truncates_at_max_rows(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [0]},
            {"timestamp": 1, "input_length": 1, "output_length": 1, "hash_ids": [1]},
            {"timestamp": 2, "input_length": 1, "output_length": 1, "hash_ids": [2]},
        ],
    )

    summary, _rows = MooncakeTraceParser(max_rows=2).parse(trace)

    assert summary.request_count == 2
    assert summary.limitations == ("Trace truncated at 2 requests.",)


def test_mooncake_parse_at_exact_row_limit_is_not_truncated(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [0]},
            {"timestamp": 1, "input_length": 1, "output_length": 1, "hash_ids": [1]},
        ],
    )

    summary, rows = MooncakeTraceParser(max_rows=2).parse(trace)

    assert len(rows) == 2
    assert summary.limitations == ()


def test_mooncake_parse_reports_nonzero_first_timestamp(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {"timestamp": 500, "input_length": 1, "output_length": 1, "hash_ids": [0]},
            {"timestamp": 1000, "input_length": 1, "output_length": 1, "hash_ids": [1]},
        ],
    )

    summary, _rows = MooncakeTraceParser().parse(trace)

    assert summary.limitations == ("The first request timestamp is not zero milliseconds.",)


def test_mooncake_parse_reports_timestamp_regression(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {"timestamp": 100, "input_length": 1, "output_length": 1, "hash_ids": [0]},
            {"timestamp": 50, "input_length": 1, "output_length": 1, "hash_ids": [1]},
        ],
    )

    summary, _rows = MooncakeTraceParser().parse(trace)

    assert any("regressed" in note for note in summary.limitations)


@pytest.mark.parametrize(
    ("line", "hint"),
    [
        ({"timestamp": 0, "input_length": 1}, "output_length"),
        ({"timestamp": -1, "input_length": 1, "output_length": 1, "hash_ids": [0]}, "timestamp"),
        (
            {"timestamp": 0, "input_length": 1.5, "output_length": 1, "hash_ids": [0]},
            "input_length",
        ),
        (
            {"timestamp": 0, "input_length": 1, "output_length": True, "hash_ids": [0]},
            "output_length",
        ),
        ({"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": "bad"}, "hash_ids"),
        ({"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [-1]}, "hash_ids"),
        ({"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [True]}, "hash_ids"),
    ],
)
def test_mooncake_rejects_malformed_lines(
    tmp_path: Path, line: dict[str, object], hint: str
) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    trace.write_text(json.dumps(line) + "\n", encoding="utf-8")

    with pytest.raises(DomainError) as error:
        MooncakeTraceParser().parse(trace)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_mooncake_rejects_non_json_line(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    trace.write_text("{not valid json}\n", encoding="utf-8")

    with pytest.raises(DomainError) as error:
        MooncakeTraceParser().parse(trace)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_mooncake_rejects_empty_file(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    trace.write_text("\n\n", encoding="utf-8")

    with pytest.raises(DomainError) as error:
        MooncakeTraceParser().parse(trace)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_mooncake_rejects_oversized_line(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    valid = {"timestamp": 0, "input_length": 1, "output_length": 1, "hash_ids": [0]}
    valid["hash_ids"] = list(range(20_000))
    line = json.dumps(valid)
    trace.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(DomainError) as error:
        MooncakeTraceParser(max_rows=10).parse(trace)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_mooncake_rejects_unterminated_oversized_line(tmp_path: Path) -> None:
    trace = tmp_path / "mooncake_trace.jsonl"
    trace.write_bytes(b"{" + (b"x" * (MooncakeTraceParser.max_line_bytes + 1)))

    with pytest.raises(DomainError) as error:
        MooncakeTraceParser().parse(trace)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


# ---------------------------------------------------------------------------
# AIPerf 0.12 per-request result normalization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# AIPerf inputs.json correlation index
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# AIPerf record correlation against inputs.json
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# vLLM aggregate result JSON normalization
# ---------------------------------------------------------------------------


def test_vllm_parse_normalizes_aggregate_metrics(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(json.dumps(_vllm_result()), encoding="utf-8")

    document, rows = VllmResultParser().parse(result)

    assert document.successful_requests == 313
    assert document.failed_requests == 0
    names = {row.name for row in rows}
    assert "vllm.request_throughput" in names
    assert "vllm.request_goodput" in names
    assert "vllm.output_throughput" in names
    assert "vllm.total_token_throughput" in names
    assert "vllm.total_input_tokens" in names
    assert "vllm.total_output_tokens" in names
    assert "vllm.completed_requests" in names
    assert "vllm.failed_requests" in names
    assert "vllm.total_requests" in names
    assert "vllm.duration_seconds" in names
    assert "vllm.time_to_first_token.mean_ms" in names
    assert "vllm.time_to_first_token.median_ms" in names
    assert "vllm.time_to_first_token.std_ms" in names
    assert "vllm.time_to_first_token.p50_ms" in names
    assert "vllm.time_to_first_token.p99_ms" in names
    assert "vllm.time_per_output_token.mean_ms" in names
    assert "vllm.inter_token_latency.mean_ms" in names
    assert "vllm.end_to_end_latency.mean_ms" in names
    assert all(row.evidence_level == "derived" for row in rows)
    assert all(
        row.unit in {"requests/sec", "tokens/sec", "count", "requests", "ms", "s"} for row in rows
    )
    throughput = next(row for row in rows if row.name == "vllm.request_throughput")
    assert throughput.value_float == pytest.approx(5.18)
    assert throughput.aggregation == "aggregate"
    assert throughput.dimensions["producer"] == "vllm.benchmark_serving"
    assert throughput.dimensions["time_scale"] == "1.0"
    by_name = {row.name: row.value_float for row in rows}
    assert by_name["vllm.request_goodput"] == pytest.approx(5.18)
    assert by_name["vllm.completed_requests"] == 313
    assert by_name["vllm.failed_requests"] == 0
    assert by_name["vllm.total_requests"] == 313
    assert by_name["vllm.duration_seconds"] == pytest.approx(60.48)


def test_vllm_parse_omits_request_goodput_when_provider_does_not_emit_it() -> None:
    payload = _vllm_result()
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    del metrics["request_goodput"]

    document, rows = VllmResultParser().parse_payload(payload)

    assert document.metrics.request_goodput is None
    assert all(row.name != "vllm.request_goodput" for row in rows)


def test_vllm_parse_drops_raw_payload_fields(tmp_path: Path) -> None:
    payload = _vllm_result()
    payload["raw_prompts"] = ["secret prompt text"]
    payload["error_log"] = "connection refused to http://10.0.0.1:8000"
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(json.dumps(payload), encoding="utf-8")

    document, rows = VllmResultParser().parse(result)

    serialized = json.dumps([row.model_dump(mode="json") for row in rows])
    assert "secret prompt text" not in serialized
    assert "10.0.0.1" not in serialized
    assert document.successful_requests == 313


def test_vllm_parse_rejects_result_above_document_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_vllm_result()))
    monkeypatch.setattr(VllmResultParser, "max_document_bytes", 8)

    with pytest.raises(DomainError) as caught:
        VllmResultParser().parse(result)

    assert caught.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_rejects_missing_metrics(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(json.dumps({"successful_requests": 1}), encoding="utf-8")

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_rejects_mismatched_totals(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(
        json.dumps(_vllm_result(successful_requests=200, failed_requests=0, total_requests=313)),
        encoding="utf-8",
    )

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_rejects_completed_mismatch(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(
        json.dumps(
            _vllm_result(
                metrics={"completed": 100},
                successful_requests=313,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_rejects_non_json(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text("{not json", encoding="utf-8")

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_rejects_negative_latency(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(
        json.dumps(_vllm_result(metrics={"mean_ttft_ms": -1.0})),
        encoding="utf-8",
    )

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


@pytest.mark.parametrize(
    "percentile_pair",
    [(-1.0, 1.0), (101.0, 1.0), (50.0, -1.0), (float("nan"), 1.0), (50.0, float("inf"))],
)
def test_vllm_parse_rejects_invalid_percentile_pairs(
    tmp_path: Path, percentile_pair: tuple[float, float]
) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(
        json.dumps(
            _vllm_result(metrics={"percentiles_ttft_ms": [percentile_pair]}),
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


@pytest.mark.parametrize("payload", [[], "not an object", 1])
def test_vllm_parse_wraps_non_object_json_as_domain_error(tmp_path: Path, payload: object) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    result.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_rejects_infinite_latency(tmp_path: Path) -> None:
    result = tmp_path / "mooncake_replay_results.json"
    payload = _vllm_result()
    payload["metrics"]["mean_ttft_ms"] = float("inf")  # type: ignore[index]
    result.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    with pytest.raises(DomainError) as error:
        VllmResultParser().parse(result)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_vllm_parse_payload_accepts_dict_directly() -> None:
    document, rows = VllmResultParser().parse_payload(_vllm_result())

    assert document.successful_requests == 313
    assert len(rows) > 0


def test_vllm_parse_accepts_native_save_result_shape() -> None:
    metrics = _vllm_metrics()
    metrics["num_prompts"] = 320
    metrics["duration"] = 60.48
    metrics["total_input_tokens"] = metrics.pop("total_input")
    metrics["total_output_tokens"] = metrics.pop("total_output")

    document, rows = VllmResultParser().parse_payload(metrics)

    assert document.successful_requests == 313
    assert document.failed_requests == 7
    assert document.total_requests == 320
    assert any(row.name == "vllm.request_throughput" for row in rows)


def test_vllm_parse_rejects_boolean_native_request_counts() -> None:
    with pytest.raises(DomainError) as error:
        VllmResultParser().parse_payload({"completed": True, "num_prompts": True})

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_mooncake_extractor_publishes_bounded_request_evidence(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    trace = tmp_path / "trace.jsonl"
    _write_mooncake_trace(
        trace,
        [{"timestamp": 0, "input_length": 10, "output_length": 2, "hash_ids": [1]}],
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.INFERENCE_REQUEST_TRACE,
            sensitivity=Sensitivity.SENSITIVE,
        )
    )

    result = InferenceArtifactExtractor(workspace).extract_trace(imported.run.run_id)
    page = EvidenceQueryService(workspace).inference_requests(run_id=imported.run.run_id, limit=1)

    assert result.request_count == 1
    assert result.corpus_commit_id.startswith("sha256:")
    assert page.requests[0].input_tokens == 10
    assert page.requests[0].cache_hit is None


def test_mooncake_extractor_enforces_independent_request_row_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace.initialize(tmp_path)
    trace = tmp_path / "trace.jsonl"
    _write_mooncake_trace(
        trace,
        [
            {"timestamp": 0, "input_length": 10, "output_length": 2},
            {"timestamp": 1, "input_length": 20, "output_length": 4},
        ],
    )
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.INFERENCE_REQUEST_TRACE,
            sensitivity=Sensitivity.SENSITIVE,
        )
    )
    monkeypatch.setattr(InferenceArtifactExtractor, "max_request_rows", 1)

    result = InferenceArtifactExtractor(workspace).extract_trace(imported.run.run_id)

    assert workspace.config.storage.max_rows_per_generation > 1
    assert result.request_count == 1
    assert result.limitations == ("Trace truncated at 1 requests.",)


def test_vllm_extractor_publishes_aggregate_measurements(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_vllm_result()))
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=result_path, kind=ArtifactKind.INFERENCE_RESULT)
    )

    result = InferenceArtifactExtractor(workspace).extract_vllm_result(imported.run.run_id)

    assert result.measurement_count > 0
    assert result.corpus_commit_id.startswith("sha256:")


def test_extraction_idempotency_is_scoped_to_target_evidence_run(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_vllm_result()))
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=result_path, kind=ArtifactKind.INFERENCE_RESULT)
    )
    extractor = InferenceArtifactExtractor(workspace)
    first_target = "canonical-run-1"
    second_target = "canonical-run-2"

    first = extractor.extract_vllm_result(imported.run.run_id, evidence_run_id=first_target)
    second = extractor.extract_vllm_result(imported.run.run_id, evidence_run_id=second_target)

    assert first.corpus_commit_id != second.corpus_commit_id
    first_rows = EvidenceQueryService(workspace).measurements(run_id=first_target, limit=100)
    second_rows = EvidenceQueryService(workspace).measurements(run_id=second_target, limit=100)
    assert first_rows.measurements
    assert second_rows.measurements
    assert {row.run_id for row in first_rows.measurements} == {first_target}
    assert {row.run_id for row in second_rows.measurements} == {second_target}


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


def test_sglang_result_parser_rejects_sensitive_detailed_output(tmp_path: Path) -> None:
    result_path = tmp_path / "result.jsonl"
    result_path.write_text(
        json.dumps(
            {
                "duration": 1.0,
                "completed": 1,
                "total_input_tokens": 4,
                "total_output_tokens": 2,
                "request_throughput": 1.0,
                "generated_texts": ["secret"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DomainError, match="aggregate SGLang JSONL"):
        SglangResultParser().parse(result_path)


def test_sglang_result_parser_emits_safe_optional_scalar_metrics(tmp_path: Path) -> None:
    result_path = tmp_path / "result.jsonl"
    result_path.write_text(
        json.dumps(
            {
                "duration": 1.0,
                "completed": 2,
                "total_input_tokens": 8,
                "total_output_tokens": 4,
                "request_throughput": 2.0,
                "total_token_throughput": 12.0,
                "accept_length": 3.5,
                "mean_ttft_ms": 4.0,
                "p95_ttft_ms": 9.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _document, rows = SglangResultParser().parse(result_path)

    values = {row.name: row.value_float for row in rows}
    assert values["sglang.accept_length"] == 3.5
    assert values["sglang.request_throughput"] == 2.0
    assert values["sglang.total_token_throughput"] == 12.0
    assert all(row.dimensions["producer"] == "sglang.bench_serving" for row in rows)
    by_name = {row.name: row for row in rows}
    assert by_name["sglang.mean_ttft_ms"].aggregation == "mean"
    assert by_name["sglang.p95_ttft_ms"].aggregation == "percentile"
    assert by_name["sglang.p95_ttft_ms"].dimensions["percentile"] == "95"


@pytest.mark.parametrize("payload", ["", "{}\n{}\n", '{"duration": NaN}\n'])
def test_sglang_result_parser_rejects_malformed_or_multiple_records(
    tmp_path: Path, payload: str
) -> None:
    result_path = tmp_path / "result.jsonl"
    result_path.write_text(payload, encoding="utf-8")

    with pytest.raises(DomainError, match="aggregate SGLang JSONL"):
        SglangResultParser().parse(result_path)
