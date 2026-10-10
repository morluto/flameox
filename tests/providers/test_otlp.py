from __future__ import annotations

from pathlib import Path

import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    EvidenceSource,
    PathSource,
    RequestLimits,
)


def _write_otlp_trace(
    path: Path,
    *,
    include_event: bool = False,
    timestamp_offset: int = 0,
    json_encoding: bool = False,
) -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    request = ExportTraceServiceRequest()
    resource_spans = request.resource_spans.add()
    scope_spans = resource_spans.scope_spans.add()
    scope_spans.scope.name = "test-scope"
    for index, start in enumerate((100, 200, 300), 1):
        span = scope_spans.spans.add()
        span.trace_id = bytes.fromhex("01" * 16)
        span.span_id = index.to_bytes(8, "big")
        span.name = f"span-{index}"
        span.start_time_unix_nano = timestamp_offset + start
        span.end_time_unix_nano = timestamp_offset + start + 10
        if include_event and index == 2:
            event = span.events.add()
            event.name = "checkpoint"
            event.time_unix_nano = timestamp_offset + start + 5
    if json_encoding:
        from google.protobuf.json_format import MessageToJson  # type: ignore[import-untyped]

        path.write_text(" " * 5_000 + MessageToJson(request))
    else:
        path.write_bytes(request.SerializeToString())


@pytest.mark.process
def test_otlp_partial_rows_continue_without_repository_state(tmp_path: Path) -> None:
    trace = tmp_path / "trace.otlp"
    _write_otlp_trace(trace)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        first = runtime.analyze(
            "trace.summary",
            [PathSource(path=str(trace), format="otlp")],
            {},
            limits=RequestLimits(max_rows=3),
        )
        second = runtime.analyze(
            "trace.summary",
            [PathSource(path=str(trace), format="otlp")],
            {},
            limits=RequestLimits(max_rows=3),
            continuation=first["continuation"],
        )
    finally:
        runtime.close()

    assert first["provider"]["id"] == "otlp"
    assert first["coverage"] == {"rows_returned": 3, "rows_observed": 5, "complete": False}
    assert [row["name"] for row in second["blocks"][1]["rows"]] == ["span-2", "span-3"]
    assert second["coverage"]["complete"] is True
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.process
@pytest.mark.parametrize("json_encoding", [False, True])
@pytest.mark.parametrize(
    ("timestamp_offset", "string_bounds"),
    [(0, False), (1_791_720_000_000_000_000, False), (1_791_720_000_000_000_000, True)],
)
def test_otlp_window_filters_inside_isolated_parser(
    tmp_path: Path, timestamp_offset: int, string_bounds: bool, json_encoding: bool
) -> None:
    trace = tmp_path / "trace.otlp"
    _write_otlp_trace(trace, timestamp_offset=timestamp_offset, json_encoding=json_encoding)
    bounds: dict[str, int | str] = {
        "start_ns": timestamp_offset + 150,
        "end_ns": timestamp_offset + 250,
    }
    if string_bounds:
        bounds = {key: str(value) for key, value in bounds.items()}
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "trace.window",
            [PathSource(path=str(trace), format="otlp")],
            bounds,
        )
        preserved = runtime.preserve_evidence(result["analysis_id"])
    finally:
        runtime.close()

    span_rows = [row for row in result["blocks"][1]["rows"] if row["table"] == "spans"]
    assert [row["name"] for row in span_rows] == ["span-2"]
    if timestamp_offset:
        assert span_rows[0]["start_time_unix_nano"] == str(timestamp_offset + 200)
    reopened = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        manifest = reopened.read_evidence(preserved["evidence_id"])
        saved_bounds = manifest["body"]["analysis_request"]["arguments"]
        assert saved_bounds == (
            {key: str(value) for key, value in bounds.items()} if timestamp_offset else bounds
        )
        projection = reopened.read_evidence_agent_projection(preserved["evidence_id"])
        replayed = reopened.analyze(
            "trace.window",
            [EvidenceSource.model_validate(item) for item in projection["analysis_sources"]],
            saved_bounds,
        )
        assert replayed["blocks"] == result["blocks"]
    finally:
        reopened.close()


@pytest.mark.process
def test_otlp_window_filters_before_applying_the_normalization_row_limit(tmp_path: Path) -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    request = ExportTraceServiceRequest()
    for _index in range(1_001):
        request.resource_spans.add().scope_spans.add()
    matching_scope = request.resource_spans.add().scope_spans.add()
    matching = matching_scope.spans.add()
    matching.trace_id = bytes.fromhex("01" * 16)
    matching.span_id = bytes.fromhex("02" * 8)
    matching.name = "matching-span"
    matching.start_time_unix_nano = 200
    matching.end_time_unix_nano = 210
    trace = tmp_path / "trace.otlp"
    trace.write_bytes(request.SerializeToString())

    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "trace.window",
            [PathSource(path=str(trace), format="otlp")],
            {"start_ns": 150, "end_ns": 250},
        )
    finally:
        runtime.close()

    assert result["coverage"]["complete"] is True
    assert [row["name"] for row in result["blocks"][1]["rows"] if row["table"] == "spans"] == [
        "matching-span"
    ]


@pytest.mark.process
def test_otlp_operation_and_lifecycle_projections_are_semantically_distinct(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.otlp"
    _write_otlp_trace(trace, include_event=True)
    source = [PathSource(path=str(trace), format="otlp")]
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        summary = runtime.analyze("trace.summary", source, {})
        operations = runtime.analyze("trace.operations", source, {})
        lifecycle = runtime.analyze("trace.lifecycle", source, {})
    finally:
        runtime.close()

    assert {row["table"] for row in summary["blocks"][1]["rows"]} >= {"spans", "events"}
    assert [row["operation"] for row in operations["blocks"][1]["rows"]] == [
        "span-1",
        "span-2",
        "span-3",
    ]
    assert all("total_duration_ns" in row for row in operations["blocks"][1]["rows"])
    transitions = lifecycle["blocks"][1]["rows"]
    assert {row["transition"] for row in transitions} == {
        "span_started",
        "span_event",
        "span_ended",
    }
    assert (
        next(row for row in transitions if row["transition"] == "span_event")["operation"]
        == "checkpoint"
    )


@pytest.mark.process
def test_otlp_protobuf_that_resembles_json_keeps_its_native_encoding(tmp_path: Path) -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    request = ExportTraceServiceRequest()
    request.resource_spans.add().scope_spans.add().scope.name = "x" * 117
    payload = request.SerializeToString()
    assert payload.startswith(b"\n{")
    artifact = tmp_path / "misleading.json"
    artifact.write_bytes(payload)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            "trace.summary", [PathSource(path=str(artifact), format="otlp")], {}
        )
        assert result["blocks"][0]["values"]["resources_count"] == 1
        assert result["blocks"][0]["values"]["scopes_count"] == 1
    finally:
        runtime.close()


@pytest.mark.process
@pytest.mark.parametrize("json_encoding", [False, True])
@pytest.mark.parametrize("label", ["NaN", "Infinity", "-Infinity"])
def test_otlp_nonfinite_attributes_preserve_native_types(
    tmp_path: Path, json_encoding: bool, label: str
) -> None:
    import json

    from google.protobuf.json_format import MessageToJson
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    request = ExportTraceServiceRequest()
    span = request.resource_spans.add().scope_spans.add().spans.add()
    span.trace_id = bytes.fromhex("01" * 16)
    span.span_id = bytes.fromhex("02" * 8)
    span.name = "nonfinite-attribute"
    span.start_time_unix_nano = 1
    span.end_time_unix_nano = 5
    span.attributes.add(key="double").value.double_value = float(label)
    span.attributes.add(key="string").value.string_value = label
    span.attributes.add(key="integer").value.int_value = 2**63 - 1
    span.attributes.add(key="array").value.array_value.values.add().double_value = float(label)
    artifact = tmp_path / "trace.otlp"
    artifact.write_bytes(
        MessageToJson(request).encode("utf-8") if json_encoding else request.SerializeToString()
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        sources = [PathSource(path=str(artifact), format="otlp")]
        summary = runtime.analyze("trace.summary", sources, {})
        attributes = json.loads(
            next(row for row in summary["blocks"][1]["rows"] if row["table"] == "spans")[
                "attributes_json"
            ]
        )
        tagged = {"type": "double", "value": label}
        assert attributes == {
            "double": tagged,
            "string": label,
            "integer": 2**63 - 1,
            "array": [tagged],
        }
        operations = runtime.analyze("trace.operations", sources, {})
        assert operations["blocks"][1]["rows"][0]["operation"] == span.name
        preserved = runtime.preserve_evidence(summary["analysis_id"])
        replay = runtime.analyze(
            "trace.summary",
            [EvidenceSource(kind="evidence", evidence_id=preserved["evidence_id"])],
            {},
        )
        assert replay["blocks"] == summary["blocks"]
    finally:
        runtime.close()
