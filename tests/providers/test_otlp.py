from __future__ import annotations

from pathlib import Path

import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    PathSource,
    RequestLimits,
)


def _write_otlp_trace(path: Path, *, include_event: bool = False) -> None:
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
        span.start_time_unix_nano = start
        span.end_time_unix_nano = start + 10
        if include_event and index == 2:
            event = span.events.add()
            event.name = "checkpoint"
            event.time_unix_nano = start + 5
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
def test_otlp_window_filters_inside_isolated_parser(tmp_path: Path) -> None:
    trace = tmp_path / "trace.otlp"
    _write_otlp_trace(trace)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "trace.window",
            [PathSource(path=str(trace), format="otlp")],
            {"start_ns": 150, "end_ns": 250},
        )
    finally:
        runtime.close()

    span_rows = [row for row in result["blocks"][1]["rows"] if row["table"] == "spans"]
    assert [row["name"] for row in span_rows] == ["span-2"]


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
