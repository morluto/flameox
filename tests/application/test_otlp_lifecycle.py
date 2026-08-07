from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("google.protobuf")
pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
from google.protobuf import json_format  # type: ignore[import-untyped]
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import Span

from flameox.application import ImportArtifactRequest, ImportService
from flameox.application.lifecycle import LifecycleEvidenceService
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import Workspace


def _attribute(key: str, value: AnyValue) -> KeyValue:
    return KeyValue(key=key, value=value)


def _request() -> ExportTraceServiceRequest:
    request = ExportTraceServiceRequest()
    resource_spans = request.resource_spans.add()
    resource_spans.schema_url = "https://example.test/resource"
    resource_spans.resource.CopyFrom(
        Resource(
            attributes=[
                _attribute("service.name", AnyValue(string_value="fixture")),
                _attribute("service.version", AnyValue(string_value="1")),
            ],
            dropped_attributes_count=2,
        )
    )
    scope_spans = resource_spans.scope_spans.add()
    scope_spans.schema_url = "https://example.test/scope"
    scope_spans.scope.name = "fixture.scope"
    scope_spans.scope.version = "0.1"
    scope_spans.scope.attributes.append(_attribute("scope.flag", AnyValue(bool_value=True)))

    def add_span(
        span_id: bytes,
        name: str,
        start: int,
        end: int,
        parent: bytes = b"",
    ) -> Span:
        span = scope_spans.spans.add(
            trace_id=b"t" * 16,
            span_id=span_id,
            parent_span_id=parent,
            name=name,
            kind=Span.SPAN_KIND_SERVER,
            start_time_unix_nano=start,
            end_time_unix_nano=end,
        )
        span.attributes.extend(
            [
                _attribute("string", AnyValue(string_value="value")),
                _attribute("integer", AnyValue(int_value=7)),
                _attribute("double", AnyValue(double_value=1.5)),
                _attribute("bytes", AnyValue(bytes_value=b"raw")),
            ]
        )
        array = span.attributes.add(key="array")
        array.value.array_value.values.extend(
            [AnyValue(string_value="nested"), AnyValue(bool_value=False)]
        )
        nested = span.attributes.add(key="nested")
        nested.value.kvlist_value.values.append(_attribute("member", AnyValue(int_value=9)))
        return cast(Span, span)

    root = add_span(b"r" * 8, "root", 100, 200)
    root.events.add(
        name="message",
        time_unix_nano=150,
        attributes=[_attribute("event.value", AnyValue(string_value="seen"))],
        dropped_attributes_count=1,
    )
    root.links.add(
        trace_id=b"l" * 16,
        span_id=b"l" * 8,
        trace_state="vendor=value",
        flags=3,
        attributes=[_attribute("link.value", AnyValue(bool_value=True))],
        dropped_attributes_count=1,
    )
    add_span(b"c" * 8, "child", 150, 300, parent=b"r" * 8)
    add_span(b"1" * 8, "repeat", 301, 400)
    add_span(b"2" * 8, "repeat", 401, 500)
    add_span(b"o" * 8, "orphan", 501, 600, parent=b"x" * 8)
    add_span(b"z" * 8, "missing", 0, 0)
    add_span(b"v" * 8, "reversed", 700, 600)
    add_span(b"r" * 8, "duplicate", 800, 900)
    invalid_trace = add_span(b"short", "invalid", 901, 902)
    invalid_trace.trace_id = b"short"
    return request


def _import(
    workspace: Workspace,
    path: Path,
    payload: bytes,
    media_type: str,
) -> tuple[str, str]:
    path.write_bytes(payload)
    result = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=path,
            kind=ArtifactKind.OTLP_TRACE,
            media_type=media_type,
        )
    )
    return result.run.run_id, result.artifact_id


@pytest.mark.parametrize("media_type", ["application/x-protobuf", "application/json"])
def test_otlp_normalization_and_lifecycle_queries_preserve_evidence(
    tmp_path: Path,
    media_type: str,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    request = _request()
    payload = (
        request.SerializeToString(deterministic=True)
        if media_type != "application/json"
        else json_format.MessageToJson(request).encode()
    )
    run_id, artifact_id = _import(workspace, tmp_path / "trace.otlp", payload, media_type)

    extracted = ImportService(workspace).extract_otlp_trace(run_id, artifact_id)
    repeated = ImportService(workspace).extract_otlp_trace(run_id, artifact_id)

    assert extracted.evidence_generation_id == repeated.evidence_generation_id
    assert (extracted.resource_count, extracted.scope_count) == (1, 1)
    assert extracted.span_count == 9
    assert extracted.event_count == 1
    assert extracted.link_count == 1
    assert any("duplicate_identity" in item for item in extracted.limitations)
    assert any("missing_timestamp" in item for item in extracted.limitations)
    assert any("end_before_start" in item for item in extracted.limitations)
    assert any("trace_id:invalid_length" in item for item in extracted.limitations)

    with Catalog(workspace).open_snapshot() as snapshot:
        attributes = snapshot.execute(
            "SELECT attributes_json FROM otel_spans WHERE artifact_id = ? AND span_id = ? LIMIT 1",
            (artifact_id, (b"r" * 8).hex()),
        ).fetchone()
    assert attributes is not None
    decoded = json.loads(attributes[0])
    assert decoded["bytes"] == {"base64": "cmF3", "type": "bytes"}
    assert decoded["nested"]["member"] == 9

    lifecycle = LifecycleEvidenceService(workspace)
    window = lifecycle.get_operation_window(
        artifact_id=artifact_id,
        start_ns=150,
        end_ns=350,
        trace_id=(b"t" * 16).hex(),
        limit=1,
    )
    assert window.evidence_level == "derived"
    assert window.query_bounds["start_ns"] == 150
    assert window.returned == 1
    assert window.next_cursor is not None
    next_window = lifecycle.get_operation_window(
        artifact_id=artifact_id,
        start_ns=150,
        end_ns=350,
        trace_id=(b"t" * 16).hex(),
        limit=1,
        cursor=window.next_cursor,
    )
    assert next_window.items[0].span_id != window.items[0].span_id
    second_request = _request()
    second_request.resource_spans[0].schema_url = "https://example.test/second"
    second_payload = (
        second_request.SerializeToString(deterministic=True)
        if media_type != "application/json"
        else json_format.MessageToJson(second_request).encode()
    )
    second_run, second_artifact = _import(
        workspace,
        tmp_path / "trace-second.otlp",
        second_payload,
        media_type,
    )
    ImportService(workspace).extract_otlp_trace(second_run, second_artifact)
    with pytest.raises(DomainError) as stale:
        lifecycle.get_operation_window(
            artifact_id=artifact_id,
            start_ns=150,
            end_ns=350,
            trace_id=(b"t" * 16).hex(),
            limit=1,
            cursor=window.next_cursor,
        )
    assert stale.value.code is ErrorCode.STALE_CURSOR

    transitions = lifecycle.get_operation_transitions(
        artifact_id=artifact_id,
        trace_id=(b"t" * 16).hex(),
        max_depth=1,
    )
    assert {item.name for item in transitions.items} >= {"root", "child"}
    assert "missing_parent_references_are_coverage_gaps" in transitions.limitations

    repeated_sequences = lifecycle.find_repeated_operation_sequences(artifact_id=artifact_id)
    assert [item.name for item in repeated_sequences.items] == ["repeat", "repeat"]
    assert all(item.details["repetition_count"] == 2 for item in repeated_sequences.items)

    gaps = lifecycle.get_lifecycle_gaps(artifact_id=artifact_id)
    assert {item.reason for item in gaps.items} >= {
        "missing_timestamp",
        "end_before_start",
        "missing_parent",
        "duplicate_identity",
    }
    assert all(item.kind == "gap" for item in gaps.items)


def test_otlp_parser_rejects_ambiguous_or_unknown_formats(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    request = _request()
    _run_id, artifact_id = _import(
        workspace,
        tmp_path / "trace.json",
        json_format.MessageToJson(request).encode(),
        "application/json",
    )
    with pytest.raises(DomainError) as unknown:
        path = tmp_path / "unknown.json"
        run_id2, _ = _import(
            workspace,
            path,
            b'{"resourceSpans": [], "unknownField": true}',
            "application/json",
        )
        ImportService(workspace).extract_otlp_trace(run_id2)
    assert unknown.value.code is ErrorCode.ARTIFACT_PARSE_FAILED

    with pytest.raises(DomainError) as media:
        path = tmp_path / "ambiguous"
        run_id3, _ = _import(
            workspace,
            path,
            request.SerializeToString(),
            "application/octet-stream",
        )
        ImportService(workspace).extract_otlp_trace(run_id3)
    assert media.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert artifact_id


def test_otlp_row_limit_refuses_partial_generation_and_retains_raw_artifact(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    request = _request()
    run_id, artifact_id = _import(
        workspace,
        tmp_path / "limited.otlp",
        request.SerializeToString(deterministic=True),
        "application/x-protobuf",
    )
    config = workspace.config.model_copy(
        update={
            "storage": workspace.config.storage.model_copy(update={"max_rows_per_generation": 1})
        }
    )
    workspace.paths.config.write_text(config.to_toml())

    result = ImportService(workspace).extract_otlp_trace(run_id, artifact_id)

    assert result.failed is True
    assert result.evidence_generation_id is None
    assert "otlp_row_limit_exceeded" in result.limitations
    assert ImportService(workspace).artifacts.get(artifact_id).payload_path.is_file()
