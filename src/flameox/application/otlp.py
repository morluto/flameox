from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, computed_field, model_validator

from flameox.application.artifact_workers import ArtifactWorker
from flameox.application.evidence_rows import _json
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace


class OtlpExtractionResult(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    schema_version: int = 1
    run_id: str
    artifact_id: str
    evidence_generation_id: str | None = None
    resource_count: int = 0
    scope_count: int = 0
    span_count: int = 0
    event_count: int = 0
    link_count: int = 0
    limitations: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def parse_legacy_failed_projection(cls, value: object) -> object:
        if not isinstance(value, dict) or "failed" not in value:
            return value
        parsed = dict(value)
        failed = parsed.pop("failed")
        if failed != (parsed.get("evidence_generation_id") is None):
            raise ValueError("failure status must agree with evidence publication")
        return parsed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed(self) -> bool:
        return self.evidence_generation_id is None


@dataclass(frozen=True, slots=True)
class _ParsedOtlp:
    resources: list[dict[str, object]]
    scopes: list[dict[str, object]]
    spans: list[dict[str, object]]
    events: list[dict[str, object]]
    links: list[dict[str, object]]
    limitations: tuple[str, ...]


class _OtlpRowLimitExceeded(Exception):
    def __init__(self, counts: dict[str, int], limitations: tuple[str, ...]) -> None:
        super().__init__("OTLP normalization row limit exceeded")
        self.counts = counts
        self.limitations = limitations


class OtlpTraceService:
    """Normalize file-imported OTLP traces into bounded authoritative tables."""

    extractor_version = "1.0"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.artifacts = ArtifactStore(workspace)
        self.runs = RunStore(workspace)
        self.publisher = GenerationPublisher(workspace)
        self.worker = ArtifactWorker(workspace)

    def extract_otlp_trace(
        self,
        run_id: str,
        artifact_id: str | None = None,
    ) -> OtlpExtractionResult:
        run = self.runs.read(run_id)
        registration = next(
            (
                item
                for item in run.artifacts
                if item.kind is ArtifactKind.OTLP_TRACE
                and (artifact_id is None or item.artifact_id == artifact_id)
            ),
            None,
        )
        if registration is None:
            raise DomainError(
                ErrorCode.RUN_NOT_FOUND,
                "The run has no registered OTLP trace artifact matching the request.",
                details={"run_id": run_id, "artifact_id": artifact_id},
            )
        path = self.artifacts.get(registration.artifact_id).payload_path
        try:
            response = self.worker.run_sync(
                "flameox.workers.otlp",
                {
                    "artifact_path": str(path),
                    "media_type": registration.media_type,
                    "row_limit": self.workspace.config.storage.max_rows_per_generation,
                },
                name="OTLP",
            )
            if response.get("row_limit_exceeded") is True:
                raw_counts = response.get("counts")
                raw_limitations = response.get("limitations")
                if not isinstance(raw_counts, dict) or not isinstance(raw_limitations, list):
                    raise ValueError("OTLP row-limit response is invalid")
                return OtlpExtractionResult(
                    run_id=run_id,
                    artifact_id=registration.artifact_id,
                    resource_count=int(raw_counts["resources"]),
                    scope_count=int(raw_counts["scopes"]),
                    span_count=int(raw_counts["spans"]),
                    event_count=int(raw_counts["events"]),
                    link_count=int(raw_counts["links"]),
                    limitations=tuple(str(item) for item in raw_limitations),
                )
            parsed = _parsed_response(response)
        except _OtlpRowLimitExceeded as error:
            return OtlpExtractionResult(
                run_id=run_id,
                artifact_id=registration.artifact_id,
                resource_count=error.counts["resources"],
                scope_count=error.counts["scopes"],
                span_count=error.counts["spans"],
                event_count=error.counts["events"],
                link_count=error.counts["links"],
                limitations=error.limitations,
            )
        except DomainError:
            raise
        except Exception as error:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "The OTLP trace payload could not be parsed.",
                run_id=run_id,
            ) from error

        rows = {
            table: table_rows
            for table, table_rows in (
                ("otel_resources", parsed.resources),
                ("otel_scopes", parsed.scopes),
                ("otel_spans", parsed.spans),
                ("otel_span_events", parsed.events),
                ("otel_span_links", parsed.links),
            )
        }
        for table_rows in rows.values():
            for row in table_rows:
                row["run_id"] = run_id
                row["artifact_id"] = registration.artifact_id
        published = self.publisher.publish_rows_idempotent(
            rows,
            publisher="flameox.otlp",
            publisher_version=self.extractor_version,
            input_run_ids=(run_id,),
            input_artifact_ids=(registration.artifact_id,),
            operation_identity={"media_type": registration.media_type},
        )
        return OtlpExtractionResult(
            run_id=run_id,
            artifact_id=registration.artifact_id,
            evidence_generation_id=published.manifest.generation_id,
            resource_count=len(parsed.resources),
            scope_count=len(parsed.scopes),
            span_count=len(parsed.spans),
            event_count=len(parsed.events),
            link_count=len(parsed.links),
            limitations=parsed.limitations,
        )

    @staticmethod
    def _normalize(request: Any, *, row_limit: int) -> _ParsedOtlp:
        resources: list[dict[str, object]] = []
        scopes: list[dict[str, object]] = []
        spans: list[dict[str, object]] = []
        events: list[dict[str, object]] = []
        links: list[dict[str, object]] = []
        limitations: list[str] = []
        identities: set[tuple[str, str]] = set()
        counts = {"resources": 0, "scopes": 0, "spans": 0, "events": 0, "links": 0}

        def append_row(table: list[dict[str, object]], name: str, row: dict[str, object]) -> None:
            counts[name] += 1
            if sum(counts.values()) > row_limit:
                raise _OtlpRowLimitExceeded(
                    counts.copy(), (*sorted(set(limitations)), "otlp_row_limit_exceeded")
                )
            table.append(row)

        source_ordinal = 0
        for resource_ordinal, resource_spans in enumerate(request.resource_spans):
            resource = resource_spans.resource
            append_row(
                resources,
                "resources",
                {
                    "resource_ordinal": resource_ordinal,
                    "schema_url": resource_spans.schema_url or None,
                    "attributes_json": _json(_attributes(resource.attributes)),
                    "dropped_attributes_count": resource.dropped_attributes_count,
                },
            )
            for scope_ordinal, scope_spans in enumerate(resource_spans.scope_spans):
                scope = scope_spans.scope
                append_row(
                    scopes,
                    "scopes",
                    {
                        "resource_ordinal": resource_ordinal,
                        "scope_ordinal": scope_ordinal,
                        "name": scope.name,
                        "version": scope.version or None,
                        "schema_url": scope_spans.schema_url or None,
                        "attributes_json": _json(_attributes(scope.attributes)),
                        "dropped_attributes_count": scope.dropped_attributes_count,
                    },
                )
                for span in scope_spans.spans:
                    trace_id = _id(span.trace_id, 16, "trace_id", limitations)
                    span_id = _id(span.span_id, 8, "span_id", limitations)
                    parent_id = (
                        _id(span.parent_span_id, 8, "parent_span_id", limitations)
                        if span.parent_span_id
                        else None
                    )
                    start = int(span.start_time_unix_nano)
                    end = int(span.end_time_unix_nano)
                    duration = end - start if start and end and end >= start else None
                    if start == 0 or end == 0:
                        limitations.append(f"span:{source_ordinal}:missing_timestamp")
                    elif end < start:
                        limitations.append(f"span:{source_ordinal}:end_before_start")
                    identity = (trace_id, span_id)
                    if identity in identities:
                        limitations.append(f"span:{source_ordinal}:duplicate_identity")
                    identities.add(identity)
                    append_row(
                        spans,
                        "spans",
                        {
                            "resource_ordinal": resource_ordinal,
                            "scope_ordinal": scope_ordinal,
                            "source_ordinal": source_ordinal,
                            "trace_id": trace_id,
                            "span_id": span_id,
                            "parent_span_id": parent_id,
                            "name": span.name,
                            "kind": int(span.kind),
                            "start_time_unix_nano": start,
                            "end_time_unix_nano": end,
                            "duration_ns": duration,
                            "trace_state": span.trace_state,
                            "flags": int(span.flags),
                            "status_code": int(span.status.code),
                            "status_message": span.status.message,
                            "attributes_json": _json(_attributes(span.attributes)),
                            "dropped_attributes_count": span.dropped_attributes_count,
                            "dropped_events_count": span.dropped_events_count,
                            "dropped_links_count": span.dropped_links_count,
                        },
                    )
                    for event_ordinal, event in enumerate(span.events):
                        append_row(
                            events,
                            "events",
                            {
                                "trace_id": trace_id,
                                "span_id": span_id,
                                "source_ordinal": source_ordinal,
                                "event_ordinal": event_ordinal,
                                "time_unix_nano": int(event.time_unix_nano),
                                "name": event.name,
                                "attributes_json": _json(_attributes(event.attributes)),
                                "dropped_attributes_count": event.dropped_attributes_count,
                            },
                        )
                    for link_ordinal, link in enumerate(span.links):
                        append_row(
                            links,
                            "links",
                            {
                                "trace_id": trace_id,
                                "span_id": span_id,
                                "source_ordinal": source_ordinal,
                                "link_ordinal": link_ordinal,
                                "linked_trace_id": _id(
                                    link.trace_id, 16, "linked_trace_id", limitations
                                ),
                                "linked_span_id": _id(
                                    link.span_id, 8, "linked_span_id", limitations
                                ),
                                "trace_state": link.trace_state,
                                "flags": int(link.flags),
                                "attributes_json": _json(_attributes(link.attributes)),
                                "dropped_attributes_count": link.dropped_attributes_count,
                            },
                        )
                    source_ordinal += 1
        return _ParsedOtlp(resources, scopes, spans, events, links, tuple(sorted(set(limitations))))


def _parse_otlp(path: Path, media_type: str, *, row_limit: int) -> _ParsedOtlp:
    payload = path.read_bytes()
    try:
        from google.protobuf import json_format  # type: ignore[import-untyped]
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )
    except ImportError as error:
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "OTLP extraction requires the optional opentelemetry-proto dependency.",
            remediation=("Install flameox with the trace extra.",),
        ) from error
    request = ExportTraceServiceRequest()
    normalized_media_type = media_type.split(";", 1)[0].strip().lower()
    if normalized_media_type in {"application/x-protobuf", "application/protobuf"}:
        try:
            request.ParseFromString(payload)
        except Exception as error:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED, "Malformed OTLP protobuf."
            ) from error
    elif normalized_media_type == "application/json":
        try:
            json_format.Parse(payload.decode("utf-8"), request, ignore_unknown_fields=False)
        except Exception as error:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Malformed OTLP protobuf JSON or unknown field.",
            ) from error
    else:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            "OTLP extraction requires an explicit protobuf or JSON media type.",
            details={"media_type": media_type},
        )
    return OtlpTraceService._normalize(request, row_limit=row_limit)


def _parsed_response(response: dict[str, Any]) -> _ParsedOtlp:
    def rows(name: str) -> list[dict[str, object]]:
        value = response.get(name)
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ValueError(f"OTLP worker field {name!r} is invalid")
        return value

    limitations = response.get("limitations")
    if not isinstance(limitations, list):
        raise ValueError("OTLP worker limitations are invalid")
    return _ParsedOtlp(
        resources=rows("resources"),
        scopes=rows("scopes"),
        spans=rows("spans"),
        events=rows("events"),
        links=rows("links"),
        limitations=tuple(str(item) for item in limitations),
    )


def _id(value: bytes, expected: int, label: str, limitations: list[str]) -> str:
    if len(value) != expected:
        limitations.append(f"{label}:invalid_length:{len(value)}")
        return f"invalid-{value.hex()}"
    return value.hex()


def _attributes(values: Any) -> dict[str, Any]:
    return {item.key: _any_value(item.value) for item in values}


def _any_value(value: Any) -> Any:
    kind = value.WhichOneof("value")
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "int_value":
        return value.int_value
    if kind == "double_value":
        return value.double_value
    if kind == "bytes_value":
        return {"type": "bytes", "base64": base64.b64encode(value.bytes_value).decode("ascii")}
    if kind == "array_value":
        return [_any_value(item) for item in value.array_value.values]
    if kind == "kvlist_value":
        return _attributes(value.kvlist_value.values)
    return None
