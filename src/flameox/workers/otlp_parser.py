from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flameox.runtime_errors import DomainError, ErrorCode


@dataclass(frozen=True, slots=True)
class ParsedOtlp:
    resources: list[dict[str, object]]
    scopes: list[dict[str, object]]
    spans: list[dict[str, object]]
    events: list[dict[str, object]]
    links: list[dict[str, object]]
    limitations: tuple[str, ...]


class OtlpRowLimitExceeded(Exception):
    def __init__(self, parsed: ParsedOtlp, counts: dict[str, int]) -> None:
        super().__init__("OTLP normalization row limit exceeded")
        self.parsed = parsed
        self.counts = counts


def parse_otlp(
    path: Path,
    media_type: str,
    *,
    row_limit: int,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> ParsedOtlp:
    payload = path.read_bytes()
    try:
        from google.protobuf import json_format  # type: ignore[import-untyped]
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )
    except ImportError as error:
        raise DomainError(
            ErrorCode.UNAVAILABLE_CAPABILITY,
            "OTLP extraction requires the optional trace provider packages.",
        ) from error
    request = ExportTraceServiceRequest()
    normalized = media_type.split(";", 1)[0].strip().lower()
    if normalized in {"application/x-protobuf", "application/protobuf"}:
        try:
            request.ParseFromString(payload)
        except Exception as error:
            raise DomainError(ErrorCode.DECODE_FAILURE, "Malformed OTLP protobuf.") from error
    elif normalized == "application/json":
        try:
            json_format.Parse(payload.decode("utf-8"), request, ignore_unknown_fields=False)
        except Exception as error:
            raise DomainError(
                ErrorCode.DECODE_FAILURE,
                "Malformed OTLP protobuf JSON or unknown field.",
            ) from error
    else:
        raise DomainError(
            ErrorCode.DECODE_FAILURE,
            "OTLP requires an explicit protobuf or JSON media type.",
        )
    return _normalize(request, row_limit=row_limit, start_ns=start_ns, end_ns=end_ns)


def _normalize(
    request: Any, *, row_limit: int, start_ns: int | None, end_ns: int | None
) -> ParsedOtlp:
    resources: list[dict[str, object]] = []
    scopes: list[dict[str, object]] = []
    spans: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    links: list[dict[str, object]] = []
    limitations: list[str] = []
    identities: set[tuple[str, str]] = set()
    counts = {"resources": 0, "scopes": 0, "spans": 0, "events": 0, "links": 0}

    def parsed() -> ParsedOtlp:
        return ParsedOtlp(
            resources,
            scopes,
            spans,
            events,
            links,
            tuple(sorted(set(limitations))),
        )

    def append(table: list[dict[str, object]], name: str, row: dict[str, object]) -> None:
        counts[name] += 1
        if sum(counts.values()) > row_limit:
            limitations.append("otlp_row_limit_exceeded")
            raise OtlpRowLimitExceeded(parsed(), counts.copy())
        table.append(row)

    source_ordinal = 0
    for resource_ordinal, resource_spans in enumerate(request.resource_spans):
        resource = resource_spans.resource
        append(
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
            append(
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
                if (
                    start_ns is not None
                    and end_ns is not None
                    and (end <= start_ns or start >= end_ns)
                ):
                    source_ordinal += 1
                    continue
                if start == 0 or end == 0:
                    limitations.append(f"span:{source_ordinal}:missing_timestamp")
                elif end < start:
                    limitations.append(f"span:{source_ordinal}:end_before_start")
                identity = (trace_id, span_id)
                if identity in identities:
                    limitations.append(f"span:{source_ordinal}:duplicate_identity")
                identities.add(identity)
                append(
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
                    append(
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
                    append(
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
                            "linked_span_id": _id(link.span_id, 8, "linked_span_id", limitations),
                            "trace_state": link.trace_state,
                            "flags": int(link.flags),
                            "attributes_json": _json(_attributes(link.attributes)),
                            "dropped_attributes_count": link.dropped_attributes_count,
                        },
                    )
                source_ordinal += 1
    return parsed()


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


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
