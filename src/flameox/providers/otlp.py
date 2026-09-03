from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flameox.providers.contracts import ProviderAnalysis
from flameox.workers.harness import IsolatedWorkerHarness
from flameox.workers.otlp_contract import OTLP_WORKER, OtlpWorkerRequest


class OtlpProvider:
    """Normalize one explicit OTLP trace through the isolated protobuf reader."""

    def __init__(self, harness: IsolatedWorkerHarness) -> None:
        self.harness = harness

    def analyze(
        self,
        path: Path,
        capability_id: str,
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
        timeout_seconds: float,
        maximum_rss_bytes: int,
        maximum_output_bytes: int,
    ) -> ProviderAnalysis:
        media_type = (
            "application/json" if path.suffix.casefold() == ".json" else "application/x-protobuf"
        )
        start_ns = arguments.get("start_ns") if capability_id == "trace.window" else None
        end_ns = arguments.get("end_ns") if capability_id == "trace.window" else None
        response = self.harness.run_typed_sync(
            OTLP_WORKER,
            OtlpWorkerRequest(
                artifact_path=str(path),
                media_type=media_type,
                row_limit=max_rows,
                start_ns=start_ns,
                end_ns=end_ns,
            ),
            timeout_seconds=timeout_seconds,
            maximum_rss_bytes=maximum_rss_bytes,
            maximum_writable_growth_bytes=maximum_output_bytes,
        )
        tables = (
            ("resources", response.resources),
            ("scopes", response.scopes),
            ("spans", response.spans),
            ("events", response.events),
            ("links", response.links),
        )
        rows: list[dict[str, Any]] = [
            {"table": table, **dict(row)} for table, values in tables for row in values
        ]
        counts: Mapping[str, int] = (
            {str(name): count for name, count in response.counts.items()}
            if response.row_limit_exceeded
            else {table: len(values) for table, values in tables}
        )
        observed = sum(counts.values())
        limitations = list(response.limitations)
        if response.row_limit_exceeded and "otlp_row_limit_exceeded" not in limitations:
            limitations.append("otlp_row_limit_exceeded")
        if capability_id == "trace.operations":
            return self._operations(response, counts, limitations, max_rows=max_rows)
        if capability_id == "trace.lifecycle":
            return self._lifecycle(response, counts, limitations, max_rows=max_rows)
        return ProviderAnalysis(
            provider_id="otlp",
            provider_version=OTLP_WORKER.implementation,
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        **{f"{name}_count": count for name, count in counts.items()},
                        "counts_are_lower_bounds": response.row_limit_exceeded,
                    },
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=not response.row_limit_exceeded,
            limitations=limitations,
        )

    @staticmethod
    def _operations(
        response: Any,
        counts: Mapping[str, int],
        limitations: list[str],
        *,
        max_rows: int,
    ) -> ProviderAnalysis:
        aggregates: dict[tuple[str, int, int], list[int]] = defaultdict(lambda: [0, 0, 0])
        missing_duration_count = 0
        for span in response.spans:
            duration = span.get("duration_ns")
            if not isinstance(duration, int):
                missing_duration_count += 1
                continue
            key = (str(span["name"]), int(span["kind"]), int(span["status_code"]))
            aggregate = aggregates[key]
            aggregate[0] += 1
            aggregate[1] += duration
            aggregate[2] = max(aggregate[2], duration)
        rows = [
            {
                "operation": operation,
                "span_kind": kind,
                "status_code": status,
                "call_count": values[0],
                "total_duration_ns": values[1],
                "mean_duration_ns": values[1] / values[0],
                "max_duration_ns": values[2],
            }
            for (operation, kind, status), values in sorted(
                aggregates.items(), key=lambda item: (-item[1][1], item[0])
            )
        ]
        if missing_duration_count:
            limitations.append(
                f"{missing_duration_count} span(s) without a valid duration were omitted."
            )
        if response.row_limit_exceeded:
            limitations.append("Operation aggregates cover only the bounded OTLP prefix.")
        complete = not response.row_limit_exceeded and len(rows) <= max_rows
        return ProviderAnalysis(
            provider_id="otlp",
            provider_version=OTLP_WORKER.implementation,
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "operation_count": len(aggregates),
                        "span_count": counts.get("spans", len(response.spans)),
                        "missing_duration_count": missing_duration_count,
                    },
                },
                {"type": "table", "rows": rows[:max_rows]},
            ],
            rows_observed=len(rows) + int(response.row_limit_exceeded),
            complete=complete,
            limitations=limitations,
        )

    @staticmethod
    def _lifecycle(
        response: Any,
        counts: Mapping[str, int],
        limitations: list[str],
        *,
        max_rows: int,
    ) -> ProviderAnalysis:
        rows: list[dict[str, Any]] = []
        for span in response.spans:
            base = {
                "trace_id": span["trace_id"],
                "span_id": span["span_id"],
                "parent_span_id": span["parent_span_id"],
                "operation": span["name"],
            }
            start = span.get("start_time_unix_nano")
            end = span.get("end_time_unix_nano")
            if isinstance(start, int) and start > 0:
                rows.append({"transition": "span_started", "time_unix_nano": start, **base})
            if isinstance(end, int) and end > 0:
                rows.append({"transition": "span_ended", "time_unix_nano": end, **base})
        for event in response.events:
            rows.append(
                {
                    "transition": "span_event",
                    "time_unix_nano": event["time_unix_nano"],
                    "trace_id": event["trace_id"],
                    "span_id": event["span_id"],
                    "parent_span_id": None,
                    "operation": event["name"],
                }
            )
        rows.sort(
            key=lambda row: (
                int(row["time_unix_nano"]),
                str(row["span_id"]),
                str(row["transition"]),
            )
        )
        if response.row_limit_exceeded:
            limitations.append("Lifecycle transitions cover only the bounded OTLP prefix.")
        complete = not response.row_limit_exceeded and len(rows) <= max_rows
        return ProviderAnalysis(
            provider_id="otlp",
            provider_version=OTLP_WORKER.implementation,
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "transition_count": len(rows),
                        "span_count": counts.get("spans", len(response.spans)),
                        "event_count": counts.get("events", len(response.events)),
                    },
                },
                {"type": "table", "rows": rows[:max_rows]},
            ],
            rows_observed=len(rows) + int(response.row_limit_exceeded),
            complete=complete,
            limitations=limitations,
        )
