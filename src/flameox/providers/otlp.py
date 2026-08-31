from __future__ import annotations

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
        counts = (
            dict(response.counts)
            if response.row_limit_exceeded
            else {table: len(values) for table, values in tables}
        )
        observed = sum(counts.values())
        limitations = list(response.limitations)
        if response.row_limit_exceeded and "otlp_row_limit_exceeded" not in limitations:
            limitations.append("otlp_row_limit_exceeded")
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
