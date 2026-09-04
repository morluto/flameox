from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from flameox.canonical import digest_model
from flameox.providers.contracts import ProviderAnalysis, ProviderFailure
from flameox.providers.inference_comparison import assess_comparison
from flameox.workers.aiperf_contract import (
    AIPERF_WORKER,
    AIPerfProjectionRow,
    AIPerfWorkerRequest,
)
from flameox.workers.harness import IsolatedWorkerHarness

_PROJECTED_LINE_BYTES = 16 * 1024


class AIPerfProvider:
    """Project one explicit AIPerf export without retaining prompts or run state."""

    def __init__(self, harness: IsolatedWorkerHarness) -> None:
        self.harness = harness

    def analyze(
        self,
        path: Path,
        *,
        max_rows: int,
        timeout_seconds: float,
        maximum_rss_bytes: int,
        maximum_output_bytes: int,
    ) -> ProviderAnalysis:
        with self.harness.run_typed_sync_session(
            AIPERF_WORKER,
            AIPerfWorkerRequest(
                artifact_path=str(path),
                max_rows=max_rows,
                max_line_bytes=1024 * 1024,
            ),
            timeout_seconds=timeout_seconds,
            maximum_rss_bytes=maximum_rss_bytes,
            maximum_writable_growth_bytes=maximum_output_bytes,
        ) as (result, job_root):
            projection = self.harness.validate_output_file(job_root, result.output)
            rows = self._read_projection(projection, expected_rows=result.row_count)
        successful = [row for row in rows if row["outcome"] == "succeeded"]
        latencies = [
            int(value) for row in successful if (value := row.get("latency_ns")) is not None
        ]
        ttfts = [int(value) for row in successful if (value := row.get("ttft_ns")) is not None]
        scheduled = [int(value) for row in rows if (value := row.get("scheduled_ns")) is not None]
        first_scheduled = min(scheduled) if scheduled else None
        workload = [
            {
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "scheduled_offset_ns": (
                    int(row["scheduled_ns"]) - first_scheduled
                    if first_scheduled is not None and row.get("scheduled_ns") is not None
                    else None
                ),
            }
            for row in rows
        ]
        limitations = [
            "Prompt and response bodies are intentionally excluded from normalized evidence."
        ]
        if result.truncated:
            limitations.append("AIPerf requests were truncated by the declared row bound.")
        return ProviderAnalysis(
            provider_id="aiperf",
            provider_version=result.aiperf_version,
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "request_count": len(rows),
                        "successful_requests": len(successful),
                        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
                        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
                        "median_ttft_ns": statistics.median(ttfts) if ttfts else None,
                        "p95_latency_ns": self._percentile(latencies, 0.95),
                        "comparison_identity": {
                            "workload": digest_model(
                                workload, projection="flameox.inference.workload/v1"
                            )
                        },
                        "comparison_identity_unavailable": ["system"],
                    },
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=result.row_count + int(result.truncated),
            complete=not result.truncated,
            limitations=limitations,
        )

    @staticmethod
    def _read_projection(path: Path, *, expected_rows: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            with path.open("rb") as stream:
                while raw := stream.readline(_PROJECTED_LINE_BYTES + 1):
                    if len(raw) > _PROJECTED_LINE_BYTES:
                        raise ValueError("AIPerf projection row exceeds its byte bound")
                    projected = AIPerfProjectionRow.model_validate_json(raw)
                    rows.append(projected.model_dump(mode="json"))
        except (OSError, ValidationError, ValueError) as error:
            raise ProviderFailure(
                "DECODE_FAILURE", "AIPerf returned an invalid prompt-free projection"
            ) from error
        if len(rows) != expected_rows:
            raise ProviderFailure(
                "DECODE_FAILURE", "AIPerf projection count contradicts its worker receipt"
            )
        return rows

    @staticmethod
    def _percentile(values: list[int], quantile: float) -> int | None:
        if not values:
            return None
        ordered = sorted(values)
        index = round((len(ordered) - 1) * quantile)
        return ordered[index]

    @staticmethod
    def compare(
        analyses: Sequence[ProviderAnalysis],
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
    ) -> ProviderAnalysis:
        if len(analyses) < 2:
            raise ProviderFailure("INVALID_INPUT", "inference.compare requires at least 2 inputs")
        if any(not item.complete for item in analyses):
            raise ProviderFailure(
                "LIMIT_EXCEEDED",
                "inference.compare requires complete request projections within the row limit",
            )
        baseline_index = int(arguments.get("baseline_index", 0))
        if baseline_index >= len(analyses):
            raise ProviderFailure("INVALID_INPUT", "baseline_index does not select an input")
        metric = str(arguments.get("metric") or "latency_ns")
        if metric not in {
            "input_tokens",
            "output_tokens",
            "ttft_ns",
            "latency_ns",
            "tpot_ns",
            "mean_itl_ns",
        }:
            raise ProviderFailure("INVALID_INPUT", f"Unsupported inference metric: {metric}")
        series = [AIPerfProvider._metric_values(item, metric) for item in analyses]
        if any(not values for values in series):
            raise ProviderFailure(
                "INVALID_INPUT", f"Every input must contain successful {metric} observations"
            )
        baseline = statistics.fmean(series[baseline_index])
        compatibility, identity_differences, identity_unavailable = assess_comparison(
            analyses, arguments
        )
        rows: list[dict[str, Any]] = []
        for index, values in enumerate(series):
            if index == baseline_index:
                continue
            candidate = statistics.fmean(values)
            rows.append(
                {
                    "metric": metric,
                    "baseline_index": baseline_index,
                    "candidate_index": index,
                    "baseline_mean": baseline,
                    "candidate_mean": candidate,
                    "ratio": candidate / baseline if baseline else None,
                    "baseline_samples": len(series[baseline_index]),
                    "candidate_samples": len(values),
                    "compatibility": compatibility,
                    "identity_differences": identity_differences,
                    "identity_unavailable": identity_unavailable,
                }
            )
        return ProviderAnalysis(
            provider_id="aiperf",
            provider_version=analyses[0].provider_version,
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "input_count": len(analyses),
                        "metric": metric,
                        "compatibility": compatibility,
                        "identity_differences": identity_differences,
                        "identity_unavailable": identity_unavailable,
                    },
                },
                {"type": "table", "rows": rows[:max_rows]},
            ],
            rows_observed=len(rows),
            complete=len(rows) <= max_rows,
            limitations=[
                "Ratios compare arithmetic means of successful prompt-free request projections.",
                "The inference system identity is unavailable from AIPerf request records.",
                *(
                    [
                        "Known-different identities were compared only because heterogeneous "
                        "mode was explicit."
                    ]
                    if compatibility == "heterogeneous"
                    else []
                ),
            ],
        )

    @staticmethod
    def _metric_values(analysis: ProviderAnalysis, metric: str) -> list[float]:
        rows = analysis.blocks[1].get("rows", [])
        if not isinstance(rows, list):
            return []
        values: list[float] = []
        for row in rows:
            if not isinstance(row, dict) or row.get("outcome") != "succeeded":
                continue
            value = row.get(metric)
            if isinstance(value, int | float) and not isinstance(value, bool):
                values.append(float(value))
        return values
