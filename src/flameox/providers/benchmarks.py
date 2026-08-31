from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from flameox.providers.contracts import ProviderAnalysis, ProviderFailure
from flameox.workers.benchmark_samples_contract import (
    BENCHMARK_SAMPLES_WORKER,
    BenchmarkSamplesWorkerRequest,
)
from flameox.workers.harness import IsolatedWorkerHarness
from flameox.workers.pyperf_contract import PYPERF_WORKER, PyperfWorkerRequest, PyperfWorkerResult


class BenchmarkProvider:
    """Bounded native benchmark parsing and explicit multi-artifact comparisons."""

    def __init__(self, harness: IsolatedWorkerHarness) -> None:
        self.harness = harness

    def analyze(
        self,
        capability_id: str,
        paths: Sequence[Path],
        formats: Sequence[str],
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
        timeout_seconds: float,
        maximum_rss_bytes: int,
        maximum_output_bytes: int,
    ) -> ProviderAnalysis | None:
        if not paths:
            return None
        if capability_id not in {"benchmark.summary", "benchmark.scaling", "benchmark.compare"}:
            return None
        if all(format_name == "samples" for format_name in formats):
            return self._structured_samples(
                paths,
                compare_arguments=arguments if capability_id == "benchmark.compare" else None,
                max_rows=max_rows,
                timeout_seconds=timeout_seconds,
                maximum_rss_bytes=maximum_rss_bytes,
                maximum_output_bytes=maximum_output_bytes,
            )
        if any(format_name != "pyperf" for format_name in formats):
            return None
        parsed = [
            self.harness.run_typed_sync(
                PYPERF_WORKER,
                PyperfWorkerRequest(artifact_path=str(path), max_rows=max_rows),
                timeout_seconds=timeout_seconds,
                maximum_rss_bytes=maximum_rss_bytes,
                maximum_writable_growth_bytes=maximum_output_bytes,
            )
            for path in paths
        ]
        if capability_id == "benchmark.compare":
            return self._compare(parsed, arguments, max_rows=max_rows)
        rows: list[dict[str, Any]] = []
        observed = 0
        limitations: list[str] = []
        for input_index, result in enumerate(parsed):
            observed += result.measurement_count + result.warmup_count
            limitations.extend(result.limitations)
            for row in result.rows:
                if len(rows) < max_rows:
                    rows.append({"input_index": input_index, **dict(row)})
        return ProviderAnalysis(
            provider_id="pyperf",
            provider_version=parsed[0].reader_version,
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "input_count": len(parsed),
                        "measurement_count": sum(item.measurement_count for item in parsed),
                        "warmup_count": sum(item.warmup_count for item in parsed),
                        "benchmark_names": sorted(
                            {name for item in parsed for name in item.benchmark_names}
                        ),
                    },
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=observed <= len(rows),
            limitations=list(dict.fromkeys(limitations)),
        )

    def _structured_samples(
        self,
        paths: Sequence[Path],
        *,
        compare_arguments: Mapping[str, Any] | None,
        max_rows: int,
        timeout_seconds: float,
        maximum_rss_bytes: int,
        maximum_output_bytes: int,
    ) -> ProviderAnalysis:
        parsed = [
            self.harness.run_typed_sync(
                BENCHMARK_SAMPLES_WORKER,
                BenchmarkSamplesWorkerRequest(artifact_path=str(path), max_rows=max_rows),
                timeout_seconds=timeout_seconds,
                maximum_rss_bytes=maximum_rss_bytes,
                maximum_writable_growth_bytes=maximum_output_bytes,
            )
            for path in paths
        ]
        if compare_arguments is not None:
            if any(item.truncated for item in parsed):
                raise ProviderFailure(
                    "LIMIT_EXCEEDED",
                    "benchmark.compare requires complete samples; raise the startup row limit",
                )
            return self._compare_row_sets(
                [item.rows for item in parsed],
                compare_arguments,
                max_rows=max_rows,
                provider_id="benchmark-samples",
                provider_version="flameox.benchmark-samples.v1",
            )
        rows: list[dict[str, Any]] = []
        limitations: list[str] = []
        observed = 0
        for input_index, result in enumerate(parsed):
            observed += result.measurement_count + result.warmup_count
            limitations.extend(result.limitations)
            for row in result.rows:
                if len(rows) < max_rows:
                    rows.append({"input_index": input_index, **dict(row)})
        versions = sorted(
            {f"{item.producer}@{item.producer_version or 'unknown'}" for item in parsed}
        )
        return ProviderAnalysis(
            provider_id="benchmark-samples",
            provider_version="flameox.benchmark-samples.v1",
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "input_count": len(parsed),
                        "measurement_count": sum(item.measurement_count for item in parsed),
                        "warmup_count": sum(item.warmup_count for item in parsed),
                        "benchmark_names": sorted(
                            {name for item in parsed for name in item.benchmark_names}
                        ),
                        "producers": versions,
                    },
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=observed <= len(rows),
            limitations=list(dict.fromkeys(limitations)),
        )

    @staticmethod
    def _compare_row_sets(
        row_sets: Sequence[Sequence[Mapping[str, Any]]],
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
        provider_id: str,
        provider_version: str,
    ) -> ProviderAnalysis:
        if len(row_sets) < 2:
            raise ProviderFailure("INVALID_INPUT", "benchmark.compare requires at least 2 inputs")
        baseline_index = int(arguments.get("baseline_index", 0))
        if baseline_index >= len(row_sets):
            raise ProviderFailure("INVALID_INPUT", "baseline_index does not select an input")
        requested_metric = arguments.get("metric")
        series: list[dict[tuple[str, str], list[float]]] = []
        for rows in row_sets:
            values: dict[tuple[str, str], list[float]] = {}
            for row in rows:
                if row["is_warmup"]:
                    continue
                identity = (str(row["benchmark"]), str(row["unit"]))
                if requested_metric is not None and identity[0] != requested_metric:
                    continue
                value = row["value_int"] if row["value_int"] is not None else row["value_float"]
                if isinstance(value, int | float) and not isinstance(value, bool):
                    values.setdefault(identity, []).append(float(value))
            series.append(values)
        common = set(series[baseline_index])
        for values in series:
            common.intersection_update(values)
        output: list[dict[str, Any]] = []
        baseline = series[baseline_index]
        for identity in sorted(common):
            baseline_mean = fmean(baseline[identity])
            for input_index, values in enumerate(series):
                if input_index != baseline_index:
                    candidate_mean = fmean(values[identity])
                    output.append(
                        {
                            "benchmark": identity[0],
                            "unit": identity[1],
                            "baseline_index": baseline_index,
                            "candidate_index": input_index,
                            "baseline_mean": baseline_mean,
                            "candidate_mean": candidate_mean,
                            "ratio": candidate_mean / baseline_mean if baseline_mean else None,
                        }
                    )
        return ProviderAnalysis(
            provider_id=provider_id,
            provider_version=provider_version,
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "input_count": len(row_sets),
                        "compatible_metric_count": len(common),
                    },
                },
                {"type": "table", "rows": output[:max_rows]},
            ],
            rows_observed=len(output),
            complete=len(output) <= max_rows,
            limitations=[
                "Ratios summarize observed sample means and do not establish causal improvement."
            ],
        )

    @staticmethod
    def _compare(
        parsed: Sequence[PyperfWorkerResult], arguments: Mapping[str, Any], *, max_rows: int
    ) -> ProviderAnalysis:
        if len(parsed) < 2:
            raise ProviderFailure("INVALID_INPUT", "benchmark.compare requires at least 2 inputs")
        if any(result.truncated for result in parsed):
            raise ProviderFailure(
                "LIMIT_EXCEEDED",
                "benchmark.compare requires complete samples; raise the startup row limit",
            )
        return BenchmarkProvider._compare_row_sets(
            [result.rows for result in parsed],
            arguments,
            max_rows=max_rows,
            provider_id="pyperf",
            provider_version=parsed[0].reader_version,
        )
