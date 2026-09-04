from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from flameox.canonical import canonical_bytes, sha256_id
from flameox.providers.contracts import ProviderAnalysis, ProviderFailure
from flameox.providers.inference_comparison import assess_comparison, field_identities

_VLLM_MAX_BYTES = 16 * 1024 * 1024
_SGLANG_MAX_BYTES = 1024 * 1024
_MOONCAKE_MAX_LINE_BYTES = 64 * 1024


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class _VllmMetrics(_Model):
    completed: int = Field(ge=0)
    total_input: int = Field(ge=0)
    total_output: int = Field(ge=0)
    request_throughput: float = Field(ge=0)
    request_goodput: float | None = Field(default=None, ge=0)
    output_throughput: float = Field(ge=0)
    total_token_throughput: float = Field(ge=0)
    mean_ttft_ms: float = Field(ge=0)
    median_ttft_ms: float = Field(ge=0)
    std_ttft_ms: float = Field(ge=0)
    percentiles_ttft_ms: list[list[int | float]] = Field(default_factory=list)
    mean_tpot_ms: float = Field(ge=0)
    median_tpot_ms: float = Field(ge=0)
    std_tpot_ms: float = Field(ge=0)
    percentiles_tpot_ms: list[list[int | float]] = Field(default_factory=list)
    mean_itl_ms: float = Field(ge=0)
    median_itl_ms: float = Field(ge=0)
    std_itl_ms: float = Field(ge=0)
    percentiles_itl_ms: list[list[int | float]] = Field(default_factory=list)
    mean_e2el_ms: float = Field(ge=0)
    median_e2el_ms: float = Field(ge=0)
    std_e2el_ms: float = Field(ge=0)
    percentiles_e2el_ms: list[list[int | float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def numbers_are_finite(self) -> _VllmMetrics:
        for name, value in self:
            values = value if isinstance(value, list) else [value]
            for item in values:
                members = item if isinstance(item, tuple) else [item]
                if any(
                    isinstance(member, float) and not math.isfinite(member) for member in members
                ):
                    raise ValueError(f"{name} must contain only finite numbers")
        for pairs in (
            self.percentiles_ttft_ms,
            self.percentiles_tpot_ms,
            self.percentiles_itl_ms,
            self.percentiles_e2el_ms,
        ):
            if any(
                len(pair) != 2
                or not math.isfinite(float(pair[0]))
                or not math.isfinite(float(pair[1]))
                or not 0 <= float(pair[0]) <= 100
                or float(pair[1]) < 0
                for pair in pairs
            ):
                raise ValueError("percentiles must contain a rank from 0 to 100 and a value >= 0")
        return self


class _VllmDocument(_Model):
    metrics: _VllmMetrics
    successful_requests: int = Field(ge=0)
    failed_requests: int = Field(ge=0)
    total_requests: int = Field(ge=0)
    actual_duration: float = Field(ge=0)
    time_scale: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def counts_match(self) -> _VllmDocument:
        if self.successful_requests + self.failed_requests != self.total_requests:
            raise ValueError("successful and failed request counts do not match total_requests")
        if self.successful_requests != self.metrics.completed:
            raise ValueError("successful_requests does not match metrics.completed")
        if not math.isfinite(self.actual_duration) or not math.isfinite(self.time_scale):
            raise ValueError("duration and time scale must be finite")
        return self


def _observed_scalars(payload: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name in names:
        value = payload.get(name)
        is_text = isinstance(value, str) and bool(value) and len(value) <= 4_096
        is_number = (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
        if is_text or is_number:
            observed[name] = value
    return observed


_SGLANG_REQUIRED = {"duration", "completed", "total_input_tokens", "total_output_tokens"}
_SGLANG_SCALARS = {
    *_SGLANG_REQUIRED,
    "num_prompts",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "accept_length",
    "concurrency",
    *{
        f"{stat}_{family}_ms"
        for family in ("e2e_latency", "ttft", "tpot", "itl")
        for stat in ("mean", "median", "std", "p90", "p95", "p99")
    },
}
_SENSITIVE_SGLANG_FIELDS = {
    "input_lens",
    "output_lens",
    "ttfts",
    "itls",
    "generated_texts",
    "errors",
}


class InferenceExportProvider:
    """Bounded offline readers that never project prompts, generations, or endpoints."""

    def analyze(
        self,
        capability_id: str,
        paths: Sequence[Path],
        formats: Sequence[str],
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
    ) -> ProviderAnalysis | None:
        supported = {"vllm-benchmark", "sglang-benchmark", "mooncake-trace"}
        if capability_id not in {"inference.summary", "inference.compare"} or not formats:
            return None
        if any(format_name not in supported for format_name in formats):
            return None
        if capability_id == "inference.summary":
            if len(paths) != 1:
                raise ProviderFailure("INVALID_INPUT", "inference.summary accepts one input")
            return self._summary(paths[0], formats[0], max_rows=max_rows)
        if len(set(formats)) != 1:
            raise ProviderFailure(
                "INVALID_INPUT", "inference.compare requires one compatible export format"
            )
        analyses = [self._summary(path, formats[0], max_rows=max_rows) for path in paths]
        return self._compare(analyses, formats[0], arguments, max_rows=max_rows)

    def _summary(self, path: Path, format_name: str, *, max_rows: int) -> ProviderAnalysis:
        if format_name == "vllm-benchmark":
            return self._vllm(path, max_rows=max_rows)
        if format_name == "sglang-benchmark":
            return self._sglang(path, max_rows=max_rows)
        return self._mooncake(path, max_rows=max_rows)

    @staticmethod
    def _load_json_object(path: Path, *, maximum_bytes: int, label: str) -> dict[str, Any]:
        try:
            if path.stat().st_size > maximum_bytes:
                raise ValueError(f"{label} exceeds its document-size bound")
            payload = json.loads(path.read_bytes())
            if not isinstance(payload, dict):
                raise ValueError(f"{label} must be a JSON object")
            return payload
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ProviderFailure("DECODE_FAILURE", f"Invalid {label} export") from error

    @staticmethod
    def _normalize_vllm(payload: dict[str, Any]) -> dict[str, Any]:
        if "metrics" in payload:
            return payload
        completed = payload.get("completed")
        total_requests = payload.get("num_prompts", completed)
        if not isinstance(completed, int) or not isinstance(total_requests, int):
            return payload
        metrics = dict(payload)
        metrics.setdefault("total_input", metrics.get("total_input_tokens"))
        metrics.setdefault("total_output", metrics.get("total_output_tokens"))
        return {
            "metrics": metrics,
            "successful_requests": completed,
            "failed_requests": max(0, total_requests - completed),
            "total_requests": total_requests,
            "actual_duration": payload.get("duration"),
            "time_scale": 1.0,
        }

    def _vllm(self, path: Path, *, max_rows: int) -> ProviderAnalysis:
        payload = self._load_json_object(
            path, maximum_bytes=_VLLM_MAX_BYTES, label="vLLM benchmark"
        )
        try:
            document = _VllmDocument.model_validate(self._normalize_vllm(payload))
        except ValidationError as error:
            raise ProviderFailure("DECODE_FAILURE", "Invalid vLLM benchmark export") from error
        metrics = document.metrics
        observed = _observed_scalars(
            payload,
            (
                "model",
                "served_model_name",
                "tokenizer",
                "backend",
                "dataset_name",
                "request_rate",
                "max_concurrency",
                "num_prompts",
            ),
        )
        system: dict[str, Any] = {
            **(
                {"model": observed.get("model", observed.get("served_model_name"))}
                if "model" in observed or "served_model_name" in observed
                else {}
            ),
            **{name: observed[name] for name in ("tokenizer", "backend") if name in observed},
        }
        workload: dict[str, Any] = {
            "total_requests": document.total_requests,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "time_scale": document.time_scale,
            **{
                name: observed[name]
                for name in ("dataset_name", "request_rate", "max_concurrency", "num_prompts")
                if name in observed
            },
        }
        comparison_identity, unavailable = field_identities(
            {
                "system": (system, ("model", "tokenizer", "backend")),
                "workload": (
                    workload,
                    (
                        "total_requests",
                        "total_input_tokens",
                        "total_output_tokens",
                        "time_scale",
                        "dataset_name",
                        "request_rate",
                        "max_concurrency",
                        "num_prompts",
                    ),
                ),
            }
        )
        rows: list[dict[str, Any]] = []

        def add(name: str, value: int | float | None, unit: str, aggregation: str) -> None:
            if value is not None:
                rows.append(
                    {
                        "name": name,
                        "value": float(value),
                        "unit": unit,
                        "aggregation": aggregation,
                    }
                )

        add("vllm.request_throughput", metrics.request_throughput, "requests/sec", "aggregate")
        add("vllm.request_goodput", metrics.request_goodput, "requests/sec", "aggregate")
        add("vllm.output_throughput", metrics.output_throughput, "tokens/sec", "aggregate")
        add(
            "vllm.total_token_throughput",
            metrics.total_token_throughput,
            "tokens/sec",
            "aggregate",
        )
        add("vllm.total_input_tokens", metrics.total_input, "tokens", "sum")
        add("vllm.total_output_tokens", metrics.total_output, "tokens", "sum")
        add("vllm.completed_requests", document.successful_requests, "requests", "count")
        add("vllm.failed_requests", document.failed_requests, "requests", "count")
        add("vllm.total_requests", document.total_requests, "requests", "count")
        add("vllm.duration_seconds", document.actual_duration, "s", "aggregate")
        for short_name, long_name in (
            ("ttft", "time_to_first_token"),
            ("tpot", "time_per_output_token"),
            ("itl", "inter_token_latency"),
            ("e2el", "end_to_end_latency"),
        ):
            add(
                f"vllm.{long_name}.mean_ms", getattr(metrics, f"mean_{short_name}_ms"), "ms", "mean"
            )
            add(
                f"vllm.{long_name}.median_ms",
                getattr(metrics, f"median_{short_name}_ms"),
                "ms",
                "median",
            )
            add(f"vllm.{long_name}.std_ms", getattr(metrics, f"std_{short_name}_ms"), "ms", "std")
            for rank_value, metric_value in getattr(metrics, f"percentiles_{short_name}_ms"):
                rank = float(rank_value)
                label = str(int(rank)) if rank.is_integer() else str(rank)
                add(
                    f"vllm.{long_name}.p{label}_ms",
                    metric_value,
                    "ms",
                    "percentile",
                )
        returned = rows[:max_rows]
        return ProviderAnalysis(
            provider_id="vllm-benchmark",
            provider_version="aggregate-v1",
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "successful_requests": document.successful_requests,
                        "failed_requests": document.failed_requests,
                        "measurement_count": len(rows),
                        "comparison_identity": comparison_identity,
                        "comparison_identity_unavailable": unavailable,
                    },
                },
                {"type": "table", "rows": returned},
            ],
            rows_observed=len(rows),
            complete=len(rows) <= max_rows,
            limitations=["Raw prompts, generations, errors, and server endpoints are excluded."],
        )

    def _sglang(self, path: Path, *, max_rows: int) -> ProviderAnalysis:
        try:
            if path.stat().st_size > _SGLANG_MAX_BYTES:
                raise ValueError("SGLang export exceeds its document-size bound")
            lines = [line for line in path.read_text().splitlines() if line.strip()]
            if len(lines) != 1:
                raise ValueError("SGLang export must contain one aggregate JSONL record")
            payload = json.loads(lines[0])
            if not isinstance(payload, dict) or not _SGLANG_REQUIRED.issubset(payload):
                raise ValueError("SGLang aggregate fields are missing")
            if _SENSITIVE_SGLANG_FIELDS.intersection(payload) or any(
                isinstance(value, list | dict) for value in payload.values()
            ):
                raise ValueError("SGLang detailed output is not accepted")
            selected = {key: payload[key] for key in _SGLANG_SCALARS.intersection(payload)}
            if any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0
                for value in selected.values()
            ):
                raise ValueError("SGLang metrics must be finite non-negative numbers")
            if "num_prompts" in selected and selected["completed"] > selected["num_prompts"]:
                raise ValueError("SGLang completed count exceeds num_prompts")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ProviderFailure("DECODE_FAILURE", "Invalid aggregate SGLang export") from error
        rows = [self._sglang_row(name, selected[name]) for name in sorted(selected)]
        observed = _observed_scalars(
            payload,
            (
                "model",
                "tokenizer",
                "backend",
                "dataset_name",
                "request_rate",
                "max_concurrency",
            ),
        )
        system: dict[str, Any] = {
            name: observed[name] for name in ("model", "tokenizer", "backend") if name in observed
        }
        workload: dict[str, Any] = {
            name: selected[name]
            for name in (
                "completed",
                "num_prompts",
                "total_input_tokens",
                "total_output_tokens",
                "concurrency",
            )
            if name in selected
        }
        workload.update(
            {
                name: observed[name]
                for name in ("dataset_name", "request_rate", "max_concurrency")
                if name in observed
            }
        )
        comparison_identity, unavailable = field_identities(
            {
                "system": (system, ("model", "tokenizer", "backend")),
                "workload": (
                    workload,
                    (
                        "completed",
                        "num_prompts",
                        "total_input_tokens",
                        "total_output_tokens",
                        "concurrency",
                        "dataset_name",
                        "request_rate",
                        "max_concurrency",
                    ),
                ),
            }
        )
        return ProviderAnalysis(
            provider_id="sglang-benchmark",
            provider_version="aggregate-v1",
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "completed_requests": int(selected["completed"]),
                        "measurement_count": len(rows),
                        "comparison_identity": comparison_identity,
                        "comparison_identity_unavailable": unavailable,
                    },
                },
                {"type": "table", "rows": rows[:max_rows]},
            ],
            rows_observed=len(rows),
            complete=len(rows) <= max_rows,
            limitations=["Detailed SGLang output is rejected to exclude prompts and generations."],
        )

    @staticmethod
    def _sglang_row(name: str, value: int | float) -> dict[str, Any]:
        if name == "duration":
            unit, aggregation = "s", "aggregate"
        elif name in {"completed", "num_prompts"}:
            unit, aggregation = "requests", "count"
        elif name in {"total_input_tokens", "total_output_tokens"}:
            unit, aggregation = "tokens", "sum"
        elif "throughput" in name:
            unit = "requests/sec" if name == "request_throughput" else "tokens/sec"
            aggregation = "rate"
        elif name == "accept_length":
            unit, aggregation = "tokens", "mean"
        elif name == "concurrency":
            unit, aggregation = "dimensionless", "mean"
        else:
            unit = "ms"
            aggregation = (
                "percentile" if name.startswith(("p90_", "p95_", "p99_")) else name.split("_", 1)[0]
            )
        return {
            "name": f"sglang.{name}",
            "value": float(value),
            "unit": unit,
            "aggregation": aggregation,
        }

    def _mooncake(self, path: Path, *, max_rows: int) -> ProviderAnalysis:
        rows: list[dict[str, Any]] = []
        limitations: list[str] = []
        observed = 0
        previous_timestamp: int | None = None
        first_timestamp: int | None = None
        max_input_length = 0
        max_output_length = 0
        workload_digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for line_index, raw in enumerate(stream):
                    if not raw.strip():
                        continue
                    if len(raw) > _MOONCAKE_MAX_LINE_BYTES:
                        raise ValueError("Mooncake trace line exceeds its byte bound")
                    payload = json.loads(raw)
                    row = self._mooncake_row(payload, line_index)
                    if first_timestamp is None:
                        first_timestamp = int(row["timestamp_ms"])
                    if previous_timestamp is not None and row["timestamp_ms"] < previous_timestamp:
                        limitations.append(
                            f"Request {line_index} timestamp regressed below the prior row."
                        )
                    previous_timestamp = int(row["timestamp_ms"])
                    observed += 1
                    max_input_length = max(max_input_length, int(row["input_length"]))
                    max_output_length = max(max_output_length, int(row["output_length"]))
                    workload_digest.update(
                        canonical_bytes(
                            [
                                row["timestamp_ms"],
                                row["input_length"],
                                row["output_length"],
                                row["prefix_hash_count"],
                            ]
                        )
                        + b"\n"
                    )
                    if len(rows) < max_rows:
                        rows.append(row)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ProviderFailure("DECODE_FAILURE", "Invalid Mooncake request trace") from error
        if observed == 0:
            raise ProviderFailure("DECODE_FAILURE", "Mooncake request trace is empty")
        if first_timestamp != 0:
            limitations.append("The first request timestamp is not zero milliseconds.")
        if observed > max_rows:
            limitations.append("Mooncake requests were truncated by the declared row bound.")
        return ProviderAnalysis(
            provider_id="mooncake-trace",
            provider_version="request-trace-v1",
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "request_count": observed,
                        "max_input_length": max_input_length,
                        "max_output_length": max_output_length,
                        "comparison_identity": {"workload": sha256_id(workload_digest.hexdigest())},
                        "comparison_identity_unavailable": ["system"],
                    },
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=observed <= max_rows,
            limitations=[
                *limitations,
                "Prompts, tools, payloads, and prefix hash values are excluded.",
            ],
        )

    @staticmethod
    def _mooncake_row(payload: Any, line_index: int) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Mooncake trace rows must be JSON objects")
        values = [payload.get(name) for name in ("timestamp", "input_length", "output_length")]
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values
        ):
            raise ValueError(
                "Mooncake trace timing and length fields must be non-negative integers"
            )
        hash_ids = payload.get("hash_ids", [])
        if not isinstance(hash_ids, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in hash_ids
        ):
            raise ValueError("Mooncake hash_ids must be non-negative integers")
        return {
            "line_index": line_index,
            "timestamp_ms": values[0],
            "input_length": values[1],
            "output_length": values[2],
            "prefix_hash_count": len(hash_ids),
        }

    @staticmethod
    def _compare(
        analyses: Sequence[ProviderAnalysis],
        format_name: str,
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
    ) -> ProviderAnalysis:
        if len(analyses) < 2:
            raise ProviderFailure("INVALID_INPUT", "inference.compare requires at least 2 inputs")
        if any(not analysis.complete for analysis in analyses):
            raise ProviderFailure(
                "LIMIT_EXCEEDED", "inference.compare requires complete bounded inputs"
            )
        baseline_index = int(arguments.get("baseline_index", 0))
        if baseline_index >= len(analyses):
            raise ProviderFailure("INVALID_INPUT", "baseline_index does not select an input")
        defaults = {
            "vllm-benchmark": "vllm.request_throughput",
            "sglang-benchmark": "sglang.request_throughput",
            "mooncake-trace": "input_length",
        }
        metric = str(arguments.get("metric") or defaults[format_name])
        series = [InferenceExportProvider._values(analysis, metric) for analysis in analyses]
        if any(not values for values in series):
            raise ProviderFailure(
                "INVALID_INPUT", f"Every input must expose the numeric metric {metric}"
            )
        baseline = statistics.fmean(series[baseline_index])
        compatibility, identity_differences, identity_unavailable = assess_comparison(
            analyses, arguments
        )
        rows = []
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
            provider_id=format_name,
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
                "Comparisons use arithmetic means of prompt-free normalized metrics.",
                *(
                    ["The inference system identity is unavailable from one or more exports."]
                    if identity_unavailable
                    else []
                ),
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
    def _values(analysis: ProviderAnalysis, metric: str) -> list[float]:
        rows = analysis.blocks[-1].get("rows")
        if not isinstance(rows, list):
            return []
        values: list[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = row.get("value") if row.get("name") == metric else row.get(metric)
            if isinstance(value, int | float) and not isinstance(value, bool):
                values.append(float(value))
        return values
