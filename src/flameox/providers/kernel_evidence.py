from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from flameox.canonical import canonical_bytes, content_id
from flameox.providers.contracts import ProviderAnalysis, ProviderFailure

_MAX_LINE_BYTES = 64 * 1024
_MAX_KERNEL_BYTES = 64 * 1024 * 1024
_MAX_KERNEL_ROWS = 100_000
_MAX_CASES = 100_000
_MAX_OUTPUTS_PER_CASE = 128
_MAX_METRICS_PER_OUTPUT = 32
_MAX_FAILURES_PER_OUTPUT = 32
_MAX_TRITON_EVENTS = 100_000
_MAX_TRITON_CANDIDATES = 64


class KernelEvidenceProvider:
    """Bounded typed readers for semantic validation and Triton autotune evidence."""

    def analyze(
        self,
        capability_id: str,
        paths: Sequence[Path],
        formats: Sequence[str],
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
    ) -> ProviderAnalysis | None:
        if not paths:
            return None
        if all(format_name == "kernel-validation" for format_name in formats):
            documents = [self._kernel_document(path) for path in paths]
            if capability_id == "kernel.compare":
                return self._compare_kernel(documents, arguments, max_rows=max_rows)
            if capability_id == "kernel.validation" and len(documents) == 1:
                return self._summarize_kernel(documents[0], max_rows=max_rows)
        if capability_id == "triton.autotune" and len(paths) == 1 and formats[0] == "triton":
            return self._triton(paths[0], max_rows=max_rows)
        return None

    @staticmethod
    def _kernel_document(path: Path) -> dict[str, Any]:
        try:
            if path.stat().st_size > _MAX_KERNEL_BYTES:
                raise ProviderFailure("LIMIT_EXCEEDED", "Kernel validation document exceeds 64 MiB")
            value = json.loads(path.read_bytes())
        except ProviderFailure:
            raise
        except (OSError, json.JSONDecodeError) as error:
            raise ProviderFailure(
                "DECODE_FAILURE", "Kernel validation document is invalid"
            ) from error
        document = _object(value, "kernel validation document")
        if document.get("schema_version") != "flameox.kernel-validation.v2":
            raise ProviderFailure(
                "UNSUPPORTED_FORMAT",
                "Kernel validation schema must be flameox.kernel-validation.v2",
            )
        if document.get("status") not in {"pass", "fail", "inconclusive", "unsupported"}:
            raise ProviderFailure("DECODE_FAILURE", "Kernel validation status is invalid")
        cases = document.get("cases")
        if not isinstance(cases, list) or len(cases) > _MAX_CASES:
            raise ProviderFailure("LIMIT_EXCEEDED", "Kernel validation case count is invalid")
        return document

    @staticmethod
    def _kernel_rows(document: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        def append(row: dict[str, Any]) -> None:
            if len(rows) >= _MAX_KERNEL_ROWS:
                raise ProviderFailure("LIMIT_EXCEEDED", "Kernel evidence row limit exceeded")
            rows.append(row)

        seen_cases: set[str] = set()
        for case_value in cast(list[object], document["cases"]):
            case = _object(case_value, "kernel validation case")
            case_id = _text(case.get("case_id"), "case_id")
            if case_id in seen_cases:
                raise ProviderFailure("DECODE_FAILURE", "Kernel validation repeats a case_id")
            seen_cases.add(case_id)
            status = _status(case.get("status"), "case status")
            outputs = case.get("outputs")
            if not isinstance(outputs, list) or len(outputs) > _MAX_OUTPUTS_PER_CASE:
                raise ProviderFailure("LIMIT_EXCEEDED", "Kernel validation output count is invalid")
            for output_value in outputs:
                output = _object(output_value, "kernel validation output")
                output_name = _text(output.get("name"), "output name")
                output_status = _status(output.get("status"), "output status")
                shape = output.get("shape")
                if (
                    not isinstance(shape, list)
                    or len(shape) > 16
                    or any(
                        not isinstance(item, int) or isinstance(item, bool) or item < 0
                        for item in shape
                    )
                ):
                    raise ProviderFailure("DECODE_FAILURE", "Kernel output shape is invalid")
                metrics = output.get("metrics", [])
                failures = output.get("representative_failures", [])
                if not isinstance(metrics, list) or len(metrics) > _MAX_METRICS_PER_OUTPUT:
                    raise ProviderFailure("LIMIT_EXCEEDED", "Kernel metric count is invalid")
                if not isinstance(failures, list) or len(failures) > _MAX_FAILURES_PER_OUTPUT:
                    raise ProviderFailure(
                        "LIMIT_EXCEEDED", "Kernel failure witness count is invalid"
                    )
                base = {
                    "case_id": case_id,
                    "case_status": status,
                    "dimensions": _json_object(case.get("dimensions", {}), "dimensions"),
                    "seed": case.get("seed"),
                    "device": case.get("device"),
                    "output": output_name,
                    "dtype": _text(output.get("dtype"), "output dtype"),
                    "shape": shape,
                    "output_status": output_status,
                }
                if not metrics and not failures:
                    append({"evidence_kind": "output", **base})
                for metric_value in metrics:
                    metric = _object(metric_value, "kernel metric")
                    metric_status = _status(metric.get("status"), "metric status")
                    comparator = metric.get("comparator")
                    if comparator not in {"<=", ">=", None}:
                        raise ProviderFailure(
                            "DECODE_FAILURE", "Kernel metric comparator is invalid"
                        )
                    append(
                        {
                            "evidence_kind": "measurement",
                            **base,
                            "metric": _text(metric.get("name"), "metric name"),
                            "value": _metric_value(metric.get("value")),
                            "comparator": comparator,
                            "threshold": _finite_or_none(metric.get("threshold"), "threshold"),
                            "unit": _text(metric.get("unit"), "metric unit"),
                            "metric_status": metric_status,
                            "limitation": metric.get("limitation"),
                        }
                    )
                for failure_index, failure_value in enumerate(failures):
                    failure = _json_object(failure_value, "kernel failure witness")
                    append(
                        {
                            "evidence_kind": "failure_witness",
                            **base,
                            "failure_index": failure_index,
                            "witness": failure,
                        }
                    )
        return rows

    def _summarize_kernel(self, document: Mapping[str, Any], *, max_rows: int) -> ProviderAnalysis:
        rows = self._kernel_rows(document)
        limitations = _string_list(document.get("limitations", []), "limitations", maximum=100)
        return ProviderAnalysis(
            provider_id="kernel-validation",
            provider_version="flameox.kernel-validation.v2",
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "status": document["status"],
                        "coverage_complete": bool(document.get("coverage_complete", False)),
                        "case_count": len(cast(list[object], document["cases"])),
                        "producer": document.get("producer"),
                        "producer_version": document.get("producer_version"),
                    },
                },
                {"type": "table", "rows": rows[:max_rows]},
            ],
            rows_observed=len(rows),
            complete=len(rows) <= max_rows,
            limitations=limitations,
        )

    def _compare_kernel(
        self,
        documents: Sequence[Mapping[str, Any]],
        arguments: Mapping[str, Any],
        *,
        max_rows: int,
    ) -> ProviderAnalysis:
        if len(documents) < 2:
            raise ProviderFailure("INVALID_INPUT", "kernel.compare requires at least 2 inputs")
        baseline_index = int(arguments.get("baseline_index", 0))
        if baseline_index >= len(documents):
            raise ProviderFailure("INVALID_INPUT", "baseline_index does not select an input")
        requested_metric = arguments.get("metric")
        series: list[dict[bytes, float]] = []
        identities: list[dict[bytes, dict[str, Any]]] = []
        statuses: list[dict[bytes, str]] = []
        for document in documents:
            values: dict[bytes, float] = {}
            members: dict[bytes, dict[str, Any]] = {}
            state: dict[bytes, str] = {}
            for row in self._kernel_rows(document):
                if row["evidence_kind"] != "measurement":
                    continue
                identity = {
                    "case_id": str(row["case_id"]),
                    "dimensions": row["dimensions"],
                    "seed": row["seed"],
                    "device": row["device"],
                    "output": str(row["output"]),
                    "dtype": str(row["dtype"]),
                    "shape": row["shape"],
                    "metric": str(row["metric"]),
                    "comparator": row["comparator"],
                    "threshold": row["threshold"],
                    "unit": str(row["unit"]),
                }
                if requested_metric is not None and identity["metric"] != requested_metric:
                    continue
                value = row["value"]
                if isinstance(value, int | float) and not isinstance(value, bool):
                    key = canonical_bytes(identity)
                    values[key] = float(value)
                    members[key] = identity
                    state[key] = str(row["metric_status"])
            series.append(values)
            identities.append(members)
            statuses.append(state)
        common = set(series[baseline_index])
        for values in series:
            common.intersection_update(values)
        all_identities = set().union(*(set(values) for values in series))
        unmatched = all_identities.difference(common)
        rows: list[dict[str, Any]] = []
        for key in sorted(common):
            baseline = series[baseline_index][key]
            for input_index, values in enumerate(series):
                if input_index == baseline_index:
                    continue
                candidate = values[key]
                rows.append(
                    {
                        **identities[baseline_index][key],
                        "baseline_index": baseline_index,
                        "candidate_index": input_index,
                        "baseline_value": baseline,
                        "candidate_value": candidate,
                        "delta": candidate - baseline,
                        "ratio": candidate / baseline if baseline else None,
                        "baseline_status": statuses[baseline_index][key],
                        "candidate_status": statuses[input_index][key],
                    }
                )
        return ProviderAnalysis(
            provider_id="kernel-validation",
            provider_version="flameox.kernel-validation.v2",
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "input_count": len(documents),
                        "compatible_metric_count": len(common),
                        "unmatched_identity_count": len(unmatched),
                    },
                },
                {"type": "table", "rows": rows[:max_rows]},
            ],
            rows_observed=len(rows),
            complete=len(rows) <= max_rows,
            limitations=[
                "Metric deltas compare declared validation outputs; they do not establish "
                "performance.",
                *(
                    ["Measurements absent from one or more inputs were not compared."]
                    if unmatched
                    else []
                ),
            ],
        )

    @staticmethod
    def _triton(path: Path, *, max_rows: int) -> ProviderAnalysis:
        rows: list[dict[str, Any]] = []
        observed = 0
        skipped = 0
        limitations: list[str] = []
        try:
            with path.open("rb") as stream:
                for raw in stream:
                    if len(raw) > _MAX_LINE_BYTES:
                        skipped += 1
                        continue
                    if not raw.strip():
                        continue
                    if observed >= _MAX_TRITON_EVENTS:
                        raise ProviderFailure(
                            "LIMIT_EXCEEDED", "Triton event count exceeds the limit"
                        )
                    try:
                        event = _object(json.loads(raw), "Triton autotune event")
                        unavailable = event.get("listener_unavailable")
                        if unavailable is not None:
                            limitations.append(_text(unavailable, "listener limitation"))
                            continue
                        row = _triton_row(event)
                    except (json.JSONDecodeError, UnicodeDecodeError, ProviderFailure):
                        skipped += 1
                        continue
                    observed += 1
                    if len(rows) < max_rows:
                        rows.append(row)
        except OSError as error:
            raise ProviderFailure("DECODE_FAILURE", "Triton event stream is unreadable") from error
        if skipped:
            limitations.append(f"{skipped} invalid or oversized Triton event(s) were omitted.")
        return ProviderAnalysis(
            provider_id="triton-autotune",
            provider_version="listener-v1",
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "selection_count": observed,
                        "cache_hit_count": sum(bool(row["cache_hit"]) for row in rows),
                    },
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=observed <= len(rows),
            limitations=limitations,
        )


def _triton_row(event: Mapping[str, Any]) -> dict[str, Any]:
    candidates = event.get("candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(candidates) > _MAX_TRITON_CANDIDATES
    ):
        raise ProviderFailure("DECODE_FAILURE", "Triton candidates are invalid")
    normalized_candidates: list[dict[str, Any]] = []
    for candidate_value in candidates:
        candidate = _object(candidate_value, "Triton candidate")
        config = _json_object(candidate.get("config"), "Triton candidate config")
        timings = candidate.get("timings_ms")
        if not isinstance(timings, list) or not timings or len(timings) > 32:
            raise ProviderFailure("DECODE_FAILURE", "Triton candidate timings are invalid")
        finite_timings = [_finite(value, "Triton timing") for value in timings]
        normalized_candidates.append(
            {
                "config_id": content_id(canonical_bytes(config)),
                "config": config,
                "timings_ms": finite_timings,
                "mean_ms": fmean(finite_timings),
            }
        )
    winner = _json_object(event.get("winner"), "Triton winner")
    winner_id = content_id(canonical_bytes(winner))
    if winner_id not in {str(item["config_id"]) for item in normalized_candidates}:
        raise ProviderFailure("DECODE_FAILURE", "Triton winner is absent from candidates")
    cache_hit = event.get("cache_hit")
    if not isinstance(cache_hit, bool):
        raise ProviderFailure("DECODE_FAILURE", "Triton cache_hit is invalid")
    duration = _finite_or_none(event.get("duration_ms"), "Triton duration")
    if cache_hit and duration is not None:
        raise ProviderFailure("DECODE_FAILURE", "Triton cache hit reports a tuning duration")
    return {
        "function_name": _text(event.get("function_name"), "Triton function name"),
        "key_digest": _digest(event.get("key_digest"), "Triton key digest"),
        "cache_hit": cache_hit,
        "duration_ms": duration,
        "winner_config_id": winner_id,
        "candidate_count": len(candidates),
        "candidates": normalized_candidates,
    }


def _object(value: object, subject: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProviderFailure("DECODE_FAILURE", f"{subject} must be an object")
    return cast(dict[str, Any], value)


def _json_object(value: object, subject: str) -> dict[str, Any]:
    result = _object(value, subject)
    try:
        return cast(dict[str, Any], json.loads(json.dumps(result, allow_nan=False)))
    except (TypeError, ValueError) as error:
        raise ProviderFailure(
            "DECODE_FAILURE", f"{subject} contains invalid JSON values"
        ) from error


def _text(value: object, subject: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 500:
        raise ProviderFailure("DECODE_FAILURE", f"{subject} is invalid")
    return value


def _digest(value: object, subject: str) -> str:
    result = _text(value, subject)
    if not result.startswith("sha256:") or len(result) != 71:
        raise ProviderFailure("DECODE_FAILURE", f"{subject} is invalid")
    return result


def _status(value: object, subject: str) -> str:
    if value not in {"pass", "fail", "inconclusive", "unsupported"}:
        raise ProviderFailure("DECODE_FAILURE", f"{subject} is invalid")
    return value


def _finite(value: object, subject: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ProviderFailure("DECODE_FAILURE", f"{subject} must be finite")
    return float(value)


def _finite_or_none(value: object, subject: str) -> float | None:
    return None if value is None else _finite(value, subject)


def _metric_value(value: object) -> float | str | None:
    if value is None:
        return None
    payload = _object(value, "kernel metric value")
    if payload.get("kind") == "finite":
        return _finite(payload.get("value"), "kernel metric value")
    if payload == {"kind": "positive_infinity", "reason": "zero_mse_exact_agreement"}:
        return "positive_infinity"
    raise ProviderFailure("DECODE_FAILURE", "Kernel metric value is invalid")


def _string_list(value: object, subject: str, *, maximum: int) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ProviderFailure("LIMIT_EXCEEDED", f"{subject} exceeds its bound")
    return [_text(item, subject) for item in value]
