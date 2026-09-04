from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from flameox.providers.contracts import ProviderAnalysis


def scaling_projection(
    rows: Sequence[Mapping[str, Any]],
    arguments: Mapping[str, Any],
    *,
    provider_id: str,
    provider_version: str,
    max_rows: int,
) -> ProviderAnalysis:
    """Fit bounded log-log scaling estimates for declared benchmark dimensions."""

    input_dimension = str(arguments["input_dimension"])
    requested_metric = arguments.get("metric")
    series: dict[tuple[str, str, str], dict[float, tuple[float, int]]] = defaultdict(
        lambda: defaultdict(lambda: (0.0, 0))
    )
    identity_dimensions: dict[tuple[str, str, str], dict[str, Any]] = {}
    omitted_measurements = 0
    for row in rows:
        if row.get("is_warmup") is True:
            continue
        benchmark = row.get("benchmark")
        unit = row.get("unit")
        if not isinstance(benchmark, str) or not isinstance(unit, str):
            continue
        if requested_metric is not None and benchmark != requested_metric:
            continue
        dimensions = row.get("dimensions")
        raw_input = dimensions.get(input_dimension) if isinstance(dimensions, Mapping) else None
        non_axis_dimensions = (
            {str(key): value for key, value in dimensions.items() if key != input_dimension}
            if isinstance(dimensions, Mapping)
            else {}
        )
        dimension_identity = json.dumps(
            non_axis_dimensions, sort_keys=True, separators=(",", ":"), default=str
        )
        identity = (benchmark, unit, dimension_identity)
        identity_dimensions[identity] = non_axis_dimensions
        sample_sum = row.get("positive_sample_sum", row.get("sample_sum"))
        sample_count = row.get("positive_sample_count", row.get("sample_count"))
        value = row.get("value_int") if sample_sum is None else sample_sum
        if value is None:
            value = row.get("value_float")
        count = sample_count if sample_sum is not None else 1
        if (
            not isinstance(raw_input, str | int | float)
            or isinstance(raw_input, bool)
            or not isinstance(value, str | int | float)
            or isinstance(value, bool)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
        ):
            omitted_measurements += 1
            continue
        try:
            input_value = float(raw_input)
            measurement_total = float(value)
        except ValueError:
            omitted_measurements += 1
            continue
        if (
            not math.isfinite(input_value)
            or not math.isfinite(measurement_total)
            or input_value <= 0
            or measurement_total / count <= 0
        ):
            omitted_measurements += 1
            continue
        prior_total, prior_count = series[identity][input_value]
        series[identity][input_value] = (
            prior_total + measurement_total,
            prior_count + count,
        )

    output: list[dict[str, Any]] = []
    estimated = 0
    for identity in sorted(identity_dimensions):
        benchmark, unit, _dimension_identity = identity
        dimensions = identity_dimensions[identity]
        points = sorted(
            (input_value, total / count) for input_value, (total, count) in series[identity].items()
        )
        if len(points) < 2:
            output.append(
                {
                    "benchmark": benchmark,
                    "unit": unit,
                    "dimensions": dimensions,
                    "status": "inconclusive",
                    "input_dimension": input_dimension,
                    "point_count": len(points),
                    "exponent": None,
                    "coefficient": None,
                    "r_squared": None,
                    "input_min": points[0][0] if points else None,
                    "input_max": points[-1][0] if points else None,
                    "reason": "fewer than two distinct positive input values",
                }
            )
            continue
        log_inputs = [math.log(point[0]) for point in points]
        log_measurements = [math.log(point[1]) for point in points]
        mean_input = fmean(log_inputs)
        mean_measurement = fmean(log_measurements)
        input_variance = sum((value - mean_input) ** 2 for value in log_inputs)
        exponent = (
            sum(
                (input_value - mean_input) * (measurement - mean_measurement)
                for input_value, measurement in zip(log_inputs, log_measurements, strict=True)
            )
            / input_variance
        )
        intercept = mean_measurement - exponent * mean_input
        residual_sum = sum(
            (measurement - (intercept + exponent * input_value)) ** 2
            for input_value, measurement in zip(log_inputs, log_measurements, strict=True)
        )
        total_sum = sum((measurement - mean_measurement) ** 2 for measurement in log_measurements)
        r_squared = 1.0 if total_sum == 0 else max(0.0, 1.0 - residual_sum / total_sum)
        output.append(
            {
                "benchmark": benchmark,
                "unit": unit,
                "dimensions": dimensions,
                "status": "estimated",
                "input_dimension": input_dimension,
                "point_count": len(points),
                "exponent": exponent,
                "coefficient": math.exp(intercept),
                "r_squared": r_squared,
                "input_min": points[0][0],
                "input_max": points[-1][0],
                "reason": None,
            }
        )
        estimated += 1

    limitations = [
        "The log-log power-law fit describes observed benchmark means and does not prove "
        "asymptotic complexity or causality."
    ]
    if omitted_measurements:
        limitations.append(
            f"{omitted_measurements} measurement(s) lacked positive numeric "
            f"{input_dimension!r} and value pairs."
        )
    if estimated == 0:
        limitations.append(
            f"No benchmark series had two distinct positive numeric {input_dimension!r} values."
        )
    return ProviderAnalysis(
        provider_id=provider_id,
        provider_version=provider_version,
        blocks=[
            {
                "type": "metrics",
                "values": {
                    "scaling_status": "estimated" if estimated else "inconclusive",
                    "input_dimension": input_dimension,
                    "series_count": len(identity_dimensions),
                    "estimated_series_count": estimated,
                    "inconclusive_series_count": len(identity_dimensions) - estimated,
                },
            },
            {"type": "table", "rows": output[:max_rows]},
        ],
        rows_observed=len(output),
        complete=len(output) <= max_rows,
        limitations=limitations,
    )
