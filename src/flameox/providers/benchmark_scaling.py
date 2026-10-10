from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from flameox.benchmark_samples import benchmark_series_identity
from flameox.canonical import canonical_bytes
from flameox.providers.contracts import ProviderAnalysis


def _positive_measurement_mean(
    raw_input: Any, row: Mapping[str, Any]
) -> tuple[float, float, int] | None:
    sample_mean = row.get("positive_sample_mean")
    sample_sum = row.get("positive_sample_sum", row.get("sample_sum"))
    sample_count = row.get("positive_sample_count", row.get("sample_count"))
    if sample_mean is not None:
        value = sample_mean
        count = sample_count
    else:
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
        return None
    try:
        input_value = float(raw_input)
        numeric_value = float(value)
    except (ValueError, OverflowError):
        return None
    measurement_mean = numeric_value if sample_mean is not None else numeric_value / count
    if (
        not math.isfinite(input_value)
        or not math.isfinite(measurement_mean)
        or input_value <= 0
        or measurement_mean <= 0
    ):
        return None
    return input_value, measurement_mean, count


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
    series: dict[bytes, dict[float, tuple[float, int]]] = defaultdict(
        lambda: defaultdict(lambda: (0.0, 0))
    )
    identities: dict[bytes, dict[str, Any]] = {}
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
        identity = benchmark_series_identity(row, input_dimension=input_dimension)
        key = canonical_bytes(identity)
        identities[key] = identity
        point = _positive_measurement_mean(raw_input, row)
        if point is None:
            omitted_measurements += 1
            continue
        input_value, measurement_mean, count = point
        prior_mean, prior_count = series[key][input_value]
        combined_count = prior_count + count
        if prior_count == 0:
            combined_mean = measurement_mean
        else:
            # A positive weighted running mean stays between its finite inputs;
            # summing finite measurements first can overflow before averaging.
            combined_mean = prior_mean + (measurement_mean - prior_mean) * (count / combined_count)
        series[key][input_value] = (combined_mean, combined_count)

    output: list[dict[str, Any]] = []
    estimated = 0
    for key in sorted(identities):
        identity = identities[key]
        points = sorted((input_value, mean) for input_value, (mean, _count) in series[key].items())
        if len(points) < 2:
            output.append(
                {
                    **identity,
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
        input_variance = math.fsum((value - mean_input) ** 2 for value in log_inputs)
        separation_floor = math.ulp(mean_input) ** 2 * len(log_inputs)
        if not math.isfinite(input_variance) or input_variance <= separation_floor:
            output.append(
                {
                    **identity,
                    "status": "inconclusive",
                    "input_dimension": input_dimension,
                    "point_count": len(points),
                    "exponent": None,
                    "coefficient": None,
                    "r_squared": None,
                    "input_min": points[0][0],
                    "input_max": points[-1][0],
                    "reason": "insufficient log-space input separation",
                }
            )
            continue
        exponent = (
            math.fsum(
                (input_value - mean_input) * (measurement - mean_measurement)
                for input_value, measurement in zip(log_inputs, log_measurements, strict=True)
            )
            / input_variance
        )
        intercept = mean_measurement - exponent * mean_input
        residual_sum = math.fsum(
            (measurement - (intercept + exponent * input_value)) ** 2
            for input_value, measurement in zip(log_inputs, log_measurements, strict=True)
        )
        total_sum = math.fsum(
            (measurement - mean_measurement) ** 2 for measurement in log_measurements
        )
        r_squared = 1.0 if total_sum == 0 else max(0.0, 1.0 - residual_sum / total_sum)
        try:
            coefficient = math.exp(intercept)
        except OverflowError:
            coefficient = None
        if not all(math.isfinite(value) for value in (exponent, intercept, r_squared)) or (
            coefficient is not None and (not math.isfinite(coefficient) or coefficient <= 0)
        ):
            coefficient = None
        if coefficient is None:
            output.append(
                {
                    **identity,
                    "status": "inconclusive",
                    "input_dimension": input_dimension,
                    "point_count": len(points),
                    "exponent": None,
                    "coefficient": None,
                    "r_squared": None,
                    "input_min": points[0][0],
                    "input_max": points[-1][0],
                    "reason": "power-law coefficients exceed finite numeric range",
                }
            )
            continue
        output.append(
            {
                **identity,
                "status": "estimated",
                "input_dimension": input_dimension,
                "point_count": len(points),
                "exponent": exponent,
                "coefficient": coefficient,
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
                    "series_count": len(identities),
                    "estimated_series_count": estimated,
                    "inconclusive_series_count": len(identities) - estimated,
                },
            },
            {"type": "table", "rows": output[:max_rows]},
        ],
        rows_observed=len(output),
        complete=len(output) <= max_rows,
        limitations=limitations,
    )
