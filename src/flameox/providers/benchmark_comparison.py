from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from flameox.providers.contracts import ProviderAnalysis, ProviderFailure

type AggregateSeries = dict[bytes, tuple[dict[str, Any], float, int]]


def compare_series(
    series: Sequence[AggregateSeries],
    arguments: Mapping[str, Any],
    *,
    max_rows: int,
    provider_id: str,
    provider_version: str,
) -> ProviderAnalysis:
    """Compare pooled means while preserving each native reader's series identities."""
    if len(series) < 2:
        raise ProviderFailure("INVALID_INPUT", "benchmark.compare requires at least 2 inputs")
    baseline_index = int(arguments.get("baseline_index", 0))
    if baseline_index >= len(series):
        raise ProviderFailure("INVALID_INPUT", "baseline_index does not select an input")
    requested_metric = arguments.get("metric")
    selected_keys = [
        {
            key
            for key, (identity, _total, count) in values.items()
            if count > 0 and (requested_metric is None or identity["benchmark"] == requested_metric)
        }
        for values in series
    ]
    common = selected_keys[baseline_index].copy()
    for keys in selected_keys:
        common.intersection_update(keys)
    unmatched = set().union(*selected_keys).difference(common)
    rows: list[dict[str, Any]] = []
    for key in sorted(common):
        identity, baseline_total, baseline_count = series[baseline_index][key]
        baseline_mean = baseline_total / baseline_count
        for input_index, values in enumerate(series):
            if input_index == baseline_index:
                continue
            _identity, candidate_total, candidate_count = values[key]
            candidate_mean = candidate_total / candidate_count
            ratio = candidate_mean / baseline_mean if baseline_mean else None
            if not math.isfinite(baseline_mean) or not math.isfinite(candidate_mean):
                raise ProviderFailure(
                    "LIMIT_EXCEEDED", "Benchmark comparison mean exceeds finite numeric range."
                )
            if ratio is not None and not math.isfinite(ratio):
                raise ProviderFailure(
                    "LIMIT_EXCEEDED", "Benchmark comparison ratio exceeds finite numeric range."
                )
            if len(rows) < max_rows:
                rows.append(
                    {
                        **identity,
                        "baseline_index": baseline_index,
                        "candidate_index": input_index,
                        "baseline_mean": baseline_mean,
                        "candidate_mean": candidate_mean,
                        "ratio": ratio,
                    }
                )
    observed = len(common) * (len(series) - 1)
    return ProviderAnalysis(
        provider_id=provider_id,
        provider_version=provider_version,
        blocks=[
            {
                "type": "metrics",
                "values": {
                    "input_count": len(series),
                    "compatible_metric_count": len(common),
                    "unmatched_identity_count": len(unmatched),
                },
            },
            {"type": "table", "rows": rows},
        ],
        rows_observed=observed,
        complete=observed <= max_rows,
        limitations=[
            "Ratios summarize observed sample means and do not establish causal improvement.",
            *(["Series absent from one or more inputs were not compared."] if unmatched else []),
        ],
    )
