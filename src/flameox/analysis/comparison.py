from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
from scipy.stats import bootstrap

from flameox.domain.identity import digest_model, new_id
from flameox.domain.models import (
    Comparison,
    ComparisonDecision,
    ComparisonMetricContract,
    ComparisonValidity,
    ConfidenceInterval,
    MeasurementSeriesSelector,
    MetricPolarity,
    MetricSource,
)
from flameox.domain.scalars import FloatingValue


def _median_log_ratio(baseline: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.median(np.log(candidate) - np.log(baseline)))


def _unpaired_median_log_ratio(baseline: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.median(np.log(candidate)) - np.median(np.log(baseline)))


def _decision(
    *,
    low: float | None,
    high: float | None,
    threshold: float,
    polarity: MetricPolarity,
) -> ComparisonDecision:
    if low is None or high is None or polarity == "neutral":
        return ComparisonDecision.INCONCLUSIVE
    if polarity == "lower_is_better":
        if high <= -threshold:
            return ComparisonDecision.MEANINGFUL_IMPROVEMENT
        if low >= threshold:
            return ComparisonDecision.MEANINGFUL_REGRESSION
    else:
        if low >= threshold:
            return ComparisonDecision.MEANINGFUL_IMPROVEMENT
        if high <= -threshold:
            return ComparisonDecision.MEANINGFUL_REGRESSION
    if low >= -threshold and high <= threshold:
        return ComparisonDecision.NO_MEANINGFUL_DIFFERENCE
    return ComparisonDecision.INCONCLUSIVE


def _bootstrap_log_interval(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    paired: bool,
    confidence_level: float,
    n_resamples: int,
    random_seed: int,
) -> tuple[float, float] | None:
    try:
        with warnings.catch_warnings(record=True) as observed_warnings:
            warnings.simplefilter("always")
            result = bootstrap(
                (baseline, candidate),
                statistic,
                paired=paired,
                vectorized=False,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                method="BCa",
                rng=np.random.default_rng(random_seed),
            )
        if observed_warnings:
            return None
        low = float(result.confidence_interval.low)
        high = float(result.confidence_interval.high)
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            return None
        transformed = (math.expm1(low), math.expm1(high))
    except (ArithmeticError, ValueError):
        return None
    if not all(math.isfinite(value) for value in transformed) or transformed[0] > transformed[1]:
        return None
    return transformed


def _comparison_from_arrays(
    *,
    comparison_id: str | None,
    baseline_run_set_id: str,
    candidate_run_set_id: str,
    baseline: np.ndarray,
    candidate: np.ndarray,
    baseline_attempted_n: int,
    candidate_attempted_n: int,
    paired: bool,
    input_mismatches: Sequence[str],
    metric: str,
    unit: str,
    polarity: MetricPolarity,
    practical_threshold: float,
    confidence_level: float = 0.95,
    random_seed: int = 0,
    n_resamples: int = 9_999,
    experiment_id: str | None = None,
    metric_source: MetricSource = MetricSource.MEASUREMENT,
    measurement_series: MeasurementSeriesSelector | None = None,
) -> Comparison:
    baseline_median = float(np.median(baseline)) if baseline.size else None
    candidate_median = float(np.median(candidate)) if candidate.size else None
    relative_change: float | None = None
    confidence_low: float | None = None
    confidence_high: float | None = None
    mismatches = list(input_mismatches)
    validity = ComparisonValidity.INVALID if mismatches else ComparisonValidity.EXPLORATORY
    method = (
        "scipy.bootstrap.bca.median_paired_log_ratio.v2"
        if paired
        else "scipy.bootstrap.bca.unpaired_median_log_ratio.v1"
    )
    estimand: Literal["median_paired_log_ratio", "difference_in_median_logs"] = (
        "median_paired_log_ratio" if paired else "difference_in_median_logs"
    )
    metric_contract = ComparisonMetricContract(
        source=metric_source,
        metric=metric,
        unit=unit,
        polarity=polarity,
        estimand=estimand,
        series=measurement_series,
    )

    minimum_n = min(baseline.size, candidate.size)
    if not mismatches and baseline.size and candidate.size:
        statistic = _median_log_ratio if paired else _unpaired_median_log_ratio
        estimate = statistic(baseline, candidate)
        try:
            relative_change = math.expm1(estimate)
        except OverflowError:
            relative_change = None
        if relative_change is None or not math.isfinite(relative_change):
            mismatches.append("median log-ratio effect is not finite")
            validity = ComparisonValidity.INVALID
        if paired:
            observed_effects = np.log(candidate) - np.log(baseline)
            exact = bool(np.all(observed_effects == observed_effects[0]))
        else:
            exact = bool(np.all(baseline == baseline[0]) and np.all(candidate == candidate[0]))
        if minimum_n >= 3 and exact and relative_change is not None:
            confidence_low = relative_change
            confidence_high = relative_change
            validity = ComparisonValidity.VALID
            method = (
                "analytic.exact_constant_paired_log_ratio.v1"
                if paired
                else "analytic.exact_constant_unpaired_median_log_ratio.v1"
            )
        elif minimum_n >= 3 and relative_change is not None:
            interval = _bootstrap_log_interval(
                baseline,
                candidate,
                statistic=statistic,
                paired=paired,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                random_seed=random_seed,
            )
            if interval is None:
                mismatches.append("BCa confidence interval is undefined for this sample")
            else:
                confidence_low, confidence_high = interval
                validity = ComparisonValidity.VALID
        elif relative_change is not None:
            mismatches.append("fewer than three independent units are available")

    decision = _decision(
        low=confidence_low,
        high=confidence_high,
        threshold=practical_threshold,
        polarity=polarity,
    )
    return Comparison(
        comparison_id=comparison_id or new_id(),
        experiment_id=experiment_id,
        baseline_run_set_id=baseline_run_set_id,
        candidate_run_set_id=candidate_run_set_id,
        metric=metric,
        unit=unit,
        metric_source=metric_source,
        metric_contract_id=metric_contract.contract_id,
        measurement_series_id=(
            digest_model(measurement_series.model_dump(mode="json"))
            if measurement_series is not None
            else None
        ),
        polarity=polarity,
        estimand=estimand,
        practical_threshold=practical_threshold,
        baseline_value=(
            FloatingValue(value=baseline_median) if baseline_median is not None else None
        ),
        candidate_value=(
            FloatingValue(value=candidate_median) if candidate_median is not None else None
        ),
        absolute_change=(
            FloatingValue(value=candidate_median - baseline_median)
            if candidate_median is not None and baseline_median is not None
            else None
        ),
        relative_change=relative_change,
        effect_size=relative_change,
        confidence_interval=(
            ConfidenceInterval(
                low=confidence_low,
                high=confidence_high,
                level=confidence_level,
            )
            if confidence_low is not None and confidence_high is not None
            else None
        ),
        method=method,
        random_seed=random_seed,
        independent_unit="block" if paired else "worker",
        baseline_attempted_n=baseline_attempted_n,
        baseline_eligible_n=int(baseline.size),
        candidate_attempted_n=candidate_attempted_n,
        candidate_eligible_n=int(candidate.size),
        complete_pair_n=int(baseline.size) if paired else None,
        decision=decision,
        validity=validity,
        mismatches=tuple(mismatches),
    )


def compare_paired_samples(
    *,
    comparison_id: str | None = None,
    baseline_run_set_id: str,
    candidate_run_set_id: str,
    baseline_by_block: Mapping[str, float],
    candidate_by_block: Mapping[str, float],
    metric: str,
    unit: str,
    polarity: MetricPolarity,
    practical_threshold: float,
    confidence_level: float = 0.95,
    random_seed: int = 0,
    n_resamples: int = 9_999,
    experiment_id: str | None = None,
    metric_source: MetricSource = MetricSource.MEASUREMENT,
    measurement_series: MeasurementSeriesSelector | None = None,
) -> Comparison:
    mismatches: list[str] = []
    baseline_blocks = set(baseline_by_block)
    candidate_blocks = set(candidate_by_block)
    if baseline_blocks != candidate_blocks:
        mismatches.append("paired block coverage differs across treatments")
    blocks = sorted(baseline_blocks) if not mismatches else []
    baseline_values = [baseline_by_block[block] for block in blocks]
    candidate_values = [candidate_by_block[block] for block in blocks]
    combined = (*baseline_values, *candidate_values)
    if any(not math.isfinite(value) for value in combined):
        mismatches.append("paired measurement is non-finite")
    if any(value <= 0 for value in combined if math.isfinite(value)):
        mismatches.append("paired measurement is outside the positive log-ratio domain")
    if mismatches:
        baseline_values = []
        candidate_values = []
    return _comparison_from_arrays(
        comparison_id=comparison_id,
        baseline_run_set_id=baseline_run_set_id,
        candidate_run_set_id=candidate_run_set_id,
        baseline=np.asarray(baseline_values, dtype=float),
        candidate=np.asarray(candidate_values, dtype=float),
        baseline_attempted_n=len(baseline_by_block),
        candidate_attempted_n=len(candidate_by_block),
        paired=True,
        input_mismatches=mismatches,
        metric=metric,
        unit=unit,
        polarity=polarity,
        practical_threshold=practical_threshold,
        confidence_level=confidence_level,
        random_seed=random_seed,
        n_resamples=n_resamples,
        experiment_id=experiment_id,
        metric_source=metric_source,
        measurement_series=measurement_series,
    )


def compare_unpaired_samples(
    *,
    comparison_id: str | None = None,
    baseline_run_set_id: str,
    candidate_run_set_id: str,
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    metric: str,
    unit: str,
    polarity: MetricPolarity,
    practical_threshold: float,
    confidence_level: float = 0.95,
    random_seed: int = 0,
    n_resamples: int = 9_999,
    metric_source: MetricSource = MetricSource.MEASUREMENT,
    measurement_series: MeasurementSeriesSelector | None = None,
) -> Comparison:
    mismatches: list[str] = []
    combined = (*baseline_values, *candidate_values)
    if any(not math.isfinite(value) for value in combined):
        mismatches.append("unpaired measurement is non-finite")
    if any(value <= 0 for value in combined if math.isfinite(value)):
        mismatches.append("unpaired measurement is outside the positive log-ratio domain")
    ordered_baseline = sorted(baseline_values) if not mismatches else []
    ordered_candidate = sorted(candidate_values) if not mismatches else []
    return _comparison_from_arrays(
        comparison_id=comparison_id,
        baseline_run_set_id=baseline_run_set_id,
        candidate_run_set_id=candidate_run_set_id,
        baseline=np.asarray(ordered_baseline, dtype=float),
        candidate=np.asarray(ordered_candidate, dtype=float),
        baseline_attempted_n=len(baseline_values),
        candidate_attempted_n=len(candidate_values),
        paired=False,
        input_mismatches=mismatches,
        metric=metric,
        unit=unit,
        polarity=polarity,
        practical_threshold=practical_threshold,
        confidence_level=confidence_level,
        random_seed=random_seed,
        n_resamples=n_resamples,
        metric_source=metric_source,
        measurement_series=measurement_series,
    )
