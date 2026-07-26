from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import numpy as np
from scipy.stats import bootstrap

from flameox.domain.identity import new_id
from flameox.domain.models import (
    Comparison,
    ComparisonDecision,
    ComparisonValidity,
)


def _median_log_ratio(baseline: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.median(np.log(candidate / baseline)))


def _decision(
    *,
    low: float | None,
    high: float | None,
    threshold: float,
    polarity: Literal["lower_is_better", "higher_is_better", "neutral"],
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


def compare_paired_samples(
    *,
    comparison_id: str | None = None,
    baseline_run_set_id: str,
    candidate_run_set_id: str,
    baseline_by_block: Mapping[str, float],
    candidate_by_block: Mapping[str, float],
    metric: str,
    unit: str,
    polarity: Literal["lower_is_better", "higher_is_better", "neutral"],
    practical_threshold: float,
    confidence_level: float = 0.95,
    random_seed: int = 0,
    n_resamples: int = 9_999,
    experiment_id: str | None = None,
) -> Comparison:
    blocks = sorted(set(baseline_by_block) & set(candidate_by_block))
    eligible = [
        block
        for block in blocks
        if math.isfinite(baseline_by_block[block])
        and math.isfinite(candidate_by_block[block])
        and baseline_by_block[block] > 0
        and candidate_by_block[block] > 0
    ]
    baseline = np.asarray([baseline_by_block[block] for block in eligible], dtype=float)
    candidate = np.asarray([candidate_by_block[block] for block in eligible], dtype=float)

    baseline_median = float(np.median(baseline)) if baseline.size else None
    candidate_median = float(np.median(candidate)) if candidate.size else None
    relative_change: float | None = None
    confidence_low: float | None = None
    confidence_high: float | None = None
    validity = ComparisonValidity.EXPLORATORY

    if baseline.size:
        estimate = _median_log_ratio(baseline, candidate)
        relative_change = math.exp(estimate) - 1
        if baseline.size >= 3 and np.allclose(np.log(candidate / baseline), estimate):
            confidence_low = relative_change
            confidence_high = relative_change
            validity = ComparisonValidity.VALID
        elif baseline.size >= 3:
            result = bootstrap(
                (baseline, candidate),
                _median_log_ratio,
                paired=True,
                vectorized=False,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                method="BCa",
                rng=np.random.default_rng(random_seed),
            )
            confidence_low = math.exp(float(result.confidence_interval.low)) - 1
            confidence_high = math.exp(float(result.confidence_interval.high)) - 1
            validity = ComparisonValidity.VALID

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
        polarity=polarity,
        estimand="median_paired_log_ratio",
        practical_threshold=practical_threshold,
        baseline_value_float=baseline_median,
        candidate_value_float=candidate_median,
        absolute_change_float=(
            candidate_median - baseline_median
            if candidate_median is not None and baseline_median is not None
            else None
        ),
        relative_change=relative_change,
        effect_size=relative_change,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        confidence_level=confidence_level if confidence_low is not None else None,
        method="scipy.bootstrap.bca.median_paired_log_ratio.v1",
        random_seed=random_seed,
        independent_unit="block",
        paired=True,
        baseline_attempted_n=len(baseline_by_block),
        baseline_eligible_n=len(eligible),
        candidate_attempted_n=len(candidate_by_block),
        candidate_eligible_n=len(eligible),
        complete_pair_n=len(eligible),
        decision=decision,
        validity=validity,
        mismatches=(),
    )
