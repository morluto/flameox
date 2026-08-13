from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import bootstrap


@dataclass(frozen=True, slots=True)
class PairedDifferenceEstimate:
    estimate: float | None
    confidence_low: float | None
    confidence_high: float | None
    method: str
    limitation: str | None = None


def paired_median_difference(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence_level: float,
    random_seed: int,
    n_resamples: int = 9_999,
) -> PairedDifferenceEstimate:
    """Estimate a paired median difference without imposing a positive value domain."""

    if len(baseline) != len(candidate):
        return PairedDifferenceEstimate(
            estimate=None,
            confidence_low=None,
            confidence_high=None,
            method="unavailable.pairing_mismatch.v1",
            limitation="baseline and candidate paired populations have different sizes",
        )
    if not baseline:
        return PairedDifferenceEstimate(
            estimate=None,
            confidence_low=None,
            confidence_high=None,
            method="unavailable.empty_population.v1",
            limitation="no eligible paired observations are available",
        )
    baseline_values = np.asarray(baseline, dtype=float)
    candidate_values = np.asarray(candidate, dtype=float)
    if not np.all(np.isfinite(baseline_values)) or not np.all(np.isfinite(candidate_values)):
        return PairedDifferenceEstimate(
            estimate=None,
            confidence_low=None,
            confidence_high=None,
            method="unavailable.nonfinite_population.v1",
            limitation="paired finite-difference inference cannot include non-finite values",
        )
    differences = candidate_values - baseline_values
    estimate = float(np.median(differences))
    if len(differences) < 3:
        return PairedDifferenceEstimate(
            estimate=estimate,
            confidence_low=None,
            confidence_high=None,
            method="descriptive.median_paired_difference.v1",
            limitation="fewer than three independent blocks are available",
        )
    if bool(np.all(differences == differences[0])):
        exact = float(differences[0])
        return PairedDifferenceEstimate(
            estimate=exact,
            confidence_low=exact,
            confidence_high=exact,
            method="analytic.exact_constant_paired_difference.v1",
        )

    def statistic(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.median(second - first))

    had_warning = False
    try:
        with warnings.catch_warnings(record=True) as observed_warnings:
            warnings.simplefilter("always")
            result = bootstrap(
                (baseline_values, candidate_values),
                statistic,
                paired=True,
                vectorized=False,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                method="BCa",
                rng=np.random.default_rng(random_seed),
            )
        had_warning = bool(observed_warnings)
        low = float(result.confidence_interval.low)
        high = float(result.confidence_interval.high)
    except (ArithmeticError, ValueError):
        low = math.nan
        high = math.nan
        had_warning = True
    if had_warning or not math.isfinite(low) or not math.isfinite(high) or low > high:
        return PairedDifferenceEstimate(
            estimate=estimate,
            confidence_low=None,
            confidence_high=None,
            method="descriptive.median_paired_difference.v1",
            limitation="BCa confidence interval is undefined for this paired population",
        )
    return PairedDifferenceEstimate(
        estimate=estimate,
        confidence_low=low,
        confidence_high=high,
        method="scipy.bootstrap.bca.median_paired_difference.v1",
    )
