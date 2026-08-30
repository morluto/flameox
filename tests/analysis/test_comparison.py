from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from flameox.analysis import compare_paired_samples, compare_unpaired_samples
from flameox.domain import (
    Comparison,
    ComparisonDecision,
    ComparisonValidity,
    MetricPolarity,
    digest_model,
)

pytestmark = pytest.mark.unit

BASELINE_SET = "sha256:" + ("a" * 64)
CANDIDATE_SET = "sha256:" + ("b" * 64)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (
            {"1": 50.0, "2": 55.0, "3": 60.0},
            ComparisonDecision.MEANINGFUL_IMPROVEMENT,
        ),
        (
            {"1": 102.0, "2": 112.2, "3": 122.4},
            ComparisonDecision.NO_MEANINGFUL_DIFFERENCE,
        ),
        (
            {"1": 150.0, "2": 165.0, "3": 180.0},
            ComparisonDecision.MEANINGFUL_REGRESSION,
        ),
    ],
)
def test_paired_comparison_uses_practical_interval(
    candidate: dict[str, float],
    expected: ComparisonDecision,
) -> None:
    result = compare_paired_samples(
        baseline_run_set_id=BASELINE_SET,
        candidate_run_set_id=CANDIDATE_SET,
        baseline_by_block={"1": 100.0, "2": 110.0, "3": 120.0},
        candidate_by_block=candidate,
        metric="benchmark.wall_time",
        unit="ns",
        polarity=MetricPolarity.LOWER_IS_BETTER,
        practical_threshold=0.05,
    )

    assert result.decision is expected
    assert result.validity is ComparisonValidity.VALID
    assert result.complete_pair_n == 3
    assert result.estimand == "median_paired_log_ratio"
    assert result.practical_threshold == 0.05
    assert result.method == "scipy.bootstrap.bca.median_paired_log_ratio.v2"
    assert result.independent_unit == "block"
    assert result.complete_pair_n is not None
    assert result.confidence_interval is not None
    assert result.confidence_interval.level == 0.95


def test_incomplete_experiment_is_not_presented_as_valid() -> None:
    result = compare_paired_samples(
        baseline_run_set_id=BASELINE_SET,
        candidate_run_set_id=CANDIDATE_SET,
        baseline_by_block={"1": 100.0, "2": 110.0},
        candidate_by_block={"2": 100.0, "3": 90.0},
        metric="benchmark.wall_time",
        unit="ns",
        polarity=MetricPolarity.LOWER_IS_BETTER,
        practical_threshold=0.05,
    )

    assert result.complete_pair_n == 0
    assert result.validity is ComparisonValidity.INVALID
    assert result.decision is ComparisonDecision.INCONCLUSIVE
    assert result.confidence_interval is None
    assert "paired block coverage differs across treatments" in result.mismatches

    with pytest.raises(ValidationError, match="complete pairs"):
        result.validated_copy(update={"complete_pair_n": 2})
    with pytest.raises(ValidationError, match="valid number"):
        result.validated_copy(
            update={"confidence_interval": {"low": -0.1, "high": None, "level": None}}
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Comparison.model_validate({**result.model_dump(mode="python"), "paired": False})


def test_exact_constant_effect_uses_versioned_analytic_interval() -> None:
    result = compare_paired_samples(
        baseline_run_set_id=BASELINE_SET,
        candidate_run_set_id=CANDIDATE_SET,
        baseline_by_block={str(index): 1.0 for index in range(4)},
        candidate_by_block={str(index): 0.5 for index in range(4)},
        metric="benchmark.wall_time",
        unit="ns",
        polarity=MetricPolarity.LOWER_IS_BETTER,
        practical_threshold=0.05,
    )

    assert result.validity is ComparisonValidity.VALID
    assert result.method == "analytic.exact_constant_paired_log_ratio.v1"
    assert result.confidence_interval is not None
    assert result.confidence_interval.low == result.confidence_interval.high == -0.5


def test_degenerate_bca_is_truthful_and_canonical_json_safe() -> None:
    result = compare_paired_samples(
        baseline_run_set_id=BASELINE_SET,
        candidate_run_set_id=CANDIDATE_SET,
        baseline_by_block={str(index): 1.0 for index in range(4)},
        candidate_by_block={"0": 0.5, "1": 0.5, "2": 0.5, "3": 1.0},
        metric="benchmark.wall_time",
        unit="ns",
        polarity=MetricPolarity.LOWER_IS_BETTER,
        practical_threshold=0.05,
    )

    assert result.validity is ComparisonValidity.EXPLORATORY
    assert result.decision is ComparisonDecision.INCONCLUSIVE
    assert result.confidence_interval is None
    assert result.mismatches == ("BCa confidence interval is undefined for this sample",)
    assert digest_model(result).startswith("sha256:")


def test_close_effects_do_not_take_the_exact_interval_shortcut() -> None:
    center = math.log(0.95)
    effects = (center - 4e-7, center, center + 4e-7)
    result = compare_paired_samples(
        baseline_run_set_id=BASELINE_SET,
        candidate_run_set_id=CANDIDATE_SET,
        baseline_by_block={str(index): 1.0 for index in range(3)},
        candidate_by_block={str(index): math.exp(effect) for index, effect in enumerate(effects)},
        metric="benchmark.wall_time",
        unit="ns",
        polarity=MetricPolarity.LOWER_IS_BETTER,
        practical_threshold=0.05,
    )

    assert result.method == "scipy.bootstrap.bca.median_paired_log_ratio.v2"
    assert result.decision is ComparisonDecision.INCONCLUSIVE
    assert result.confidence_interval is not None
    assert result.confidence_interval.low < result.confidence_interval.high


@pytest.mark.parametrize("invalid_value", (0.0, -1.0, float("nan"), float("inf")))
def test_log_ratio_rejects_every_out_of_domain_observation(invalid_value: float) -> None:
    result = compare_paired_samples(
        baseline_run_set_id=BASELINE_SET,
        candidate_run_set_id=CANDIDATE_SET,
        baseline_by_block={"0": 1.0, "1": 1.0, "2": 1.0},
        candidate_by_block={"0": 0.5, "1": invalid_value, "2": 0.5},
        metric="benchmark.wall_time",
        unit="ns",
        polarity=MetricPolarity.LOWER_IS_BETTER,
        practical_threshold=0.05,
    )

    assert result.validity is ComparisonValidity.INVALID
    assert result.decision is ComparisonDecision.INCONCLUSIVE
    assert result.complete_pair_n == 0
    assert digest_model(result).startswith("sha256:")


def test_unpaired_comparison_is_permutation_invariant_and_reports_worker_unit() -> None:
    baseline = (1.0, 10.0, 100.0, 1_000.0)
    candidate = (2.0, 20.0, 200.0, 2_000.0)
    forward = compare_unpaired_samples(
        baseline_run_set_id=BASELINE_SET,
        candidate_run_set_id=CANDIDATE_SET,
        baseline_values=baseline,
        candidate_values=candidate,
        metric="benchmark.wall_time",
        unit="ns",
        polarity=MetricPolarity.LOWER_IS_BETTER,
        practical_threshold=0.05,
        comparison_id="same",
    )
    reordered = compare_unpaired_samples(
        baseline_run_set_id=BASELINE_SET,
        candidate_run_set_id=CANDIDATE_SET,
        baseline_values=tuple(reversed(baseline)),
        candidate_values=(200.0, 2.0, 2_000.0, 20.0),
        metric="benchmark.wall_time",
        unit="ns",
        polarity=MetricPolarity.LOWER_IS_BETTER,
        practical_threshold=0.05,
        comparison_id="same",
    )

    assert reordered == forward
    assert forward.complete_pair_n is None
    assert forward.independent_unit == "worker"
    assert forward.estimand == "difference_in_median_logs"


def test_comparison_model_rejects_non_finite_effect_fields() -> None:
    result = compare_paired_samples(
        baseline_run_set_id=BASELINE_SET,
        candidate_run_set_id=CANDIDATE_SET,
        baseline_by_block={"0": 1.0, "1": 1.0, "2": 1.0},
        candidate_by_block={"0": 0.5, "1": 0.5, "2": 0.5},
        metric="benchmark.wall_time",
        unit="ns",
        polarity=MetricPolarity.LOWER_IS_BETTER,
        practical_threshold=0.05,
    )

    with pytest.raises(ValidationError):
        result.validated_copy(update={"relative_change": float("nan")})
