from __future__ import annotations

import pytest
from pydantic import ValidationError

from flameox.analysis import compare_paired_samples
from flameox.domain import Comparison, ComparisonDecision, ComparisonValidity, MetricPolarity

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
    assert result.method == "scipy.bootstrap.bca.median_paired_log_ratio.v1"
    assert result.independent_unit == "block"
    assert result.paired
    assert result.confidence_level == 0.95


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

    assert result.complete_pair_n == 1
    assert result.validity is ComparisonValidity.EXPLORATORY
    assert result.decision is ComparisonDecision.INCONCLUSIVE
    assert result.confidence_low is None

    with pytest.raises(ValidationError, match="complete pairs"):
        Comparison.model_validate({**result.model_dump(mode="python"), "complete_pair_n": 2})
    with pytest.raises(ValidationError, match="confidence bounds and level"):
        Comparison.model_validate({**result.model_dump(mode="python"), "confidence_low": -0.1})

    with pytest.raises(ValidationError, match="paired projection contradicts"):
        Comparison.model_validate({**result.model_dump(mode="python"), "paired": False})
