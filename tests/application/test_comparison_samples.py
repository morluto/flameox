from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application import (
    ComparisonService,
    FreezeRunIdsRequest,
    MeasurementCompareRunSetsRequest,
    RunSetService,
    RuntimeResourceCompareRunSetsRequest,
)
from flameox.catalog import Catalog
from flameox.domain import ComparisonValidity, DomainError, ErrorCode, MetricPolarity
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace
from tests.support.comparisons import imported_benchmark, measurement_row


def _comparison_request(
    workspace: Workspace,
    baseline_id: str,
    candidate_id: str,
) -> MeasurementCompareRunSetsRequest:
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))
    return MeasurementCompareRunSetsRequest(
        baseline_run_set_id=baseline.run_set_id,
        candidate_run_set_id=candidate.run_set_id,
        metric="pyperf.scan",
        unit="ns",
        polarity=MetricPolarity.LOWER_IS_BETTER,
        practical_threshold=0.05,
    )


def _benchmark_pair(workspace: Workspace, root: Path) -> tuple[str, str]:
    Catalog(workspace).rebuild()
    baseline_id = imported_benchmark(
        workspace,
        root / "baseline.json",
        (0.010, 0.011, 0.012),
    )
    candidate_id = imported_benchmark(
        workspace,
        root / "candidate.json",
        (0.005, 0.0055, 0.006),
    )
    return baseline_id, candidate_id


def test_comparison_rejects_duplicate_measurement_keys_without_block_identity(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_id, candidate_id = _benchmark_pair(workspace, tmp_path)
    duplicate = measurement_row(
        baseline_id,
        20_000_000,
        measurement_id="duplicate-baseline-measurement",
        worker_id="scan:0",
        worker_run_index=0,
        value_index=0,
    )
    GenerationPublisher(workspace).publish_rows(
        {"measurements": [duplicate]},
        publisher="comparison-sample-fixture",
        publisher_version="1",
        input_run_ids=(baseline_id,),
    )
    request = _comparison_request(workspace, baseline_id, candidate_id)

    with pytest.raises(DomainError) as error:
        ComparisonService(workspace).compare(request)

    assert error.value.code is ErrorCode.COMPARISON_INVALID
    assert error.value.details == {"run_id": baseline_id, "key": "scan:0:0:0"}


def test_comparison_pairs_distinct_measurement_keys_from_published_evidence(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_id, candidate_id = _benchmark_pair(workspace, tmp_path)
    GenerationPublisher(workspace).publish_rows(
        {
            "measurements": [
                measurement_row(baseline_id, 13_000_000),
                measurement_row(candidate_id, 6_500_000),
            ]
        },
        publisher="comparison-sample-fixture",
        publisher_version="1",
        input_run_ids=(baseline_id, candidate_id),
    )
    request = _comparison_request(workspace, baseline_id, candidate_id)

    result = ComparisonService(workspace).compare(request)

    assert result.comparison.baseline_eligible_n == 4
    assert result.comparison.candidate_eligible_n == 4
    assert result.comparison.complete_pair_n == 4
    assert result.comparison.relative_change is not None
    assert result.comparison.relative_change < 0


def test_comparison_pairs_runtime_resource_catalog_metric(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_id, candidate_id = _benchmark_pair(workspace, tmp_path)
    GenerationPublisher(workspace).publish_rows(
        {
            "runtime_resource_summaries": [
                {
                    "run_id": baseline_id,
                    "sampling_interval_ms": 100,
                    "minimum_free_bytes": 1000,
                    "staging_growth_bytes": 20,
                    "peak_rss_bytes": 200,
                    "peak_rss_backend": "psutil_recursive_polling",
                    "policy_termination": None,
                    "unavailable_metrics": [],
                },
                {
                    "run_id": candidate_id,
                    "sampling_interval_ms": 100,
                    "minimum_free_bytes": 900,
                    "staging_growth_bytes": 10,
                    "peak_rss_bytes": 100,
                    "peak_rss_backend": "psutil_recursive_polling",
                    "policy_termination": None,
                    "unavailable_metrics": [],
                },
            ]
        },
        publisher="comparison-resource-fixture",
        publisher_version="1",
        input_run_ids=(baseline_id, candidate_id),
    )
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))

    result = ComparisonService(workspace).compare(
        RuntimeResourceCompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate.run_set_id,
            metric="runtime_resource.peak_rss_bytes",
            metric_source="runtime_resource",
            unit="bytes",
            polarity=MetricPolarity.LOWER_IS_BETTER,
            practical_threshold=0.05,
        )
    )

    assert result.comparison.baseline_value is not None
    assert result.comparison.baseline_value.kind == "floating"
    assert result.comparison.baseline_value.value == 200
    assert result.comparison.candidate_value is not None
    assert result.comparison.candidate_value.kind == "floating"
    assert result.comparison.candidate_value.value == 100
    assert result.comparison.complete_pair_n == 1


def test_runtime_resource_comparison_rejects_wrong_unit() -> None:
    with pytest.raises(ValueError, match="bytes"):
        RuntimeResourceCompareRunSetsRequest.model_validate(
            {
                "baseline_run_set_id": "baseline",
                "candidate_run_set_id": "candidate",
                "metric": "runtime_resource.peak_rss_bytes",
                "metric_source": "runtime_resource",
                "unit": "ns",
                "polarity": "lower_is_better",
                "practical_threshold": 0,
            }
        )


def test_runtime_resource_zero_is_invalid_for_log_ratio_estimation(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_id, candidate_id = _benchmark_pair(workspace, tmp_path)
    GenerationPublisher(workspace).publish_rows(
        {
            "runtime_resource_summaries": [
                {
                    "run_id": baseline_id,
                    "sampling_interval_ms": 100,
                    "minimum_free_bytes": 1,
                    "staging_growth_bytes": 10,
                    "peak_rss_bytes": 100,
                    "peak_rss_backend": "psutil_recursive_polling",
                    "policy_termination": None,
                    "unavailable_metrics": [],
                },
                {
                    "run_id": candidate_id,
                    "sampling_interval_ms": 100,
                    "minimum_free_bytes": 1,
                    "staging_growth_bytes": 0,
                    "peak_rss_bytes": 100,
                    "peak_rss_backend": "psutil_recursive_polling",
                    "policy_termination": None,
                    "unavailable_metrics": [],
                },
            ]
        },
        publisher="comparison-resource-fixture",
        publisher_version="1",
        input_run_ids=(baseline_id, candidate_id),
    )
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))

    result = ComparisonService(workspace).compare(
        RuntimeResourceCompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate.run_set_id,
            metric="runtime_resource.staging_growth_bytes",
            metric_source="runtime_resource",
            unit="bytes",
            polarity=MetricPolarity.LOWER_IS_BETTER,
            practical_threshold=0,
        )
    )

    assert result.comparison.validity is ComparisonValidity.INVALID
    assert result.comparison.candidate_eligible_n == 0
    assert any("zero-valued" in reason for reason in result.comparison.mismatches)
