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
from flameox.domain import (
    ComparisonValidity,
    MetricPolarity,
    MetricSource,
)
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace
from tests.support.comparisons import (
    imported_benchmark,
    imported_benchmark_workers,
    measurement_row,
)

pytestmark = pytest.mark.integration


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


def test_comparison_rejects_an_unbound_row_mixed_into_an_artifact_series(
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

    result = ComparisonService(workspace).compare(request)

    assert result.comparison.validity is ComparisonValidity.INVALID
    assert result.comparison.baseline_missing_n == 1
    assert any("matching measurement series" in reason for reason in result.comparison.mismatches)


def test_unpaired_comparison_aggregates_inner_values_at_the_worker_boundary(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    baseline_id = imported_benchmark_workers(
        workspace,
        tmp_path / "baseline-workers.json",
        ((0.010, 0.011, 0.012), (0.012, 0.013, 0.014)),
    )
    candidate_id = imported_benchmark_workers(
        workspace,
        tmp_path / "candidate-workers.json",
        ((0.005, 0.0055, 0.006), (0.006, 0.0065, 0.007)),
    )
    request = _comparison_request(workspace, baseline_id, candidate_id)

    result = ComparisonService(workspace).compare(request)

    assert result.comparison.baseline_eligible_n == 2
    assert result.comparison.candidate_eligible_n == 2
    assert result.comparison.complete_pair_n is None
    assert result.comparison.independent_unit == "worker"
    assert result.comparison.relative_change is not None
    assert result.comparison.relative_change < 0


def test_comparison_refuses_to_coerce_exact_uint64_measurements_to_float(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_id, candidate_id = _benchmark_pair(workspace, tmp_path)
    rows = []
    for run_id, value in ((baseline_id, 2**64 - 1), (candidate_id, 2**64 - 2)):
        row = measurement_row(
            run_id,
            1,
            measurement_id=f"uint64-{run_id}",
            worker_id="counter:0",
            worker_run_index=0,
            value_index=0,
        )
        row.update(
            {
                "name": "device.counter",
                "unit": "count",
                "value_int": None,
                "value_float": None,
                "value_uint": value,
                "value_kind": "unsigned_integer",
            }
        )
        rows.append(row)
    GenerationPublisher(workspace).publish_rows(
        {"measurements": rows},
        publisher="uint64-comparison-fixture",
        publisher_version="1",
        input_run_ids=(baseline_id, candidate_id),
    )
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))

    result = ComparisonService(workspace).compare(
        MeasurementCompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate.run_set_id,
            metric="device.counter",
            unit="count",
            polarity=MetricPolarity.HIGHER_IS_BETTER,
            practical_threshold=0,
        )
    )

    assert result.comparison.validity is ComparisonValidity.INVALID
    assert any("exact uint64" in reason for reason in result.comparison.mismatches)


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
            metric_source=MetricSource.RUNTIME_RESOURCE,
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
    assert result.comparison.complete_pair_n is None
    assert result.comparison.independent_unit == "worker"


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
            metric_source=MetricSource.RUNTIME_RESOURCE,
            unit="bytes",
            polarity=MetricPolarity.LOWER_IS_BETTER,
            practical_threshold=0,
        )
    )

    assert result.comparison.validity is ComparisonValidity.INVALID
    assert result.comparison.candidate_eligible_n == 0
    assert result.comparison.candidate_out_of_domain_n == 1
    assert any("strictly-positive" in reason for reason in result.comparison.mismatches)
