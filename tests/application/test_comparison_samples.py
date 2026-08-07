from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application import (
    CompareRunSetsRequest,
    ComparisonService,
    FreezeRunSetRequest,
    RunSetService,
)
from flameox.catalog import Catalog
from flameox.domain import DomainError, ErrorCode
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace
from tests.support.comparisons import imported_benchmark, measurement_row


def _comparison_request(
    workspace: Workspace,
    baseline_id: str,
    candidate_id: str,
) -> CompareRunSetsRequest:
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunSetRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunSetRequest(run_ids=(candidate_id,)))
    return CompareRunSetsRequest(
        baseline_run_set_id=baseline.run_set_id,
        candidate_run_set_id=candidate.run_set_id,
        metric="pyperf.scan",
        unit="ns",
        polarity="lower_is_better",
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
