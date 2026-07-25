from __future__ import annotations

from pathlib import Path

import pyperf
import pytest

from flamo.adapters import PyPerfExtractor
from flamo.application import (
    CompareRunSetsRequest,
    ComparisonService,
    CreateInvestigationRequest,
    EvidenceInput,
    FindingService,
    FreezeRunSetMember,
    FreezeRunSetRequest,
    ImportArtifactRequest,
    ImportService,
    InvestigationService,
    RecordFindingRequest,
    RecordHypothesisRequest,
    RunSetService,
)
from flamo.catalog import Catalog
from flamo.domain import (
    ArtifactKind,
    ComparisonValidity,
    DomainError,
    ErrorCode,
    EvidenceLevel,
    FindingAssessment,
)
from flamo.evidence import GenerationPublisher
from flamo.storage import Workspace


def benchmark(path: Path, values: tuple[float, float, float]) -> None:
    run = pyperf.Run(
        values,
        metadata={"name": "scan", "unit": "second", "loops": 1},
        collect_metadata=False,
    )
    pyperf.BenchmarkSuite([pyperf.Benchmark([run])]).dump(
        str(path),
        replace=True,
    )


def imported_benchmark(
    workspace: Workspace,
    path: Path,
    values: tuple[float, float, float],
) -> str:
    benchmark(path, values)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=path,
            kind=ArtifactKind.BENCHMARK_SAMPLES,
        )
    )
    PyPerfExtractor(workspace).extract(imported.run.run_id)
    return imported.run.run_id


def test_investigation_hypothesis_revision_uses_compare_and_swap(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    service = InvestigationService(workspace)
    investigation = service.create(
        CreateInvestigationRequest(
            question="Why does the reverse scan scale linearly?",
        )
    )
    first = service.record_hypothesis(
        RecordHypothesisRequest(
            investigation_id=investigation.investigation_id,
            claim="Python loop overhead dominates.",
            prediction="Runtime doubles with length.",
            discriminating_condition="A vectorized implementation removes the slope.",
        )
    )
    second = service.record_hypothesis(
        RecordHypothesisRequest(
            investigation_id=investigation.investigation_id,
            hypothesis_id=first.hypothesis_id,
            expected_revision=1,
            claim=first.claim,
            prediction="Runtime approximately doubles with length.",
            discriminating_condition=first.discriminating_condition,
        )
    )

    assert second.revision == 2
    with pytest.raises(DomainError) as stale:
        service.record_hypothesis(
            RecordHypothesisRequest(
                investigation_id=investigation.investigation_id,
                hypothesis_id=first.hypothesis_id,
                expected_revision=1,
                claim=first.claim,
                prediction=first.prediction,
                discriminating_condition=first.discriminating_condition,
            )
        )
    assert stale.value.code is ErrorCode.REVISION_CONFLICT


def test_frozen_run_set_comparison_and_evidence_linked_finding(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    baseline_id = imported_benchmark(
        workspace,
        tmp_path / "baseline.json",
        (0.010, 0.011, 0.012),
    )
    candidate_id = imported_benchmark(
        workspace,
        tmp_path / "candidate.json",
        (0.005, 0.0055, 0.006),
    )
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunSetRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunSetRequest(run_ids=(candidate_id,)))

    comparison_request = CompareRunSetsRequest(
        baseline_run_set_id=baseline.run_set_id,
        candidate_run_set_id=candidate.run_set_id,
        metric="pyperf.scan",
        unit="ns",
        polarity="lower_is_better",
        practical_threshold=0.05,
    )
    service = ComparisonService(workspace)
    head_before = workspace.corpus.read_head().commit_id
    preview = service.compare(comparison_request)
    repeated = service.compare(comparison_request)
    assert workspace.corpus.read_head().commit_id == head_before
    assert preview.comparison.comparison_id == repeated.comparison.comparison_id
    assert preview.analysis is None

    result = service.record(comparison_request)
    assert result.analysis is not None
    assert result.materialized_commit_id == workspace.corpus.read_head().commit_id
    assert result.materialized_commit_id != head_before
    request = RecordFindingRequest(
        kind="performance",
        title="Candidate halves reverse-scan time",
        claim="The candidate is materially faster on the frozen cohort.",
        evidence_level=EvidenceLevel.DERIVED,
        confidence="high",
        assessment=FindingAssessment.SUPPORTED,
        evidence=(
            EvidenceInput(
                ref_type="comparison",
                ref_id=result.comparison.comparison_id,
                relation="supports",
            ),
        ),
    )

    assert result.comparison.validity is ComparisonValidity.INVALID
    assert result.comparison.complete_pair_n == 3
    with pytest.raises(DomainError) as invalid_proof:
        FindingService(workspace).record(request)
    assert invalid_proof.value.code is ErrorCode.COMPARISON_INVALID
    finding = FindingService(workspace).record(
        request.model_copy(
            update={
                "assessment": FindingAssessment.INCONCLUSIVE,
                "confidence": "low",
            }
        )
    )
    assert finding.evidence[0].ref_id == result.comparison.comparison_id
    assert finding.finding.revision == 1


def test_trial_block_identity_makes_pairing_independent_of_member_order(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    runs: dict[str, str] = {}
    for name, values in {
        "baseline-1": (0.010, 0.010, 0.010),
        "baseline-2": (0.020, 0.020, 0.020),
        "candidate-1": (0.005, 0.005, 0.005),
        "candidate-2": (0.015, 0.015, 0.015),
    }.items():
        runs[name] = imported_benchmark(workspace, tmp_path / f"{name}.json", values)
    trials = [
        ("baseline-trial-1", runs["baseline-1"], "baseline", "block-1"),
        ("baseline-trial-2", runs["baseline-2"], "baseline", "block-2"),
        ("candidate-trial-1", runs["candidate-1"], "candidate", "block-1"),
        ("candidate-trial-2", runs["candidate-2"], "candidate", "block-2"),
    ]
    GenerationPublisher(workspace).publish_rows(
        {
            "trials": [
                {
                    "trial_id": trial_id,
                    "experiment_id": "experiment",
                    "variant_id": variant,
                    "run_id": run_id,
                    "block_id": block_id,
                    "order_in_block": 0,
                    "parameter_name": None,
                    "parameter_value_int": None,
                    "parameter_value_float": None,
                    "attempt": 1,
                    "outcome": "succeeded",
                    "exclusion_reason": None,
                    "validation_status": "not_requested",
                }
                for trial_id, run_id, variant, block_id in trials
            ]
        },
        publisher="trial-fixture",
        publisher_version="1",
        input_run_ids=tuple(run_id for _, run_id, _, _ in trials),
    )
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(
        FreezeRunSetRequest(
            members=tuple(
                FreezeRunSetMember(run_id=run_id, trial_id=trial_id)
                for trial_id, run_id, variant, _ in trials
                if variant == "baseline"
            )
        )
    )
    candidates = [
        (trial_id, run_id)
        for trial_id, run_id, variant, _ in trials
        if variant == "candidate"
    ]
    candidate_forward = run_sets.freeze(
        FreezeRunSetRequest(
            members=tuple(
                FreezeRunSetMember(run_id=run_id, trial_id=trial_id)
                for trial_id, run_id in candidates
            )
        )
    )
    candidate_reversed = run_sets.freeze(
        FreezeRunSetRequest(
            members=tuple(
                FreezeRunSetMember(run_id=run_id, trial_id=trial_id)
                for trial_id, run_id in reversed(candidates)
            )
        )
    )
    service = ComparisonService(workspace)

    forward = service.compare(
        CompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate_forward.run_set_id,
            metric="pyperf.scan",
            unit="ns",
            polarity="lower_is_better",
            practical_threshold=0.01,
        )
    )
    reversed_result = service.compare(
        CompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate_reversed.run_set_id,
            metric="pyperf.scan",
            unit="ns",
            polarity="lower_is_better",
            practical_threshold=0.01,
        )
    )

    assert forward.comparison.complete_pair_n == 2
    assert forward.comparison.relative_change == reversed_result.comparison.relative_change
