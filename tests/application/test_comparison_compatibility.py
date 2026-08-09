from __future__ import annotations

import math
from pathlib import Path

import pytest

from flameox.application import (
    ComparisonService,
    EvidenceInput,
    EvidenceLookupService,
    FindingService,
    FreezeRunIdsRequest,
    MeasurementCompareRunSetsRequest,
    RecordFindingRequest,
    RunSetService,
)
from flameox.application.environment import collect_environment
from flameox.application.evidence_rows import environment_row
from flameox.application.run_rows import run_row
from flameox.catalog import Catalog
from flameox.domain import (
    AcceleratorIdentityFacet,
    ComparisonValidity,
    DomainError,
    ErrorCode,
    EvidenceLevel,
    ExecutionIdentityInput,
    ExternalExecutionContext,
    FindingAssessment,
    IdentityQuality,
    WorkloadExecutionIdentity,
    digest_model,
)
from flameox.evidence import GenerationPublisher
from flameox.storage import RunStore, Workspace
from tests.support.comparisons import (
    imported_benchmark,
    measurement_row,
)


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
    GenerationPublisher(workspace).publish_rows(
        {
            "frames": [
                {
                    "frame_id": "scan-frame",
                    "language": "Python",
                    "function": "reverse_scan",
                    "module": "scan",
                    "file": "scan.py",
                    "line": 10,
                    "column": None,
                    "address": None,
                    "build_id": None,
                    "module_relative_address": None,
                    "inline_chain_id": None,
                    "source_state_id": None,
                    "artifact_id": "sha256:" + "b" * 64,
                    "inlined": False,
                    "symbolization": "complete",
                }
            ]
            + [
                {
                    "frame_id": frame_id,
                    "language": "Python",
                    "function": function,
                    "module": "scan",
                    "file": "scan.py",
                    "line": line,
                    "column": None,
                    "address": None,
                    "build_id": None,
                    "module_relative_address": None,
                    "inline_chain_id": None,
                    "source_state_id": None,
                    "artifact_id": "sha256:" + "b" * 64,
                    "inlined": False,
                    "symbolization": "complete",
                }
                for frame_id, function, line in (
                    ("baseline-only", "removed_helper", 20),
                    ("candidate-only", "added_helper", 30),
                )
            ],
            "frame_measurements": [
                {
                    "run_id": run_id,
                    "artifact_id": "sha256:" + "b" * 64,
                    "frame_id": "scan-frame",
                    "metric": "cpu.time",
                    "self_value": value,
                    "inclusive_value": value,
                    "unit": "ns",
                    "sample_count": 1,
                    "thread_name": "main",
                    "process_name": "python",
                    "phase": "steady_state",
                }
                for run_id, value in ((baseline_id, 100), (candidate_id, 25))
            ]
            + [
                {
                    "run_id": run_id,
                    "artifact_id": "sha256:" + "b" * 64,
                    "frame_id": frame_id,
                    "metric": "cpu.time",
                    "self_value": value,
                    "inclusive_value": value,
                    "unit": "ns",
                    "sample_count": 1,
                    "thread_name": "main",
                    "process_name": "python",
                    "phase": "steady_state",
                }
                for run_id, frame_id, value in (
                    (baseline_id, "baseline-only", 10),
                    (candidate_id, "candidate-only", 5),
                )
            ],
        },
        publisher="profile-comparison-fixture",
        publisher_version="1",
    )
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))

    comparison_request = MeasurementCompareRunSetsRequest(
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
    changes = {item.function: item for item in preview.profile_changes}
    assert changes["reverse_scan"].direction == "improved"
    assert changes["reverse_scan"].relative_change == -0.75
    assert changes["removed_helper"].candidate_value == 0
    assert changes["removed_helper"].direction == "improved"
    assert changes["added_helper"].baseline_value == 0
    assert changes["added_helper"].direction == "regressed"
    neutral = service.compare(comparison_request.model_copy(update={"polarity": "neutral"}))
    assert {
        item.direction
        for item in neutral.profile_changes
        if not math.isclose(item.absolute_change, 0.0)
    } == {"changed"}

    result = service.record(comparison_request)
    assert result.analysis is not None
    assert result.materialized_commit_id == workspace.corpus.read_head().commit_id
    assert result.materialized_commit_id != head_before
    persisted = EvidenceLookupService(workspace).get("comparison", result.comparison.comparison_id)
    assert persisted.data["schema_version"] == 1
    assert (persisted.data["baseline_value_int"] is None) is not (
        persisted.data["baseline_value_float"] is None
    )
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


def test_comparison_reads_both_run_sets_from_one_pinned_corpus_commit(
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
    baseline = run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))
    GenerationPublisher(workspace).publish_rows(
        {
            "measurements": [
                measurement_row(baseline_id, 13_000_000),
                measurement_row(candidate_id, 6_500_000),
            ]
        },
        publisher="snapshot-regression",
        publisher_version="1",
        input_run_ids=(baseline_id, candidate_id),
    )
    candidate = run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))

    result = ComparisonService(workspace).compare(
        MeasurementCompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate.run_set_id,
            metric="pyperf.scan",
            unit="ns",
            polarity="lower_is_better",
            practical_threshold=0.05,
        )
    )

    assert baseline.corpus_commit_id != candidate.corpus_commit_id
    assert result.corpus_commit_id == workspace.corpus.read_head().commit_id
    assert result.comparison.complete_pair_n == 4


def test_comparison_rejects_different_or_partial_accelerator_identity(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_id = imported_benchmark(
        workspace,
        tmp_path / "baseline-accelerator.json",
        (0.010, 0.011, 0.012),
    )
    candidate_id = imported_benchmark(
        workspace,
        tmp_path / "candidate-accelerator.json",
        (0.005, 0.0055, 0.006),
    )
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))

    candidate_environment = collect_environment(
        AcceleratorIdentityFacet(
            provider="cuda",
            status="available",
            identity_quality=IdentityQuality.EXACT,
            driver_version="575.57.08",
            runtime_version="12.9",
        )
    )
    candidate_run = (
        RunStore(workspace)
        .read(candidate_id)
        .model_copy(update={"environment_id": candidate_environment.environment_id})
    )
    GenerationPublisher(workspace).publish_rows(
        {
            "environments": [environment_row(candidate_environment)],
            "runs": [run_row(candidate_run)],
        },
        publisher="accelerator-identity-test",
        publisher_version="1",
        input_run_ids=(candidate_id,),
    )
    candidate = run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))
    request = MeasurementCompareRunSetsRequest(
        baseline_run_set_id=baseline.run_set_id,
        candidate_run_set_id=candidate.run_set_id,
        metric="pyperf.scan",
        unit="ns",
        polarity="lower_is_better",
        practical_threshold=0.05,
    )

    different = ComparisonService(workspace).compare(request)

    assert "environment_id differs across treatments" in different.comparison.mismatches

    partial_environment = collect_environment(
        AcceleratorIdentityFacet(
            provider="cuda",
            status="missing",
            identity_quality=IdentityQuality.PARTIAL,
            missing_fields=("cuda.devices",),
        )
    )
    GenerationPublisher(workspace).publish_rows(
        {
            "environments": [environment_row(partial_environment)],
            "runs": [
                run_row(
                    RunStore(workspace)
                    .read(run_id)
                    .model_copy(update={"environment_id": partial_environment.environment_id})
                )
                for run_id in (baseline_id, candidate_id)
            ],
        },
        publisher="accelerator-identity-test",
        publisher_version="2",
        input_run_ids=(baseline_id, candidate_id),
    )

    partial = ComparisonService(workspace).compare(request)

    assert "environment identity is partial or unavailable" in partial.comparison.mismatches


def test_comparison_reports_differing_declared_artifact_paths_and_digests(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_id = imported_benchmark(
        workspace,
        tmp_path / "baseline-execution-identity.json",
        (0.010, 0.011, 0.012),
    )
    candidate_id = imported_benchmark(
        workspace,
        tmp_path / "candidate-execution-identity.json",
        (0.005, 0.0055, 0.006),
    )

    def identity(digest_character: str) -> WorkloadExecutionIdentity:
        item = ExecutionIdentityInput(
            kind="native_file",
            requested="build/libtilelang.so",
            configured_path="/project/build/libtilelang.so",
            resolved_path="/project/build/libtilelang.so",
            loaded_path="/project/build/libtilelang.so",
            content_digest="sha256:" + digest_character * 64,
            status="exact",
        )
        values = {"quality": "exact", "inputs": [item.model_dump(mode="json")]}
        return WorkloadExecutionIdentity(
            identity_id=digest_model(values),
            quality="exact",
            inputs=(item,),
        )

    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [
                run_row(
                    RunStore(workspace)
                    .read(run_id)
                    .model_copy(update={"execution_identity": identity(character)})
                )
                for run_id, character in ((baseline_id, "a"), (candidate_id, "b"))
            ]
        },
        publisher="execution-identity-test",
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
            metric="pyperf.scan",
            unit="ns",
            polarity="lower_is_better",
            practical_threshold=0.05,
        )
    )

    assert "declared execution identity differs across treatments" in (result.comparison.mismatches)
    detail = next(
        mismatch
        for mismatch in result.comparison.mismatches
        if mismatch.startswith("execution identity input")
    )
    assert "build/libtilelang.so" in detail
    assert "sha256:" + "a" * 64 in detail
    assert "sha256:" + "b" * 64 in detail


def test_distinct_remote_leases_do_not_change_environment_compatibility(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_id = imported_benchmark(
        workspace,
        tmp_path / "baseline-lease.json",
        (0.010, 0.011, 0.012),
    )
    candidate_id = imported_benchmark(
        workspace,
        tmp_path / "candidate-lease.json",
        (0.005, 0.0055, 0.006),
    )

    def context(lease_id: str) -> ExternalExecutionContext:
        return ExternalExecutionContext(
            orchestrator="crabbox",
            provider="runpod",
            lease_id=lease_id,
            worker_id=f"worker-{lease_id}",
            orchestration_run_id=f"run-{lease_id}",
        )

    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [
                run_row(
                    RunStore(workspace)
                    .read(run_id)
                    .model_copy(update={"external_context": context(lease_id)})
                )
                for run_id, lease_id in (
                    (baseline_id, "lease-a"),
                    (candidate_id, "lease-b"),
                )
            ]
        },
        publisher="lease-compatibility-test",
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
            metric="pyperf.scan",
            unit="ns",
            polarity="lower_is_better",
            practical_threshold=0.05,
        )
    )

    assert not any("lease" in mismatch for mismatch in result.comparison.mismatches)
    assert "environment_id differs across treatments" not in result.comparison.mismatches
