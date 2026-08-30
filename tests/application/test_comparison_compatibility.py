from __future__ import annotations

import math
from pathlib import Path

import pytest

from flameox.application.comparisons import (
    ComparisonService,
    ExcludedFreezeRunSetMember,
    FreezeRunIdsRequest,
    FreezeRunMembersRequest,
    IncludedFreezeRunSetMember,
    MeasurementCompareRunSetsRequest,
    RunSetService,
)
from flameox.application.environment import collect_environment
from flameox.application.evidence_lookup import EvidenceLookupService
from flameox.application.evidence_rows import environment_row
from flameox.application.records import (
    EvidenceInput,
    FindingService,
    RecordFindingRequest,
)
from flameox.application.run_rows import run_row
from flameox.catalog import Catalog
from flameox.domain import (
    AcceleratorIdentityFacet,
    AcceleratorIdentityStatus,
    CaptureScope,
    ComparisonValidity,
    DomainError,
    ErrorCode,
    EvidenceLevel,
    EvidenceReferenceType,
    EvidenceRelation,
    ExecutionIdentityInput,
    ExecutionIdentityInputKind,
    ExecutionIdentityInputStatus,
    ExecutionIdentityQuality,
    ExternalExecutionContext,
    FindingAssessment,
    FindingConfidence,
    IdentityQuality,
    MetricPolarity,
    RunManifest,
    RunSemantics,
    SemanticOption,
    WorkloadExecutionIdentity,
    digest_model,
)
from flameox.evidence import GenerationPublisher
from flameox.storage import RunStore, Workspace
from tests.support.comparisons import (
    imported_benchmark,
    measurement_row,
)

pytestmark = [pytest.mark.integration, pytest.mark.serial]


def _revised_run(
    workspace: Workspace,
    run_id: str,
    *,
    revision_increment: int = 1,
    **updates: object,
) -> RunManifest:
    current = RunStore(workspace).read(run_id)
    return current.model_copy(update={"revision": current.revision + revision_increment, **updates})


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
        polarity=MetricPolarity.LOWER_IS_BETTER,
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
    neutral = service.compare(
        comparison_request.model_copy(update={"polarity": MetricPolarity.NEUTRAL})
    )
    assert {
        item.direction
        for item in neutral.profile_changes
        if not math.isclose(item.absolute_change, 0.0)
    } == {"changed"}

    result = service.record(comparison_request)
    assert result.analysis is not None
    assert result.materialized_commit_id == workspace.corpus.read_head().commit_id
    assert result.materialized_commit_id != head_before
    Catalog(workspace).rebuild()
    run_sets = RunSetService(workspace)
    later_baseline = run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))
    later_candidate = run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))
    after_unrelated_publication = service.compare(
        comparison_request.model_copy(
            update={
                "baseline_run_set_id": later_baseline.run_set_id,
                "candidate_run_set_id": later_candidate.run_set_id,
            }
        )
    )
    assert after_unrelated_publication.corpus_commit_id != preview.corpus_commit_id
    assert after_unrelated_publication.comparison.comparison_id == preview.comparison.comparison_id
    GenerationPublisher(workspace).publish_rows(
        {"measurements": [measurement_row(baseline_id, 13_000_000)]},
        publisher="new-comparison-input",
        publisher_version="1",
    )
    Catalog(workspace).rebuild()
    changed_run_sets = RunSetService(workspace)
    changed_baseline = changed_run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))
    changed_candidate = changed_run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))
    changed = service.compare(
        comparison_request.model_copy(
            update={
                "baseline_run_set_id": changed_baseline.run_set_id,
                "candidate_run_set_id": changed_candidate.run_set_id,
            }
        )
    )
    assert changed.comparison.comparison_id != preview.comparison.comparison_id
    persisted = EvidenceLookupService(workspace).get(
        EvidenceReferenceType.COMPARISON,
        result.comparison.comparison_id,
    )
    assert (persisted.data["baseline_value_int"] is None) is not (
        persisted.data["baseline_value_float"] is None
    )
    request = RecordFindingRequest(
        kind="performance",
        title="Candidate halves reverse-scan time",
        claim="The candidate is materially faster on the frozen cohort.",
        evidence_level=EvidenceLevel.DERIVED,
        confidence=FindingConfidence.HIGH,
        assessment=FindingAssessment.SUPPORTED,
        evidence=(
            EvidenceInput(
                ref_type=EvidenceReferenceType.COMPARISON,
                ref_id=result.comparison.comparison_id,
                relation=EvidenceRelation.SUPPORTS,
            ),
        ),
    )

    assert result.comparison.validity is ComparisonValidity.INVALID
    assert result.comparison.complete_pair_n is None
    assert result.comparison.baseline_eligible_n == 1
    assert result.comparison.independent_unit == "worker"
    with pytest.raises(DomainError) as invalid_proof:
        FindingService(workspace).record(request)
    assert invalid_proof.value.code is ErrorCode.COMPARISON_INVALID
    finding = FindingService(workspace).record(
        request.model_copy(
            update={
                "assessment": FindingAssessment.INCONCLUSIVE,
                "confidence": "low",
                "evidence": (
                    request.evidence[0].model_copy(update={"relation": EvidenceRelation.CONTEXT}),
                ),
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
            polarity=MetricPolarity.LOWER_IS_BETTER,
            practical_threshold=0.05,
        )
    )

    assert baseline.corpus_commit_id == candidate.corpus_commit_id
    assert result.corpus_commit_id == baseline.corpus_commit_id
    assert result.comparison.complete_pair_n is None
    assert result.comparison.baseline_eligible_n == 1


def test_comparison_rejects_mismatched_memray_region_semantics(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_id = imported_benchmark(
        workspace,
        tmp_path / "baseline-memray-semantics.json",
        (0.010, 0.011, 0.012),
    )
    candidate_id = imported_benchmark(
        workspace,
        tmp_path / "candidate-memray-semantics.json",
        (0.005, 0.0055, 0.006),
    )

    def semantics(region: str, warmup_count: int) -> RunSemantics:
        return RunSemantics(
            origin="import",
            adapter="memray",
            adapter_version="1.19.3",
            scope=CaptureScope(
                mode=SemanticOption(name="mode", value="sdk"),
                process_scope=SemanticOption(name="process_scope", value="workload_process"),
                bounds={"warmup_count": warmup_count},
                filters={"region": region, "thread_scope": "all_threads"},
            ),
        )

    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [
                run_row(
                    _revised_run(
                        workspace,
                        baseline_id,
                        semantics=semantics("steady_step", 2),
                    )
                ),
                run_row(
                    _revised_run(
                        workspace,
                        candidate_id,
                        semantics=semantics("steady_step", 3),
                    )
                ),
            ]
        },
        publisher="memray-semantic-compatibility-test",
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
            polarity=MetricPolarity.LOWER_IS_BETTER,
            practical_threshold=0.05,
        )
    )

    assert result.comparison.validity is ComparisonValidity.INVALID
    assert "run semantics differ across treatments" in result.comparison.mismatches


def test_comparison_rejects_run_sets_frozen_from_different_snapshots(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
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
    baseline = RunSetService(workspace).freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))
    candidate = RunSetService(workspace).freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))
    assert baseline.corpus_commit_id != candidate.corpus_commit_id

    with pytest.raises(DomainError) as error:
        ComparisonService(workspace).compare(
            MeasurementCompareRunSetsRequest(
                baseline_run_set_id=baseline.run_set_id,
                candidate_run_set_id=candidate.run_set_id,
                metric="pyperf.scan",
                unit="ns",
                polarity=MetricPolarity.LOWER_IS_BETTER,
                practical_threshold=0.05,
            )
        )

    assert error.value.code is ErrorCode.COMPARISON_INVALID


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
    candidate_environment = collect_environment(
        AcceleratorIdentityFacet(
            provider="cuda",
            status=AcceleratorIdentityStatus.AVAILABLE,
            identity_quality=IdentityQuality.EXACT,
            driver_version="575.57.08",
            runtime_version="12.9",
        )
    )
    candidate_run = _revised_run(
        workspace,
        candidate_id,
        environment_id=candidate_environment.environment_id,
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
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))
    request = MeasurementCompareRunSetsRequest(
        baseline_run_set_id=baseline.run_set_id,
        candidate_run_set_id=candidate.run_set_id,
        metric="pyperf.scan",
        unit="ns",
        polarity=MetricPolarity.LOWER_IS_BETTER,
        practical_threshold=0.05,
    )

    different = ComparisonService(workspace).compare(request)

    assert "environment_id differs across treatments" in different.comparison.mismatches

    partial_environment = collect_environment(
        AcceleratorIdentityFacet(
            provider="cuda",
            status=AcceleratorIdentityStatus.MISSING,
            identity_quality=IdentityQuality.PARTIAL,
            missing_fields=("cuda.devices",),
        )
    )
    GenerationPublisher(workspace).publish_rows(
        {
            "environments": [environment_row(partial_environment)],
            "runs": [
                run_row(
                    _revised_run(
                        workspace,
                        run_id,
                        revision_increment=2,
                        environment_id=partial_environment.environment_id,
                    )
                )
                for run_id in (baseline_id, candidate_id)
            ],
        },
        publisher="accelerator-identity-test",
        publisher_version="2",
        input_run_ids=(baseline_id, candidate_id),
    )

    partial_run_sets = RunSetService(workspace)
    partial_baseline = partial_run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))
    partial_candidate = partial_run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))
    partial = ComparisonService(workspace).compare(
        request.validated_copy(
            update={
                "baseline_run_set_id": partial_baseline.run_set_id,
                "candidate_run_set_id": partial_candidate.run_set_id,
            }
        )
    )

    assert "environment identity is partial or unavailable" in partial.comparison.mismatches
    assert partial.comparison.comparison_id != different.comparison.comparison_id


def test_excluded_member_is_coverage_evidence_not_a_compatibility_observation(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_id = imported_benchmark(
        workspace,
        tmp_path / "baseline-included.json",
        (0.010, 0.011, 0.012),
    )
    candidate_id = imported_benchmark(
        workspace,
        tmp_path / "candidate-included.json",
        (0.005, 0.0055, 0.006),
    )
    excluded_id = imported_benchmark(
        workspace,
        tmp_path / "baseline-excluded.json",
        (0.050, 0.051, 0.052),
    )
    incompatible_environment = collect_environment(
        AcceleratorIdentityFacet(
            provider="cuda",
            status=AcceleratorIdentityStatus.AVAILABLE,
            identity_quality=IdentityQuality.EXACT,
            driver_version="different",
            runtime_version="different",
        )
    )
    GenerationPublisher(workspace).publish_rows(
        {
            "environments": [environment_row(incompatible_environment)],
            "runs": [
                run_row(
                    _revised_run(
                        workspace,
                        excluded_id,
                        environment_id=incompatible_environment.environment_id,
                    )
                )
            ],
        },
        publisher="excluded-cohort-test",
        publisher_version="1",
        input_run_ids=(excluded_id,),
    )
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(
        FreezeRunMembersRequest(
            members=(
                IncludedFreezeRunSetMember(run_id=baseline_id),
                ExcludedFreezeRunSetMember(
                    run_id=excluded_id,
                    reason="predeclared unsupported host",
                ),
            )
        )
    )
    candidate = run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))

    result = ComparisonService(workspace).compare(
        MeasurementCompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate.run_set_id,
            metric="pyperf.scan",
            unit="ns",
            polarity=MetricPolarity.LOWER_IS_BETTER,
            practical_threshold=0.05,
        )
    )

    assert result.comparison.baseline_excluded_n == 1
    assert result.comparison.baseline_eligible_n == 1
    assert "environment_id differs across treatments" not in result.comparison.mismatches


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
            kind=ExecutionIdentityInputKind.NATIVE_FILE,
            requested="build/libtilelang.so",
            configured_path="/project/build/libtilelang.so",
            resolved_path="/project/build/libtilelang.so",
            loaded_path="/project/build/libtilelang.so",
            content_digest="sha256:" + digest_character * 64,
            status=ExecutionIdentityInputStatus.EXACT,
        )
        values = {
            "quality": "exact",
            "inputs": [item.model_dump(mode="json")],
            "missing_inputs": [],
        }
        return WorkloadExecutionIdentity(
            identity_id=digest_model(values),
            quality=ExecutionIdentityQuality.EXACT,
            inputs=(item,),
        )

    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [
                run_row(
                    _revised_run(
                        workspace,
                        run_id,
                        execution_identity=identity(character),
                    )
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
            polarity=MetricPolarity.LOWER_IS_BETTER,
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
                    _revised_run(
                        workspace,
                        run_id,
                        external_context=context(lease_id),
                    )
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
            polarity=MetricPolarity.LOWER_IS_BETTER,
            practical_threshold=0.05,
        )
    )

    assert not any("lease" in mismatch for mismatch in result.comparison.mismatches)
    assert "environment_id differs across treatments" not in result.comparison.mismatches
