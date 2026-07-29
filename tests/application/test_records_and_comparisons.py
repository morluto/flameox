from __future__ import annotations

import asyncio
import json
import math
import threading
from pathlib import Path

import pyperf
import pytest

from flameox.adapters import PyPerfExtractor
from flameox.application import (
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
from flameox.application.environment import collect_environment
from flameox.application.evidence_rows import environment_row
from flameox.application.run_rows import run_row
from flameox.catalog import Catalog, Snapshot
from flameox.domain import (
    AcceleratorIdentityFacet,
    ArtifactKind,
    ComparisonValidity,
    DomainError,
    ErrorCode,
    EvidenceLevel,
    ExecutionIdentityInput,
    ExternalExecutionContext,
    FindingAssessment,
    IdentityQuality,
    RunSet,
    WorkloadExecutionIdentity,
    digest_model,
)
from flameox.evidence import GenerationPublisher
from flameox.storage import RunStore, Workspace


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


def additional_measurement(run_id: str, value: int) -> dict[str, object]:
    return {
        "measurement_id": f"additional-{run_id}",
        "run_id": run_id,
        "artifact_id": None,
        "name": "pyperf.scan",
        "value_int": value,
        "value_float": None,
        "unit": "ns",
        "aggregation": "sample",
        "scope": "process",
        "trial_id": None,
        "worker_id": "additional-worker",
        "worker_run_index": 0,
        "value_index": 0,
        "loop_count": 1,
        "is_warmup": False,
        "block_id": None,
        "variant_id": None,
        "order_in_block": None,
        "phase": None,
        "dimensions": {},
        "evidence_level": "observed",
    }


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
    baseline = run_sets.freeze(FreezeRunSetRequest(run_ids=(baseline_id,)))
    GenerationPublisher(workspace).publish_rows(
        {
            "measurements": [
                additional_measurement(baseline_id, 13_000_000),
                additional_measurement(candidate_id, 6_500_000),
            ]
        },
        publisher="snapshot-regression",
        publisher_version="1",
        input_run_ids=(baseline_id, candidate_id),
    )
    candidate = run_sets.freeze(FreezeRunSetRequest(run_ids=(candidate_id,)))

    result = ComparisonService(workspace).compare(
        CompareRunSetsRequest(
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
    baseline = run_sets.freeze(FreezeRunSetRequest(run_ids=(baseline_id,)))

    candidate_environment = collect_environment(
        AcceleratorIdentityFacet(
            provider="cuda",
            status="available",
            identity_quality=IdentityQuality.EXACT,
            driver_version="575.57.08",
            runtime_version="12.9",
        )
    )
    candidate_run = RunStore(workspace).read(candidate_id).model_copy(
        update={"environment_id": candidate_environment.environment_id}
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
    candidate = run_sets.freeze(FreezeRunSetRequest(run_ids=(candidate_id,)))
    request = CompareRunSetsRequest(
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
    baseline = run_sets.freeze(FreezeRunSetRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunSetRequest(run_ids=(candidate_id,)))

    result = ComparisonService(workspace).compare(
        CompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate.run_set_id,
            metric="pyperf.scan",
            unit="ns",
            polarity="lower_is_better",
            practical_threshold=0.05,
        )
    )

    assert "declared execution identity differs across treatments" in (
        result.comparison.mismatches
    )
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
    baseline = run_sets.freeze(FreezeRunSetRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunSetRequest(run_ids=(candidate_id,)))

    result = ComparisonService(workspace).compare(
        CompareRunSetsRequest(
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


@pytest.mark.anyio
async def test_async_comparison_cancellation_interrupts_duckdb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunSetRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunSetRequest(run_ids=(candidate_id,)))
    request = CompareRunSetsRequest(
        baseline_run_set_id=baseline.run_set_id,
        candidate_run_set_id=candidate.run_set_id,
        metric="pyperf.scan",
        unit="ns",
        polarity="lower_is_better",
        practical_threshold=0.05,
    )
    query_started = threading.Event()
    original = ComparisonService._compatibility_mismatches

    def slow_compatibility(
        service: ComparisonService,
        snapshot: Snapshot,
        baseline_set: RunSet,
        candidate_set: RunSet,
    ) -> list[str]:
        query_started.set()
        snapshot.execute("SELECT sum(sin(i)) FROM range(100000000000) values(i)").fetchall()
        return original(service, snapshot, baseline_set, candidate_set)

    monkeypatch.setattr(ComparisonService, "_compatibility_mismatches", slow_compatibility)
    task = asyncio.create_task(ComparisonService(workspace).compare_async(request))
    assert await asyncio.to_thread(query_started.wait, 5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


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
                    "combination_id": digest_model(
                        {"variant": variant, "block": block_id}
                    ),
                    "factors_json": json.dumps({"variant": variant}, sort_keys=True),
                    "block_id": block_id,
                    "order_in_block": 0,
                    "parameter_name": None,
                    "parameter_value_int": None,
                    "parameter_value_float": None,
                    "attempt": 1,
                    "outcome": "succeeded",
                    "exclusion_reason": None,
                    "validation_status": "not_requested",
                    "failure_class": "none",
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
        (trial_id, run_id) for trial_id, run_id, variant, _ in trials if variant == "candidate"
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
