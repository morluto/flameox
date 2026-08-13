"""Focused tests for inference replay comparison compatibility.

These tests prove that:
- Inference replay runs with incomplete identity (existing_local, no oracle)
  produce EXPLORATORY comparisons, not INVALID.
- Differing protocol facets still produce INVALID comparisons.
- Non-inference runs are unaffected by the inference compatibility path.
- Mixed inference/non-inference run sets are INVALID.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from flameox.analysis.inference_protocol import (
    HardwareIdentity,
    InferenceProtocolIdentity,
    ModelIdentity,
    OracleIdentity,
    OracleResult,
    ProfilerState,
    ScheduleIdentity,
    ServerConfigIdentity,
    TraceIdentity,
)
from flameox.application import (
    ComparisonService,
    FreezeRunIdsRequest,
    FreezeRunMembersRequest,
    MeasurementCompareRunSetsRequest,
    RunSetService,
)
from flameox.application.environment import collect_environment
from flameox.application.evidence_rows import environment_row
from flameox.application.run_rows import run_row
from flameox.catalog import Catalog
from flameox.domain import (
    CaptureStatus,
    ComparisonValidity,
    ExecutionStatus,
    MetricPolarity,
    OracleStatus,
    ValidationStatus,
    digest_model,
)
from flameox.domain.models import ImportRunManifest
from flameox.evidence import GenerationPublisher
from flameox.storage import RunStore, Workspace

pytestmark = pytest.mark.unit


def _protocol_identity(
    *,
    model_id: str = "test-model",
    backend: Literal["vllm", "openai-chat", "custom"] = "vllm",
    provider: str = "aiperf",
    oracle_kind: str = "none",
) -> InferenceProtocolIdentity:
    return InferenceProtocolIdentity(
        provider=provider,
        provider_version="0.1.0",
        trace=TraceIdentity(producer="flameox-test"),
        schedule=ScheduleIdentity(preserve_timing=True),
        model=ModelIdentity(model_id=model_id),
        server=ServerConfigIdentity(backend=backend),
        hardware=HardwareIdentity(),
        profiler=ProfilerState(),
        oracle=OracleIdentity(kind=oracle_kind),  # type: ignore[arg-type]
    )


def _complete_protocol_identity() -> InferenceProtocolIdentity:
    return InferenceProtocolIdentity(
        provider="aiperf",
        provider_version="0.12.0",
        provider_executable_digest="sha256:" + "1" * 64,
        trace=TraceIdentity(
            producer="flameox-test",
            artifact_digest="sha256:" + "2" * 64,
        ),
        schedule=ScheduleIdentity(preserve_timing=True),
        model=ModelIdentity(
            model_id="test-model",
            model_revision="revision-1",
            tokenizer_id="test-tokenizer",
            tokenizer_revision="revision-1",
            quantization="none",
        ),
        server=ServerConfigIdentity(
            backend="vllm",
            managed_server_command_digest="sha256:" + "3" * 64,
            server_executable_digest="sha256:" + "5" * 64,
            server_version="0.26.0",
        ),
        hardware=HardwareIdentity(accelerator_kind="cpu", accelerator_count=0),
        profiler=ProfilerState(),
        oracle=OracleIdentity(
            kind="contract_check",
            command_digest="sha256:" + "4" * 64,
        ),
        oracle_result=OracleResult(status=OracleStatus.PASS, reason="contract_passed"),
    )


def _publish_inference_run(
    workspace: Workspace,
    *,
    run_id: str,
    protocol: InferenceProtocolIdentity,
    values: tuple[int, ...],
    metric: str = "inference.latency",
    unit: str = "ns",
    validation_status: ValidationStatus = ValidationStatus.NOT_REQUESTED,
) -> str:
    """Publish a run with an inference protocol identity and measurements."""
    environment = collect_environment()
    protocol_json = json.dumps(
        protocol.model_dump(mode="json"),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    manifest = ImportRunManifest(
        run_id=run_id,
        execution_status=ExecutionStatus.NOT_APPLICABLE,
        capture_status=CaptureStatus.REGISTERED,
        validation_status=validation_status,
        environment_id=environment.environment_id,
        collector="inference-replay",
        inference_protocol_identity_id=digest_model(protocol_json),
        inference_protocol_identity_json=protocol_json,
    )
    RunStore(workspace).create(manifest)
    artifact_id = "sha256:" + "f" * 64
    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row(manifest)],
            "environments": [environment_row(environment)],
            "measurements": [
                {
                    "measurement_id": f"{run_id}-m{i}",
                    "run_id": run_id,
                    "artifact_id": artifact_id,
                    "name": metric,
                    "value_int": value,
                    "value_float": None,
                    "unit": unit,
                    "aggregation": "sample",
                    "scope": "process",
                    "trial_id": None,
                    "worker_id": "worker-0",
                    "worker_run_index": 0,
                    "value_index": i,
                    "loop_count": 1,
                    "is_warmup": False,
                    "block_id": f"block-{i}",
                    "variant_id": None,
                    "order_in_block": 0,
                    "phase": "steady_state",
                    "dimensions": {},
                    "evidence_level": "observed",
                }
                for i, value in enumerate(values)
            ],
        },
        publisher="inference-comparison-test",
        publisher_version="1",
        input_run_ids=(run_id,),
    )
    return run_id


def _compare(
    workspace: Workspace,
    baseline_id: str,
    candidate_id: str,
    *,
    metric: str = "inference.latency",
    unit: str = "ns",
) -> ComparisonValidity:
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))
    result = ComparisonService(workspace).compare(
        MeasurementCompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate.run_set_id,
            metric=metric,
            unit=unit,
            polarity=MetricPolarity.LOWER_IS_BETTER,
            practical_threshold=0.05,
        )
    )
    return result.comparison.validity


def test_inference_runs_with_incomplete_identity_are_exploratory_not_invalid(
    tmp_path: Path,
) -> None:
    """Existing-local inference runs lack source_state_id, execution identity,
    validation, and oracle. These are exploratory limitations, not proof
    invalidators."""
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    protocol = _protocol_identity()
    baseline_id = _publish_inference_run(
        workspace, run_id="sha256:" + "1" * 64, protocol=protocol, values=(100, 110, 120)
    )
    candidate_id = _publish_inference_run(
        workspace, run_id="sha256:" + "2" * 64, protocol=protocol, values=(80, 85, 90)
    )

    validity = _compare(workspace, baseline_id, candidate_id)

    assert validity is ComparisonValidity.EXPLORATORY


def test_complete_inference_protocol_does_not_promote_one_worker_to_three_units(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    protocol = _complete_protocol_identity()
    baseline_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "5" * 64,
        protocol=protocol,
        values=(100, 110, 120),
        validation_status=ValidationStatus.PASSED,
    )
    candidate_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "6" * 64,
        protocol=protocol,
        values=(80, 85, 90),
        validation_status=ValidationStatus.PASSED,
    )

    validity = _compare(workspace, baseline_id, candidate_id)

    assert validity is ComparisonValidity.EXPLORATORY


def test_inference_runs_with_differing_protocol_facets_are_invalid(
    tmp_path: Path,
) -> None:
    """A present-but-differing protocol facet (model_id) invalidates the
    comparison, even for inference runs."""
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    baseline_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "3" * 64,
        protocol=_protocol_identity(model_id="model-a"),
        values=(100, 110, 120),
    )
    candidate_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "4" * 64,
        protocol=_protocol_identity(model_id="model-b"),
        values=(80, 85, 90),
    )

    validity = _compare(workspace, baseline_id, candidate_id)

    assert validity is ComparisonValidity.INVALID


def test_inference_runs_with_differing_server_backend_are_invalid(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    baseline_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "5" * 64,
        protocol=_protocol_identity(backend="vllm"),
        values=(100, 110, 120),
    )
    candidate_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "6" * 64,
        protocol=_protocol_identity(backend="openai-chat"),
        values=(80, 85, 90),
    )

    validity = _compare(workspace, baseline_id, candidate_id)

    assert validity is ComparisonValidity.INVALID


def test_non_inference_runs_still_produce_invalid_for_missing_source_state(
    tmp_path: Path,
) -> None:
    """Non-inference runs (no protocol identity) with missing source_state_id
    are still INVALID, preserving the existing behavior."""
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    # Publish runs without protocol identity and without source_state_id
    environment = collect_environment()
    for run_id, values, char in (
        ("sha256:" + "7" * 64, (100, 110, 120), "a"),
        ("sha256:" + "8" * 64, (80, 85, 90), "b"),
    ):
        manifest = ImportRunManifest(
            run_id=run_id,
            execution_status=ExecutionStatus.NOT_APPLICABLE,
            capture_status=CaptureStatus.REGISTERED,
            validation_status=ValidationStatus.NOT_REQUESTED,
            environment_id=environment.environment_id,
            collector="import",
        )
        RunStore(workspace).create(manifest)
        GenerationPublisher(workspace).publish_rows(
            {
                "runs": [run_row(manifest)],
                "environments": [environment_row(environment)],
                "measurements": [
                    {
                        "measurement_id": f"{run_id}-m{i}",
                        "run_id": run_id,
                        "artifact_id": "sha256:" + char * 64,
                        "name": "inference.latency",
                        "value_int": value,
                        "value_float": None,
                        "unit": "ns",
                        "aggregation": "sample",
                        "scope": "process",
                        "trial_id": None,
                        "worker_id": "worker-0",
                        "worker_run_index": 0,
                        "value_index": i,
                        "loop_count": 1,
                        "is_warmup": False,
                        "block_id": f"block-{i}",
                        "variant_id": None,
                        "order_in_block": 0,
                        "phase": "steady_state",
                        "dimensions": {},
                        "evidence_level": "observed",
                    }
                    for i, value in enumerate(values)
                ],
            },
            publisher="non-inference-test",
            publisher_version="1",
            input_run_ids=(run_id,),
        )

    validity = _compare(workspace, "sha256:" + "7" * 64, "sha256:" + "8" * 64)

    assert validity is ComparisonValidity.INVALID


def test_mixed_inference_and_non_inference_runs_are_invalid(
    tmp_path: Path,
) -> None:
    """Comparing an inference run with a non-inference run is invalid —
    they are fundamentally different workload types."""
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    inference_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "9" * 64,
        protocol=_protocol_identity(),
        values=(100, 110, 120),
    )
    # Non-inference run
    environment = collect_environment()
    non_inference_id = "sha256:" + "0" * 64
    manifest = ImportRunManifest(
        run_id=non_inference_id,
        execution_status=ExecutionStatus.NOT_APPLICABLE,
        capture_status=CaptureStatus.REGISTERED,
        validation_status=ValidationStatus.NOT_REQUESTED,
        environment_id=environment.environment_id,
        collector="import",
    )
    RunStore(workspace).create(manifest)
    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row(manifest)],
            "environments": [environment_row(environment)],
            "measurements": [
                {
                    "measurement_id": f"{non_inference_id}-m{i}",
                    "run_id": non_inference_id,
                    "artifact_id": "sha256:" + "e" * 64,
                    "name": "inference.latency",
                    "value_int": value,
                    "value_float": None,
                    "unit": "ns",
                    "aggregation": "sample",
                    "scope": "process",
                    "trial_id": None,
                    "worker_id": "worker-0",
                    "worker_run_index": 0,
                    "value_index": i,
                    "loop_count": 1,
                    "is_warmup": False,
                    "block_id": f"block-{i}",
                    "variant_id": None,
                    "order_in_block": 0,
                    "phase": "steady_state",
                    "dimensions": {},
                    "evidence_level": "observed",
                }
                for i, value in enumerate((80, 85, 90))
            ],
        },
        publisher="mixed-test",
        publisher_version="1",
        input_run_ids=(non_inference_id,),
    )

    validity = _compare(workspace, inference_id, non_inference_id)

    assert validity is ComparisonValidity.INVALID


def test_inference_comparison_mismatches_include_protocol_and_exploratory_reasons(
    tmp_path: Path,
) -> None:
    """The comparison's mismatch list includes both protocol mismatches
    (invalidating) and exploratory limitations."""
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    baseline_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "a" * 64,
        protocol=_protocol_identity(model_id="model-a"),
        values=(100, 110, 120),
    )
    candidate_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "b" * 64,
        protocol=_protocol_identity(model_id="model-b"),
        values=(80, 85, 90),
    )
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_id,)))
    candidate = run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_id,)))
    result = ComparisonService(workspace).compare(
        MeasurementCompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate.run_set_id,
            metric="inference.latency",
            unit="ns",
            polarity=MetricPolarity.LOWER_IS_BETTER,
            practical_threshold=0.05,
        )
    )

    mismatches = list(result.comparison.mismatches)
    # Protocol mismatch for model_id
    assert any("inference protocol mismatch" in m and "model.model_id" in m for m in mismatches)
    # Exploratory reasons for undeclared facets
    assert any("inference protocol exploratory" in m for m in mismatches)
    # Generic source-state and cross-treatment validation artifacts do not
    # duplicate the inference protocol's provenance and oracle contracts.
    assert not any("source_state" in m for m in mismatches)
    assert not any("cross-treatment validation outputs" in m for m in mismatches)
    # Missing validation is exploratory
    assert any("lack passing validation" in m for m in mismatches)


def test_within_treatment_protocol_difference_is_invalid(
    tmp_path: Path,
) -> None:
    """Two baseline runs with different model_ids produce an invalidating
    within-treatment mismatch, reported with exact dotted field names."""
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    baseline_run_a = "sha256:" + "c" * 64
    baseline_run_b = "sha256:" + "d" * 64
    # Two baseline runs with different model_ids
    _publish_inference_run(
        workspace,
        run_id=baseline_run_a,
        protocol=_protocol_identity(model_id="model-a"),
        values=(100, 110),
    )
    _publish_inference_run(
        workspace,
        run_id=baseline_run_b,
        protocol=_protocol_identity(model_id="model-b"),
        values=(120, 130),
    )
    candidate_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "e" * 64,
        protocol=_protocol_identity(model_id="model-a"),
        values=(80, 85),
    )
    # Publish trials so multi-run run sets can reference them
    experiment_id = "sha256:" + "0" * 64
    GenerationPublisher(workspace).publish_rows(
        {
            "trials": [
                {
                    "trial_id": f"trial-{run_id}",
                    "experiment_id": experiment_id,
                    "variant_id": "default",
                    "run_id": run_id,
                    "combination_id": "default",
                    "factors_json": "{}",
                    "block_id": f"block-{run_id}",
                    "order_in_block": 0,
                    "parameter_name": None,
                    "parameter_value_int": None,
                    "parameter_value_float": None,
                    "attempt": 1,
                    "outcome": "succeeded",
                    "exclusion_reason": None,
                    "validation_status": "not_requested",
                    "failure_class": "none",
                    "oracle_receipt_json": None,
                }
                for run_id in (baseline_run_a, baseline_run_b, candidate_id)
            ],
        },
        publisher="trials-test",
        publisher_version="1",
        input_run_ids=(baseline_run_a, baseline_run_b, candidate_id),
    )
    from flameox.application import IncludedFreezeRunSetMember

    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(
        FreezeRunMembersRequest(
            members=(
                IncludedFreezeRunSetMember(
                    run_id=baseline_run_a, trial_id=f"trial-{baseline_run_a}"
                ),
                IncludedFreezeRunSetMember(
                    run_id=baseline_run_b, trial_id=f"trial-{baseline_run_b}"
                ),
            )
        )
    )
    candidate = run_sets.freeze(
        FreezeRunMembersRequest(
            members=(
                IncludedFreezeRunSetMember(run_id=candidate_id, trial_id=f"trial-{candidate_id}"),
            )
        )
    )
    result = ComparisonService(workspace).compare(
        MeasurementCompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate.run_set_id,
            metric="inference.latency",
            unit="ns",
            polarity=MetricPolarity.LOWER_IS_BETTER,
            practical_threshold=0.05,
        )
    )

    assert result.comparison.validity is ComparisonValidity.INVALID
    within = [
        m
        for m in result.comparison.mismatches
        if "within-treatment mismatch" in m and "model.model_id" in m
    ]
    assert len(within) >= 1
    assert any("model-a" in m and "model-b" in m for m in within)


def test_malformed_protocol_json_is_invalidating_not_leaking(
    tmp_path: Path,
) -> None:
    """Malformed persisted protocol identity JSON is reported as an
    invalidating compatibility reason, not a leaked ValidationError."""
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    # Publish a run with malformed protocol JSON
    environment = collect_environment()
    run_id = "sha256:" + "f" * 64
    manifest = ImportRunManifest(
        run_id=run_id,
        execution_status=ExecutionStatus.NOT_APPLICABLE,
        capture_status=CaptureStatus.REGISTERED,
        validation_status=ValidationStatus.NOT_REQUESTED,
        environment_id=environment.environment_id,
        collector="inference-replay",
        inference_protocol_identity_id=digest_model("garbage"),
        inference_protocol_identity_json="{not valid json",
    )
    RunStore(workspace).create(manifest)
    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row(manifest)],
            "environments": [environment_row(environment)],
            "measurements": [
                {
                    "measurement_id": f"{run_id}-m{i}",
                    "run_id": run_id,
                    "artifact_id": "sha256:" + "g" * 64,
                    "name": "inference.latency",
                    "value_int": value,
                    "value_float": None,
                    "unit": "ns",
                    "aggregation": "sample",
                    "scope": "process",
                    "trial_id": None,
                    "worker_id": "worker-0",
                    "worker_run_index": 0,
                    "value_index": i,
                    "loop_count": 1,
                    "is_warmup": False,
                    "block_id": f"block-{i}",
                    "variant_id": None,
                    "order_in_block": 0,
                    "phase": "steady_state",
                    "dimensions": {},
                    "evidence_level": "observed",
                }
                for i, value in enumerate((100, 110))
            ],
        },
        publisher="malformed-test",
        publisher_version="1",
        input_run_ids=(run_id,),
    )
    candidate_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "1" * 64,
        protocol=_protocol_identity(),
        values=(80, 85),
    )

    validity = _compare(workspace, run_id, candidate_id)

    assert validity is ComparisonValidity.INVALID


def test_missing_provider_version_is_exploratory_not_failing(
    tmp_path: Path,
) -> None:
    """When the discovered executable has no package metadata, provider_version
    is absent. The comparison is exploratory (undeclared facet), not invalid."""
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    protocol = _protocol_identity()
    # Override provider_version to None (simulating no package metadata)
    protocol_no_version = protocol.model_copy(update={"provider_version": None})
    baseline_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "2" * 64,
        protocol=protocol_no_version,
        values=(100, 110),
    )
    candidate_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "3" * 64,
        protocol=protocol_no_version,
        values=(80, 85),
    )

    validity = _compare(workspace, baseline_id, candidate_id)

    assert validity is ComparisonValidity.EXPLORATORY


def test_managed_server_command_digest_mismatch_is_invalid(
    tmp_path: Path,
) -> None:
    """Differing managed_server_command_digest invalidates the comparison."""
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    digest_a = "sha256:" + "a" * 64
    digest_b = "sha256:" + "b" * 64
    baseline_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "4" * 64,
        protocol=_protocol_identity().model_copy(
            update={
                "server": _protocol_identity().server.model_copy(
                    update={"managed_server_command_digest": digest_a}
                )
            }
        ),
        values=(100, 110),
    )
    candidate_id = _publish_inference_run(
        workspace,
        run_id="sha256:" + "5" * 64,
        protocol=_protocol_identity().model_copy(
            update={
                "server": _protocol_identity().server.model_copy(
                    update={"managed_server_command_digest": digest_b}
                )
            }
        ),
        values=(80, 85),
    )

    validity = _compare(workspace, baseline_id, candidate_id)

    assert validity is ComparisonValidity.INVALID
