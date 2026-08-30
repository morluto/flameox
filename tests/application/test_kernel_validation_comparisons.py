from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from flameox.adapters.kernel_validation import (
    KernelValidationExtractor,
    KernelValidationV2,
)
from flameox.application.comparisons import (
    FreezeRunIdsRequest,
    FreezeRunMembersRequest,
    IncludedFreezeRunSetMember,
    RunSetService,
)
from flameox.application.evidence_lookup import EvidenceLookupService
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.application.kernel_validation_comparisons import (
    KernelMetricChangeKind,
    KernelMetricDirection,
    KernelValidationCompareRequest,
    KernelValidationComparisonProtocol,
    KernelValidationComparisonService,
    KernelValidationInputIdentity,
    KernelValidationMetricSelector,
)
from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactKind,
    ComparisonDecision,
    ComparisonValidity,
    EvidenceReferenceType,
    Experiment,
    ExperimentRole,
    MetricPolarity,
    MetricSource,
    digest_model,
)
from flameox.evidence import GenerationPublisher
from flameox.storage import ControlRecordStore, Workspace

pytestmark = [pytest.mark.integration, pytest.mark.serial]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "kernel_validation" / "pass.json"


def _payload() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE.read_text()))


def _finite_document(
    value: float,
    *,
    threshold: float = 0.01,
    reference_identity: str | None = None,
    coverage_complete: bool = True,
) -> dict[str, Any]:
    payload = _payload()
    if reference_identity is not None:
        payload["reference"]["identity"] = reference_identity
    metric = {
        "name": "max_abs_error",
        "value": {"kind": "finite", "value": value},
        "comparator": "<=",
        "threshold": threshold,
        "unit": "absolute",
        "status": "pass" if value <= threshold else "fail",
    }
    output = payload["cases"][0]["outputs"][0]
    output["metrics"] = [metric]
    if value > threshold:
        output["status"] = "fail"
        output["representative_failures"] = [{"coordinates": [0], "expected": 0.0, "actual": value}]
        payload["cases"][0]["status"] = "fail"
        payload["status"] = "fail"
    if not coverage_complete:
        payload["coverage_complete"] = False
        payload["status"] = "inconclusive"
        payload["limitations"] = ["producer sampled only part of the declared population"]
    return payload


def _import_document(
    workspace: Workspace,
    root: Path,
    name: str,
    payload: dict[str, Any],
) -> str:
    document = KernelValidationV2.model_validate(payload)
    path = root / f"{name}.json"
    path.write_text(document.model_dump_json())
    run_id = (
        ImportService(workspace)
        .import_artifact(ImportArtifactRequest(path=path, kind=ArtifactKind.VALIDATION_OUTPUT))
        .run.run_id
    )
    KernelValidationExtractor(workspace).extract(run_id)
    return run_id


def _request(
    workspace: Workspace,
    baseline_run_id: str,
    candidate_run_id: str,
) -> KernelValidationCompareRequest:
    run_sets = RunSetService(workspace)
    baseline = run_sets.freeze(FreezeRunIdsRequest(run_ids=(baseline_run_id,)))
    candidate = run_sets.freeze(FreezeRunIdsRequest(run_ids=(candidate_run_id,)))
    return KernelValidationCompareRequest(
        baseline_run_set_id=baseline.run_set_id,
        candidate_run_set_id=candidate.run_set_id,
    )


def test_kernel_validation_compares_both_pass_values_below_a_loose_threshold(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline = _import_document(workspace, tmp_path, "baseline", _finite_document(0.009))
    candidate = _import_document(workspace, tmp_path, "candidate", _finite_document(0.004))

    result = KernelValidationComparisonService(workspace).compare(
        _request(workspace, baseline, candidate)
    )

    assert result.comparison.validity is ComparisonValidity.EXPLORATORY
    assert len(result.comparison.pairs) == 1
    pair = result.comparison.pairs[0]
    assert pair.status_transition == "pass_to_pass"
    assert pair.signed_change == -0.004999999999999999
    assert pair.relative_change_percent == pytest.approx(-55.55555555555556)
    assert pair.direction is KernelMetricDirection.IMPROVED


def test_kernel_validation_exposes_within_tolerance_regression_and_threshold_crossing(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline = _import_document(workspace, tmp_path, "baseline", _finite_document(0.001))
    candidate = _import_document(workspace, tmp_path, "candidate", _finite_document(0.005))
    within = (
        KernelValidationComparisonService(workspace)
        .compare(_request(workspace, baseline, candidate))
        .comparison.pairs[0]
    )
    assert within.status_transition == "pass_to_pass"
    assert within.direction is KernelMetricDirection.REGRESSED

    crossed_workspace = Workspace.initialize(tmp_path / "crossed")
    crossed_baseline = _import_document(
        crossed_workspace,
        tmp_path / "crossed",
        "baseline",
        _finite_document(0.005),
    )
    crossed_candidate = _import_document(
        crossed_workspace,
        tmp_path / "crossed",
        "candidate",
        _finite_document(0.02),
    )
    crossed = (
        KernelValidationComparisonService(crossed_workspace)
        .compare(_request(crossed_workspace, crossed_baseline, crossed_candidate))
        .comparison.pairs[0]
    )
    assert crossed.status_transition == "pass_to_fail"
    assert crossed.direction is KernelMetricDirection.REGRESSED


def test_kernel_validation_zero_baseline_has_explicit_percentage_semantics(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline = _import_document(workspace, tmp_path, "baseline", _finite_document(0.0))
    candidate = _import_document(workspace, tmp_path, "candidate", _finite_document(0.001))

    pair = (
        KernelValidationComparisonService(workspace)
        .compare(_request(workspace, baseline, candidate))
        .comparison.pairs[0]
    )

    assert pair.signed_change == 0.001
    assert pair.relative_change_percent is None
    assert pair.relative_change_unavailable_reason == "baseline is zero"


def test_kernel_validation_reference_mismatch_and_incomplete_coverage_are_not_dropped(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline = _import_document(workspace, tmp_path, "baseline", _finite_document(0.005))
    candidate_payload = _finite_document(
        0.004,
        reference_identity="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        coverage_complete=False,
    )
    candidate = _import_document(workspace, tmp_path, "candidate", candidate_payload)

    comparison = (
        KernelValidationComparisonService(workspace)
        .compare(_request(workspace, baseline, candidate))
        .comparison
    )

    assert not comparison.pairs
    assert comparison.validity is ComparisonValidity.INVALID
    assert comparison.compatibility_mismatches[0].reasons == ("reference differs",)
    assert any("incomplete coverage" in reason for reason in comparison.mismatches)


def test_kernel_validation_missing_metric_output_is_explicit_not_an_exception(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline = _import_document(workspace, tmp_path, "baseline", _finite_document(0.005))
    candidate_payload = _payload()
    output = candidate_payload["cases"][0]["outputs"][0]
    output.update(
        status="inconclusive",
        metrics=[],
        limitations=["metric provider unavailable"],
    )
    candidate_payload["cases"][0]["status"] = "inconclusive"
    candidate_payload["status"] = "inconclusive"
    candidate = _import_document(workspace, tmp_path, "candidate", candidate_payload)

    comparison = (
        KernelValidationComparisonService(workspace)
        .compare(_request(workspace, baseline, candidate))
        .comparison
    )

    assert comparison.validity is ComparisonValidity.INVALID
    assert len(comparison.baseline_only) == 1
    candidate_artifact_id = KernelValidationExtractor(workspace).extract(candidate).artifact_id
    assert set(comparison.input_artifact_ids) == {
        comparison.baseline_only[0].artifact_id,
        candidate_artifact_id,
    }
    assert any("has no metric observations" in reason for reason in comparison.mismatches)


def test_kernel_validation_orders_exact_psnr_above_finite_without_a_sentinel(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_payload = _payload()
    candidate_payload = deepcopy(baseline_payload)
    profile = {
        "identity_quality": "exact",
        "data_range": 1.0,
        "log_base": 10,
        "reduction": "mean_squared_error",
        "zero_mse_convention": "positive_infinity",
    }
    for payload, mse, psnr in (
        (baseline_payload, 1e-8, {"kind": "finite", "value": 80.0}),
        (
            candidate_payload,
            0.0,
            {
                "kind": "positive_infinity",
                "reason": "zero_mse_exact_agreement",
            },
        ),
    ):
        payload["cases"][0]["outputs"][0]["metrics"] = [
            {
                "name": "mse",
                "value": {"kind": "finite", "value": mse},
                "comparator": "<=",
                "threshold": 1e-7,
                "unit": "squared_error",
                "status": "pass",
            },
            {
                "name": "psnr",
                "value": psnr,
                "comparator": ">=",
                "threshold": 60.0,
                "unit": "dB",
                "profile": profile,
                "status": "pass",
            },
        ]
    baseline = _import_document(workspace, tmp_path, "baseline", baseline_payload)
    candidate = _import_document(workspace, tmp_path, "candidate", candidate_payload)

    pairs = (
        KernelValidationComparisonService(workspace)
        .compare(_request(workspace, baseline, candidate))
        .comparison.pairs
    )
    psnr_pair = next(item for item in pairs if item.metric_name == "psnr")

    assert psnr_pair.change_kind is KernelMetricChangeKind.POSITIVE_INFINITY
    assert psnr_pair.signed_change is None
    assert psnr_pair.direction is KernelMetricDirection.IMPROVED


def test_recorded_kernel_validation_comparison_preserves_pairs_and_provenance(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline = _import_document(workspace, tmp_path, "baseline", _finite_document(0.009))
    candidate = _import_document(workspace, tmp_path, "candidate", _finite_document(0.004))

    result = KernelValidationComparisonService(workspace).record(
        _request(workspace, baseline, candidate)
    )

    assert result.analysis is not None
    assert result.materialized_commit_id is not None
    lookup = EvidenceLookupService(workspace).get(
        EvidenceReferenceType.COMPARISON,
        result.comparison.comparison_id,
    )
    assert lookup.data["metric_source"] == "kernel_validation"
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute(
            "SELECT count(*) FROM kernel_validation_comparisons WHERE comparison_id = ?",
            (result.comparison.comparison_id,),
        ).fetchone() == (1,)
        assert snapshot.execute(
            "SELECT signed_change, direction FROM kernel_validation_comparison_pairs "
            "WHERE comparison_id = ?",
            (result.comparison.comparison_id,),
        ).fetchone() == (-0.004999999999999999, "improved")


def test_preregistered_randomized_blocks_produce_confirmatory_uncertainty(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_runs: list[str] = []
    candidate_runs: list[str] = []
    for block, baseline_value in enumerate((0.009, 0.008, 0.007)):
        baseline_runs.append(
            _import_document(
                workspace,
                tmp_path,
                f"baseline-{block}",
                _finite_document(baseline_value),
            )
        )
        candidate_runs.append(
            _import_document(
                workspace,
                tmp_path,
                f"candidate-{block}",
                _finite_document(baseline_value - 0.005),
            )
        )
    protocol = KernelValidationComparisonProtocol(
        reference_name="torch.reference",
        reference_version="2.9",
        reference_identity=(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ),
        case_population=(
            KernelValidationMetricSelector(
                case_id="square-fp32-128",
                dimensions={"size": 128, "transposed": False},
                inputs={
                    "left": KernelValidationInputIdentity(
                        dtype="float32",
                        shape=(128, 128),
                        role="input",
                    )
                },
                seed=42,
                output_name="result",
                output_dtype="float32",
                output_shape=(128, 128),
                metric_name="max_abs_error",
                unit="absolute",
                comparator="<=",
                threshold=0.01,
                device="cuda:0-sm86",
            ),
        ),
    )
    experiment = Experiment(
        experiment_id="kernel-correctness-experiment",
        investigation_id="kernel-correctness-investigation",
        recipe="kernel-validation-comparison",
        recipe_version="1",
        workload_definition_id=digest_model({"workload": "kernel"}),
        experiment_design_id=digest_model({"design": "randomized-block"}),
        measurement_protocol_id=digest_model({"measurement": "kernel-validation-v2"}),
        validation_spec_id=protocol.protocol_id,
        primary_metric="max_abs_error",
        metric_source=MetricSource.KERNEL_VALIDATION,
        primary_metric_unit="absolute",
        polarity=MetricPolarity.LOWER_IS_BETTER,
        estimand="median_blockwise_median_difference",
        practical_threshold=0.001,
        confidence_level=0.95,
        stopping_rule={"fixed_blocks": 3},
        random_seed=7,
        role=ExperimentRole.CONFIRMATORY,
    )
    ControlRecordStore(
        workspace,
        kind="experiments",
        model=Experiment,
        id_field="experiment_id",
    ).create(experiment)
    trial_rows: list[dict[str, object]] = []
    for treatment, variant_id, run_ids in (
        ("baseline", "baseline-variant", baseline_runs),
        ("candidate", "candidate-variant", candidate_runs),
    ):
        for block, run_id in enumerate(run_ids):
            trial_rows.append(
                {
                    "trial_id": f"{treatment}-trial-{block}",
                    "experiment_id": experiment.experiment_id,
                    "variant_id": variant_id,
                    "run_id": run_id,
                    "combination_id": digest_model({"treatment": treatment, "block": block}),
                    "factors_json": "{}",
                    "block_id": f"block-{block}",
                    "order_in_block": 0 if treatment == "baseline" else 1,
                    "parameter_name": None,
                    "parameter_value_int": None,
                    "parameter_value_float": None,
                    "attempt": 1,
                    "outcome": "succeeded",
                    "exclusion_reason": None,
                    "validation_status": "passed",
                    "failure_class": "none",
                    "oracle_receipt_json": None,
                    "oracle_receipt_artifact_id": None,
                }
            )
    GenerationPublisher(workspace).publish_rows(
        {"trials": trial_rows},
        publisher="kernel-validation-comparison-fixture",
        publisher_version="1",
        input_run_ids=tuple((*baseline_runs, *candidate_runs)),
    )
    run_sets = RunSetService(workspace)
    baseline_set = run_sets.freeze(
        FreezeRunMembersRequest(
            selection={
                "experiment_id": experiment.experiment_id,
                "variant_id": "baseline-variant",
            },
            members=tuple(
                IncludedFreezeRunSetMember(
                    run_id=run_id,
                    trial_id=f"baseline-trial-{block}",
                )
                for block, run_id in enumerate(baseline_runs)
            ),
        )
    )
    candidate_set = run_sets.freeze(
        FreezeRunMembersRequest(
            selection={
                "experiment_id": experiment.experiment_id,
                "variant_id": "candidate-variant",
            },
            members=tuple(
                IncludedFreezeRunSetMember(
                    run_id=run_id,
                    trial_id=f"candidate-trial-{block}",
                )
                for block, run_id in enumerate(candidate_runs)
            ),
        )
    )

    comparison = (
        KernelValidationComparisonService(workspace)
        .compare(
            KernelValidationCompareRequest(
                baseline_run_set_id=baseline_set.run_set_id,
                candidate_run_set_id=candidate_set.run_set_id,
                experiment_id=experiment.experiment_id,
                protocol=protocol,
            )
        )
        .comparison
    )

    assert comparison.validity is ComparisonValidity.VALID
    assert comparison.decision is ComparisonDecision.MEANINGFUL_IMPROVEMENT
    assert comparison.complete_pair_n == 3
    assert comparison.estimate == pytest.approx(-0.005)
    assert comparison.confidence_interval is not None
    assert comparison.confidence_interval.low == pytest.approx(-0.005)
    assert comparison.confidence_interval.high == pytest.approx(-0.005)

    changed_protocol = protocol.validated_copy(
        update={
            "reference_identity": (
                "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            )
        }
    )
    invalid = (
        KernelValidationComparisonService(workspace)
        .compare(
            KernelValidationCompareRequest(
                baseline_run_set_id=baseline_set.run_set_id,
                candidate_run_set_id=candidate_set.run_set_id,
                experiment_id=experiment.experiment_id,
                protocol=changed_protocol,
            )
        )
        .comparison
    )
    assert invalid.validity is ComparisonValidity.INVALID
    assert invalid.decision is ComparisonDecision.INCONCLUSIVE
    assert any("validation_spec_id" in reason for reason in invalid.mismatches)
