from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from flameox.action_graph import ActionId, ManualAction
from flameox.analysis import RecipeService, ScalingAnalysisResult
from flameox.catalog import Catalog
from flameox.domain import canonical_json
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace

pytestmark = pytest.mark.integration

FIXTURE_CREATED_AT = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)


def _scaling_variant_row() -> dict[str, object]:
    return {
        "variant_id": "variant-baseline",
        "experiment_id": "scaling-experiment",
        "name": "baseline",
        "identity_quality": "heterogeneous",
        "source_state_id": "source",
        "workload_instance_id": None,
        "environment_id": "stable-environment",
        "source_state_ids": ["source"],
        "workload_instance_ids": [
            "workload-32768",
            "workload-32768.0",
            "workload-65536",
            "workload-131072",
        ],
        "environment_ids": ["stable-environment"],
        "combination_ids": [
            "combination-32768",
            "combination-32768.0",
            "combination-65536",
            "combination-131072",
        ],
        "environment_requirements_json": "{}",
        "parameters_json": '{"implementation":"baseline"}',
        "varying_factors_json": '{"length":[32768,65536,131072]}',
        "limitations": [],
    }


def _publish_scaling_fixture(
    workspace: Workspace,
    trials: tuple[tuple[str, str, int | float | None, bool, str], ...],
) -> None:
    variants = tuple(sorted({variant for variant, *_ in trials}))
    variant_rows = []
    for variant in variants:
        row = _scaling_variant_row()
        row.update(
            {
                "variant_id": f"variant-{variant}",
                "name": variant,
                "parameters_json": canonical_json({"implementation": variant}),
                "workload_instance_ids": [f"workload-{variant}"],
                "combination_ids": [f"combination-{variant}"],
            }
        )
        variant_rows.append(row)
    run_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    measurement_rows: list[dict[str, object]] = []
    for index, (variant, block_id, input_value, has_measurement, outcome) in enumerate(trials):
        run_id = f"run-{index}"
        trial_id = f"trial-{index}"
        is_integer = type(input_value) is int
        factors: dict[str, str | int | float] = {"implementation": variant}
        if input_value is not None:
            factors["length"] = input_value
        run_rows.append(
            {
                "run_id": run_id,
                "created_at": FIXTURE_CREATED_AT,
                "run_type": "execution",
                "execution_status": "succeeded",
                "capture_status": "complete",
                "validation_status": "passed",
                "workload_definition_id": "workload",
                "workload_instance_id": f"workload-{variant}-{index}",
                "measurement_protocol_id": "protocol",
                "environment_id": "stable-environment",
                "source_state_id": "source",
                "adapter": "pyperf",
                "adapter_version": "1",
                "run_semantic_id": "sha256:" + "f" * 64,
                "exit_code": 0,
                "wall_time_ns": int(input_value) if input_value is not None else 1,
            }
        )
        trial_rows.append(
            {
                "trial_id": trial_id,
                "experiment_id": "scaling-experiment",
                "variant_id": f"variant-{variant}",
                "run_id": run_id,
                "combination_id": f"combination-{variant}-{input_value}",
                "factors_json": canonical_json(factors),
                "block_id": block_id,
                "order_in_block": index,
                "parameter_name": "length" if input_value is not None else None,
                "parameter_value_int": input_value if is_integer else None,
                "parameter_value_float": (
                    input_value if input_value is not None and not is_integer else None
                ),
                "attempt": 1,
                "outcome": outcome,
                "exclusion_reason": None,
                "validation_status": "passed",
                "failure_class": "none" if outcome == "succeeded" else "process_failure",
            }
        )
        if has_measurement:
            measurement_rows.append(
                {
                    "measurement_id": f"measurement-{index}",
                    "run_id": run_id,
                    "artifact_id": None,
                    "name": "scan.time",
                    "value_int": index + 1,
                    "value_float": None,
                    "unit": "ns",
                    "aggregation": "single",
                    "scope": "process",
                    "trial_id": trial_id,
                    "worker_id": None,
                    "worker_run_index": None,
                    "value_index": 0,
                    "loop_count": 1,
                    "is_warmup": False,
                    "block_id": block_id,
                    "variant_id": f"variant-{variant}",
                    "order_in_block": index,
                    "phase": None,
                    "dimensions": {},
                    "evidence_level": "observed",
                }
            )
    GenerationPublisher(workspace).publish_rows(
        {
            "experiments": [
                {
                    "experiment_id": "scaling-experiment",
                    "investigation_id": "investigation",
                    "hypothesis_id": None,
                    "recipe": "scaling",
                    "recipe_version": "1",
                    "workload_definition_id": "workload",
                    "experiment_design_id": "design",
                    "measurement_protocol_id": "protocol",
                    "validation_spec_id": None,
                    "primary_metric": "scan.time",
                    "polarity": "lower_is_better",
                    "estimand": "median",
                    "practical_threshold": 0.01,
                    "confidence_level": 0.95,
                    "stopping_rule_json": "{}",
                    "random_seed": 1,
                    "role": "exploratory",
                    "created_at": FIXTURE_CREATED_AT,
                }
            ],
            "variants": variant_rows,
            "trials": trial_rows,
            "runs": run_rows,
            "measurements": measurement_rows,
        },
        publisher="scaling-fixture",
        publisher_version="1",
    )
    Catalog(workspace).rebuild()


def _requirements(result: ScalingAnalysisResult) -> set[str]:
    return {item.kind.value for item in result.sufficiency.missing_requirements}


def test_scaling_reports_dispersion_models_and_supported_range(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    now = FIXTURE_CREATED_AT
    variants = (32_768, 32_768.0, 65_536, 131_072)
    run_rows = []
    trial_rows = []
    measurement_rows: list[dict[str, object]] = []
    frame_measurement_rows = []
    variant_rows = [_scaling_variant_row()]
    for input_value in variants:
        is_integer = type(input_value) is int
        variant_id = "variant-baseline"
        for block in range(3):
            run_id = f"run-{input_value}-{block}"
            trial_id = f"trial-{input_value}-{block}"
            run_rows.append(
                {
                    "run_id": run_id,
                    "created_at": now,
                    "run_type": "execution",
                    "execution_status": "succeeded",
                    "capture_status": "complete",
                    "validation_status": "passed",
                    "workload_definition_id": "workload",
                    "workload_instance_id": f"workload-{input_value}",
                    "measurement_protocol_id": "protocol",
                    "environment_id": "stable-environment",
                    "source_state_id": "source",
                    "adapter": "pyperf",
                    "adapter_version": "1",
                    "run_semantic_id": "sha256:" + "f" * 64,
                    "exit_code": 0,
                    "wall_time_ns": int(input_value * 2),
                }
            )
            trial_rows.append(
                {
                    "trial_id": trial_id,
                    "experiment_id": "scaling-experiment",
                    "variant_id": variant_id,
                    "run_id": run_id,
                    "combination_id": f"combination-{input_value}",
                    "factors_json": canonical_json(
                        {"implementation": "baseline", "length": input_value}
                    ),
                    "block_id": f"block-{input_value}-{block}",
                    "order_in_block": block,
                    "parameter_name": "length",
                    "parameter_value_int": input_value if is_integer else None,
                    "parameter_value_float": None if is_integer else input_value,
                    "attempt": 1,
                    "outcome": "succeeded",
                    "exclusion_reason": None,
                    "validation_status": "passed",
                    "failure_class": "none",
                }
            )
            measurement_rows.append(
                {
                    "measurement_id": f"measurement-{input_value}-{block}",
                    "run_id": run_id,
                    "artifact_id": None,
                    "name": "scan.time",
                    "value_int": int(input_value * 2 + block) if is_integer else None,
                    "value_float": None if is_integer else input_value * 2 + block,
                    "unit": "ns",
                    "aggregation": "single",
                    "scope": "process",
                    "trial_id": trial_id,
                    "worker_id": None,
                    "worker_run_index": None,
                    "value_index": 0,
                    "loop_count": 1,
                    "is_warmup": False,
                    "block_id": f"block-{input_value}-{block}",
                    "variant_id": variant_id,
                    "order_in_block": block,
                    "phase": None,
                    "dimensions": {},
                    "evidence_level": "observed",
                }
            )
            frame_measurement_rows.append(
                {
                    "run_id": run_id,
                    "artifact_id": "sha256:" + "a" * 64,
                    "frame_id": "reverse-scan-frame",
                    "metric": "cpu.time",
                    "self_value": input_value + block,
                    "inclusive_value": input_value + block,
                    "unit": "ns",
                    "sample_count": 1,
                    "thread_name": "main",
                    "process_name": "python",
                    "phase": "steady_state",
                }
            )
    GenerationPublisher(workspace).publish_rows(
        {
            "experiments": [
                {
                    "experiment_id": "scaling-experiment",
                    "investigation_id": "investigation",
                    "hypothesis_id": None,
                    "recipe": "scaling",
                    "recipe_version": "1",
                    "workload_definition_id": "workload",
                    "experiment_design_id": "design",
                    "measurement_protocol_id": "protocol",
                    "validation_spec_id": None,
                    "primary_metric": "scan.time",
                    "polarity": "lower_is_better",
                    "estimand": "median",
                    "practical_threshold": 0.01,
                    "confidence_level": 0.95,
                    "stopping_rule_json": "{}",
                    "random_seed": 1,
                    "role": "exploratory",
                    "created_at": now,
                }
            ],
            "variants": variant_rows,
            "trials": trial_rows,
            "runs": run_rows,
            "measurements": measurement_rows,
            "frames": [
                {
                    "frame_id": "reverse-scan-frame",
                    "language": "Python",
                    "function": "reverse_scan",
                    "module": "fixture",
                    "file": "scan.py",
                    "line": 10,
                    "column": None,
                    "address": None,
                    "build_id": None,
                    "module_relative_address": None,
                    "inline_chain_id": None,
                    "source_state_id": "source",
                    "artifact_id": "sha256:" + "a" * 64,
                    "inlined": False,
                    "symbolization": "complete",
                }
            ],
            "frame_measurements": frame_measurement_rows,
        },
        publisher="scaling-fixture",
        publisher_version="1",
    )
    Catalog(workspace).rebuild()

    result = RecipeService(workspace).scaling("scaling-experiment")

    assert result.attempted_trials == 12
    assert result.complete_blocks == 12
    assert result.fits == ()
    assert result.sufficiency.status == "incompatible"
    assert _requirements(result) == {"distinct_input_values", "consistent_input_kind"}
    assert (
        "Fits were excluded for variants with mixed integer and floating scaling inputs: baseline."
        in result.warnings
    )
    assert len(result.trials) == 12
    assert {(point.input_value, point.input_kind) for point in result.points} == {
        (32_768.0, "integer"),
        (32_768.0, "floating"),
        (65_536.0, "integer"),
        (131_072.0, "integer"),
    }
    assert all(point.confidence_interval is not None for point in result.points)
    assert result.correlated_hotspots[0].function == "reverse_scan"
    assert result.correlated_hotspots[0].spearman_rho > 0.9
    assert result.correlated_hotspots[0].multiplicity_method == "benjamini-hochberg-fdr"
    assert result.correlated_hotspots[0].tested_hypothesis_count == len(result.correlated_hotspots)
    assert result.correlated_hotspots[0].adjusted_p_value >= result.correlated_hotspots[0].p_value
    payload = result.model_dump(mode="json")
    assert type(result).model_validate(payload) == result


def test_scaling_correlated_hotspots_empty_when_all_filtered(tmp_path: Path) -> None:
    """Regression test for the tested_hypothesis_count invariant.

    When the scaling recipe produces no correlated hotspots (e.g. because the
    frame measurements do not correlate with the scaling axis), the result must
    contain zero rows rather than rows carrying a misleading
    ``tested_hypothesis_count=0`` with a declared multiplicity method.
    """
    workspace = Workspace.initialize(tmp_path)
    now = FIXTURE_CREATED_AT
    # A single scaling point with a single block: no correlation can be computed
    # (n < 2 after the rank filter), so correlated_hotspots must be empty.
    run_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    measurement_rows: list[dict[str, object]] = []
    frame_measurement_rows: list[dict[str, object]] = []
    variant_rows = [_scaling_variant_row()]
    for input_value in (32_768,):
        for block in range(1):
            run_id = f"run-{input_value}-{block}"
            trial_id = f"trial-{input_value}-{block}"
            run_rows.append(
                {
                    "run_id": run_id,
                    "created_at": now,
                    "run_type": "execution",
                    "execution_status": "succeeded",
                    "capture_status": "complete",
                    "validation_status": "passed",
                    "workload_definition_id": "workload",
                    "workload_instance_id": f"workload-{input_value}",
                    "measurement_protocol_id": "protocol",
                    "environment_id": "stable-environment",
                    "source_state_id": "source",
                    "adapter": "pyperf",
                    "adapter_version": "1",
                    "run_semantic_id": "sha256:" + "f" * 64,
                    "exit_code": 0,
                    "wall_time_ns": input_value * 2,
                }
            )
            trial_rows.append(
                {
                    "trial_id": trial_id,
                    "experiment_id": "scaling-experiment",
                    "variant_id": "variant-baseline",
                    "run_id": run_id,
                    "combination_id": f"combination-{input_value}",
                    "factors_json": canonical_json(
                        {"implementation": "baseline", "length": input_value}
                    ),
                    "block_id": f"block-{input_value}-{block}",
                    "order_in_block": block,
                    "parameter_name": "length",
                    "parameter_value_int": input_value,
                    "parameter_value_float": None,
                    "attempt": 1,
                    "outcome": "succeeded",
                    "exclusion_reason": None,
                    "validation_status": "passed",
                    "failure_class": "none",
                }
            )
            measurement_rows.append(
                {
                    "measurement_id": f"measurement-{input_value}-{block}",
                    "run_id": run_id,
                    "artifact_id": None,
                    "name": "scan.time",
                    "value_int": input_value * 2 + block,
                    "value_float": None,
                    "unit": "ns",
                    "aggregation": "single",
                    "scope": "process",
                    "trial_id": trial_id,
                    "worker_id": None,
                    "worker_run_index": None,
                    "value_index": 0,
                    "loop_count": 1,
                    "is_warmup": False,
                    "block_id": f"block-{input_value}-{block}",
                    "variant_id": "variant-baseline",
                    "order_in_block": block,
                    "phase": None,
                    "dimensions": {},
                    "evidence_level": "observed",
                }
            )
            frame_measurement_rows.append(
                {
                    "run_id": run_id,
                    "artifact_id": "sha256:" + "a" * 64,
                    "frame_id": "reverse-scan-frame",
                    "metric": "cpu.time",
                    "self_value": input_value + block,
                    "inclusive_value": input_value + block,
                    "unit": "ns",
                    "sample_count": 1,
                    "thread_name": "main",
                    "process_name": "python",
                    "phase": "steady_state",
                }
            )
    GenerationPublisher(workspace).publish_rows(
        {
            "experiments": [
                {
                    "experiment_id": "scaling-experiment",
                    "investigation_id": "investigation",
                    "hypothesis_id": None,
                    "recipe": "scaling",
                    "recipe_version": "1",
                    "workload_definition_id": "workload",
                    "experiment_design_id": "design",
                    "measurement_protocol_id": "protocol",
                    "validation_spec_id": None,
                    "primary_metric": "scan.time",
                    "polarity": "lower_is_befter",
                    "estimand": "median",
                    "practical_threshold": 0.01,
                    "confidence_level": 0.95,
                    "stopping_rule_json": "{}",
                    "random_seed": 1,
                    "role": "exploratory",
                    "created_at": now,
                }
            ],
            "variants": variant_rows,
            "trials": trial_rows,
            "runs": run_rows,
            "measurements": measurement_rows,
            "frames": [
                {
                    "frame_id": "reverse-scan-frame",
                    "language": "Python",
                    "function": "reverse_scan",
                    "module": "fixture",
                    "file": "scan.py",
                    "line": 10,
                    "column": None,
                    "address": None,
                    "build_id": None,
                    "module_relative_address": None,
                    "inline_chain_id": None,
                    "source_state_id": "source",
                    "artifact_id": "sha256:" + "a" * 64,
                    "inlined": False,
                    "symbolization": "complete",
                }
            ],
            "frame_measurements": frame_measurement_rows,
        },
        publisher="scaling-fixture",
        publisher_version="1",
    )
    Catalog(workspace).rebuild()

    result = RecipeService(workspace).scaling("scaling-experiment")
    assert result.correlated_hotspots == ()


def test_scaling_reports_missing_numeric_input_as_incompatible(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _publish_scaling_fixture(
        workspace,
        (
            ("baseline", "block-1", None, True, "succeeded"),
            ("baseline", "block-2", None, True, "succeeded"),
        ),
    )

    result = RecipeService(workspace).scaling("scaling-experiment")

    assert result.evidence.status == "available"
    assert result.sufficiency.status == "incompatible"
    assert _requirements(result) == {"numeric_input_factor"}
    assert isinstance(result.sufficiency.next_action, ManualAction)
    assert result.sufficiency.next_action.suggested_action is ActionId.GET_DECLARED_WORKFLOW
    assert result.sufficiency.next_action.missing_arguments == ("name",)
    assert result.experiment_id in result.sufficiency.next_action.instruction
    assert "Do not extend" in result.sufficiency.next_action.instruction


def test_scaling_reports_too_few_input_values(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _publish_scaling_fixture(
        workspace,
        tuple(
            ("baseline", f"length-{value}-block-{block}", value, True, "succeeded")
            for value in (1, 2, 3)
            for block in (1, 2)
        ),
    )

    result = RecipeService(workspace).scaling("scaling-experiment")

    assert result.evidence.status == "available"
    assert result.sufficiency.status == "insufficient"
    requirement = next(
        item
        for item in result.sufficiency.missing_requirements
        if item.kind.value == "distinct_input_values"
    )
    assert (requirement.variants, requirement.observed, requirement.required) == (
        ("baseline",),
        3,
        4,
    )
    assert isinstance(result.sufficiency.next_action, ManualAction)
    assert "replacement" in result.sufficiency.next_action.instruction


def test_scaling_reports_incomplete_blocks_without_marking_rows_unavailable(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    trials = [
        (variant, f"length-{value}-block-{block}", value, True, "succeeded")
        for value in (1, 2, 3, 4)
        for block in (1, 2)
        for variant in ("baseline", "candidate")
    ]
    trials[-1] = ("candidate", "length-4-block-2", 4, False, "failed")
    _publish_scaling_fixture(workspace, tuple(trials))

    result = RecipeService(workspace).scaling("scaling-experiment")

    assert result.evidence.status == "available"
    assert result.sufficiency.status == "insufficient"
    requirement = next(
        item
        for item in result.sufficiency.missing_requirements
        if item.kind.value == "complete_blocks"
    )
    assert (requirement.observed, requirement.required) == (7, 8)


def test_scaling_routes_missing_primary_measurements_to_protocol_repair(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    trials = tuple(
        (
            "baseline",
            f"length-{value}-block-{block}",
            value,
            not (value == 1 and block == 1),
            "succeeded",
        )
        for value in (1, 2, 3, 4)
        for block in (1, 2)
    )
    _publish_scaling_fixture(workspace, trials)

    result = RecipeService(workspace).scaling("scaling-experiment")

    assert result.evidence.status == "partial"
    assert result.sufficiency.status == "incompatible"
    assert "primary_measurements" in _requirements(result)
    assert isinstance(result.sufficiency.next_action, ManualAction)
    assert "Do not extend" in result.sufficiency.next_action.instruction


def test_scaling_reports_insufficient_replication(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _publish_scaling_fixture(
        workspace,
        tuple(
            ("baseline", f"length-{value}-block-1", value, True, "succeeded")
            for value in (1, 2, 3, 4)
        ),
    )

    result = RecipeService(workspace).scaling("scaling-experiment")

    assert result.sufficiency.status == "insufficient"
    requirement = next(
        item for item in result.sufficiency.missing_requirements if item.kind.value == "replication"
    )
    assert (requirement.observed, requirement.required) == (1, 2)


def test_scaling_marks_fit_ready_evidence_sufficient(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _publish_scaling_fixture(
        workspace,
        tuple(
            ("baseline", f"length-{value}-block-{block}", value, True, "succeeded")
            for value in (1, 2, 3, 4)
            for block in (1, 2)
        ),
    )

    result = RecipeService(workspace).scaling("scaling-experiment")

    assert result.evidence.status == "available"
    assert result.sufficiency.status == "sufficient"
    assert result.sufficiency.missing_requirements == ()
    assert result.sufficiency.next_action is None
