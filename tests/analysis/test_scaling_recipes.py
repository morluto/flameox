from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.analysis import RecipeService
from flameox.catalog import Catalog
from flameox.domain import canonical_json
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace

FIXTURE_CREATED_AT = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)


def test_scaling_reports_dispersion_models_and_supported_range(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    now = FIXTURE_CREATED_AT
    variants = (32_768, 65_536, 131_072)
    run_rows = []
    trial_rows = []
    measurement_rows: list[dict[str, object]] = []
    frame_measurement_rows = []
    variant_rows = [
        {
            "variant_id": "variant-baseline",
            "experiment_id": "scaling-experiment",
            "name": "baseline",
            "source_state_id": "source",
            "workload_instance_id": "workload-family",
            "environment_requirements_json": "{}",
            "parameters_json": '{"implementation":"baseline"}',
        }
    ]
    for input_value in variants:
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
                    "collector": "pyperf",
                    "collector_version": "1",
                    "exit_code": 0,
                    "wall_time_ns": input_value * 2,
                    "manifest_path": f"runs/{run_id}/manifest.json",
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

    assert result.attempted_trials == 9
    assert result.complete_blocks == 9
    assert result.environment_stable
    assert any(fit.model == "linear" for fit in result.fits)
    assert {fit.variant for fit in result.fits} == {"baseline"}
    assert {fit.supported_min for fit in result.fits} == {32_768}
    assert {fit.supported_max for fit in result.fits} == {131_072}
    assert len(result.trials) == 9
    assert all(point.confidence_low is not None for point in result.points)
    assert result.correlated_hotspots[0].function == "reverse_scan"
    assert result.correlated_hotspots[0].spearman_rho > 0.9
    assert result.correlated_hotspots[0].multiplicity_method == "benjamini-hochberg-fdr"
    assert result.correlated_hotspots[0].tested_hypothesis_count == len(result.correlated_hotspots)
    assert result.correlated_hotspots[0].adjusted_p_value >= result.correlated_hotspots[0].p_value
    payload = result.model_dump(mode="json")
    assert type(result).model_validate(payload) == result
    with pytest.raises(ValidationError, match="environment stability must derive"):
        type(result).model_validate({**payload, "environment_stable": False})


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
    variant_rows = [
        {
            "variant_id": "variant-baseline",
            "experiment_id": "scaling-experiment",
            "name": "baseline",
            "source_state_id": "source",
            "workload_instance_id": "workload-family",
            "environment_requirements_json": "{}",
            "parameters_json": '{"implementation":"baseline"}',
        }
    ]
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
                    "collector": "pyperf",
                    "collector_version": "1",
                    "exit_code": 0,
                    "wall_time_ns": input_value * 2,
                    "manifest_path": f"runs/{run_id}/manifest.json",
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
