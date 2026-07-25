from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from flamo.analysis import RecipeService
from flamo.catalog import Catalog
from flamo.evidence import GenerationPublisher
from flamo.storage import Workspace


def test_scaling_reports_dispersion_models_and_supported_range(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    now = datetime.now(UTC)
    variants = (32_768, 65_536, 131_072)
    run_rows = []
    trial_rows = []
    measurement_rows = []
    variant_rows = []
    for input_value in variants:
        variant_id = f"variant-{input_value}"
        variant_rows.append(
            {
                "variant_id": variant_id,
                "experiment_id": "scaling-experiment",
                "name": str(input_value),
                "source_state_id": "source",
                "workload_instance_id": f"workload-{input_value}",
                "environment_requirements_json": "{}",
                "parameters_json": f'{{"length":{input_value}}}',
            }
        )
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
                    "block_id": f"block-{block}",
                    "order_in_block": block,
                    "parameter_name": "length",
                    "parameter_value_int": input_value,
                    "parameter_value_float": None,
                    "attempt": 1,
                    "outcome": "succeeded",
                    "exclusion_reason": None,
                    "validation_status": "passed",
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
                    "block_id": f"block-{block}",
                    "variant_id": variant_id,
                    "order_in_block": block,
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
                    "created_at": now,
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

    result = RecipeService(workspace).scaling("scaling-experiment")

    assert result.attempted_trials == 9
    assert result.complete_blocks == 3
    assert result.environment_stable
    assert any(fit.model == "linear" for fit in result.fits)
    assert {fit.supported_min for fit in result.fits} == {32_768}
    assert {fit.supported_max for fit in result.fits} == {131_072}
