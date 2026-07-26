from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import pytest

from flameox.analysis import RecipeService
from flameox.catalog import Catalog
from flameox.domain import DomainError, ErrorCode
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace


def _run_row(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC),
        "run_type": "execution",
        "execution_status": "succeeded",
        "capture_status": "complete",
        "validation_status": "passed",
        "workload_definition_id": "workload",
        "workload_instance_id": "workload-instance",
        "measurement_protocol_id": "protocol",
        "environment_id": "environment",
        "source_state_id": "source",
        "collector": "fixture",
        "collector_version": "1",
        "exit_code": 0,
        "wall_time_ns": 1,
        "manifest_path": f"runs/{run_id}/manifest.json",
    }


def test_scaling_reports_dispersion_models_and_supported_range(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    now = datetime.now(UTC)
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
                    "block_id": f"block-{input_value}-{block}",
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


def test_memory_reports_phase_correlated_growth(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    rows: list[dict[str, object]] = []
    for index, (phase, value) in enumerate(
        (("warmup", 100), ("steady_state", 240), ("shutdown", 260))
    ):
        rows.append(
            {
                "measurement_id": f"memory-{index}",
                "run_id": "memory-run",
                "artifact_id": None,
                "name": "memory.retained_end",
                "value_int": value,
                "value_float": None,
                "unit": "bytes",
                "aggregation": "single",
                "scope": "process",
                "trial_id": None,
                "worker_id": None,
                "worker_run_index": 0,
                "value_index": index,
                "loop_count": None,
                "is_warmup": phase == "warmup",
                "block_id": None,
                "variant_id": None,
                "order_in_block": None,
                "phase": phase,
                "dimensions": {},
                "evidence_level": "observed",
            }
        )
    GenerationPublisher(workspace).publish_rows(
        {"runs": [_run_row("memory-run")], "measurements": rows},
        publisher="memory-fixture",
        publisher_version="1",
    )

    result = RecipeService(workspace).memory("memory-run")

    assert [point.phase for point in result.phase_growth] == [
        "warmup",
        "steady_state",
        "shutdown",
    ]
    assert [point.delta for point in result.phase_growth] == [None, 140.0, 20.0]


def test_execution_compares_path_and_semantic_observation_changes(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)

    def observation(
        run_id: str,
        observation_id: str,
        name: str,
        value: str,
        line: int,
    ) -> dict[str, object]:
        return {
            "observation_id": observation_id,
            "run_id": run_id,
            "artifact_id": None,
            "kind": "configuration",
            "name": name,
            "value_json": value,
            "file": "policy.py",
            "line_from": line,
            "line_to": line,
            "context": "update",
            "evidence_level": "observed",
        }

    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [_run_row("baseline"), _run_row("candidate")],
            "observations": [
                observation("baseline", "old-source", "old_log_prob_source", '"rollout"', 10),
                observation("baseline", "removed", "legacy_branch", "true", 20),
                observation("candidate", "new-source", "old_log_prob_source", '"epoch"', 10),
                observation("candidate", "added", "clip_fraction", "0.21", 30),
            ],
        },
        publisher="execution-fixture",
        publisher_version="1",
    )

    result = RecipeService(workspace).execution(
        "baseline",
        comparison_input_id="candidate",
    )

    assert [item.name for item in result.added] == ["clip_fraction"]
    assert [item.name for item in result.removed] == ["legacy_branch"]
    assert [item.name for item in result.changed] == ["old_log_prob_source"]
    assert result.changed[0].baseline_value_json == '"rollout"'
    assert result.changed[0].candidate_value_json == '"epoch"'


def test_execution_comparison_diffs_complete_inputs_before_limiting(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)

    def observation(
        run_id: str,
        observation_id: str,
        name: str,
        value: str,
        line: int,
    ) -> dict[str, object]:
        return {
            "observation_id": observation_id,
            "run_id": run_id,
            "artifact_id": None,
            "kind": "configuration",
            "name": name,
            "value_json": value,
            "file": "policy.py",
            "line_from": line,
            "line_to": line,
            "context": "update",
            "evidence_level": "observed",
        }

    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [_run_row("baseline"), _run_row("candidate")],
            "observations": [
                observation("baseline", "shared-old", "shared", '"old"', 10),
                observation("baseline", "baseline-only", "baseline_only", "true", 20),
                observation("candidate", "candidate-early", "candidate_early", "true", 1),
                observation("candidate", "shared-new", "shared", '"new"', 10),
            ],
        },
        publisher="bounded-execution-fixture",
        publisher_version="1",
    )

    result = RecipeService(workspace).execution(
        "baseline",
        comparison_input_id="candidate",
        limit=1,
    )

    assert [item.name for item in result.added] == ["candidate_early"]
    assert [item.name for item in result.removed] == ["baseline_only"]
    assert [item.name for item in result.changed] == ["shared"]
    assert result.returned == 1
    assert result.truncated


def test_read_only_analysis_pins_snapshot_without_workspace_write_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    head = workspace.corpus.read_head().commit_id

    def fail_write_lock(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("read-only analysis acquired the workspace write lock")

    monkeypatch.setattr(workspace, "write_locked", fail_write_lock)

    result = RecipeService(workspace).failures(limit=10)

    assert result.corpus_commit_id == head
    assert workspace.corpus.read_head().commit_id == head


def test_analysis_rejects_input_absent_from_pinned_corpus(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    with pytest.raises(DomainError) as error:
        RecipeService(workspace).hotspots("missing-run")

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
