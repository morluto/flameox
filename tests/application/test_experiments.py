from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.analysis import RecipeService
from flameox.application import (
    ComparisonService,
    CreateInvestigationRequest,
    ExecutionPolicy,
    ExperimentPlan,
    ExperimentService,
    FreezeRunIdsRequest,
    FreezeRunMembersRequest,
    IncludedFreezeRunSetMember,
    InvestigationService,
    MeasurementCompareRunSetsRequest,
    RunSetService,
    parse_experiment_config,
)
from flameox.application.experiments import OutcomeExperimentResult
from flameox.application.runtime_resources import RUNTIME_RESOURCE_METRIC_CATALOG
from flameox.catalog import Catalog
from flameox.config import WorkspaceConfig
from flameox.domain import (
    ComparisonValidity,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ExperimentOutcomeDisposition,
    ExperimentOutcomeGoal,
    TrialFailureClass,
    TrialOutcome,
    VariantIdentityQuality,
)
from flameox.storage import RunStore, Workspace

pytestmark = [pytest.mark.integration, pytest.mark.serial]


def _git(project: Path, *args: str) -> None:
    subprocess.run(
        ("git", *args),
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )


def test_outcome_result_rejects_half_a_first_failure() -> None:
    with pytest.raises(ValidationError):
        OutcomeExperimentResult.model_validate(
            {
                "experiment_id": "experiment",
                "goal": ExperimentOutcomeGoal.ABSENCE_OF_FAILURE,
                "disposition": ExperimentOutcomeDisposition.INSUFFICIENT_EVIDENCE,
                "counts": (),
                "complete_pairs": 0,
                "unmatched_cells": 0,
                "first_failure_trial_id": "trial",
            }
        )


@pytest.mark.parametrize(
    "metric",
    (
        "runtime_resource.peak_rss_bytes",
        "runtime_resource.minimum_free_bytes",
        "runtime_resource.staging_growth_bytes",
    ),
)
def test_experiment_config_accepts_closed_runtime_resource_catalog(metric: str) -> None:
    config = parse_experiment_config(
        {
            "workload": "scan",
            "variants": ("baseline", "candidate"),
            "primary_metric": metric,
        }
    )

    assert config.primary_metric == metric


def test_runtime_resource_catalog_records_the_admission_contract() -> None:
    assert set(RUNTIME_RESOURCE_METRIC_CATALOG) == {
        "runtime_resource.peak_rss_bytes",
        "runtime_resource.minimum_free_bytes",
        "runtime_resource.staging_growth_bytes",
    }
    for metric, definition in RUNTIME_RESOURCE_METRIC_CATALOG.items():
        assert definition.metric == metric
        assert definition.unit == "bytes"
        assert definition.scope
        assert definition.aggregation
        assert definition.collection_backend
        assert definition.unavailable_behavior
        assert definition.limitations
        assert definition.supports_confirmatory_paired_comparison is True


def test_experiment_config_rejects_writable_root_growth_metric() -> None:
    with pytest.raises(ValueError, match="runtime-resource primary_metric"):
        parse_experiment_config(
            {
                "workload": "scan",
                "variants": ("baseline", "candidate"),
                "primary_metric": "runtime_resource.writable_root_growth_bytes",
            }
        )


def test_experiment_plan_defaults_metric_source_for_persisted_plans() -> None:
    assert ExperimentPlan.model_fields["metric_source"].default == "measurement"


@pytest.mark.anyio
async def test_randomized_experiment_records_trials_and_compares_run_sets(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "bench.py").write_text(
        "import sys\n"
        "count = 6000 if sys.argv[1] == 'implementation=' else 1500\n"
        "assert sum(range(count)) >= 0\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1

[workloads.scan]
argv = ["python", "bench.py", "implementation={implementation}"]
cwd = "."
timeout_seconds = 30

[workloads.scan.parameters]
implementation = ["", "candidate"]

[workloads.scan.oracle]
strength = "cross_treatment_equivalence"
argv = ["python", "-c", "print('same-output')"]

[experiments.scan_comparison]
workload = "scan"
treatment_factor = "implementation"
baseline_value = ""
design = "randomized_complete_blocks"
blocks = 2
primary_metric = "pyperf.workload"
polarity = "lower_is_better"
estimand = "median_paired_log_ratio"
practical_threshold = 0.01
confidence_level = 0.95
random_seed = 1984
[experiments.scan_comparison.factors]
implementation = ["candidate", ""]
"""
    )
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    assert isinstance(config, WorkspaceConfig)
    workspace.paths.config.write_text(config.to_toml())
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "bench.py", "flameox.toml")
    _git(
        tmp_path,
        "-c",
        "user.name=flameox Test",
        "-c",
        "user.email=flameox@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Does the candidate remove scan overhead?")
    )
    service = ExperimentService(workspace)

    plan = await service.plan(
        experiment_name="scan_comparison",
        investigation_id=investigation.investigation_id,
        adapter="pyperf",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    progress: list[tuple[float, float, str]] = []

    async def record_progress(
        completed: float,
        total: float,
        message: str,
    ) -> None:
        progress.append((completed, total, message))

    result = await service.run(plan.plan_token, progress=record_progress)

    assert len(plan.blocks) == 2
    assert all(set(block.order) == {"", "candidate"} for block in plan.blocks)
    assert plan.baseline_variant == ""
    assert len(result.trials) == 4
    assert all(trial.outcome is TrialOutcome.SUCCEEDED for trial in result.trials)
    assert len(result.run_sets) == 2
    assert result.comparison is not None
    assert result.comparison.comparison.experiment_id == result.experiment.experiment_id
    assert result.comparison.comparison.validity is ComparisonValidity.EXPLORATORY
    assert result.comparison.comparison.complete_pair_n == 2
    assert result.comparison.comparison.protocol_identity_id is not None
    assert result.comparison.comparison.metric_contract_id.startswith("sha256:")
    assert result.comparison.baseline_run_set.selection["variant"] == ""
    assert result.comparison.candidate_run_set.selection["variant"] == "candidate"
    assert result.limitations == ()
    with Catalog(workspace).open_snapshot(result.corpus_commit_id) as snapshot:
        assert snapshot.execute(
            "SELECT count(*) FROM experiments WHERE experiment_id = ?",
            (result.experiment.experiment_id,),
        ).fetchone() == (1,)
        assert snapshot.execute(
            "SELECT count(DISTINCT variant_id) FROM variants WHERE experiment_id = ?",
            (result.experiment.experiment_id,),
        ).fetchone() == (len(result.variants),)
        assert snapshot.execute(
            "SELECT count(DISTINCT trial_id) FROM trials WHERE experiment_id = ?",
            (result.experiment.experiment_id,),
        ).fetchone() == (len(result.trials),)
        assert snapshot.execute(
            "SELECT count(*) FROM run_sets WHERE run_set_id IN (?, ?)",
            tuple(value.run_set_id for value in result.run_sets),
        ).fetchone() == (len(result.run_sets),)
        assert snapshot.execute(
            "SELECT count(*) FROM comparisons WHERE comparison_id = ?",
            (result.comparison.comparison.comparison_id,),
        ).fetchone() == (1,)
    assert service.experiments.read(result.experiment.experiment_id) == result.experiment
    scaling = RecipeService(workspace).scaling(result.experiment.experiment_id)
    assert scaling.attempted_trials == 4
    assert scaling.complete_blocks == 2
    assert {point.variant for point in scaling.points} == {
        "",
        "candidate",
    }
    assert [item[0] for item in progress] == list(range(9))
    assert {item[1] for item in progress} == {8}
    assert all(item[2] for item in progress)

    declared_request = MeasurementCompareRunSetsRequest(
        baseline_run_set_id=result.comparison.baseline_run_set.run_set_id,
        candidate_run_set_id=result.comparison.candidate_run_set.run_set_id,
        experiment_id=result.experiment.experiment_id,
        metric=result.experiment.primary_metric,
        unit="ns",
        polarity=result.experiment.polarity,
        estimand="median_paired_log_ratio",
        practical_threshold=result.experiment.practical_threshold,
        confidence_level=result.experiment.confidence_level,
        random_seed=result.experiment.random_seed,
    )
    for update, field_name in (
        ({"metric": "other.metric"}, "metric"),
        ({"unit": "milliseconds"}, "unit"),
        ({"estimand": "difference_in_median_logs"}, "estimand"),
        ({"practical_threshold": 0.9}, "practical_threshold"),
        ({"confidence_level": 0.8}, "confidence_level"),
        ({"random_seed": 999}, "random_seed"),
    ):
        mismatched = ComparisonService(workspace).compare(
            declared_request.validated_copy(update=update)
        )
        assert mismatched.comparison.validity is ComparisonValidity.INVALID
        assert any(
            f"comparison {field_name} differs" in reason
            for reason in mismatched.comparison.mismatches
        )

    partial_run_sets = RunSetService(workspace)
    partial_baseline_member = result.comparison.baseline_run_set.members[0]
    partial_candidate_member = result.comparison.candidate_run_set.members[0]
    assert partial_baseline_member.trial_id is not None
    assert partial_candidate_member.trial_id is not None
    partial_baseline = partial_run_sets.freeze(
        FreezeRunMembersRequest(
            selection=result.comparison.baseline_run_set.selection,
            members=(
                IncludedFreezeRunSetMember(
                    run_id=partial_baseline_member.run_id,
                    trial_id=partial_baseline_member.trial_id,
                ),
            ),
        )
    )
    partial_candidate = partial_run_sets.freeze(
        FreezeRunMembersRequest(
            selection=result.comparison.candidate_run_set.selection,
            members=(
                IncludedFreezeRunSetMember(
                    run_id=partial_candidate_member.run_id,
                    trial_id=partial_candidate_member.trial_id,
                ),
            ),
        )
    )
    partial = ComparisonService(workspace).compare(
        declared_request.validated_copy(
            update={
                "baseline_run_set_id": partial_baseline.run_set_id,
                "candidate_run_set_id": partial_candidate.run_set_id,
            }
        )
    )
    assert partial.comparison.validity is ComparisonValidity.INVALID
    assert any(
        "full declared trial population" in reason for reason in partial.comparison.mismatches
    )

    unbound_baseline = partial_run_sets.freeze(
        FreezeRunIdsRequest(run_ids=(partial_baseline_member.run_id,))
    )
    unbound_candidate = partial_run_sets.freeze(
        FreezeRunIdsRequest(run_ids=(partial_candidate_member.run_id,))
    )
    unbound = ComparisonService(workspace).compare(
        declared_request.validated_copy(
            update={
                "baseline_run_set_id": unbound_baseline.run_set_id,
                "candidate_run_set_id": unbound_candidate.run_set_id,
            }
        )
    )
    assert unbound.comparison.validity is ComparisonValidity.INVALID
    assert any("without a trial identity" in reason for reason in unbound.comparison.mismatches)


@pytest.mark.anyio
async def test_runtime_resource_primary_metric_runs_paired_comparison(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "bench.py").write_text("assert sum(range(1000)) >= 0\n")
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.scan]
argv = ["python", "bench.py"]
cwd = "."
[workloads.scan.parameters]
implementation = ["baseline", "candidate"]
[workloads.scan.oracle]
strength = "cross_treatment_equivalence"
argv = ["python", "-c", "print('same-output')"]
[experiments.resources]
workload = "scan"
treatment_factor = "implementation"
blocks = 1
primary_metric = "runtime_resource.peak_rss_bytes"
polarity = "lower_is_better"
[experiments.resources.factors]
implementation = ["baseline", "candidate"]
"""
    )
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "bench.py", "flameox.toml")
    _git(
        tmp_path,
        "-c",
        "user.name=flameox Test",
        "-c",
        "user.email=flameox@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Does resource use differ?")
    )
    service = ExperimentService(workspace)
    plan = await service.plan(
        experiment_name="resources",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.run(plan.plan_token)

    assert result.comparison is not None
    assert result.comparison.comparison.metric == "runtime_resource.peak_rss_bytes"


@pytest.mark.anyio
async def test_structured_benchmark_samples_run_the_automatic_paired_analysis(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "structured_bench.py").write_text(
        "import json, os, sys\n"
        "treatment = sys.argv[1].split('=', 1)[1]\n"
        "samples = [50, 50, 50] if treatment == 'candidate' else [100, 100, 100]\n"
        "document = {\n"
        "  'schema_version': 'flameox.benchmark-samples.v1',\n"
        "  'producer': 'fixture-benchmark', 'producer_version': '1',\n"
        "  'benchmarks': [{\n"
        "    'name': 'kernel.time', 'unit': 'ns',\n"
        "    'measurement_clock': 'device_event',\n"
        "    'synchronization': 'event_synchronize',\n"
        "    'scope': 'device', 'phase': 'steady_state', 'loop_count': 1,\n"
        "    'device': {'type': 'mps', 'index': 0},\n"
        "    'dimensions': {'shape': 'small'}, 'warmups': [110],\n"
        "    'samples': samples\n"
        "  }]\n"
        "}\n"
        "with open(os.environ['FLAMEOX_BENCHMARK_OUTPUT'], 'w') as stream:\n"
        "    json.dump(document, stream)\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.kernel]
argv = ["python", "structured_bench.py", "implementation={implementation}"]
[workloads.kernel.parameters]
implementation = ["baseline", "candidate"]
[workloads.kernel.oracle]
strength = "cross_treatment_equivalence"
argv = ["python", "-c", "print('same-output')"]

[experiments.kernel]
workload = "kernel"
design = "randomized_complete_blocks"
blocks = 3
treatment_factor = "implementation"
baseline_value = "baseline"
primary_metric = "kernel.time"
primary_metric_unit = "ns"
polarity = "lower_is_better"
estimand = "median_paired_log_ratio"
practical_threshold = 0.05
random_seed = 7
[experiments.kernel.factors]
implementation = ["baseline", "candidate"]
[experiments.kernel.measurement_series]
scope = "device"
aggregation = "sample"
phase = "steady_state"
loop_count = 1
[experiments.kernel.measurement_series.dimensions]
measurement_clock = "device_event"
synchronization = "event_synchronize"
producer = "fixture-benchmark"
producer_version = "1"
"device.type" = "mps"
"device.index" = "0"
shape = "small"
"""
    )
    workspace.paths.config.write_text(
        workspace.config.validated_copy(
            update={
                "execution": workspace.config.execution.validated_copy(
                    update={"containment": "disabled"}
                )
            }
        ).to_toml()
    )
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "structured_bench.py", "flameox.toml")
    _git(
        tmp_path,
        "-c",
        "user.name=flameox Test",
        "-c",
        "user.email=flameox@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Does the candidate reduce device time?")
    )
    service = ExperimentService(workspace)
    plan = await service.plan(
        experiment_name="kernel",
        investigation_id=investigation.investigation_id,
        adapter="benchmark-samples",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.run(plan.plan_token)

    assert result.comparison is not None
    comparison = result.comparison.comparison
    assert comparison.validity is ComparisonValidity.VALID
    assert comparison.complete_pair_n == 3
    assert comparison.independent_unit == "block"
    assert comparison.relative_change == pytest.approx(-0.5)
    assert comparison.measurement_series_id is not None


@pytest.mark.anyio
async def test_explicit_factor_matrix_completes_when_progress_delivery_fails(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "bench.py").write_text("assert sum(range(1000)) >= 0\n")
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.scan]
argv = ["python", "bench.py"]
cwd = "."
[workloads.scan.parameters]
implementation = ["baseline", "candidate", "other"]
[workloads.scan.oracle]
strength = "cross_treatment_equivalence"
argv = ["python", "-c", "print('same-output')"]
[experiments.partial]
workload = "scan"
treatment_factor = "implementation"
baseline_value = "baseline"
combination_policy = "explicit"
combinations = [{implementation = "candidate"}, {implementation = "other"}]
blocks = 1
primary_metric = "runtime_resource.peak_rss_bytes"
polarity = "lower_is_better"
[experiments.partial.factors]
implementation = ["baseline", "candidate", "other"]
"""
    )
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "bench.py", "flameox.toml")
    _git(
        tmp_path,
        "-c",
        "user.name=flameox Test",
        "-c",
        "user.email=flameox@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Does the partial matrix affect resource use?")
    )
    service = ExperimentService(workspace)
    plan = await service.plan(
        experiment_name="partial",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    async def failed_notification(completed: float, total: float, message: str) -> None:
        raise RuntimeError("progress transport closed")

    result = await service.run(plan.plan_token, progress=failed_notification)

    assert len(result.trials) == 2
    assert result.comparison is None
    assert (
        "Automatic paired comparison requires the declared baseline and exactly one candidate "
        "treatment." in (result.limitations)
    )


@pytest.mark.anyio
async def test_experiment_plan_rejects_multiplicative_trial_explosion(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    scaling_values = ", ".join(str(value) for value in range(32))
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1

[workloads.scan]
argv = ["python", "-c", "print('{{variant}}', '{{length}}')"]
cwd = "."

[workloads.scan.parameters]
variant = ["baseline", "candidate"]
length = [{scaling_values}]

[experiments.oversized]
workload = "scan"
treatment_factor = "variant"
design = "randomized_complete_blocks"
blocks = 1000
primary_metric = "wall_time"
polarity = "lower_is_better"
estimand = "median_paired_log_ratio"
practical_threshold = 0.01
[experiments.oversized.factors]
variant = ["baseline", "candidate"]
length = [{scaling_values}]
"""
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Should an oversized plan be rejected?")
    )

    with pytest.raises(DomainError) as error:
        await ExperimentService(workspace).plan(
            experiment_name="oversized",
            investigation_id=investigation.investigation_id,
            adapter="command",
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )

    assert error.value.code is ErrorCode.QUERY_BUDGET_EXCEEDED
    assert error.value.details["trial_count"] == 64_000


@pytest.mark.anyio
async def test_cancelled_experiment_preserves_attempted_trial(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "wait.py").write_text("import time\ntime.sleep(30)\n")
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1

[workloads.wait]
argv = ["python", "wait.py", "{variant}"]
cwd = "."
timeout_seconds = 60

[workloads.wait.parameters]
variant = ["baseline", "candidate"]

[experiments.cancelled]
workload = "wait"
treatment_factor = "variant"
design = "randomized_complete_blocks"
blocks = 1
primary_metric = "wall_time"
polarity = "lower_is_better"
estimand = "median_paired_log_ratio"
practical_threshold = 0.01
confidence_level = 0.95
random_seed = 7
[experiments.cancelled.factors]
variant = ["baseline", "candidate"]
"""
    )
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Does cancellation preserve the attempt?")
    )
    service = ExperimentService(workspace)
    plan = await service.plan(
        experiment_name="cancelled",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    task = asyncio.create_task(service.run(plan.plan_token))
    running_run_id: str | None = None
    for _ in range(500):
        for run in RunStore(workspace).list():
            if run.execution_status is ExecutionStatus.RUNNING:
                running_run_id = run.run_id
                break
        if running_run_id is not None:
            break
        await asyncio.sleep(0.01)
    assert running_run_id is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with Catalog(workspace).open_snapshot() as snapshot:
        rows = snapshot.execute(
            "SELECT DISTINCT run_id, outcome FROM trials WHERE experiment_id = ?",
            (plan.experiment.experiment_id,),
        ).fetchall()
        assert len(rows) == 2
        assert (None, "unattempted") in rows
        attempted = next(row for row in rows if row[0] is not None)
        run_rows = snapshot.execute(
            "SELECT execution_status FROM runs WHERE run_id = ? ORDER BY published_at DESC LIMIT 1",
            (str(attempted[0]),),
        ).fetchall()
    assert attempted[1] == "cancelled"
    run = RunStore(workspace).read(str(attempted[0]))
    assert run.execution_status is ExecutionStatus.CANCELLED
    assert run_rows == [("cancelled",)]


@pytest.mark.anyio
async def test_failed_experiment_preserves_failed_and_unattempted_cells(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.probe]
argv = ["python", "-c", "print('{variant}')"]
[workloads.probe.parameters]
variant = ["base", "candidate"]
[experiments.failed]
workload = "probe"
treatment_factor = "variant"
blocks = 1
primary_metric = "wall_time"
polarity = "neutral"
estimand = "descriptive"
practical_threshold = 0
[experiments.failed.factors]
variant = ["base", "candidate"]
"""
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Are unattempted cells preserved?")
    )
    service = ExperimentService(workspace)
    plan = await service.plan(
        experiment_name="failed",
        investigation_id=investigation.investigation_id,
        adapter="missing-adapter",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    with pytest.raises(DomainError):
        await service.run(plan.plan_token)

    with Catalog(workspace).open_snapshot() as snapshot:
        rows = snapshot.execute(
            "SELECT outcome, failure_class, run_id FROM trials "
            "WHERE experiment_id = ? ORDER BY outcome",
            (plan.experiment.experiment_id,),
        ).fetchall()
    assert rows == [
        ("infrastructure_failed", "infrastructure_failure", None),
        ("unattempted", "unattempted", None),
    ]


@pytest.mark.anyio
async def test_multifactor_cartesian_exclusions_and_order_are_materialized_stably(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.matrix]
argv = ["python", "-c", "print('{mode}', '{dtype}', '{case}')"]
[workloads.matrix.parameters]
mode = ["base", "candidate"]
dtype = ["int8", "int16"]
case = ["empty", "boundary", "ordinary"]

[experiments.matrix]
workload = "matrix"
design = "fixed_order"
blocks = 1
treatment_factor = "mode"
max_trials = 20
primary_metric = "wall_time"
polarity = "neutral"
estimand = "descriptive"
practical_threshold = 0
exclude = [{mode = "candidate", case = "empty"}]
[experiments.matrix.factors]
mode = ["base", "candidate"]
dtype = ["int8", "int16"]
case = ["empty", "boundary", "ordinary"]
"""
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Which matrix cells differ?")
    )
    service = ExperimentService(workspace)

    first = await service.plan(
        experiment_name="matrix",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    second = await service.plan(
        experiment_name="matrix",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    first_cells = [cell for block in first.blocks for cell in block.cells]
    second_cells = [cell for block in second.blocks for cell in block.cells]

    assert len(first_cells) == 10
    assert [cell.trial_id for cell in first_cells] == [cell.trial_id for cell in second_cells]
    assert [cell.combination_id for cell in first_cells] == [
        cell.combination_id for cell in second_cells
    ]
    assert all(set(cell.factors) == {"mode", "dtype", "case"} for cell in first_cells)
    assert not any(
        cell.factors["mode"] == "candidate" and cell.factors["case"] == "empty"
        for cell in first_cells
    )
    assert first.experiment.experiment_design_id == second.experiment.experiment_design_id


@pytest.mark.anyio
async def test_factorial_variants_preserve_cohort_identity_without_a_representative_cell(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.matrix]
argv = ["python", "-c", "print('{implementation}', '{size}')"]
[workloads.matrix.parameters]
implementation = ["baseline", "candidate"]
size = [1, 2]

[experiments.matrix]
workload = "matrix"
design = "fixed_order"
blocks = 1
treatment_factor = "implementation"
baseline_value = "baseline"
primary_metric = "runtime_resource.peak_rss_bytes"
polarity = "lower_is_better"
[experiments.matrix.factors]
implementation = ["baseline", "candidate"]
size = [1, 2]
"""
    )
    workspace.paths.config.write_text(
        workspace.config.validated_copy(
            update={
                "execution": workspace.config.execution.validated_copy(
                    update={"containment": "disabled"}
                )
            }
        ).to_toml()
    )
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "flameox.toml")
    _git(
        tmp_path,
        "-c",
        "user.name=flameox Test",
        "-c",
        "user.email=flameox@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="How does size interact with implementation?")
    )
    service = ExperimentService(workspace)
    plan = await service.plan(
        experiment_name="matrix",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.run(plan.plan_token)

    assert len(result.trials) == 4
    assert len(result.variants) == 2
    for variant in result.variants:
        assert variant.identity_quality is VariantIdentityQuality.HETEROGENEOUS
        assert variant.parameters == {"implementation": variant.treatment_value}
        assert variant.varying_factors == {"size": (1, 2)}
        assert variant.workload_instance_id is None
        assert len(variant.workload_instance_ids) == 2
        assert len(variant.combination_ids) == 2
        selected_trials = [
            trial for trial in result.trials if trial.variant_id == variant.variant_id
        ]
        assert {trial.combination_id for trial in selected_trials} == set(variant.combination_ids)
        run_set = next(
            item for item in result.run_sets if item.selection["variant_id"] == variant.variant_id
        )
        assert run_set.selection["treatment_identity_id"] == variant.treatment_identity_id
        assert run_set.selection["combination_count"] == 2
    with Catalog(workspace).open_snapshot(result.corpus_commit_id) as snapshot:
        rows = snapshot.execute(
            "SELECT identity_quality, workload_instance_id, "
            "array_length(workload_instance_ids), varying_factors_json "
            "FROM variants WHERE experiment_id = ? ORDER BY name",
            (result.experiment.experiment_id,),
        ).fetchall()
    assert rows == [
        ("heterogeneous", None, 2, '{"size":[1,2]}'),
        ("heterogeneous", None, 2, '{"size":[1,2]}'),
    ]

    baseline_cells = [
        (block, order, cell)
        for block in plan.blocks
        for order, cell in enumerate(block.cells)
        if cell.treatment == "baseline"
    ]
    first_block, first_order, first_cell = baseline_cells[0]
    missing_first = service._make_trial(
        plan=plan,
        cell=first_cell,
        run=None,
        block_id=first_block.block_id,
        order=first_order,
        outcome=TrialOutcome.UNSUPPORTED,
        failure_class=TrialFailureClass.UNSUPPORTED_ENVIRONMENT,
    )
    surviving = next(
        trial
        for trial in result.trials
        if trial.factors == {"implementation": "baseline", "size": 2}
    )
    partial = service._variant_for_treatment(
        plan,
        "baseline",
        [missing_first, surviving],
    )
    assert partial.identity_quality is VariantIdentityQuality.INCOMPLETE
    assert partial.parameters == {"implementation": "baseline"}
    assert partial.varying_factors == {"size": (1, 2)}
    assert partial.workload_instance_id is None
    assert len(partial.workload_instance_ids) == 1


@pytest.mark.anyio
async def test_multifactor_explicit_partial_matrix_and_budget_validation(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.matrix]
argv = ["python", "-c", "print('{mode}', '{case}')"]
[workloads.matrix.parameters]
mode = ["base", "candidate"]
case = ["clean", "bad"]

[experiments.matrix]
workload = "matrix"
design = "randomized"
blocks = 2
treatment_factor = "mode"
combination_policy = "explicit"
combinations = [
  {mode = "base", case = "clean"},
  {mode = "candidate", case = "clean"},
  {mode = "base", case = "bad"},
]
max_trials = 6
primary_metric = "wall_time"
polarity = "neutral"
estimand = "descriptive"
practical_threshold = 0
[experiments.matrix.factors]
mode = ["base", "candidate"]
case = ["clean", "bad"]
"""
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Is the explicit matrix bounded?")
    )
    service = ExperimentService(workspace)

    plan = await service.plan(
        experiment_name="matrix",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    regenerated = await service.plan(
        experiment_name="matrix",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    assert sum(len(block.cells) for block in plan.blocks) == 6
    assert any(len(block.cells) == 1 for block in plan.blocks)
    assert {block.order for block in plan.blocks if len(block.order) == 2} <= {
        ("base", "candidate"),
        ("candidate", "base"),
    }
    assert [block.order for block in plan.blocks] == [block.order for block in regenerated.blocks]
    with pytest.raises(DomainError) as conflicting:
        await service.plan(
            experiment_name="matrix",
            investigation_id=investigation.investigation_id,
            adapter="command",
            parameter_overrides={"case": "clean"},
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
    assert conflicting.value.code is ErrorCode.INVALID_CAPTURE_PLAN

    config_path = tmp_path / "flameox.toml"
    config_path.write_text(config_path.read_text().replace("max_trials = 6", "max_trials = 5"))
    with pytest.raises(DomainError) as bounded:
        await service.plan(
            experiment_name="matrix",
            investigation_id=investigation.investigation_id,
            adapter="command",
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
    assert bounded.value.code is ErrorCode.QUERY_BUDGET_EXCEEDED

    config_path.write_text(
        config_path.read_text()
        .replace("max_trials = 5", "max_trials = 6")
        .replace('{mode = "base", case = "bad"}', '{mode = "base", case = "unknown"}')
    )
    with pytest.raises(DomainError) as undeclared:
        await service.plan(
            experiment_name="matrix",
            investigation_id=investigation.investigation_id,
            adapter="command",
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
    assert undeclared.value.code is ErrorCode.WORKSPACE_INVALID


@pytest.mark.anyio
async def test_outcome_experiment_classifies_oracle_failure_and_pairs_coordinates(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.semantic]
argv = ["python", "-c", "print('{mode}', '{case}')"]
[workloads.semantic.parameters]
mode = ["", "candidate"]
case = ["bad", "clean"]
[workloads.semantic.oracle]
strength = "contract_check"
argv = [
  "python", "-c",
  "import sys; raise SystemExit(1 if sys.argv[1:] == ['mode=', 'bad'] else 0)",
  "mode={mode}", "{case}",
]

[experiments.semantic]
workload = "semantic"
design = "fixed_order"
blocks = 1
treatment_factor = "mode"
baseline_value = ""
analysis = "outcome"
outcome_goal = "absence_of_failure"
minimum_attempts = 1
maximum_attempts = 1
[experiments.semantic.factors]
mode = ["candidate", ""]
case = ["bad", "clean"]
"""
    )
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Does the candidate preserve semantics?")
    )
    service = ExperimentService(workspace)
    plan = await service.plan(
        experiment_name="semantic",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.run(plan.plan_token)

    assert result.comparison is None
    assert result.outcome is not None
    assert result.outcome.disposition == "base_only_failure"
    assert result.outcome.complete_pairs == 2
    assert result.outcome.unmatched_cells == 0
    assert plan.baseline_variant == ""
    assert result.outcome.first_failure_factors == {"mode": "", "case": "bad"}
    counts = {item.treatment: item for item in result.outcome.counts}
    assert counts[""].oracle_failed == 1
    assert counts[""].failure_rate == 0.5
    assert counts[""].failure_rate_upper_bound is not None
    assert counts[""].failure_rate_upper_bound > 0.5
    assert counts["candidate"].pass_rate == 1
    assert counts["candidate"].failure_rate_upper_bound is not None
    assert counts["candidate"].failure_rate_upper_bound > 0
    with Catalog(workspace).open_snapshot(result.corpus_commit_id) as snapshot:
        row = snapshot.execute(
            "SELECT method, disposition, complete_pairs FROM experiment_outcomes "
            "WHERE experiment_id = ?",
            (result.experiment.experiment_id,),
        ).fetchone()
    assert row == ("absence_of_failure_fixed_attempts_v1", "base_only_failure", 2)


@pytest.mark.anyio
async def test_structured_receipts_round_trip_through_trials_and_lookup(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "oracle.py").write_text(
        """import json
import os
import sys
from pathlib import Path

treatment = sys.argv[1]
status = "fail" if treatment == "base" else "pass"
reason = "contract_mismatch" if status == "fail" else "expected_rejection"
Path(os.environ["FLAMEOX_ORACLE_RECEIPT"]).write_text(json.dumps({
    "schema_version": "flameox.oracle-receipt.v1",
    "status": status,
    "reason": reason,
    "case_id": treatment,
}))
"""
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.semantic]
argv = ["python", "-c", "print('{treatment}')"]
[workloads.semantic.parameters]
treatment = ["base", "candidate"]
[workloads.semantic.oracle]
strength = "contract_check"
argv = ["python", "oracle.py", "{treatment}"]
receipt_schema = "flameox.oracle-receipt.v1"

[experiments.semantic]
workload = "semantic"
design = "fixed_order"
blocks = 1
treatment_factor = "treatment"
analysis = "outcome"
outcome_goal = "absence_of_failure"
minimum_attempts = 1
maximum_attempts = 1
[experiments.semantic.factors]
treatment = ["base", "candidate"]
"""
    )
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Do structured outcomes survive publication?")
    )
    service = ExperimentService(workspace)
    plan = await service.plan(
        experiment_name="semantic",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.run(plan.plan_token)

    assert result.outcome is not None
    assert result.outcome.disposition == "base_only_failure"
    collection = service.list_trials(result.experiment.experiment_id)
    assert collection.returned == 2
    by_treatment = {trial.factors["treatment"]: trial for trial in collection.trials}
    assert by_treatment["base"].failure_class == "oracle_failure"
    assert by_treatment["candidate"].outcome is TrialOutcome.SUCCEEDED
    assert by_treatment["candidate"].oracle_receipt is not None
    assert by_treatment["candidate"].oracle_receipt.reason == "expected_rejection"
    loaded = service.get_trial(
        by_treatment["base"].trial_id,
        experiment_id=result.experiment.experiment_id,
    )
    assert loaded.oracle_receipt_artifact_id is not None
    assert loaded.oracle_receipt == by_treatment["base"].oracle_receipt

    first_page = service.list_trials(result.experiment.experiment_id, limit=1)
    assert first_page.returned == 1
    assert first_page.truncated is True
    assert first_page.next_cursor is not None
    second_page = service.list_trials(
        result.experiment.experiment_id,
        limit=1,
        cursor=first_page.next_cursor,
    )
    assert second_page.returned == 1
    assert second_page.truncated is False

    repeated_plan = await service.plan(
        experiment_name="semantic",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    repeated = await service.run(repeated_plan.plan_token)
    assert repeated.experiment.experiment_id != result.experiment.experiment_id
    with pytest.raises(DomainError, match="ambiguous"):
        service.get_trial(by_treatment["base"].trial_id)
    historical = service.get_trial(
        by_treatment["base"].trial_id,
        experiment_id=result.experiment.experiment_id,
    )
    assert historical.experiment_id == result.experiment.experiment_id


@pytest.mark.anyio
async def test_outcome_partial_matrix_reports_unmatched_and_insufficient_evidence(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.semantic]
argv = ["python", "-c", "print('{mode}', '{case}')"]
[workloads.semantic.parameters]
mode = ["base", "candidate"]
case = ["bad"]

[experiments.partial]
workload = "semantic"
design = "fixed_order"
blocks = 1
treatment_factor = "mode"
combination_policy = "explicit"
combinations = [{mode = "base", case = "bad"}]
analysis = "outcome"
outcome_goal = "absence_of_failure"
minimum_attempts = 1
maximum_attempts = 1
[experiments.partial.factors]
mode = ["base", "candidate"]
case = ["bad"]
"""
    )
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Is a partial matrix conclusive?")
    )
    service = ExperimentService(workspace)
    plan = await service.plan(
        experiment_name="partial",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.run(plan.plan_token)

    assert result.outcome is not None
    assert result.outcome.disposition == "insufficient_evidence"
    assert result.outcome.unmatched_cells == 1
    assert {item.treatment: item.attempted for item in result.outcome.counts} == {
        "base": 1,
        "candidate": 0,
    }


@pytest.mark.anyio
async def test_outcome_experiment_distinguishes_unsupported_environment(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.semantic]
argv = ["python", "-c", "print('{mode}')"]
[workloads.semantic.parameters]
mode = ["base", "candidate"]
[workloads.semantic.requirements]
executables = ["flameox-definitely-missing-executable"]

[experiments.unsupported]
workload = "semantic"
blocks = 1
treatment_factor = "mode"
analysis = "outcome"
outcome_goal = "absence_of_failure"
[experiments.unsupported.factors]
mode = ["base", "candidate"]
"""
    )
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Is this environment supported?")
    )
    service = ExperimentService(workspace)
    plan = await service.plan(
        experiment_name="unsupported",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.run(plan.plan_token)

    assert result.outcome is not None
    assert result.outcome.disposition == "unsupported"
    assert {item.treatment: item.unsupported for item in result.outcome.counts} == {
        "base": 1,
        "candidate": 1,
    }
    assert len(result.trials) == 2
    assert all(trial.run_id is None for trial in result.trials)
    assert all(trial.outcome is TrialOutcome.UNSUPPORTED for trial in result.trials)


@pytest.mark.anyio
async def test_outcome_experiment_classifies_hangs_as_timeouts(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.hang]
argv = ["python", "-c", "import time; time.sleep(1)", "{mode}"]
timeout_seconds = 0.1
[workloads.hang.parameters]
mode = ["base", "candidate"]

[experiments.hangs]
workload = "hang"
blocks = 1
treatment_factor = "mode"
analysis = "outcome"
outcome_goal = "absence_of_failure"
[experiments.hangs.factors]
mode = ["base", "candidate"]
"""
    )
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Do either treatment hangs?")
    )
    service = ExperimentService(workspace)
    plan = await service.plan(
        experiment_name="hangs",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.run(plan.plan_token)

    assert result.outcome is not None
    assert {item.treatment: item.timed_out for item in result.outcome.counts} == {
        "base": 1,
        "candidate": 1,
    }
    assert len(result.trials) == 2
    assert all(trial.failure_class == "timeout" for trial in result.trials)
    assert all(trial.outcome is TrialOutcome.TIMED_OUT for trial in result.trials)
