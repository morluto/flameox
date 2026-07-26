from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from flameox.analysis import RecipeService
from flameox.application import (
    CreateInvestigationRequest,
    ExecutionPolicy,
    ExperimentService,
    InvestigationService,
)
from flameox.catalog import Catalog
from flameox.config import WorkspaceConfig
from flameox.domain import (
    ComparisonValidity,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    TrialOutcome,
)
from flameox.storage import RunStore, Workspace


def _git(project: Path, *args: str) -> None:
    subprocess.run(
        ("git", *args),
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.anyio
async def test_randomized_experiment_records_trials_and_compares_run_sets(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "bench.py").write_text(
        "import sys\n"
        "count = 6000 if sys.argv[1] == 'baseline' else 1500\n"
        "assert sum(range(count)) >= 0\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1

[workloads.scan]
argv = ["python", "bench.py", "{implementation}"]
cwd = "."
timeout_seconds = 30

[workloads.scan.parameters]
implementation = ["baseline", "candidate"]

[workloads.scan.oracle]
strength = "cross_treatment_equivalence"
argv = ["python", "-c", "print('same-output')"]

[experiments.scan_comparison]
workload = "scan"
variants = ["baseline", "candidate"]
design = "randomized_complete_blocks"
blocks = 1
primary_metric = "pyperf.workload"
polarity = "lower_is_better"
estimand = "median_paired_log_ratio"
practical_threshold = 0.01
confidence_level = 0.95
random_seed = 1984
"""
    )
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
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

    result = await service.run(plan.plan_id, progress=record_progress)

    assert len(plan.blocks) == 1
    assert set(plan.blocks[0].order) == {"baseline", "candidate"}
    assert len(result.trials) == 2
    assert all(trial.outcome is TrialOutcome.SUCCEEDED for trial in result.trials)
    assert len(result.run_sets) == 2
    assert result.comparison is not None
    assert result.comparison.comparison.experiment_id == result.experiment.experiment_id
    assert result.comparison.comparison.validity is ComparisonValidity.EXPLORATORY
    assert result.comparison.comparison.complete_pair_n == 1
    assert result.limitations == ()
    assert service.experiments.read(result.experiment.experiment_id) == result.experiment
    scaling = RecipeService(workspace).scaling(result.experiment.experiment_id)
    assert scaling.attempted_trials == 2
    assert scaling.complete_blocks == 1
    assert {point.variant for point in scaling.points} == {
        "baseline",
        "candidate",
    }
    assert [item[0] for item in progress] == list(range(7))
    assert {item[1] for item in progress} == {6}
    assert all(item[2] for item in progress)


@pytest.mark.anyio
async def test_experiment_plan_rejects_multiplicative_trial_explosion(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    scaling_values = ", ".join(str(value) for value in range(1_000))
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
variants = ["baseline", "candidate"]
design = "randomized_complete_blocks"
blocks = 1000
scaling_parameter = "length"
scaling_values = [{scaling_values}]
primary_metric = "wall_time"
polarity = "lower_is_better"
estimand = "median_paired_log_ratio"
practical_threshold = 0.01
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
    assert error.value.details["trial_count"] == 2_000_000


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
variants = ["baseline", "candidate"]
design = "randomized_complete_blocks"
blocks = 1
primary_metric = "wall_time"
polarity = "lower_is_better"
estimand = "median_paired_log_ratio"
practical_threshold = 0.01
confidence_level = 0.95
random_seed = 7
"""
    )
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
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

    task = asyncio.create_task(service.run(plan.plan_id))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with Catalog(workspace).open_snapshot() as snapshot:
        rows = snapshot.execute(
            "SELECT DISTINCT run_id, outcome FROM trials WHERE experiment_id = ?",
            (plan.experiment.experiment_id,),
        ).fetchall()
        assert len(rows) == 1
        run_rows = snapshot.execute(
            "SELECT execution_status FROM runs WHERE run_id = ? ORDER BY published_at DESC LIMIT 1",
            (str(rows[0][0]),),
        ).fetchall()
    assert rows[0][1] == "cancelled"
    run = RunStore(workspace).read(str(rows[0][0]))
    assert run.execution_status is ExecutionStatus.CANCELLED
    assert run_rows == [("cancelled",)]
