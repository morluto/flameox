from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from flameox.application import (
    ImportArtifactRequest,
    ImportService,
    PlanReductionRequest,
    ReductionLimits,
    ReductionService,
    WorkloadService,
)
from flameox.domain import ArtifactKind
from flameox.storage import Workspace


def _configure(project: Path, reducer_code: str, predicate_code: str) -> None:
    (project / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.reducer]
argv = ["python", "-c", {json.dumps(reducer_code)}]
cwd = "."
timeout_seconds = 30
[workloads.predicate]
argv = ["python", "-c", {json.dumps(predicate_code)}]
cwd = "."
timeout_seconds = 30
"""
    )


def _approve(workspace: Workspace) -> None:
    workloads = WorkloadService(workspace)
    workloads.approve("reducer")
    workloads.approve("predicate")


def _original(workspace: Workspace, path: Path, content: str = "large FAIL input") -> str:
    path.write_text(content)
    return (
        ImportService(workspace)
        .import_artifact(ImportArtifactRequest(path=path, kind=ArtifactKind.COLLECTOR_METADATA))
        .artifact_id
    )


@pytest.mark.anyio
async def test_reduction_binds_predicate_and_revalidates_final_candidate(
    tmp_path: Path,
) -> None:
    reducer = (
        "import os,pathlib,subprocess;"
        "p=pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']);"
        "p.write_text('FAIL');"
        "raise SystemExit(subprocess.run("
        "[os.environ['FLAMEOX_PREDICATE_WRAPPER'],str(p)]).returncode)"
    )
    predicate = (
        "import os,pathlib;"
        "raise SystemExit(0 if 'FAIL' in "
        "pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']).read_text() else 1)"
    )
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, reducer, predicate)
    _approve(workspace)
    service = ReductionService(workspace)
    plan = service.plan(
        PlanReductionRequest(
            original_artifact_id=_original(workspace, tmp_path / "original.txt"),
            reducer_workload="reducer",
            predicate_workload="predicate",
            limits=ReductionLimits(predicate_repetitions=2),
            expected_determinism="repeated",
        )
    )

    result = await service.execute(plan.plan_id)

    assert result.disposition == "succeeded"
    assert result.final_artifact_id is not None
    assert result.final_artifact_id != result.original_artifact_id
    assert result.attempts.attempted == 3
    assert result.attempts.passed == 3
    assert result.cleanup_complete is True
    assert service.get(result.reduction_id) == result


@pytest.mark.anyio
async def test_concurrent_reduction_execution_reconnects_to_one_result(
    tmp_path: Path,
) -> None:
    reducer = (
        "import os,pathlib,subprocess,time;time.sleep(.2);"
        "p=pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']);p.write_text('FAIL');"
        "raise SystemExit(subprocess.run("
        "[os.environ['FLAMEOX_PREDICATE_WRAPPER'],str(p)]).returncode)"
    )
    predicate = "raise SystemExit(0)"
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, reducer, predicate)
    _approve(workspace)
    service = ReductionService(workspace)
    plan = service.plan(
        PlanReductionRequest(
            original_artifact_id=_original(workspace, tmp_path / "original.txt"),
            reducer_workload="reducer",
            predicate_workload="predicate",
        )
    )

    first, second = await asyncio.gather(
        service.execute(plan.plan_id),
        ReductionService(workspace).execute(plan.plan_id),
    )

    assert first == second


@pytest.mark.anyio
async def test_reduction_drains_accepted_predicate_requests_before_finalizing(
    tmp_path: Path,
) -> None:
    reducer = (
        "import os,pathlib,subprocess,time;"
        "p=pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']);p.write_text('FAIL');"
        "subprocess.Popen([os.environ['FLAMEOX_PREDICATE_WRAPPER'],str(p)]);time.sleep(.1)"
    )
    predicate = "import time;time.sleep(.3);raise SystemExit(0)"
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, reducer, predicate)
    _approve(workspace)
    service = ReductionService(workspace)
    plan = service.plan(
        PlanReductionRequest(
            original_artifact_id=_original(workspace, tmp_path / "original.txt"),
            reducer_workload="reducer",
            predicate_workload="predicate",
        )
    )

    result = await service.execute(plan.plan_id)

    assert result.disposition == "succeeded"
    assert result.attempts.attempted == 2


@pytest.mark.anyio
async def test_reduction_reports_contradictory_predicate_as_inconclusive(
    tmp_path: Path,
) -> None:
    reducer = (
        "import os,pathlib,subprocess;"
        "p=pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']);p.write_text('FAIL');"
        "raise SystemExit(subprocess.run("
        "[os.environ['FLAMEOX_PREDICATE_WRAPPER'],str(p)]).returncode)"
    )
    predicate = (
        "import os,pathlib;"
        "p=pathlib.Path('toggle');n=int(p.read_text())+1 if p.exists() else 1;p.write_text(str(n));"
        "raise SystemExit(n%2)"
    )
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, reducer, predicate)
    _approve(workspace)
    service = ReductionService(workspace)
    plan = service.plan(
        PlanReductionRequest(
            original_artifact_id=_original(workspace, tmp_path / "original.txt"),
            reducer_workload="reducer",
            predicate_workload="predicate",
            limits=ReductionLimits(predicate_repetitions=2),
            expected_determinism="repeated",
        )
    )

    result = await service.execute(plan.plan_id)

    assert result.disposition == "inconclusive"
    assert result.attempts.contradictory == 1
    assert result.final_artifact_id is None


@pytest.mark.anyio
async def test_reduction_rejects_candidate_symlink(tmp_path: Path) -> None:
    reducer = (
        "import os,pathlib;"
        "pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']).symlink_to("
        "os.environ['FLAMEOX_REDUCTION_ORIGINAL'])"
    )
    predicate = "raise SystemExit(0)"
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, reducer, predicate)
    _approve(workspace)
    service = ReductionService(workspace)
    plan = service.plan(
        PlanReductionRequest(
            original_artifact_id=_original(workspace, tmp_path / "original.txt"),
            reducer_workload="reducer",
            predicate_workload="predicate",
        )
    )

    result = await service.execute(plan.plan_id)

    assert result.disposition == "failed"
    assert result.final_artifact_id is None
    assert result.cleanup_complete is True


@pytest.mark.anyio
async def test_reduction_cancellation_publishes_terminal_result(tmp_path: Path) -> None:
    reducer = "import time;time.sleep(30)"
    predicate = "raise SystemExit(0)"
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, reducer, predicate)
    _approve(workspace)
    service = ReductionService(workspace)
    plan = service.plan(
        PlanReductionRequest(
            original_artifact_id=_original(workspace, tmp_path / "original.txt"),
            reducer_workload="reducer",
            predicate_workload="predicate",
        )
    )
    task = asyncio.create_task(service.execute(plan.plan_id))
    await asyncio.sleep(0.1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    reduction_id = next(
        path.name
        for path in (workspace.paths.records / "reduction_results").iterdir()
        if path.is_dir()
    )
    result = service.get(reduction_id)
    assert result.disposition == "cancelled"
    assert result.cleanup_complete is True
