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
    ReductionPlan,
    ReductionService,
)
from flameox.domain import ArtifactKind
from flameox.storage import Workspace


def _configure(project: Path, predicate_code: str) -> None:
    (project / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.predicate]
argv = ["python", "-c", {json.dumps(predicate_code)}]
cwd = "."
timeout_seconds = 30
"""
    )


def _original(workspace: Workspace, path: Path, content: str = "discard\nKEEP\ndiscard\n") -> str:
    path.write_text(content)
    return (
        ImportService(workspace)
        .import_artifact(ImportArtifactRequest(path=path, kind=ArtifactKind.COLLECTOR_METADATA))
        .artifact_id
    )


def _plan(workspace: Workspace, artifact_id: str) -> ReductionPlan:
    return ReductionService(workspace).plan(
        PlanReductionRequest(
            original_artifact_id=artifact_id,
            partitioner="text_lines",
            predicate_workload="predicate",
            limits=ReductionLimits(predicate_repetitions=2),
            expected_determinism="repeated",
        )
    )


@pytest.mark.anyio
async def test_native_reduction_revalidates_final_candidate(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(
        tmp_path,
        "import os,pathlib; raise SystemExit(0 if 'KEEP' in "
        "pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']).read_text() else 1)",
    )
    service = ReductionService(workspace)
    plan = _plan(workspace, _original(workspace, tmp_path / "original.txt"))

    assert plan.schema_version == 2
    assert plan.engine == "native_ddmin"
    result = await service.execute(plan.plan_id)

    assert result.disposition == "succeeded"
    assert result.final_artifact_id is not None
    assert result.final_artifact_id != result.original_artifact_id
    assert result.attempts.attempted >= 2
    assert result.attempts.passed >= 1
    assert result.final_revalidation_status == "interesting"
    assert result.minimality == "one_minimal"
    assert result.cleanup_complete is True
    assert service.get(result.reduction_id) == result


@pytest.mark.anyio
async def test_native_reduction_publishes_empty_final_artifact(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, "raise SystemExit(0)")
    original_id = _original(workspace, tmp_path / "original.txt", "remove-me\n")
    result = await ReductionService(workspace).execute(_plan(workspace, original_id).plan_id)

    assert result.disposition == "succeeded"
    assert result.final_unit_count == 0
    assert result.final_artifact_id is not None
    assert result.final_artifact_id != original_id
    final_path = ReductionService(workspace).artifacts.get(result.final_artifact_id).payload_path
    assert final_path.read_bytes() == b""


@pytest.mark.anyio
async def test_concurrent_native_execution_reconnects_to_one_result(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, "raise SystemExit(0)")
    artifact_id = _original(workspace, tmp_path / "original.txt")
    plan = _plan(workspace, artifact_id)

    first, second = await asyncio.gather(
        ReductionService(workspace).execute(plan.plan_id),
        ReductionService(workspace).execute(plan.plan_id),
    )

    assert first == second


@pytest.mark.anyio
async def test_native_reduction_does_not_read_original_bytes_in_application_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(
        tmp_path,
        "import os,pathlib; raise SystemExit(0 if 'KEEP' in "
        "pathlib.Path(os.environ['FLAMEOX_REDUCTION_CANDIDATE']).read_text() else 1)",
    )
    artifact_id = _original(workspace, tmp_path / "worker-original.txt")
    service = ReductionService(workspace)
    original_path = service.artifacts.get(artifact_id).payload_path
    original_read_bytes = Path.read_bytes

    def refuse_original_read(path: Path) -> bytes:
        if path == original_path:
            raise AssertionError("application process read the whole reduction artifact")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", refuse_original_read)

    result = await service.execute(_plan(workspace, artifact_id).plan_id)

    assert result.disposition == "succeeded"
    assert result.final_artifact_id is not None


@pytest.mark.anyio
async def test_waiter_retries_after_reduction_worker_owner_is_cancelled(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, "raise SystemExit(0)")
    plan = _plan(workspace, _original(workspace, tmp_path / "retry-original.txt"))
    first = asyncio.create_task(ReductionService(workspace).execute(plan.plan_id))
    await asyncio.sleep(0.1)
    second = asyncio.create_task(ReductionService(workspace).execute(plan.plan_id))
    await asyncio.sleep(0.1)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    result = await asyncio.wait_for(second, timeout=8)

    assert result.disposition == "succeeded"


@pytest.mark.anyio
async def test_predicate_timeout_is_reported_without_killing_reduction_worker(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _configure(tmp_path, "import time; time.sleep(10)")
    artifact_id = _original(workspace, tmp_path / "timeout-original.txt")
    plan = ReductionService(workspace).plan(
        PlanReductionRequest(
            original_artifact_id=artifact_id,
            partitioner="text_lines",
            predicate_workload="predicate",
            limits=ReductionLimits(
                max_attempts=2,
                wall_time_seconds=0.5,
                predicate_timeout_seconds=0.05,
            ),
        )
    )

    result = await ReductionService(workspace).execute(plan.plan_id)

    assert result.disposition == "inconclusive"
    assert result.attempts.attempted == 1
    assert result.attempts.timed_out == 1
