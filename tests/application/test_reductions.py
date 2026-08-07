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
