from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application import CaptureService, ExecutionPolicy
from flameox.domain import ExecutionStatus
from flameox.storage import Workspace
from tests.support.capture import disable_containment


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_coverage
@pytest.mark.process
async def test_coverage_capture_uses_supported_launcher_and_registers_native_data(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "workload.py").write_text("value = sum(range(10))\nprint(value)\n")
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.script]
argv = ["python", "workload.py"]
cwd = "."
timeout_seconds = 10
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="script",
        adapter="coverage",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_token)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert any(item.kind.value == "execution_coverage" for item in result.run.artifacts)
