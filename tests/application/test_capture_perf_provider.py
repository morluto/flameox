from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application import CapabilityService, CaptureService, ExecutionPolicy
from flameox.domain import ExecutionStatus
from flameox.storage import Workspace
from tests.support.capture import disable_containment


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_perf
@pytest.mark.process
async def test_perf_capture_registers_native_profile_when_kernel_allows_sampling(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    if CapabilityService(workspace).get("perf").executable is None:
        pytest.skip("perf is not installed.")
    (tmp_path / "busy.py").write_text(
        "total = 0\nfor value in range(1000000):\n    total += value * value\nprint(total)\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.busy]
argv = ["python", "busy.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="busy",
        adapter="perf",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_id)
    if result.run.execution_status is not ExecutionStatus.SUCCEEDED:
        pytest.skip("The host kernel does not permit perf sampling for this process.")

    assert any(
        registration.kind.value == "sample_profile" and registration.display_name == "perf.data"
        for registration in result.run.artifacts
    )
