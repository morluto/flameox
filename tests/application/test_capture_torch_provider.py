from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application import CapabilityService, CaptureService, ExecutionPolicy
from flameox.domain import CapabilityStatus, ExecutionStatus
from flameox.storage import Workspace
from tests.support.capture import disable_containment


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_torch
@pytest.mark.process
async def test_torch_profiler_capture_registers_public_chrome_trace(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    if CapabilityService(workspace).get("torch.profiler").status is not CapabilityStatus.AVAILABLE:
        pytest.skip("PyTorch is not installed.")
    (tmp_path / "torch_workload.py").write_text(
        "import torch\n"
        "left = torch.ones((16, 16))\n"
        "right = torch.ones((16, 16))\n"
        "print(torch.mm(left, right).sum().item())\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.torch]
argv = ["python", "torch_workload.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="torch",
        adapter="torch.profiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_id)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    trace = next(
        registration
        for registration in result.run.artifacts
        if registration.kind.value == "execution_trace"
    )
    assert trace.display_name == "torch-trace.json"
