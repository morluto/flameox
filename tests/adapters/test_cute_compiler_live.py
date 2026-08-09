from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from flameox.application import CaptureService, ExecutionPolicy
from flameox.domain import (
    ArtifactKind,
    CapabilityReport,
    CapabilityStatus,
    CaptureStatus,
    ExecutionStatus,
)
from flameox.storage import Workspace
from tests.support.capture import disable_containment


@pytest.mark.requires_cute
@pytest.mark.anyio
async def test_configured_cute_workload_emits_kernel_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = os.environ.get("FLAMEOX_CUTE_WORKLOAD")
    assert workload is not None
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.compile]
argv = [{json.dumps(workload)}]
timeout_seconds = 120
"""
    )
    report = CapabilityReport(
        adapter="cute.compiler",
        status=CapabilityStatus.AVAILABLE,
        version="configured-workload",
        supported_modes=("compile",),
        supported_formats=("kernel-build-manifest",),
        permission_status="granted",
        probe_kind="passive",
    )
    service = CaptureService(workspace)
    monkeypatch.setattr(service.capabilities, "get", lambda _adapter: report)

    async def probe(_adapter: str, *, refresh: bool = False) -> CapabilityReport:
        assert refresh
        return report

    monkeypatch.setattr(service.capabilities, "probe", probe)

    plan = await service.plan(
        workload_name="compile",
        adapter="cute.compiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_id)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    assert any(
        registration.kind is ArtifactKind.KERNEL_BUILD
        and registration.role.startswith("compiler_stage")
        for registration in result.run.artifacts
    )
