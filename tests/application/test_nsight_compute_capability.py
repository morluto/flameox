from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from flameox.application import CapabilityService
from flameox.domain import CapabilityStatus, ProcessResult, process_termination_from_returncode
from flameox.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    ProcessContainment,
    SubprocessBroker,
)
from flameox.storage import Workspace


class _NcuProbeBroker(SubprocessBroker):
    def __init__(
        self,
        *,
        probe_stderr: bytes = (
            b"==ERROR== ERR_NVGPUCTRPERM - Permission to access GPU counters denied\n"
        ),
    ) -> None:
        self.requests: list[ExecutionRequest] = []
        self.probe_stderr = probe_stderr

    async def run(self, request: ExecutionRequest, **_: Any) -> ExecutionOutcome:
        self.requests.append(request)
        if len(self.requests) == 1:
            stdout = b"NVIDIA (R) Nsight Compute\nVersion 2026.2.1.0\n"
            stderr = b""
            exit_code = 0
        else:
            stdout = b""
            stderr = self.probe_stderr
            exit_code = 1
        return ExecutionOutcome(
            process=ProcessResult(
                termination=process_termination_from_returncode(exit_code),
                cleanup_complete=True,
            ),
            stdout=stdout,
            stderr=stderr,
            resolved_executable=Path(request.argv[0]),
            containment=ProcessContainment.PROCESS_GROUP,
        )


@pytest.mark.anyio
async def test_ncu_probe_maps_counter_denial_to_permission_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    broker = _NcuProbeBroker()
    service = CapabilityService(workspace, broker=broker)
    monkeypatch.setattr(
        service,
        "_resolved_executable",
        lambda adapter, executable: "/usr/bin/ncu" if adapter == "nsight.compute" else None,
    )
    monkeypatch.setattr(
        "flameox.application.capabilities.find_ncu_report_interface",
        lambda **_: Path("/opt/nvidia/nsight-compute/extras/python/ncu_report.py"),
    )
    monkeypatch.setattr(service, "_nvidia_counter_access_restriction", lambda: None)

    report = await service.probe("nsight.compute")

    assert report.status is CapabilityStatus.PERMISSION_REQUIRED
    assert report.permission_status == "denied"
    assert report.version == "Version 2026.2.1.0"
    assert "ERR_NVGPUCTRPERM" in report.limitations[0]
    assert "will not change system privileges" in report.remediation[0]
    assert broker.requests[1].argv[1:6] == (
        "--set",
        "basic",
        "--launch-count",
        "1",
        "--export",
    )
    assert broker.requests[1].resource_policy is not None
    assert broker.requests[1].resource_policy.staging_root is not None
    assert not broker.requests[1].resource_policy.staging_root.exists()


@pytest.mark.anyio
async def test_ncu_probe_reports_non_permission_failure_as_degraded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    service = CapabilityService(
        workspace,
        broker=_NcuProbeBroker(probe_stderr=b"==ERROR== CUDA driver initialization failed\n"),
    )
    monkeypatch.setattr(
        service,
        "_resolved_executable",
        lambda adapter, executable: "/usr/bin/ncu" if adapter == "nsight.compute" else None,
    )
    monkeypatch.setattr(
        "flameox.application.capabilities.find_ncu_report_interface",
        lambda **_: Path("/opt/nvidia/nsight-compute/extras/python/ncu_report.py"),
    )
    monkeypatch.setattr(service, "_nvidia_counter_access_restriction", lambda: None)

    report = await service.probe("nsight.compute")

    assert report.status is CapabilityStatus.DEGRADED
    assert report.permission_status == "unknown"
    assert "CUDA driver initialization failed" in report.limitations[0]
