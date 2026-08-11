from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from flameox.application import CapabilityService
from flameox.domain import CapabilityStatus, ProcessResult
from flameox.execution import ExecutionOutcome, ExecutionRequest, SubprocessBroker
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
            process=ProcessResult(exit_code=exit_code, cleanup_complete=True),
            stdout=stdout,
            stderr=stderr,
            resolved_executable=Path(request.argv[0]),
            containment="process_group",
        )


def _install_ncu_with_report_interface(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installation = project / "nsight-compute"
    executable = installation / "bin" / "ncu"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    interface = installation / "extras" / "python" / "ncu_report.py"
    interface.parent.mkdir(parents=True)
    interface.write_text("# official interface location fixture\n")
    monkeypatch.setenv("PATH", f"{executable.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr("flameox.application.capabilities.os.geteuid", lambda: 0)


@pytest.mark.anyio
async def test_ncu_probe_maps_counter_denial_to_permission_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    broker = _NcuProbeBroker()
    service = CapabilityService(workspace, broker=broker)
    _install_ncu_with_report_interface(tmp_path, monkeypatch)

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
    _install_ncu_with_report_interface(tmp_path, monkeypatch)

    report = await service.probe("nsight.compute")

    assert report.status is CapabilityStatus.DEGRADED
    assert report.permission_status == "unknown"
    assert "CUDA driver initialization failed" in report.limitations[0]
