from __future__ import annotations

import stat
from pathlib import Path

import pytest

from flameox.application import CaptureService, ExecutionPolicy
from flameox.domain import (
    ArtifactKind,
    CapabilityPermissionStatus,
    CapabilityReport,
    CapabilityStatus,
    CaptureStatus,
    ExecutionStatus,
    ProbeKind,
)
from flameox.storage import ArtifactStore, RunStore, Workspace
from tests.support.capture import disable_containment


def _fake_perf(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys
import time

output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])
mode = next((value for value in sys.argv if value.startswith('mode=')), 'mode=ok')
if mode != 'mode=missing':
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b'' if mode == 'mode=empty' else b'fake profile')
if mode == 'mode=failed':
    print('collector failed', file=sys.stderr)
    raise SystemExit(7)
if mode == 'mode=timeout':
    time.sleep(10)
"""
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("mode", "execution", "capture", "sample_role", "quarantine"),
    (
        ("ok", ExecutionStatus.SUCCEEDED, CaptureStatus.REGISTERED, "primary", False),
        ("empty", ExecutionStatus.SUCCEEDED, CaptureStatus.FAILED, None, True),
        ("missing", ExecutionStatus.SUCCEEDED, CaptureStatus.FAILED, None, False),
        ("failed", ExecutionStatus.FAILED, CaptureStatus.FAILED, None, True),
        ("timeout", ExecutionStatus.TIMED_OUT, CaptureStatus.REGISTERED, "partial", False),
    ),
)
async def test_native_output_publication_gate_preserves_invalid_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    execution: ExecutionStatus,
    capture: CaptureStatus,
    sample_role: str | None,
    quarantine: bool,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    collector = tmp_path / "fake-perf"
    _fake_perf(collector)
    timeout = 0.1 if mode == "timeout" else 5
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.profile]
argv = ["python", "-c", "print('workload')", "mode={mode}"]
timeout_seconds = {timeout}
"""
    )
    report = CapabilityReport(
        adapter="perf",
        status=CapabilityStatus.AVAILABLE,
        executable=str(collector),
        version="fake-perf",
        supported_modes=("record",),
        supported_formats=("perf.data",),
        permission_status=CapabilityPermissionStatus.GRANTED,
        probe_kind=ProbeKind.ACTIVE,
    )
    service = CaptureService(workspace)
    monkeypatch.setattr(service.capabilities, "get", lambda _adapter: report)

    async def probe(_adapter: str, *, refresh: bool = False) -> CapabilityReport:
        assert refresh
        return report

    monkeypatch.setattr(service.capabilities, "probe", probe)
    plan = await service.plan(
        workload_name="profile",
        adapter="perf",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.execute(plan.plan_id)

    assert result.run.execution_status is execution
    assert result.run.capture_status is capture
    profile_registrations = [
        item for item in result.run.artifacts if item.kind is ArtifactKind.SAMPLE_PROFILE
    ]
    if sample_role is None:
        assert profile_registrations == []
    else:
        assert [item.role for item in profile_registrations] == [sample_role]
    if mode == "failed":
        stderr = next(item for item in result.run.artifacts if item.role == "stderr")
        assert ArtifactStore(workspace).get(stderr.artifact_id).payload_path.read_text() == (
            "collector failed\n"
        )
        assert any(item.source == "collector" for item in result.run.limitation_details)
    if quarantine:
        manifests = list(service.workspace.paths.quarantine.glob("*/manifest.json"))
        assert manifests
        quarantine_manifest = RunStore(workspace).read(plan.run_id)
        assert any(
            detail.code == "native_output_quarantined"
            for detail in quarantine_manifest.limitation_details
        )
    if mode == "missing":
        assert any(
            detail.code == "expected_output_invalid" for detail in result.run.limitation_details
        )
