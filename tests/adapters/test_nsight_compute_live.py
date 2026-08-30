import json
import os
import shutil
from pathlib import Path

import pytest

from flameox.adapters.nsight_compute import NsightComputeExtractor, find_ncu_report_interface
from flameox.application.capabilities import CapabilityService
from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.domain import (
    ArtifactKind,
    CapabilityStatus,
    CaptureStatus,
    ExecutionStatus,
)
from flameox.storage import Workspace
from tests.support.capture import disable_containment

pytestmark = [
    pytest.mark.integration,
    pytest.mark.optional,
    pytest.mark.process,
    pytest.mark.serial,
    pytest.mark.requires_ncu,
]


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_ncu
@pytest.mark.requires_nvbench
async def test_managed_capture_collects_and_extracts_live_counters(tmp_path: Path) -> None:
    workload = os.environ.get("FLAMEOX_NVBENCH_EXECUTABLE")
    assert workload is not None
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    (tmp_path / "flameox.toml").write_text(
        f"""
[workloads.profile]
argv = [{json.dumps(workload)}, "--timeout", "0.1", "--min-time", "1e-5"]
timeout_seconds = 120
"""
    )

    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="profile",
        adapter="nsight.compute",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_token)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    assert any(
        registration.kind is ArtifactKind.KERNEL_PROFILE and registration.role == "primary"
        for registration in result.run.artifacts
    )
    extracted = NsightComputeExtractor(workspace).extract(result.run.run_id)
    assert extracted.action_count >= 1
    assert extracted.metric_count >= 1


@pytest.mark.optional
@pytest.mark.requires_ncu
def test_installed_official_interface_extracts_bundled_sample(tmp_path: Path) -> None:
    executable = shutil.which("ncu")
    assert executable is not None
    interface = find_ncu_report_interface(executable=Path(executable))
    assert interface is not None
    sample = interface.parent.parent / "samples" / "instructionMix" / "sobelFloat.ncu-rep"
    if not sample.is_file():
        pytest.skip(f"selected Nsight Compute installation has no bundled sample at {sample}")
    workspace = Workspace.initialize(tmp_path)
    run_id = (
        ImportService(workspace)
        .import_artifact(
            ImportArtifactRequest(
                path=sample,
                kind=ArtifactKind.KERNEL_PROFILE,
                producer="nsight.compute",
                allow_external_path=True,
            )
        )
        .run.run_id
    )

    result = NsightComputeExtractor(workspace).extract(run_id)

    assert result.report_version
    assert result.action_count >= 1
    assert result.metric_count >= 1


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_ncu
async def test_local_driver_policy_reports_counter_permission_requirement(tmp_path: Path) -> None:
    service = CapabilityService(Workspace.initialize(tmp_path))
    report = await service.probe("nsight.compute", refresh=True)
    if report.status is not CapabilityStatus.PERMISSION_REQUIRED:
        pytest.skip("local NVIDIA driver does not expose a restricted counter policy")

    assert report.status is CapabilityStatus.PERMISSION_REQUIRED
    assert report.permission_status == "denied"
    assert "ERR_NVGPUCTRPERM" in report.limitations[0]
