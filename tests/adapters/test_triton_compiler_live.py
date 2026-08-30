from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.domain import ArtifactKind, CaptureStatus, ExecutionStatus
from flameox.storage import Workspace
from tests.support.capture import disable_containment

pytestmark = [
    pytest.mark.integration,
    pytest.mark.optional,
    pytest.mark.process,
    pytest.mark.serial,
    pytest.mark.requires_triton,
]


def _write_triton_workload(project: Path, *, python: str, script: str) -> None:
    (project / "flameox.toml").write_text(
        f"""

[workloads.compile]
argv = ["{python}", "{script}"]
cwd = "."
timeout_seconds = 60

[workloads.compile.environment]
TRITON_CACHE_DIR = "{project / "triton-cache"}"
"""
    )


@pytest.mark.requires_triton
@pytest.mark.anyio
async def test_triton_compiler_live_capture_emits_manifest_with_ir(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    python = os.environ.get("FLAMEOX_TRITON_PYTHON", sys.executable)
    script = tmp_path / "triton_kernel.py"
    fixture = Path(__file__).parent.parent / "fixtures" / "triton" / "vector_add.py.txt"
    shutil.copyfile(fixture, script)
    _write_triton_workload(tmp_path, python=python, script=str(script))
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="compile",
        adapter="triton.compiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_token)
    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    manifest_regs = [reg for reg in result.run.artifacts if reg.role == "kernel_build_manifest"]
    assert len(manifest_regs) == 1
    assert manifest_regs[0].kind is ArtifactKind.KERNEL_BUILD
    native_regs = [reg for reg in result.run.artifacts if reg.role == "compiler_output"]
    assert len(native_regs) >= 1
    extensions = {reg.display_name.rsplit(".", 1)[1] for reg in native_regs}
    assert "ttir" in extensions or "ptx" in extensions or "cubin" in extensions
