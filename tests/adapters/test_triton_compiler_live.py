from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from flameox.application import CaptureService, ExecutionPolicy
from flameox.domain import ArtifactKind, CaptureStatus, ExecutionStatus
from flameox.storage import Workspace
from tests.support.capture import disable_containment


def _write_triton_workload(project: Path, *, python: str, script: str) -> None:
    (project / "flameox.toml").write_text(
        f"""
schema_version = 1

[workloads.compile]
argv = ["{python}", "{script}"]
cwd = "."
timeout_seconds = 60

[workloads.compile.environment]
TRITON_CACHE_DIR = "{project / "triton-cache"}"
"""
    )


def _triton_kernel_script(path: Path) -> None:
    path.write_text(
        """
import os
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

import torch
x = torch.randn(128, device="cuda")
y = torch.randn(128, device="cuda")
out = torch.empty_like(x)
add_kernel[(1,)](x, y, out, 128, BLOCK=128)
print(out.sum().item())
"""
    )


@pytest.mark.requires_triton
@pytest.mark.anyio
async def test_triton_compiler_live_capture_emits_manifest_with_ir(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    python = os.environ.get("FLAMEOX_TRITON_PYTHON", sys.executable)
    script = tmp_path / "triton_kernel.py"
    _triton_kernel_script(script)
    _write_triton_workload(tmp_path, python=python, script=str(script))
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="compile",
        adapter="triton.compiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_id)
    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    manifest_regs = [reg for reg in result.run.artifacts if reg.role == "kernel_build_manifest"]
    assert len(manifest_regs) == 1
    assert manifest_regs[0].kind is ArtifactKind.KERNEL_BUILD
    native_regs = [reg for reg in result.run.artifacts if reg.role.startswith("compiler_stage")]
    assert len(native_regs) >= 1
    extensions = {reg.display_name.rsplit(".", 1)[1] for reg in native_regs}
    assert "ttir" in extensions or "ptx" in extensions or "cubin" in extensions
