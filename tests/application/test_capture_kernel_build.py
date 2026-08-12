"""Kernel-build capture lifecycle integration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flameox.application import (
    ArtifactPipelineService,
    CaptureService,
    ExecutionPolicy,
)
from flameox.domain import ArtifactKind, CaptureStatus, DomainError, ErrorCode, ExecutionStatus
from flameox.storage import ArtifactStore, Workspace
from tests.support.capture import disable_containment


def _write_workload(
    project: Path,
    *,
    script: str,
    environment: dict[str, str] | None = None,
) -> None:
    env_section = ""
    if environment:
        env_lines = "\n".join(f'  {key} = "{value}"' for key, value in environment.items())
        env_section = f"\n[workloads.compile.environment]\n{env_lines}"
    (project / "flameox.toml").write_text(
        f"""
schema_version = 1

[workloads.compile]
argv = ["python", "{script}"]
cwd = "."
timeout_seconds = 30{env_section}
"""
    )


def _dump_script(path: Path, *, exit_code: int = 0) -> None:
    """A fake compiler that writes IR files to TRITON_DUMP_DIR and exits."""
    path.write_text(
        f"""
import os, sys, pathlib
dump = pathlib.Path(os.environ["TRITON_DUMP_DIR"])
dump.mkdir(parents=True, exist_ok=True)
(dump / "kernel.ttir").write_text("ttir content")
(dump / "kernel.ptx").write_text("ptx content")
(dump / "kernel.cubin").write_bytes(b"\\x00\\x01\\x02\\x03")
sys.exit({exit_code})
"""
    )


@pytest.mark.anyio
async def test_capture_plan_rejects_conflicting_compiler_environment(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script)
    _write_workload(
        tmp_path,
        script=str(script),
        environment={"TRITON_DUMP_DIR": "/elsewhere"},
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    with pytest.raises(DomainError) as error:
        await service.plan(
            workload_name="compile",
            adapter="triton.compiler",
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
    assert error.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    assert "conflict" in error.value.message.lower()


@pytest.mark.anyio
async def test_capture_plan_accepts_identical_compiler_environment(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script)
    _write_workload(
        tmp_path,
        script=str(script),
        environment={"TRITON_KERNEL_DUMP": "1"},
    )
    disable_containment(workspace)

    plan = await CaptureService(workspace).plan(
        workload_name="compile",
        adapter="triton.compiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    assert plan.collector_environment["TRITON_KERNEL_DUMP"] == "1"


@pytest.mark.anyio
async def test_triton_compiler_capture_emits_manifest_with_native_artifacts(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script)
    _write_workload(tmp_path, script=str(script))
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
    manifest_reg = next(reg for reg in result.run.artifacts if reg.role == "kernel_build_manifest")
    assert manifest_reg.kind is ArtifactKind.KERNEL_BUILD
    native_regs = [reg for reg in result.run.artifacts if reg.role.startswith("compiler_stage")]
    assert len(native_regs) == 3
    extensions = {reg.display_name.rsplit(".", 1)[1] for reg in native_regs}
    assert extensions == {"ttir", "ptx", "cubin"}
    media_types = {reg.display_name.rsplit(".", 1)[1]: reg.media_type for reg in native_regs}
    assert media_types == {
        "ttir": "text/plain",
        "ptx": "text/plain",
        "cubin": "application/octet-stream",
    }
    pipelines = ArtifactPipelineService(workspace).pipelines.list()
    assert len(pipelines) == 1
    assert pipelines[0].run_id == result.run.run_id
    assert {stage.artifact_id for stage in pipelines[0].stages} == {
        registration.artifact_id for registration in native_regs
    }
    artifact = ArtifactStore(workspace).get(manifest_reg.artifact_id)
    manifest = json.loads(artifact.payload_path.read_text())
    assert manifest["schema_version"] == "flameox.kernel-build.v1"
    assert manifest["producer"] == "triton"
    assert manifest["workload_identity"] == "compile"
    assert manifest["outcome"] == "succeeded"
    assert len(manifest["stages"]) == 3
    native_artifacts = {
        registration.display_name: ArtifactStore(workspace)
        .get(registration.artifact_id)
        .payload_path
        for registration in native_regs
    }
    for stage in manifest["stages"]:
        declaration = stage["artifact"]
        assert declaration is not None
        payload = native_artifacts[declaration["path"]].read_bytes()
        assert declaration["byte_length"] == len(payload)
        assert declaration["sha256"] == hashlib.sha256(payload).hexdigest()
    ttir_stage = next(stage for stage in manifest["stages"] if stage["name"] == "ttir")
    assert ttir_stage["format_schema"] == "triton-ttir"


@pytest.mark.anyio
async def test_triton_compiler_capture_preserves_manifest_on_nonzero_exit(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script, exit_code=1)
    _write_workload(tmp_path, script=str(script))
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="compile",
        adapter="triton.compiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_token)
    assert result.run.execution_status is ExecutionStatus.FAILED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    manifest_regs = [reg for reg in result.run.artifacts if reg.role == "kernel_build_manifest"]
    assert len(manifest_regs) == 1
    native_regs = [reg for reg in result.run.artifacts if reg.role.startswith("compiler_stage")]
    assert len(native_regs) == 3


@pytest.mark.anyio
async def test_cute_compiler_capture_emits_manifest(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    script.write_text(
        """
import os, sys, pathlib
dump = pathlib.Path(os.environ["CUTE_DSL_DUMP_DIR"])
dump.mkdir(parents=True, exist_ok=True)
(dump / ("cute_dsl_" + "long_kernel_identity_" * 6 + ".mlir")).write_text("cute ir")
(dump / "kernel.ptx").write_text("ptx")
sys.exit(0)
"""
    )
    _write_workload(tmp_path, script=str(script))
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="compile",
        adapter="cute.compiler",
        adapter_options={"keep_allowlist": ["ir", "ptx"]},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_token)
    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    native_regs = [reg for reg in result.run.artifacts if reg.role.startswith("compiler_stage")]
    assert len(native_regs) == 2
    assert all(len(reg.role) <= 100 for reg in native_regs)
    assert any("long_kernel_identity" in reg.display_name for reg in native_regs)
    pipelines = ArtifactPipelineService(workspace).pipelines.list()
    assert len(pipelines) == 1
    assert {stage.artifact_id for stage in pipelines[0].stages} == {
        registration.artifact_id for registration in native_regs
    }
