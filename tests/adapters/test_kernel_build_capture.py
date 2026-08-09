from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from flameox.adapters.builtins import build_capture_invocation
from flameox.adapters.options import bind_adapter_options
from flameox.application import (
    ArtifactPipelineService,
    CaptureService,
    ExecutionPolicy,
    KernelBuildCaptureCollector,
)
from flameox.domain import ArtifactKind, CaptureStatus, DomainError, ErrorCode, ExecutionStatus
from flameox.storage import Workspace
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


def _dump_script(path: Path, dump_dir: Path, *, exit_code: int = 0) -> None:
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


def test_triton_compiler_capture_invocation_sets_env_vars(tmp_path: Path) -> None:
    invocation = build_capture_invocation(
        "triton.compiler",
        ("python", "compile.py"),
        tmp_path / "output",
        executable=None,
        options={"dump_subdir": "triton-dumps", "kernel_dump": True},
    )
    assert invocation.argv == ("python", "compile.py")
    assert invocation.environment["TRITON_KERNEL_DUMP"] == "1"
    assert "triton-dumps" in invocation.environment["TRITON_DUMP_DIR"]
    assert "TRITON_REPRODUCER_PATH" not in invocation.environment


def test_triton_compiler_reproducer_option_sets_env_var(tmp_path: Path) -> None:
    invocation = build_capture_invocation(
        "triton.compiler",
        ("python", "compile.py"),
        tmp_path / "output",
        executable=None,
        options={"reproducer_filename": "triton-reproducer.mlir"},
    )
    assert "TRITON_REPRODUCER_PATH" in invocation.environment
    assert invocation.environment["TRITON_REPRODUCER_PATH"].endswith("triton-reproducer.mlir")


def test_cute_compiler_capture_invocation_sets_env_vars(tmp_path: Path) -> None:
    invocation = build_capture_invocation(
        "cute.compiler",
        ("python", "compile.py"),
        tmp_path / "output",
        executable=None,
        options={"keep_allowlist": ("ir", "ptx")},
    )
    assert invocation.argv == ("python", "compile.py")
    assert "CUTE_DSL_DUMP_DIR" in invocation.environment
    assert invocation.environment["CUTE_DSL_KEEP"] == "ir,ptx"


def test_cute_compiler_rejects_all_mixed_with_other_tokens(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as error:
        bind_adapter_options(
            "cute.compiler",
            {"keep_allowlist": ["all", "ptx"]},
            project_root=tmp_path,
        )
    assert error.value.code is ErrorCode.INVALID_CAPTURE_PLAN


def test_cute_compiler_rejects_duplicate_keep_allowlist(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as error:
        bind_adapter_options(
            "cute.compiler",
            {"keep_allowlist": ["ptx", "ptx"]},
            project_root=tmp_path,
        )
    assert error.value.code is ErrorCode.INVALID_CAPTURE_PLAN


def test_triton_compiler_rejects_invalid_dump_subdir(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as error:
        bind_adapter_options(
            "triton.compiler",
            {"dump_subdir": "../escape"},
            project_root=tmp_path,
        )
    assert error.value.code is ErrorCode.INVALID_CAPTURE_PLAN


@pytest.mark.anyio
async def test_env_conflict_rejected_unless_identical(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script, tmp_path / "triton-dumps")
    _write_workload(
        tmp_path,
        script=str(script),
        environment={"TRITON_DUMP_DIR": "/elsewhere"},
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    with pytest.raises(DomainError) as error:
        await service._adapter_command(
            "triton.compiler",
            service.workloads.resolve("compile").command,
            tmp_path / "staging",
            options=bind_adapter_options(
                "triton.compiler",
                {},
                project_root=tmp_path,
            ),
        )
    assert error.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    assert "conflict" in error.value.message.lower()


@pytest.mark.anyio
async def test_env_conflict_identical_value_accepted(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    staging = tmp_path / "staging"
    invocation = build_capture_invocation(
        "triton.compiler",
        ("python", "compile.py"),
        staging,
        executable=None,
        options={},
    )
    dump_dir = invocation.environment["TRITON_DUMP_DIR"]
    script = tmp_path / "compile.py"
    _dump_script(script, Path(dump_dir))
    _write_workload(
        tmp_path,
        script=str(script),
        environment={"TRITON_DUMP_DIR": dump_dir},
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    binding = await service._adapter_command(
        "triton.compiler",
        service.workloads.resolve("compile").command,
        staging,
        options=bind_adapter_options(
            "triton.compiler",
            {},
            project_root=tmp_path,
        ),
    )
    assert binding.environment["TRITON_DUMP_DIR"] == dump_dir


@pytest.mark.anyio
async def test_triton_compiler_capture_emits_manifest_with_native_artifacts(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script, tmp_path / "triton-dumps")
    _write_workload(tmp_path, script=str(script))
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


@pytest.mark.anyio
async def test_triton_compiler_capture_preserves_manifest_on_nonzero_exit(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script, tmp_path / "triton-dumps", exit_code=1)
    _write_workload(tmp_path, script=str(script))
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="compile",
        adapter="triton.compiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_id)
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
    result = await service.execute(plan.plan_id)
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


@pytest.mark.anyio
async def test_kernel_build_manifest_content_matches_artifacts(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    script = tmp_path / "compile.py"
    _dump_script(script, tmp_path / "triton-dumps")
    _write_workload(tmp_path, script=str(script))
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="compile",
        adapter="triton.compiler",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_id)
    from flameox.storage import ArtifactStore

    manifest_reg = next(reg for reg in result.run.artifacts if reg.role == "kernel_build_manifest")
    artifact = ArtifactStore(workspace).get(manifest_reg.artifact_id)
    manifest = json.loads(artifact.payload_path.read_text())
    assert manifest["schema_version"] == "flameox.kernel-build.v1"
    assert manifest["producer"] == "triton"
    assert manifest["workload_identity"] == "compile"
    assert manifest["outcome"] == "succeeded"
    assert len(manifest["stages"]) == 3
    for stage in manifest["stages"]:
        assert stage["artifact"] is not None
        assert len(stage["artifact"]["sha256"]) == 64
    ttir_stage = next(s for s in manifest["stages"] if s["name"] == "ttir")
    assert ttir_stage["format"] == "ttir"
    assert ttir_stage["format_schema"] == "triton-ttir"
    ptx_stage = next(s for s in manifest["stages"] if s["name"] == "ptx")
    assert ptx_stage["predecessor"] is None
    assert any("predecessor lineage" in item for item in manifest["limitations"])


def test_collector_rejects_symlink_in_dump_dir(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    dump_dir.mkdir()
    target = tmp_path / "real.ttir"
    target.write_text("real")
    link = dump_dir / "link.ttir"
    link.symlink_to(target)
    collector = KernelBuildCaptureCollector(workspace)
    _, _, native_paths = collector.collect(
        adapter="triton.compiler",
        dump_dir=dump_dir,
        output_root=tmp_path,
        workload_name="compile",
        exit_code=0,
        producer_version="1",
        source_environment={},
    )
    assert native_paths == ()


def test_collector_ignores_non_allowlisted_extensions(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    dump_dir.mkdir()
    (dump_dir / "kernel.ttir").write_text("ttir")
    (dump_dir / "readme.txt").write_text("ignore me")
    (dump_dir / "data.bin").write_bytes(b"binary")
    collector = KernelBuildCaptureCollector(workspace)
    _, _, native_paths = collector.collect(
        adapter="triton.compiler",
        dump_dir=dump_dir,
        output_root=tmp_path,
        workload_name="compile",
        exit_code=0,
        producer_version="1",
        source_environment={},
    )
    assert len(native_paths) == 1
    assert native_paths[0].suffix == ".ttir"


def test_collector_emits_inconclusive_when_no_artifacts(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    dump_dir.mkdir()
    collector = KernelBuildCaptureCollector(workspace)
    manifest, _, native_paths = collector.collect(
        adapter="triton.compiler",
        dump_dir=dump_dir,
        output_root=tmp_path,
        workload_name="compile",
        exit_code=0,
        producer_version="1",
        source_environment={},
    )
    assert manifest.outcome == "inconclusive"
    assert native_paths == ()
    assert any("no allowlisted" in lim.lower() for lim in manifest.limitations)


def test_collector_emits_failed_when_nonzero_and_no_artifacts(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    dump_dir.mkdir()
    collector = KernelBuildCaptureCollector(workspace)
    manifest, _, _ = collector.collect(
        adapter="triton.compiler",
        dump_dir=dump_dir,
        output_root=tmp_path,
        workload_name="compile",
        exit_code=1,
        producer_version="1",
        source_environment={},
    )
    assert manifest.outcome == "failed"


def test_collector_handles_missing_dump_dir(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    collector = KernelBuildCaptureCollector(workspace)
    manifest, _, native_paths = collector.collect(
        adapter="triton.compiler",
        dump_dir=tmp_path / "nonexistent",
        output_root=tmp_path,
        workload_name="compile",
        exit_code=0,
        producer_version="1",
        source_environment={},
    )
    assert manifest.outcome == "inconclusive"
    assert native_paths == ()


def test_collector_cute_keep_allowlist_filters_extensions(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "cute-dumps"
    dump_dir.mkdir()
    (dump_dir / "kernel.mlir").write_text("ir")
    (dump_dir / "kernel.ptx").write_text("ptx")
    (dump_dir / "kernel.cubin").write_bytes(b"cubin")
    collector = KernelBuildCaptureCollector(workspace)
    _, _, native_paths = collector.collect(
        adapter="cute.compiler",
        dump_dir=dump_dir,
        output_root=tmp_path,
        workload_name="compile",
        exit_code=0,
        producer_version="1",
        source_environment={},
        cute_keep_allowlist=("ptx",),
    )
    assert len(native_paths) == 1
    assert native_paths[0].suffix == ".ptx"


def test_collector_recognizes_cute_multi_suffix_debug_ir(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "cute-dumps"
    dump_dir.mkdir()
    debug_ir = dump_dir / "kernel.mlir"
    debug_ir.write_text("debug ir")

    manifest, _, native_paths = KernelBuildCaptureCollector(workspace).collect(
        adapter="cute.compiler",
        dump_dir=dump_dir,
        output_root=tmp_path,
        workload_name="compile",
        exit_code=0,
        producer_version="1",
        source_environment={"CUTE_DSL_DUMP_DIR": str(dump_dir)},
        cute_keep_allowlist=("ir-debug",),
    )

    assert native_paths == (debug_ir,)
    assert manifest.stages[0].format == "cute_dsl_ir"
    assert manifest.source_environment["CUTE_DSL_DUMP_DIR"] == "<staging>/cute-dumps"


def test_collector_hardlinks_rejected(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    dump_dir.mkdir()
    original = tmp_path / "original.ttir"
    original.write_text("content")
    hardlink = dump_dir / "link.ttir"
    os.link(original, hardlink)
    collector = KernelBuildCaptureCollector(workspace)
    _, _, native_paths = collector.collect(
        adapter="triton.compiler",
        dump_dir=dump_dir,
        output_root=tmp_path,
        workload_name="compile",
        exit_code=0,
        producer_version="1",
        source_environment={},
    )
    if hardlink.lstat().st_nlink > 1:
        assert native_paths == ()
    else:
        assert len(native_paths) == 1


def test_collector_inventories_amdgcn_and_hsaco_extensions(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    dump_dir.mkdir()
    (dump_dir / "kernel.amdgcn").write_text("amdgcn source")
    (dump_dir / "kernel.hsaco").write_bytes(b"hsaco binary")
    collector = KernelBuildCaptureCollector(workspace)
    manifest, _, native_paths = collector.collect(
        adapter="triton.compiler",
        dump_dir=dump_dir,
        output_root=tmp_path,
        workload_name="compile",
        exit_code=0,
        producer_version="1",
        source_environment={},
    )
    assert len(native_paths) == 2
    extensions = {p.suffix for p in native_paths}
    assert extensions == {".amdgcn", ".hsaco"}
    formats = {s.format for s in manifest.stages if s.artifact is not None}
    assert "amdgcn" in formats
    assert "hsaco" in formats


def test_collector_inventories_reproducer_file_outside_dump_dir(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    dump_dir.mkdir()
    (dump_dir / "kernel.ttir").write_text("ttir")
    reproducer = tmp_path / "triton-reproducer.mlir"
    reproducer.write_text("reproducer content")
    collector = KernelBuildCaptureCollector(workspace)
    manifest, _, native_paths = collector.collect(
        adapter="triton.compiler",
        dump_dir=dump_dir,
        output_root=tmp_path,
        workload_name="compile",
        exit_code=0,
        producer_version="1",
        source_environment={},
        reproducer_path=reproducer,
    )
    assert len(native_paths) == 2
    assert reproducer in native_paths
    reproducer_stage = next(s for s in manifest.stages if s.format == "reproducer")
    assert reproducer_stage.artifact is not None
    assert reproducer_stage.artifact.path == "triton-reproducer.mlir"


def test_collector_does_not_infer_predecessor_lineage_from_paths(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    kernel_a = dump_dir / "kernel_a"
    kernel_b = dump_dir / "kernel_b"
    kernel_a.mkdir(parents=True)
    kernel_b.mkdir(parents=True)
    (kernel_a / "a.ttir").write_text("a ttir")
    (kernel_a / "a.ptx").write_text("a ptx")
    (kernel_b / "b.ttir").write_text("b ttir")
    (kernel_b / "b.ptx").write_text("b ptx")
    collector = KernelBuildCaptureCollector(workspace)
    manifest, _, _ = collector.collect(
        adapter="triton.compiler",
        dump_dir=dump_dir,
        output_root=tmp_path,
        workload_name="compile",
        exit_code=0,
        producer_version="1",
        source_environment={},
    )
    available = [stage for stage in manifest.stages if stage.artifact is not None]
    assert [stage.format for stage in available] == ["ttir", "ttir", "ptx", "ptx"]
    assert all(stage.predecessor is None for stage in available)
    assert any("predecessor lineage" in item for item in manifest.limitations)
