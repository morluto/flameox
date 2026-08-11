from __future__ import annotations

import os
from pathlib import Path

import pytest

from flameox.adapters.builtins import build_capture_invocation
from flameox.adapters.options import bind_adapter_options
from flameox.application import KernelBuildCaptureCollector
from flameox.domain import DomainError, ErrorCode
from flameox.storage import Workspace


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


@pytest.mark.parametrize(
    ("dump_exists", "exit_code", "expected_outcome"),
    [
        (True, 0, "inconclusive"),
        (False, 0, "inconclusive"),
        (True, 1, "failed"),
    ],
)
def test_collector_records_empty_or_failed_compilation(
    tmp_path: Path,
    dump_exists: bool,
    exit_code: int,
    expected_outcome: str,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    if dump_exists:
        dump_dir.mkdir()

    manifest, _, native_paths = KernelBuildCaptureCollector(workspace).collect(
        adapter="triton.compiler",
        dump_dir=dump_dir,
        output_root=tmp_path,
        workload_name="compile",
        exit_code=exit_code,
        producer_version="1",
        source_environment={},
    )

    assert manifest.outcome == expected_outcome
    assert native_paths == ()
    if exit_code == 0:
        assert any("no allowlisted" in limitation.lower() for limitation in manifest.limitations)


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


def test_collector_maps_cute_ir_debug_allowlist_to_mlir(tmp_path: Path) -> None:
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
    assert hardlink.lstat().st_nlink > 1
    assert native_paths == ()


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
