from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import JsonValue

from flameox.adapters.builtins import build_capture_invocation
from flameox.adapters.kernel_build import KernelBuildManifest
from flameox.adapters.options import bind_adapter_options
from flameox.application.kernel_builds import KernelBuildCaptureCollector
from flameox.domain import DomainError, ErrorCode
from flameox.storage import Workspace

pytestmark = pytest.mark.unit


def test_triton_compiler_capture_invocation_sets_env_vars(tmp_path: Path) -> None:
    invocation = build_capture_invocation(
        "triton.compiler",
        ("python", "compile.py"),
        tmp_path / "output",
        executable=None,
        options={"dump_subdir": "triton-dumps", "kernel_dump": True},
    )

    assert invocation.argv[:2] == ("python", "-c")
    assert any(argument.endswith("triton-autotune.jsonl") for argument in invocation.argv)
    assert invocation.argv[-2:] == ("--", "compile.py")
    assert invocation.implementation_id is not None
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

    assert invocation.environment["TRITON_REPRODUCER_PATH"].endswith("triton-reproducer.mlir")


def test_triton_compiler_capture_rejects_inline_python(tmp_path: Path) -> None:
    with pytest.raises(DomainError, match="Inline Python commands"):
        build_capture_invocation(
            "triton.compiler",
            ("python", "-c", "print('kernel')"),
            tmp_path / "output",
            executable=None,
        )


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


@pytest.mark.parametrize(
    ("adapter", "options"),
    [
        ("cute.compiler", {"keep_allowlist": ["all", "ptx"]}),
        ("cute.compiler", {"keep_allowlist": ["ptx", "ptx"]}),
        ("triton.compiler", {"dump_subdir": "../escape"}),
    ],
)
def test_kernel_build_options_reject_invalid_bounded_values(
    tmp_path: Path,
    adapter: str,
    options: dict[str, JsonValue],
) -> None:
    with pytest.raises(DomainError) as error:
        bind_adapter_options(adapter, options, project_root=tmp_path)

    assert error.value.code is ErrorCode.INVALID_CAPTURE_PLAN


def _collect(
    workspace: Workspace,
    *,
    adapter: str,
    dump_dir: Path,
    exit_code: int = 0,
    cute_keep_allowlist: tuple[str, ...] | None = None,
    reproducer_path: Path | None = None,
) -> tuple[KernelBuildManifest, Path, tuple[Path, ...], tuple[str, ...]]:
    return KernelBuildCaptureCollector(workspace).collect(
        adapter=adapter,
        dump_dir=dump_dir,
        output_root=dump_dir.parent,
        exit_code=exit_code,
        cute_keep_allowlist=cute_keep_allowlist,
        reproducer_path=reproducer_path,
    )


def test_collector_rejects_symlink_and_hardlink_outputs(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    dump_dir.mkdir()
    target = tmp_path / "real.ttir"
    target.write_text("real")
    (dump_dir / "link.ttir").symlink_to(target)
    os.link(target, dump_dir / "hardlink.ttir")

    _, _, native_paths, limitations = _collect(
        workspace,
        adapter="triton.compiler",
        dump_dir=dump_dir,
    )

    assert native_paths == ()
    assert any("non-linked" in limitation for limitation in limitations)


def test_collector_preserves_groups_by_each_triton_artifact_parent(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    paths = (
        dump_dir / "root.ttir",
        dump_dir / "root.ptx",
        dump_dir / "source-hash-a" / "a.ttir",
        dump_dir / "source-hash-a" / "nested" / "a.ptx",
        dump_dir / "source-hash-b" / "b.ttir",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.suffix)

    manifest, _, native_paths, _ = _collect(
        workspace,
        adapter="triton.compiler",
        dump_dir=dump_dir,
    )

    assert native_paths == tuple(sorted(paths))
    assert [group.path for group in manifest.native_groups] == [
        "triton-dumps",
        "triton-dumps/source-hash-a",
        "triton-dumps/source-hash-a/nested",
        "triton-dumps/source-hash-b",
    ]
    assert [artifact.path for artifact in manifest.native_groups[-1].artifacts] == [
        "triton-dumps/source-hash-b/b.ttir",
    ]
    assert set(manifest.model_dump(mode="json")) == {
        "producer",
        "native_groups",
        "attachments",
    }


def test_collector_reports_empty_compilation_inline_not_in_the_manifest(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    dump_dir.mkdir()

    manifest, _, native_paths, limitations = _collect(
        workspace,
        adapter="triton.compiler",
        dump_dir=dump_dir,
    )

    assert manifest.native_groups == ()
    assert native_paths == ()
    assert any("produced no allowlisted" in limitation for limitation in limitations)


def test_collector_cute_keep_allowlist_filters_extensions(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "cute-dumps"
    dump_dir.mkdir()
    (dump_dir / "kernel.mlir").write_text("ir")
    (dump_dir / "kernel.ptx").write_text("ptx")
    (dump_dir / "kernel.cubin").write_bytes(b"cubin")

    manifest, _, native_paths, _ = _collect(
        workspace,
        adapter="cute.compiler",
        dump_dir=dump_dir,
        cute_keep_allowlist=("ptx",),
    )

    assert native_paths == (dump_dir / "kernel.ptx",)
    assert manifest.native_groups[0].path == "cute-dumps"


def test_collector_preserves_reproducer_as_an_attachment_not_a_stage(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    dump_dir.mkdir()
    (dump_dir / "kernel.ttir").write_text("ttir")
    reproducer = tmp_path / "triton-reproducer.mlir"
    reproducer.write_text("reproducer content")

    manifest, _, native_paths, _ = _collect(
        workspace,
        adapter="triton.compiler",
        dump_dir=dump_dir,
        reproducer_path=reproducer,
    )

    assert native_paths == (dump_dir / "kernel.ttir", reproducer)
    assert [artifact.path for artifact in manifest.attachments] == ["triton-reproducer.mlir"]


def test_collector_inventories_amdgcn_and_hsaco_extensions(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    dump_dir = tmp_path / "triton-dumps"
    dump_dir.mkdir()
    (dump_dir / "kernel.amdgcn").write_text("amdgcn source")
    (dump_dir / "kernel.hsaco").write_bytes(b"hsaco")

    manifest, _, native_paths, _ = _collect(
        workspace,
        adapter="triton.compiler",
        dump_dir=dump_dir,
    )

    assert {path.suffix for path in native_paths} == {".amdgcn", ".hsaco"}
    assert {artifact.path for artifact in manifest.native_groups[0].artifacts} == {
        "triton-dumps/kernel.amdgcn",
        "triton-dumps/kernel.hsaco",
    }
