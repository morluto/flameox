from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from flameox.action_graph import ActionId, ToolAction
from flameox.application import (
    ImportArtifactRequest,
    ImportService,
    NativeViewerPlan,
    NativeViewerService,
)
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode, Sensitivity
from flameox.storage import Workspace

pytestmark = pytest.mark.integration


def test_benchmark_artifact_uses_pyperf_viewer(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    artifact_path = tmp_path / "benchmark.json"
    artifact_path.write_text("{}")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=artifact_path,
            kind=ArtifactKind.BENCHMARK_SAMPLES,
        )
    )

    plan = NativeViewerService(workspace).plan(imported.run.artifacts[0].artifact_id)

    assert plan.viewer == "pyperf show"
    assert plan.argv[1:4] == ("-m", "pyperf", "show")
    assert plan.provider_environment_id is None


def test_memray_viewer_uses_verified_provider_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    artifact_path = tmp_path / "memory.bin"
    artifact_path.write_bytes(b"memray")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=artifact_path,
            kind=ArtifactKind.MEMORY_PROFILE,
            producer="memray",
            producer_version="1.20.0",
        )
    )
    executable = tmp_path / "provider-runtimes" / "runtime" / "bin" / "memray"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    service = NativeViewerService(workspace)
    monkeypatch.setattr(
        service.provider_runtimes,
        "find_distribution",
        lambda **_kwargs: SimpleNamespace(
            root=executable.parents[2],
            executable=executable,
            receipt=SimpleNamespace(environment_id="sha256:" + "a" * 64),
        ),
    )

    plan = service.plan(imported.artifact_id)

    assert plan.argv[0] == str(executable)
    assert plan.provider_environment_id == "sha256:" + "a" * 64
    assert not any("host environment" in limitation for limitation in plan.limitations)


def test_memray_viewer_rejects_unverified_runtime_with_setup_recovery(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    artifact_path = tmp_path / "memory.bin"
    artifact_path.write_bytes(b"memray")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=artifact_path,
            kind=ArtifactKind.MEMORY_PROFILE,
            producer="memray",
            producer_version="1.20.0",
        )
    )
    invalid_runtime = workspace.paths.records / "provider-runtimes" / ("a" * 64)
    invalid_runtime.mkdir(parents=True)
    (invalid_runtime / "provider-runtime.json").write_text("{}")

    with pytest.raises(DomainError) as captured:
        NativeViewerService(workspace).plan(imported.artifact_id)

    error = captured.value
    assert error.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert isinstance(error.next_action, ToolAction)
    assert error.next_action.action is ActionId.START_CAPABILITY_SETUP


@pytest.mark.parametrize("versions", [("not-a-version",), ("1.19.0", "1.20.0")])
def test_memray_viewer_rejects_ambiguous_producer_versions(
    tmp_path: Path,
    versions: tuple[str, ...],
) -> None:
    workspace = Workspace.initialize(tmp_path)
    artifact_path = tmp_path / "memory.bin"
    artifact_path.write_bytes(b"memray")
    artifact_id = ""
    for version in versions:
        imported = ImportService(workspace).import_artifact(
            ImportArtifactRequest(
                path=artifact_path,
                kind=ArtifactKind.MEMORY_PROFILE,
                producer="memray",
                producer_version=version,
            )
        )
        artifact_id = imported.artifact_id

    with pytest.raises(DomainError) as captured:
        NativeViewerService(workspace).plan(artifact_id)

    assert captured.value.code is ErrorCode.ADAPTER_INCOMPATIBLE


@pytest.mark.parametrize(
    ("kind", "expected_viewer"),
    (
        (ArtifactKind.BENCHMARK_SAMPLES, "pyperf show"),
        (ArtifactKind.EXECUTION_TRACE, "trace_processor_shell"),
        (ArtifactKind.SAMPLE_PROFILE, "trace_processor_shell"),
        (ArtifactKind.CORE_DUMP, "gdb"),
        (ArtifactKind.COLLECTOR_METADATA, "xdg-open"),
    ),
)
def test_every_supported_native_kind_dispatches_to_ecosystem_viewer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: ArtifactKind,
    expected_viewer: str,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    executable = tmp_path / "viewer"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setattr(
        "flameox.command_binding.shutil.which",
        lambda _name, path=None: str(executable),
    )
    artifact_path = tmp_path / f"{kind.value}.bin"
    artifact_path.write_bytes(b"native")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=artifact_path,
            kind=kind,
            sensitivity=(
                Sensitivity.SENSITIVE if kind is ArtifactKind.CORE_DUMP else Sensitivity.INTERNAL
            ),
        )
    )

    plan = NativeViewerService(workspace).plan(imported.artifact_id)

    assert plan.viewer == expected_viewer
    expected_executable = (
        sys.executable if kind is ArtifactKind.BENCHMARK_SAMPLES else str(executable)
    )
    assert plan.argv[0] == expected_executable


@pytest.mark.anyio
async def test_explicit_viewer_launch_uses_bounded_subprocess_broker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    executable = tmp_path / "viewer"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setattr(
        "flameox.command_binding.shutil.which",
        lambda _name, path=None: str(executable),
    )
    artifact_path = tmp_path / "metadata.bin"
    artifact_path.write_bytes(b"native")
    imported = ImportService(workspace).import_artifact(ImportArtifactRequest(path=artifact_path))

    result = await NativeViewerService(workspace).launch(imported.artifact_id)

    assert result.plan.launches is True
    assert result.validated_copy().plan.launches is True
    assert result.process.exit_code == 0


def test_unlaunched_viewer_plan_cannot_claim_launch_side_effect() -> None:
    with pytest.raises(ValidationError):
        NativeViewerPlan.model_validate(
            {
                "artifact_id": "sha256:artifact",
                "artifact_path": "/artifact",
                "artifact_kinds": [ArtifactKind.COLLECTOR_METADATA],
                "viewer": "xdg-open",
                "argv": ["xdg-open", "/artifact"],
                "launches": True,
            }
        )
