from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from flameox.application import (
    ImportArtifactRequest,
    ImportService,
    QuarantineService,
    RepairService,
)
from flameox.domain import DomainError, ErrorCode
from flameox.storage import RunStore, Workspace


def test_repair_quarantines_invalid_projection_and_rebuilds_from_revision(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "profile.bin"
    source.write_bytes(b"profile")
    imported = ImportService(workspace).import_artifact(ImportArtifactRequest(path=source))
    projection = workspace.paths.runs / imported.run.run_id / "manifest.json"
    projection.write_text("{partial")

    service = RepairService(workspace)
    plan = service.plan()
    result = service.apply(plan)

    assert result.repaired_paths == (f"runs/{imported.run.run_id}/manifest.json",)
    assert RunStore(workspace).read(imported.run.run_id) == imported.run
    quarantine = result.quarantine[0]
    assert quarantine.operation == f"repair:{plan.plan_id}"
    assert quarantine.originating_run_id == imported.run.run_id

    projection.unlink()
    restored = QuarantineService(workspace).restore(quarantine.quarantine_id)
    assert restored.restored_path == f"runs/{imported.run.run_id}/manifest.json"
    assert projection.read_text() == "{partial"


def test_repair_revalidates_projection_after_acquiring_write_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "profile.bin"
    source.write_bytes(b"profile")
    imported = ImportService(workspace).import_artifact(ImportArtifactRequest(path=source))
    projection = workspace.paths.runs / imported.run.run_id / "manifest.json"
    projection.write_text("{partial")
    service = RepairService(workspace)
    plan = service.plan()
    original_write_lock = workspace.write_locked

    @contextmanager
    def concurrent_projection_update() -> Iterator[None]:
        with original_write_lock():
            projection.write_text(imported.run.model_dump_json())
            yield

    monkeypatch.setattr(workspace, "write_locked", concurrent_projection_update)

    with pytest.raises(DomainError) as error:
        service.apply(plan)

    assert error.value.code is ErrorCode.REVISION_CONFLICT
    assert RunStore(workspace).read(imported.run.run_id) == imported.run
    assert service.quarantine.list_manifests() == ()


def test_quarantine_resume_completes_crash_after_manifest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = workspace.paths.staging / "partial.bin"
    source.write_bytes(b"partial")
    real_replace = os.replace

    def fail_move(move_source: str | Path, destination: str | Path) -> None:
        if Path(move_source) == source:
            raise OSError("simulated crash")
        real_replace(move_source, destination)

    monkeypatch.setattr(
        "flameox.application.recoverable_move.os.replace",
        fail_move,
    )
    with pytest.raises(OSError, match="simulated crash"):
        QuarantineService(workspace).quarantine(
            source,
            reason="fixture",
            operation="test",
        )
    (manifest,) = QuarantineService(workspace).list_manifests()

    monkeypatch.setattr("flameox.application.recoverable_move.os.replace", real_replace)
    resumed = QuarantineService(workspace).resume(manifest.quarantine_id)

    assert resumed.state == "quarantined"
    assert not source.exists()


def test_quarantine_rejects_manifest_paths_outside_recovery_storage(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = workspace.paths.staging / "partial.bin"
    source.write_bytes(b"partial")
    service = QuarantineService(workspace)
    quarantined = service.quarantine(
        source,
        reason="fixture",
        operation="test",
    )
    manifest_path = workspace.paths.quarantine / quarantined.quarantine_id / "manifest.json"
    manifest_path.write_text(
        quarantined.model_copy(update={"original_path": "../../outside"}).model_dump_json()
    )

    with pytest.raises(DomainError) as error:
        service.restore(quarantined.quarantine_id)

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_quarantine_rejects_symbolic_link_sources(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    target = workspace.paths.staging / "target.bin"
    target.write_bytes(b"authoritative")
    source = workspace.paths.staging / "link.bin"
    try:
        source.symlink_to(target.name)
    except OSError:
        pytest.skip("The platform does not permit symbolic links.")

    with pytest.raises(DomainError) as error:
        QuarantineService(workspace).quarantine(
            source,
            reason="fixture",
            operation="test",
        )

    assert error.value.code is ErrorCode.EXECUTION_REFUSED
    assert target.read_bytes() == b"authoritative"
    assert source.is_symlink()


def test_quarantine_digest_distinguishes_empty_file_and_directory(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = workspace.paths.staging / "empty"
    source.touch()
    service = QuarantineService(workspace)
    manifest = service.quarantine(
        source,
        reason="fixture",
        operation="test",
    )
    stored = workspace.paths.quarantine / manifest.quarantine_id / manifest.stored_path
    stored.unlink()
    stored.mkdir()

    with pytest.raises(DomainError) as error:
        service.restore(manifest.quarantine_id)

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_quarantine_resume_revalidates_content_after_interrupted_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = workspace.paths.staging / "partial.bin"
    source.write_bytes(b"original")
    real_replace = os.replace

    def fail_move(move_source: str | Path, destination: str | Path) -> None:
        if Path(move_source) == source:
            raise OSError("simulated crash")
        real_replace(move_source, destination)

    monkeypatch.setattr(
        "flameox.application.recoverable_move.os.replace",
        fail_move,
    )
    with pytest.raises(OSError, match="simulated crash"):
        QuarantineService(workspace).quarantine(
            source,
            reason="fixture",
            operation="test",
        )
    (manifest,) = QuarantineService(workspace).list_manifests()
    source.write_bytes(b"changed")
    monkeypatch.setattr("flameox.application.recoverable_move.os.replace", real_replace)

    with pytest.raises(DomainError) as error:
        QuarantineService(workspace).resume(manifest.quarantine_id)

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert QuarantineService(workspace).list_manifests()[0].state == "moving"


def test_quarantine_resume_completes_interrupted_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = workspace.paths.staging / "partial.bin"
    source.write_bytes(b"partial")
    service = QuarantineService(workspace)
    quarantined = service.quarantine(
        source,
        reason="fixture",
        operation="test",
    )
    real_replace = os.replace

    def fail_after_move(move_source: str | Path, destination: str | Path) -> None:
        real_replace(move_source, destination)
        if Path(destination) == source:
            raise OSError("injected restore crash")

    monkeypatch.setattr(
        "flameox.application.recoverable_move.os.replace",
        fail_after_move,
    )
    with pytest.raises(OSError, match="injected restore crash"):
        service.restore(quarantined.quarantine_id)

    monkeypatch.setattr("flameox.application.recoverable_move.os.replace", real_replace)
    assert service.moving_manifests() == (quarantined.quarantine_id,)
    manifest = service.resume(quarantined.quarantine_id)

    assert manifest.state == "restored"
    assert source.read_bytes() == b"partial"
