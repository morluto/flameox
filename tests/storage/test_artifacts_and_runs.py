from __future__ import annotations

import os
from pathlib import Path

import pytest

from flameox.domain import (
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    RunManifest,
    RunType,
    ValidationStatus,
)
from flameox.storage import ArtifactStore, RunStore, Workspace

DIGEST = "sha256:" + ("a" * 64)


def import_manifest(run_id: str, *, revision: int = 0) -> RunManifest:
    return RunManifest(
        revision=revision,
        run_id=run_id,
        run_type=RunType.IMPORT,
        execution_status=ExecutionStatus.NOT_APPLICABLE,
        capture_status=CaptureStatus.PENDING,
        validation_status=ValidationStatus.NOT_REQUESTED,
        environment_id=DIGEST,
    )


def test_identical_artifact_bytes_share_one_content_object(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    first = tmp_path / "first.bin"
    second = tmp_path / "second.dat"
    first.write_bytes(b"same bytes")
    second.write_bytes(b"same bytes")
    store = ArtifactStore(workspace)

    left = store.import_path(first, allowed_roots=(tmp_path,), max_bytes=100)
    right = store.import_path(second, allowed_roots=(tmp_path,), max_bytes=100)

    assert left.content.artifact_id == right.content.artifact_id
    assert left.payload_path == right.payload_path
    assert len(list(workspace.paths.artifacts.glob("*/*/artifact.json"))) == 1


def test_artifact_import_rejects_symlinks_and_hard_links(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    original = tmp_path / "source.bin"
    original.write_bytes(b"evidence")
    symlink = tmp_path / "symlink.bin"
    symlink.symlink_to(original)
    hardlink = tmp_path / "hardlink.bin"
    os.link(original, hardlink)
    store = ArtifactStore(workspace)

    with pytest.raises(DomainError) as symlink_error:
        store.import_path(symlink, allowed_roots=(tmp_path,), max_bytes=100)
    with pytest.raises(DomainError) as hardlink_error:
        store.import_path(hardlink, allowed_roots=(tmp_path,), max_bytes=100)

    assert symlink_error.value.code is ErrorCode.EXECUTION_REFUSED
    assert hardlink_error.value.code is ErrorCode.EXECUTION_REFUSED


def test_artifact_import_enforces_size_and_root(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path / "project")
    source = tmp_path / "source.bin"
    source.write_bytes(b"1234")
    store = ArtifactStore(workspace)

    with pytest.raises(DomainError) as root_error:
        store.import_path(
            source,
            allowed_roots=(workspace.project_root,),
            max_bytes=100,
        )
    with pytest.raises(DomainError) as size_error:
        store.import_path(source, allowed_roots=(tmp_path,), max_bytes=3)

    assert root_error.value.code is ErrorCode.EXECUTION_REFUSED
    assert size_error.value.code is ErrorCode.ARTIFACT_TOO_LARGE
    assert list(workspace.paths.staging.iterdir()) == []


def test_run_revisions_use_compare_and_swap(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    store = RunStore(workspace)
    original = store.create(import_manifest("run"))
    completed = original.model_copy(
        update={
            "revision": 1,
            "capture_status": CaptureStatus.REGISTERED,
        }
    )

    store.append(completed, expected_revision=0)

    with pytest.raises(DomainError) as error:
        store.append(
            original.model_copy(update={"revision": 1}),
            expected_revision=0,
        )

    assert error.value.code is ErrorCode.REVISION_CONFLICT
    assert store.read("run") == completed
    revisions = list((workspace.paths.runs / "run" / "revisions").glob("*.json"))
    assert len(revisions) == 2
