from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import TypeAdapter

import flameox.filesystem as filesystem_module
from flameox.domain import (
    CaptureLease,
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    RunManifest,
    RunSemantics,
    ValidationStatus,
)
from flameox.domain.models import ExecutionRunManifest, ImportRunManifest
from flameox.filesystem import BoundedFileSystem
from flameox.storage import ArtifactStore, RunStore, Workspace

pytestmark = [pytest.mark.integration, pytest.mark.serial]

DIGEST = "sha256:" + ("a" * 64)


def import_manifest(run_id: str, *, revision: int = 0) -> RunManifest:
    return ImportRunManifest(
        revision=revision,
        run_id=run_id,
        execution_status=ExecutionStatus.NOT_APPLICABLE,
        capture_status=CaptureStatus.PENDING,
        validation_status=ValidationStatus.NOT_REQUESTED,
        environment_id=DIGEST,
        semantics=RunSemantics.unavailable(origin="import", adapter="import"),
    )


def test_missing_run_has_distinct_recovery_guidance(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    with pytest.raises(DomainError) as error:
        RunStore(workspace).read("missing-run")

    assert error.value.code is ErrorCode.RUN_NOT_FOUND
    assert error.value.details == {"missing_entity": "run"}
    assert error.value.remediation == ("Call list_runs to choose an existing run.",)


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


def test_deduplicating_import_reauthenticates_existing_cas_payload(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    first_source = tmp_path / "first.bin"
    second_source = tmp_path / "second.bin"
    first_source.write_bytes(b"trusted")
    second_source.write_bytes(b"trusted")
    store = ArtifactStore(workspace)
    first = store.import_path(first_source, allowed_roots=(tmp_path,), max_bytes=100)
    first.payload_path.write_bytes(b"corrupt")

    with pytest.raises(DomainError) as caught:
        store.import_path(second_source, allowed_roots=(tmp_path,), max_bytes=100)

    assert caught.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_artifact_import_rejects_bytes_that_do_not_match_the_expected_receipt(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "handoff.bin"
    source.write_bytes(b"replacement")
    expected = "sha256:" + hashlib.sha256(b"validated").hexdigest()

    with pytest.raises(DomainError) as error:
        ArtifactStore(workspace).import_path(
            source,
            allowed_roots=(tmp_path,),
            max_bytes=100,
            expected_artifact_id=expected,
            expected_byte_length=len(b"validated"),
        )

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert not list(workspace.paths.artifacts.glob("*/*/artifact.json"))


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


def test_artifact_import_pins_parent_directories_before_source_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source_root = tmp_path / "safe"
    source_root.mkdir()
    source = source_root / "source.bin"
    source.write_bytes(b"approved bytes")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / source.name).write_bytes(b"outside bytes")
    store = ArtifactStore(workspace)
    real_open = os.open
    swapped = False

    def raced_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        path_value = Path(os.fsdecode(path))
        if not swapped and (
            path_value == source or (dir_fd is not None and path_value == Path(source.name))
        ):
            source_root.rename(tmp_path / "safe-original")
            source_root.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", raced_open)
    stored = store.import_path(source, allowed_roots=(tmp_path,), max_bytes=100)

    assert swapped
    assert stored.payload_path.read_bytes() == b"approved bytes"


def test_windows_artifact_import_fallback_opens_regular_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "source.bin"
    source.write_bytes(b"windows-compatible bytes")
    store = ArtifactStore(workspace)
    monkeypatch.setattr(
        BoundedFileSystem,
        "_open_beneath",
        staticmethod(filesystem_module._open_windows_beneath),
    )
    monkeypatch.setattr(
        filesystem_module,
        "_windows_final_path",
        lambda descriptor: source,
    )

    stored = store.import_path(source, allowed_roots=(tmp_path,), max_bytes=100)

    assert stored.payload_path.read_bytes() == b"windows-compatible bytes"


def test_windows_artifact_import_fallback_opens_nested_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    nested = tmp_path / "nested" / "directory"
    nested.mkdir(parents=True)
    source = nested / "source.bin"
    source.write_bytes(b"nested windows-compatible bytes")
    store = ArtifactStore(workspace)
    monkeypatch.setattr(
        BoundedFileSystem,
        "_open_beneath",
        staticmethod(filesystem_module._open_windows_beneath),
    )
    monkeypatch.setattr(
        filesystem_module,
        "_windows_final_path",
        lambda descriptor: source,
    )

    stored = store.import_path(source, allowed_roots=(tmp_path,), max_bytes=100)

    assert stored.payload_path.read_bytes() == b"nested windows-compatible bytes"


def test_windows_artifact_import_fallback_rejects_reparse_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    source = tmp_path / "source.bin"
    source.symlink_to(target)
    store = ArtifactStore(workspace)
    monkeypatch.setattr(
        BoundedFileSystem,
        "_open_beneath",
        staticmethod(filesystem_module._open_windows_beneath),
    )

    with pytest.raises(DomainError) as error:
        store.import_path(source, allowed_roots=(tmp_path,), max_bytes=100)

    assert error.value.code is ErrorCode.EXECUTION_REFUSED


def test_windows_artifact_import_fallback_rejects_hard_link_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    original = tmp_path / "original.bin"
    original.write_bytes(b"mutable source")
    source = tmp_path / "source.bin"
    os.link(original, source)
    store = ArtifactStore(workspace)
    monkeypatch.setattr(
        BoundedFileSystem,
        "_open_beneath",
        staticmethod(filesystem_module._open_windows_beneath),
    )
    monkeypatch.setattr(
        filesystem_module,
        "_windows_final_path",
        lambda descriptor: source,
    )

    with pytest.raises(DomainError) as error:
        store.import_path(source, allowed_roots=(tmp_path,), max_bytes=100)

    assert error.value.code is ErrorCode.EXECUTION_REFUSED


def test_windows_artifact_import_fallback_checks_open_handle_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"outside")
    store = ArtifactStore(workspace)
    monkeypatch.setattr(
        BoundedFileSystem,
        "_open_beneath",
        staticmethod(filesystem_module._open_windows_beneath),
    )
    monkeypatch.setattr(
        filesystem_module,
        "_windows_final_path",
        lambda descriptor: outside,
    )

    with pytest.raises(DomainError) as error:
        store.import_path(source, allowed_roots=(tmp_path,), max_bytes=100)

    assert error.value.code is ErrorCode.EXECUTION_REFUSED
    assert not list(workspace.paths.staging.iterdir())


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


@pytest.mark.parametrize(
    "payload_name",
    ["", "../outside", r"..\outside", "D:outside", ".", ".."],
)
def test_artifact_get_rejects_non_local_payload_name(
    tmp_path: Path,
    payload_name: str,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "source.bin"
    source.write_bytes(b"evidence")
    store = ArtifactStore(workspace)
    stored = store.import_path(source, allowed_roots=(tmp_path,), max_bytes=100)
    metadata_path = stored.payload_path.parent / "artifact.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["payload_name"] = payload_name
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(DomainError) as error:
        store.get(stored.content.artifact_id)

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_artifact_get_rejects_symlink_payload(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "source.bin"
    source.write_bytes(b"evidence")
    store = ArtifactStore(workspace)
    stored = store.import_path(source, allowed_roots=(tmp_path,), max_bytes=100)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    stored.payload_path.unlink()
    stored.payload_path.symlink_to(outside)

    with pytest.raises(DomainError) as error:
        store.get(stored.content.artifact_id)

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


@pytest.mark.parametrize("component", ["object", "metadata"])
def test_artifact_get_rejects_symlink_storage_component(
    tmp_path: Path,
    component: str,
) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    source = tmp_path / "source.bin"
    source.write_bytes(b"evidence")
    store = ArtifactStore(workspace)
    stored = store.import_path(source, allowed_roots=(tmp_path,), max_bytes=100)
    object_root = stored.payload_path.parent
    metadata_path = object_root / "artifact.json"
    outside = tmp_path / "outside"
    outside.mkdir()

    if component == "metadata":
        outside_metadata = outside / "artifact.json"
        outside_metadata.write_bytes(metadata_path.read_bytes())
        metadata_path.unlink()
        metadata_path.symlink_to(outside_metadata)
    else:
        outside_object = outside / "object"
        object_root.rename(outside_object)
        object_root.symlink_to(outside_object, target_is_directory=True)

    with pytest.raises(DomainError) as error:
        store.get(stored.content.artifact_id)

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


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
    with workspace.paths.control_plane.open("rb") as stream:
        assert stream.read(16) == b"SQLite format 3\x00"
    assert [run.run_id for run in store.list()] == ["run"]


def test_run_store_reparses_unchecked_updates_before_persistence(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    started = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    run = ExecutionRunManifest(
        run_id="invalid-transition",
        started_at=started,
        execution_status=ExecutionStatus.RUNNING,
        capture_status=CaptureStatus.RUNNING,
        validation_status=ValidationStatus.PENDING,
        environment_id=DIGEST,
        semantics=RunSemantics.unavailable(origin="internal", adapter=None),
    )
    invalid = run.model_copy(update={"finished_at": started - timedelta(seconds=1)})

    with pytest.raises(DomainError) as error:
        RunStore(workspace).create(invalid)

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
    assert RunStore(workspace).list() == ()


def test_run_semantics_cannot_change_across_revisions(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    store = RunStore(workspace)
    current = store.create(import_manifest("stable-semantics"))
    changed = current.validated_copy(
        update={
            "revision": 1,
            "semantics": RunSemantics.unavailable(origin="import", adapter="other"),
        }
    )

    with pytest.raises(DomainError, match="semantics are immutable"):
        store.append(changed, expected_revision=0)

    assert store.read("stable-semantics") == current


def test_record_store_reparses_unchecked_updates_before_persistence(tmp_path: Path) -> None:
    from flameox.storage.records import ControlRecordStore

    workspace = Workspace.initialize(tmp_path)
    observed = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    lease = CaptureLease(
        process_id=123,
        process_start_identity="456",
        boot_id="boot-id",
        heartbeat_monotonic_ns=0,
        observed_at=observed,
        expires_at=observed + timedelta(seconds=1),
    )
    invalid = lease.model_copy(update={"expires_at": observed})
    store = ControlRecordStore(
        workspace,
        kind="leases",
        model=CaptureLease,
        id_field="boot_id",
    )

    with pytest.raises(DomainError) as error:
        store.create(invalid)

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
    assert not (workspace.paths.records / "leases" / lease.boot_id).exists()


def test_artifact_get_rejects_race_symlink_swap(tmp_path: Path) -> None:
    """Regression: O_NOFOLLOW must block a symlink swapped in after the
    is_symlink() check but before open()."""
    workspace = Workspace.initialize(tmp_path / "workspace")
    source = tmp_path / "source.bin"
    source.write_bytes(b"evidence")
    store = ArtifactStore(workspace)
    stored = store.import_path(source, allowed_roots=(tmp_path,), max_bytes=100)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    # Replace the payload with a symlink *after* it was imported.
    stored.payload_path.unlink()
    stored.payload_path.symlink_to(outside)

    with pytest.raises(DomainError) as error:
        store.get(stored.content.artifact_id)

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_record_store_rejects_dotdot_identifier(tmp_path: Path) -> None:
    """Regression: ControlRecordStore must reject '..' identifiers to prevent
    directory traversal outside the records directory."""
    from flameox.domain.models import RunManifest
    from flameox.storage.records import ControlRecordStore

    workspace = Workspace.initialize(tmp_path)
    store: ControlRecordStore[RunManifest] = ControlRecordStore(
        workspace,
        kind="test",
        model=TypeAdapter(RunManifest),
        id_field="run_id",
    )

    with pytest.raises(DomainError) as error:
        store.read("..")

    assert error.value.code is ErrorCode.WORKSPACE_INVALID


def test_record_store_rejects_dot_prefix_identifier(tmp_path: Path) -> None:
    """Regression: ControlRecordStore must reject dot-prefixed identifiers
    to prevent hidden-file traversal and .dotfile access."""
    from flameox.domain.models import RunManifest
    from flameox.storage.records import ControlRecordStore

    workspace = Workspace.initialize(tmp_path)
    store: ControlRecordStore[RunManifest] = ControlRecordStore(
        workspace,
        kind="test",
        model=TypeAdapter(RunManifest),
        id_field="run_id",
    )

    with pytest.raises(DomainError) as error:
        store.read(".hidden")

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
