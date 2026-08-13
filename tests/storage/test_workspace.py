from __future__ import annotations

import errno
import json
import threading
from pathlib import Path

import pytest

from flameox.config import WorkspaceConfig
from flameox.domain import DomainError, ErrorCode
from flameox.storage import CorpusCommit, Workspace
from flameox.storage.locks import (
    CATALOG_SHARED,
    RETENTION_EXCLUSIVE,
    RETENTION_SHARED,
    WRITE_EXCLUSIVE,
)

pytestmark = [pytest.mark.integration, pytest.mark.serial]


def test_initialize_creates_static_identity_and_empty_corpus(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    first_identity = workspace.identity
    head = workspace.corpus.read_head()

    same_workspace = Workspace.initialize(tmp_path)

    assert same_workspace.identity == first_identity
    assert head.parent_commit_id is None
    assert head.generation_manifests == ()
    assert workspace.paths.config.is_file()
    assert WorkspaceConfig.from_path(workspace.paths.config) == WorkspaceConfig()


def test_removed_ad_hoc_command_setting_is_rejected(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    path = workspace.paths.config
    path.write_text(
        "# Preserve this project note.\n"
        "schema_version = 1\n\n"
        "[execution]\n"
        "# Preserve this execution note.\n"
        "allow_mcp_ad_hoc_commands = true\n",
        encoding="utf-8",
    )

    with pytest.raises(DomainError, match="Invalid workspace configuration"):
        _ = workspace.config


def test_initialize_maps_filesystem_quota_failures_to_domain_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "simulated quota exhaustion")

    monkeypatch.setattr("flameox.storage.workspace.atomic_write_json", fail_write)

    with pytest.raises(DomainError) as error:
        Workspace.initialize(tmp_path)

    assert error.value.code is ErrorCode.STORAGE_QUOTA_EXCEEDED
    assert error.value.details == {
        "operation": "workspace_initialization",
        "errno": errno.ENOSPC,
    }
    assert error.value.remediation == (
        "Free local storage or increase the filesystem quota, then retry initialization.",
    )


def test_initialize_maps_cross_root_relpath_failures_to_domain_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_relpath(*args: object, **kwargs: object) -> str:
        raise ValueError("path roots differ")

    monkeypatch.setattr("flameox.storage.workspace.os.path.relpath", fail_relpath)

    with pytest.raises(DomainError) as error:
        Workspace.initialize(tmp_path, workspace_root=tmp_path / "evidence")

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
    assert error.value.details == {"operation": "workspace_initialization"}


def test_discovery_selects_nearest_workspace(tmp_path: Path) -> None:
    outer = Workspace.initialize(tmp_path)
    nested_project = tmp_path / "packages" / "child"
    nested_project.mkdir(parents=True)
    inner = Workspace.initialize(nested_project)
    start = nested_project / "src"
    start.mkdir()

    assert Workspace.discover(start).identity.workspace_id == inner.identity.workspace_id
    assert Workspace.discover(tmp_path).identity.workspace_id == outer.identity.workspace_id


def test_explicit_workspace_rejects_a_different_project_root(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    workspace = Workspace.initialize(first, workspace_root=tmp_path / "evidence")

    with pytest.raises(DomainError) as error:
        Workspace.discover(
            second,
            explicit=workspace.paths.root,
            project_root=second,
        )

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
    assert error.value.details["bound_project_root"] == str(first.resolve())
    assert error.value.details["requested_project_root"] == str(second.resolve())


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    workspace.paths.config.write_text("schema_version = 1\nunknown = true\n")

    with pytest.raises(DomainError) as error:
        _ = workspace.config

    assert error.value.code is ErrorCode.WORKSPACE_INVALID


def test_corrupt_head_is_not_treated_as_empty_corpus(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    workspace.corpus.head_path.write_text("sha256:" + ("f" * 64))

    with pytest.raises(DomainError) as error:
        workspace.corpus.read_head()

    assert error.value.code is ErrorCode.WORKSPACE_INVALID


def test_commit_digest_detects_tampering(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    head = workspace.corpus.read_head()
    path = workspace.corpus.commit_path(head.commit_id)
    payload = json.loads(path.read_text())
    payload["generation_manifests"] = ["generations/fake/manifest.json"]
    path.write_text(json.dumps(payload))

    with pytest.raises(DomainError) as error:
        workspace.corpus.read_head()

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_corpus_commit_rejects_contradictory_persisted_digest_projections(
    tmp_path: Path,
) -> None:
    commit = Workspace.initialize(tmp_path).corpus.read_head()
    payload = commit.model_dump(mode="json")

    payload["inventory_digest"] = "sha256:" + ("a" * 64)
    with pytest.raises(ValueError, match="inventory digest"):
        CorpusCommit.model_validate(payload)

    payload = commit.model_dump(mode="json")
    payload["commit_id"] = "sha256:" + ("b" * 64)
    with pytest.raises(ValueError, match="commit digest"):
        CorpusCommit.model_validate(payload)


def test_reentrant_workspace_lock_is_rejected_before_blocking(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    with (
        workspace.write_locked(),
        pytest.raises(DomainError) as error,
        workspace.write_locked(timeout=0.01),
    ):
        raise AssertionError("reentrant lock was unexpectedly acquired")

    assert error.value.code is ErrorCode.LOCK_ORDER_VIOLATION


def test_ranked_workspace_locks_reject_inversion_and_upgrades(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    with (
        workspace.locked(WRITE_EXCLUSIVE),
        pytest.raises(DomainError) as inversion,
        workspace.locked(CATALOG_SHARED),
    ):
        pass
    assert inversion.value.code is ErrorCode.LOCK_ORDER_VIOLATION

    with (
        workspace.locked(RETENTION_SHARED),
        pytest.raises(DomainError) as upgrade,
        workspace.locked(RETENTION_EXCLUSIVE),
    ):
        pass
    assert upgrade.value.code is ErrorCode.LOCK_ORDER_VIOLATION

    with workspace.locked(CATALOG_SHARED, WRITE_EXCLUSIVE, RETENTION_SHARED):
        pass


def test_lock_timeout_names_resource_and_phase(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    held = threading.Event()
    release = threading.Event()

    def hold_write_lock() -> None:
        with workspace.write_locked():
            held.set()
            release.wait()

    holder = threading.Thread(target=hold_write_lock)
    holder.start()
    assert held.wait(1)
    try:
        with (
            pytest.raises(DomainError) as error,
            workspace.locked(
                WRITE_EXCLUSIVE,
                timeout=0.01,
                phase="test publication",
            ),
        ):
            pass
    finally:
        release.set()
        holder.join(timeout=1)

    assert error.value.code is ErrorCode.WRITE_LOCK_TIMEOUT
    assert error.value.retryable
    assert error.value.details["resource"] == "write"
    assert error.value.details["phase"] == "test publication"
