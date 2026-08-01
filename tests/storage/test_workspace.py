from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from flameox.config import WorkspaceConfig
from flameox.domain import DomainError, ErrorCode
from flameox.storage import Workspace


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


def test_removed_ad_hoc_command_setting_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "schema_version = 1\n\n[execution]\nallow_mcp_ad_hoc_commands = true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allow_mcp_ad_hoc_commands"):
        WorkspaceConfig.from_path(path)
    assert "allow_mcp_ad_hoc_commands" in path.read_text()


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


def test_discovery_selects_nearest_workspace(tmp_path: Path) -> None:
    outer = Workspace.initialize(tmp_path)
    nested_project = tmp_path / "packages" / "child"
    nested_project.mkdir(parents=True)
    inner = Workspace.initialize(nested_project)
    start = nested_project / "src"
    start.mkdir()

    assert Workspace.discover(start).identity.workspace_id == inner.identity.workspace_id
    assert Workspace.discover(tmp_path).identity.workspace_id == outer.identity.workspace_id


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


def test_lock_timeout_maps_to_structured_retryable_error(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    with (
        workspace.write_locked(),
        pytest.raises(DomainError) as error,
        workspace.write_locked(timeout=0.01),
    ):
        raise AssertionError("contended lock was unexpectedly acquired")

    assert error.value.code is ErrorCode.WRITE_LOCK_TIMEOUT
    assert error.value.retryable
