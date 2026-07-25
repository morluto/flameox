from __future__ import annotations

import json
from pathlib import Path

import pytest

from flamo.config import WorkspaceConfig
from flamo.domain import DomainError, ErrorCode
from flamo.storage import Workspace


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
