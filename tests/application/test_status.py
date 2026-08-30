from __future__ import annotations

from pathlib import Path

import pytest

from flameox import __version__
from flameox.action_graph import ActionId
from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.application.quarantine import QuarantineService
from flameox.application.status import workspace_status
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, new_id
from flameox.observability import OperationLogger
from flameox.storage import Workspace
from flameox.storage.corpus import build_commit

pytestmark = pytest.mark.integration


def test_status_reports_integrity_storage_recovery_and_capabilities(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    artifact = tmp_path / "profile.bin"
    artifact.write_bytes(b"evidence")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=artifact,
            kind=ArtifactKind.COLLECTOR_METADATA,
        )
    )
    staging = workspace.paths.staging / "invalid.tmp"
    staging.write_bytes(b"bad")
    QuarantineService(workspace).quarantine(
        staging,
        reason="fixture",
        operation="test",
        originating_run_id=imported.run.run_id,
    )

    result = workspace_status(workspace)

    assert result.server_version == __version__

    assert result.workspace_valid
    assert result.catalog_valid
    assert result.last_catalog_rebuild_at is not None
    assert result.storage_by_artifact_kind == {"collector_metadata": 8}
    assert result.quarantined_run_ids == (imported.run.run_id,)
    assert result.extractor_versions["perfetto"] == "1"
    assert result.capability_warnings


def test_operation_log_failure_does_not_change_operation_outcome(tmp_path: Path) -> None:
    logger = OperationLogger(tmp_path)
    logger.path.mkdir(parents=True)

    emitted = logger.emit(
        operation_id=new_id(),
        operation="test",
        phase="complete",
    )

    assert emitted is False


def test_status_reports_corrupt_rebuildable_catalog_without_crashing(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    workspace.paths.catalog.write_bytes(b"not duckdb")

    result = workspace_status(workspace)

    assert result.workspace_valid
    assert result.catalog_valid is False
    assert any("catalog" in warning.lower() for warning in result.warnings)
    assert result.next_action is not None
    assert result.next_action.action is ActionId.REBUILD_CATALOG


def test_status_does_not_treat_a_new_corpus_head_as_stale_catalog_state(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    head = workspace.corpus.read_head()
    newer = build_commit(
        parent_commit_id=head.commit_id,
        generation_ids=head.generation_ids,
    )
    workspace.corpus.write_commit(newer)
    workspace.corpus.publish_head(newer.commit_id)

    result = workspace_status(workspace)

    assert result.catalog_valid is True
    assert result.corpus_commit_id == newer.commit_id
    assert result.next_action is None
