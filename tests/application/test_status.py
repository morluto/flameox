from __future__ import annotations

from pathlib import Path

from flamo.application import (
    ImportArtifactRequest,
    ImportService,
    QuarantineService,
    workspace_status,
)
from flamo.domain import ArtifactKind
from flamo.observability import OperationLogger
from flamo.storage import Workspace


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
        operation_id=logger.new_id(),
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
