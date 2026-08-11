from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.application import (
    ImportArtifactRequest,
    ImportService,
    IntegrityIssue,
    IntegrityLevel,
    IntegrityResult,
    IntegrityService,
    IntegritySeverity,
)
from flameox.catalog import Catalog
from flameox.storage import ArtifactStore, Workspace


def test_full_integrity_detects_altered_artifact_bytes(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "profile.bin"
    source.write_bytes(b"immutable")
    imported = ImportService(workspace).import_artifact(ImportArtifactRequest(path=source))
    stored = ArtifactStore(workspace).get(imported.artifact_id)
    stored.payload_path.write_bytes(b"tampered!")

    quick = IntegrityService(workspace).validate(IntegrityLevel.QUICK)
    full = IntegrityService(workspace).validate(IntegrityLevel.FULL)

    # With verify-on-retrieval (M1), the quick path now also catches tampered
    # artifact bytes because ArtifactStore.get() re-hashes the payload on
    # every retrieval. Both levels must report the artifact as invalid.
    assert quick.valid is False
    assert full.valid is False
    assert any(issue.code == "INVALID_ARTIFACT" for issue in quick.issues)
    assert any(issue.code == "INVALID_ARTIFACT" for issue in full.issues)


def test_integrity_validity_is_derived_from_issue_severity() -> None:
    result = IntegrityResult(
        level=IntegrityLevel.QUICK,
        corpus_commit_id="sha256:commit",
        checked_artifacts=0,
        checked_generations=0,
        checked_parquet_files=0,
        issues=(
            IntegrityIssue(
                severity=IntegritySeverity.WARNING,
                code="STALE",
                message="Catalog is stale.",
            ),
        ),
    )

    assert result.valid is True
    assert IntegrityResult.model_validate(result.model_dump()).valid is True

    contradictory = {**result.model_dump(), "valid": False}
    with pytest.raises(ValidationError, match="validity must agree"):
        IntegrityResult.model_validate(contradictory)
