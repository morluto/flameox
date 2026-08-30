from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.application.imports import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.application.integrity import (
    IntegrityIssue,
    IntegrityLevel,
    IntegrityResult,
    IntegrityService,
    IntegritySeverity,
)
from flameox.catalog import Catalog
from flameox.storage import ArtifactStore, Workspace
from flameox.storage.corpus import build_commit

pytestmark = pytest.mark.integration


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
    assert result.validated_copy().valid is True

    contradictory = {**result.model_dump(), "valid": False}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        IntegrityResult.model_validate(contradictory)


def test_integrity_uses_pinned_corpus_without_stale_catalog_bookkeeping(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    head = workspace.corpus.read_head()
    newer = build_commit(
        parent_commit_id=head.commit_id,
        generation_ids=head.generation_ids,
    )
    workspace.corpus.write_commit(newer)
    workspace.corpus.publish_head(newer.commit_id)

    result = IntegrityService(workspace).validate(IntegrityLevel.QUICK)
    assert result.corpus_commit_id == newer.commit_id
    assert all(issue.code != "STALE_CATALOG" for issue in result.issues)
