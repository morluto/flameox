from __future__ import annotations

from pathlib import Path

from flamo.application import ImportArtifactRequest, ImportService, IntegrityService
from flamo.catalog import Catalog
from flamo.storage import ArtifactStore, Workspace


def test_full_integrity_detects_altered_artifact_bytes(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "profile.bin"
    source.write_bytes(b"immutable")
    imported = ImportService(workspace).import_artifact(ImportArtifactRequest(path=source))
    stored = ArtifactStore(workspace).get(imported.artifact_id)
    stored.payload_path.write_bytes(b"tampered!")

    quick = IntegrityService(workspace).validate(full=False)
    full = IntegrityService(workspace).validate(full=True)

    assert quick.valid is True
    assert full.valid is False
    assert any(issue.code == "INVALID_ARTIFACT" for issue in full.issues)
