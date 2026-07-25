from __future__ import annotations

from pathlib import Path

import pytest

from flamo.application import (
    ArtifactService,
    ImportArtifactRequest,
    ImportService,
    NativeViewerService,
)
from flamo.catalog import Catalog
from flamo.domain import Sensitivity
from flamo.storage import Workspace


def test_artifact_metadata_keeps_registrations_and_max_sensitivity(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "same.bin"
    source.write_bytes(b"same")
    first = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, sensitivity=Sensitivity.NORMAL)
    )
    second = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, sensitivity=Sensitivity.SENSITIVE)
    )

    result = ArtifactService(workspace).get(first.artifact_id)
    listed = ArtifactService(workspace).list()

    assert first.artifact_id == second.artifact_id
    assert result.total_registrations == 2
    assert {item.run_id for item in result.registrations} == {
        first.run.run_id,
        second.run.run_id,
    }
    assert result.effective_sensitivity is Sensitivity.SENSITIVE
    assert listed.artifacts[0].registration_count == 2


def test_native_viewer_plan_is_read_only_and_uses_content_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "profile.bin"
    source.write_bytes(b"profile")
    imported = ImportService(workspace).import_artifact(ImportArtifactRequest(path=source))
    monkeypatch.setattr(
        "flamo.application.viewers.shutil.which",
        lambda _: "/usr/bin/xdg-open",
    )

    plan = NativeViewerService(workspace).plan(imported.artifact_id)

    assert plan.argv[0] == "/usr/bin/xdg-open"
    assert Path(plan.argv[1]).is_file()
    assert not plan.launches
