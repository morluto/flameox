from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application import (
    ArtifactService,
    EvidenceLookupService,
    ImportArtifactRequest,
    ImportService,
    NativeViewerService,
)
from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactKind,
    DomainError,
    ErrorCode,
    EvidenceReferenceType,
    Sensitivity,
)
from flameox.storage import ArtifactStore, Workspace

pytestmark = pytest.mark.integration


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


def test_artifact_metadata_effective_sensitivity_includes_registrations_outside_page(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "same.bin"
    source.write_bytes(b"same")
    imports = ImportService(workspace)
    sensitive = imports.import_artifact(
        ImportArtifactRequest(path=source, sensitivity=Sensitivity.SENSITIVE)
    )
    normal = imports.import_artifact(
        ImportArtifactRequest(path=source, sensitivity=Sensitivity.NORMAL)
    )

    result = ArtifactService(workspace).get(sensitive.artifact_id, limit=1)

    assert normal.artifact_id == sensitive.artifact_id
    assert result.total_registrations == 2
    assert len(result.registrations) == 1
    assert result.registrations[0].sensitivity is Sensitivity.NORMAL
    assert result.effective_sensitivity is Sensitivity.SENSITIVE


def test_evidence_lookup_rejects_cas_object_outside_the_snapshot(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "unregistered.bin"
    source.write_bytes(b"unregistered")
    stored = ArtifactStore(workspace).import_path(
        source,
        allowed_roots=(tmp_path,),
        max_bytes=1024,
    )

    with pytest.raises(DomainError) as error:
        EvidenceLookupService(workspace).get(
            EvidenceReferenceType.ARTIFACT,
            stored.content.artifact_id,
        )

    assert error.value.code is ErrorCode.WORKSPACE_INVALID


def test_native_viewer_plan_is_read_only_and_uses_content_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "profile.bin"
    source.write_bytes(b"profile")
    imported = ImportService(workspace).import_artifact(ImportArtifactRequest(path=source))
    monkeypatch.setattr(
        "flameox.command_binding.shutil.which",
        lambda _name, path=None: "/usr/bin/xdg-open",
    )

    plan = NativeViewerService(workspace).plan(imported.artifact_id)

    assert plan.argv[0] == "/usr/bin/xdg-open"
    assert Path(plan.argv[1]).is_file()
    assert not plan.launches


@pytest.mark.parametrize("sensitivity", [Sensitivity.NORMAL, Sensitivity.INTERNAL])
def test_import_refuses_underclassified_aiperf_result(
    tmp_path: Path, sensitivity: Sensitivity
) -> None:
    workspace = Workspace.initialize(tmp_path)
    export = tmp_path / "profile_export.jsonl"
    export.write_text('{"metadata":{"prompt":"secret"}}\n')

    with pytest.raises(DomainError) as error:
        ImportService(workspace).import_artifact(
            ImportArtifactRequest(
                path=export,
                kind=ArtifactKind.INFERENCE_RESULT,
                producer="aiperf",
                sensitivity=sensitivity,
            )
        )

    assert error.value.code is ErrorCode.SENSITIVE_ARTIFACT_REFUSED
