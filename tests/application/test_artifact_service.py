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
from flameox.storage import ArtifactStore, StoredArtifact, Workspace

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


def test_process_output_preview_is_bounded_and_continuable(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "stdout.txt"
    source.write_text("first\nsecond\nthird\n")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=source,
            kind=ArtifactKind.PROCESS_OUTPUT,
            sensitivity=Sensitivity.INTERNAL,
        )
    )

    first = ArtifactService(workspace).preview_text(
        imported.artifact_id,
        offset=0,
        max_bytes=64,
        max_lines=1,
    )
    second = ArtifactService(workspace).preview_text(
        imported.artifact_id,
        offset=first.next_offset or 0,
        max_bytes=64,
        max_lines=2,
    )

    assert first.text == "first\n"
    assert first.returned_bytes == 6
    assert first.total_bytes == 19
    assert first.truncated is True
    assert first.next_offset == 6
    assert second.text == "second\nthird\n"
    assert second.truncated is False
    assert second.next_offset is None


@pytest.mark.parametrize(
    ("payload", "kind", "sensitivity", "code"),
    (
        (
            b"binary",
            ArtifactKind.COLLECTOR_METADATA,
            Sensitivity.INTERNAL,
            ErrorCode.ARTIFACT_PARSE_FAILED,
        ),
        (
            b"secret",
            ArtifactKind.PROCESS_OUTPUT,
            Sensitivity.SENSITIVE,
            ErrorCode.SENSITIVE_ARTIFACT_REFUSED,
        ),
        (
            b"\xff",
            ArtifactKind.PROCESS_OUTPUT,
            Sensitivity.INTERNAL,
            ErrorCode.ARTIFACT_PARSE_FAILED,
        ),
    ),
)
def test_artifact_preview_returns_typed_refusals(
    tmp_path: Path,
    payload: bytes,
    kind: ArtifactKind,
    sensitivity: Sensitivity,
    code: ErrorCode,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "payload.bin"
    source.write_bytes(payload)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, kind=kind, sensitivity=sensitivity)
    )

    with pytest.raises(DomainError) as error:
        ArtifactService(workspace).preview_text(
            imported.artifact_id,
            offset=0,
            max_bytes=64,
            max_lines=10,
        )

    assert error.value.code is code


def test_empty_process_output_preview_and_invalid_offset(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "empty.txt"
    source.write_bytes(b"")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, kind=ArtifactKind.PROCESS_OUTPUT)
    )
    service = ArtifactService(workspace)

    result = service.preview_text(imported.artifact_id, offset=0, max_bytes=1, max_lines=1)

    assert result.text == ""
    assert result.returned_bytes == 0
    assert result.truncated is False
    with pytest.raises(DomainError) as error:
        service.preview_text(imported.artifact_id, offset=1, max_bytes=1, max_lines=1)
    assert error.value.code is ErrorCode.INVALID_ARGUMENTS


def test_preview_applies_maximum_sensitivity_across_shared_registrations(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "shared.txt"
    source.write_text("shared output")
    imports = ImportService(workspace)
    process_output = imports.import_artifact(
        ImportArtifactRequest(
            path=source,
            kind=ArtifactKind.PROCESS_OUTPUT,
            sensitivity=Sensitivity.NORMAL,
        )
    )
    metadata = imports.import_artifact(
        ImportArtifactRequest(
            path=source,
            kind=ArtifactKind.COLLECTOR_METADATA,
            sensitivity=Sensitivity.SENSITIVE,
        )
    )

    assert process_output.artifact_id == metadata.artifact_id
    with pytest.raises(DomainError) as error:
        ArtifactService(workspace).preview_text(
            process_output.artifact_id,
            offset=0,
            max_bytes=64,
            max_lines=10,
        )
    assert error.value.code is ErrorCode.SENSITIVE_ARTIFACT_REFUSED


def test_preview_rejects_payload_replaced_after_metadata_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    source = tmp_path / "stdout.txt"
    source.write_text("captured")
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(path=source, kind=ArtifactKind.PROCESS_OUTPUT)
    )
    secret = tmp_path / "secret.txt"
    secret.write_text("must not escape")
    original_get = ArtifactStore.get

    def replace_after_verification(store: ArtifactStore, artifact_id: str) -> StoredArtifact:
        stored = original_get(store, artifact_id)
        stored.payload_path.unlink()
        stored.payload_path.symlink_to(secret)
        return stored

    monkeypatch.setattr(ArtifactStore, "get", replace_after_verification)

    with pytest.raises(DomainError) as error:
        ArtifactService(workspace).preview_text(
            imported.artifact_id,
            offset=0,
            max_bytes=64,
            max_lines=10,
        )
    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


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
