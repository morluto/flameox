from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flameox.application import (
    ImportArtifactRequest,
    ImportProfile,
    ImportService,
    ProjectionCoordinator,
    QualifyArtifactImportRequest,
)
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import ProjectionIntentStore, Workspace

pytestmark = pytest.mark.integration

_PYSPY_FIXTURE = Path(__file__).parents[1] / "fixtures" / "pyspy" / "chrometrace-0.4.2.json"


def _pyspy_trace(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "args": {"filename": "scan.py", "line": 12},
                    "cat": "py-spy",
                    "name": "scan",
                    "ph": phase,
                    "pid": 10,
                    "tid": 20,
                    "ts": timestamp,
                }
                for phase, timestamp in (("B", 30), ("E", 40))
            ]
        )
    )


def test_validated_pyspy_import_owns_provider_semantics(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    qualified = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=_PYSPY_FIXTURE,
            kind=ArtifactKind.SAMPLE_PROFILE,
            media_type="application/json",
            producer="py-spy",
            producer_version="0.4.2",
            profile=ImportProfile.PYSPY_CHROMETRACE,
            allow_external_path=True,
        )
    )
    generic = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=_PYSPY_FIXTURE,
            kind=ArtifactKind.SAMPLE_PROFILE,
            media_type="application/json",
            producer="py-spy",
            producer_version="0.4.2",
            allow_external_path=True,
        )
    )
    qualified_without_producer_claim = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=_PYSPY_FIXTURE,
            kind=ArtifactKind.SAMPLE_PROFILE,
            media_type="application/json",
            profile=ImportProfile.PYSPY_CHROMETRACE,
            allow_external_path=True,
        )
    )
    requalified = ImportService(workspace).qualify_artifact(
        QualifyArtifactImportRequest(
            run_id=generic.run.run_id,
            artifact_id=generic.artifact_id,
            profile=ImportProfile.PYSPY_CHROMETRACE,
        )
    )

    assert qualified.artifact_id == generic.artifact_id
    assert requalified.artifact_id == generic.artifact_id
    assert requalified.run.semantics.adapter == "py-spy"
    assert qualified.run.semantics.origin == "import"
    assert qualified.run.semantics.adapter == "py-spy"
    assert qualified.run.semantics.adapter_version is None
    assert qualified.run.semantics.configuration == {"import_profile": "py-spy-chrometrace"}
    assert qualified.run.semantics.unavailable_fields == ("adapter_version", "scope")
    assert qualified.run.artifacts[0].producer == "py-spy"
    assert qualified_without_producer_claim.run.semantics.adapter == "py-spy"
    assert qualified_without_producer_claim.run.artifacts[0].producer == "flameox.import"
    assert generic.run.semantics.adapter == "import"
    assert generic.run.semantics.unavailable_fields == ("configuration", "scope")


def test_invalid_pyspy_profile_preserves_failed_import_evidence(tmp_path: Path) -> None:
    trace = tmp_path / "generic-trace.json"
    trace.write_text(
        '[{"args":{"filename":"scan.py","line":12},"cat":"generic",'
        '"name":"scan","ph":"B","pid":1,"tid":2,"ts":3}]'
    )
    workspace = Workspace.initialize(tmp_path)
    service = ImportService(workspace)

    with pytest.raises(DomainError) as failure:
        service.import_artifact(
            ImportArtifactRequest(
                path=trace,
                kind=ArtifactKind.SAMPLE_PROFILE,
                profile=ImportProfile.PYSPY_CHROMETRACE,
            )
        )

    assert failure.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert failure.value.run_id is not None
    failed_run = service.runs.read(failure.value.run_id)
    assert failed_run.capture_status.value == "failed"
    assert len(failed_run.artifacts) == 1
    assert service.artifacts.get(failed_run.artifacts[0].artifact_id).payload_path.read_bytes()
    assert failed_run.semantics.adapter == "import"
    assert failed_run.semantics.configuration == {"attempted_import_profile": "py-spy-chrometrace"}


def test_pyspy_profile_reports_unverified_producer_version(tmp_path: Path) -> None:
    trace = tmp_path / "pyspy-trace.json"
    _pyspy_trace(trace)
    workspace = Workspace.initialize(tmp_path)

    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.SAMPLE_PROFILE,
            producer_version="future",
            profile=ImportProfile.PYSPY_CHROMETRACE,
        )
    )

    assert imported.run.semantics.adapter == "py-spy"
    assert imported.run.semantics.adapter_version is None
    assert imported.run.limitations == (
        "Imported py-spy future is caller-declared and not independently verified.",
    )


@pytest.mark.parametrize(
    "event",
    [
        {
            "args": {"filename": "scan.py", "line": 12},
            "cat": "py-spy",
            "name": "scan",
            "ph": "B",
            "pid": True,
            "tid": 2,
            "ts": 3,
        },
        {
            "args": {"filename": "scan.py", "line": 12},
            "cat": "py-spy",
            "name": "scan",
            "ph": "E",
            "pid": 1,
            "tid": 2,
            "ts": 3,
        },
    ],
)
def test_pyspy_profile_rejects_invalid_or_unbalanced_events(
    tmp_path: Path,
    event: dict[str, object],
) -> None:
    trace = tmp_path / "invalid-pyspy.json"
    trace.write_text(json.dumps([event]))
    workspace = Workspace.initialize(tmp_path)

    with pytest.raises(DomainError) as failure:
        ImportService(workspace).import_artifact(
            ImportArtifactRequest(
                path=trace,
                kind=ArtifactKind.SAMPLE_PROFILE,
                profile=ImportProfile.PYSPY_CHROMETRACE,
            )
        )

    assert failure.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_pyspy_profile_rejects_conflicting_producer_claim(tmp_path: Path) -> None:
    trace = tmp_path / "pyspy-trace.json"
    _pyspy_trace(trace)
    workspace = Workspace.initialize(tmp_path)

    with pytest.raises(DomainError) as failure:
        ImportService(workspace).import_artifact(
            ImportArtifactRequest(
                path=trace,
                kind=ArtifactKind.SAMPLE_PROFILE,
                producer="torch.profiler",
                profile=ImportProfile.PYSPY_CHROMETRACE,
            )
        )

    assert failure.value.code is ErrorCode.INVALID_ARGUMENTS


def test_profiled_import_records_failure_before_artifact_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = tmp_path / "pyspy-trace.json"
    _pyspy_trace(trace)
    workspace = Workspace.initialize(tmp_path)
    service = ImportService(workspace)

    def fail_publication(*_args: object, **_kwargs: object) -> object:
        raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_FAILED, "injected publication failure")

    monkeypatch.setattr(service.artifacts, "import_snapshot", fail_publication)
    with pytest.raises(DomainError) as failure:
        service.import_artifact(
            ImportArtifactRequest(
                path=trace,
                kind=ArtifactKind.SAMPLE_PROFILE,
                profile=ImportProfile.PYSPY_CHROMETRACE,
            )
        )

    assert failure.value.run_id is not None
    failed_run = service.runs.read(failure.value.run_id)
    assert failed_run.capture_status.value == "failed"
    assert failed_run.artifacts == ()
    assert failed_run.semantics.adapter == "py-spy"


def test_profiled_import_durably_binds_published_artifact_if_projection_fails(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "pyspy-trace.json"
    _pyspy_trace(trace)
    workspace = Workspace.initialize(tmp_path)
    service = ImportService(workspace)

    def fail_projection(phase: object, _intent: object) -> None:
        if phase == "after_domain_commit":
            raise RuntimeError("injected projection failure")

    service.projections = ProjectionCoordinator(workspace, fault_injector=fail_projection)
    with pytest.raises(RuntimeError, match="injected projection failure"):
        service.import_artifact(
            ImportArtifactRequest(
                path=trace,
                kind=ArtifactKind.SAMPLE_PROFILE,
                profile=ImportProfile.PYSPY_CHROMETRACE,
            )
        )

    runs = service.runs.list()
    assert len(runs) == 1
    assert runs[0].capture_status.value == "registered"
    assert runs[0].semantics.adapter == "py-spy"
    artifact_id = f"sha256:{hashlib.sha256(trace.read_bytes()).hexdigest()}"
    assert runs[0].artifacts[0].artifact_id == artifact_id
    assert service.artifacts.get(artifact_id).content.artifact_id == artifact_id
    intents = ProjectionIntentStore(workspace).list()
    assert len(intents) == 1
    assert intents[0].domain_revision == 0
    assert intents[0].state.value == "pending"


def test_profile_cannot_bypass_sensitive_artifact_admission(tmp_path: Path) -> None:
    trace = tmp_path / "dump.json"
    _pyspy_trace(trace)
    workspace = Workspace.initialize(tmp_path)

    with pytest.raises(DomainError) as failure:
        ImportService(workspace).import_artifact(
            ImportArtifactRequest(
                path=trace,
                kind=ArtifactKind.CORE_DUMP,
                profile=ImportProfile.PYSPY_CHROMETRACE,
            )
        )

    assert failure.value.code is ErrorCode.SENSITIVE_ARTIFACT_REFUSED
    assert ImportService(workspace).runs.list() == ()


def test_pyspy_profile_bounds_frame_text(tmp_path: Path) -> None:
    trace = tmp_path / "oversized-frame.json"
    trace.write_text(
        json.dumps(
            [
                {
                    "args": {"filename": "scan.py", "line": 12},
                    "cat": "py-spy",
                    "name": "x" * 4_097,
                    "ph": "B",
                    "pid": 1,
                    "tid": 2,
                    "ts": 3,
                }
            ]
        )
    )
    workspace = Workspace.initialize(tmp_path)

    with pytest.raises(DomainError) as failure:
        ImportService(workspace).import_artifact(
            ImportArtifactRequest(
                path=trace,
                kind=ArtifactKind.SAMPLE_PROFILE,
                profile=ImportProfile.PYSPY_CHROMETRACE,
            )
        )

    assert failure.value.code is ErrorCode.QUERY_BUDGET_EXCEEDED
    assert failure.value.run_id is not None
    assert ImportService(workspace).runs.read(failure.value.run_id).artifacts
