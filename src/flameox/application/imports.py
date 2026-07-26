from __future__ import annotations

import mimetypes
from pathlib import Path

from flameox.application.environment import collect_environment
from flameox.application.evidence_rows import (
    artifact_registration_row,
    environment_row,
    source_state_row,
)
from flameox.application.run_rows import run_row
from flameox.application.source import collect_partial_source_state
from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.identity import new_id
from flameox.domain.models import (
    ArtifactKind,
    ArtifactRegistration,
    CaptureStatus,
    ExecutionStatus,
    RunManifest,
    RunType,
    Sensitivity,
    ValidationStatus,
)
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace


class ImportArtifactRequest(ContractModel):
    path: Path
    kind: ArtifactKind = ArtifactKind.COLLECTOR_METADATA
    media_type: str | None = None
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    role: str = "primary"
    producer: str = "flameox.import"
    producer_version: str | None = None
    allow_external_path: bool = False


class ImportResult(ContractModel):
    run: RunManifest
    artifact_id: str
    corpus_commit_id: str


class ImportService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.artifacts = ArtifactStore(workspace)
        self.runs = RunStore(workspace)
        self.publisher = GenerationPublisher(workspace)

    def import_artifact(self, request: ImportArtifactRequest) -> ImportResult:
        if (
            request.kind in {ArtifactKind.CORE_DUMP, ArtifactKind.SOURCE_SNAPSHOT}
            and request.sensitivity is not Sensitivity.SENSITIVE
        ):
            raise DomainError(
                code=ErrorCode.SENSITIVE_ARTIFACT_REFUSED,
                message=f"{request.kind.value} artifacts must be marked sensitive.",
            )
        environment = collect_environment()
        source_state = collect_partial_source_state(self.workspace)
        run_id = new_id()
        initial = RunManifest(
            run_id=run_id,
            run_type=RunType.IMPORT,
            execution_status=ExecutionStatus.NOT_APPLICABLE,
            capture_status=CaptureStatus.PENDING,
            validation_status=ValidationStatus.NOT_REQUESTED,
            environment_id=environment.environment_id,
            source_state_id=source_state.source_state_id,
            collector="import",
        )
        self.runs.create(initial)
        allowed_roots = [self.workspace.project_root]
        if request.allow_external_path:
            allowed_roots.append(request.path.absolute().parent)
        try:
            stored = self.artifacts.import_path(
                request.path,
                allowed_roots=tuple(allowed_roots),
                max_bytes=self.workspace.config.capture.max_artifact_bytes,
            )
        except DomainError as error:
            failed = initial.model_copy(
                update={
                    "revision": 1,
                    "capture_status": CaptureStatus.FAILED,
                    "limitations": (error.message,),
                }
            )
            self.runs.append(failed, expected_revision=0)
            error.run_id = run_id
            raise

        media_type = request.media_type or mimetypes.guess_type(request.path.name)[0]
        registration = ArtifactRegistration(
            registration_id=new_id(),
            run_id=run_id,
            artifact_id=stored.content.artifact_id,
            display_name=request.path.name,
            media_type=media_type or "application/octet-stream",
            kind=request.kind,
            role=request.role,
            producer=request.producer,
            producer_version=request.producer_version,
            sensitivity=request.sensitivity,
        )
        registered = initial.model_copy(
            update={
                "revision": 1,
                "capture_status": CaptureStatus.REGISTERED,
                "artifacts": (registration,),
            }
        )
        self.runs.append(registered, expected_revision=0)
        published = self.publisher.publish_rows(
            {
                "runs": [run_row(registered)],
                "artifact_registrations": [
                    artifact_registration_row(
                        registration,
                        byte_length=stored.content.byte_length,
                    )
                ],
                "environments": [environment_row(environment)],
                "source_states": [source_state_row(source_state)],
            },
            publisher="flameox.import",
            publisher_version="1",
            input_run_ids=(run_id,),
            input_artifact_ids=(stored.content.artifact_id,),
        )
        return ImportResult(
            run=registered,
            artifact_id=stored.content.artifact_id,
            corpus_commit_id=published.commit.commit_id,
        )
