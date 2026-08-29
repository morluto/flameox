from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, StringConstraints

from flameox.adapters.kernel_validation import (
    KernelValidationStatus,
    load_kernel_validation_document,
)
from flameox.application.evidence_lookup import EvidenceLookupService
from flameox.application.projections import ProjectionCoordinator
from flameox.domain import DomainError, ErrorCode, new_id
from flameox.domain.models import (
    ArtifactKind,
    ArtifactRegistration,
    ExecutionRunManifest,
    ExecutionStatus,
    Sensitivity,
    ValidationStatus,
)
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace


class RegisterKernelValidationRequest(ContractModel):
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    expected_run_revision: Annotated[int, Field(ge=0)]
    path: Path
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    allow_external_path: bool = False


class RegisterKernelValidationResult(ContractModel):
    run_id: str
    run_revision: int
    registration_id: str
    artifact_id: str
    validation_status: ValidationStatus
    workload_definition_id: str
    workload_instance_id: str
    environment_id: str
    source_state_id: str | None
    execution_identity_id: str
    corpus_commit_id: str


class KernelValidationRegistrationService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.artifacts = ArtifactStore(workspace)
        self.runs = RunStore(workspace)
        self.projections = ProjectionCoordinator(workspace)

    def register(
        self,
        request: RegisterKernelValidationRequest,
    ) -> RegisterKernelValidationResult:
        run = self.runs.read(request.run_id)
        if run.revision != request.expected_run_revision:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "The source run changed after it was reviewed; inspect it and retry.",
                details={
                    "run_id": run.run_id,
                    "expected_revision": request.expected_run_revision,
                    "actual_revision": run.revision,
                },
            )
        if not isinstance(run, ExecutionRunManifest) or (
            run.execution_status is not ExecutionStatus.SUCCEEDED
        ):
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "Kernel validation can be linked only to a succeeded execution run.",
                details={"run_id": run.run_id, "run_type": run.run_type.value},
            )
        if (
            run.workload_definition_id is None
            or run.workload_instance_id is None
            or run.execution_identity is None
        ):
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "The source run lacks the workload or execution identity required for validation.",
                details={"run_id": run.run_id},
            )
        workload_definition_id = run.workload_definition_id
        workload_instance_id = run.workload_instance_id
        execution_identity_id = run.execution_identity.identity_id
        with EvidenceLookupService(self.workspace).session() as evidence:
            environment = evidence.environment(run.environment_id)
            source_state = (
                evidence.source_state(run.source_state_id)
                if run.source_state_id is not None
                else None
            )
        if any(item.role == "kernel_validation" for item in run.artifacts):
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "The source run already has kernel-validation evidence.",
                details={"run_id": run.run_id},
            )

        allowed_roots = [self.workspace.project_root]
        if request.allow_external_path:
            allowed_roots.append(request.path.absolute().parent)
        with self.artifacts.temporary_snapshot(
            request.path,
            allowed_roots=tuple(allowed_roots),
            max_bytes=self.workspace.config.capture.max_artifact_bytes,
        ) as snapshot:
            document, source_schema = load_kernel_validation_document(snapshot.payload_path)
            if source_schema != "flameox.kernel-validation.v2":
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    "Run-linked kernel validation requires flameox.kernel-validation.v2.",
                )
            status = {
                KernelValidationStatus.PASS: ValidationStatus.PASSED,
                KernelValidationStatus.FAIL: ValidationStatus.FAILED,
                KernelValidationStatus.INCONCLUSIVE: ValidationStatus.INCONCLUSIVE,
                KernelValidationStatus.UNSUPPORTED: ValidationStatus.UNSUPPORTED,
            }[document.status]
            if run.validation_status not in {ValidationStatus.NOT_REQUESTED, status}:
                raise DomainError(
                    ErrorCode.INVALID_ARGUMENTS,
                    "Kernel-validation status conflicts with existing run validation semantics.",
                    details={
                        "run_id": run.run_id,
                        "existing_status": run.validation_status.value,
                        "kernel_validation_status": status.value,
                    },
                )
            stored = self.artifacts.import_snapshot(snapshot, display_name=request.path.name)
        registration = ArtifactRegistration(
            registration_id=new_id(),
            run_id=run.run_id,
            artifact_id=stored.content.artifact_id,
            display_name=request.path.name,
            media_type="application/json",
            kind=ArtifactKind.VALIDATION_OUTPUT,
            role="kernel_validation",
            producer=document.producer,
            producer_version=document.producer_version,
            sensitivity=request.sensitivity,
        )
        updated = run.validated_copy(
            update={
                "revision": run.revision + 1,
                "validation_status": status,
                "artifacts": (*run.artifacts, registration),
            }
        )
        projected = self.projections.append_run(
            updated,
            expected_revision=run.revision,
            environment=environment,
            source_state=source_state,
        )
        return RegisterKernelValidationResult(
            run_id=updated.run_id,
            run_revision=updated.revision,
            registration_id=registration.registration_id,
            artifact_id=registration.artifact_id,
            validation_status=status,
            workload_definition_id=workload_definition_id,
            workload_instance_id=workload_instance_id,
            environment_id=updated.environment_id,
            source_state_id=updated.source_state_id,
            execution_identity_id=execution_identity_id,
            corpus_commit_id=projected.publication.commit.commit_id,
        )
