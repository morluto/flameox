from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from flameox.application.evidence_lookup import EvidenceLookupService
from flameox.domain import (
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ProjectionIntent,
    ProjectionState,
    RunManifest,
    RunType,
    Sensitivity,
    ValidationStatus,
    digest_model,
)
from flameox.models import ContractModel
from flameox.storage import ProjectionIntentStore, RunStore, Workspace


class AgentCommandProjection(ContractModel):
    executable_name: str
    argument_count: Annotated[int, Field(ge=0)]
    arguments_digest: str | None = None
    environment_names: tuple[str, ...] = ()
    timeout_seconds: float


class AgentExternalContextProjection(ContractModel):
    orchestrator: str
    provider: str
    sensitivity: Sensitivity
    lease_id: str
    worker_id: str
    orchestration_run_id: str


class ProjectionVisibilityState(StrEnum):
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"
    INTENTIONALLY_ABSENT = "intentionally_absent"
    UNTRACKED = "untracked"


class AgentRunProjectionStatus(ContractModel):
    state: ProjectionVisibilityState
    current: bool
    authoritative_revision: Annotated[int, Field(ge=0)]
    projected_revision: Annotated[int, Field(ge=0)] | None = None
    intent_id: str | None = None
    failure_code: str | None = None


class AgentRunSemanticsProjection(ContractModel):
    semantic_id: str
    origin: Literal["capture", "import", "internal"]
    adapter: str | None = None
    adapter_version: str | None = None
    mode: JsonValue | None = None
    process_scope: JsonValue | None = None
    bounds: dict[str, JsonValue] = Field(default_factory=dict, max_length=32)
    filters: dict[str, JsonValue] = Field(default_factory=dict, max_length=32)
    unavailable_fields: tuple[str, ...] = ()


class AgentRunProjection(ContractModel):
    """Privacy-safe, snapshot-bound run projection for ordinary agent transports."""

    corpus_commit_id: str
    manifest_digest: str
    manifest_source: Literal["corpus_projection", "control_plane"]
    projection: AgentRunProjectionStatus
    run_id: str
    revision: Annotated[int, Field(ge=0)]
    run_type: RunType
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    execution_status: ExecutionStatus
    capture_status: CaptureStatus
    validation_status: ValidationStatus
    workload_definition_id: str | None = None
    workload_instance_id: str | None = None
    measurement_protocol_id: str | None = None
    source_measurement_run_id: str | None = None
    environment_id: str
    source_state_id: str | None = None
    semantics: AgentRunSemanticsProjection
    command: AgentCommandProjection | None = None
    external_context: AgentExternalContextProjection | None = None
    artifact_ids: Annotated[tuple[str, ...], Field(max_length=100)] = ()
    artifact_count: Annotated[int, Field(ge=0)] = 0
    artifact_list_truncated: bool = False
    limitation_count: Annotated[int, Field(ge=0)] = 0
    limitation_codes: Annotated[tuple[str, ...], Field(max_length=100)] = ()
    redactions: tuple[
        Literal[
            "command_arguments",
            "environment_values",
            "sensitive_external_identifiers",
            "host_paths",
            "limitation_text",
        ],
        ...,
    ]


def safe_command_projection(run: RunManifest) -> AgentCommandProjection | None:
    if run.command is None:
        return None
    argv = run.command.argv
    return AgentCommandProjection(
        executable_name=Path(argv[0]).name,
        argument_count=max(0, len(argv) - 1),
        arguments_digest=digest_model(argv[1:]) if len(argv) > 1 else None,
        environment_names=tuple(sorted(run.command.env_overrides)),
        timeout_seconds=run.command.timeout_seconds,
    )


def safe_external_context(run: RunManifest) -> AgentExternalContextProjection | None:
    context = run.external_context
    if context is None:
        return None
    redact = context.sensitivity is Sensitivity.SENSITIVE
    return AgentExternalContextProjection(
        orchestrator=context.orchestrator,
        provider=context.provider,
        sensitivity=context.sensitivity,
        lease_id="[redacted]" if redact else context.lease_id,
        worker_id="[redacted]" if redact else context.worker_id,
        orchestration_run_id="[redacted]" if redact else context.orchestration_run_id,
    )


def safe_run_semantics(run: RunManifest) -> AgentRunSemanticsProjection:
    semantics = run.semantics
    return AgentRunSemanticsProjection(
        semantic_id=semantics.semantic_id,
        origin=semantics.origin,
        adapter=semantics.adapter,
        adapter_version=semantics.adapter_version,
        mode=semantics.scope.mode.value if semantics.scope.mode else None,
        process_scope=(
            semantics.scope.process_scope.value if semantics.scope.process_scope else None
        ),
        bounds=semantics.scope.bounds,
        filters=semantics.scope.filters,
        unavailable_fields=semantics.unavailable_fields,
    )


class RunProjectionService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def get(self, run_id: str) -> AgentRunProjection:
        authoritative = RunStore(self.workspace).read(run_id)
        intent = ProjectionIntentStore(self.workspace).latest(
            domain_kind="run",
            domain_id=run_id,
            projection_kind="run.core",
        )
        try:
            with EvidenceLookupService(self.workspace).session() as evidence:
                projected = evidence.run(run_id)
                return self._build(
                    projected,
                    corpus_commit_id=evidence.commit_id,
                    manifest_source="corpus_projection",
                    projection=self._status(authoritative, projected, intent),
                )
        except DomainError as error:
            if error.code is not ErrorCode.RUN_NOT_FOUND:
                raise
        return self._build(
            authoritative,
            corpus_commit_id=self.workspace.corpus.read_head().commit_id,
            manifest_source="control_plane",
            projection=self._status(authoritative, None, intent),
        )

    def _build(
        self,
        run: RunManifest,
        *,
        corpus_commit_id: str,
        manifest_source: Literal["corpus_projection", "control_plane"],
        projection: AgentRunProjectionStatus,
    ) -> AgentRunProjection:
        artifact_ids = tuple(dict.fromkeys(item.artifact_id for item in run.artifacts))
        redactions: list[str] = [
            "command_arguments",
            "environment_values",
            "host_paths",
            "limitation_text",
        ]
        if (
            run.external_context is not None
            and run.external_context.sensitivity is Sensitivity.SENSITIVE
        ):
            redactions.append("sensitive_external_identifiers")
        return AgentRunProjection.model_validate(
            {
                "corpus_commit_id": corpus_commit_id,
                "manifest_digest": digest_model(run.model_dump(mode="json")),
                "manifest_source": manifest_source,
                "projection": projection,
                "run_id": run.run_id,
                "revision": run.revision,
                "run_type": run.run_type,
                "created_at": run.created_at.isoformat(),
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "execution_status": run.execution_status,
                "capture_status": run.capture_status,
                "validation_status": run.validation_status,
                "workload_definition_id": run.workload_definition_id,
                "workload_instance_id": run.workload_instance_id,
                "measurement_protocol_id": run.measurement_protocol_id,
                "source_measurement_run_id": run.source_measurement_run_id,
                "environment_id": run.environment_id,
                "source_state_id": run.source_state_id,
                "semantics": safe_run_semantics(run),
                "command": safe_command_projection(run),
                "external_context": safe_external_context(run),
                "artifact_ids": artifact_ids[:100],
                "artifact_count": len(artifact_ids),
                "artifact_list_truncated": len(artifact_ids) > 100,
                "limitation_count": len(run.limitations) + len(run.limitation_details),
                "limitation_codes": tuple(
                    dict.fromkeys(item.code for item in run.limitation_details)
                )[:100],
                "redactions": tuple(redactions),
            }
        )

    @staticmethod
    def _status(
        authoritative: RunManifest,
        projected: RunManifest | None,
        intent: ProjectionIntent | None,
    ) -> AgentRunProjectionStatus:
        projected_revision = projected.revision if projected is not None else None
        current = (
            projected is not None
            and projected.revision == authoritative.revision
            and digest_model(projected.model_dump(mode="json"))
            == digest_model(authoritative.model_dump(mode="json"))
        )
        if intent is None:
            state = (
                ProjectionVisibilityState.UNTRACKED
                if projected is not None
                else ProjectionVisibilityState.INTENTIONALLY_ABSENT
            )
        elif intent.state is ProjectionState.PENDING:
            state = ProjectionVisibilityState.PENDING
        elif intent.state is ProjectionState.FAILED:
            state = ProjectionVisibilityState.FAILED
        else:
            state = ProjectionVisibilityState.PUBLISHED
        return AgentRunProjectionStatus(
            state=state,
            current=current,
            authoritative_revision=authoritative.revision,
            projected_revision=projected_revision,
            intent_id=intent.intent_id if intent is not None else None,
            failure_code=intent.failure_code if intent is not None else None,
        )
