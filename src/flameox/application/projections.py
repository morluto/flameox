from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from flameox.application.evidence_rows import (
    artifact_registration_row,
    environment_row,
    source_state_row,
)
from flameox.application.run_rows import run_row
from flameox.domain import (
    DomainError,
    EnvironmentRecord,
    ErrorCode,
    ProjectionIntent,
    ProjectionIntentSpec,
    ProjectionState,
    RunManifest,
    RunProjectionContext,
    SourceState,
    digest_model,
    projection_intent_id,
)
from flameox.evidence import (
    GenerationPublisher,
    PublishedGeneration,
)
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, ControlPlane, RunStore, Workspace

RUN_PROJECTION_PUBLISHER = "flameox.run_projection"
RUN_PROJECTION_PUBLISHER_VERSION = "1"

type ProjectionPhase = Literal[
    "after_domain_commit",
    "after_corpus_publish",
    "after_intent_commit",
]
type ProjectionFaultInjector = Callable[[ProjectionPhase, ProjectionIntent], None]


@dataclass(frozen=True, slots=True)
class PublishedRunProjection:
    run: RunManifest
    publication: PublishedGeneration


class ProjectionReconciliationResult(ContractModel):
    inspected: int
    published: int
    failed: int
    intent_ids: tuple[str, ...]


class ProjectionCoordinator:
    """Publish run revisions through a recoverable, idempotent outbox."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        fault_injector: ProjectionFaultInjector | None = None,
    ) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)
        self.artifacts = ArtifactStore(workspace)
        self.control = ControlPlane(workspace)
        self.publisher = GenerationPublisher(workspace)
        self.fault_injector = fault_injector

    def create_run(
        self,
        run: RunManifest,
        *,
        environment: EnvironmentRecord | None,
        source_state: SourceState | None,
    ) -> PublishedRunProjection:
        spec = self.run_projection_spec(
            run,
            environment=environment,
            source_state=source_state,
        )
        stored = self.runs.create(run, projection_intent=spec)
        intent = self.control.read_projection_intent(spec.intent_id)
        self._inject("after_domain_commit", intent)
        return PublishedRunProjection(
            run=stored,
            publication=self._publish_run_intent(intent),
        )

    def append_run(
        self,
        run: RunManifest,
        *,
        expected_revision: int,
        environment: EnvironmentRecord | None,
        source_state: SourceState | None,
    ) -> PublishedRunProjection:
        spec = self.run_projection_spec(
            run,
            environment=environment,
            source_state=source_state,
        )
        stored = self.runs.append(
            run,
            expected_revision=expected_revision,
            projection_intent=spec,
        )
        intent = self.control.read_projection_intent(spec.intent_id)
        self._inject("after_domain_commit", intent)
        return PublishedRunProjection(
            run=stored,
            publication=self._publish_run_intent(intent),
        )

    def run_projection_spec(
        self,
        run: RunManifest,
        *,
        environment: EnvironmentRecord | None,
        source_state: SourceState | None,
    ) -> ProjectionIntentSpec:
        if environment is not None and environment.environment_id != run.environment_id:
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "Run projection environment does not match the run revision.",
            )
        if source_state is not None and source_state.source_state_id != run.source_state_id:
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "Run projection source state does not match the run revision.",
            )
        context = RunProjectionContext(
            environment=environment,
            source_state=source_state,
        )
        intent_id = projection_intent_id(
            workspace_id=self.workspace.identity.workspace_id,
            run_id=run.run_id,
            run_revision=run.revision,
        )
        return ProjectionIntentSpec(
            intent_id=intent_id,
            run_id=run.run_id,
            run_revision=run.revision,
            run_digest=digest_model(run.model_dump(mode="json")),
            context=context,
        )

    def reconcile(self) -> ProjectionReconciliationResult:
        candidates = list(self.control.list_projection_intents(state=ProjectionState.PENDING))
        candidates.extend(self.control.list_projection_intents(state=ProjectionState.FAILED))
        published = 0
        failed = 0
        intent_ids: list[str] = []
        for candidate in candidates:
            intent_ids.append(candidate.intent_id)
            try:
                result = self.reconcile_intent(candidate.intent_id)
            except Exception:
                failed += 1
                continue
            if result.state is ProjectionState.PUBLISHED:
                published += 1
            else:
                failed += 1
        return ProjectionReconciliationResult(
            inspected=len(candidates),
            published=published,
            failed=failed,
            intent_ids=tuple(intent_ids),
        )

    def reconcile_intent(self, intent_id: str) -> ProjectionIntent:
        intent = self.control.read_projection_intent(intent_id)
        if intent.state is ProjectionState.PUBLISHED:
            return intent
        if intent.state is ProjectionState.FAILED:
            intent = self.control.retry_projection(intent.intent_id)
        latest = self.control.latest_projection_intent(run_id=intent.run_id)
        if (
            latest is not None
            and latest.run_revision > intent.run_revision
            and latest.state is ProjectionState.PUBLISHED
        ):
            return self.control.mark_projection_failed(
                intent_id=intent.intent_id,
                failure_code="superseded_revision",
                failure_message="A newer domain revision already owns the current projection.",
            )
        self._publish_run_intent(intent)
        return self.control.read_projection_intent(intent.intent_id)

    def _publish_run_intent(self, intent: ProjectionIntent) -> PublishedGeneration:
        run, rows = self._run_projection_rows(intent)
        if run.run_id != intent.run_id:
            raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_FAILED, "Run projection id changed.")
        try:
            published = self.publisher.publish_rows_idempotent(
                rows,
                publisher=RUN_PROJECTION_PUBLISHER,
                publisher_version=RUN_PROJECTION_PUBLISHER_VERSION,
                # Rows carry exact artifact identity. Keeping publication scope
                # run-only lets each revision supersede the previous one.
                input_run_ids=(intent.run_id,),
                input_artifact_ids=(),
                operation_identity={"projection_intent_id": intent.intent_id},
            )
        except Exception as error:
            code = (
                error.code.value.lower() if isinstance(error, DomainError) else "publication_failed"
            )
            message = (
                error.message
                if isinstance(error, DomainError)
                else f"Projection publication failed: {type(error).__name__}."
            )
            try:
                self.control.mark_projection_failed(
                    intent_id=intent.intent_id,
                    failure_code=code,
                    failure_message=message[:500],
                )
            except Exception as status_error:
                error.add_note(
                    "Projection failure status could not be recorded: "
                    f"{type(status_error).__name__}."
                )
            raise
        self._inject("after_corpus_publish", intent)
        completed = self.control.mark_projection_published(
            intent_id=intent.intent_id,
            generation_id=published.manifest.generation_id,
            corpus_commit_id=published.commit.commit_id,
        )
        self._inject("after_intent_commit", completed)
        return published

    def _run_projection_rows(
        self,
        intent: ProjectionIntent,
    ) -> tuple[RunManifest, Mapping[str, Sequence[Mapping[str, Any]]]]:
        run = self.runs.read_revision(intent.run_id, intent.run_revision)
        if digest_model(run.model_dump(mode="json")) != intent.run_digest:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Projection intent no longer matches its immutable run revision.",
            )
        context = intent.context
        if (
            context.environment is not None
            and context.environment.environment_id != run.environment_id
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Projection environment does not match the run revision.",
            )
        if (
            context.source_state is not None
            and context.source_state.source_state_id != run.source_state_id
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Projection source state does not match the run revision.",
            )
        rows: dict[str, Sequence[Mapping[str, Any]]] = {
            "runs": [run_row(run)],
            "artifact_registrations": [
                artifact_registration_row(
                    registration,
                    byte_length=self.artifacts.get(registration.artifact_id).content.byte_length,
                )
                for registration in run.artifacts
            ],
        }
        if context.environment is not None:
            rows["environments"] = [environment_row(context.environment)]
        if context.source_state is not None:
            rows["source_states"] = [source_state_row(context.source_state)]
        return run, rows

    def _inject(self, phase: ProjectionPhase, intent: ProjectionIntent) -> None:
        if self.fault_injector is not None:
            self.fault_injector(phase, intent)
