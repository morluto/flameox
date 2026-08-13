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
    SourceState,
    digest_model,
    projection_intent_id,
)
from flameox.evidence import (
    GenerationPublisher,
    PublishedGeneration,
    publication_operation_digest,
)
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, ProjectionIntentStore, RunStore, Workspace

RUN_PROJECTION_KIND = "run.core"
RUN_PROJECTION_SCHEMA_VERSION = 1
RUN_PROJECTION_PUBLISHER = "flameox.run_projection"
RUN_PROJECTION_PUBLISHER_VERSION = "1"
_RUN_PROJECTION_BASE_TABLES = ("runs", "artifact_registrations")

type ProjectionPhase = Literal[
    "after_domain_commit",
    "after_corpus_publish",
    "after_intent_commit",
]
type ProjectionFaultInjector = Callable[[ProjectionPhase, ProjectionIntent], None]


class RunProjectionReplayContext(ContractModel):
    environment: EnvironmentRecord | None = None
    source_state: SourceState | None = None


@dataclass(frozen=True, slots=True)
class PublishedRunProjection:
    run: RunManifest
    intent: ProjectionIntent
    publication: PublishedGeneration


class ProjectionReconciliationResult(ContractModel):
    inspected: int
    published: int
    failed: int
    already_terminal: int
    deferred: int = 0
    intent_ids: tuple[str, ...]


class ProjectionCoordinator:
    """Couple domain revisions to replayable, idempotent evidence projections."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        fault_injector: ProjectionFaultInjector | None = None,
    ) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)
        self.artifacts = ArtifactStore(workspace)
        self.intents = ProjectionIntentStore(workspace)
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
        intent = self.intents.read(spec.intent_id)
        self._inject("after_domain_commit", intent)
        return PublishedRunProjection(
            run=stored,
            intent=intent,
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
        intent = self.intents.read(spec.intent_id)
        self._inject("after_domain_commit", intent)
        return PublishedRunProjection(
            run=stored,
            intent=intent,
            publication=self._publish_run_intent(intent),
        )

    def project_existing_run(
        self,
        run: RunManifest,
        *,
        environment: EnvironmentRecord | None,
        source_state: SourceState | None,
    ) -> PublishedRunProjection:
        """Attach a projection intent to an already durable legacy/recovery revision."""

        durable = self.runs.read_revision(run.run_id, run.revision)
        if durable != run:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "The requested run projection does not match its durable revision.",
            )
        spec = self.run_projection_spec(
            durable,
            environment=environment,
            source_state=source_state,
        )
        intent = self.intents.create(spec)
        self._inject("after_domain_commit", intent)
        return PublishedRunProjection(
            run=durable,
            intent=intent,
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
        context = RunProjectionReplayContext(
            environment=environment,
            source_state=source_state,
        )
        tables = [*_RUN_PROJECTION_BASE_TABLES]
        if environment is not None:
            tables.append("environments")
        if source_state is not None:
            tables.append("source_states")
        intent_id = projection_intent_id(
            workspace_id=self.workspace.identity.workspace_id,
            domain_kind="run",
            domain_id=run.run_id,
            domain_revision=run.revision,
            projection_kind=RUN_PROJECTION_KIND,
            projection_schema_version=RUN_PROJECTION_SCHEMA_VERSION,
        )
        operation_identity = {"projection_intent_id": intent_id}
        return ProjectionIntentSpec(
            intent_id=intent_id,
            workspace_id=self.workspace.identity.workspace_id,
            domain_kind="run",
            domain_id=run.run_id,
            domain_revision=run.revision,
            domain_digest=digest_model(run.model_dump(mode="json")),
            projection_kind=RUN_PROJECTION_KIND,
            projection_schema_version=RUN_PROJECTION_SCHEMA_VERSION,
            publisher=RUN_PROJECTION_PUBLISHER,
            publisher_version=RUN_PROJECTION_PUBLISHER_VERSION,
            input_run_ids=(run.run_id,),
            # The run row and artifact-registration rows carry exact artifact
            # identity. Keeping the manifest scope run-only lets each revision
            # supersede its predecessor even when the artifact set changes.
            input_artifact_ids=(),
            expected_tables=tuple(tables),
            operation_digest=publication_operation_digest(
                publisher=RUN_PROJECTION_PUBLISHER,
                publisher_version=RUN_PROJECTION_PUBLISHER_VERSION,
                input_run_ids=(run.run_id,),
                input_artifact_ids=(),
                operation_identity=operation_identity,
            ),
            replay_context=context.model_dump(mode="json"),
        )

    def stage(self, spec: ProjectionIntentSpec) -> ProjectionIntent:
        """Persist an immutable replay recipe before a separately built projection."""

        return self.intents.create(spec)

    def publish_rows(
        self,
        intent_id: str,
        rows: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> PublishedGeneration:
        intent = self.intents.read(intent_id)
        if intent.state is ProjectionState.FAILED:
            intent = self.intents.retry(intent_id)
        if tuple(rows) != intent.expected_tables:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Projection rows do not match the immutable intent table set.",
            )
        return self._publish(intent, rows)

    def reconcile(self, *, include_failed: bool = True) -> ProjectionReconciliationResult:
        candidates = list(self.intents.list(state=ProjectionState.PENDING))
        if include_failed:
            candidates.extend(self.intents.list(state=ProjectionState.FAILED))
        published = 0
        failed = 0
        deferred = 0
        already_terminal = 0
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
            elif result.state is ProjectionState.FAILED:
                failed += 1
            else:
                deferred += 1
        return ProjectionReconciliationResult(
            inspected=len(candidates),
            published=published,
            failed=failed,
            deferred=deferred,
            already_terminal=already_terminal,
            intent_ids=tuple(intent_ids),
        )

    def reconcile_intent(self, intent_id: str) -> ProjectionIntent:
        intent = self.intents.read(intent_id)
        if intent.state is ProjectionState.PUBLISHED:
            return intent
        if intent.state is ProjectionState.FAILED and intent.projection_kind == RUN_PROJECTION_KIND:
            intent = self.intents.retry(intent.intent_id)
        elif intent.state is ProjectionState.FAILED:
            return intent
        latest = self.intents.latest(
            domain_kind=intent.domain_kind,
            domain_id=intent.domain_id,
            projection_kind=intent.projection_kind,
        )
        if (
            latest is not None
            and latest.domain_revision > intent.domain_revision
            and latest.state is ProjectionState.PUBLISHED
        ):
            return self.intents.failed(
                intent.intent_id,
                failure_code="superseded_revision",
                failure_message="A newer domain revision already owns the current projection.",
            )
        if intent.projection_kind == RUN_PROJECTION_KIND:
            self._publish_run_intent(intent)
            return self.intents.read(intent.intent_id)
        # A feature-specific projector may replay this recipe later. Keep it
        # pending rather than turning missing in-process routing into a durable
        # claim that the source data is invalid.
        return intent

    def _publish_run_intent(self, intent: ProjectionIntent) -> PublishedGeneration:
        run, rows = self._run_projection_rows(intent)
        if run.run_id != intent.domain_id:
            raise DomainError(ErrorCode.ARTIFACT_INTEGRITY_FAILED, "Run projection id changed.")
        return self._publish(intent, rows)

    def _run_projection_rows(
        self,
        intent: ProjectionIntent,
    ) -> tuple[RunManifest, Mapping[str, Sequence[Mapping[str, Any]]]]:
        run = self.runs.read_revision(intent.domain_id, intent.domain_revision)
        if digest_model(run.model_dump(mode="json")) != intent.domain_digest:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Projection intent no longer matches its immutable run revision.",
            )
        context = RunProjectionReplayContext.model_validate(intent.replay_context)
        if (
            context.environment is not None
            and context.environment.environment_id != run.environment_id
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Projection replay environment does not match the run revision.",
            )
        if (
            context.source_state is not None
            and context.source_state.source_state_id != run.source_state_id
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Projection replay source state does not match the run revision.",
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
        if tuple(rows) != intent.expected_tables:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Projection replay tables do not match the immutable intent.",
            )
        return run, rows

    def _publish(
        self,
        intent: ProjectionIntent,
        rows: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> PublishedGeneration:
        try:
            published = self.publisher.publish_rows_idempotent(
                rows,
                publisher=intent.publisher,
                publisher_version=intent.publisher_version,
                input_run_ids=intent.input_run_ids,
                input_artifact_ids=intent.input_artifact_ids,
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
                self.intents.failed(
                    intent.intent_id,
                    failure_code=code,
                    failure_message=message[:500],
                )
            except Exception as status_error:
                error.add_note(
                    "Projection failure status could not be recorded: "
                    f"{type(status_error).__name__}."
                )
            raise
        if published.manifest.operation_digest != intent.operation_digest:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Published generation does not match its projection operation identity.",
            )
        self._inject("after_corpus_publish", intent)
        completed = self.intents.published(
            intent.intent_id,
            generation_id=published.manifest.generation_id,
            corpus_commit_id=published.commit.commit_id,
        )
        self._inject("after_intent_commit", completed)
        return published

    def _inject(self, phase: ProjectionPhase, intent: ProjectionIntent) -> None:
        if self.fault_injector is not None:
            self.fault_injector(phase, intent)
