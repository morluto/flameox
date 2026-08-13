from __future__ import annotations

from flameox.domain.projections import ProjectionIntent, ProjectionIntentSpec, ProjectionState
from flameox.storage.control_plane import ControlPlane
from flameox.storage.workspace import Workspace


class ProjectionIntentStore:
    """Typed access to the projection outbox in the SQLite control plane."""

    def __init__(self, workspace: Workspace) -> None:
        self.control_plane = ControlPlane(workspace)

    def create(self, spec: ProjectionIntentSpec) -> ProjectionIntent:
        return self.control_plane.create_projection_intent(spec)

    def read(self, intent_id: str) -> ProjectionIntent:
        return self.control_plane.read_projection_intent(intent_id)

    def list(self, *, state: ProjectionState | None = None) -> tuple[ProjectionIntent, ...]:
        return self.control_plane.list_projection_intents(state=state)

    def latest(
        self,
        *,
        domain_kind: str,
        domain_id: str,
        projection_kind: str,
    ) -> ProjectionIntent | None:
        return self.control_plane.latest_projection_intent(
            domain_kind=domain_kind,
            domain_id=domain_id,
            projection_kind=projection_kind,
        )

    def published(
        self,
        intent_id: str,
        *,
        generation_id: str,
        corpus_commit_id: str,
    ) -> ProjectionIntent:
        return self.control_plane.mark_projection_published(
            intent_id=intent_id,
            generation_id=generation_id,
            corpus_commit_id=corpus_commit_id,
        )

    def failed(
        self,
        intent_id: str,
        *,
        failure_code: str,
        failure_message: str,
    ) -> ProjectionIntent:
        return self.control_plane.mark_projection_failed(
            intent_id=intent_id,
            failure_code=failure_code,
            failure_message=failure_message,
        )

    def retry(self, intent_id: str) -> ProjectionIntent:
        return self.control_plane.retry_projection(intent_id)
