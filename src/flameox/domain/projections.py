from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import Field, StringConstraints, model_validator

from flameox.domain.identity import digest_model
from flameox.domain.models import Digest, EnvironmentRecord, SourceState
from flameox.models import ContractModel

ProjectionName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=100, pattern=r"^[a-z0-9_.-]+$"),
]


class ProjectionState(StrEnum):
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"


class RunProjectionContext(ContractModel):
    environment: EnvironmentRecord | None = None
    source_state: SourceState | None = None


class ProjectionIntentSpec(ContractModel):
    """Immutable run-projection publication intent."""

    intent_id: Digest
    run_id: str
    run_revision: Annotated[int, Field(ge=0)]
    run_digest: Digest
    context: RunProjectionContext = Field(default_factory=RunProjectionContext)


class ProjectionIntent(ProjectionIntentSpec):
    state: ProjectionState
    generation_id: str | None = None
    corpus_commit_id: str | None = None
    failure_code: ProjectionName | None = None
    failure_message: Annotated[str, StringConstraints(min_length=1, max_length=500)] | None = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def state_fields_are_coherent(self) -> ProjectionIntent:
        publication = self.generation_id is not None or self.corpus_commit_id is not None
        failure = self.failure_code is not None or self.failure_message is not None
        if self.state is ProjectionState.PENDING and (publication or failure):
            raise ValueError("pending projection intents cannot contain terminal fields")
        if self.state is ProjectionState.PUBLISHED and (
            self.generation_id is None or self.corpus_commit_id is None or failure
        ):
            raise ValueError("published projection intents require one generation and commit")
        if self.state is ProjectionState.FAILED and (
            self.failure_code is None or self.failure_message is None or publication
        ):
            raise ValueError("failed projection intents require a bounded failure")
        return self


def projection_intent_id(
    *,
    workspace_id: str,
    run_id: str,
    run_revision: int,
) -> str:
    return digest_model(
        {
            "workspace_id": workspace_id,
            "run_id": run_id,
            "run_revision": run_revision,
        }
    )
