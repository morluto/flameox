from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import Field, JsonValue, StringConstraints, model_validator

from flameox.domain.identity import digest_model
from flameox.models import ContractModel

ProjectionName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=100, pattern=r"^[a-z0-9_.-]+$"),
]


class ProjectionState(StrEnum):
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"


class ProjectionIntentSpec(ContractModel):
    """Immutable identity and replay recipe for one domain projection."""

    intent_id: str
    workspace_id: str
    domain_kind: ProjectionName
    domain_id: str
    domain_revision: Annotated[int, Field(ge=0)]
    domain_digest: str
    projection_kind: ProjectionName
    publisher: ProjectionName
    publisher_version: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    input_run_ids: Annotated[tuple[str, ...], Field(max_length=100)] = ()
    input_artifact_ids: Annotated[tuple[str, ...], Field(max_length=100)] = ()
    expected_tables: Annotated[tuple[ProjectionName, ...], Field(min_length=1, max_length=64)]
    operation_digest: str
    replay_context: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def identity_matches_content(self) -> ProjectionIntentSpec:
        expected = projection_intent_id(
            workspace_id=self.workspace_id,
            domain_kind=self.domain_kind,
            domain_id=self.domain_id,
            domain_revision=self.domain_revision,
            projection_kind=self.projection_kind,
        )
        if self.intent_id != expected:
            raise ValueError("projection intent id must match its domain projection identity")
        if len(set(self.input_run_ids)) != len(self.input_run_ids):
            raise ValueError("projection input run ids must be unique")
        if len(set(self.input_artifact_ids)) != len(self.input_artifact_ids):
            raise ValueError("projection input artifact ids must be unique")
        if len(set(self.expected_tables)) != len(self.expected_tables):
            raise ValueError("projection tables must be unique")
        return self


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
    domain_kind: str,
    domain_id: str,
    domain_revision: int,
    projection_kind: str,
) -> str:
    return digest_model(
        {
            "workspace_id": workspace_id,
            "domain_kind": domain_kind,
            "domain_id": domain_id,
            "domain_revision": domain_revision,
            "projection_kind": projection_kind,
        }
    )
