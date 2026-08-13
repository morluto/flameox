from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from flameox.domain import DomainError, ErrorCode, digest_model
from flameox.domain.models import utc_now
from flameox.models import ContractModel
from flameox.storage.control_plane import ControlPlane, canonical_json
from flameox.storage.workspace import Workspace


class _RetentionIntent(ContractModel):
    schema_version: Literal[1] = 1
    intent_id: str
    revision: int = Field(ge=1)
    corpus_commit_id: str
    owner_kind: str
    owner_id: str
    operation_digest: str
    created_at: datetime


class PendingRetentionIntent(_RetentionIntent):
    state: Literal["pending"] = "pending"
    revision: Literal[1] = 1

    def complete(self, *, materialized_commit_id: str) -> CompletedRetentionIntent:
        return CompletedRetentionIntent.model_validate(
            {
                **self.model_dump(mode="python"),
                "state": "completed",
                "revision": 2,
                "materialized_commit_id": materialized_commit_id,
                "completed_at": utc_now(),
            }
        )


class CompletedRetentionIntent(_RetentionIntent):
    state: Literal["completed"] = "completed"
    revision: Literal[2] = 2
    materialized_commit_id: str
    completed_at: datetime


type RetentionIntent = Annotated[
    PendingRetentionIntent | CompletedRetentionIntent,
    Field(discriminator="state"),
]


_RETENTION_INTENT_ADAPTER: TypeAdapter[RetentionIntent] = TypeAdapter(RetentionIntent)


class RetentionIntentStore:
    """Durable GC roots that bridge a live snapshot to its published result.

    This store uses the SQLite transaction directly because intent creation is control-plane
    state, not corpus publication. The ranked workspace lock manager separately orders
    retention, catalog, and publication locks.
    """

    _KIND = "retention_intents"

    def __init__(self, workspace: Workspace) -> None:
        self.control_plane = ControlPlane(workspace)

    def acquire(
        self,
        *,
        corpus_commit_id: str,
        owner_kind: str,
        owner_id: str,
        operation_digest: str,
    ) -> RetentionIntent:
        intent_id = digest_model(
            {
                "schema_version": 1,
                "corpus_commit_id": corpus_commit_id,
                "owner_kind": owner_kind,
                "owner_id": owner_id,
                "operation_digest": operation_digest,
            }
        )
        pending = PendingRetentionIntent(
            intent_id=intent_id,
            corpus_commit_id=corpus_commit_id,
            owner_kind=owner_kind,
            owner_id=owner_id,
            operation_digest=operation_digest,
            created_at=utc_now(),
        )
        try:
            self.control_plane.create_record(
                kind=self._KIND,
                record_id=intent_id,
                revision=pending.revision,
                payload_json=canonical_json(pending.model_dump(mode="json")),
            )
            return pending
        except DomainError as error:
            if error.code is not ErrorCode.REVISION_CONFLICT:
                raise
        existing = self.read(intent_id)
        if (
            existing.corpus_commit_id != corpus_commit_id
            or existing.owner_kind != owner_kind
            or existing.owner_id != owner_id
            or existing.operation_digest != operation_digest
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "A retention intent identity is bound to different immutable inputs.",
                details={"retention_intent_id": intent_id},
            )
        return existing

    def complete(
        self,
        intent: RetentionIntent,
        *,
        materialized_commit_id: str,
    ) -> CompletedRetentionIntent:
        if isinstance(intent, CompletedRetentionIntent):
            if intent.materialized_commit_id != materialized_commit_id:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "A completed retention intent names a different materialized commit.",
                    details={"retention_intent_id": intent.intent_id},
                )
            return intent
        completed = intent.complete(materialized_commit_id=materialized_commit_id)
        try:
            self.control_plane.append_record(
                kind=self._KIND,
                record_id=intent.intent_id,
                expected_revision=intent.revision,
                next_revision=completed.revision,
                payload_json=canonical_json(completed.model_dump(mode="json")),
            )
            return completed
        except DomainError as error:
            if error.code is not ErrorCode.REVISION_CONFLICT:
                raise
        current = self.read(intent.intent_id)
        if (
            not isinstance(current, CompletedRetentionIntent)
            or current.materialized_commit_id != materialized_commit_id
        ):
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "The retention intent changed before completion.",
                retryable=True,
                details={"retention_intent_id": intent.intent_id},
            )
        return current

    def read(self, intent_id: str) -> RetentionIntent:
        try:
            return _RETENTION_INTENT_ADAPTER.validate_json(
                self.control_plane.read_record(kind=self._KIND, record_id=intent_id)
            )
        except ValueError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                f"Retention intent {intent_id!r} is invalid.",
            ) from exc

    def pending(self) -> tuple[PendingRetentionIntent, ...]:
        values: list[PendingRetentionIntent] = []
        for payload in self.control_plane.list_records(kind=self._KIND):
            try:
                intent = _RETENTION_INTENT_ADAPTER.validate_json(payload)
            except ValueError as exc:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "A retention intent is invalid.",
                ) from exc
            if isinstance(intent, PendingRetentionIntent):
                values.append(intent)
        return tuple(values)

    def pending_commit_ids(self) -> tuple[str, ...]:
        return tuple(sorted({intent.corpus_commit_id for intent in self.pending()}))
