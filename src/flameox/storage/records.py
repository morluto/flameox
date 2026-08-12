from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

from flameox.domain import DomainError, ErrorCode
from flameox.storage.control_plane import ControlPlane, canonical_json
from flameox.storage.workspace import Workspace

RecordT = TypeVar("RecordT", bound=BaseModel)
type ModelFieldSelection = (
    set[int]
    | set[str]
    | Mapping[int, ModelFieldSelection | bool]
    | Mapping[str, ModelFieldSelection | bool]
)


class JsonRecordStore[RecordT: BaseModel]:
    """Immutable JSON revisions plus an atomic current projection."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        kind: str,
        model: type[RecordT] | TypeAdapter[RecordT],
        id_field: str,
        revision_field: str | None = None,
        output_only_fields: ModelFieldSelection | None = None,
    ) -> None:
        self.workspace = workspace
        self.kind = kind
        self._adapter = TypeAdapter(model) if isinstance(model, type) else model
        self.id_field = id_field
        self.revision_field = revision_field
        self.output_only_fields = output_only_fields or set()
        self.control_plane = ControlPlane(workspace)

    def create(self, record: RecordT) -> RecordT:
        with self.workspace.write_locked():
            self.create_locked(record)
        return record

    def create_locked(self, record: RecordT) -> RecordT:
        """Create a record while the caller owns the workspace write lock.

        This is intentionally a small escape hatch for compound operations
        that must inspect and create records in one cross-process critical
        section. Callers must hold ``workspace.write_locked()``.
        """
        persisted = self._canonical(record)
        identifier = self._identifier(persisted)
        self.control_plane.create_record(
            kind=self.kind,
            record_id=identifier,
            revision=(self._revision(persisted) if self.revision_field is not None else None),
            payload_json=self._json(persisted),
        )
        return record

    def read(self, identifier: str) -> RecordT:
        try:
            return self._adapter.validate_json(
                self.control_plane.read_record(kind=self.kind, record_id=identifier)
            )
        except ValueError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"{self.kind} {identifier!r} does not exist or is invalid.",
            ) from exc

    def create_idempotent(
        self,
        record: RecordT,
        *,
        idempotency_digest: str,
        intent_digest: str,
    ) -> tuple[RecordT, bool]:
        persisted = self._canonical(record)
        stored_intent, payload, created = self.control_plane.create_idempotent_record(
            kind=self.kind,
            record_id=self._identifier(persisted),
            revision=(self._revision(persisted) if self.revision_field is not None else None),
            payload_json=self._json(persisted),
            idempotency_digest=idempotency_digest,
            intent_digest=intent_digest,
        )
        if stored_intent != intent_digest:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The idempotency key is already bound to another intent.",
            )
        try:
            return self._adapter.validate_json(payload), created
        except ValueError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                f"Invalid {self.kind} idempotency record in the control plane.",
            ) from exc

    def read_idempotent(
        self,
        *,
        idempotency_digest: str,
    ) -> tuple[str, RecordT] | None:
        stored = self.control_plane.read_idempotent_record(
            kind=self.kind,
            idempotency_digest=idempotency_digest,
        )
        if stored is None:
            return None
        intent_digest, payload = stored
        try:
            return intent_digest, self._adapter.validate_json(payload)
        except ValueError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                f"Invalid {self.kind} idempotency record in the control plane.",
            ) from exc

    def list(self) -> tuple[RecordT, ...]:
        records: list[RecordT] = []
        for payload in self.control_plane.list_records(kind=self.kind):
            try:
                records.append(self._adapter.validate_json(payload))
            except ValueError as exc:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Invalid {self.kind} record in the control plane.",
                ) from exc
        return tuple(records)

    def append(self, record: RecordT, *, expected_revision: int) -> RecordT:
        with self.workspace.write_locked():
            self.append_locked(record, expected_revision=expected_revision)
        return record

    def append_locked(self, record: RecordT, *, expected_revision: int) -> RecordT:
        """Append a revision while the caller owns the workspace write lock."""
        if self.revision_field is None:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                f"{self.kind} records do not support revisions.",
            )
        persisted = self._canonical(record)
        identifier = self._identifier(persisted)
        next_revision = self._revision(persisted)
        self.control_plane.append_record(
            kind=self.kind,
            record_id=identifier,
            expected_revision=expected_revision,
            next_revision=next_revision,
            payload_json=self._json(persisted),
        )
        return record

    def _canonical(self, record: RecordT) -> RecordT:
        try:
            return self._adapter.validate_python(
                record.model_dump(mode="python", exclude=self.output_only_fields)
            )
        except ValueError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Invalid {self.kind} record cannot be persisted.",
            ) from exc

    def _validate_identifier(self, identifier: str) -> str:
        if (
            not identifier
            or "/" in identifier
            or "\\" in identifier
            or "\x00" in identifier
            or identifier == ".."
            or identifier.startswith(".")
        ):
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Invalid record identifier.")
        return identifier

    def _identifier(self, record: RecordT) -> str:
        value = getattr(record, self.id_field, None)
        if not isinstance(value, str):
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Record identifier is invalid.")
        return self._validate_identifier(value)

    def _revision(self, record: RecordT) -> int:
        if self.revision_field is None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Record revision field is not configured.",
            )
        value = getattr(record, self.revision_field, None)
        if not isinstance(value, int):
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Record revision is invalid.")
        return value

    def _json(self, record: RecordT) -> str:
        return canonical_json(record.model_dump(mode="json", exclude=self.output_only_fields))
