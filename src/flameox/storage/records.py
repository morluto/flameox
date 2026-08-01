from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from flameox.atomic import atomic_write_json
from flameox.domain import DomainError, ErrorCode
from flameox.storage.workspace import Workspace

RecordT = TypeVar("RecordT", bound=BaseModel)


class JsonRecordStore[RecordT: BaseModel]:
    """Immutable JSON revisions plus an atomic current projection."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        kind: str,
        model: type[RecordT],
        id_field: str,
        revision_field: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.kind = kind
        self.model = model
        self.id_field = id_field
        self.revision_field = revision_field

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
        identifier = self._identifier(record)
        root = self._root(identifier)
        if root.exists():
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                f"{self.kind} {identifier!r} already exists.",
            )
        root.mkdir(parents=True)
        if self.revision_field is not None:
            (root / "revisions").mkdir()
            self._write_revision(record)
        self._write_projection(record)
        return record

    def read(self, identifier: str) -> RecordT:
        try:
            return self.model.model_validate_json(
                (self._root(identifier) / "record.json").read_text()
            )
        except (FileNotFoundError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"{self.kind} {identifier!r} does not exist or is invalid.",
            ) from exc

    def list(self) -> tuple[RecordT, ...]:
        root = self.workspace.paths.records / self.kind
        if not root.exists():
            return ()
        records: list[RecordT] = []
        for path in sorted(root.glob("*/record.json")):
            try:
                records.append(self.model.model_validate_json(path.read_text()))
            except ValueError as exc:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Invalid {self.kind} record at {path}.",
                ) from exc
        return tuple(records)

    def append(self, record: RecordT, *, expected_revision: int) -> RecordT:
        if self.revision_field is None:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                f"{self.kind} records do not support revisions.",
            )
        identifier = self._identifier(record)
        with self.workspace.write_locked():
            current = self.read(identifier)
            actual = self._revision(current)
            next_revision = self._revision(record)
            if next_revision != expected_revision + 1:
                # Caller bug: the supplied next revision does not match the
                # expected sequence. Retrying would loop forever, so this is
                # not retryable.
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"{self.kind} {identifier!r} has a stale expected revision.",
                    details={
                        "expected_revision": expected_revision,
                        "supplied_next_revision": next_revision,
                    },
                )
            if actual != expected_revision:
                # Genuine race: another writer committed first. Retryable.
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"{self.kind} {identifier!r} changed before the update.",
                    retryable=True,
                    details={
                        "expected_revision": expected_revision,
                        "actual_revision": actual,
                    },
                )
            self._write_revision(record)
            self._write_projection(record)
        return record

    def _root(self, identifier: str) -> Path:
        if not identifier or "/" in identifier or "\\" in identifier or "\x00" in identifier:
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Invalid record identifier.")
        return self.workspace.paths.records / self.kind / identifier

    def _identifier(self, record: RecordT) -> str:
        value = getattr(record, self.id_field, None)
        if not isinstance(value, str):
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Record identifier is invalid.")
        return value

    def _revision(self, record: RecordT) -> int:
        assert self.revision_field is not None
        value = getattr(record, self.revision_field, None)
        if not isinstance(value, int):
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Record revision is invalid.")
        return value

    def _write_revision(self, record: RecordT) -> None:
        revision = self._revision(record)
        path = self._root(self._identifier(record)) / "revisions" / f"{revision:08d}.json"
        if path.exists():
            existing = self.model.model_validate_json(path.read_text())
            if existing != record:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "An immutable record revision already contains different data.",
                )
            return
        atomic_write_json(path, record.model_dump(mode="json"))

    def _write_projection(self, record: RecordT) -> None:
        atomic_write_json(
            self._root(self._identifier(record)) / "record.json",
            record.model_dump(mode="json"),
        )
