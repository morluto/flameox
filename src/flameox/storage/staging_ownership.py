from __future__ import annotations

import sqlite3
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import model_validator

from flameox.domain import CaptureLease, DomainError, ErrorCode
from flameox.domain.models import utc_now
from flameox.models import ContractModel
from flameox.storage.control_plane import ControlPlane, canonical_json
from flameox.storage.workspace import Workspace


class StagingOwnerState(StrEnum):
    ACTIVE = "active"
    RELEASED = "released"


class StagingOwnerRecord(ContractModel):
    schema_version: Literal[1] = 1
    path: str
    owner_kind: str
    owner_id: str
    state: StagingOwnerState = StagingOwnerState.ACTIVE
    process_lease: CaptureLease
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def update_follows_creation(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("staging ownership update predates creation")
        return self

    def release(self) -> StagingOwnerRecord:
        return self.validated_copy(
            update={"state": StagingOwnerState.RELEASED, "updated_at": utc_now()}
        )


class StagingOwnershipStore:
    """Durable exact-process ownership for recoverable staging roots."""

    def __init__(self, workspace: Workspace) -> None:
        self.control_plane = ControlPlane(workspace)

    def acquire(self, record: StagingOwnerRecord) -> StagingOwnerRecord:
        lease = record.process_lease
        payload = canonical_json(record.model_dump(mode="json"))
        try:
            with self.control_plane.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO staging_owners(
                        path, owner_kind, owner_id, state, process_id,
                        process_start_identity, boot_id, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.path,
                        record.owner_kind,
                        record.owner_id,
                        record.state,
                        lease.process_id,
                        lease.process_start_identity,
                        lease.boot_id,
                        payload,
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "The staging root already has a durable owner.",
                details={"path": record.path},
            ) from exc
        return record

    def read(self, path: str) -> StagingOwnerRecord | None:
        with self.control_plane.transaction() as connection:
            row = connection.execute(
                "SELECT payload_json FROM staging_owners WHERE path = ?",
                (path,),
            ).fetchone()
        if row is None:
            return None
        try:
            return StagingOwnerRecord.model_validate_json(str(row["payload_json"]))
        except ValueError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "A staging ownership record is invalid.",
                details={"path": path},
            ) from exc

    def release(self, record: StagingOwnerRecord) -> StagingOwnerRecord:
        released = record.release()
        lease = record.process_lease
        with self.control_plane.transaction() as connection:
            changed = connection.execute(
                """
                UPDATE staging_owners
                SET state = 'released', payload_json = ?, updated_at = ?
                WHERE path = ? AND owner_id = ? AND state = 'active'
                  AND process_id = ? AND process_start_identity = ? AND boot_id = ?
                """,
                (
                    canonical_json(released.model_dump(mode="json")),
                    released.updated_at.isoformat(),
                    record.path,
                    record.owner_id,
                    lease.process_id,
                    lease.process_start_identity,
                    lease.boot_id,
                ),
            ).rowcount
        if changed != 1:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Staging ownership changed before release.",
                details={"path": record.path},
            )
        return released

    def delete(self, record: StagingOwnerRecord) -> None:
        lease = record.process_lease
        with self.control_plane.transaction() as connection:
            changed = connection.execute(
                """
                DELETE FROM staging_owners
                WHERE path = ? AND owner_id = ?
                  AND process_id = ? AND process_start_identity = ? AND boot_id = ?
                """,
                (
                    record.path,
                    record.owner_id,
                    lease.process_id,
                    lease.process_start_identity,
                    lease.boot_id,
                ),
            ).rowcount
        if changed != 1:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Staging ownership changed before deletion.",
                details={"path": record.path},
            )
