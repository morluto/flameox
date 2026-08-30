from __future__ import annotations

from datetime import datetime

from flameox.domain import CaptureLease, DomainError, ErrorCode
from flameox.models import ContractModel
from flameox.storage.control_plane import ControlPlane, _serialize_control_payload
from flameox.storage.workspace import Workspace


class CaptureAdmissionRecord(ContractModel):
    run_id: str
    owner_id: str
    process_lease: CaptureLease
    acquired_at: datetime


class CaptureAdmissionStore:
    """Transactional workspace-wide collector admission leases."""

    def __init__(self, workspace: Workspace) -> None:
        self.control_plane = ControlPlane(workspace)

    def try_acquire(self, record: CaptureAdmissionRecord, *, limit: int) -> bool:
        if limit < 1:
            raise ValueError("capture admission limit must be positive")
        payload = _serialize_control_payload(record.model_dump(mode="json"))
        lease = record.process_lease
        with self.control_plane.transaction() as connection:
            existing = connection.execute(
                "SELECT owner_id FROM capture_admissions WHERE run_id = ?",
                (record.run_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["owner_id"]) == record.owner_id:
                    return True
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "The capture run already has a different admission owner.",
                    details={"run_id": record.run_id},
                )
            count = connection.execute("SELECT count(*) FROM capture_admissions").fetchone()
            assert count is not None
            if int(count[0]) >= limit:
                return False
            connection.execute(
                """
                INSERT INTO capture_admissions(
                    run_id, owner_id, process_id, process_start_identity, boot_id,
                    payload_json, acquired_at, heartbeat_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.owner_id,
                    lease.process_id,
                    lease.process_start_identity,
                    lease.boot_id,
                    payload,
                    record.acquired_at.isoformat(),
                    lease.observed_at.isoformat(),
                    lease.expires_at.isoformat(),
                ),
            )
        return True

    def heartbeat(
        self,
        record: CaptureAdmissionRecord,
        *,
        process_lease: CaptureLease,
    ) -> CaptureAdmissionRecord:
        previous = record.process_lease
        if (
            process_lease.process_id != previous.process_id
            or process_lease.process_start_identity != previous.process_start_identity
            or process_lease.boot_id != previous.boot_id
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Capture admission heartbeat changed exact process identity.",
                details={"run_id": record.run_id},
            )
        updated = record.validated_copy(update={"process_lease": process_lease})
        with self.control_plane.transaction() as connection:
            changed = connection.execute(
                """
                UPDATE capture_admissions
                SET payload_json = ?, heartbeat_at = ?, expires_at = ?
                WHERE run_id = ? AND owner_id = ?
                  AND process_id = ? AND process_start_identity = ? AND boot_id = ?
                """,
                (
                    _serialize_control_payload(updated.model_dump(mode="json")),
                    process_lease.observed_at.isoformat(),
                    process_lease.expires_at.isoformat(),
                    record.run_id,
                    record.owner_id,
                    previous.process_id,
                    previous.process_start_identity,
                    previous.boot_id,
                ),
            ).rowcount
        if changed != 1:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Capture admission ownership changed before heartbeat.",
                details={"run_id": record.run_id},
            )
        return updated

    def release(self, record: CaptureAdmissionRecord) -> None:
        lease = record.process_lease
        with self.control_plane.transaction() as connection:
            changed = connection.execute(
                """
                DELETE FROM capture_admissions
                WHERE run_id = ? AND owner_id = ?
                  AND process_id = ? AND process_start_identity = ? AND boot_id = ?
                """,
                (
                    record.run_id,
                    record.owner_id,
                    lease.process_id,
                    lease.process_start_identity,
                    lease.boot_id,
                ),
            ).rowcount
        if changed != 1:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Capture admission ownership changed before release.",
                details={"run_id": record.run_id},
            )

    def reclaim(self, record: CaptureAdmissionRecord) -> bool:
        lease = record.process_lease
        with self.control_plane.transaction() as connection:
            changed = connection.execute(
                """
                DELETE FROM capture_admissions
                WHERE run_id = ? AND owner_id = ?
                  AND process_id = ? AND process_start_identity = ? AND boot_id = ?
                """,
                (
                    record.run_id,
                    record.owner_id,
                    lease.process_id,
                    lease.process_start_identity,
                    lease.boot_id,
                ),
            ).rowcount
        return changed == 1

    def list(self) -> tuple[CaptureAdmissionRecord, ...]:
        with self.control_plane.transaction() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM capture_admissions ORDER BY acquired_at, run_id"
            ).fetchall()
        try:
            return tuple(
                CaptureAdmissionRecord.model_validate_json(str(row["payload_json"])) for row in rows
            )
        except ValueError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "A capture admission lease is invalid.",
            ) from exc
