from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import psutil

from flameox.application.proc import read_boot_id, read_proc_stat_start_identity
from flameox.domain import CaptureLease, DomainError, ErrorCode, digest_model
from flameox.domain.models import utc_now
from flameox.storage import (
    StagingOwnerRecord,
    StagingOwnershipStore,
    StagingOwnerState,
    Workspace,
)


def observe_process_lease(process_id: int | None = None) -> CaptureLease:
    process_id = process_id or os.getpid()
    observed = utc_now()
    return CaptureLease(
        process_id=process_id,
        process_start_identity=read_proc_stat_start_identity(process_id),
        boot_id=read_boot_id(),
        heartbeat_monotonic_ns=time.monotonic_ns(),
        observed_at=observed,
        expires_at=observed + timedelta(seconds=60),
    )


def exact_process_is_dead(lease: CaptureLease) -> bool:
    try:
        if lease.boot_id != read_boot_id():
            return True
        return read_proc_stat_start_identity(lease.process_id) != lease.process_start_identity
    except (FileNotFoundError, psutil.NoSuchProcess):
        return True
    except (OSError, ValueError, psutil.Error):
        # Inability to prove death preserves the staging data.
        return False


@dataclass(slots=True)
class StagingOwnership:
    store: StagingOwnershipStore
    record: StagingOwnerRecord
    forgotten: bool = False

    def release(self) -> StagingOwnerRecord:
        if self.record.state is StagingOwnerState.RELEASED:
            return self.record
        self.record = self.store.release(self.record)
        return self.record

    def forget_if_removed(self, path: Path) -> None:
        if self.forgotten or path.exists() or path.is_symlink():
            return
        self.store.delete(self.record)
        self.forgotten = True


class StagingOwnershipService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.store = StagingOwnershipStore(workspace)

    def acquire(
        self,
        path: Path,
        *,
        owner_kind: str,
        owner_id: str | None = None,
    ) -> StagingOwnership:
        relative = self._relative(path)
        now = utc_now()
        try:
            lease = observe_process_lease()
        except (OSError, ValueError, psutil.Error) as exc:
            raise DomainError(
                ErrorCode.PROCESS_FAILED,
                "Could not establish exact staging ownership.",
                details={"path": relative},
            ) from exc
        record = StagingOwnerRecord(
            path=relative,
            owner_kind=owner_kind,
            owner_id=owner_id or uuid4().hex,
            process_lease=lease,
            created_at=now,
            updated_at=now,
        )
        return StagingOwnership(self.store, self.store.acquire(record))

    def collectible(self, path: Path) -> tuple[StagingOwnerRecord | None, str] | None:
        relative = self._relative(path)
        record = self.store.read(relative)
        if record is None:
            return None, digest_model({"path": relative, "ownership": "unowned"})
        if record.state is StagingOwnerState.ACTIVE and not exact_process_is_dead(
            record.process_lease
        ):
            return None
        return record, digest_model(record.model_dump(mode="json"))

    def _relative(self, path: Path) -> str:
        try:
            relative = path.relative_to(self.workspace.paths.root)
        except ValueError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Staging ownership path escapes the workspace.",
            ) from exc
        if not relative.parts or relative.parts[0] != "staging" or ".." in relative.parts:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Staging ownership path is not a workspace staging root.",
            )
        return relative.as_posix()


__all__ = [
    "StagingOwnership",
    "StagingOwnershipService",
    "exact_process_is_dead",
    "observe_process_lease",
]
