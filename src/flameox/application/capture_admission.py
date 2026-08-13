from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from contextlib import suppress
from functools import partial
from uuid import uuid4

import anyio
import psutil

from flameox.application.staging_ownership import exact_process_is_dead, observe_process_lease
from flameox.domain import DomainError, ErrorCode
from flameox.domain.models import utc_now
from flameox.storage import CaptureAdmissionRecord, CaptureAdmissionStore, Workspace


class CaptureAdmission:
    def __init__(
        self,
        store: CaptureAdmissionStore,
        record: CaptureAdmissionRecord,
    ) -> None:
        self._store = store
        self._record = record
        self._released = False

    async def run[T](self, work: Awaitable[T]) -> T:
        heartbeat = asyncio.create_task(
            self._heartbeat(),
            name=f"capture-admission-{self._record.run_id}",
        )
        try:
            return await work
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

    def release(self) -> None:
        if self._released:
            return
        self._store.release(self._record)
        self._released = True

    async def _heartbeat(self) -> None:
        while True:
            await anyio.sleep(10)
            lease = await anyio.to_thread.run_sync(
                observe_process_lease,
                self._record.process_lease.process_id,
            )
            self._record = await anyio.to_thread.run_sync(
                partial(self._store.heartbeat, self._record, process_lease=lease)
            )


class CaptureAdmissionService:
    def __init__(self, workspace: Workspace, *, limit: int) -> None:
        self.workspace = workspace
        self.limit = limit
        self.store = CaptureAdmissionStore(workspace)

    async def acquire(self, run_id: str) -> CaptureAdmission:
        owner_id = uuid4().hex
        while True:
            await anyio.to_thread.run_sync(self._reclaim_dead_owners)
            try:
                lease = await anyio.to_thread.run_sync(observe_process_lease)
            except (OSError, ValueError, psutil.Error) as exc:
                raise DomainError(
                    ErrorCode.PROCESS_FAILED,
                    "Could not establish workspace capture-admission ownership.",
                    run_id=run_id,
                ) from exc
            record = CaptureAdmissionRecord(
                run_id=run_id,
                owner_id=owner_id,
                process_lease=lease,
                acquired_at=utc_now(),
            )
            acquired = await anyio.to_thread.run_sync(
                partial(self.store.try_acquire, record, limit=self.limit)
            )
            if acquired:
                return CaptureAdmission(self.store, record)
            await anyio.sleep(0.1)

    def _reclaim_dead_owners(self) -> None:
        for record in self.store.list():
            if exact_process_is_dead(record.process_lease):
                self.store.reclaim(record)
