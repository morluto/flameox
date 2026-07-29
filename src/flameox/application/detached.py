from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field

from flameox.application.capture import CaptureService
from flameox.domain import (
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    digest_model,
)
from flameox.domain.models import utc_now
from flameox.models import ContractModel
from flameox.storage import JsonRecordStore, RunStore, Workspace


class DetachedProgress(ContractModel):
    completed: Annotated[float, Field(ge=0)]
    total: Annotated[float, Field(gt=0)]
    message: Annotated[str, Field(min_length=1, max_length=500)]
    observed_at: datetime = Field(default_factory=utc_now)


class DetachedCaptureRecord(ContractModel):
    schema_version: Literal[1] = 1
    run_id: str
    revision: Annotated[int, Field(ge=0)] = 0
    idempotency_digest: str
    plan_digest: str
    state: Literal["starting", "running", "terminal", "failed_to_start"] = "starting"
    progress: Annotated[tuple[DetachedProgress, ...], Field(max_length=16)] = ()
    failure_code: str | None = None
    failure_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class DetachedCaptureStatus(ContractModel):
    schema_version: Literal[1] = 1
    run_id: str
    state: Literal[
        "starting",
        "running",
        "terminal",
        "failed_to_start",
        "unmanaged_after_restart",
    ]
    execution_status: ExecutionStatus | None = None
    capture_status: CaptureStatus | None = None
    artifact_count: Annotated[int, Field(ge=0)] = 0
    progress: tuple[DetachedProgress, ...] = ()
    failure_code: str | None = None
    failure_message: str | None = None
    limitations: tuple[str, ...] = ()


class DetachedCaptureManager:
    """Own server-lifetime capture tasks while persisting their bounded status."""

    def __init__(self, workspace: Workspace, captures: CaptureService) -> None:
        self.workspace = workspace
        self.captures = captures
        self.runs = RunStore(workspace)
        self.records = JsonRecordStore(
            workspace,
            kind="detached_captures",
            model=DetachedCaptureRecord,
            id_field="run_id",
            revision_field="revision",
        )
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._idempotency: dict[str, str] = {}
        self._start_lock = asyncio.Lock()
        self._record_lock = asyncio.Lock()

    async def start(self, plan_id: str, idempotency_key: str) -> DetachedCaptureStatus:
        idempotency_digest = digest_model(
            {
                "workspace_id": self.workspace.identity.workspace_id,
                "idempotency_key": idempotency_key,
            }
        )
        async with self._start_lock:
            existing_run_id = self._idempotency.get(idempotency_digest)
            if existing_run_id is not None:
                record = self.records.read(existing_run_id)
                if record.plan_digest != digest_model({"plan_id": plan_id}):
                    raise DomainError(
                        ErrorCode.INVALID_CAPTURE_PLAN,
                        "The detached idempotency key is already bound to another plan.",
                    )
                return self.status(existing_run_id)

            plan = await self.captures.plans.inspect(plan_id)
            plan_digest = digest_model({"plan_id": plan_id})
            record = DetachedCaptureRecord(
                run_id=plan.run_id,
                idempotency_digest=idempotency_digest,
                plan_digest=plan_digest,
            )
            self.records.create(record)
            self._idempotency[idempotency_digest] = plan.run_id
            task = asyncio.create_task(
                self._run(plan_id, plan.run_id),
                name=f"flameox-detached-{plan.run_id}",
            )
            self._tasks[plan.run_id] = task
        await self._wait_until_owned_or_terminal(plan.run_id)
        return self.status(plan.run_id)

    def status(self, run_id: str) -> DetachedCaptureStatus:
        record = self.records.read(run_id)
        try:
            run = self.runs.read(run_id)
        except DomainError:
            return DetachedCaptureStatus(
                run_id=run_id,
                state=record.state,
                progress=record.progress,
                failure_code=record.failure_code,
                failure_message=record.failure_message,
            )
        terminal = run.execution_status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.TIMED_OUT,
            ExecutionStatus.CANCELLED,
        }
        task = self._tasks.get(run_id)
        managed = task is not None
        state: Literal[
            "starting",
            "running",
            "terminal",
            "failed_to_start",
            "unmanaged_after_restart",
        ]
        limitations: tuple[str, ...] = ()
        if terminal and (task is None or task.done()):
            state = "terminal"
        elif terminal:
            state = "running"
        elif not managed and run.execution_status is ExecutionStatus.RUNNING:
            state = "unmanaged_after_restart"
            limitations = (
                "The server that owned this capture is unavailable; status is read-only. "
                "Run recovery after the exact process lease disappears.",
            )
        elif run.execution_status is ExecutionStatus.RUNNING:
            state = "running"
        else:
            state = record.state
        return DetachedCaptureStatus(
            run_id=run_id,
            state=state,
            execution_status=run.execution_status,
            capture_status=run.capture_status,
            artifact_count=len(run.artifacts),
            progress=record.progress,
            failure_code=record.failure_code,
            failure_message=record.failure_message,
            limitations=limitations,
        )

    async def cancel(self, run_id: str) -> DetachedCaptureStatus:
        status = self.status(run_id)
        if status.state == "terminal":
            return status
        task = self._tasks.get(run_id)
        if task is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "This server does not own the detached capture task.",
                remediation=(
                    "Inspect the exact process lease; recover only after it disappears.",
                ),
                run_id=run_id,
            )
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return self.status(run_id)

    async def shutdown(self) -> None:
        active = [task for task in self._tasks.values() if not task.done()]
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)

    async def _run(self, plan_id: str, run_id: str) -> None:
        try:
            await self.captures.execute(
                plan_id,
                progress=lambda completed, total, message: self._record_progress(
                    run_id,
                    completed,
                    total,
                    message,
                ),
            )
        except asyncio.CancelledError:
            await self._sync_state(run_id)
        except DomainError as error:
            await self._sync_state(
                run_id,
                failure_code=error.code.value,
                failure_message=error.message,
            )
        except Exception as error:
            await self._sync_state(
                run_id,
                failure_code=ErrorCode.INTERNAL_ERROR.value,
                failure_message=f"Detached capture failed: {type(error).__name__}.",
            )
        else:
            await self._sync_state(run_id)

    async def _record_progress(
        self,
        run_id: str,
        completed: float,
        total: float,
        message: str,
    ) -> None:
        async with self._record_lock:
            current = self.records.read(run_id)
            progress = (
                *current.progress[-15:],
                DetachedProgress(completed=completed, total=total, message=message),
            )
            self.records.append(
                current.model_copy(
                    update={
                        "revision": current.revision + 1,
                        "state": "running",
                        "progress": progress,
                        "updated_at": utc_now(),
                    }
                ),
                expected_revision=current.revision,
            )

    async def _sync_state(
        self,
        run_id: str,
        *,
        failure_code: str | None = None,
        failure_message: str | None = None,
    ) -> None:
        async with self._record_lock:
            current = self.records.read(run_id)
            try:
                run = self.runs.read(run_id)
                state = (
                    "terminal"
                    if run.execution_status
                    in {
                        ExecutionStatus.SUCCEEDED,
                        ExecutionStatus.FAILED,
                        ExecutionStatus.TIMED_OUT,
                        ExecutionStatus.CANCELLED,
                    }
                    else "running"
                )
            except DomainError:
                state = "failed_to_start"
            self.records.append(
                current.model_copy(
                    update={
                        "revision": current.revision + 1,
                        "state": state,
                        "failure_code": failure_code,
                        "failure_message": failure_message,
                        "updated_at": utc_now(),
                    }
                ),
                expected_revision=current.revision,
            )

    async def _wait_until_owned_or_terminal(self, run_id: str) -> None:
        for _ in range(1_200):
            task = self._tasks[run_id]
            if task.done():
                return
            try:
                run = self.runs.read(run_id)
            except DomainError:
                await asyncio.sleep(0.025)
                continue
            if run.lease is not None or run.execution_status in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.TIMED_OUT,
                ExecutionStatus.CANCELLED,
            }:
                return
            await asyncio.sleep(0.025)
