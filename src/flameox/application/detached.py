from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from flameox.action_graph import ActionId, NextAction, next_action_for_action
from flameox.application.capture import CaptureService
from flameox.application.operations import (
    OperationAdapter,
    OperationRecord,
    OperationRunner,
    OperationStatus,
    operation_digests,
)
from flameox.application.task_supervisor import TaskSupervisor
from flameox.domain import (
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    digest_model,
)
from flameox.domain.models import utc_now
from flameox.models import ContractModel
from flameox.storage import RunStore, Workspace

_CANCELLATION_PENDING_LIMITATION = (
    "Cancellation was requested; owned cleanup is still in progress. Poll this run for its "
    "terminal status."
)


class DetachedProgress(ContractModel):
    completed: Annotated[float, Field(ge=0)]
    total: Annotated[float, Field(gt=0)]
    message: Annotated[str, Field(min_length=1, max_length=500)]
    observed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def completed_does_not_exceed_total(self) -> Self:
        if self.completed > self.total:
            raise ValueError("detached progress cannot exceed its total")
        return self


class DetachedFailure(ContractModel):
    code: str
    message: str


def _replan_recovery(arguments: dict[str, object]) -> NextAction:
    return next_action_for_action(
        ActionId.PLAN_CAPTURE,
        context=arguments,
        instruction="Supply the complete inputs required to plan this capture again.",
    )


class DetachedCaptureStatus(ContractModel):
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
    failure: DetachedFailure | None = None
    limitations: tuple[str, ...] = ()
    recovery: NextAction | None = None

    @model_validator(mode="after")
    def projected_state_is_coherent(self) -> Self:
        if (self.execution_status is None) != (self.capture_status is None):
            raise ValueError("detached execution and capture status must appear together")
        if self.recovery is not None and self.state != "failed_to_start":
            raise ValueError("only a failed-to-start capture can carry recovery")
        if self.state == "unmanaged_after_restart" and (
            self.execution_status is not ExecutionStatus.RUNNING
            or self.capture_status is not CaptureStatus.RUNNING
        ):
            raise ValueError("an unmanaged capture must still be running")
        if self.state == "terminal":
            if self.execution_status not in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.TIMED_OUT,
                ExecutionStatus.CANCELLED,
            }:
                raise ValueError("a terminal detached capture requires terminal execution")
            if self.capture_status in {CaptureStatus.PENDING, CaptureStatus.RUNNING}:
                raise ValueError("a terminal detached capture requires terminal capture state")
        return self


class DetachedCaptureManager:
    """Project capture domain state over the shared durable operation kernel."""

    _ADAPTER = OperationAdapter(
        kind="capture.detached",
        start_action=ActionId.START_DETACHED_CAPTURE,
        status_action=ActionId.GET_DETACHED_CAPTURE,
        status_identifier="run_id",
    )

    def __init__(
        self,
        workspace: Workspace,
        captures: CaptureService,
        *,
        supervisor: TaskSupervisor | None = None,
    ) -> None:
        self.workspace = workspace
        self.captures = captures
        self.runs = RunStore(workspace)
        self.runner = OperationRunner(
            workspace,
            self._ADAPTER,
            supervisor=supervisor,
        )

    async def start(self, plan_token: str, idempotency_key: str) -> DetachedCaptureStatus:
        plan_digest = digest_model({"plan_token": plan_token})
        _, idempotency_digest = operation_digests(
            self.workspace,
            self._ADAPTER.kind,
            {},
            idempotency_key,
        )
        existing = self.runner.store.find(
            operation=self._ADAPTER.kind,
            idempotency_digest=idempotency_digest,
        )
        if existing is not None:
            if existing.request.get("plan_digest") != plan_digest:
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    "The detached idempotency key is already bound to another plan.",
                )
            if existing.subject_id is None:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "The detached operation has no run identity.",
                )
            return self.status(existing.subject_id)

        plan = await self.captures.plans.inspect(plan_token)
        plan_request: dict[str, object] = {
            "workload_name": plan.workload_name,
            "adapter": plan.adapter,
            "parameters": plan.workload_instance.parameters,
            "preflight_mode": "auto",
            "capture_mode": (
                "managed" if plan.execution_policy == "approved_agent" else "trusted_local"
            ),
        }
        request: dict[str, Any] = {
            "run_id": plan.run_id,
            "plan_digest": plan_digest,
            "plan_request": plan_request,
        }
        ready = asyncio.Event()

        async def execute(
            operation_id: str,
            progress: Callable[..., Awaitable[None]],
        ) -> dict[str, Any]:
            return await self._run(
                plan_token,
                operation_id,
                ready=ready,
                progress=progress,
            )

        try:
            operation = await self.runner.start(
                request,
                idempotency_key,
                execute,
                subject_id=plan.run_id,
            )
        except DomainError as error:
            if error.code is ErrorCode.REVISION_CONFLICT:
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    "The detached idempotency key or run is already bound to another plan.",
                ) from error
            raise
        if operation.operation_id in self.runner.tasks:
            await ready.wait()
        return self.status(plan.run_id)

    def status(self, run_id: str) -> DetachedCaptureStatus:
        record = self.runner.store.find_subject(
            operation=self._ADAPTER.kind,
            subject_id=run_id,
        )
        if record is None:
            raise DomainError(
                ErrorCode.RUN_NOT_FOUND,
                "No detached capture operation exists for this run.",
                run_id=run_id,
            )
        return self._project(record)

    def _project(self, record: OperationRecord) -> DetachedCaptureStatus:
        run_id = record.subject_id
        if run_id is None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "The detached operation has no run identity.",
            )
        operation = OperationStatus.from_record(record, adapter=self._ADAPTER)
        progress = tuple(
            DetachedProgress(
                completed=item.completed,
                total=item.total,
                message=item.message,
                observed_at=item.observed_at,
            )
            for item in operation.progress
            if item.completed is not None and item.total is not None
        )
        try:
            run = self.runs.read(run_id)
        except DomainError:
            managed = operation.operation_id in self.runner.tasks
            startup_limitations = (
                (_CANCELLATION_PENDING_LIMITATION,)
                if managed and operation.cancellation_requested
                else (
                    (
                        "The server stopped before publishing a run manifest. Re-plan the "
                        "capture and start it with a new idempotency key."
                    ),
                )
                if not managed
                else ()
            )
            return DetachedCaptureStatus(
                run_id=run_id,
                state="starting" if managed else "failed_to_start",
                progress=progress,
                failure=self._failure(operation),
                limitations=startup_limitations,
                recovery=(
                    _replan_recovery(self._plan_request(record))
                    if not managed and self._plan_request(record)
                    else None
                ),
            )
        terminal = run.execution_status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.TIMED_OUT,
            ExecutionStatus.CANCELLED,
        }
        managed = operation.operation_id in self.runner.tasks
        state: Literal[
            "starting",
            "running",
            "terminal",
            "failed_to_start",
            "unmanaged_after_restart",
        ]
        limitations: tuple[str, ...] = ()
        if terminal:
            # The run manifest is domain-authoritative, but the operation still owns
            # preservation/publication. Do not advertise terminal completion until
            # that task has committed its receipt and evicted local ownership.
            state = "running" if managed else "terminal"
        elif run.execution_status is ExecutionStatus.RUNNING:
            if managed:
                state = "running"
            else:
                state = "unmanaged_after_restart"
                limitations = (
                    "The server that owned this capture is unavailable; status is read-only. "
                    "Run recovery after the exact process lease disappears.",
                )
        elif managed:
            state = "starting"
        else:
            state = "failed_to_start"
        if managed and operation.cancellation_requested:
            limitations = (_CANCELLATION_PENDING_LIMITATION,)
        return DetachedCaptureStatus(
            run_id=run_id,
            state=state,
            execution_status=run.execution_status,
            capture_status=run.capture_status,
            artifact_count=len(run.artifacts),
            progress=progress,
            failure=self._failure(operation),
            limitations=limitations,
            recovery=(
                _replan_recovery(self._plan_request(record))
                if state == "failed_to_start" and self._plan_request(record)
                else None
            ),
        )

    async def cancel(self, run_id: str) -> DetachedCaptureStatus:
        record = self.runner.store.find_subject(
            operation=self._ADAPTER.kind,
            subject_id=run_id,
        )
        if record is None:
            return self.status(run_id)
        await self.runner.cancel(record.operation_id)
        return self.status(run_id)

    async def shutdown(self) -> None:
        await self.runner.shutdown()

    async def _run(
        self,
        plan_token: str,
        operation_id: str,
        *,
        ready: asyncio.Event,
        progress: Callable[..., Awaitable[None]],
    ) -> dict[str, Any]:
        task = self.runner.tasks[operation_id]
        self.runner.set_cancel_hook(operation_id, task.cancel)

        async def report(completed: float, total: float, message: str) -> None:
            if completed >= 4:
                ready.set()
            await progress("capturing", completed, total, message)

        try:
            result = await self.captures.execute(
                plan_token,
                progress=report,
            )
            return {
                "run_id": result.run.run_id,
                "execution_status": result.run.execution_status.value,
                "capture_status": result.run.capture_status.value,
                "artifact_count": len(result.run.artifacts),
            }
        finally:
            ready.set()
            self.runner.clear_cancel_hook(operation_id)

    @staticmethod
    def _failure(operation: OperationStatus) -> DetachedFailure | None:
        if operation.failure_code is None or operation.failure_message is None:
            return None
        return DetachedFailure(
            code=operation.failure_code,
            message=operation.failure_message,
        )

    @staticmethod
    def _plan_request(record: OperationRecord) -> dict[str, object]:
        value = record.request.get("plan_request")
        return value if isinstance(value, dict) else {}
