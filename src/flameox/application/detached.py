from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, computed_field, model_validator

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
from flameox.storage import ControlRecordStore, RunStore, Workspace


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


def _advertise_detached_failure_projections(schema: dict[str, Any]) -> None:
    properties = schema.setdefault("properties", {})
    assert isinstance(properties, dict)
    properties.pop("failure", None)
    properties.update(
        {
            "failure_code": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Failure Code",
            },
            "failure_message": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Failure Message",
            },
        }
    )
    required = schema.setdefault("required", [])
    assert isinstance(required, list)
    for field_name in ("failure", "failure_code", "failure_message"):
        if field_name in required:
            required.remove(field_name)


class _DetachedFailureFields(ContractModel):
    model_config = ConfigDict(json_schema_extra=_advertise_detached_failure_projections)

    failure: DetachedFailure | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def parse_failure_projections(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        has_code = "failure_code" in value
        has_message = "failure_message" in value
        if not has_code and not has_message:
            return value
        if "failure" in value:
            raise ValueError("use either failure or flattened detached-failure fields")
        if has_code != has_message:
            raise ValueError("detached capture failure code and message must appear together")
        parsed = dict(value)
        code = parsed.pop("failure_code")
        message = parsed.pop("failure_message")
        if (code is None) != (message is None):
            raise ValueError("detached capture failure code and message must appear together")
        parsed["failure"] = None if code is None else {"code": code, "message": message}
        return parsed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failure_code(self) -> str | None:
        return self.failure.code if self.failure is not None else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failure_message(self) -> str | None:
        return self.failure.message if self.failure is not None else None


class DetachedCaptureRecord(_DetachedFailureFields):
    schema_version: Literal[1] = 1
    run_id: str
    revision: Annotated[int, Field(ge=0)] = 0
    idempotency_digest: str
    plan_digest: str
    plan_request: dict[str, object] = Field(default_factory=dict)
    progress: Annotated[tuple[DetachedProgress, ...], Field(max_length=16)] = ()
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def with_progress(self, progress: DetachedProgress) -> Self:
        return self.__class__.model_validate(
            {
                **self.model_dump(mode="python"),
                "revision": self.revision + 1,
                "progress": (*self.progress[-15:], progress),
                "updated_at": utc_now(),
            }
        )

    def with_failure(self, *, code: str, message: str) -> Self:
        payload = self.model_dump(mode="python")
        payload.pop("failure_code", None)
        payload.pop("failure_message", None)
        return self.__class__.model_validate(
            {
                **payload,
                "revision": self.revision + 1,
                "failure": DetachedFailure(code=code, message=message),
                "updated_at": utc_now(),
            }
        )


class DetachedRecovery(ContractModel):
    action: Literal["replan"] = "replan"
    tool: Literal["plan_capture"] = "plan_capture"
    arguments: dict[str, object]


class DetachedCaptureStatus(_DetachedFailureFields):
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
    limitations: tuple[str, ...] = ()
    recovery: DetachedRecovery | None = None

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
    """Own server-lifetime capture tasks while persisting their bounded status."""

    def __init__(self, workspace: Workspace, captures: CaptureService) -> None:
        self.workspace = workspace
        self.captures = captures
        self.runs = RunStore(workspace)
        self.records = ControlRecordStore(
            workspace,
            kind="detached_captures",
            model=DetachedCaptureRecord,
            id_field="run_id",
            revision_field="revision",
        )
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._start_lock = asyncio.Lock()
        self._record_lock = asyncio.Lock()

    async def start(self, plan_token: str, idempotency_key: str) -> DetachedCaptureStatus:
        idempotency_digest = digest_model(
            {
                "workspace_id": self.workspace.identity.workspace_id,
                "idempotency_key": idempotency_key,
            }
        )
        plan_digest = digest_model({"plan_token": plan_token})
        async with self._start_lock:
            existing = self.records.read_idempotent(
                idempotency_digest=idempotency_digest,
            )
            if existing is not None:
                stored_digest, record = existing
                if stored_digest != plan_digest:
                    raise DomainError(
                        ErrorCode.INVALID_CAPTURE_PLAN,
                        "The detached idempotency key is already bound to another plan.",
                    )
                return self.status(record.run_id)
            plan = await self.captures.plans.inspect(plan_token)
            record = DetachedCaptureRecord(
                run_id=plan.run_id,
                idempotency_digest=idempotency_digest,
                plan_digest=plan_digest,
                plan_request={
                    "workload_name": plan.workload_name,
                    "adapter": plan.adapter,
                    "parameters": plan.workload_instance.parameters,
                    "preflight_mode": "auto",
                    "capture_mode": (
                        "managed" if plan.execution_policy == "approved_agent" else "trusted_local"
                    ),
                },
            )
            record, created = self.records.create_idempotent(
                record,
                idempotency_digest=idempotency_digest,
                intent_digest=plan_digest,
            )
            if not created:
                return self.status(record.run_id)
            task = asyncio.create_task(
                self._run(plan_token, plan.run_id),
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
            managed = run_id in self._tasks and not self._tasks[run_id].done()
            return DetachedCaptureStatus(
                run_id=run_id,
                state="starting" if managed else "failed_to_start",
                progress=record.progress,
                failure=record.failure,
                limitations=(
                    (
                        "The server stopped before publishing a run manifest. Re-plan the "
                        "capture and start it with a new idempotency key.",
                    )
                    if not managed
                    else ()
                ),
                recovery=(
                    DetachedRecovery(arguments=record.plan_request)
                    if not managed and record.plan_request
                    else None
                ),
            )
        terminal = run.execution_status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.TIMED_OUT,
            ExecutionStatus.CANCELLED,
        }
        task = self._tasks.get(run_id)
        managed = task is not None and not task.done()
        state: Literal[
            "starting",
            "running",
            "terminal",
            "failed_to_start",
            "unmanaged_after_restart",
        ]
        limitations: tuple[str, ...] = ()
        if terminal:
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
        return DetachedCaptureStatus(
            run_id=run_id,
            state=state,
            execution_status=run.execution_status,
            capture_status=run.capture_status,
            artifact_count=len(run.artifacts),
            progress=record.progress,
            failure=record.failure,
            limitations=limitations,
            recovery=(
                DetachedRecovery(arguments=record.plan_request)
                if state == "failed_to_start" and record.plan_request
                else None
            ),
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
                remediation=("Inspect the exact process lease; recover only after it disappears.",),
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

    async def _run(self, plan_token: str, run_id: str) -> None:
        try:
            await self.captures.execute(
                plan_token,
                progress=lambda completed, total, message: self._record_progress(
                    run_id,
                    completed,
                    total,
                    message,
                ),
            )
        except asyncio.CancelledError:
            pass
        except DomainError as error:
            await self._record_failure(
                run_id,
                failure_code=error.code.value,
                failure_message=error.message,
            )
        except Exception as error:
            await self._record_failure(
                run_id,
                failure_code=ErrorCode.INTERNAL_ERROR.value,
                failure_message=f"Detached capture failed: {type(error).__name__}.",
            )

    async def _record_progress(
        self,
        run_id: str,
        completed: float,
        total: float,
        message: str,
    ) -> None:
        async with self._record_lock:
            current = self.records.read(run_id)
            updated = current.with_progress(
                DetachedProgress(completed=completed, total=total, message=message)
            )
            self.records.append(
                updated,
                expected_revision=current.revision,
            )

    async def _record_failure(
        self,
        run_id: str,
        *,
        failure_code: str,
        failure_message: str,
    ) -> None:
        async with self._record_lock:
            current = self.records.read(run_id)
            updated = current.with_failure(code=failure_code, message=failure_message)
            self.records.append(
                updated,
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
