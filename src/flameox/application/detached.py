from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, computed_field, model_validator

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
    next_action: NextAction


def _replan_recovery(arguments: dict[str, object]) -> DetachedRecovery:
    return DetachedRecovery(
        next_action=next_action_for_action(
            ActionId.PLAN_CAPTURE,
            context=arguments,
            instruction="Supply the complete inputs required to plan this capture again.",
        )
    )


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
        # Version-one records are read-only migration input. New captures write only
        # the shared operation envelope and authoritative run manifest.
        self.records = ControlRecordStore(
            workspace,
            kind="detached_captures",
            model=DetachedCaptureRecord,
            id_field="run_id",
            revision_field="revision",
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
            return self._legacy_status(self.records.read(run_id))
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
            return DetachedCaptureStatus(
                run_id=run_id,
                state="starting" if managed else "failed_to_start",
                progress=progress,
                failure=self._failure(operation),
                limitations=(
                    (
                        "The server stopped before publishing a run manifest. Re-plan the "
                        "capture and start it with a new idempotency key.",
                    )
                    if not managed
                    else ()
                ),
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
            status = self.status(run_id)
            if status.state == "terminal":
                return status
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "This server does not own the legacy detached capture task.",
                run_id=run_id,
            )
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

    def _legacy_status(self, record: DetachedCaptureRecord) -> DetachedCaptureStatus:
        try:
            run = self.runs.read(record.run_id)
        except DomainError:
            return DetachedCaptureStatus(
                run_id=record.run_id,
                state="failed_to_start",
                progress=record.progress,
                failure=record.failure,
                limitations=(
                    "This version-one detached record has no run manifest and cannot resume.",
                ),
                recovery=(_replan_recovery(record.plan_request) if record.plan_request else None),
            )
        terminal = run.execution_status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.TIMED_OUT,
            ExecutionStatus.CANCELLED,
        }
        return DetachedCaptureStatus(
            run_id=record.run_id,
            state=(
                "terminal"
                if terminal
                else (
                    "unmanaged_after_restart"
                    if run.execution_status is ExecutionStatus.RUNNING
                    else "failed_to_start"
                )
            ),
            execution_status=run.execution_status,
            capture_status=run.capture_status,
            artifact_count=len(run.artifacts),
            progress=record.progress,
            failure=record.failure,
            limitations=("Legacy detached lifecycle state is read-only after migration.",),
            recovery=(
                _replan_recovery(record.plan_request)
                if not terminal
                and run.execution_status is not ExecutionStatus.RUNNING
                and record.plan_request
                else None
            ),
        )
