from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from uuid import uuid4

from pydantic import Field, TypeAdapter, computed_field, model_validator

from flameox.domain import DomainError, ErrorCode, digest_model
from flameox.domain.models import Digest, utc_now
from flameox.models import ContractModel
from flameox.storage import JsonRecordStore, Workspace


class OperationState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    TERMINAL = "terminal"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNMANAGED_AFTER_RESTART = "unmanaged_after_restart"


class OperationItemStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    RETRYABLE = "retryable"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class OperationCleanupStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class OperationRecoveryAction(StrEnum):
    POLL = "poll"
    RETRY_SAME_REQUEST = "retry_same_request"
    RETRY_NEW_OPERATION = "retry_new_operation"
    RETRY_FAILED_ITEMS = "retry_failed_items"
    INSPECT_CAPABILITIES = "inspect_capabilities"
    EXTRACT_REQUIRED_EVIDENCE = "extract_required_evidence"
    REREAD_SNAPSHOT = "reread_snapshot"


class OperationProgress(ContractModel):
    phase: str = Field(min_length=1, max_length=100)
    completed: float | None = Field(default=None, ge=0)
    total: float | None = Field(default=None, gt=0)
    message: str = Field(min_length=1, max_length=500)
    observed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def bounded_progress_is_coherent(self) -> OperationProgress:
        if (self.completed is None) != (self.total is None):
            raise ValueError("operation progress completed and total must appear together")
        if self.completed is not None and self.total is not None and self.completed > self.total:
            raise ValueError("operation progress cannot exceed its total")
        return self


class OperationItemOutcome(ContractModel):
    item: str = Field(min_length=1, max_length=200)
    status: OperationItemStatus
    message: str | None = Field(default=None, max_length=500)

    def with_status(self, status: OperationItemStatus) -> Self:
        return self.__class__.model_validate({**self.model_dump(mode="python"), "status": status})


class OperationRecovery(ContractModel):
    action: OperationRecoveryAction
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class OperationFailure(Exception):
    """A domain failure with item-level completion known to the operation."""

    def __init__(self, error: DomainError, *, completed_items: tuple[str, ...] = ()) -> None:
        super().__init__(error.message)
        self.error = error
        self.completed_items = completed_items


class _OperationRecord(ContractModel):
    schema_version: Literal[2] = 2
    operation: str
    workspace_id: str
    request: dict[str, Any]
    idempotency_digest: Digest
    phase: str = "starting"
    revision: int = Field(default=0, ge=0)
    progress: tuple[OperationProgress, ...] = Field(default=(), max_length=32)
    item_outcomes: tuple[OperationItemOutcome, ...] = Field(default=(), max_length=64)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def parse_identity_projections(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        supplied_request_digest = payload.pop("request_digest", None)
        if (
            supplied_request_digest is not None
            and "request" in payload
            and supplied_request_digest != digest_model(payload["request"])
        ):
            raise ValueError("operation request digest does not match its request")
        supplied_operation_id = payload.pop("operation_id", None)
        idempotency_digest = payload.get("idempotency_digest")
        if supplied_operation_id is not None and isinstance(idempotency_digest, str):
            expected_operation_id = f"op-{idempotency_digest.removeprefix('sha256:')}"
            if supplied_operation_id != expected_operation_id:
                raise ValueError("operation identifier does not match its idempotency digest")
        return payload

    @computed_field  # type: ignore[prop-decorator]
    @property
    def request_digest(self) -> Digest:
        return digest_model(self.request)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def operation_id(self) -> str:
        return f"op-{self.idempotency_digest.removeprefix('sha256:')}"

    def _next_revision(self) -> dict[str, Any]:
        payload = {name: getattr(self, name) for name in type(self).model_fields}
        payload["revision"] = self.revision + 1
        payload["updated_at"] = utc_now()
        return payload


def _parse_active_cancellation_projection(value: Any) -> Any:
    if not isinstance(value, Mapping) or "cancellation_requested" not in value:
        return value
    payload = dict(value)
    supplied = payload.pop("cancellation_requested")
    cleanup_status = payload.get(
        "cleanup_status",
        OperationCleanupStatus.NOT_REQUIRED,
    )
    if cleanup_status in {
        OperationCleanupStatus.NOT_REQUIRED,
        OperationCleanupStatus.PENDING,
    } and supplied != (cleanup_status == OperationCleanupStatus.PENDING):
        raise ValueError("operation cancellation must agree with pending cleanup")
    return payload


class ActiveOperationRecord(_OperationRecord):
    """A locally owned operation that can still transition through runner callbacks."""

    state: Literal[OperationState.STARTING, OperationState.RUNNING] = OperationState.STARTING
    failure_code: Literal[None] = None
    failure_message: Literal[None] = None
    failure_details: Literal[None] = None
    cleanup_status: Literal[
        OperationCleanupStatus.NOT_REQUIRED,
        OperationCleanupStatus.PENDING,
    ] = OperationCleanupStatus.NOT_REQUIRED
    terminal_receipt: Literal[None] = None
    recovery: Literal[None] = None
    owner_id: str
    owner_heartbeat_at: datetime

    @model_validator(mode="before")
    @classmethod
    def parse_cancellation_projection(cls, value: Any) -> Any:
        return _parse_active_cancellation_projection(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cancellation_requested(self) -> bool:
        return self.cleanup_status is OperationCleanupStatus.PENDING

    def running(self) -> Self:
        payload = self._next_revision()
        payload.update(
            {
                "state": OperationState.RUNNING,
                "phase": "cancelling" if self.cancellation_requested else "starting",
                "owner_heartbeat_at": payload["updated_at"],
            }
        )
        return self.__class__.model_validate(payload)

    def request_cancellation(self) -> Self:
        payload = self._next_revision()
        payload.update(
            {
                "phase": "cancelling",
                "cleanup_status": OperationCleanupStatus.PENDING,
                "owner_heartbeat_at": payload["updated_at"],
            }
        )
        return self.__class__.model_validate(payload)

    def heartbeat(self) -> Self:
        payload = self._next_revision()
        payload["owner_heartbeat_at"] = payload["updated_at"]
        return self.__class__.model_validate(payload)

    def report_progress(self, event: OperationProgress, *, owner_id: str) -> Self:
        payload = self._next_revision()
        payload.update(
            {
                "phase": event.phase,
                "progress": (*self.progress[-31:], event),
                "owner_heartbeat_at": (
                    payload["updated_at"] if self.owner_id == owner_id else self.owner_heartbeat_at
                ),
            }
        )
        return self.__class__.model_validate(payload)

    def completed(
        self,
        *,
        receipt: dict[str, Any],
        item_outcomes: tuple[OperationItemOutcome, ...],
    ) -> CompletedOperationRecord:
        return CompletedOperationRecord.model_validate(
            {
                **self._next_revision(),
                "state": OperationState.TERMINAL,
                "phase": "completed",
                "cancellation_requested": False,
                "cleanup_status": OperationCleanupStatus.COMPLETE,
                "terminal_receipt": receipt,
                "item_outcomes": item_outcomes,
                "owner_id": None,
                "owner_heartbeat_at": None,
            }
        )

    def cancelled(
        self,
        *,
        recovery: OperationRecovery,
        item_outcomes: tuple[OperationItemOutcome, ...],
        receipt: dict[str, Any] | None = None,
        failure_code: str | None = None,
        failure_message: str | None = None,
        failure_details: dict[str, Any] | None = None,
    ) -> CancelledOperationRecord:
        return CancelledOperationRecord.model_validate(
            {
                **self._next_revision(),
                "state": OperationState.CANCELLED,
                "phase": "cancelled",
                "cancellation_requested": self.cancellation_requested,
                "cleanup_status": OperationCleanupStatus.COMPLETE,
                "terminal_receipt": receipt,
                "item_outcomes": item_outcomes,
                "failure_code": failure_code,
                "failure_message": failure_message,
                "failure_details": failure_details,
                "recovery": recovery,
                "owner_id": None,
                "owner_heartbeat_at": None,
            }
        )

    def failed(
        self,
        *,
        phase: str,
        failure_code: str,
        failure_message: str,
        failure_details: dict[str, Any] | None,
        item_outcomes: tuple[OperationItemOutcome, ...],
        recovery: OperationRecovery,
    ) -> FailedOperationRecord:
        return FailedOperationRecord.model_validate(
            {
                **self._next_revision(),
                "state": OperationState.FAILED,
                "phase": phase,
                "cancellation_requested": self.cancellation_requested,
                "cleanup_status": OperationCleanupStatus.COMPLETE,
                "failure_code": failure_code,
                "failure_message": failure_message,
                "failure_details": failure_details,
                "item_outcomes": item_outcomes,
                "recovery": recovery,
                "owner_id": None,
                "owner_heartbeat_at": None,
            }
        )

    def unmanaged(self, *, recovery: OperationRecovery) -> UnmanagedOperationRecord:
        return UnmanagedOperationRecord.model_validate(
            {
                **self._next_revision(),
                "state": OperationState.UNMANAGED_AFTER_RESTART,
                "recovery": recovery,
                "owner_id": None,
                "owner_heartbeat_at": None,
            }
        )


class CompletedOperationRecord(_OperationRecord):
    state: Literal[OperationState.TERMINAL] = OperationState.TERMINAL
    phase: Literal["completed"] = "completed"
    failure_code: Literal[None] = None
    failure_message: Literal[None] = None
    failure_details: Literal[None] = None
    cancellation_requested: Literal[False] = False
    cleanup_status: Literal[OperationCleanupStatus.COMPLETE] = OperationCleanupStatus.COMPLETE
    terminal_receipt: dict[str, Any]
    recovery: Literal[None] = None
    owner_id: Literal[None] = None
    owner_heartbeat_at: Literal[None] = None


class FailedOperationRecord(_OperationRecord):
    state: Literal[OperationState.FAILED] = OperationState.FAILED
    failure_code: str
    failure_message: str
    failure_details: dict[str, Any] | None = None
    cancellation_requested: bool
    cleanup_status: Literal[OperationCleanupStatus.COMPLETE] = OperationCleanupStatus.COMPLETE
    terminal_receipt: Literal[None] = None
    recovery: OperationRecovery
    owner_id: Literal[None] = None
    owner_heartbeat_at: Literal[None] = None


class CancelledOperationRecord(_OperationRecord):
    state: Literal[OperationState.CANCELLED] = OperationState.CANCELLED
    phase: Literal["cancelled"] = "cancelled"
    failure_code: str | None = None
    failure_message: str | None = None
    failure_details: dict[str, Any] | None = None
    cancellation_requested: bool
    cleanup_status: Literal[OperationCleanupStatus.COMPLETE] = OperationCleanupStatus.COMPLETE
    terminal_receipt: dict[str, Any] | None = None
    recovery: OperationRecovery
    owner_id: Literal[None] = None
    owner_heartbeat_at: Literal[None] = None

    @model_validator(mode="after")
    def failure_fields_are_atomic(self) -> Self:
        if (self.failure_code is None) != (self.failure_message is None):
            raise ValueError("cancelled operation failure code and message must appear together")
        if self.failure_code is None and self.failure_details is not None:
            raise ValueError("cancelled operation failure details require a failure")
        return self


class UnmanagedOperationRecord(_OperationRecord):
    state: Literal[OperationState.UNMANAGED_AFTER_RESTART] = OperationState.UNMANAGED_AFTER_RESTART
    failure_code: Literal[None] = None
    failure_message: Literal[None] = None
    failure_details: Literal[None] = None
    cleanup_status: Literal[
        OperationCleanupStatus.NOT_REQUIRED,
        OperationCleanupStatus.PENDING,
    ]
    terminal_receipt: Literal[None] = None
    recovery: OperationRecovery
    owner_id: Literal[None] = None
    owner_heartbeat_at: Literal[None] = None

    @model_validator(mode="before")
    @classmethod
    def parse_cancellation_projection(cls, value: Any) -> Any:
        return _parse_active_cancellation_projection(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cancellation_requested(self) -> bool:
        return self.cleanup_status is OperationCleanupStatus.PENDING


type OperationRecord = Annotated[
    ActiveOperationRecord
    | CompletedOperationRecord
    | FailedOperationRecord
    | CancelledOperationRecord
    | UnmanagedOperationRecord,
    Field(discriminator="state"),
]

_OPERATION_RECORD: TypeAdapter[OperationRecord] = TypeAdapter(OperationRecord)


class OperationStatus(ContractModel):
    schema_version: Literal[1] = 1
    operation_id: str
    operation: str
    workspace_id: str
    request_digest: str
    request: dict[str, Any]
    idempotency_digest: str
    revision: int
    state: OperationState
    phase: str
    progress: tuple[OperationProgress, ...]
    item_outcomes: tuple[OperationItemOutcome, ...]
    cancellation_requested: bool
    cleanup_status: OperationCleanupStatus
    failure_code: str | None
    failure_message: str | None
    failure_details: dict[str, Any] | None
    terminal_receipt: dict[str, Any] | None
    recovery: OperationRecovery | None
    poll_after_ms: int | None = Field(default=None, ge=100, le=30_000)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record: OperationRecord) -> OperationStatus:
        payload = record.model_dump(exclude={"owner_id", "owner_heartbeat_at"})
        payload["schema_version"] = 1
        if record.operation == "capability.setup" and isinstance(record, ActiveOperationRecord):
            payload["poll_after_ms"] = _capability_setup_poll_after_ms(record.phase)
            payload["recovery"] = OperationRecovery(
                action=OperationRecoveryAction.POLL,
                tool="get_capability_setup",
                arguments={"operation_id": record.operation_id},
            ).model_dump()
        else:
            payload["poll_after_ms"] = None
        return cls.model_validate(payload)


def _capability_setup_poll_after_ms(phase: str) -> int:
    if phase == "validating_request":
        return 250
    if phase == "verifying":
        return 500
    return 1_000


class OperationStore:
    def __init__(self, workspace: Workspace) -> None:
        self.records: JsonRecordStore[OperationRecord] = JsonRecordStore(
            workspace,
            kind="operations",
            model=_OPERATION_RECORD,
            id_field="operation_id",
            revision_field="revision",
        )

    def read(self, operation_id: str) -> OperationRecord:
        return self.records.read(operation_id)

    def find(self, *, operation: str, idempotency_digest: str) -> OperationRecord | None:
        return next(
            (
                item
                for item in self.records.list()
                if item.operation == operation and item.idempotency_digest == idempotency_digest
            ),
            None,
        )


def operation_digests(
    workspace: Workspace,
    operation: str,
    request: dict[str, Any],
    idempotency_key: str,
) -> tuple[str, str]:
    request_digest = digest_model(request)
    idempotency_digest = digest_model(
        {
            "workspace_id": workspace.identity.workspace_id,
            "operation": operation,
            "idempotency_key": idempotency_key,
        }
    )
    return request_digest, idempotency_digest


class OperationRunner:
    """Small durable runner used by MCP operations with external side effects."""

    _LEASE_TIMEOUT = timedelta(seconds=30)
    _LEASE_HEARTBEAT_INTERVAL = 5.0

    def __init__(self, workspace: Workspace, operation: str) -> None:
        self.workspace = workspace
        self.operation = operation
        self.store = OperationStore(workspace)
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.cancel_events: dict[str, asyncio.Event] = {}
        self.cancel_hooks: dict[str, Callable[[], None]] = {}
        self.lease_tasks: dict[str, asyncio.Task[None]] = {}
        self.owner_id = uuid4().hex
        self.record_lock = asyncio.Lock()
        self.lock = asyncio.Lock()

    async def start(
        self,
        request: dict[str, Any],
        idempotency_key: str,
        run: Callable[[str, Callable[..., Awaitable[None]]], Awaitable[dict[str, Any]]],
        *,
        items: tuple[str, ...] = (),
    ) -> OperationStatus:
        request_digest, idempotency_digest = operation_digests(
            self.workspace,
            self.operation,
            request,
            idempotency_key,
        )
        async with self.lock:
            existing = self.store.find(
                operation=self.operation,
                idempotency_digest=idempotency_digest,
            )
            if existing is not None:
                if existing.request_digest != request_digest:
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        "The idempotency key is already bound to a different request.",
                        details={"operation_id": existing.operation_id},
                    )
                if (
                    isinstance(existing, ActiveOperationRecord)
                    and existing.operation_id not in self.tasks
                    and not self._lease_is_active(existing)
                ):
                    existing = self._mark_unmanaged(existing)
                    self.store.records.append(existing, expected_revision=existing.revision - 1)
                return OperationStatus.from_record(existing)

            # The digest-derived identity makes the create itself the cross-process
            # idempotency gate. A process-local asyncio lock cannot protect two MCP
            # server instances sharing one workspace.
            record = ActiveOperationRecord(
                operation=self.operation,
                workspace_id=self.workspace.identity.workspace_id,
                request=request,
                idempotency_digest=idempotency_digest,
                owner_id=self.owner_id,
                owner_heartbeat_at=utc_now(),
                item_outcomes=tuple(
                    OperationItemOutcome(item=item, status=OperationItemStatus.PENDING)
                    for item in items
                ),
            )
            operation_id = record.operation_id
            try:
                self.store.records.create(record)
            except DomainError as error:
                if error.code is not ErrorCode.REVISION_CONFLICT:
                    raise
                existing = self.store.read(operation_id)
                if existing.request_digest != request_digest:
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        "The idempotency key is already bound to a different request.",
                        details={"operation_id": existing.operation_id},
                    ) from error
                return OperationStatus.from_record(existing)
            cancel_event = asyncio.Event()
            self.cancel_events[operation_id] = cancel_event
            self.tasks[operation_id] = asyncio.create_task(
                self._execute(operation_id, run, cancel_event),
                name=f"flameox-operation-{operation_id}",
            )
            self.lease_tasks[operation_id] = asyncio.create_task(
                self._heartbeat(operation_id),
                name=f"flameox-operation-lease-{operation_id}",
            )
        await asyncio.sleep(0)
        return OperationStatus.from_record(self.store.read(operation_id))

    async def status(self, operation_id: str) -> OperationStatus:
        record = self.store.read(operation_id)
        if (
            isinstance(record, ActiveOperationRecord)
            and operation_id not in self.tasks
            and not self._lease_is_active(record)
        ):
            record = self._mark_unmanaged(record)
            self.store.records.append(record, expected_revision=record.revision - 1)
        return OperationStatus.from_record(record)

    def set_cancel_hook(self, operation_id: str, hook: Callable[[], None]) -> None:
        self.cancel_hooks[operation_id] = hook
        event = self.cancel_events.get(operation_id)
        if event is not None and event.is_set():
            hook()

    def clear_cancel_hook(self, operation_id: str) -> None:
        self.cancel_hooks.pop(operation_id, None)

    async def cancel(self, operation_id: str) -> OperationStatus:
        record = self.store.read(operation_id)
        if isinstance(
            record,
            (CompletedOperationRecord, FailedOperationRecord, CancelledOperationRecord),
        ):
            return OperationStatus.from_record(record)
        event = self.cancel_events.get(operation_id)
        if event is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "The server no longer owns this operation.",
                details={"operation_id": operation_id},
                remediation=(
                    "Poll the operation status; retry the exact request only if it is unmanaged.",
                ),
            )
        event.set()
        hook = self.cancel_hooks.get(operation_id)
        if hook is not None:
            hook()
        await self._append_active_transition(
            operation_id,
            lambda current: current.request_cancellation(),
        )
        task = self.tasks.get(operation_id)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        return OperationStatus.from_record(self.store.read(operation_id))

    async def shutdown(self) -> None:
        for event in self.cancel_events.values():
            event.set()
        for hook in self.cancel_hooks.values():
            hook()
        tasks = [task for task in self.tasks.values() if not task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute(
        self,
        operation_id: str,
        run: Callable[[str, Callable[..., Awaitable[None]]], Awaitable[dict[str, Any]]],
        cancel_event: asyncio.Event,
    ) -> None:
        try:
            started = await self._append_active_transition(
                operation_id,
                lambda current: current.running(),
            )
            if started is None:
                return

            async def progress(
                phase: str,
                completed: float | None = None,
                total: float | None = None,
                message: str = "Operation in progress.",
            ) -> None:
                await self._progress(operation_id, phase, completed, total, message)

            receipt = await run(operation_id, progress)
            await self._finish_success(operation_id, cancel_event, receipt)
        except asyncio.CancelledError:
            await self._finish_cancelled(operation_id)
        except OperationFailure as failure:
            await self._finish_domain_failure(
                operation_id,
                cancel_event,
                failure.error,
                completed_items=failure.completed_items,
            )
        except DomainError as error:
            await self._finish_domain_failure(operation_id, cancel_event, error)
        except Exception as error:
            await self._finish_unexpected_failure(operation_id, error)
        finally:
            lease_task = self.lease_tasks.pop(operation_id, None)
            if lease_task is not None:
                lease_task.cancel()
                await asyncio.gather(lease_task, return_exceptions=True)

    async def _finish_success(
        self,
        operation_id: str,
        cancel_event: asyncio.Event,
        receipt: dict[str, Any],
    ) -> None:
        def transition(current: ActiveOperationRecord) -> OperationRecord:
            if cancel_event.is_set() or current.cancellation_requested:
                return current.cancelled(
                    recovery=self._retry_recovery(current),
                    item_outcomes=self._items(current, OperationItemStatus.PENDING),
                    receipt=receipt,
                )
            return current.completed(
                receipt=receipt,
                item_outcomes=self._items(current, OperationItemStatus.COMPLETE),
            )

        await self._append_active_transition(operation_id, transition)

    async def _finish_cancelled(self, operation_id: str) -> None:
        await self._append_active_transition(
            operation_id,
            lambda current: current.cancelled(
                recovery=self._retry_recovery(current),
                item_outcomes=self._items(current, OperationItemStatus.PENDING),
            ),
        )

    async def _finish_domain_failure(
        self,
        operation_id: str,
        cancel_event: asyncio.Event,
        error: DomainError,
        *,
        completed_items: tuple[str, ...] = (),
    ) -> None:
        failure_details = self._failure_details(error)

        def transition(current: ActiveOperationRecord) -> OperationRecord:
            if (
                cancel_event.is_set()
                or current.cancellation_requested
                or error.code is ErrorCode.PROCESS_CANCELLED
            ):
                return current.cancelled(
                    failure_code=error.code.value,
                    failure_message=error.message,
                    failure_details=failure_details,
                    item_outcomes=self._items(
                        current,
                        OperationItemStatus.PENDING,
                        completed_items,
                    ),
                    recovery=self._retry_recovery(current),
                )
            failure_phase = failure_details.get("phase")
            return current.failed(
                phase=(failure_phase if isinstance(failure_phase, str) else "failed"),
                failure_code=error.code.value,
                failure_message=error.message,
                failure_details=failure_details,
                item_outcomes=self._items(
                    current,
                    (
                        OperationItemStatus.RETRYABLE
                        if error.retryable
                        else OperationItemStatus.FAILED
                    ),
                    completed_items,
                ),
                recovery=(
                    self._retry_recovery(current)
                    if error.retryable
                    else OperationRecovery(
                        action=OperationRecoveryAction.INSPECT_CAPABILITIES,
                        tool=f"start_{self.operation.replace('.', '_')}",
                        arguments=current.request,
                    )
                ),
            )

        await self._append_active_transition(operation_id, transition)

    @staticmethod
    def _failure_details(error: DomainError) -> dict[str, Any]:
        """Persist only bounded, recovery-relevant diagnostics from a domain failure."""
        details: dict[str, Any] = {}
        for key in ("phase", "failure_category", "adapter", "next_tool"):
            value = error.details.get(key)
            if isinstance(value, str) and value:
                details[key] = value[:200]
        detail = error.details.get("failure_detail") or error.details.get("error")
        if detail is not None:
            normalized = " ".join(str(detail).split())[:500]
            if normalized:
                details["failure_detail"] = normalized
        return details

    async def _finish_unexpected_failure(self, operation_id: str, error: Exception) -> None:
        await self._append_active_transition(
            operation_id,
            lambda current: current.failed(
                phase="failed",
                failure_code=ErrorCode.INTERNAL_ERROR.value,
                failure_message=f"Operation failed with {type(error).__name__}.",
                failure_details=None,
                item_outcomes=self._items(current, OperationItemStatus.FAILED),
                recovery=self._retry_recovery(current),
            ),
        )

    async def _heartbeat(self, operation_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(self._LEASE_HEARTBEAT_INTERVAL)
                current = self.store.read(operation_id)
                if not isinstance(current, ActiveOperationRecord) or (
                    current.owner_id != self.owner_id
                ):
                    return
                updated = await self._append_active_transition(
                    operation_id,
                    lambda active: active.heartbeat() if active.owner_id == self.owner_id else None,
                )
                if updated is None:
                    return
        except asyncio.CancelledError:
            return

    async def _progress(
        self,
        operation_id: str,
        phase: str,
        completed: float | None,
        total: float | None,
        message: str,
    ) -> None:
        event = OperationProgress(
            phase=phase,
            completed=completed,
            total=total,
            message=message,
        )

        def transition(current: ActiveOperationRecord) -> ActiveOperationRecord:
            if current.progress:
                previous = current.progress[-1]
                if (
                    completed is not None
                    and previous.completed is not None
                    and completed < previous.completed
                ):
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        "Operation progress must be monotonic.",
                    )
            return current.report_progress(event, owner_id=self.owner_id)

        await self._append_active_transition(operation_id, transition)

    async def _append_active_transition(
        self,
        operation_id: str,
        transition: Callable[[ActiveOperationRecord], OperationRecord | None],
    ) -> OperationRecord | None:
        async with self.record_lock:
            current = self.store.read(operation_id)
            if not isinstance(current, ActiveOperationRecord):
                return None
            updated = transition(current)
            if updated is None:
                return None
            self.store.records.append(
                updated,
                expected_revision=current.revision,
            )
            return updated

    def _items(
        self,
        record: _OperationRecord,
        status: OperationItemStatus,
        completed_items: tuple[str, ...] = (),
    ) -> tuple[OperationItemOutcome, ...]:
        completed = set(completed_items)
        return tuple(
            item.with_status(OperationItemStatus.COMPLETE if item.item in completed else status)
            for item in record.item_outcomes
        )

    def _mark_unmanaged(self, record: ActiveOperationRecord) -> UnmanagedOperationRecord:
        return record.unmanaged(recovery=self._retry_recovery(record))

    def _retry_recovery(self, record: _OperationRecord) -> OperationRecovery:
        return OperationRecovery(
            action=OperationRecoveryAction.RETRY_NEW_OPERATION,
            tool=f"start_{self.operation.replace('.', '_')}",
            arguments={
                **record.request,
                "idempotency_key": f"{record.operation_id}:retry:{record.revision + 1}",
            },
        )

    def _lease_is_active(self, record: ActiveOperationRecord) -> bool:
        return utc_now() - record.owner_heartbeat_at < self._LEASE_TIMEOUT
