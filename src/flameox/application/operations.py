from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from flameox.domain import DomainError, ErrorCode, digest_model
from flameox.domain.models import utc_now
from flameox.models import ContractModel
from flameox.storage import JsonRecordStore, Workspace

OperationState = Literal[
    "starting",
    "running",
    "terminal",
    "failed",
    "cancelled",
    "unmanaged_after_restart",
]


class OperationProgress(ContractModel):
    phase: str = Field(min_length=1, max_length=100)
    completed: float | None = Field(default=None, ge=0)
    total: float | None = Field(default=None, gt=0)
    message: str = Field(min_length=1, max_length=500)
    observed_at: datetime = Field(default_factory=utc_now)


class OperationItemOutcome(ContractModel):
    item: str = Field(min_length=1, max_length=200)
    status: Literal["pending", "running", "complete", "retryable", "unavailable", "failed"]
    message: str | None = Field(default=None, max_length=500)


class OperationRecovery(ContractModel):
    action: Literal[
        "poll",
        "retry_same_request",
        "retry_failed_items",
        "inspect_capabilities",
        "extract_required_evidence",
        "reread_snapshot",
    ]
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class OperationRecord(ContractModel):
    """Durable state for local work that may outlive one MCP request."""

    schema_version: Literal[1] = 1
    operation_id: str
    operation: str
    workspace_id: str
    request_digest: str
    request: dict[str, Any]
    idempotency_digest: str
    state: OperationState = "starting"
    phase: str = "starting"
    revision: int = 0
    progress: tuple[OperationProgress, ...] = Field(default=(), max_length=32)
    item_outcomes: tuple[OperationItemOutcome, ...] = Field(default=(), max_length=64)
    failure_code: str | None = None
    failure_message: str | None = None
    cancellation_requested: bool = False
    cleanup_status: Literal["not_required", "pending", "complete", "incomplete"] = "not_required"
    terminal_receipt: dict[str, Any] | None = None
    recovery: OperationRecovery | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


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
    cleanup_status: Literal["not_required", "pending", "complete", "incomplete"]
    failure_code: str | None
    failure_message: str | None
    terminal_receipt: dict[str, Any] | None
    recovery: OperationRecovery | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record: OperationRecord) -> OperationStatus:
        return cls(**record.model_dump())


class OperationStore:
    def __init__(self, workspace: Workspace) -> None:
        self.records = JsonRecordStore(
            workspace,
            kind="operations",
            model=OperationRecord,
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

    def __init__(self, workspace: Workspace, operation: str) -> None:
        self.workspace = workspace
        self.operation = operation
        self.store = OperationStore(workspace)
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.cancel_events: dict[str, asyncio.Event] = {}
        self.cancel_hooks: dict[str, Callable[[], None]] = {}
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
                    existing.state in {"starting", "running"}
                    and existing.operation_id not in self.tasks
                ):
                    existing = existing.model_copy(
                        update={
                            "state": "unmanaged_after_restart",
                            "recovery": OperationRecovery(
                                action="retry_same_request",
                                tool=f"start_{self.operation.replace('.', '_')}",
                                arguments=request,
                            ),
                        }
                    )
                    self.store.records.append(existing, expected_revision=existing.revision)
                return OperationStatus.from_record(existing)

            operation_id = f"op-{uuid4().hex}"
            record = OperationRecord(
                operation_id=operation_id,
                operation=self.operation,
                workspace_id=self.workspace.identity.workspace_id,
                request_digest=request_digest,
                request=request,
                idempotency_digest=idempotency_digest,
                item_outcomes=tuple(
                    OperationItemOutcome(item=item, status="pending") for item in items
                ),
            )
            self.store.records.create(record)
            cancel_event = asyncio.Event()
            self.cancel_events[operation_id] = cancel_event
            self.tasks[operation_id] = asyncio.create_task(
                self._execute(operation_id, run, cancel_event),
                name=f"flameox-operation-{operation_id}",
            )
        await asyncio.sleep(0)
        return OperationStatus.from_record(self.store.read(operation_id))

    async def status(self, operation_id: str) -> OperationStatus:
        record = self.store.read(operation_id)
        if record.state in {"starting", "running"} and operation_id not in self.tasks:
            record = record.model_copy(
                update={
                    "revision": record.revision + 1,
                    "state": "unmanaged_after_restart",
                    "recovery": OperationRecovery(
                        action="retry_same_request",
                        tool=f"start_{self.operation.replace('.', '_')}",
                        arguments=record.request,
                    ),
                    "updated_at": utc_now(),
                }
            )
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
        if record.state in {"terminal", "failed", "cancelled"}:
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
        current = self.store.read(operation_id)
        updated = current.model_copy(
            update={
                "revision": current.revision + 1,
                "cancellation_requested": True,
                "phase": "cancelling",
                "cleanup_status": "pending",
                "updated_at": utc_now(),
            }
        )
        self.store.records.append(updated, expected_revision=current.revision)
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
            await self._update(operation_id, state="running", phase="starting")

            async def progress(
                phase: str,
                completed: float | None = None,
                total: float | None = None,
                message: str = "Operation in progress.",
            ) -> None:
                await self._progress(operation_id, phase, completed, total, message)

            receipt = await run(operation_id, progress)
            current = self.store.read(operation_id)
            if cancel_event.is_set() or current.cancellation_requested:
                await self._update(
                    operation_id,
                    state="cancelled",
                    phase="cancelled",
                    cleanup_status="complete",
                    terminal_receipt=receipt,
                    item_outcomes=self._items(operation_id, "pending"),
                    recovery=OperationRecovery(
                        action="retry_same_request",
                        tool=f"start_{self.operation.replace('.', '_')}",
                        arguments=self.store.read(operation_id).request,
                    ),
                )
            else:
                await self._update(
                    operation_id,
                    state="terminal",
                    phase="completed",
                    cleanup_status="complete",
                    terminal_receipt=receipt,
                    item_outcomes=self._items(operation_id, "complete"),
                )
        except asyncio.CancelledError:
            await self._update(
                operation_id,
                state="cancelled",
                phase="cancelled",
                cleanup_status="complete",
                item_outcomes=self._items(operation_id, "pending"),
            )
        except DomainError as error:
            if cancel_event.is_set() or error.code is ErrorCode.PROCESS_CANCELLED:
                await self._update(
                    operation_id,
                    state="cancelled",
                    phase="cancelled",
                    cleanup_status="complete",
                    failure_code=error.code.value,
                    failure_message=error.message,
                    item_outcomes=self._items(operation_id, "pending"),
                )
            else:
                await self._update(
                    operation_id,
                    state="failed",
                    phase="failed",
                    cleanup_status="complete",
                    failure_code=error.code.value,
                    failure_message=error.message,
                    item_outcomes=self._items(
                        operation_id,
                        "retryable" if error.retryable else "failed",
                    ),
                    recovery=OperationRecovery(
                        action="retry_same_request" if error.retryable else "inspect_capabilities",
                        tool=f"start_{self.operation.replace('.', '_')}",
                        arguments=self.store.read(operation_id).request,
                    ),
                )
        except Exception as error:
            await self._update(
                operation_id,
                state="failed",
                phase="failed",
                cleanup_status="complete",
                failure_code=ErrorCode.INTERNAL_ERROR.value,
                failure_message=f"Operation failed with {type(error).__name__}.",
                item_outcomes=self._items(operation_id, "failed"),
            )

    async def _progress(
        self,
        operation_id: str,
        phase: str,
        completed: float | None,
        total: float | None,
        message: str,
    ) -> None:
        current = self.store.read(operation_id)
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
        event = OperationProgress(
            phase=phase,
            completed=completed,
            total=total,
            message=message,
        )
        self.store.records.append(
            current.model_copy(
                update={
                    "revision": current.revision + 1,
                    "phase": phase,
                    "progress": (*current.progress[-31:], event),
                    "updated_at": utc_now(),
                }
            ),
            expected_revision=current.revision,
        )

    async def _update(self, operation_id: str, **updates: Any) -> None:
        current = self.store.read(operation_id)
        self.store.records.append(
            current.model_copy(
                update={"revision": current.revision + 1, "updated_at": utc_now(), **updates}
            ),
            expected_revision=current.revision,
        )

    def _items(
        self,
        operation_id: str,
        status: Literal["pending", "running", "complete", "retryable", "unavailable", "failed"],
    ) -> tuple[OperationItemOutcome, ...]:
        current = self.store.read(operation_id)
        return tuple(item.model_copy(update={"status": status}) for item in current.item_outcomes)
