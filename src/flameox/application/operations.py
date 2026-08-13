from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from uuid import uuid4

import anyio
from pydantic import Field, TypeAdapter, computed_field, model_validator

from flameox.action_graph import ActionId, NextAction, next_action_for_action
from flameox.application.task_supervisor import (
    TaskHandle,
    TaskSupervisor,
    start_local_task,
)
from flameox.domain import DomainError, ErrorCode, digest_model
from flameox.domain.models import Digest, utc_now
from flameox.models import ContractModel
from flameox.storage import ControlPlane, Workspace
from flameox.storage.control_plane import canonical_json


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
    FOLLOW_NEXT_ACTION = "follow_next_action"


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
    next_action: NextAction


class OperationAdapter(ContractModel):
    """Stable action bindings supplied to the durable lifecycle kernel."""

    kind: str = Field(min_length=1, max_length=100)
    start_action: ActionId
    status_action: ActionId
    status_identifier: Literal["operation_id", "run_id", "reduction_id"] = "operation_id"
    retry_with_idempotency_key: bool = True
    poll_after_ms: int = Field(default=1_000, ge=100, le=30_000)
    recover_unmanaged: bool = False


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
    subject_id: str | None = None
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

    def recover(self, *, owner_id: str) -> ActiveOperationRecord:
        return ActiveOperationRecord.model_validate(
            {
                **self._next_revision(),
                "state": OperationState.STARTING,
                "phase": "recovering",
                "failure_code": None,
                "failure_message": None,
                "failure_details": None,
                "terminal_receipt": None,
                "recovery": None,
                "owner_id": owner_id,
                "owner_heartbeat_at": utc_now(),
            }
        )


type OperationRecord = Annotated[
    ActiveOperationRecord
    | CompletedOperationRecord
    | FailedOperationRecord
    | CancelledOperationRecord
    | UnmanagedOperationRecord,
    Field(discriminator="state"),
]

_OPERATION_RECORD: TypeAdapter[OperationRecord] = TypeAdapter(OperationRecord)


def _validate_operation_transition(
    current: OperationRecord,
    updated: OperationRecord,
    *,
    expected_revision: int,
) -> None:
    if current.revision != expected_revision or updated.revision != expected_revision + 1:
        raise DomainError(
            ErrorCode.REVISION_CONFLICT,
            "The operation transition revision is not consecutive.",
        )
    immutable_fields = (
        "operation",
        "workspace_id",
        "request",
        "subject_id",
        "idempotency_digest",
        "created_at",
    )
    if any(getattr(current, field) != getattr(updated, field) for field in immutable_fields):
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            "An operation transition changed immutable identity or intent.",
            details={"operation_id": current.operation_id},
        )
    if isinstance(current, UnmanagedOperationRecord) and isinstance(
        updated,
        ActiveOperationRecord,
    ):
        if updated.state is not OperationState.STARTING:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "An unmanaged operation can recover only through the starting state.",
                details={"operation_id": current.operation_id},
            )
        return
    if not isinstance(current, ActiveOperationRecord):
        raise DomainError(
            ErrorCode.REVISION_CONFLICT,
            "A terminal or unmanaged operation cannot transition again.",
            details={"operation_id": current.operation_id, "state": current.state.value},
        )
    if isinstance(updated, ActiveOperationRecord):
        if current.owner_id != updated.owner_id:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "An active operation transition cannot replace its lease owner.",
                details={"operation_id": current.operation_id},
            )
        if current.state is OperationState.RUNNING and updated.state is OperationState.STARTING:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "A running operation cannot transition back to starting.",
                details={"operation_id": current.operation_id},
            )


class OperationStatus(ContractModel):
    schema_version: Literal[1] = 1
    operation_id: str
    operation: str
    workspace_id: str
    request_digest: str
    request: dict[str, Any]
    subject_id: str | None = None
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
    def from_record(
        cls,
        record: OperationRecord,
        *,
        adapter: OperationAdapter | None = None,
    ) -> OperationStatus:
        payload = record.model_dump(exclude={"owner_id", "owner_heartbeat_at"})
        payload["schema_version"] = 1
        if adapter is not None and adapter.kind != record.operation:
            raise ValueError("operation adapter kind does not match the durable record")
        if adapter is not None and isinstance(record, ActiveOperationRecord):
            payload["poll_after_ms"] = adapter.poll_after_ms
            payload["recovery"] = OperationRecovery(
                action=OperationRecoveryAction.POLL,
                next_action=next_action_for_action(
                    adapter.status_action,
                    context={adapter.status_identifier: record.operation_id},
                    instruction="Supply the operation identity required to inspect its status.",
                ),
            ).model_dump()
        else:
            payload["poll_after_ms"] = None
        return cls.model_validate(payload)


class _OperationRecords:
    def __init__(self, workspace: Workspace) -> None:
        self.control_plane = ControlPlane(workspace)

    def create(self, record: OperationRecord) -> OperationRecord:
        canonical = _OPERATION_RECORD.validate_python(record.model_dump(mode="python"))
        self.control_plane.create_operation(
            operation_id=canonical.operation_id,
            kind=canonical.operation,
            state=canonical.state,
            revision=canonical.revision,
            idempotency_digest=canonical.idempotency_digest,
            intent_digest=canonical.request_digest,
            payload_json=canonical_json(canonical.model_dump(mode="json")),
            run_id=canonical.subject_id,
        )
        return record

    def read(self, operation_id: str) -> OperationRecord:
        try:
            return _OPERATION_RECORD.validate_json(self.control_plane.read_operation(operation_id))
        except ValueError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Operation {operation_id!r} does not exist or is invalid.",
            ) from exc

    def list(self) -> tuple[OperationRecord, ...]:
        try:
            return tuple(
                _OPERATION_RECORD.validate_json(payload)
                for payload in self.control_plane.list_operations()
            )
        except ValueError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "An operation record is invalid.",
            ) from exc

    def append(self, record: OperationRecord, *, expected_revision: int) -> OperationRecord:
        canonical = _OPERATION_RECORD.validate_python(record.model_dump(mode="python"))
        current = self.read(canonical.operation_id)
        _validate_operation_transition(
            current,
            canonical,
            expected_revision=expected_revision,
        )
        self.control_plane.append_operation(
            operation_id=canonical.operation_id,
            state=canonical.state,
            expected_revision=expected_revision,
            next_revision=canonical.revision,
            payload_json=canonical_json(canonical.model_dump(mode="json")),
        )
        return record


class OperationStore:
    def __init__(self, workspace: Workspace) -> None:
        self.records = _OperationRecords(workspace)

    def read(self, operation_id: str) -> OperationRecord:
        return self.records.read(operation_id)

    def find(self, *, operation: str, idempotency_digest: str) -> OperationRecord | None:
        payload = self.records.control_plane.find_operation(
            kind=operation,
            idempotency_digest=idempotency_digest,
        )
        return _OPERATION_RECORD.validate_json(payload) if payload is not None else None

    def find_subject(self, *, operation: str, subject_id: str) -> OperationRecord | None:
        payload = self.records.control_plane.find_operation_by_run_id(
            kind=operation,
            run_id=subject_id,
        )
        return _OPERATION_RECORD.validate_json(payload) if payload is not None else None


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

    def __init__(
        self,
        workspace: Workspace,
        adapter: OperationAdapter,
        *,
        supervisor: TaskSupervisor | None = None,
    ) -> None:
        self.workspace = workspace
        self.adapter = adapter
        self.operation = adapter.kind
        self.store = OperationStore(workspace)
        self.supervisor = supervisor
        self.tasks: dict[str, TaskHandle] = {}
        self.cancel_events: dict[str, asyncio.Event] = {}
        self.cancel_hooks: dict[str, Callable[[], None]] = {}
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
        subject_id: str | None = None,
    ) -> OperationStatus:
        request_digest, idempotency_digest = operation_digests(
            self.workspace,
            self.operation,
            request,
            idempotency_key,
        )
        started_event: anyio.Event | None = None
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
                if existing.subject_id != subject_id:
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        "The idempotency key is already bound to a different operation subject.",
                        details={"operation_id": existing.operation_id},
                    )
                if (
                    isinstance(existing, ActiveOperationRecord)
                    and existing.operation_id not in self.tasks
                    and not self._lease_is_active(existing)
                ):
                    unmanaged = self._mark_unmanaged(existing)
                    try:
                        self.store.records.append(
                            unmanaged,
                            expected_revision=existing.revision,
                        )
                        existing = unmanaged
                    except DomainError as error:
                        if error.code is not ErrorCode.REVISION_CONFLICT:
                            raise
                        existing = self.store.read(existing.operation_id)
                if (
                    isinstance(existing, UnmanagedOperationRecord)
                    and self.adapter.recover_unmanaged
                ):
                    recovered = existing.recover(owner_id=self.owner_id)
                    try:
                        self.store.records.append(
                            recovered,
                            expected_revision=existing.revision,
                        )
                    except DomainError as error:
                        if error.code is not ErrorCode.REVISION_CONFLICT:
                            raise
                        return OperationStatus.from_record(
                            self.store.read(existing.operation_id),
                            adapter=self.adapter,
                        )
                    operation_id = recovered.operation_id
                    started_event = self._schedule(operation_id, run)
                else:
                    return OperationStatus.from_record(existing, adapter=self.adapter)

            if started_event is None:
                # The digest-derived identity makes the create itself the cross-process
                # idempotency gate. A process-local asyncio lock cannot protect two MCP
                # server instances sharing one workspace.
                record = ActiveOperationRecord(
                    operation=self.operation,
                    workspace_id=self.workspace.identity.workspace_id,
                    request=request,
                    subject_id=subject_id,
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
                    existing = self.store.find(
                        operation=self.operation,
                        idempotency_digest=idempotency_digest,
                    )
                    if existing is None and subject_id is not None:
                        existing = self.store.find_subject(
                            operation=self.operation,
                            subject_id=subject_id,
                        )
                    if existing is None:
                        raise
                    if existing.request_digest != request_digest:
                        raise DomainError(
                            ErrorCode.REVISION_CONFLICT,
                            "The idempotency key is already bound to a different request.",
                            details={"operation_id": existing.operation_id},
                        ) from error
                    if existing.subject_id != subject_id:
                        raise DomainError(
                            ErrorCode.REVISION_CONFLICT,
                            "The operation subject is already bound to a different request.",
                            details={"operation_id": existing.operation_id},
                        ) from error
                    return OperationStatus.from_record(existing, adapter=self.adapter)
                started_event = self._schedule(operation_id, run)
        assert started_event is not None
        await started_event.wait()
        return OperationStatus.from_record(
            self.store.read(operation_id),
            adapter=self.adapter,
        )

    def _schedule(
        self,
        operation_id: str,
        run: Callable[[str, Callable[..., Awaitable[None]]], Awaitable[dict[str, Any]]],
    ) -> anyio.Event:
        cancel_event = asyncio.Event()
        started_event = anyio.Event()
        self.cancel_events[operation_id] = cancel_event
        task_name = f"flameox-operation-{operation_id}"

        async def function() -> None:
            await self._execute(
                operation_id,
                run,
                cancel_event,
                started_event,
            )

        self.tasks[operation_id] = (
            self.supervisor.start(function, name=task_name)
            if self.supervisor is not None
            else start_local_task(function, name=task_name)
        )
        return started_event

    async def status(self, operation_id: str) -> OperationStatus:
        record = self.store.read(operation_id)
        if (
            isinstance(record, ActiveOperationRecord)
            and operation_id not in self.tasks
            and not self._lease_is_active(record)
        ):
            unmanaged = self._mark_unmanaged(record)
            try:
                self.store.records.append(unmanaged, expected_revision=record.revision)
                record = unmanaged
            except DomainError as error:
                if error.code is not ErrorCode.REVISION_CONFLICT:
                    raise
                record = self.store.read(operation_id)
        return OperationStatus.from_record(record, adapter=self.adapter)

    async def wait(
        self,
        operation_id: str,
        *,
        timeout_seconds: float,
    ) -> OperationStatus:
        task = self.tasks.get(operation_id)
        if task is not None:
            with anyio.move_on_after(timeout_seconds):
                await task.wait()
            return await self.status(operation_id)
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            status = await self.status(operation_id)
            if status.state not in {OperationState.STARTING, OperationState.RUNNING}:
                return status
            if asyncio.get_running_loop().time() >= deadline:
                return status
            await anyio.sleep(0.1)

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
            return OperationStatus.from_record(record, adapter=self.adapter)
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
        await self._append_active_transition(
            operation_id,
            lambda current: current.request_cancellation(),
        )
        hook = self.cancel_hooks.get(operation_id)
        if hook is not None:
            hook()
        task = self.tasks.get(operation_id)
        if task is not None:
            await task.wait()
        return OperationStatus.from_record(
            self.store.read(operation_id),
            adapter=self.adapter,
        )

    async def shutdown(self) -> None:
        for event in self.cancel_events.values():
            event.set()
        for hook in self.cancel_hooks.values():
            hook()
        tasks = [task for task in self.tasks.values() if not task.done]
        if tasks:
            await asyncio.gather(*(task.wait() for task in tasks))

    async def _execute(
        self,
        operation_id: str,
        run: Callable[[str, Callable[..., Awaitable[None]]], Awaitable[dict[str, Any]]],
        cancel_event: asyncio.Event,
        started_event: anyio.Event,
    ) -> None:
        try:
            async with anyio.create_task_group() as lifetime:
                lifetime.start_soon(
                    self._heartbeat,
                    operation_id,
                    name=f"flameox-operation-lease-{operation_id}",
                )
                try:
                    started = await self._append_active_transition(
                        operation_id,
                        lambda current: current.running(),
                    )
                    started_event.set()
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
                    lifetime.cancel_scope.cancel()
        finally:
            started_event.set()
            self.tasks.pop(operation_id, None)
            self.cancel_events.pop(operation_id, None)
            self.cancel_hooks.pop(operation_id, None)

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
                        action=(
                            OperationRecoveryAction.FOLLOW_NEXT_ACTION
                            if error.next_action is not None
                            else OperationRecoveryAction.RETRY_NEW_OPERATION
                        ),
                        next_action=(
                            error.next_action
                            if error.next_action is not None
                            else next_action_for_action(
                                self.adapter.start_action,
                                context=current.request,
                                instruction=(
                                    "Supply the complete request required to start a replacement "
                                    "operation."
                                ),
                            )
                        ),
                    )
                ),
            )

        await self._append_active_transition(operation_id, transition)

    @staticmethod
    def _failure_details(error: DomainError) -> dict[str, Any]:
        """Persist only bounded, recovery-relevant diagnostics from a domain failure."""
        details: dict[str, Any] = {}
        for key in ("phase", "failure_category", "adapter"):
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
            if current.owner_id != self.owner_id:
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
        arguments = dict(record.request)
        if self.adapter.retry_with_idempotency_key:
            arguments["idempotency_key"] = f"{record.operation_id}:retry:{record.revision + 1}"
        return OperationRecovery(
            action=OperationRecoveryAction.RETRY_NEW_OPERATION,
            next_action=next_action_for_action(
                self.adapter.start_action,
                context=arguments,
                instruction=(
                    "Supply the complete request required to start a replacement operation."
                ),
            ),
        )

    def _lease_is_active(self, record: ActiveOperationRecord) -> bool:
        return utc_now() - record.owner_heartbeat_at < self._LEASE_TIMEOUT
