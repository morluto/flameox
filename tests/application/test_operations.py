from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest

from flameox.application.operations import (
    ActiveOperationRecord,
    OperationFailure,
    OperationProgress,
    OperationRunner,
    OperationStore,
    operation_digests,
)
from flameox.domain import DomainError, ErrorCode
from flameox.domain.models import utc_now
from flameox.storage import Workspace


@pytest.mark.parametrize(
    ("completed", "total"),
    ((1, None), (None, 1), (2, 1)),
)
def test_operation_progress_rejects_incoherent_bounds(
    completed: float | None,
    total: float | None,
) -> None:
    with pytest.raises(ValueError):
        OperationProgress(
            phase="running",
            completed=completed,
            total=total,
            message="invalid progress",
        )


def test_operation_store_rejects_terminal_state_without_a_receipt(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    request = {"value": 1}
    _, idempotency_digest = operation_digests(
        workspace,
        "test.operation",
        request,
        "terminal-test",
    )
    record = ActiveOperationRecord(
        operation="test.operation",
        workspace_id=workspace.identity.workspace_id,
        request=request,
        idempotency_digest=idempotency_digest,
        owner_id="test-owner",
        owner_heartbeat_at=utc_now(),
    )
    store = OperationStore(workspace)
    store.records.create(record)
    serialized = record.model_dump(mode="json")
    serialized.update(
        {
            "state": "terminal",
            "phase": "completed",
            "cleanup_status": "complete",
            "owner_id": None,
            "owner_heartbeat_at": None,
        }
    )
    with sqlite3.connect(workspace.paths.control_plane) as connection:
        connection.execute(
            "UPDATE operations SET payload_json = ? WHERE operation_id = ?",
            (json.dumps(serialized), record.operation_id),
        )

    with pytest.raises(DomainError) as error:
        store.read(record.operation_id)

    assert error.value.code is ErrorCode.WORKSPACE_INVALID


def test_operation_store_rejects_version_one_records(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    request = {"value": 1}
    _, idempotency_digest = operation_digests(
        workspace,
        "test.operation",
        request,
        "version-one-test",
    )
    record = ActiveOperationRecord(
        operation="test.operation",
        workspace_id=workspace.identity.workspace_id,
        request=request,
        idempotency_digest=idempotency_digest,
        owner_id="test-owner",
        owner_heartbeat_at=utc_now(),
    )
    store = OperationStore(workspace)
    store.records.create(record)
    serialized = record.model_dump(mode="json")
    serialized["schema_version"] = 1
    with sqlite3.connect(workspace.paths.control_plane) as connection:
        connection.execute(
            "UPDATE operations SET payload_json = ? WHERE operation_id = ?",
            (json.dumps(serialized), record.operation_id),
        )

    with pytest.raises(DomainError) as error:
        store.read(record.operation_id)

    assert error.value.code is ErrorCode.WORKSPACE_INVALID


def test_operation_record_rejects_contradictory_identity_projections(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    request = {"value": 1}
    _, idempotency_digest = operation_digests(
        workspace,
        "test.operation",
        request,
        "identity-test",
    )
    record = ActiveOperationRecord(
        operation="test.operation",
        workspace_id=workspace.identity.workspace_id,
        request=request,
        idempotency_digest=idempotency_digest,
        owner_id="test-owner",
        owner_heartbeat_at=utc_now(),
    )
    payload = record.model_dump(mode="python")

    with pytest.raises(ValueError, match="request digest"):
        ActiveOperationRecord.model_validate({**payload, "request_digest": "sha256:" + "a" * 64})
    with pytest.raises(ValueError, match="operation identifier"):
        ActiveOperationRecord.model_validate({**payload, "operation_id": "op-" + "b" * 64})

    cancelling = record.request_cancellation()
    assert cancelling.cancellation_requested is True
    with pytest.raises(ValueError, match="cancellation must agree"):
        ActiveOperationRecord.model_validate(
            {**cancelling.model_dump(mode="python"), "cancellation_requested": False}
        )


@pytest.mark.anyio
async def test_operation_runner_persists_exact_request_and_reconnects(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    runner = OperationRunner(workspace, "test.operation")

    async def run(
        operation_id: str,
        progress: object,
    ) -> dict[str, object]:
        report = cast(
            Callable[[str, float | None, float | None, str], Awaitable[None]],
            progress,
        )
        await report("working", 0, 2, "Started")
        await report("working", 2, 2, "Finished")
        return {"receipt": "ok"}

    first = await runner.start({"run_id": "run-1"}, "same-key", run, items=("run-1",))
    second = await runner.start({"run_id": "run-1"}, "same-key", run, items=("run-1",))
    assert second.operation_id == first.operation_id
    assert ":" not in first.operation_id
    assert second.request == {"run_id": "run-1"}

    for _ in range(50):
        status = await runner.status(first.operation_id)
        if status.state == "terminal":
            break
        await asyncio.sleep(0.01)
    assert status.state == "terminal"
    assert [item.status for item in status.item_outcomes] == ["complete"]
    assert [item.completed for item in status.progress] == [0, 2]
    assert status.terminal_receipt == {"receipt": "ok"}

    with pytest.raises(DomainError, match="different request"):
        await runner.start({"run_id": "run-2"}, "same-key", run)


@pytest.mark.anyio
async def test_operation_runner_cancellation_persists_cleanup_and_is_replayable(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    runner = OperationRunner(workspace, "test.operation")
    stopped = asyncio.Event()

    async def run(
        operation_id: str,
        progress: object,
    ) -> dict[str, object]:
        report = cast(
            Callable[[str, float | None, float | None, str], Awaitable[None]],
            progress,
        )
        cancel_event = asyncio.Event()
        runner.set_cancel_hook(operation_id, cancel_event.set)
        await report("waiting", None, None, "Waiting for cancellation")
        await cancel_event.wait()
        stopped.set()
        return {"cleanup": "complete"}

    started = await runner.start({"value": 1}, "cancel-key", run, items=("item",))
    cancelled = await runner.cancel(started.operation_id)
    assert stopped.is_set()
    assert cancelled.state == "cancelled"
    assert cancelled.cleanup_status == "complete"
    assert cancelled.cancellation_requested is True
    assert cancelled.recovery is not None
    assert cancelled.recovery.action == "retry_new_operation"
    assert cancelled.recovery.arguments["value"] == 1
    assert cancelled.recovery.arguments["idempotency_key"]


@pytest.mark.anyio
async def test_operation_runner_cancellation_preserves_failure_details(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    runner = OperationRunner(workspace, "test.operation")

    async def run(
        operation_id: str,
        progress: object,
    ) -> dict[str, object]:
        del progress
        cancel_event = asyncio.Event()
        runner.set_cancel_hook(operation_id, cancel_event.set)
        await cancel_event.wait()
        raise DomainError(
            ErrorCode.PROCESS_TIMEOUT,
            "The operation was interrupted during staging.",
            details={"phase": "staging", "failure_category": "timeout"},
        )

    started = await runner.start({"value": 1}, "cancel-details-key", run)
    cancelled = await runner.cancel(started.operation_id)

    assert cancelled.state == "cancelled"
    assert cancelled.failure_details == {
        "phase": "staging",
        "failure_category": "timeout",
    }


@pytest.mark.anyio
async def test_operation_runner_idempotency_is_shared_by_runners(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    first_runner = OperationRunner(workspace, "test.operation")
    second_runner = OperationRunner(workspace, "test.operation")
    started = 0
    invoked = asyncio.Event()
    release = asyncio.Event()

    async def run(
        operation_id: str,
        progress: object,
    ) -> dict[str, object]:
        nonlocal started
        started += 1
        invoked.set()
        await release.wait()
        return {"operation_id": operation_id}

    first, second = await asyncio.gather(
        first_runner.start({"value": 1}, "same-key", run),
        second_runner.start({"value": 1}, "same-key", run),
    )
    assert first.operation_id == second.operation_id
    await asyncio.wait_for(invoked.wait(), timeout=5)
    release.set()
    try:
        status = first
        for _ in range(50):
            status = await first_runner.status(first.operation_id)
            if status.state == "terminal":
                break
            await asyncio.sleep(0.01)

        assert started == 1
        assert status.state == "terminal"
    finally:
        await first_runner.shutdown()
        await second_runner.shutdown()


@pytest.mark.anyio
async def test_operation_runner_does_not_steal_a_live_cross_process_lease(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    first_runner = OperationRunner(workspace, "test.operation")
    second_runner = OperationRunner(workspace, "test.operation")
    stopped = asyncio.Event()

    async def run(
        operation_id: str,
        progress: object,
    ) -> dict[str, object]:
        cancel_event = asyncio.Event()
        first_runner.set_cancel_hook(operation_id, cancel_event.set)
        await cancel_event.wait()
        stopped.set()
        return {"receipt": "cancelled"}

    started = await first_runner.start({"value": 1}, "lease-key", run)
    observed = await second_runner.status(started.operation_id)

    assert observed.state == "running"
    assert observed.recovery is None
    await first_runner.cancel(started.operation_id)
    assert stopped.is_set()


@pytest.mark.anyio
async def test_operation_runner_preserves_completed_items_on_failure(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    runner = OperationRunner(workspace, "test.operation")

    async def run(
        operation_id: str,
        progress: object,
    ) -> dict[str, object]:
        raise OperationFailure(
            DomainError(ErrorCode.PROCESS_FAILED, "The second item failed.", retryable=True),
            completed_items=("first",),
        )

    started = await runner.start(
        {"value": 1},
        "failure-key",
        run,
        items=("first", "second"),
    )
    status = started
    for _ in range(50):
        status = await runner.status(started.operation_id)
        if status.state == "failed":
            break
        await asyncio.sleep(0.01)

    assert status.state == "failed"
    assert [item.status for item in status.item_outcomes] == ["complete", "retryable"]
