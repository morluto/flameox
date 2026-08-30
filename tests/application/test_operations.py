from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import cast

import anyio
import pytest

from flameox.action_graph import ActionId, ManualAction, ToolAction
from flameox.application.operations import (
    ActiveOperationRecord,
    OperationAdapter,
    OperationFailure,
    OperationProgress,
    OperationRunner,
    OperationState,
    OperationStore,
    operation_digests,
)
from flameox.application.task_supervisor import TaskSupervisor
from flameox.domain import DomainError, ErrorCode
from flameox.domain.models import utc_now
from flameox.storage import Workspace
from flameox.storage.control_plane import ControlPlane

pytestmark = pytest.mark.unit

_TEST_OPERATION = OperationAdapter(
    kind="test.operation",
    start_action=ActionId.START_CAPABILITY_SETUP,
    status_action=ActionId.GET_CAPABILITY_SETUP,
)


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
    store.create(record)
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


def test_operation_record_keeps_derived_status_fields_out_of_storage(tmp_path: Path) -> None:
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

    cancelling = record.request_cancellation()
    assert not {"operation_id", "request_digest", "cancellation_requested"} & payload.keys()
    assert record.operation_id.startswith("op-")
    assert record.request_digest.startswith("sha256:")
    assert cancelling.cancellation_requested is True


def test_operation_store_rejects_transition_after_terminal_state(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    request = {"value": 1}
    _, idempotency_digest = operation_digests(
        workspace,
        "test.operation",
        request,
        "terminal-immutability",
    )
    active = ActiveOperationRecord(
        operation="test.operation",
        workspace_id=workspace.identity.workspace_id,
        request=request,
        idempotency_digest=idempotency_digest,
        owner_id="test-owner",
        owner_heartbeat_at=utc_now(),
    )
    store = OperationStore(workspace)
    store.create(active)
    terminal = active.completed(receipt={"result": "done"}, item_outcomes=())
    store.append(terminal, expected_revision=active.revision)
    reactivated = ActiveOperationRecord(
        operation=active.operation,
        workspace_id=active.workspace_id,
        request=active.request,
        idempotency_digest=active.idempotency_digest,
        revision=terminal.revision + 1,
        state=OperationState.RUNNING,
        owner_id="new-owner",
        owner_heartbeat_at=utc_now(),
        created_at=active.created_at,
    )

    with pytest.raises(DomainError, match="cannot transition again"):
        store.append(reactivated, expected_revision=terminal.revision)

    assert store.read(active.operation_id) == terminal


def test_operation_store_rejects_running_to_starting_transition(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    request = {"value": 1}
    _, idempotency_digest = operation_digests(
        workspace,
        "test.operation",
        request,
        "state-regression",
    )
    active = ActiveOperationRecord(
        operation="test.operation",
        workspace_id=workspace.identity.workspace_id,
        request=request,
        idempotency_digest=idempotency_digest,
        owner_id="test-owner",
        owner_heartbeat_at=utc_now(),
    )
    store = OperationStore(workspace)
    store.create(active)
    running = active.running()
    store.append(running, expected_revision=active.revision)
    regressed = ActiveOperationRecord.model_validate(
        {
            **running.model_dump(mode="python"),
            "revision": running.revision + 1,
            "state": "starting",
        }
    )

    with pytest.raises(DomainError, match="back to starting"):
        store.append(regressed, expected_revision=running.revision)


def test_operation_revision_history_retains_creation_and_a_bounded_tail(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    request = {"value": 1}
    _, idempotency_digest = operation_digests(
        workspace,
        "test.operation",
        request,
        "bounded-history",
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
    store.create(record)

    for _ in range(ControlPlane.MAX_OPERATION_REVISIONS * 2):
        updated = record.heartbeat()
        store.append(updated, expected_revision=record.revision)
        record = updated

    with sqlite3.connect(workspace.paths.control_plane) as connection:
        revisions = [
            int(row[0])
            for row in connection.execute(
                "SELECT revision FROM operation_revisions WHERE operation_id = ? ORDER BY revision",
                (record.operation_id,),
            )
        ]

    assert len(revisions) == ControlPlane.MAX_OPERATION_REVISIONS
    assert revisions[0] == 0
    assert revisions[-1] == record.revision


@pytest.mark.anyio
async def test_operation_runner_persists_exact_request_and_reconnects(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    runner = OperationRunner(workspace, _TEST_OPERATION)

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

    for _ in range(50):
        if not runner.tasks:
            break
        await asyncio.sleep(0.01)
    assert runner.tasks == {}
    assert runner.cancel_events == {}
    assert runner.cancel_hooks == {}

    restarted = OperationRunner(workspace, _TEST_OPERATION)
    reconnected = await restarted.start({"run_id": "run-1"}, "same-key", run)
    assert reconnected.state == "terminal"
    assert restarted.tasks == {}

    with pytest.raises(DomainError, match="different request"):
        await runner.start({"run_id": "run-2"}, "same-key", run)


@pytest.mark.anyio
async def test_operation_runner_uses_lifespan_task_supervisor(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    async with anyio.create_task_group() as tasks:
        runner = OperationRunner(
            workspace,
            _TEST_OPERATION,
            supervisor=TaskSupervisor(tasks),
        )

        async def run(operation_id: str, progress: object) -> dict[str, object]:
            return {"operation_id": operation_id}

        started = await runner.start({"value": 1}, "supervised-key", run)
        status = await runner.status(started.operation_id)

        assert status.state == "terminal"
        for _ in range(50):
            if not runner.tasks:
                break
            await asyncio.sleep(0.01)
        assert runner.tasks == {}
        tasks.cancel_scope.cancel()


@pytest.mark.anyio
async def test_operation_runner_cancellation_persists_cleanup(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    runner = OperationRunner(workspace, _TEST_OPERATION)
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
        raise DomainError(ErrorCode.PROCESS_CANCELLED, "cancelled")

    started = await runner.start({"value": 1}, "cancel-key", run, items=("item",))
    cancelled = await runner.cancel(started.operation_id)
    assert stopped.is_set()
    assert cancelled.state == "cancelled"
    assert cancelled.cleanup_status == "complete"
    assert cancelled.cancellation_requested is True
    assert cancelled.recovery is not None
    assert cancelled.recovery.action == "retry_new_operation"
    assert isinstance(cancelled.recovery.next_action, ManualAction)
    assert cancelled.recovery.next_action.suggested_action is ActionId.START_CAPABILITY_SETUP
    assert set(cancelled.recovery.next_action.missing_arguments) == {"adapters"}


@pytest.mark.anyio
async def test_operation_runner_cancellation_preserves_failure_details(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    runner = OperationRunner(workspace, _TEST_OPERATION)

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
async def test_operation_runner_committed_completion_wins_after_cancellation_request(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    runner = OperationRunner(workspace, _TEST_OPERATION)
    cancellation_seen = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def run(operation_id: str, progress: object) -> dict[str, object]:
        del progress
        runner.set_cancel_hook(operation_id, cancellation_seen.set)
        await cancellation_seen.wait()
        await release_cleanup.wait()
        return {"cleanup": "complete"}

    started = await runner.start(
        {"value": 1},
        "bounded-cancel-key",
        run,
        items=("item",),
    )
    before = asyncio.get_running_loop().time()
    cancelling = await runner.cancel(started.operation_id)

    assert asyncio.get_running_loop().time() - before < 1
    assert cancelling.state == "running"
    assert cancelling.phase == "cancelling"
    assert cancelling.cancellation_requested is True
    assert cancelling.cleanup_status == "pending"

    release_cleanup.set()
    terminal = await runner.wait(started.operation_id, timeout_seconds=1)
    assert terminal.state == "terminal"
    assert terminal.cleanup_status == "complete"
    assert terminal.cancellation_requested is False
    assert terminal.terminal_receipt == {"cleanup": "complete"}
    assert terminal.item_outcomes[0].status == "complete"


@pytest.mark.anyio
async def test_operation_runner_idempotency_is_shared_by_runners(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    first_runner = OperationRunner(workspace, _TEST_OPERATION)
    second_runner = OperationRunner(workspace, _TEST_OPERATION)
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
    first_runner = OperationRunner(workspace, _TEST_OPERATION)
    second_runner = OperationRunner(workspace, _TEST_OPERATION)
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
    assert observed.recovery is not None
    assert observed.recovery.action == "poll"
    assert isinstance(observed.recovery.next_action, ToolAction)
    assert observed.recovery.next_action.action is ActionId.GET_CAPABILITY_SETUP
    assert observed.recovery.next_action.arguments == {"operation_id": started.operation_id}
    await first_runner.cancel(started.operation_id)
    assert stopped.is_set()


@pytest.mark.anyio
async def test_operation_runner_recovers_one_stale_owner_under_cas(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    adapter = OperationAdapter(
        kind="test.recoverable",
        start_action=ActionId.START_CAPABILITY_SETUP,
        status_action=ActionId.GET_CAPABILITY_SETUP,
        recover_unmanaged=True,
    )
    request = {"value": 1}
    _, idempotency_digest = operation_digests(
        workspace,
        adapter.kind,
        request,
        "recover-key",
    )
    stale = ActiveOperationRecord(
        operation=adapter.kind,
        workspace_id=workspace.identity.workspace_id,
        request=request,
        idempotency_digest=idempotency_digest,
        owner_id="dead-owner",
        owner_heartbeat_at=utc_now() - timedelta(minutes=1),
    )
    OperationStore(workspace).create(stale)
    first_runner = OperationRunner(workspace, adapter)
    second_runner = OperationRunner(workspace, adapter)
    executions = 0

    async def run(operation_id: str, progress: object) -> dict[str, object]:
        nonlocal executions
        executions += 1
        return {"operation_id": operation_id}

    first, second = await asyncio.gather(
        first_runner.start(request, "recover-key", run),
        second_runner.start(request, "recover-key", run),
    )
    status = await first_runner.wait(first.operation_id, timeout_seconds=5)

    assert first.operation_id == second.operation_id == stale.operation_id
    assert status.state == "terminal"
    assert executions == 1


@pytest.mark.anyio
async def test_operation_runner_preserves_completed_items_on_failure(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    runner = OperationRunner(workspace, _TEST_OPERATION)

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
