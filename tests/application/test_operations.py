from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest

from flameox.application.operations import OperationRunner
from flameox.domain import DomainError
from flameox.storage import Workspace


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
    assert cancelled.recovery.arguments == {"value": 1}
