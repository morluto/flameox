from __future__ import annotations

import asyncio

import pytest

from flameox.application.progress import ProgressReporter

pytestmark = pytest.mark.unit


@pytest.mark.anyio
async def test_progress_failure_is_dropped_without_changing_control_flow() -> None:
    calls = 0

    async def failing_callback(completed: float, total: float, message: str) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("transport closed")

    reporter = ProgressReporter(failing_callback)

    await reporter.report(0, 2, "starting")
    await reporter.report(1, 2, "running")

    assert calls == 1
    assert reporter.dropped_count == 1


@pytest.mark.anyio
async def test_progress_callback_cannot_forge_task_cancellation() -> None:
    async def misleading_callback(completed: float, total: float, message: str) -> None:
        raise asyncio.CancelledError

    reporter = ProgressReporter(misleading_callback)

    await reporter.report(0, 1, "starting")

    assert reporter.dropped_count == 1


@pytest.mark.anyio
async def test_progress_reporter_preserves_real_caller_cancellation() -> None:
    entered = asyncio.Event()

    async def blocked_callback(completed: float, total: float, message: str) -> None:
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(ProgressReporter(blocked_callback).report(0, 1, "starting"))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
