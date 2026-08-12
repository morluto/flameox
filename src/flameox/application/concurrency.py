from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import cast

import anyio


async def race_with_cancellation[T](
    work: Awaitable[T],
    wait_for_cancellation: Callable[[], Awaitable[object]],
    cancellation_error: Callable[[], BaseException],
) -> T:
    """Own, cancel, and join one operation and its cancellation watcher."""

    missing = object()
    result: T | object = missing
    work_error: BaseException | None = None
    cancellation_requested = False

    async with anyio.create_task_group() as tasks:

        async def run_work() -> None:
            nonlocal result, work_error
            try:
                result = await work
            except BaseException as exc:
                work_error = exc
            finally:
                tasks.cancel_scope.cancel()

        async def watch_cancellation() -> None:
            nonlocal cancellation_requested
            await wait_for_cancellation()
            cancellation_requested = True
            tasks.cancel_scope.cancel()

        tasks.start_soon(run_work, name="flameox-scoped-work")
        tasks.start_soon(watch_cancellation, name="flameox-cancellation-watcher")

    if cancellation_requested:
        raise cancellation_error()
    if work_error is not None:
        if isinstance(work_error, asyncio.CancelledError):
            raise work_error
        raise work_error
    if result is missing:
        raise RuntimeError("structured task scope exited without a result")
    return cast(T, result)
