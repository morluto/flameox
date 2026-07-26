from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress


def _consume_completion(task: asyncio.Task[object]) -> None:
    with suppress(asyncio.CancelledError):
        task.exception()


async def run_atomic_thread[T](
    operation: Callable[[], T],
    *,
    cleanup_timeout_seconds: float = 35,
) -> T:
    """Run finite local mutation work without abandoning it on caller cancellation."""

    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        try:
            async with asyncio.timeout(cleanup_timeout_seconds):
                await asyncio.shield(asyncio.gather(task, return_exceptions=True))
        except TimeoutError:
            task.add_done_callback(_consume_completion)
        finally:
            raise cancellation
