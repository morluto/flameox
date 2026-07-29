from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress

import anyio


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
        deadline = asyncio.get_running_loop().time() + cleanup_timeout_seconds
        with anyio.CancelScope(shield=True):
            while not task.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait({task}, timeout=remaining)
                except asyncio.CancelledError:
                    # asyncio cancellation is edge-triggered and bypasses an
                    # AnyIO shield. Keep the durability barrier intact when a
                    # caller repeats cancellation while the thread settles.
                    continue
        if task.done():
            _consume_completion(task)
        else:
            task.add_done_callback(_consume_completion)
        raise cancellation
