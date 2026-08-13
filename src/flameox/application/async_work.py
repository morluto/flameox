from __future__ import annotations

import asyncio
from collections.abc import Callable

import anyio


async def run_atomic_thread[T](
    operation: Callable[[], T],
) -> T:
    """Run finite local mutation work and retain ownership through completion."""

    async def run_worker() -> T:
        return await anyio.to_thread.run_sync(operation, abandon_on_cancel=False)

    worker = asyncio.create_task(run_worker())
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(worker)
        except asyncio.CancelledError as error:
            # A raw asyncio Task.cancel() bypasses AnyIO cancel-scope shielding.
            # Suppress it until the non-cancellable mutation has produced its exact
            # receipt or error; repeated cancellation must not orphan the worker.
            cancellation = cancellation or error
            if worker.done():
                worker.result()
                raise cancellation from None
        else:
            if cancellation is not None:
                raise cancellation
            return result
