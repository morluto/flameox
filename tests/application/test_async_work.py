from __future__ import annotations

import asyncio
import threading

import pytest

from flameox.application.async_work import run_atomic_thread


@pytest.mark.anyio
async def test_atomic_thread_settles_before_repeated_cancellation_returns() -> None:
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def mutation() -> None:
        started.set()
        release.wait()
        completed.set()

    task = asyncio.create_task(run_atomic_thread(mutation))
    assert await asyncio.to_thread(started.wait, 1)

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()

    done, _ = await asyncio.wait({task}, timeout=0.1)
    try:
        assert not done
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed.is_set()
