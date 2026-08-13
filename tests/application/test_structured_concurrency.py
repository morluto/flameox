from __future__ import annotations

import asyncio

import pytest

from flameox.application.concurrency import race_with_cancellation
from flameox.domain import DomainError, ErrorCode

pytestmark = pytest.mark.unit


@pytest.mark.anyio
async def test_race_scope_cancels_and_joins_losing_task() -> None:
    cancellation = asyncio.Event()
    worker_finished = asyncio.Event()

    async def worker() -> str:
        try:
            await asyncio.Future()
        finally:
            worker_finished.set()
        return "unreachable"

    cancellation.set()
    with pytest.raises(DomainError) as stopped:
        await race_with_cancellation(
            worker(),
            cancellation.wait,
            lambda: DomainError(ErrorCode.PROCESS_CANCELLED, "cancelled"),
        )

    assert stopped.value.code is ErrorCode.PROCESS_CANCELLED
    assert worker_finished.is_set()


@pytest.mark.anyio
async def test_race_scope_returns_work_and_joins_cancellation_watcher() -> None:
    cancellation = asyncio.Event()

    result = await race_with_cancellation(
        _return_value(),
        cancellation.wait,
        lambda: DomainError(ErrorCode.PROCESS_CANCELLED, "cancelled"),
    )

    assert result == "complete"


async def _return_value() -> str:
    await asyncio.sleep(0)
    return "complete"
