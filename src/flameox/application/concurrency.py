from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import cast

import anyio

from flameox.action_graph import NextAction
from flameox.domain import DomainError, ErrorCode
from flameox.execution import ExecutionOutcome, ExecutionRequest, SubprocessBroker


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


async def _run_brokered(
    broker: SubprocessBroker,
    request: ExecutionRequest,
    *,
    cancel_event: threading.Event | None,
    cancellation_message: str,
    cancellation_details: dict[str, str],
    cancellation_next_action: NextAction,
) -> ExecutionOutcome:
    if cancel_event is None:
        return await broker.run(request)
    return await race_with_cancellation(
        broker.run(request),
        lambda: _wait_for_cancellation(cancel_event),
        lambda: DomainError(
            ErrorCode.PROCESS_CANCELLED,
            cancellation_message,
            retryable=True,
            details=cancellation_details,
            next_action=cancellation_next_action,
        ),
    )


async def _wait_for_cancellation(cancel_event: threading.Event) -> None:
    while not cancel_event.is_set():
        await asyncio.sleep(0.05)


def run_brokered_from_worker(
    broker: SubprocessBroker,
    request: ExecutionRequest,
    *,
    cancel_event: threading.Event | None,
    cancellation_message: str,
    cancellation_details: dict[str, str],
    cancellation_next_action: NextAction,
) -> ExecutionOutcome:
    """Run brokered setup work from its synchronous worker thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("brokered synchronous setup must run in a worker thread")
    return asyncio.run(
        _run_brokered(
            broker,
            request,
            cancel_event=cancel_event,
            cancellation_message=cancellation_message,
            cancellation_details=cancellation_details,
            cancellation_next_action=cancellation_next_action,
        )
    )


def process_output_detail(outcome: ExecutionOutcome) -> str:
    return (
        outcome.stderr.decode("utf-8", errors="replace").strip()
        or outcome.stdout.decode("utf-8", errors="replace").strip()
    )
