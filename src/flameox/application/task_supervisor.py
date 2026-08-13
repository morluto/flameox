from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

import anyio


class TaskHandle(Protocol):
    @property
    def done(self) -> bool: ...

    def cancel(self) -> None: ...

    async def wait(self) -> None: ...


class _LocalTaskHandle:
    def __init__(self, task: asyncio.Task[None]) -> None:
        self._task = task

    @property
    def done(self) -> bool:
        return self._task.done()

    def cancel(self) -> None:
        self._task.cancel()

    async def wait(self) -> None:
        await asyncio.shield(asyncio.gather(self._task, return_exceptions=True))


class SupervisedTask:
    def __init__(self) -> None:
        self._scope: anyio.CancelScope | None = None
        self._cancel_requested = False
        self._done = anyio.Event()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def cancel(self) -> None:
        self._cancel_requested = True
        if self._scope is not None:
            self._scope.cancel()

    async def wait(self) -> None:
        await self._done.wait()

    def _started(self, scope: anyio.CancelScope) -> None:
        self._scope = scope
        if self._cancel_requested:
            scope.cancel()

    def _finished(self) -> None:
        self._scope = None
        self._done.set()


class TaskSupervisor:
    """One lifespan-owned parent for application background tasks."""

    def __init__(self, task_group: anyio.abc.TaskGroup) -> None:
        self._task_group = task_group

    def start(
        self,
        function: Callable[[], Awaitable[None]],
        *,
        name: str,
    ) -> SupervisedTask:
        handle = SupervisedTask()
        self._task_group.start_soon(self._run, handle, function, name=name)
        return handle

    @staticmethod
    async def _run(
        handle: SupervisedTask,
        function: Callable[[], Awaitable[None]],
    ) -> None:
        try:
            with anyio.CancelScope() as scope:
                handle._started(scope)
                await function()
        finally:
            handle._finished()


def start_local_task(
    function: Callable[[], Awaitable[None]],
    *,
    name: str,
) -> TaskHandle:
    """Compatibility owner for non-lifespan callers such as direct application tests."""

    async def invoke() -> None:
        await function()

    return _LocalTaskHandle(asyncio.create_task(invoke(), name=name))
