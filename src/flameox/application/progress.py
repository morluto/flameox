from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

type ProgressCallback = Callable[[float, float, str], Awaitable[None]]


class ProgressReporter:
    """Best-effort observer that cannot decide a domain operation's outcome."""

    def __init__(self, callback: ProgressCallback | None) -> None:
        self._callback = callback
        self.dropped_count = 0

    async def report(self, completed: float, total: float, message: str) -> None:
        callback = self._callback
        if callback is None:
            return
        try:
            await callback(completed, total, message)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            self._drop()
        except Exception:
            self._drop()

    def _drop(self) -> None:
        self.dropped_count += 1
        self._callback = None
        logging.getLogger("flameox.progress").warning(
            "Progress delivery failed; domain work will continue without notifications."
        )
