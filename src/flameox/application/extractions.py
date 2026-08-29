from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable

from flameox.action_graph import ActionId
from flameox.adapters.memray import MemrayExtractionResult, MemrayExtractor
from flameox.application.async_work import run_atomic_thread
from flameox.application.operations import OperationAdapter, OperationRunner, OperationStatus
from flameox.application.task_supervisor import TaskSupervisor
from flameox.domain import DomainError, ErrorCode
from flameox.storage import Workspace


class ExtractionManager:
    """Own durable lifecycle for provider extraction that may outlive one request."""

    _OPERATION = OperationAdapter(
        kind="evidence.extract.memray",
        start_action=ActionId.EXTRACT_MEMRAY,
        status_action=ActionId.GET_EXTRACTION,
    )

    def __init__(
        self,
        workspace: Workspace,
        *,
        supervisor: TaskSupervisor | None = None,
        memray_factory: Callable[[Workspace], MemrayExtractor] = MemrayExtractor,
    ) -> None:
        self.workspace = workspace
        self.memray_factory = memray_factory
        self.runner = OperationRunner(workspace, self._OPERATION, supervisor=supervisor)

    async def start_memray(self, run_id: str, idempotency_key: str) -> OperationStatus:
        return await self.runner.start(
            {"run_id": run_id},
            idempotency_key,
            self._run_memray,
            items=(run_id,),
        )

    async def status(self, operation_id: str) -> OperationStatus:
        return await self.runner.status(operation_id)

    async def cancel(self, operation_id: str) -> OperationStatus:
        return await self.runner.cancel(operation_id)

    async def shutdown(self) -> None:
        await self.runner.shutdown()

    async def _run_memray(
        self,
        operation_id: str,
        progress: Callable[[str, float | None, float | None, str], Awaitable[None]],
    ) -> dict[str, object]:
        cancel_event = threading.Event()
        self.runner.set_cancel_hook(operation_id, cancel_event.set)
        loop = asyncio.get_running_loop()

        def check_cancelled() -> None:
            if cancel_event.is_set():
                raise DomainError(ErrorCode.PROCESS_CANCELLED, "Memray extraction was cancelled.")

        def report(phase: str, completed: int, total: int | None) -> None:
            async def emit() -> None:
                await progress(
                    phase,
                    None if total is None else completed,
                    total,
                    (
                        f"Memray extraction {phase.replace('_', ' ')}: {completed} records."
                        if total is None
                        else f"Memray extraction {phase.replace('_', ' ')}: "
                        f"{completed} of {total} records."
                    ),
                )

            future = asyncio.run_coroutine_threadsafe(emit(), loop)
            future.result()

        try:
            extractor = self.memray_factory(self.workspace)
            result: MemrayExtractionResult = await run_atomic_thread(
                lambda: extractor.extract(
                    self._run_id(operation_id),
                    cancel_check=check_cancelled,
                    progress=report,
                )
            )
            return {"extraction": result.model_dump(mode="json")}
        finally:
            self.runner.clear_cancel_hook(operation_id)

    def _run_id(self, operation_id: str) -> str:
        record = self.runner.store.read(operation_id)
        run_id = record.request.get("run_id")
        if not isinstance(run_id, str):
            raise DomainError(
                ErrorCode.INTERNAL_ERROR,
                "Extraction operation does not contain a valid run identifier.",
            )
        return run_id
