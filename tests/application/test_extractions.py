from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from flameox.adapters.memray import MemrayExtractionResult, MemrayExtractor
from flameox.application.extractions import ExtractionManager
from flameox.domain import DomainError
from flameox.storage import Workspace

pytestmark = pytest.mark.unit


def _result(run_id: str) -> MemrayExtractionResult:
    return MemrayExtractionResult(
        run_id=run_id,
        artifact_id="artifact-fixture",
        peak_memory_bytes=100,
        retained_end_bytes=40,
        total_allocations=3,
        frame_count=2,
        corpus_commit_id="commit-fixture",
    )


@pytest.mark.anyio
async def test_memray_extraction_is_durable_reconnectable_and_idempotent(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    class FakeExtractor(MemrayExtractor):
        def __init__(self, workspace: Workspace) -> None:
            del workspace

        def extract(
            self,
            run_id: str,
            *,
            cancel_check: Callable[[], None] | None = None,
            progress: Callable[[str, int, int | None], None] | None = None,
        ) -> MemrayExtractionResult:
            del cancel_check
            assert progress is not None
            progress("reading_high_watermark", 1_024, None)
            progress("aggregating_high_watermark", 5, 10)
            return _result(run_id)

    manager = ExtractionManager(
        workspace,
        memray_factory=FakeExtractor,
    )
    try:
        started = await manager.start_memray("run-fixture", "stable-request")
        replay = await manager.start_memray("run-fixture", "stable-request")
        assert replay.operation_id == started.operation_id

        terminal = await manager.runner.wait(started.operation_id, timeout_seconds=2)
        reconnected = await manager.status(started.operation_id)

        assert terminal.state == "terminal"
        assert reconnected == terminal
        reading, aggregation = terminal.progress[-2:]
        assert reading.phase == "reading_high_watermark"
        assert reading.completed is None
        assert reading.total is None
        assert "1024 records" in reading.message
        assert aggregation.phase == "aggregating_high_watermark"
        assert aggregation.completed == 5
        assert aggregation.total == 10
        assert terminal.terminal_receipt is not None
        assert terminal.terminal_receipt["extraction"]["corpus_commit_id"] == "commit-fixture"
    finally:
        await manager.shutdown()


@pytest.mark.anyio
async def test_memray_extraction_cancel_returns_before_cleanup_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    entered = threading.Event()
    release_cleanup = threading.Event()

    class BlockedExtractor(MemrayExtractor):
        def __init__(self, workspace: Workspace) -> None:
            del workspace

        def extract(
            self,
            run_id: str,
            *,
            cancel_check: Callable[[], None] | None = None,
            progress: Callable[[str, int, int | None], None] | None = None,
        ) -> MemrayExtractionResult:
            del run_id, progress
            assert cancel_check is not None
            entered.set()
            while True:
                try:
                    cancel_check()
                except DomainError:
                    assert release_cleanup.wait(timeout=2)
                    raise
                threading.Event().wait(0.01)

    manager = ExtractionManager(
        workspace,
        memray_factory=BlockedExtractor,
    )
    try:
        started = await manager.start_memray("run-fixture", "cancel-request")
        assert await asyncio.to_thread(entered.wait, 1)

        cancelling = await manager.cancel(started.operation_id)
        assert cancelling.state == "running"
        assert cancelling.phase == "cancelling"
        assert cancelling.cancellation_requested is True
        assert cancelling.cleanup_status == "pending"

        release_cleanup.set()
        terminal = await manager.runner.wait(started.operation_id, timeout_seconds=2)
        assert terminal.state == "cancelled"
        assert terminal.cleanup_status == "complete"
        assert terminal.terminal_receipt is None
    finally:
        release_cleanup.set()
        await manager.shutdown()
