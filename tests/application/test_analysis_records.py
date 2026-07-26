from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from flameox.analysis import FailureAnalysisResult, RecipeService
from flameox.application import (
    AnalysisMaterializationService,
    MaterializeAnalysisRequest,
)
from flameox.catalog import Catalog, Snapshot
from flameox.storage import Workspace


@pytest.mark.anyio
async def test_materialized_analysis_cancellation_interrupts_duckdb_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    original_head = workspace.corpus.read_head().commit_id
    query_started = threading.Event()
    query_released = threading.Event()
    interrupt_attempted = threading.Event()
    original_failures = RecipeService.failures
    original_interrupt = Snapshot.interrupt

    def tracked_interrupt(snapshot: Snapshot) -> None:
        interrupt_attempted.set()
        original_interrupt(snapshot)

    def slow_failures(
        service: RecipeService,
        *,
        limit: int | None = None,
        corpus_commit_id: str | None = None,
    ) -> FailureAnalysisResult:
        assert service.snapshot is not None
        query_started.set()
        assert query_released.wait(timeout=5)
        service.snapshot.execute("SELECT sum(sin(i)) FROM range(100000000000) values(i)").fetchall()
        return original_failures(
            service,
            limit=limit,
            corpus_commit_id=corpus_commit_id,
        )

    monkeypatch.setattr(RecipeService, "failures", slow_failures)
    monkeypatch.setattr(Snapshot, "interrupt", tracked_interrupt)
    task = asyncio.create_task(
        AnalysisMaterializationService(workspace).record_async(
            MaterializeAnalysisRequest(recipe="failures")
        )
    )
    assert await asyncio.to_thread(query_started.wait, 5)
    task.cancel()
    assert await asyncio.to_thread(interrupt_attempted.wait, 5)
    query_released.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert workspace.corpus.read_head().commit_id == original_head
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute("SELECT count(*) FROM analyses").fetchone() == (0,)
