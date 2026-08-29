from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from flameox.adapters.memray import MemrayExtractionResult, MemrayExtractor
from flameox.application.extractions import ExtractionManager
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace
from flameox.workers.memray_contract import (
    MEMRAY_WORKER,
    MemrayExtractionCoverage,
    MemrayExtractionLimits,
    MemrayMetricCoverage,
)

pytestmark = pytest.mark.unit


def _result(
    run_id: str,
    *,
    evidence_generation_id: str = "generation-fixture",
    corpus_commit_id: str = "commit-fixture",
) -> MemrayExtractionResult:
    limits = MemrayExtractionLimits(
        max_input_bytes=1_000,
        max_provider_records=100,
        max_frames=100,
        max_stack_depth=100,
        max_aggregate_rows=200,
        max_unique_edges=200,
        max_representative_stacks=100,
        max_output_bytes=1_000,
        wall_time_seconds=30,
        max_worker_memory_bytes=1_000_000,
    )
    coverage = MemrayExtractionCoverage(
        high_watermark=MemrayMetricCoverage(
            records_seen=1,
            records_selected=1,
            record_bytes_seen=100,
            record_bytes_selected=100,
            dropped_stack_frames=0,
            dropped_stack_frame_bytes=0,
        ),
        retained_end=MemrayMetricCoverage(
            records_seen=1,
            records_selected=1,
            record_bytes_seen=40,
            record_bytes_selected=40,
            dropped_stack_frames=0,
            dropped_stack_frame_bytes=0,
        ),
        frames_published=2,
        aggregate_rows_published=2,
        frame_contributions_dropped=0,
        frame_contribution_bytes_dropped=0,
        aggregate_rows_dropped=0,
        aggregate_inclusive_bytes_dropped=0,
        edge_rows_published=2,
        edge_rows_dropped=0,
        edge_weight_bytes_dropped=0,
        representative_stacks_published=1,
        representative_stacks_dropped=0,
        representative_stack_weight_bytes_dropped=0,
        output_bytes=100,
    )
    return MemrayExtractionResult(
        run_id=run_id,
        artifact_id="artifact-fixture",
        producer_version="1.20.0",
        reader_version="1.20.0",
        reader_environment_id="sha256:" + "e" * 64,
        extractor_profile=MEMRAY_WORKER.implementation,
        peak_memory_bytes=100,
        retained_end_bytes=40,
        allocation_operations=3,
        total_allocated_bytes=120,
        capture_records=5,
        limits=limits,
        coverage=coverage,
        evidence_generation_id=evidence_generation_id,
        corpus_commit_id=corpus_commit_id,
    )


@pytest.mark.anyio
async def test_memray_extraction_is_durable_reconnectable_and_idempotent(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    calls = 0
    published = GenerationPublisher(workspace).publish_rows(
        {"frames": []},
        publisher="test",
        publisher_version="1",
    )

    class FakeExtractor(MemrayExtractor):
        def __init__(self, workspace: Workspace) -> None:
            del workspace

        async def extract(
            self,
            run_id: str,
            *,
            limits: MemrayExtractionLimits,
            progress: Callable[[str, int, int | None], Awaitable[None]] | None = None,
        ) -> MemrayExtractionResult:
            nonlocal calls
            calls += 1
            assert progress is not None
            await progress("reading_high_watermark", 1_024, None)
            await progress("aggregating_high_watermark", 5, 10)
            return _result(
                run_id,
                evidence_generation_id=published.manifest.generation_id,
                corpus_commit_id=published.commit.commit_id,
            )

    manager = ExtractionManager(
        workspace,
        memray_factory=FakeExtractor,
    )
    try:
        started = await manager.start_memray("run-fixture", "stable-request")
        assert started.request["extractor_profile"] == MEMRAY_WORKER.implementation
        assert set(started.request["limits"]) == {
            "max_input_bytes",
            "max_provider_records",
            "max_frames",
            "max_stack_depth",
            "max_aggregate_rows",
            "max_unique_edges",
            "max_representative_stacks",
            "max_output_bytes",
            "wall_time_seconds",
            "max_worker_memory_bytes",
        }
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
        assert (
            terminal.terminal_receipt["extraction"]["corpus_commit_id"]
            == published.commit.commit_id
        )

        repeated = await manager.start_memray("run-fixture", "another-request")
        reused = await manager.runner.wait(repeated.operation_id, timeout_seconds=2)
        assert reused.terminal_receipt == terminal.terminal_receipt
        assert reused.progress[-1].phase == "reusing_completed_generation"
        assert calls == 1

        first_file = workspace.paths.root / published.manifest.files[0].path
        first_file.write_bytes(b"corrupt")
        replacement = await manager.start_memray("run-fixture", "corrupt-evidence-request")
        await manager.runner.wait(replacement.operation_id, timeout_seconds=2)
        assert calls == 2

        first_file.unlink()
        missing = await manager.start_memray("run-fixture", "missing-evidence-request")
        await manager.runner.wait(missing.operation_id, timeout_seconds=2)
        assert calls == 3
    finally:
        await manager.shutdown()


@pytest.mark.anyio
async def test_memray_extraction_cancel_returns_before_cleanup_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    entered = asyncio.Event()
    release_cleanup = asyncio.Event()

    class BlockedExtractor(MemrayExtractor):
        def __init__(self, workspace: Workspace) -> None:
            del workspace

        async def extract(
            self,
            run_id: str,
            *,
            limits: MemrayExtractionLimits,
            progress: Callable[[str, int, int | None], Awaitable[None]] | None = None,
        ) -> MemrayExtractionResult:
            del run_id, progress
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release_cleanup.wait()
                raise
            raise AssertionError("unreachable")

    manager = ExtractionManager(
        workspace,
        memray_factory=BlockedExtractor,
    )
    try:
        started = await manager.start_memray("run-fixture", "cancel-request")
        await asyncio.wait_for(entered.wait(), 1)

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
