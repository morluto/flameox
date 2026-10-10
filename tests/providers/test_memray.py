from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    CaptureTarget,
    PathSource,
    RequestLimits,
)
from flameox.workers.harness import IsolatedWorkerHarness, WorkerRuntimeConfig
from flameox.workers.memray_contract import (
    MEMRAY_WORKER,
    MemrayExtractionLimits,
    MemrayWorkerRequest,
)


@pytest.mark.integration
@pytest.mark.requires_memray
def test_memray_native_profile_attributes_and_bounds_nested_allocations(tmp_path: Path) -> None:
    memray = pytest.importorskip("memray")
    capture = tmp_path / "nested-memory.bin"

    def allocate_leaf() -> list[bytearray]:
        return [bytearray(8_192) for _ in range(4)]

    def call_leaf() -> list[bytearray]:
        return allocate_leaf()

    with memray.Tracker(str(capture)):
        retained = call_leaf()

    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "memory.hotspots",
            [PathSource(path=str(capture), format="memray", producer="memray")],
            {},
            limits=RequestLimits(max_rows=1),
        )
    finally:
        runtime.close()

    assert retained
    assert result["blocks"][0]["values"]["total_allocated_bytes"] > 0
    row = result["blocks"][1]["rows"][0]
    assert row["function"] == "allocate_leaf"
    assert row["inclusive_bytes"] >= row["self_bytes"]
    assert row["allocation_count"] == 4
    assert result["coverage"]["rows_returned"] == 1
    assert result["coverage"]["rows_observed"] > 1
    assert result["coverage"]["complete"] is False
    assert result["truncation"]["reason"] == "row_limit"
    assert any("bounded" in item for item in result["limitations"])


@pytest.mark.process
@pytest.mark.requires_memray
def test_direct_memray_capture_uses_typed_argv_and_preserves_native_output(
    tmp_path: Path,
) -> None:
    pytest.importorskip("memray")

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "retained = [bytearray(4096) for _ in range(8)]"],
                    cwd=str(tmp_path),
                    provider_id="memray",
                    capture_arguments={"native": False},
                ),
                "memory.hotspots",
                limits=RequestLimits(max_rows=10),
            )
            preserved = runtime.preserve_evidence(result["analysis_id"])
            manifest = runtime.read_evidence(preserved["evidence_id"])
        finally:
            runtime.close()

        execution = result["capture"]["executions"][0]
        assert execution["capture_argv"][1:4] == ["-m", "memray", "run"]
        assert result["provider"]["id"] == "memray"
        assert {item["role"] for item in manifest["body"]["artifacts"]} == {
            "capture-0001/memory",
        }

    asyncio.run(exercise())


@pytest.mark.process
@pytest.mark.requires_memray
def test_memray_worker_emits_only_frames_referenced_by_bounded_measurements(tmp_path: Path) -> None:
    memray = pytest.importorskip("memray")
    capture = tmp_path / "many-frames.bin"
    allocators: list[Callable[[], bytearray]] = []
    for index in range(100):
        namespace: dict[str, object] = {}
        exec(
            compile("def allocate(): return bytearray(8192)", f"frame{index}.py", "exec"), namespace
        )
        allocator = namespace["allocate"]
        assert callable(allocator)
        allocators.append(allocator)
    with memray.Tracker(str(capture)):
        retained = [allocate() for allocate in allocators]
    assert len(retained) == 100
    request = MemrayWorkerRequest(
        artifact_path=str(capture),
        limits=MemrayExtractionLimits(
            max_input_bytes=1 << 20,
            max_provider_records=10_000,
            max_frames=3,
            max_stack_depth=100,
            max_aggregate_rows=4,
            max_output_bytes=1 << 20,
        ),
    )
    harness = IsolatedWorkerHarness(WorkerRuntimeConfig(tmp_path, tmp_path, tmp_path))
    with harness.run_typed_sync_session(MEMRAY_WORKER, request) as (result, job_root):
        frames = pq.read_table(job_root / "frames.parquet").to_pylist()
        measurements = pq.read_table(job_root / "frame_measurements.parquet").to_pylist()
        assert 0 < len(frames) <= 3
        assert 0 < len(measurements) <= 4
        assert {row["frame_id"] for row in frames} == {row["frame_id"] for row in measurements}
        assert result.coverage.frame_contributions_dropped > 0
