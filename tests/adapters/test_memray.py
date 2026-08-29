from __future__ import annotations

import asyncio
import ctypes
import mmap
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from flameox.action_graph import ActionId, tool_action
from flameox.adapters import MemrayExtractor
from flameox.adapters.memray import memray_extraction_limits
from flameox.analysis import RecipeService
from flameox.application import DrilldownService, ImportArtifactRequest, ImportService
from flameox.catalog import Catalog
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.evidence import GenerationPublisher
from flameox.storage import ArtifactStore, GenerationManifest, Workspace

pytestmark = [pytest.mark.integration, pytest.mark.optional, pytest.mark.requires_memray]


def _memray_module() -> ModuleType:
    return cast(
        ModuleType,
        pytest.importorskip("memray", reason="optional provider unavailable: install memray"),
    )


def _extractor(
    workspace: Workspace,
    version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> MemrayExtractor:
    extractor = MemrayExtractor(workspace)
    runtime = SimpleNamespace(
        python=Path(sys.executable),
        receipt=SimpleNamespace(
            environment_id="sha256:" + "e" * 64,
            distributions={"memray": version},
            limitations=(),
        ),
    )
    monkeypatch.setattr(
        extractor.provider_runtimes,
        "find_distribution",
        lambda **_kwargs: runtime,
    )
    monkeypatch.setattr(
        extractor.provider_runtimes,
        "verified_use",
        lambda _runtime: nullcontext(runtime),
    )
    return extractor


def _exercise_allocator_case(case: str) -> None:
    libc = ctypes.CDLL(None)
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.malloc.restype = ctypes.c_void_p
    libc.realloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.realloc.restype = ctypes.c_void_p
    libc.free.argtypes = [ctypes.c_void_p]
    if case == "malloc_free":
        for _ in range(10):
            pointer = libc.malloc(4_096)
            libc.free(pointer)
    elif case == "realloc":
        pointer = libc.malloc(1_024)
        pointer = libc.realloc(pointer, 8_192)
        libc.free(pointer)
    elif case == "mmap_munmap":
        mapping = mmap.mmap(-1, 4_096)
        mapping.close()
    else:
        allocations = [bytearray(64) for _ in range(100)]
        del allocations


def _allocation_a() -> bytearray:
    return bytearray(1_000)


def _allocation_b() -> bytearray:
    return bytearray(2_000)


def _nested_allocation_leaf() -> bytearray:
    return bytearray(100_000)


def _nested_allocation_parent() -> bytearray:
    return _nested_allocation_leaf()


@pytest.mark.anyio
async def test_memray_extractor_preserves_native_capture_and_names_memory_concepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memray = _memray_module()
    capture = tmp_path / "memory.bin"
    with memray.Tracker(str(capture)):
        retained = _nested_allocation_parent()
    assert len(retained) == 100_000
    from memray._memray import compute_statistics

    provider_stats = compute_statistics(str(capture), report_progress=False, num_largest=1)

    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
            producer="memray",
            producer_version=memray.__version__,
        )
    )
    progress: list[tuple[str, int, int | None]] = []

    async def record_progress(phase: str, completed: int, total: int | None) -> None:
        progress.append((phase, completed, total))

    result = await _extractor(workspace, memray.__version__, monkeypatch).extract(
        imported.run.run_id,
        limits=memray_extraction_limits(workspace),
        progress=record_progress,
    )

    assert result.peak_memory_bytes >= 100_000
    assert result.retained_end_bytes >= 100_000
    assert result.allocation_operations == provider_stats.total_num_allocations
    assert result.total_allocated_bytes == provider_stats.total_memory_allocated
    assert result.capture_records >= result.allocation_operations
    assert result.coverage.frames_published >= 1
    assert result.producer_version == memray.__version__
    assert result.reader_version == memray.__version__
    assert result.reader_environment_id == "sha256:" + "e" * 64
    phases = {phase for phase, _completed, _total in progress}
    assert {"reading_profile", "publishing_evidence"} <= phases
    assert phases <= {
        "reading_profile",
        "computing_statistics",
        "normalizing_high_watermark",
        "normalizing_retained_end",
        "normalizing_allocation_volume",
        "normalizing_temporary",
        "writing_evidence",
        "publishing_evidence",
    }
    with Catalog(workspace).open_snapshot() as snapshot:
        measurements = snapshot.execute(
            "SELECT name, value_int, unit FROM measurements ORDER BY name"
        ).fetchall()
        frames = snapshot.execute(
            "SELECT metric, sum(self_value), sum(inclusive_value) "
            "FROM frame_measurements GROUP BY metric ORDER BY metric"
        ).fetchall()
        edge = snapshot.execute(
            "SELECT parent_frame_id, child_frame_id, metric, weight_value, unit "
            "FROM call_edges WHERE run_id = ? ORDER BY weight_value DESC LIMIT 1",
            (imported.run.run_id,),
        ).fetchone()
        stack = snapshot.execute(
            "SELECT leaf_frame_id, metric, weight_value, unit, frame_ids "
            "FROM stacks WHERE run_id = ? ORDER BY weight_value DESC LIMIT 1",
            (imported.run.run_id,),
        ).fetchone()
    assert ("memory.peak", result.peak_memory_bytes, "bytes") in measurements
    assert ("memory.retained_end", result.retained_end_bytes, "bytes") in measurements
    assert (
        "memory.allocation_operations",
        provider_stats.total_num_allocations,
        "count",
    ) in measurements
    assert (
        "memory.allocated_bytes",
        provider_stats.total_memory_allocated,
        "bytes",
    ) in measurements
    assert ("memory.capture_records", result.capture_records, "count") in measurements
    assert {row[0] for row in frames} == {
        "memory.allocated",
        "memory.high_watermark",
        "memory.retained_end",
    }
    assert result.coverage.allocation_volume.status == "available"
    assert result.coverage.temporary.status == "available"
    assert result.temporary_allocation_threshold == 1
    assert edge is not None
    assert stack is not None
    assert edge[2] in {"memory.allocated", "memory.high_watermark", "memory.retained_end"}
    assert edge[3] > 0
    assert edge[4] == "bytes"
    assert stack[1] in {"memory.allocated", "memory.high_watermark", "memory.retained_end"}
    assert stack[2] > 0
    assert stack[3] == "bytes"
    assert stack[4][-1] == stack[0]
    drilldown = DrilldownService(workspace)
    callers = drilldown.callers(
        imported.run.run_id,
        str(edge[1]),
        metric=str(edge[2]),
    )
    callees = drilldown.callees(
        imported.run.run_id,
        str(edge[0]),
        metric=str(edge[2]),
    )
    examples = drilldown.examples(
        imported.run.run_id,
        str(stack[0]),
        metric=str(stack[1]),
    )
    assert callers.frames[0].frame_id == edge[0]
    assert callees.frames[0].frame_id == edge[1]
    assert callers.frames[0].unit == callees.frames[0].unit == "bytes"
    assert examples.examples[0].frames[-1].frame_id == stack[0]
    assert examples.examples[0].unit == "bytes"
    absent = drilldown.callers(
        imported.run.run_id,
        str(edge[1]),
        metric="memory.unavailable",
    )
    assert absent.returned == 0
    assert absent.recovery == tool_action(
        ActionId.GET_NATIVE_VIEWER_PLAN,
        artifact_id=imported.artifact_id,
    )
    assert result.recovery is None
    assert "The capture does not contain native stack traces." in result.limitations
    pinned_commit_id = workspace.corpus.read_head().commit_id
    GenerationPublisher(workspace).publish_rows(
        {
            "frames": [
                {
                    "frame_id": "later-frame",
                    "language": "python",
                    "function": "published_later",
                    "module": None,
                    "file": "later.py",
                    "line": 1,
                    "column": None,
                    "address": None,
                    "build_id": None,
                    "module_relative_address": None,
                    "inline_chain_id": None,
                    "source_state_id": None,
                    "artifact_id": imported.artifact_id,
                    "inlined": False,
                    "symbolization": "complete",
                }
            ],
            "frame_measurements": [
                {
                    "run_id": imported.run.run_id,
                    "artifact_id": imported.artifact_id,
                    "frame_id": "later-frame",
                    "metric": "memory.retained_end",
                    "self_value": 1,
                    "inclusive_value": 1,
                    "unit": "bytes",
                    "sample_count": 1,
                    "thread_name": None,
                    "process_name": None,
                    "phase": None,
                }
            ],
        },
        publisher="pinned-recipe-regression",
        publisher_version="1",
        input_run_ids=(imported.run.run_id,),
        input_artifact_ids=(imported.artifact_id,),
    )

    analysis = RecipeService(
        workspace,
        snapshot_handle=Catalog(workspace).pin(pinned_commit_id),
    ).memory(
        imported.run.run_id,
        corpus_commit_id=pinned_commit_id,
    )
    assert analysis.corpus_commit_id == pinned_commit_id
    assert {item.name for item in analysis.measurements} == {
        "memory.allocated_bytes",
        "memory.allocation_operations",
        "memory.capture_records",
            "memory.peak",
            "memory.retained_end",
            "memory.temporary",
        }
    assert analysis.hotspots
    assert all(item.frame_id != "later-frame" for item in analysis.hotspots)


@pytest.mark.parametrize(
    ("case", "trace_python_allocators"),
    (
        ("malloc_free", False),
        ("realloc", False),
        ("mmap_munmap", False),
        ("python_allocators", True),
    ),
)
@pytest.mark.anyio
async def test_memray_allocation_operations_match_provider_stats_and_exclude_deallocations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    trace_python_allocators: bool,
) -> None:
    memray = _memray_module()
    from memray._memray import compute_statistics

    capture = tmp_path / "memory.bin"
    with memray.Tracker(
        str(capture),
        trace_python_allocators=trace_python_allocators,
    ):
        _exercise_allocator_case(case)
    provider_stats = compute_statistics(str(capture), report_progress=False, num_largest=1)
    provider_records = memray.FileReader(str(capture)).metadata.total_allocations
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
            producer="memray",
            producer_version=memray.__version__,
        )
    )

    result = await _extractor(workspace, memray.__version__, monkeypatch).extract(
        imported.run.run_id, limits=memray_extraction_limits(workspace)
    )

    assert result.allocation_operations == provider_stats.total_num_allocations
    assert result.total_allocated_bytes == provider_stats.total_memory_allocated
    assert result.capture_records == provider_records
    assert result.allocation_operations is not None
    assert result.capture_records > result.allocation_operations


@pytest.mark.anyio
async def test_memray_extraction_rejects_a_reader_that_disagrees_with_its_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memray = _memray_module()
    capture = tmp_path / "memory.bin"
    with memray.Tracker(str(capture)):
        bytearray(1_000)
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
            producer="memray",
            producer_version=memray.__version__,
        )
    )
    original_head = workspace.corpus.read_head().commit_id

    with pytest.raises(DomainError) as raised:
        await _extractor(workspace, "0.0", monkeypatch).extract(
            imported.run.run_id, limits=memray_extraction_limits(workspace)
        )

    assert raised.value.code is ErrorCode.ADAPTER_INCOMPATIBLE
    assert raised.value.details["reader_version"] == "0.0"
    assert workspace.corpus.read_head().commit_id == original_head


@pytest.mark.anyio
async def test_memray_aggregated_capture_does_not_invent_allocation_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memray = _memray_module()
    capture = tmp_path / "memory.bin"
    with memray.Tracker(
        str(capture),
        file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS,
    ):
        bytearray(1_000)
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
            producer="memray",
            producer_version=memray.__version__,
        )
    )

    result = await _extractor(workspace, memray.__version__, monkeypatch).extract(
        imported.run.run_id, limits=memray_extraction_limits(workspace)
    )

    assert result.allocation_operations is None
    assert result.total_allocated_bytes is None
    assert result.temporary_allocated_bytes is None
    assert result.coverage.allocation_volume.status == "unavailable"
    assert result.coverage.temporary.status == "unavailable"
    assert result.capture_records >= 1
    assert any(
        "structured allocation statistics are unavailable" in item for item in result.limitations
    )
    assert any("temporary-allocation record stream" in item for item in result.limitations)


@pytest.mark.anyio
async def test_memray_limits_report_bounded_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memray = _memray_module()
    capture = tmp_path / "memory.bin"
    with memray.Tracker(str(capture)):
        retained = (_allocation_a(), _allocation_b())
    assert sum(map(len, retained)) == 3_000
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
            producer="memray",
            producer_version=memray.__version__,
        )
    )
    limits = memray_extraction_limits(workspace).validated_copy(
        update={
            "max_provider_records": 1,
            "max_frames": 1,
            "max_stack_depth": 1,
            "max_aggregate_rows": 2,
        }
    )
    extractor = _extractor(workspace, memray.__version__, monkeypatch)

    first = await extractor.extract(imported.run.run_id, limits=limits)
    assert first.coverage.complete is False
    assert any(
        metric.records_seen > metric.records_selected
        for metric in (first.coverage.high_watermark, first.coverage.retained_end)
    )
    assert first.coverage.frames_published <= 1
    assert first.coverage.aggregate_rows_published <= 2
    assert any("reached an extraction limit" in item for item in first.limitations)
    assert first.recovery == tool_action(
        ActionId.GET_NATIVE_VIEWER_PLAN,
        artifact_id=imported.artifact_id,
    )
    analysis = RecipeService(workspace).memory(imported.run.run_id)
    if not analysis.hotspots:
        assert analysis.hotspot_evidence.status == "unavailable"
        assert analysis.hotspot_evidence.next_action is not None
        assert analysis.hotspot_evidence.next_action.action is ActionId.EXTRACT_MEMRAY
    second = await extractor.extract(
        imported.run.run_id,
        limits=memray_extraction_limits(workspace),
    )
    active_generations = [
        GenerationManifest.model_validate_json(
            (workspace.paths.root / relative_path).read_text()
        )
        for relative_path in workspace.corpus.read_head().generation_manifests
    ]
    assert [item.generation_id for item in active_generations if item.publisher == "memray"] == [
        second.evidence_generation_id
    ]
    preserved = ArtifactStore(workspace).get(imported.artifact_id).payload_path
    assert capture.read_bytes() == preserved.read_bytes()


@pytest.mark.parametrize("payload", (b"", b"truncated"))
@pytest.mark.anyio
async def test_memray_rejects_empty_and_truncated_captures(
    tmp_path: Path,
    payload: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    capture = tmp_path / "memory.bin"
    capture.write_bytes(payload)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
            producer="memray",
            producer_version="1.20.0",
        )
    )

    with pytest.raises(DomainError) as error:
        await _extractor(workspace, "1.20.0", monkeypatch).extract(
            imported.run.run_id, limits=memray_extraction_limits(workspace)
        )

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


@pytest.mark.anyio
async def test_memray_cancellation_before_publication_preserves_the_corpus_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memray = _memray_module()
    capture = tmp_path / "memory.bin"
    with memray.Tracker(str(capture)):
        retained = [bytearray(64) for _ in range(100_000)]
    assert retained

    workspace = Workspace.initialize(tmp_path)
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=capture,
            kind=ArtifactKind.MEMORY_PROFILE,
            producer="memray",
            producer_version=memray.__version__,
        )
    )
    head_before = workspace.corpus.read_head().commit_id
    worker_started = asyncio.Event()

    async def observe_progress(phase: str, _completed: int, _total: int | None) -> None:
        if phase not in {"reading_profile", "publishing_evidence"}:
            worker_started.set()

    extraction = asyncio.create_task(
        _extractor(workspace, memray.__version__, monkeypatch).extract(
            imported.run.run_id,
            limits=memray_extraction_limits(workspace),
            progress=observe_progress,
        )
    )
    await asyncio.wait_for(worker_started.wait(), timeout=10)
    extraction.cancel()

    with pytest.raises(asyncio.CancelledError):
        await extraction
    assert workspace.corpus.read_head().commit_id == head_before
    assert ArtifactStore(workspace).get(imported.artifact_id).payload_path.is_file()
    worker_staging = workspace.paths.staging / "artifact-workers"
    assert not worker_staging.exists() or tuple(worker_staging.iterdir()) == ()
