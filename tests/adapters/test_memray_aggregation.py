from __future__ import annotations

import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pytest

from flameox.workers import memray as memray_worker
from flameox.workers.memray_contract import MemrayExtractionLimits


@dataclass(frozen=True)
class _AllocationRecord:
    size: int
    n_allocations: int
    stack: tuple[tuple[str, str, int], ...]

    def stack_trace(self) -> tuple[tuple[str, str, int], ...]:
        return self.stack


@dataclass(frozen=True)
class _AllocationRecordWithNativeFrames(_AllocationRecord):
    native_stack: tuple[tuple[str, str, int], ...]

    def native_stack_trace(self) -> tuple[tuple[str, str, int], ...]:
        return self.native_stack


def _state(tmp_path: Path, **overrides: int) -> memray_worker._AggregationState:
    values = {
        "max_input_bytes": 1_000_000,
        "max_provider_records": 1_000_000,
        "max_frames": 1_000,
        "max_stack_depth": 100,
        "max_aggregate_rows": 2_000,
        "max_unique_edges": 2_000,
        "max_representative_stacks": 1_000,
        "max_output_bytes": 16_000_000,
        "wall_time_seconds": 30,
        "max_worker_memory_bytes": 1_000_000,
        **overrides,
    }
    return memray_worker._AggregationState(
        limits=MemrayExtractionLimits.model_validate(values),
        run_id="run",
        artifact_id="artifact",
        workload_cwd=tmp_path,
        project_root=tmp_path,
        source_state_id=None,
        database_path=tmp_path / f"aggregation-{uuid4()}.sqlite",
    )


def test_memray_frame_identity_is_computed_once_across_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized: list[str] = []

    def normalize(
        filename: str,
        *,
        workload_cwd: Path,
        project_root: Path,
        source_state_id: str | None,
    ) -> tuple[str, str | None, str]:
        assert workload_cwd == tmp_path
        assert project_root == tmp_path
        assert source_state_id is None
        normalized.append(filename)
        return filename, None, "complete"

    monkeypatch.setattr(memray_worker, "_normalize", normalize)
    stack = (("allocate", "agent.py", 10), ("main", "runner.py", 20))
    records = [_AllocationRecord(size=5, n_allocations=1, stack=stack) for _ in range(100)]
    state = _state(tmp_path)

    for metric in ("memory.high_watermark", "memory.retained_end"):
        memray_worker._aggregate(
            records,
            metric=metric,
            state=state,
        )

    projection = state.finalize()
    state.close()
    assert normalized == ["agent.py", "runner.py"]
    assert len(projection.frame_rows) == 2
    assert len(projection.aggregates) == 4


@pytest.mark.performance
def test_memray_repeated_frame_aggregation_budget(tmp_path: Path) -> None:
    stack = (("allocate", "agent.py", 10), ("main", "runner.py", 20))
    record = _AllocationRecord(size=5, n_allocations=1, stack=stack)
    records = [record] * 500_000
    state = _state(tmp_path)

    started = perf_counter()
    memray_worker._aggregate(
        records,
        metric="memory.high_watermark",
        state=state,
    )
    elapsed = perf_counter() - started

    projection = state.finalize()
    state.close()
    assert len(projection.frame_rows) == len(projection.aggregates) == 2
    assert elapsed < 5


@pytest.mark.performance
def test_memray_high_cardinality_peak_memory_is_bound_by_limits(tmp_path: Path) -> None:
    def measure(record_count: int) -> tuple[int, float]:
        state = _state(
            tmp_path,
            max_provider_records=1_000,
            max_frames=1_000,
            max_aggregate_rows=1_000,
        )
        records = (
            _AllocationRecord(
                size=index + 1,
                n_allocations=1,
                stack=((f"allocate_{index}", f"module_{index}.py", index + 1),),
            )
            for index in range(record_count)
        )
        tracemalloc.start()
        started = perf_counter()
        _total, coverage = memray_worker._aggregate(
            records,
            metric="memory.high_watermark",
            state=state,
        )
        elapsed = perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert coverage.records_seen == record_count
        assert coverage.records_selected == 1_000
        projection = state.finalize()
        state.close()
        assert len(projection.frame_rows) == len(projection.aggregates) == 1_000
        return peak, elapsed

    modest_peak, modest_elapsed = measure(100_000)
    large_peak, large_elapsed = measure(1_000_000)

    assert large_peak <= modest_peak + 2 * 1024 * 1024
    assert modest_elapsed < 5
    assert large_elapsed < 30


def test_memray_aggregation_reports_record_frame_and_depth_coverage(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        max_provider_records=2,
        max_frames=1,
        max_stack_depth=1,
        max_aggregate_rows=1,
    )
    records = (
        _AllocationRecord(
            size=10,
            n_allocations=1,
            stack=(("first", "first.py", 1), ("caller", "caller.py", 2)),
        ),
        _AllocationRecord(
            size=20,
            n_allocations=1,
            stack=(("second", "second.py", 3),),
        ),
        _AllocationRecord(
            size=30,
            n_allocations=1,
            stack=(("third", "third.py", 4),),
        ),
    )

    total, coverage = memray_worker._aggregate(
        records,
        metric="memory.high_watermark",
        state=state,
    )

    assert total == 60
    assert coverage.model_dump() == {
        "records_seen": 3,
        "records_selected": 2,
        "record_bytes_seen": 60,
        "record_bytes_selected": 50,
        "dropped_stack_frames": 0,
        "dropped_stack_frame_bytes": 0,
    }
    assert coverage.complete is False
    projection = state.finalize()
    state.close()
    assert projection.frame_contributions_dropped == 1
    assert projection.frame_contribution_bytes_dropped == 20
    assert len(projection.frame_rows) == len(projection.aggregates) == 1
    assert projection.frame_rows[0]["function"] == "third"

    aggregate_state = _state(tmp_path, max_aggregate_rows=1)
    memray_worker._aggregate(
        records[:2],
        metric="memory.high_watermark",
        state=aggregate_state,
    )
    aggregate_projection = aggregate_state.finalize()
    aggregate_state.close()
    assert aggregate_projection.aggregate_rows_dropped == 2
    assert aggregate_projection.aggregate_inclusive_bytes_dropped == 20
    assert len(aggregate_projection.frame_rows) == 3
    assert len(aggregate_projection.aggregates) == 1


def test_memray_aggregate_limit_retains_top_inclusive_contributor(tmp_path: Path) -> None:
    state = _state(tmp_path, max_frames=1, max_aggregate_rows=1)
    records = (
        _AllocationRecord(
            size=100,
            n_allocations=1,
            stack=(("leaf_a", "a.py", 1), ("common", "common.py", 1)),
        ),
        _AllocationRecord(
            size=90,
            n_allocations=1,
            stack=(("leaf_b", "b.py", 1), ("common", "common.py", 1)),
        ),
    )

    memray_worker._aggregate(records, metric="memory.high_watermark", state=state)
    projection = state.finalize()
    state.close()

    assert len(projection.aggregates) == 1
    _metric, _frame_id, _self_value, inclusive, _samples = projection.aggregates[0]
    assert projection.frame_rows[0]["function"] == "common"
    assert inclusive == 190
    assert projection.frame_contributions_dropped == 2
    assert projection.frame_contribution_bytes_dropped == 190
    assert projection.aggregate_rows_dropped == 0


def test_memray_preserves_root_to_leaf_recursion_and_combines_identical_stacks(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    stack = (
        ("leaf", "work.py", 30),
        ("recursive", "work.py", 20),
        ("recursive", "work.py", 20),
        ("root", "work.py", 10),
    )
    records = (
        _AllocationRecord(size=40, n_allocations=2, stack=stack),
        _AllocationRecord(size=10, n_allocations=1, stack=stack),
    )

    memray_worker._aggregate(records, metric="memory.high_watermark", state=state)
    projection = state.finalize()
    state.close()

    functions = {row["frame_id"]: row["function"] for row in projection.frame_rows}
    assert len(projection.stack_rows) == 1
    representative = projection.stack_rows[0]
    assert [functions[frame_id] for frame_id in representative["frame_ids"]] == [
        "root",
        "recursive",
        "recursive",
        "leaf",
    ]
    assert representative["weight_value"] == 50
    assert representative["sample_count"] == 3
    edges = {
        (functions[row["parent_frame_id"]], functions[row["child_frame_id"]]): (
            row["weight_value"],
            row["sample_count"],
        )
        for row in projection.edge_rows
    }
    assert edges == {
        ("root", "recursive"): (50, 3),
        ("recursive", "recursive"): (50, 3),
        ("recursive", "leaf"): (50, 3),
    }


def test_memray_bounds_edges_and_representative_stacks_with_exact_drop_weight(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path, max_unique_edges=1, max_representative_stacks=1)
    records = (
        _AllocationRecord(
            size=100,
            n_allocations=2,
            stack=(("hot_leaf", "hot.py", 3), ("middle", "hot.py", 2), ("root", "hot.py", 1)),
        ),
        _AllocationRecord(
            size=10,
            n_allocations=1,
            stack=(("cold_leaf", "cold.py", 2), ("cold_root", "cold.py", 1)),
        ),
    )

    memray_worker._aggregate(records, metric="memory.retained_end", state=state)
    projection = state.finalize()
    state.close()

    assert len(projection.edge_rows) == 1
    assert projection.edge_rows[0]["weight_value"] == 100
    assert projection.edge_rows_dropped == 2
    assert projection.edge_weight_bytes_dropped == 110
    assert len(projection.stack_rows) == 1
    assert projection.stack_rows[0]["weight_value"] == 100
    assert projection.representative_stacks_dropped == 1
    assert projection.representative_stack_weight_bytes_dropped == 10


def test_memray_stack_depth_limit_is_reported_and_applied_to_navigation(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path, max_stack_depth=2)
    record = _AllocationRecord(
        size=25,
        n_allocations=1,
        stack=(("leaf", "work.py", 3), ("middle", "work.py", 2), ("root", "work.py", 1)),
    )

    _total, coverage = memray_worker._aggregate(
        (record,), metric="memory.high_watermark", state=state
    )
    projection = state.finalize()
    state.close()

    assert coverage.dropped_stack_frames == 1
    assert coverage.dropped_stack_frame_bytes == 25
    assert len(projection.stack_rows[0]["frame_ids"]) == 2
    assert len(projection.edge_rows) == 1


def test_memray_navigation_does_not_guess_native_frame_identity(tmp_path: Path) -> None:
    state = _state(tmp_path)
    record = _AllocationRecordWithNativeFrames(
        size=25,
        n_allocations=1,
        stack=(("python_leaf", "work.py", 2), ("python_root", "work.py", 1)),
        native_stack=(("malloc", "libc.so", 0),),
    )

    memray_worker._aggregate((record,), metric="memory.high_watermark", state=state)
    projection = state.finalize()
    state.close()

    assert {row["function"] for row in projection.frame_rows} == {
        "python_leaf",
        "python_root",
    }
    assert {row["language"] for row in projection.frame_rows} == {"Python"}


def test_memray_paths_use_captured_cwd_and_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    workload = project / "workload"
    source_state_id = "sha256:" + "a" * 64
    elsewhere = tmp_path / "extractor"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert memray_worker._normalize(
        "entry.py",
        workload_cwd=workload,
        project_root=project,
        source_state_id=source_state_id,
    ) == ("workload/entry.py", source_state_id, "complete")
    assert memray_worker._normalize(
        "entry.py",
        workload_cwd=None,
        project_root=project,
        source_state_id=None,
    ) == ("entry.py", None, "partial")
    assert memray_worker._normalize(
        "entry.py",
        workload_cwd=workload,
        project_root=project,
        source_state_id=None,
    ) == ("workload/entry.py", None, "partial")
    assert memray_worker._normalize(
        "../../outside.py",
        workload_cwd=workload,
        project_root=project,
        source_state_id=source_state_id,
    ) == ("../../outside.py", None, "partial")
    assert memray_worker._normalize(
        "<listcomp>",
        workload_cwd=workload,
        project_root=project,
        source_state_id=source_state_id,
    ) == ("<listcomp>", None, "partial")
    external = tmp_path / "site-packages" / "dependency.py"
    assert memray_worker._normalize(
        str(external),
        workload_cwd=workload,
        project_root=project,
        source_state_id=source_state_id,
    ) == (str(external), None, "partial")

    moved_project = tmp_path / "moved"
    assert memray_worker._normalize(
        "entry.py",
        workload_cwd=moved_project / "workload",
        project_root=moved_project,
        source_state_id=source_state_id,
    ) == ("workload/entry.py", source_state_id, "complete")
