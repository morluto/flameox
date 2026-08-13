from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flameox.application import CompactionService, workspace_status
from flameox.catalog import Catalog
from flameox.evidence import GenerationPublisher
from flameox.storage import ArtifactStore, Workspace

pytestmark = [pytest.mark.performance, pytest.mark.serial]


def _run_row(index: int) -> dict[str, object]:
    return {
        "run_id": f"synthetic-{index:06d}",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "run_type": "import",
        "execution_status": "not_applicable",
        "capture_status": "complete",
        "validation_status": "not_requested",
        "workload_definition_id": None,
        "workload_instance_id": None,
        "measurement_protocol_id": None,
        "environment_id": "synthetic-environment",
        "source_state_id": None,
        "collector": "baseline" if index % 2 == 0 else "candidate",
        "collector_version": None,
        "exit_code": None,
        "wall_time_ns": None,
        "manifest_path": "synthetic",
    }


def _measurement_row(index: int) -> dict[str, object]:
    return {
        "measurement_id": f"synthetic-measurement-{index:06d}",
        "run_id": f"synthetic-{index:06d}",
        "artifact_id": None,
        "name": "wall_time",
        "value_int": index + 1,
        "value_float": None,
        "unit": "ns",
        "aggregation": "single",
        "scope": "process",
        "trial_id": None,
        "worker_id": None,
        "worker_run_index": None,
        "value_index": 0,
        "loop_count": 1,
        "is_warmup": False,
        "block_id": None,
        "variant_id": None,
        "order_in_block": None,
        "phase": "steady_state",
        "dimensions": {},
        "evidence_level": "observed",
    }


@pytest.mark.parametrize(
    ("run_count", "publication_budget", "rebuild_budget", "startup_budget"),
    (
        (10, 5.0, 5.0, 2.0),
        (1_000, 10.0, 5.0, 2.0),
        pytest.param(
            100_000,
            30.0,
            15.0,
            5.0,
            marks=pytest.mark.skipif(
                os.environ.get("FLAMEOX_RUN_PERFORMANCE") != "1",
                reason="set FLAMEOX_RUN_PERFORMANCE=1 for the 100k acceptance check",
            ),
        ),
    ),
)
@pytest.mark.performance
def test_catalog_scale_budget_matrix(
    tmp_path: Path,
    run_count: int,
    publication_budget: float,
    rebuild_budget: float,
    startup_budget: float,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    publisher = GenerationPublisher(workspace)
    started = time.perf_counter()
    chunk_size = min(10_000, run_count)
    for start in range(0, run_count, chunk_size):
        publisher.publish_rows(
            {
                "runs": [
                    _run_row(index) for index in range(start, min(start + chunk_size, run_count))
                ],
                "measurements": [
                    _measurement_row(index)
                    for index in range(start, min(start + chunk_size, run_count))
                ],
            },
            publisher="scale-fixture",
            publisher_version="1",
        )
    publication_seconds = time.perf_counter() - started

    compacted = CompactionService(workspace).compact()
    assert compacted.reachable_file_count_after == 2

    catalog = Catalog(workspace)
    started = time.perf_counter()
    catalog.rebuild()
    rebuild_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with catalog.open_snapshot() as snapshot:
        count = snapshot.execute(
            "SELECT count(*) FROM runs WHERE environment_id = ?",
            ("synthetic-environment",),
        ).fetchone()
        cohort_comparison = snapshot.execute(
            "SELECT runs.collector, avg(measurements.value_int) "
            "FROM runs JOIN measurements USING (run_id) "
            "WHERE runs.environment_id = ? AND measurements.name = ? "
            "GROUP BY runs.collector ORDER BY runs.collector",
            ("synthetic-environment", "wall_time"),
        ).fetchall()
    startup_query_seconds = time.perf_counter() - started
    started = time.perf_counter()
    encoded = workspace_status(workspace).model_dump_json()
    serialization_seconds = time.perf_counter() - started

    assert count == (run_count,)
    assert len(cohort_comparison) == 2
    assert publication_seconds < publication_budget
    assert rebuild_seconds < rebuild_budget
    assert startup_query_seconds < startup_budget
    assert serialization_seconds < 5
    assert len(encoded) < 1_000_000


@pytest.mark.performance
def test_artifact_hashing_throughput_budget(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "hash-source.bin"
    byte_count = 16 * 1024 * 1024
    source.write_bytes(b"x" * byte_count)

    started = time.perf_counter()
    stored = ArtifactStore(workspace).import_path(
        source,
        allowed_roots=(tmp_path,),
        max_bytes=byte_count,
    )
    seconds = time.perf_counter() - started

    assert stored.content.byte_length == byte_count
    assert byte_count / max(seconds, 0.001) > 10 * 1024 * 1024
