from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flamo.application import CompactionService
from flamo.catalog import Catalog
from flamo.evidence import GenerationPublisher
from flamo.storage import Workspace


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
        "collector": None,
        "collector_version": None,
        "exit_code": None,
        "wall_time_ns": None,
        "manifest_path": "synthetic",
    }


@pytest.mark.performance
@pytest.mark.skipif(
    os.environ.get("FLAMO_RUN_PERFORMANCE") != "1",
    reason="set FLAMO_RUN_PERFORMANCE=1 to run the 100k corpus acceptance check",
)
def test_100k_run_catalog_after_compaction(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    publisher = GenerationPublisher(workspace)
    for start in range(0, 100_000, 10_000):
        publisher.publish_rows(
            {"runs": [_run_row(index) for index in range(start, start + 10_000)]},
            publisher="scale-fixture",
            publisher_version="1",
        )

    compacted = CompactionService(workspace).compact()
    assert compacted.reachable_file_count_after == 1

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
    query_seconds = time.perf_counter() - started

    assert count == (100_000,)
    assert rebuild_seconds < 15
    assert query_seconds < 5
