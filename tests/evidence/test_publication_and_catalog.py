from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flamo.application import CompactionService
from flamo.catalog import Catalog
from flamo.evidence import GenerationPublisher, schema_for
from flamo.storage import Workspace

DIGEST = "sha256:" + ("a" * 64)


def run_row(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "created_at": datetime(2026, 7, 25, tzinfo=UTC),
        "run_type": "import",
        "execution_status": "not_applicable",
        "capture_status": "registered",
        "validation_status": "not_requested",
        "workload_definition_id": None,
        "workload_instance_id": None,
        "measurement_protocol_id": None,
        "environment_id": DIGEST,
        "source_state_id": None,
        "collector": None,
        "collector_version": None,
        "exit_code": None,
        "wall_time_ns": None,
        "manifest_path": "runs/run/manifest.json",
    }


def test_publication_is_visible_only_through_new_corpus_commit(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    catalog = Catalog(workspace)
    catalog.rebuild()
    publisher = GenerationPublisher(workspace)

    with catalog.open_snapshot() as old_snapshot:
        assert old_snapshot.execute("SELECT count(*) FROM runs").fetchone() == (0,)

        published = publisher.publish_rows(
            {"runs": [run_row("run-1")]},
            publisher="test",
            publisher_version="1",
        )

        assert old_snapshot.execute("SELECT count(*) FROM runs").fetchone() == (0,)

    with catalog.open_snapshot() as new_snapshot:
        assert new_snapshot.commit.commit_id == published.commit.commit_id
        assert new_snapshot.execute("SELECT run_id FROM runs").fetchall() == [("run-1",)]


def test_rogue_parquet_file_is_invisible_without_manifest(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    catalog = Catalog(workspace)
    catalog.rebuild()
    rogue = workspace.paths.evidence / "runs" / "rogue.parquet"
    rogue.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([], schema=schema_for("runs")), rogue)

    with catalog.open_snapshot() as snapshot:
        assert snapshot.execute("SELECT count(*) FROM runs").fetchone() == (0,)


def test_snapshot_connection_cannot_read_outside_workspace(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    catalog = Catalog(workspace)
    catalog.rebuild()

    with catalog.open_snapshot() as snapshot:
        with pytest.raises(duckdb.PermissionException):
            snapshot.execute("SELECT * FROM read_csv_auto('/etc/passwd')").fetchall()
        with pytest.raises(duckdb.InvalidInputException):
            snapshot.execute("SET threads=99")


def test_catalog_is_rebuildable_after_deletion(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    publisher = GenerationPublisher(workspace)
    publisher.publish_rows(
        {"runs": [run_row("run-1")]},
        publisher="test",
        publisher_version="1",
    )
    catalog = Catalog(workspace)
    catalog.rebuild()
    workspace.paths.catalog.unlink()

    catalog.rebuild()

    with catalog.open_snapshot() as snapshot:
        assert snapshot.execute("SELECT run_id FROM runs").fetchone() == ("run-1",)


def test_concurrent_publishers_retry_contention_without_losing_generations(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()

    def publish(index: int) -> str:
        result = GenerationPublisher(workspace).publish_rows(
            {
                "investigations": [
                    {
                        "investigation_id": f"investigation-{index}",
                        "question": f"question {index}",
                        "symptom": None,
                        "project_root": ".",
                        "status": "open",
                        "parent_investigation_id": None,
                        "created_at": datetime.now(UTC),
                    }
                ]
            },
            publisher="concurrency-test",
            publisher_version="1",
        )
        return result.commit.commit_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        commit_ids = list(executor.map(publish, range(8)))

    assert len(set(commit_ids)) == 8
    head = workspace.corpus.read_head()
    assert len(head.generation_manifests) == 8
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute("SELECT count(*) FROM investigations").fetchone() == (8,)


def test_compaction_replaces_reachable_small_generations(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    publisher = GenerationPublisher(workspace)
    for index in range(3):
        publisher.publish_rows(
            {"runs": [run_row(f"run-{index}")]},
            publisher="test",
            publisher_version="1",
        )

    result = CompactionService(workspace).compact()

    assert result.superseded_generation_count == 3
    assert result.reachable_file_count_before == 3
    assert result.reachable_file_count_after == 1
    assert len(workspace.corpus.read_head().generation_manifests) == 1
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute("SELECT count(*) FROM runs").fetchone() == (3,)
