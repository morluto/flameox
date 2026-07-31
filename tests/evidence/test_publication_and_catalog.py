from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flameox.application import CompactionService
from flameox.catalog import Catalog
from flameox.domain import DomainError, ErrorCode
from flameox.evidence import GenerationPublisher, schema_for
from flameox.storage import CorpusCommit, Workspace
from flameox.storage.atomic import atomic_write_json

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


def test_catalog_projects_additive_trial_columns_for_old_generations(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    published = GenerationPublisher(workspace).publish_rows(
        {
            "trials": [
                {
                    "trial_id": "trial-1",
                    "experiment_id": "experiment-1",
                    "variant_id": "variant-1",
                    "run_id": None,
                    "combination_id": "combination-1",
                    "factors_json": '{"treatment":"base"}',
                    "block_id": "block-1",
                    "order_in_block": 1,
                    "parameter_name": None,
                    "parameter_value_int": None,
                    "parameter_value_float": None,
                    "attempt": 1,
                    "outcome": "succeeded",
                    "exclusion_reason": None,
                    "validation_status": "not_requested",
                    "failure_class": "none",
                    "oracle_receipt_json": None,
                    "oracle_receipt_artifact_id": None,
                }
            ]
        },
        publisher="test",
        publisher_version="1",
    )
    path = workspace.paths.root / published.manifest.files[0].path
    table = pq.read_table(path).drop(["oracle_receipt_json", "oracle_receipt_artifact_id"])
    pq.write_table(table, path)

    Catalog(workspace).rebuild()
    with Catalog(workspace).open_snapshot() as snapshot:
        row = snapshot.execute(
            "SELECT trial_id, oracle_receipt_json, oracle_receipt_artifact_id FROM trials"
        ).fetchone()
    assert row == ("trial-1", None, None)


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


@pytest.mark.anyio
async def test_interruptible_catalog_query_cancels_and_releases_snapshot(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    catalog = Catalog(workspace)
    catalog.rebuild()
    task = asyncio.create_task(
        catalog.run_interruptible(
            lambda snapshot: snapshot.execute(
                "SELECT sum(sin(i)) FROM range(100000000000) values(i)"
            ).fetchall()
        )
    )
    await asyncio.sleep(0.05)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with catalog.open_snapshot() as snapshot:
        assert snapshot.execute("SELECT 1").fetchone() == (1,)


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


def test_generation_row_quota_is_enforced_before_staging(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.model_copy(
        update={
            "storage": workspace.config.storage.model_copy(update={"max_rows_per_generation": 1})
        }
    )
    workspace.paths.config.write_text(config.to_toml())

    with pytest.raises(DomainError) as error:
        GenerationPublisher(workspace).publish_rows(
            {"runs": [run_row("run-1"), run_row("run-2")]},
            publisher="test",
            publisher_version="1",
        )

    assert error.value.code is ErrorCode.STORAGE_QUOTA_EXCEEDED
    assert not any(workspace.paths.staging.iterdir())


def test_publication_failure_removes_its_staging_directory(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    with pytest.raises(ValueError, match="Unknown evidence table"):
        GenerationPublisher(workspace).publish_rows(
            {"unknown_table": []},
            publisher="test",
            publisher_version="1",
        )

    assert not any(workspace.paths.staging.iterdir())


@pytest.mark.parametrize(
    "boundary",
    (
        "parquet_staged",
        "manifest_staged",
        "evidence_published",
        "manifest_published",
        "commit_written",
        "before_head",
    ),
)
def test_publication_crashes_before_head_never_expose_partial_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    publisher = GenerationPublisher(workspace)
    original_head = workspace.corpus.read_head().commit_id

    if boundary == "parquet_staged":
        original_write_table = cast(Any, pq.write_table)

        def fail_after_parquet(*args: object, **kwargs: object) -> None:
            original_write_table(*args, **kwargs)
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(pq, "write_table", fail_after_parquet)
    elif boundary == "manifest_staged":
        original_atomic_write_json = atomic_write_json

        def fail_after_manifest(
            path: Path,
            value: object,
            *,
            mode: int = 0o600,
        ) -> None:
            original_atomic_write_json(path, value, mode=mode)
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(
            "flameox.evidence.publisher.atomic_write_json",
            fail_after_manifest,
        )
    elif boundary in {"evidence_published", "manifest_published"}:
        original_replace = os.replace

        def fail_after_move(source: Path, destination: Path) -> None:
            original_replace(source, destination)
            relative = Path(destination).relative_to(workspace.paths.root)
            target = (
                relative.parts[0] == "evidence"
                if boundary == "evidence_published"
                else relative.parts[0] == "generations"
            )
            if target:
                raise RuntimeError("simulated crash")

        monkeypatch.setattr(
            "flameox.evidence.publisher.os.replace",
            fail_after_move,
        )
    elif boundary == "commit_written":
        original_write_commit = workspace.corpus.write_commit

        def fail_after_commit(commit: CorpusCommit) -> None:
            original_write_commit(commit)
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(workspace.corpus, "write_commit", fail_after_commit)
    else:

        def fail_before_head(_commit_id: str) -> None:
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(workspace.corpus, "publish_head", fail_before_head)

    with pytest.raises(RuntimeError, match="simulated crash"):
        publisher.publish_rows(
            {"runs": [run_row(f"run-{boundary}")]},
            publisher="test",
            publisher_version="1",
        )

    assert workspace.corpus.read_head().commit_id == original_head
    with Catalog(workspace).open_snapshot(original_head) as snapshot:
        assert snapshot.execute("SELECT count(*) FROM runs").fetchone() == (0,)


def test_crash_after_head_only_exposes_complete_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    original_publish_head = workspace.corpus.publish_head

    def fail(commit_id: str) -> None:
        original_publish_head(commit_id)
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(workspace.corpus, "publish_head", fail)
    with pytest.raises(RuntimeError, match="simulated crash"):
        GenerationPublisher(workspace).publish_rows(
            {"runs": [run_row("run-after-head")]},
            publisher="test",
            publisher_version="1",
        )

    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute("SELECT run_id FROM runs").fetchall() == [("run-after-head",)]
