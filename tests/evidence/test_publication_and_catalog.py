from __future__ import annotations

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flameox.atomic import atomic_write_json
from flameox.catalog import Catalog
from flameox.domain import DomainError, ErrorCode
from flameox.evidence import GenerationPublisher, schema_for, table_names
from flameox.storage import CorpusCommit, GenerationManifest, Workspace
from flameox.storage.locks import RETENTION_EXCLUSIVE, WorkspaceLockIntent

pytestmark = [pytest.mark.integration, pytest.mark.serial]

DIGEST = "sha256:" + ("a" * 64)


@pytest.mark.anyio
async def test_async_prepared_publication_cancellation_removes_staging(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    publisher = GenerationPublisher(workspace)
    initial_head = workspace.corpus.read_head().commit_id
    entered = asyncio.Event()

    async def prepare(
        _root: Path,
    ) -> dict[str, Path]:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    task = asyncio.create_task(
        publisher.publish_prepared_parquet(
            prepare,
            publisher="test",
            publisher_version="1",
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert workspace.corpus.read_head().commit_id == initial_head
    assert tuple(workspace.paths.staging.iterdir()) == ()


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
        "adapter": None,
        "adapter_version": None,
        "run_semantic_id": "sha256:" + "f" * 64,
        "exit_code": None,
        "wall_time_ns": None,
    }


def test_publication_is_visible_only_through_new_corpus_commit(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    catalog = Catalog(workspace)
    catalog.rebuild()
    publisher = GenerationPublisher(workspace)

    with catalog.open_snapshot() as old_snapshot:
        assert old_snapshot.execute("SELECT count(*) FROM runs").fetchone() == (0,)
        assert old_snapshot.execute(
            "SELECT count(*) FROM runtime_resource_summaries"
        ).fetchone() == (0,)

        published = publisher.publish_rows(
            {"runs": [run_row("run-1")]},
            publisher="test",
            publisher_version="1",
        )

        assert old_snapshot.execute("SELECT count(*) FROM runs").fetchone() == (0,)

    with catalog.open_snapshot() as new_snapshot:
        assert new_snapshot.commit.commit_id == published.commit.commit_id
        assert new_snapshot.execute("SELECT run_id FROM runs").fetchall() == [("run-1",)]
        assert new_snapshot.execute(
            "SELECT count(*) FROM runtime_writable_root_growth"
        ).fetchone() == (0,)


def test_empty_snapshot_preserves_struct_evidence_columns(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    with Catalog(workspace).open_snapshot() as snapshot:
        fields = snapshot.execute("DESCRIBE inference_requests").fetchall()

    outcome = next(field for field in fields if field[0] == "outcome")
    assert outcome[1] == "STRUCT(kind VARCHAR, error_type VARCHAR, error_code VARCHAR)"


def test_current_runs_uses_domain_revision_not_projection_completion_order(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    publisher = GenerationPublisher(workspace)
    revision_two = {
        **run_row("run-1"),
        "run_revision": 2,
        "run_manifest_digest": "sha256:" + "2" * 64,
        "execution_status": "succeeded",
    }
    delayed_revision_one = {
        **run_row("run-1"),
        "run_revision": 1,
        "run_manifest_digest": "sha256:" + "1" * 64,
        "execution_status": "running",
    }

    publisher.publish_rows(
        {"runs": [revision_two]},
        publisher="revision-test",
        publisher_version="1",
    )
    publisher.publish_rows(
        {"runs": [delayed_revision_one]},
        publisher="revision-test",
        publisher_version="1",
    )

    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute(
            "SELECT run_revision, execution_status FROM current_runs WHERE run_id = 'run-1'"
        ).fetchone() == (2, "succeeded")


def test_snapshot_rejects_conflicting_content_for_one_run_revision(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    publisher = GenerationPublisher(workspace)
    for digest in ("1", "2"):
        publisher.publish_rows(
            {
                "runs": [
                    {
                        **run_row("run-1"),
                        "run_revision": 1,
                        "run_manifest_digest": "sha256:" + digest * 64,
                    }
                ]
            },
            publisher="revision-test",
            publisher_version="1",
        )

    with pytest.raises(DomainError) as conflict, Catalog(workspace).open_snapshot():
        pass
    assert conflict.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_idempotent_publication_reuses_or_supersedes_exact_operation(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    publisher = GenerationPublisher(workspace)

    first = publisher.publish_rows_idempotent(
        {"runs": [run_row("normalized-run")]},
        publisher="extractor",
        publisher_version="1",
        input_run_ids=("source-run",),
        operation_identity={"tool_digest": "sha256:first"},
    )
    repeated = publisher.publish_rows_idempotent(
        {"runs": [run_row("normalized-run")]},
        publisher="extractor",
        publisher_version="1",
        input_run_ids=("source-run",),
        operation_identity={"tool_digest": "sha256:first"},
    )
    changed = publisher.publish_rows_idempotent(
        {"runs": [run_row("normalized-run")]},
        publisher="extractor",
        publisher_version="1",
        input_run_ids=("source-run",),
        operation_identity={"tool_digest": "sha256:second"},
    )

    assert repeated.commit.commit_id == first.commit.commit_id
    assert repeated.manifest.generation_id == first.manifest.generation_id
    assert changed.manifest.supersedes == (first.manifest.generation_id,)
    assert changed.manifest.operation_digest != first.manifest.operation_digest
    assert len(workspace.corpus.read_head().generation_ids) == 1
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute("SELECT count(*) FROM runs").fetchone() == (1,)


def test_rogue_parquet_file_is_invisible_without_manifest(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    catalog = Catalog(workspace)
    catalog.rebuild()
    rogue = workspace.paths.evidence / "runs" / "rogue.parquet"
    rogue.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([], schema=schema_for("runs")), rogue)

    with catalog.open_snapshot() as snapshot:
        assert snapshot.execute("SELECT count(*) FROM runs").fetchone() == (0,)


def test_catalog_rejects_parquet_schema_drift(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    published = GenerationPublisher(workspace).publish_rows(
        {"frames": []},
        publisher="test",
        publisher_version="1",
    )
    path = workspace.paths.root / published.manifest.files[0].path
    table = pq.read_table(path).append_column(
        "unexpected_column",
        pa.array([], type=pa.int32()),
    )
    pq.write_table(table, path)

    with pytest.raises(DomainError) as raised, Catalog(workspace).open_snapshot():
        pass

    assert raised.value.code is ErrorCode.EVIDENCE_SCHEMA_MISMATCH


def test_manifest_owns_normalized_provenance_and_physical_rows_do_not_repeat_it(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    published = GenerationPublisher(workspace).publish_rows(
        {"runs": [run_row("run-1")]},
        publisher="test",
        publisher_version="1",
    )
    path = workspace.paths.root / published.manifest.files[0].path
    physical_columns = set(pq.read_schema(path).names)
    assert physical_columns.isdisjoint(
        {
            "evidence_generation_id",
            "published_at",
            "extractor_name",
            "extractor_version",
        }
    )
    assert all(
        set(schema_for(name).names).isdisjoint(
            {"evidence_generation_id", "published_at", "extractor_name"}
        )
        for name in table_names()
    )
    assert "extractor_version" in schema_for("adapter_extractions").names

    with Catalog(workspace).open_snapshot() as snapshot:
        provenance = snapshot.execute(
            "SELECT evidence_generation_id, published_at, extractor_name, extractor_version "
            "FROM runs WHERE run_id = 'run-1'"
        ).fetchone()
    assert provenance == (
        published.manifest.generation_id,
        published.manifest.created_at,
        "test",
        "1",
    )


def test_catalog_preserves_table_owned_extractor_version(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    published = GenerationPublisher(workspace).publish_rows(
        {
            "adapter_extractions": [
                {
                    "extraction_id": "extraction-1",
                    "run_id": "run-1",
                    "input_artifact_id": "sha256:" + "a" * 64,
                    "adapter": "adapter",
                    "adapter_package_identity": "adapter-package",
                    "extractor_version": "provider-3",
                    "summary_json": "{}",
                    "limitations": [],
                }
            ]
        },
        publisher="test",
        publisher_version="1",
    )

    with Catalog(workspace).open_snapshot() as snapshot:
        provenance = snapshot.execute(
            "SELECT evidence_generation_id, published_at, extractor_name, extractor_version "
            "FROM adapter_extractions"
        ).fetchone()

    assert provenance == (
        published.manifest.generation_id,
        published.manifest.created_at,
        "test",
        "provider-3",
    )


def test_catalog_projects_manifest_provenance_for_each_generation_file(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    first = GenerationPublisher(workspace).publish_rows(
        {"runs": [run_row("run-1")]},
        publisher="first-publisher",
        publisher_version="1",
    )
    second = GenerationPublisher(workspace).publish_rows(
        {"runs": [run_row("run-2")]},
        publisher="second-publisher",
        publisher_version="2",
    )

    with Catalog(workspace).open_snapshot() as snapshot:
        provenance = snapshot.execute(
            "SELECT run_id, evidence_generation_id, published_at, "
            "extractor_name, extractor_version FROM runs ORDER BY run_id"
        ).fetchall()

    assert provenance == [
        (
            "run-1",
            first.manifest.generation_id,
            first.manifest.created_at,
            "first-publisher",
            "1",
        ),
        (
            "run-2",
            second.manifest.generation_id,
            second.manifest.created_at,
            "second-publisher",
            "2",
        ),
    ]


def test_snapshot_rejects_manifest_replacement_behind_committed_generation_id(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    published = GenerationPublisher(workspace).publish_rows(
        {"runs": [run_row("run-1")]},
        publisher="test",
        publisher_version="1",
    )
    manifest_path = workspace.corpus.generation_path(published.manifest.generation_id)
    manifest = json.loads(manifest_path.read_text())
    assert manifest_path.name == f"{published.manifest.generation_id.removeprefix('sha256:')}.json"
    assert "generation_id" not in manifest
    assert workspace.corpus.read_head().generation_ids == (published.manifest.generation_id,)
    manifest["publisher"] = "replacement"
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(DomainError) as raised, Catalog(workspace).open_snapshot():
        pass

    assert raised.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_catalog_rejects_manifest_row_count_mismatch(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    published = GenerationPublisher(workspace).publish_rows(
        {"runs": [run_row("run-1"), run_row("run-2")]},
        publisher="test",
        publisher_version="1",
    )
    manifest_path = workspace.corpus.generation_path(published.manifest.generation_id)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["row_count"] = 1
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(DomainError) as raised, Catalog(workspace).open_snapshot():
        pass

    assert raised.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_catalog_rejects_manifest_byte_length_mismatch(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    published = GenerationPublisher(workspace).publish_rows(
        {"runs": [run_row("run-1")]},
        publisher="test",
        publisher_version="1",
    )
    manifest_path = workspace.corpus.generation_path(published.manifest.generation_id)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["byte_length"] += 1
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(DomainError) as raised, Catalog(workspace).open_snapshot():
        pass

    assert raised.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_generation_manifest_rejects_persisted_derived_identity(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    published = GenerationPublisher(workspace).publish_rows(
        {"frames": []},
        publisher="test",
        publisher_version="1",
    )
    manifest = published.manifest.model_dump(mode="json")
    manifest["generation_id"] = published.manifest.generation_id
    with pytest.raises(ValueError):
        GenerationManifest.model_validate(manifest)


def test_snapshot_connection_cannot_read_outside_workspace(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    catalog = Catalog(workspace)
    catalog.rebuild()

    with catalog.open_snapshot() as snapshot:
        with pytest.raises(duckdb.PermissionException):
            snapshot.execute("SELECT * FROM read_csv_auto('/etc/passwd')").fetchall()
        with pytest.raises(duckdb.InvalidInputException):
            snapshot.execute("SET threads=99")


def test_concurrent_read_snapshots_keep_configuration_connection_local(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    catalog = Catalog(workspace)
    catalog.rebuild()
    ready = threading.Barrier(2)

    def read_head() -> int:
        with catalog.open_snapshot() as snapshot:
            ready.wait(timeout=5)
            row = snapshot.execute("SELECT 1").fetchone()
            assert row is not None
            return cast(int, row[0])

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: read_head(), range(2)))

    assert results == (1, 1)


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
    query_started = threading.Event()

    def run_query(snapshot: Any) -> list[tuple[object, ...]]:
        query_started.set()
        return cast(
            list[tuple[object, ...]],
            snapshot.execute("SELECT sum(sin(i)) FROM range(100000000000) values(i)").fetchall(),
        )

    task = asyncio.create_task(catalog.run_interruptible(run_query))
    assert await asyncio.to_thread(query_started.wait, 5)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with catalog.open_snapshot() as snapshot:
        assert snapshot.execute("SELECT 1").fetchone() == (1,)


@pytest.mark.anyio
async def test_interruptible_catalog_cancellation_aborts_pre_snapshot_lock_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    catalog = Catalog(workspace)
    original_locked = workspace.locked
    lock_held = threading.Event()
    lock_attempted = threading.Event()
    release_lock = threading.Event()
    operation_started = threading.Event()

    def hold_retention_lock() -> None:
        with original_locked(RETENTION_EXCLUSIVE):
            lock_held.set()
            release_lock.wait()

    @contextmanager
    def observed_locked(
        *intents: WorkspaceLockIntent,
        timeout: float = 30,
        phase: str = "workspace mutation",
    ) -> Any:
        lock_attempted.set()
        with original_locked(*intents, timeout=timeout, phase=phase) as streams:
            yield streams

    holder = threading.Thread(target=hold_retention_lock)
    holder.start()
    assert await asyncio.to_thread(lock_held.wait, 1)
    monkeypatch.setattr(workspace, "locked", observed_locked)

    def query_after_lock(snapshot: Any) -> list[tuple[object, ...]]:
        operation_started.set()
        return cast(list[tuple[object, ...]], snapshot.execute("SELECT 1").fetchall())

    task = asyncio.create_task(
        catalog.run_interruptible(
            query_after_lock,
            query_name="blocked-before-snapshot",
        )
    )
    assert await asyncio.to_thread(lock_attempted.wait, 1)
    task.cancel()
    try:
        async with asyncio.timeout(1):
            with pytest.raises(asyncio.CancelledError):
                await task
        assert holder.is_alive()
        assert not operation_started.is_set()

        events = [
            json.loads(line)
            for line in workspace.paths.operation_log.read_text().splitlines()
            if '"query_name":"blocked-before-snapshot"' in line
        ]
        assert events[-2]["cleanup_status"] == "pending"
        assert events[-1]["phase"] == "query cancelled"
        assert events[-1]["cleanup_status"] == "complete"
    finally:
        release_lock.set()
        holder.join(timeout=1)


@pytest.mark.anyio
async def test_interruptible_catalog_retains_slow_worker_until_locks_are_released(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    catalog = Catalog(workspace)
    operation_started = threading.Event()
    release_operation = threading.Event()
    operation_finished = threading.Event()

    def slow_operation(snapshot: Any) -> list[tuple[object, ...]]:
        operation_started.set()
        try:
            release_operation.wait()
            return cast(list[tuple[object, ...]], snapshot.execute("SELECT 1").fetchall())
        finally:
            operation_finished.set()

    task = asyncio.create_task(
        catalog.run_interruptible(slow_operation, query_name="slow-python-operation")
    )
    assert await asyncio.to_thread(operation_started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()

    done, _ = await asyncio.wait({task}, timeout=0.1)
    try:
        assert not done
        assert not operation_finished.is_set()
        assert '"phase":"query cancelled"' not in "\n".join(
            line
            for line in workspace.paths.operation_log.read_text().splitlines()
            if '"query_name":"slow-python-operation"' in line
        )
    finally:
        release_operation.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert operation_finished.is_set()
    with (
        workspace.retention_locked(shared=False, timeout=1),
        workspace.catalog_locked(shared=False, timeout=1),
    ):
        pass

    events = [
        json.loads(line)
        for line in workspace.paths.operation_log.read_text().splitlines()
        if '"query_name":"slow-python-operation"' in line
    ]
    assert events[-1]["phase"] == "query cancelled"
    assert events[-1]["cleanup_status"] == "complete"


@pytest.mark.anyio
async def test_interruptible_catalog_admission_is_bounded_and_cancellable(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    catalog = Catalog(workspace)
    release_operations = threading.Event()
    active_lock = threading.Lock()
    active = 0
    maximum_active = 0
    four_started = threading.Event()

    def slow_operation(snapshot: Any) -> list[tuple[object, ...]]:
        nonlocal active, maximum_active
        with active_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 4:
                four_started.set()
        try:
            release_operations.wait()
            return cast(list[tuple[object, ...]], snapshot.execute("SELECT 1").fetchall())
        finally:
            with active_lock:
                active -= 1

    active_tasks = [
        asyncio.create_task(catalog.run_interruptible(slow_operation)) for _ in range(4)
    ]
    assert await asyncio.to_thread(four_started.wait, 2)
    waiting = asyncio.create_task(catalog.run_interruptible(slow_operation))
    await asyncio.sleep(0.05)
    waiting.cancel()
    try:
        async with asyncio.timeout(1):
            with pytest.raises(asyncio.CancelledError):
                await waiting
        assert maximum_active == 4
    finally:
        release_operations.set()

    assert all(await asyncio.gather(*active_tasks))


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
                        "created_at": datetime(2025, 1, 2, 3, 4, index, tzinfo=UTC),
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
    assert len(head.generation_ids) == 8
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute("SELECT count(*) FROM investigations").fetchone() == (8,)


def test_generation_row_quota_is_enforced_before_staging(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "storage": workspace.config.storage.validated_copy(
                update={"max_rows_per_generation": 1}
            )
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
    assert not list(workspace.paths.evidence.rglob("*.parquet"))
    assert not list(workspace.paths.generations.glob("*.json"))
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


def test_uncertain_head_read_after_publication_failure_keeps_referenced_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    original_publish_head = workspace.corpus.publish_head
    original_read_head = workspace.corpus.read_head
    head_advanced = False

    def fail_after_head(commit_id: str) -> None:
        nonlocal head_advanced
        original_publish_head(commit_id)
        head_advanced = True
        raise RuntimeError("simulated failure after HEAD replacement")

    def unreadable_after_head() -> CorpusCommit:
        if head_advanced:
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "simulated unreadable HEAD")
        return original_read_head()

    monkeypatch.setattr(workspace.corpus, "publish_head", fail_after_head)
    monkeypatch.setattr(workspace.corpus, "read_head", unreadable_after_head)

    with pytest.raises(RuntimeError, match="failure after HEAD"):
        GenerationPublisher(workspace).publish_rows(
            {"runs": [run_row("run-after-uncertain-head")]},
            publisher="test",
            publisher_version="1",
        )

    monkeypatch.setattr(workspace.corpus, "read_head", original_read_head)
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute("SELECT run_id FROM runs").fetchall() == [
            ("run-after-uncertain-head",)
        ]
