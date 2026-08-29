from __future__ import annotations

import asyncio
import os
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import duckdb
import pyarrow as pa
from pydantic import BaseModel

from flameox.atomic import fsync_directory
from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.models import RunManifest, parse_run_manifest_json
from flameox.evidence.schemas import SCHEMA_MAJOR, SCHEMA_MINOR, schema_for, table_names
from flameox.observability import OperationLogger, elapsed_ms
from flameox.storage.corpus import CorpusCommit, GenerationManifest
from flameox.storage.locks import (
    CATALOG_EXCLUSIVE,
    CATALOG_SHARED,
    RETENTION_SHARED,
    WRITE_EXCLUSIVE,
)
from flameox.storage.workspace import Workspace

_CATALOG_QUERY_WORKERS = 4
_CANCELLABLE_LOCK_SLICE_SECONDS = 0.1
_CATALOG_QUERY_EXECUTOR = ThreadPoolExecutor(
    max_workers=_CATALOG_QUERY_WORKERS,
    thread_name_prefix="flameox-catalog",
)
_QUERY_ADMISSIONS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)
_QUERY_ADMISSIONS_LOCK = threading.Lock()


def _query_admission() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    with _QUERY_ADMISSIONS_LOCK:
        admission = _QUERY_ADMISSIONS.get(loop)
        if admission is None:
            admission = asyncio.Semaphore(_CATALOG_QUERY_WORKERS)
            _QUERY_ADMISSIONS[loop] = admission
        return admission


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _duckdb_type(data_type: pa.DataType) -> str:
    if pa.types.is_int32(data_type):
        return "INTEGER"
    if pa.types.is_int64(data_type):
        return "BIGINT"
    if pa.types.is_uint64(data_type):
        return "UBIGINT"
    if pa.types.is_uint32(data_type):
        return "UINTEGER"
    if pa.types.is_float64(data_type):
        return "DOUBLE"
    if pa.types.is_boolean(data_type):
        return "BOOLEAN"
    if pa.types.is_string(data_type):
        return "VARCHAR"
    if pa.types.is_timestamp(data_type):
        return "TIMESTAMPTZ"
    if pa.types.is_list(data_type):
        return f"{_duckdb_type(data_type.value_type)}[]"
    if pa.types.is_map(data_type):
        return f"MAP({_duckdb_type(data_type.key_type)}, {_duckdb_type(data_type.item_type)})"
    raise TypeError(f"Unsupported Arrow type for an empty DuckDB view: {data_type}")


@dataclass(frozen=True, slots=True)
class SnapshotHandle:
    """An immutable corpus identity acquired once at an analysis boundary."""

    commit: CorpusCommit

    @property
    def commit_id(self) -> str:
        return self.commit.commit_id


class Snapshot:
    def __init__(
        self,
        *,
        handle: SnapshotHandle,
        connection: duckdb.DuckDBPyConnection,
    ) -> None:
        self.handle = handle
        self.commit = handle.commit
        self.connection = connection

    def execute(
        self,
        sql: str,
        parameters: tuple[object, ...] = (),
    ) -> duckdb.DuckDBPyConnection:
        return self.connection.execute(sql, parameters)

    def interrupt(self) -> None:
        self.connection.interrupt()

    def run(self, run_id: str) -> RunManifest:
        row = self.execute(
            "SELECT manifest_json FROM current_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise DomainError(
                ErrorCode.RUN_NOT_FOUND,
                f"Run {run_id!r} is absent from snapshot {self.handle.commit_id!r}.",
                details={"missing_entity": "run", "corpus_commit_id": self.handle.commit_id},
            )
        if not isinstance(row[0], str):
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                "The pinned run row predates snapshot-contained run manifests.",
                run_id=run_id,
                remediation=("Republish or re-import the run before analyzing it.",),
            )
        try:
            return parse_run_manifest_json(row[0])
        except ValueError as exc:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                "The pinned snapshot contains an invalid run manifest.",
                run_id=run_id,
            ) from exc


class _CancelledBeforeQuery(Exception):
    pass


class Catalog:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def rebuild(self) -> None:
        operation_id = OperationLogger(self.workspace.paths.root).new_id()
        started = time.monotonic()
        head = self.workspace.corpus.read_head()
        inventory = self._inventory(head)
        temporary = self.workspace.paths.catalog.with_name(f".catalog.{uuid4().hex}.duckdb")
        connection = duckdb.connect(str(temporary))
        try:
            connection.execute(
                """
                CREATE TABLE flameox_catalog_metadata (
                    built_at VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO flameox_catalog_metadata VALUES (?)",
                (datetime.now(UTC).isoformat(),),
            )
            connection.execute(
                """
                CREATE TABLE flameox_schema_registry (
                    table_name VARCHAR PRIMARY KEY,
                    schema_major INTEGER NOT NULL,
                    schema_minor INTEGER NOT NULL
                )
                """
            )
            connection.executemany(
                "INSERT INTO flameox_schema_registry VALUES (?, ?, ?)",
                [(name, SCHEMA_MAJOR, SCHEMA_MINOR) for name in table_names()],
            )
            connection.execute("CHECKPOINT")
        finally:
            connection.close()

        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        lock_started = time.monotonic()
        lock_wait_ms = 0.0
        try:
            with self.workspace.locked(
                WRITE_EXCLUSIVE,
                CATALOG_EXCLUSIVE,
                phase="catalog publication",
            ):
                lock_wait_ms = elapsed_ms(lock_started)
                current = self.workspace.corpus.read_head()
                if current.commit_id != head.commit_id:
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        "The corpus changed while the catalog was rebuilt.",
                        retryable=True,
                    )
                self._validate_inventory_paths(inventory)
                os.replace(temporary, self.workspace.paths.catalog)
                fsync_directory(self.workspace.paths.catalog.parent)
        finally:
            temporary.unlink(missing_ok=True)
        OperationLogger(self.workspace.paths.root).emit(
            operation_id=operation_id,
            operation="catalog.rebuild",
            phase="catalog published",
            elapsed_ms=elapsed_ms(started),
            lock_wait_ms=lock_wait_ms,
        )

    def pin(self, commit_id: str | None = None) -> SnapshotHandle:
        commit = (
            self.workspace.corpus.read_head()
            if commit_id is None
            else self.workspace.corpus.read_commit(commit_id)
        )
        return SnapshotHandle(commit=commit)

    @contextmanager
    def open_snapshot(
        self,
        handle: SnapshotHandle | str | None = None,
    ) -> Iterator[Snapshot]:
        with self._open_snapshot(handle) as snapshot:
            yield snapshot

    @contextmanager
    def _open_snapshot(
        self,
        handle: SnapshotHandle | str | None = None,
        *,
        cancellation_requested: threading.Event | None = None,
        snapshot_observer: Callable[[Snapshot | None], None] | None = None,
    ) -> Iterator[Snapshot]:
        if not isinstance(handle, SnapshotHandle):
            handle = self.pin(handle)
        commit = handle.commit
        with self._snapshot_locks(cancellation_requested):
            self._raise_if_cancelled(cancellation_requested)
            if not self.workspace.paths.catalog.exists():
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "The DuckDB catalog is missing.",
                    remediation=("Run `flameox catalog rebuild`.",),
                )
            # Snapshot views are derived entirely from the immutable corpus inventory.
            # Giving each snapshot its own transient database keeps connection security
            # settings local; DuckDB otherwise shares `lock_configuration` across
            # concurrent connections to the same catalog file.
            connection = duckdb.connect(":memory:")
            snapshot = Snapshot(handle=handle, connection=connection)
            if snapshot_observer is not None:
                snapshot_observer(snapshot)
            try:
                self._raise_if_cancelled(cancellation_requested)
                inventory = self._inventory(commit)
                self._configure_connection(connection)
                self._raise_if_cancelled(cancellation_requested)
                self._create_snapshot_views(connection, inventory)
                self._raise_if_cancelled(cancellation_requested)
                yield snapshot
            finally:
                try:
                    connection.close()
                finally:
                    if snapshot_observer is not None:
                        snapshot_observer(None)

    @contextmanager
    def _snapshot_locks(
        self,
        cancellation_requested: threading.Event | None,
    ) -> Iterator[None]:
        if cancellation_requested is None:
            with self.workspace.locked(
                RETENTION_SHARED,
                CATALOG_SHARED,
                phase="snapshot acquisition",
            ):
                yield
            return

        while True:
            self._raise_if_cancelled(cancellation_requested)
            locks = ExitStack()
            try:
                locks.enter_context(
                    self.workspace.locked(
                        RETENTION_SHARED,
                        CATALOG_SHARED,
                        timeout=_CANCELLABLE_LOCK_SLICE_SECONDS,
                        phase="cancellable snapshot acquisition",
                    )
                )
            except DomainError as error:
                locks.close()
                if error.code is not ErrorCode.WRITE_LOCK_TIMEOUT:
                    raise
                if cancellation_requested.wait(_CANCELLABLE_LOCK_SLICE_SECONDS):
                    raise _CancelledBeforeQuery from None
                continue
            except BaseException:
                locks.close()
                raise

            try:
                self._raise_if_cancelled(cancellation_requested)
                with locks:
                    yield
                return
            except BaseException:
                locks.close()
                raise

    @staticmethod
    def _raise_if_cancelled(cancellation_requested: threading.Event | None) -> None:
        if cancellation_requested is not None and cancellation_requested.is_set():
            raise _CancelledBeforeQuery

    async def run_interruptible[T](
        self,
        operation: Callable[[Snapshot], T],
        *,
        handle: SnapshotHandle | None = None,
        query_name: str | None = None,
    ) -> T:
        pinned = handle or self.pin()
        cancellation_requested = threading.Event()
        holder_lock = threading.Lock()
        active_snapshot: list[Snapshot] = []

        def observe_snapshot(snapshot: Snapshot | None) -> None:
            with holder_lock:
                active_snapshot.clear()
                if snapshot is not None:
                    active_snapshot.append(snapshot)

        def run() -> T:
            with self._open_snapshot(
                pinned,
                cancellation_requested=cancellation_requested,
                snapshot_observer=observe_snapshot,
            ) as snapshot:
                self._raise_if_cancelled(cancellation_requested)
                return operation(snapshot)

        operation_id = OperationLogger(self.workspace.paths.root).new_id()
        started = time.monotonic()
        name = query_name or getattr(operation, "__name__", "analysis")
        logger = OperationLogger(self.workspace.paths.root)
        admission = _query_admission()
        try:
            await admission.acquire()
        except asyncio.CancelledError as cancellation:
            logger.emit(
                operation_id=operation_id,
                operation="catalog.query",
                phase="query cancelled before admission",
                query_name=name,
                query_duration_ms=elapsed_ms(started),
                error_code="cancelled",
                cleanup_status="complete",
            )
            raise cancellation from None

        loop = asyncio.get_running_loop()
        worker = loop.run_in_executor(_CATALOG_QUERY_EXECUTOR, run)
        try:
            result = await asyncio.shield(worker)
            rows_returned = getattr(result, "returned", None)
            bytes_returned = (
                len(result.model_dump_json().encode("utf-8"))
                if isinstance(result, BaseModel)
                else None
            )
            logger.emit(
                operation_id=operation_id,
                operation="catalog.query",
                phase="query complete",
                query_name=name,
                query_duration_ms=elapsed_ms(started),
                rows_returned=(
                    int(rows_returned)
                    if isinstance(rows_returned, int) and rows_returned >= 0
                    else None
                ),
                bytes_returned=bytes_returned,
            )
            return result
        except Exception as error:
            logger.emit(
                operation_id=operation_id,
                operation="catalog.query",
                phase="query failed",
                query_name=name,
                query_duration_ms=elapsed_ms(started),
                error_code=type(error).__name__,
            )
            raise
        except asyncio.CancelledError as cancellation:
            cancellation_requested.set()
            logger.emit(
                operation_id=operation_id,
                operation="catalog.query",
                phase="query cancellation requested",
                query_name=name,
                query_duration_ms=elapsed_ms(started),
                error_code="cancelled",
                cleanup_status="pending",
            )
            worker_error: BaseException | None = None
            try:
                while not worker.done():
                    with holder_lock:
                        snapshot = active_snapshot[0] if active_snapshot else None
                    if snapshot is not None:
                        with suppress(duckdb.ConnectionException):
                            snapshot.interrupt()
                    try:
                        await asyncio.wait({worker}, timeout=0.05)
                    except asyncio.CancelledError:
                        # Repeated transport cancellation must not detach the query.
                        continue
                try:
                    worker.result()
                except BaseException as error:
                    worker_error = error
            finally:
                logger.emit(
                    operation_id=operation_id,
                    operation="catalog.query",
                    phase="query cancelled",
                    query_name=name,
                    query_duration_ms=elapsed_ms(started),
                    error_code=(
                        "cancelled" if worker_error is None else type(worker_error).__name__
                    ),
                    cleanup_status="complete",
                )
                raise cancellation from None
        finally:
            admission.release()

    def _inventory(self, commit: CorpusCommit) -> dict[str, list[Path]]:
        inventory: dict[str, list[Path]] = {name: [] for name in table_names()}
        for manifest_relative in commit.generation_manifests:
            manifest_path = (self.workspace.paths.root / manifest_relative).resolve()
            self._require_workspace_path(manifest_path)
            try:
                manifest = GenerationManifest.model_validate_json(manifest_path.read_text())
            except (FileNotFoundError, ValueError) as exc:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Generation manifest is missing or invalid: {manifest_relative}",
                ) from exc
            for file in manifest.files:
                if file.table not in inventory:
                    raise DomainError(
                        ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                        f"Unknown table in generation manifest: {file.table}",
                    )
                if file.schema_major != SCHEMA_MAJOR:
                    raise DomainError(
                        ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                        "Evidence generation uses an unsupported schema major.",
                        details={
                            "table": file.table,
                            "observed_schema_major": file.schema_major,
                            "supported_schema_major": SCHEMA_MAJOR,
                        },
                    )
                path = (self.workspace.paths.root / file.path).resolve()
                self._require_workspace_path(path)
                inventory[file.table].append(path)
        self._validate_inventory_paths(inventory)
        return inventory

    def _require_workspace_path(self, path: Path) -> None:
        try:
            path.relative_to(self.workspace.paths.root)
        except ValueError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Corpus inventory escapes the workspace: {path}",
            ) from exc

    def _validate_inventory_paths(self, inventory: dict[str, list[Path]]) -> None:
        missing = [
            str(path) for paths in inventory.values() for path in paths if not path.is_file()
        ]
        if missing:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Corpus inventory references missing Parquet files.",
                details={"paths": missing[:100]},
            )

    def _configure_connection(self, connection: duckdb.DuckDBPyConnection) -> None:
        allowed = [
            self.workspace.paths.evidence.resolve(),
            (self.workspace.paths.staging / "duckdb-temp").resolve(),
        ]
        allowed[1].mkdir(parents=True, exist_ok=True)
        allowed_sql = ", ".join(_sql_string(str(path)) for path in allowed)
        connection.execute(f"SET allowed_directories=[{allowed_sql}]")
        connection.execute("SET autoinstall_known_extensions=false")
        connection.execute("SET autoload_known_extensions=false")
        connection.execute("SET allow_community_extensions=false")
        connection.execute("SET threads=4")
        connection.execute("SET memory_limit='1GiB'")
        connection.execute(f"SET temp_directory={_sql_string(str(allowed[1]))}")
        connection.execute("SET enable_external_access=false")
        connection.execute("SET lock_configuration=true")

    def _create_snapshot_views(
        self,
        connection: duckdb.DuckDBPyConnection,
        inventory: dict[str, list[Path]],
    ) -> None:
        for name in table_names():
            identifier = _sql_identifier(name)
            paths = inventory[name]
            if paths:
                files = ", ".join(_sql_string(str(path)) for path in paths)
                available_columns = {
                    str(row[0])
                    for row in connection.execute(
                        f"DESCRIBE SELECT * FROM read_parquet([{files}], union_by_name=true)"
                    ).fetchall()
                }
                projected_columns = ", ".join(
                    (
                        _sql_identifier(field.name)
                        if field.name in available_columns
                        else (
                            f"CAST(NULL AS {_duckdb_type(field.type)}) AS "
                            f"{_sql_identifier(field.name)}"
                        )
                    )
                    for field in schema_for(name)
                )
                connection.execute(
                    f"CREATE TEMP VIEW {identifier} AS "
                    f"SELECT {projected_columns} FROM read_parquet([{files}], union_by_name=true)"
                )
                continue
            columns = ", ".join(
                f"CAST(NULL AS {_duckdb_type(field.type)}) AS {_sql_identifier(field.name)}"
                for field in schema_for(name)
            )
            connection.execute(f"CREATE TEMP VIEW {identifier} AS SELECT {columns} WHERE FALSE")
        conflict = connection.execute(
            "SELECT run_id, run_revision FROM runs "
            "WHERE run_revision IS NOT NULL AND run_manifest_digest IS NOT NULL "
            "GROUP BY run_id, run_revision "
            "HAVING count(DISTINCT run_manifest_digest) > 1 LIMIT 1"
        ).fetchone()
        if conflict is not None:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "The pinned corpus contains conflicting projections for one run revision.",
                run_id=str(conflict[0]),
                details={"run_revision": int(conflict[1])},
            )
        connection.execute(
            "CREATE TEMP VIEW current_runs AS "
            "SELECT * EXCLUDE (revision_order) FROM ("
            "SELECT *, row_number() OVER (PARTITION BY run_id "
            "ORDER BY run_revision DESC NULLS LAST, published_at DESC, "
            "run_manifest_digest DESC NULLS LAST) AS revision_order FROM runs"
            ") WHERE revision_order = 1"
        )

    def _validated_metadata(self) -> dict[str, object]:
        if not self.workspace.paths.catalog.exists():
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "The DuckDB catalog is missing.")
        try:
            connection = duckdb.connect(str(self.workspace.paths.catalog), read_only=True)
            try:
                row = connection.execute(
                    "SELECT built_at FROM flameox_catalog_metadata"
                ).fetchone()
            finally:
                connection.close()
        except duckdb.Error as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "The DuckDB catalog is unreadable or has invalid metadata.",
                remediation=("Run `flameox catalog rebuild`.",),
            ) from exc
        if row is None:
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Catalog metadata is missing.")
        return {"built_at": row[0]}

    def status(self) -> dict[str, object]:
        return self._validated_metadata()
