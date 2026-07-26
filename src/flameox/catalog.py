from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import duckdb
import pyarrow as pa
from pydantic import BaseModel

from flameox.domain.errors import DomainError, ErrorCode
from flameox.evidence.schemas import SCHEMA_MAJOR, SCHEMA_MINOR, schema_for, table_names
from flameox.observability import OperationLogger, elapsed_ms
from flameox.storage.atomic import fsync_directory
from flameox.storage.corpus import CorpusCommit, GenerationManifest
from flameox.storage.workspace import Workspace


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


class Snapshot:
    def __init__(
        self,
        *,
        commit: CorpusCommit,
        connection: duckdb.DuckDBPyConnection,
    ) -> None:
        self.commit = commit
        self.connection = connection

    def execute(
        self,
        sql: str,
        parameters: tuple[object, ...] = (),
    ) -> duckdb.DuckDBPyConnection:
        return self.connection.execute(sql, parameters)

    def interrupt(self) -> None:
        self.connection.interrupt()


class _CancelledBeforeQuery(Exception):
    pass


class Catalog:
    FORMAT_VERSION = 1

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
                    format_version INTEGER NOT NULL,
                    schema_major INTEGER NOT NULL,
                    schema_minor INTEGER NOT NULL,
                    built_at VARCHAR NOT NULL,
                    validated_commit_id VARCHAR NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO flameox_catalog_metadata VALUES (?, ?, ?, ?, ?)",
                (
                    self.FORMAT_VERSION,
                    SCHEMA_MAJOR,
                    SCHEMA_MINOR,
                    datetime.now(UTC).isoformat(),
                    head.commit_id,
                ),
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
            with (
                self.workspace.write_locked(),
                self.workspace.catalog_locked(shared=False),
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

    @contextmanager
    def open_snapshot(self, commit_id: str | None = None) -> Iterator[Snapshot]:
        commit = (
            self.workspace.corpus.read_head()
            if commit_id is None
            else self.workspace.corpus.read_commit(commit_id)
        )
        with (
            self.workspace.retention_locked(shared=True),
            self.workspace.catalog_locked(shared=True),
        ):
            if not self.workspace.paths.catalog.exists():
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "The DuckDB catalog is missing.",
                    remediation=("Run `flameox catalog rebuild`.",),
                )
            connection = duckdb.connect(
                str(self.workspace.paths.catalog),
                read_only=True,
            )
            try:
                inventory = self._inventory(commit)
                self._configure_connection(connection)
                self._create_snapshot_views(connection, inventory)
                yield Snapshot(commit=commit, connection=connection)
            finally:
                connection.close()

    async def run_interruptible[T](
        self,
        operation: Callable[[Snapshot], T],
        *,
        commit_id: str | None = None,
        cleanup_timeout_seconds: float = 5,
        query_name: str | None = None,
    ) -> T:
        pinned_commit_id = (
            self.workspace.corpus.read_head().commit_id if commit_id is None else commit_id
        )
        cancellation_requested = threading.Event()
        holder_lock = threading.Lock()
        active_snapshot: list[Snapshot] = []

        def run() -> T:
            with self.open_snapshot(pinned_commit_id) as snapshot:
                with holder_lock:
                    active_snapshot.append(snapshot)
                try:
                    if cancellation_requested.is_set():
                        raise _CancelledBeforeQuery
                    return operation(snapshot)
                finally:
                    with holder_lock:
                        active_snapshot.clear()

        operation_id = OperationLogger(self.workspace.paths.root).new_id()
        started = time.monotonic()
        name = query_name or getattr(operation, "__name__", "analysis")
        task = asyncio.create_task(asyncio.to_thread(run))
        try:
            result = await asyncio.shield(task)
            rows_returned = getattr(result, "returned", None)
            bytes_returned = (
                len(result.model_dump_json().encode("utf-8"))
                if isinstance(result, BaseModel)
                else None
            )
            OperationLogger(self.workspace.paths.root).emit(
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
            OperationLogger(self.workspace.paths.root).emit(
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
            try:
                deadline = asyncio.get_running_loop().time() + cleanup_timeout_seconds
                while not task.done():
                    with holder_lock:
                        snapshot = active_snapshot[0] if active_snapshot else None
                    if snapshot is not None:
                        with suppress(duckdb.ConnectionException):
                            snapshot.interrupt()
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    await asyncio.wait(
                        {task},
                        timeout=min(0.05, remaining),
                    )
                if task.done():
                    await asyncio.gather(task, return_exceptions=True)
            finally:
                OperationLogger(self.workspace.paths.root).emit(
                    operation_id=operation_id,
                    operation="catalog.query",
                    phase="query cancelled",
                    query_name=name,
                    query_duration_ms=elapsed_ms(started),
                    error_code="cancelled",
                )
                raise cancellation

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
                connection.execute(
                    f"CREATE TEMP VIEW {identifier} AS "
                    f"SELECT * FROM read_parquet([{files}], union_by_name=true)"
                )
                continue
            columns = ", ".join(
                f"CAST(NULL AS {_duckdb_type(field.type)}) AS {_sql_identifier(field.name)}"
                for field in schema_for(name)
            )
            connection.execute(f"CREATE TEMP VIEW {identifier} AS SELECT {columns} WHERE FALSE")

    def _validated_metadata(self) -> dict[str, object]:
        if not self.workspace.paths.catalog.exists():
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "The DuckDB catalog is missing.")
        try:
            connection = duckdb.connect(str(self.workspace.paths.catalog), read_only=True)
            try:
                row = connection.execute(
                    "SELECT format_version, schema_major, schema_minor, "
                    "built_at, validated_commit_id FROM flameox_catalog_metadata"
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
        return {
            "format_version": row[0],
            "schema_major": row[1],
            "schema_minor": row[2],
            "built_at": row[3],
            "validated_commit_id": row[4],
        }

    def status(self) -> dict[str, object]:
        metadata = self._validated_metadata()
        metadata["current_commit_id"] = self.workspace.corpus.read_head().commit_id
        metadata["fresh"] = metadata["validated_commit_id"] == metadata["current_commit_id"]
        return metadata
