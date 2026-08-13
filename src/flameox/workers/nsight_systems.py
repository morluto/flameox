from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import cast
from urllib.parse import quote

from pydantic import JsonValue

from flameox.domain import DomainError, ErrorCode
from flameox.workers.nsight_systems_contract import (
    NSIGHT_SYSTEMS_WORKER,
    NsightSystemsWorkerRequest,
    NsightSystemsWorkerResult,
)
from flameox.workers.protocol import (
    WorkerApplication,
    WorkerContext,
    WorkerFailureKind,
    run_typed_worker,
)


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _column(columns: dict[str, str], *candidates: str) -> str | None:
    for candidate in candidates:
        found = columns.get(candidate.casefold())
        if found is not None:
            return found
    return None


def _required_column(table: str, columns: dict[str, str], *candidates: str) -> str:
    found = _column(columns, *candidates)
    if found is None:
        raise ValueError(f"Unsupported Nsight Systems schema: {table} lacks one of {candidates!r}")
    return found


def _table_columns(connection: sqlite3.Connection, table: str) -> dict[str, str]:
    rows = connection.execute(f"PRAGMA table_info({_identifier(table)})").fetchall()
    return {str(row[1]).casefold(): str(row[1]) for row in rows}


def _rows(
    connection: sqlite3.Connection,
    table: str,
    selected: dict[str, str | None],
    *,
    limit: int,
) -> tuple[list[dict[str, object]], bool]:
    available = {key: value for key, value in selected.items() if value is not None}
    expressions = ", ".join(
        f"{_identifier(value)} AS {_identifier(key)}" for key, value in available.items()
    )
    ordering = ", ".join(_identifier(key) for key in available)
    query = f"SELECT {expressions} FROM {_identifier(table)} ORDER BY {ordering} LIMIT ?"
    records = connection.execute(query, (limit + 1,)).fetchall()
    return (
        [{key: record[index] for index, key in enumerate(available)} for record in records[:limit]],
        len(records) > limit,
    )


def _name(value: object, strings: dict[int, str], fallback: str) -> str:
    if isinstance(value, int):
        return strings.get(value, f"{fallback}:{value}")
    if value not in {None, ""}:
        return str(value)
    return fallback


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str | bytes | bytearray):
        raise ValueError(f"expected an integer-compatible SQLite value, got {type(value).__name__}")
    return int(value)


def _string_values(
    connection: sqlite3.Connection,
    tables: dict[str, str],
    *,
    limit: int,
) -> tuple[dict[int, str], bool, str | None]:
    table = tables.get("stringids")
    if table is None:
        return {}, False, None
    columns = _table_columns(connection, table)
    id_column = _column(columns, "id")
    value_column = _column(columns, "value", "string")
    if id_column is None or value_column is None:
        return {}, False, table
    rows = connection.execute(
        f"SELECT {_identifier(id_column)}, {_identifier(value_column)} "
        f"FROM {_identifier(table)} ORDER BY {_identifier(id_column)} LIMIT ?",
        (limit + 1,),
    ).fetchall()
    values = {
        _integer(identifier): str(value)
        for identifier, value in rows[:limit]
        if identifier is not None and value is not None
    }
    return values, len(rows) > limit, table


def _api_events(
    connection: sqlite3.Connection,
    table: str,
    strings: dict[int, str],
    *,
    category: str,
    limit: int,
) -> tuple[list[dict[str, object]], bool]:
    columns = _table_columns(connection, table)
    rows, truncated = _rows(
        connection,
        table,
        {
            "start": _required_column(table, columns, "start"),
            "end": _required_column(table, columns, "end"),
            "name": _required_column(table, columns, "nameId", "name"),
            "correlation": _column(columns, "correlationId", "correlation"),
            "thread": _column(columns, "globalTid"),
            "process": _column(columns, "processId", "pid"),
        },
        limit=limit,
    )
    events: list[dict[str, object]] = []
    for row in rows:
        start = _integer(row["start"])
        end = _integer(row["end"])
        events.append(
            {
                "name": _name(row.get("name"), strings, category),
                "category": category,
                "start_ns": start,
                "duration_ns": max(0, end - start),
                "correlation_id": row.get("correlation"),
                "thread": row.get("thread"),
                "process": row.get("process"),
            }
        )
    return events, truncated


def _memory_events(
    connection: sqlite3.Connection,
    table: str,
    *,
    category: str,
    limit: int,
) -> tuple[list[dict[str, object]], bool, bool]:
    columns = _table_columns(connection, table)
    start_column = _column(columns, "start")
    end_column = _column(columns, "end")
    if start_column is None or end_column is None:
        return [], False, False
    rows, truncated = _rows(
        connection,
        table,
        {
            "start": start_column,
            "end": end_column,
            "bytes": _column(columns, "bytes", "size"),
            "copy_kind": _column(columns, "copyKind", "kind"),
            "value": _column(columns, "value"),
            "device": _column(columns, "deviceId", "device"),
            "context": _column(columns, "contextId", "context"),
            "stream": _column(columns, "streamId", "stream"),
        },
        limit=limit,
    )
    events: list[dict[str, object]] = []
    for row in rows:
        start = _integer(row["start"])
        end = _integer(row["end"])
        events.append(
            {
                "name": (
                    f"cuda_memcpy:{row.get('copy_kind', 'unknown')}"
                    if category == "memcpy"
                    else "cuda_memset"
                ),
                "category": category,
                "start_ns": start,
                "duration_ns": max(0, end - start),
                "bytes": row.get("bytes"),
                "copy_kind": row.get("copy_kind"),
                "value": row.get("value"),
                "device": row.get("device"),
                "context": row.get("context"),
                "stream": row.get("stream"),
            }
        )
    return events, truncated, True


def _graph_launch_events(
    connection: sqlite3.Connection,
    table: str,
    *,
    limit: int,
) -> tuple[list[dict[str, object]], bool]:
    columns = _table_columns(connection, table)
    rows, truncated = _rows(
        connection,
        table,
        {
            "start": _required_column(table, columns, "start"),
            "end": _required_column(table, columns, "end"),
            "correlation": _column(columns, "correlationId", "correlation"),
            "device": _column(columns, "deviceId", "device"),
            "context": _column(columns, "contextId", "context"),
            "stream": _column(columns, "streamId", "stream"),
            "process": _column(columns, "globalPid", "processId", "pid"),
        },
        limit=limit,
    )
    events: list[dict[str, object]] = []
    for row in rows:
        start = _integer(row["start"])
        end = _integer(row["end"])
        events.append(
            {
                "name": "cudaGraphLaunch",
                "category": "cuda_runtime",
                "start_ns": start,
                "duration_ns": max(0, end - start),
                "correlation_id": row.get("correlation"),
                "device": row.get("device"),
                "context": row.get("context"),
                "stream": row.get("stream"),
                "process": row.get("process"),
            }
        )
    return events, truncated


def _extract(path: Path, *, limit: int) -> dict[str, object]:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
        ).fetchall()
        tables = {str(row[0]).casefold(): str(row[0]) for row in table_rows}
        runtime_table = tables.get("cupti_activity_kind_runtime")
        kernel_table = tables.get("cupti_activity_kind_kernel")
        if runtime_table is None or kernel_table is None:
            raise ValueError(
                "Unsupported Nsight Systems schema: CUDA runtime and kernel tables are required"
            )
        schema = {
            table: sorted(_table_columns(connection, table).values())
            for table in sorted(tables.values())
        }
        schema_fingerprint = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(schema, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest()
        )

        strings, strings_truncated, string_table = _string_values(
            connection,
            tables,
            limit=limit,
        )

        events: list[dict[str, object]] = []
        truncated_tables: list[str] = []
        if strings_truncated and string_table is not None:
            truncated_tables.append(string_table)
        runtime_events, truncated = _api_events(
            connection,
            runtime_table,
            strings,
            category="cuda_runtime",
            limit=limit,
        )
        events.extend(runtime_events)
        if truncated:
            truncated_tables.append(runtime_table)

        graph_table = tables.get("cupti_activity_kind_graph_trace")
        runtime_has_graph_launch = any(
            "graphlaunch"
            in "".join(
                character for character in str(event["name"]).casefold() if character.isalnum()
            )
            for event in runtime_events
        )
        if graph_table is not None and not runtime_has_graph_launch:
            graph_events, truncated = _graph_launch_events(
                connection,
                graph_table,
                limit=limit,
            )
            events.extend(graph_events)
            if truncated:
                truncated_tables.append(graph_table)

        driver_table = tables.get("cupti_activity_kind_driver")
        if driver_table is not None:
            driver_events, truncated = _api_events(
                connection,
                driver_table,
                strings,
                category="cuda_driver",
                limit=limit,
            )
            events.extend(driver_events)
            if truncated:
                truncated_tables.append(driver_table)

        kernel_columns = _table_columns(connection, kernel_table)
        kernel_rows, truncated = _rows(
            connection,
            kernel_table,
            {
                "start": _required_column(kernel_table, kernel_columns, "start"),
                "end": _required_column(kernel_table, kernel_columns, "end"),
                "name": _required_column(
                    kernel_table,
                    kernel_columns,
                    "demangledName",
                    "shortName",
                    "name",
                ),
                "correlation": _column(kernel_columns, "correlationId", "correlation"),
                "device": _column(kernel_columns, "deviceId", "device"),
                "context": _column(kernel_columns, "contextId", "context"),
                "stream": _column(kernel_columns, "streamId", "stream"),
            },
            limit=limit,
        )
        if truncated:
            truncated_tables.append(kernel_table)
        for row in kernel_rows:
            start = _integer(row["start"])
            end = _integer(row["end"])
            events.append(
                {
                    "name": _name(row.get("name"), strings, "cuda_kernel"),
                    "category": "kernel",
                    "start_ns": start,
                    "duration_ns": max(0, end - start),
                    "correlation_id": row.get("correlation"),
                    "device": row.get("device"),
                    "context": row.get("context"),
                    "stream": row.get("stream"),
                }
            )

        coverage = {
            "cuda_runtime": True,
            "cuda_driver": driver_table is not None,
            "cuda_kernels": True,
            "nvtx": False,
            "memory_copies": False,
            "memory_sets": False,
            "correlation_ids": any(event.get("correlation_id") is not None for event in events),
            "stream_identity": any(event.get("stream") is not None for event in events),
            "thread_identity": any(event.get("thread") is not None for event in events),
            "process_identity": any(event.get("process") is not None for event in events),
        }
        nvtx_table = tables.get("nvtx_events")
        if nvtx_table is not None:
            columns = _table_columns(connection, nvtx_table)
            start_column = _column(columns, "start")
            end_column = _column(columns, "end")
            text_column = _column(columns, "text", "name")
            text_id_column = _column(columns, "textId")
            if (
                start_column is not None
                and end_column is not None
                and (text_column is not None or text_id_column is not None)
            ):
                nvtx_rows, truncated = _rows(
                    connection,
                    nvtx_table,
                    {
                        "start": start_column,
                        "end": end_column,
                        "name_text": text_column,
                        "name_id": text_id_column,
                        "thread": _column(columns, "globalTid"),
                        "process": _column(columns, "processId", "pid"),
                    },
                    limit=limit,
                )
                if truncated:
                    truncated_tables.append(nvtx_table)
                for row in nvtx_rows:
                    start = _integer(row["start"])
                    end = _integer(row["end"])
                    events.append(
                        {
                            "name": _name(
                                row.get("name_text")
                                if row.get("name_text") not in {None, ""}
                                else row.get("name_id"),
                                strings,
                                "nvtx",
                            ),
                            "category": "nvtx",
                            "start_ns": start,
                            "duration_ns": max(0, end - start),
                            "thread": row.get("thread"),
                            "process": row.get("process"),
                        }
                    )
                coverage["nvtx"] = True

        memcpy_table = tables.get("cupti_activity_kind_memcpy")
        if memcpy_table is not None:
            memory_events, truncated, supported = _memory_events(
                connection,
                memcpy_table,
                category="memcpy",
                limit=limit,
            )
            events.extend(memory_events)
            if truncated:
                truncated_tables.append(memcpy_table)
            coverage["memory_copies"] = supported

        memset_table = tables.get("cupti_activity_kind_memset")
        if memset_table is not None:
            memory_events, truncated, supported = _memory_events(
                connection,
                memset_table,
                category="memset",
                limit=limit,
            )
            events.extend(memory_events)
            if truncated:
                truncated_tables.append(memset_table)
            coverage["memory_sets"] = supported

        return {
            "ok": True,
            "schema_fingerprint": schema_fingerprint,
            "tables": sorted(tables.values()),
            "events": sorted(
                events,
                key=lambda event: (
                    _integer(event["start_ns"]),
                    str(event["category"]),
                    json.dumps(event, allow_nan=False, separators=(",", ":"), sort_keys=True),
                ),
            ),
            "coverage": coverage,
            "truncated_tables": sorted(set(truncated_tables)),
        }
    finally:
        connection.close()


def _handle(
    request: NsightSystemsWorkerRequest,
    _context: WorkerContext,
) -> NsightSystemsWorkerResult:
    try:
        result = _extract(Path(request.artifact_path), limit=request.max_rows_per_table)
    except ValueError as exc:
        if str(exc).startswith("Unsupported Nsight Systems schema:"):
            raise DomainError(ErrorCode.ARTIFACT_PARSE_FAILED, str(exc)) from exc
        raise
    return NsightSystemsWorkerResult(
        schema_fingerprint=cast(str, result["schema_fingerprint"]),
        tables=tuple(cast(list[str], result["tables"])),
        events=cast(
            tuple[dict[str, JsonValue], ...],
            tuple(cast(list[dict[str, object]], result["events"])),
        ),
        coverage=cast(dict[str, bool], result["coverage"]),
        truncated_tables=tuple(cast(list[str], result["truncated_tables"])),
    )


def main() -> int:
    return run_typed_worker(
        WorkerApplication(
            definition=NSIGHT_SYSTEMS_WORKER,
            handler=_handle,
            invalid_failure=WorkerFailureKind.INPUT_MALFORMED,
            invalid_message="Nsight Systems structured export is unsupported or invalid",
            caught=(OSError, sqlite3.DatabaseError, ValueError, KeyError, TypeError),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
