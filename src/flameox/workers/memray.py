from __future__ import annotations

import hashlib
import heapq
import importlib.metadata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from flameox.canonical import digest_model, sha256_id
from flameox.runtime_errors import DomainError, ErrorCode
from flameox.workers.memray_contract import (
    MEMRAY_WORKER,
    MemrayExtractionCoverage,
    MemrayExtractionLimits,
    MemrayMetricCoverage,
    MemrayWorkerRequest,
    MemrayWorkerResult,
)
from flameox.workers.parquet_schemas import SCHEMAS
from flameox.workers.protocol import (
    WorkerApplication,
    WorkerFailureKind,
    WorkerOutputFile,
    run_typed_worker,
)


def _normalize_filename(filename: str) -> str:
    if filename.startswith("<") and filename.endswith(">"):
        return filename
    return Path(filename).as_posix()


@dataclass(frozen=True)
class _AggregationProjection:
    frame_rows: list[dict[str, Any]]
    aggregates: list[tuple[str, str, int, int, int]]
    frame_contributions_dropped: int
    aggregate_rows_dropped: int


class _AggregationState:
    def __init__(
        self,
        *,
        limits: MemrayExtractionLimits,
    ) -> None:
        self.limits = limits
        self.frame_cache: dict[tuple[str, str, int], str] = {}
        self.frame_batch: dict[str, dict[str, Any]] = {}
        self.aggregate_batch: list[dict[str, Any]] = []
        self.connection = duckdb.connect(
            ":memory:",
            config={
                "threads": "1",
                "preserve_insertion_order": "false",
            },
        )
        self.connection.begin()
        self.connection.execute(
            """
            CREATE TABLE frames (
                frame_id TEXT PRIMARY KEY,
                function TEXT NOT NULL,
                file TEXT NOT NULL,
                line INTEGER NOT NULL
            );
            CREATE TABLE aggregates (
                metric TEXT NOT NULL,
                frame_id TEXT NOT NULL,
                self_value BIGINT NOT NULL,
                inclusive_value BIGINT NOT NULL,
                samples UBIGINT NOT NULL,
                occurrences BIGINT NOT NULL,
                PRIMARY KEY (metric, frame_id)
            );
            """
        )

    def add(
        self,
        metric: str,
        raw_frame: tuple[str, str, int],
        *,
        contribution_bytes: int,
        allocations: int,
        is_leaf: bool,
    ) -> None:
        frame_id = self.frame_cache.get(raw_frame)
        if frame_id is None:
            normalized = _normalize_filename(raw_frame[1])
            frame_id = digest_model(
                {
                    "language": "Python",
                    "function": raw_frame[0],
                    "file": normalized,
                    "line": raw_frame[2],
                }
            )
            self.frame_batch[frame_id] = {
                "frame_id": frame_id,
                "function": raw_frame[0],
                "file": normalized,
                "line": raw_frame[2],
            }
            if len(self.frame_cache) < self.limits.max_frames:
                self.frame_cache[raw_frame] = frame_id
        self.aggregate_batch.append(
            {
                "metric": metric,
                "frame_id": frame_id,
                "self_value": contribution_bytes if is_leaf else 0,
                "inclusive_value": contribution_bytes,
                "samples": allocations,
            }
        )
        if len(self.aggregate_batch) >= 1_024:
            self._flush_aggregates()

    def _execute_batch(self, rows: list[dict[str, Any]], query: str) -> None:
        if not rows:
            return
        batch = pa.table(
            {
                name: pa.array(
                    [row[name] for row in rows], type=pa.uint64() if name == "samples" else None
                )
                for name in rows[0]
            }
        )
        self.connection.register("memray_batch", batch)
        try:
            self.connection.execute(query)
        finally:
            self.connection.unregister("memray_batch")

    def _flush_aggregates(self) -> None:
        self._execute_batch(
            list(self.frame_batch.values()),
            "INSERT OR IGNORE INTO frames SELECT * FROM memray_batch",
        )
        self.frame_batch.clear()
        self._execute_batch(
            self.aggregate_batch,
            """
            INSERT INTO aggregates
            SELECT metric, frame_id, sum(self_value), sum(inclusive_value), sum(samples), count(*)
            FROM memray_batch GROUP BY metric, frame_id
            ON CONFLICT(metric, frame_id) DO UPDATE SET
                self_value = self_value + excluded.self_value,
                inclusive_value = inclusive_value + excluded.inclusive_value,
                samples = samples + excluded.samples,
                occurrences = occurrences + excluded.occurrences
            """,
        )
        self.aggregate_batch.clear()

    def finalize(self) -> _AggregationProjection:
        self._flush_aggregates()
        self.connection.commit()
        self.connection.execute(
            """
            CREATE TEMP TABLE selected_frames (frame_id TEXT PRIMARY KEY);
            CREATE TEMP TABLE selected_aggregates (
                metric TEXT NOT NULL,
                frame_id TEXT NOT NULL,
                PRIMARY KEY (metric, frame_id)
            );
            """
        )
        self.connection.execute(
            """
            INSERT INTO selected_frames
            SELECT frame_id FROM aggregates
            GROUP BY frame_id
            ORDER BY max(inclusive_value) DESC, sum(inclusive_value) DESC, frame_id
            LIMIT ?
            """,
            (self.limits.max_frames,),
        )
        self.connection.execute(
            """
            INSERT INTO selected_aggregates
            SELECT a.metric, a.frame_id
            FROM aggregates AS a
            JOIN selected_frames AS f USING (frame_id)
            ORDER BY a.inclusive_value DESC, a.self_value DESC, a.samples DESC,
                     a.metric, a.frame_id
            LIMIT ?
            """,
            (self.limits.max_aggregate_rows,),
        )
        frame_drop = self.connection.execute(
            """
            SELECT coalesce(sum(occurrences), 0)
            FROM aggregates
            WHERE frame_id NOT IN (SELECT frame_id FROM selected_frames)
            """
        ).fetchone()
        aggregate_drop = self.connection.execute(
            """
            SELECT count(*)
            FROM aggregates AS a
            JOIN selected_frames AS f USING (frame_id)
            LEFT JOIN selected_aggregates AS s
              ON s.metric = a.metric AND s.frame_id = a.frame_id
            WHERE s.frame_id IS NULL
            """
        ).fetchone()
        aggregate_rows = [
            (str(metric), str(frame_id), int(self_value), int(inclusive), int(samples))
            for metric, frame_id, self_value, inclusive, samples in self.connection.execute(
                """
                SELECT a.metric, a.frame_id, a.self_value, a.inclusive_value, a.samples
                FROM aggregates AS a
                JOIN selected_aggregates AS s
                  ON s.metric = a.metric AND s.frame_id = a.frame_id
                ORDER BY a.metric, a.frame_id
                """
            ).fetchall()
        ]
        selected_frame_rows = self.connection.execute(
            """
            SELECT frame_id, function, file, line FROM frames
            WHERE frame_id IN (SELECT frame_id FROM selected_aggregates)
            ORDER BY frame_id
            """
        ).fetchall()
        frame_rows = [
            {
                "frame_id": str(frame_id),
                "function": str(function),
                "file": str(file),
                "line": int(line),
            }
            for frame_id, function, file, line in selected_frame_rows
        ]
        assert frame_drop is not None
        assert aggregate_drop is not None
        return _AggregationProjection(
            frame_rows=frame_rows,
            aggregates=aggregate_rows,
            frame_contributions_dropped=int(frame_drop[0]),
            aggregate_rows_dropped=int(aggregate_drop[0]),
        )

    def close(self) -> None:
        self.connection.close()


def _aggregate(
    records: Iterable[Any],
    *,
    metric: Literal[
        "memory.high_watermark",
        "memory.retained_end",
    ],
    state: _AggregationState,
) -> tuple[int, MemrayMetricCoverage]:
    total_bytes = 0
    records_seen = 0
    records_selected = 0
    dropped_stack_frames = 0

    retained: list[tuple[int, int, Any]] = []
    for ordinal, record in enumerate(records):
        size = int(record.size)
        records_seen += 1
        total_bytes += size
        candidate = (size, -ordinal, record)
        if len(retained) < state.limits.max_provider_records:
            heapq.heappush(retained, candidate)
        elif candidate[:2] > retained[0][:2]:
            heapq.heapreplace(retained, candidate)

    for size, _ordinal, record in sorted(retained, reverse=True, key=lambda item: item[:2]):
        allocations = int(record.n_allocations)
        records_selected += 1
        for index, (function, filename, line) in enumerate(record.stack_trace()):
            if index >= state.limits.max_stack_depth:
                dropped_stack_frames += 1
                continue
            raw_frame = (str(function), str(filename), int(line))
            state.add(
                metric,
                raw_frame,
                contribution_bytes=size,
                allocations=allocations,
                is_leaf=index == 0,
            )
    return total_bytes, MemrayMetricCoverage(
        records_seen=records_seen,
        records_selected=records_selected,
        dropped_stack_frames=dropped_stack_frames,
    )


def _write_table(
    root: Path,
    name: str,
    rows: list[dict[str, Any]],
) -> WorkerOutputFile:
    schema = SCHEMAS[name]
    path = root / f"{name}.parquet"
    with pq.ParquetWriter(
        path,
        schema,
        compression="zstd",
        version="2.6",
        write_statistics=True,
    ) as writer:
        for start in range(0, len(rows), 16_384):
            table = pa.Table.from_pylist(
                rows[start : start + 16_384],
                schema=schema,
            )
            writer.write_table(table, row_group_size=16_384)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return WorkerOutputFile(
        role=name,
        relative_path=path.name,
        media_type="application/vnd.apache.parquet",
        byte_length=path.stat().st_size,
        sha256=sha256_id(digest.hexdigest()),
    )


def _handle(request: MemrayWorkerRequest, job_root: Path) -> MemrayWorkerResult:
    try:
        import memray
    except ImportError as error:
        raise DomainError(
            ErrorCode.UNAVAILABLE_CAPABILITY,
            "Memray reader is unavailable.",
        ) from error
    try:
        if Path(request.artifact_path).stat().st_size > request.limits.max_input_bytes:
            raise DomainError(
                ErrorCode.LIMIT_EXCEEDED,
                "Memray capture exceeds the extraction input-byte limit.",
            )
        state = _AggregationState(
            limits=request.limits,
        )
        try:
            with memray.FileReader(request.artifact_path) as reader:
                metadata = reader.metadata
                has_allocation_history = metadata.file_format == memray.FileFormat.ALL_ALLOCATIONS
                if request.metric == "memory.high_watermark":
                    _, metric_coverage = _aggregate(
                        reader.get_high_watermark_allocation_records(),
                        metric=request.metric,
                        state=state,
                    )
                    retained_end = sum(
                        int(record.size) for record in reader.get_leaked_allocation_records()
                    )
                else:
                    retained_end, metric_coverage = _aggregate(
                        reader.get_leaked_allocation_records(),
                        metric=request.metric,
                        state=state,
                    )
                allocation_operations: int | None = None
                allocated_bytes: int | None = None
                if has_allocation_history:
                    try:
                        allocated_bytes = allocation_operations = 0
                        for record in reader.get_allocation_records():
                            size = int(record.size)
                            if size > 0:
                                allocated_bytes += size
                                allocation_operations += int(record.n_allocations)
                    except NotImplementedError:
                        allocated_bytes = allocation_operations = None
                projection = state.finalize()
        finally:
            state.close()
    except (OSError, duckdb.Error, ValueError) as error:
        diagnostic = str(error)
        raise DomainError(
            (
                ErrorCode.UNSUPPORTED_FORMAT
                if "incompatible" in diagnostic.casefold()
                else ErrorCode.DECODE_FAILURE
            ),
            f"Memray reader rejected the capture: {diagnostic}",
        ) from error

    frame_measurements = [
        {
            "frame_id": frame_id,
            "metric": metric,
            "self_value": self_value,
            "inclusive_value": inclusive_value,
            "sample_count": samples,
        }
        for metric, frame_id, self_value, inclusive_value, samples in projection.aggregates
    ]
    frame_rows = projection.frame_rows
    files: list[WorkerOutputFile] = []
    output_bytes = 0
    for name, rows in (
        ("frames", frame_rows),
        ("frame_measurements", frame_measurements),
    ):
        output = _write_table(job_root, name, rows)
        output_bytes += output.byte_length
        if output_bytes > request.limits.max_output_bytes:
            raise DomainError(
                ErrorCode.LIMIT_EXCEEDED,
                "Memray normalized evidence exceeds the extraction output-byte limit.",
            )
        files.append(output)
    return MemrayWorkerResult(
        reader_version=importlib.metadata.version("memray"),
        peak_memory_bytes=int(metadata.peak_memory),
        retained_end_bytes=retained_end,
        allocation_operations=allocation_operations,
        total_allocated_bytes=allocated_bytes,
        capture_records=int(metadata.total_allocations),
        has_native_traces=bool(metadata.has_native_traces),
        coverage=MemrayExtractionCoverage(
            metric=metric_coverage,
            frame_contributions_dropped=projection.frame_contributions_dropped,
            aggregate_rows_dropped=projection.aggregate_rows_dropped,
        ),
        files=tuple(files),
    )


def main() -> int:
    return run_typed_worker(
        WorkerApplication(
            definition=MEMRAY_WORKER,
            handler=_handle,
            invalid_failure=WorkerFailureKind.INPUT_MALFORMED,
            invalid_message="Memray capture is unsupported or invalid",
            caught=(OSError, ValueError, KeyError, TypeError),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
