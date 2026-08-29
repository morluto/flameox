from __future__ import annotations

import hashlib
import heapq
import importlib.metadata
import os
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from flameox.atomic import atomic_write_json
from flameox.domain import DomainError, ErrorCode, digest_model
from flameox.evidence.schemas import SCHEMA_MAJOR, schema_for
from flameox.workers.memray_contract import (
    MEMRAY_EXTRACTOR_NAME,
    MEMRAY_EXTRACTOR_VERSION,
    MEMRAY_WORKER,
    MemrayExtractionCoverage,
    MemrayExtractionLimits,
    MemrayMetricCoverage,
    MemrayWorkerProgress,
    MemrayWorkerRequest,
    MemrayWorkerResult,
)
from flameox.workers.protocol import (
    WorkerApplication,
    WorkerContext,
    WorkerFailureKind,
    WorkerOutputFile,
    run_typed_worker,
)


def _normalize(
    filename: str,
    *,
    workload_cwd: Path | None,
    project_root: Path,
    source_state_id: str | None,
) -> tuple[str, str | None, str]:
    if filename.startswith("<") and filename.endswith(">"):
        return filename, None, "partial"
    provider_path = Path(filename)
    if provider_path.is_absolute():
        candidate = provider_path
    else:
        if workload_cwd is None:
            return provider_path.as_posix(), None, "partial"
        candidate = workload_cwd / provider_path
    path = Path(os.path.normpath(candidate))
    if not provider_path.is_absolute() and not path.is_relative_to(project_root):
        return provider_path.as_posix(), None, "partial"
    try:
        normalized = path.relative_to(project_root).as_posix()
        return normalized, source_state_id, "complete" if source_state_id else "partial"
    except ValueError:
        return str(path), None, "partial"


@dataclass(frozen=True)
class _AggregationProjection:
    frame_rows: list[dict[str, Any]]
    aggregates: list[tuple[str, str, int, int, int]]
    frame_contributions_dropped: int
    frame_contribution_bytes_dropped: int
    aggregate_rows_dropped: int
    aggregate_inclusive_bytes_dropped: int


class _AggregationState:
    def __init__(
        self,
        *,
        limits: MemrayExtractionLimits,
        artifact_id: str,
        workload_cwd: Path | None,
        project_root: Path,
        source_state_id: str | None,
        database_path: Path,
    ) -> None:
        self.limits = limits
        self.artifact_id = artifact_id
        self.workload_cwd = workload_cwd
        self.project_root = project_root
        self.source_state_id = source_state_id
        self.database_path = database_path
        self.frame_cache: dict[tuple[str, str, int], str] = {}
        self.contributions = 0
        self.connection = sqlite3.connect(database_path)
        self.connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            CREATE TABLE frames (
                frame_id TEXT PRIMARY KEY,
                function TEXT NOT NULL,
                file TEXT NOT NULL,
                line INTEGER NOT NULL,
                source_state_id TEXT,
                symbolization TEXT NOT NULL
            );
            CREATE TABLE aggregates (
                metric TEXT NOT NULL,
                frame_id TEXT NOT NULL,
                self_value INTEGER NOT NULL,
                inclusive_value INTEGER NOT NULL,
                samples INTEGER NOT NULL,
                occurrences INTEGER NOT NULL,
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
            normalized, frame_source_state_id, symbolization = _normalize(
                raw_frame[1],
                workload_cwd=self.workload_cwd,
                project_root=self.project_root,
                source_state_id=self.source_state_id,
            )
            frame_id = digest_model(
                {
                    "language": "Python",
                    "function": raw_frame[0],
                    "file": normalized,
                    "line": raw_frame[2],
                }
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO frames VALUES (?, ?, ?, ?, ?, ?)",
                (
                    frame_id,
                    raw_frame[0],
                    normalized,
                    raw_frame[2],
                    frame_source_state_id,
                    symbolization,
                ),
            )
            if len(self.frame_cache) < self.limits.max_frames:
                self.frame_cache[raw_frame] = frame_id
        self.connection.execute(
            """
            INSERT INTO aggregates VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(metric, frame_id) DO UPDATE SET
                self_value = self_value + excluded.self_value,
                inclusive_value = inclusive_value + excluded.inclusive_value,
                samples = samples + excluded.samples,
                occurrences = occurrences + 1
            """,
            (
                metric,
                frame_id,
                contribution_bytes if is_leaf else 0,
                contribution_bytes,
                allocations,
            ),
        )
        self.contributions += 1
        if self.contributions % 1_024 == 0:
            self._check_budget()

    def finalize(self) -> _AggregationProjection:
        self.connection.commit()
        self._check_budget()
        self.connection.executescript(
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
            SELECT coalesce(sum(occurrences), 0), coalesce(sum(inclusive_value), 0)
            FROM aggregates
            WHERE frame_id NOT IN (SELECT frame_id FROM selected_frames)
            """
        ).fetchone()
        aggregate_drop = self.connection.execute(
            """
            SELECT count(*), coalesce(sum(a.inclusive_value), 0)
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
            )
        ]
        referenced = {frame_id for _metric, frame_id, *_values in aggregate_rows}
        frame_rows = [
            {
                "frame_id": str(frame_id),
                "language": "Python",
                "function": str(function),
                "module": None,
                "file": str(file),
                "line": int(line),
                "column": None,
                "address": None,
                "build_id": None,
                "module_relative_address": None,
                "inline_chain_id": None,
                "source_state_id": source_state_id,
                "artifact_id": self.artifact_id,
                "inlined": False,
                "symbolization": str(symbolization),
            }
            for frame_id, function, file, line, source_state_id, symbolization
            in self.connection.execute(
                "SELECT frame_id, function, file, line, source_state_id, symbolization "
                "FROM frames ORDER BY frame_id"
            )
            if frame_id in referenced
        ]
        return _AggregationProjection(
            frame_rows=frame_rows,
            aggregates=aggregate_rows,
            frame_contributions_dropped=int(frame_drop[0]),
            frame_contribution_bytes_dropped=int(frame_drop[1]),
            aggregate_rows_dropped=int(aggregate_drop[0]),
            aggregate_inclusive_bytes_dropped=int(aggregate_drop[1]),
        )

    def close(self) -> None:
        self.connection.close()
        self.database_path.unlink(missing_ok=True)

    def _check_budget(self) -> None:
        self.connection.commit()
        if self.database_path.stat().st_size > self.limits.max_output_bytes:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Memray aggregation workspace exceeds the extraction output-byte limit.",
            )


def _aggregate(
    records: Iterable[Any],
    *,
    metric: Literal["memory.high_watermark", "memory.retained_end"],
    state: _AggregationState,
    progress: Callable[[MemrayWorkerProgress], None] | None = None,
) -> tuple[int, MemrayMetricCoverage]:
    total_bytes = 0
    records_seen = 0
    records_selected = 0
    record_bytes_selected = 0
    dropped_stack_frames = 0
    dropped_stack_frame_bytes = 0
    phase: Literal["normalizing_high_watermark", "normalizing_retained_end"] = (
        "normalizing_high_watermark"
        if metric == "memory.high_watermark"
        else "normalizing_retained_end"
    )

    def emit_progress() -> None:
        if progress is not None:
            progress(
                MemrayWorkerProgress(
                    phase=phase,
                    records_seen=records_seen,
                    records_selected=records_selected,
                    record_bytes_seen=total_bytes,
                )
            )

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
        if records_seen % 1_024 == 0:
            emit_progress()

    for size, _ordinal, record in sorted(retained, reverse=True, key=lambda item: item[:2]):
        allocations = int(record.n_allocations)
        records_selected += 1
        record_bytes_selected += size
        for index, (function, filename, line) in enumerate(record.stack_trace()):
            if index >= state.limits.max_stack_depth:
                dropped_stack_frames += 1
                dropped_stack_frame_bytes += size
                continue
            raw_frame = (str(function), str(filename), int(line))
            state.add(
                metric,
                raw_frame,
                contribution_bytes=size,
                allocations=allocations,
                is_leaf=index == 0,
            )
        if records_selected % 1_024 == 0:
            emit_progress()
    emit_progress()
    return total_bytes, MemrayMetricCoverage(
        records_seen=records_seen,
        records_selected=records_selected,
        record_bytes_seen=total_bytes,
        record_bytes_selected=record_bytes_selected,
        dropped_stack_frames=dropped_stack_frames,
        dropped_stack_frame_bytes=dropped_stack_frame_bytes,
    )


def _write_table(
    root: Path,
    name: str,
    rows: list[dict[str, Any]],
    request: MemrayWorkerRequest,
) -> WorkerOutputFile:
    schema = schema_for(name)
    common = {
        "schema_version": SCHEMA_MAJOR,
        "evidence_generation_id": request.generation_id,
        "published_at": request.published_at,
        "extractor_name": MEMRAY_EXTRACTOR_NAME,
        "extractor_version": MEMRAY_EXTRACTOR_VERSION,
    }
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
                [{**common, **row} for row in rows[start : start + 16_384]],
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
        sha256="sha256:" + digest.hexdigest(),
    )


def _handle(request: MemrayWorkerRequest, context: WorkerContext) -> MemrayWorkerResult:
    try:
        import memray
        from memray._memray import compute_statistics
    except ImportError as error:
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "Memray reader is unavailable.",
        ) from error
    try:
        if Path(request.artifact_path).stat().st_size > request.limits.max_input_bytes:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Memray capture exceeds the extraction input-byte limit.",
            )
        reader = memray.FileReader(request.artifact_path)
        metadata = reader.metadata
        progress_path = context.job_root / "progress.json"

        def report(progress: MemrayWorkerProgress) -> None:
            atomic_write_json(progress_path, progress.model_dump(mode="json"))

        report(
            MemrayWorkerProgress(
                phase="computing_statistics",
                records_seen=0,
                records_selected=0,
                record_bytes_seen=0,
            )
        )
        try:
            stats = compute_statistics(
                request.artifact_path,
                report_progress=False,
                num_largest=1,
            )
        except NotImplementedError:
            stats = None
        state = _AggregationState(
            limits=request.limits,
            artifact_id=request.artifact_id,
            workload_cwd=Path(request.workload_cwd) if request.workload_cwd else None,
            project_root=Path(request.project_root),
            source_state_id=request.source_state_id,
            database_path=context.job_root / "aggregation.sqlite",
        )
        try:
            _, high_water_coverage = _aggregate(
                reader.get_high_watermark_allocation_records(),
                metric="memory.high_watermark",
                state=state,
                progress=report,
            )
            retained_end, retained_coverage = _aggregate(
                reader.get_leaked_allocation_records(),
                metric="memory.retained_end",
                state=state,
                progress=report,
            )
            projection = state.finalize()
        finally:
            state.close()
    except (OSError, sqlite3.Error, ValueError) as error:
        diagnostic = str(error)
        raise DomainError(
            (
                ErrorCode.ADAPTER_INCOMPATIBLE
                if "incompatible" in diagnostic.casefold()
                else ErrorCode.ARTIFACT_PARSE_FAILED
            ),
            f"Memray reader rejected the capture: {diagnostic}",
        ) from error

    metrics = [
        ("memory.peak", int(metadata.peak_memory), "bytes", "peak"),
        ("memory.retained_end", retained_end, "bytes", "total"),
        ("memory.capture_records", int(metadata.total_allocations), "count", "total"),
    ]
    if stats is not None:
        metrics.extend(
            (
                (
                    "memory.allocation_operations",
                    int(stats.total_num_allocations),
                    "count",
                    "total",
                ),
                ("memory.allocated_bytes", int(stats.total_memory_allocated), "bytes", "total"),
            )
        )
    measurement_rows: list[dict[str, Any]] = [
        {
            "measurement_id": digest_model(
                {"run_id": request.run_id, "artifact_id": request.artifact_id, "name": name}
            ),
            "run_id": request.run_id,
            "artifact_id": request.artifact_id,
            "name": name,
            "value_int": value,
            "value_float": None,
            "unit": unit,
            "aggregation": aggregation,
            "scope": "process",
            "trial_id": None,
            "worker_id": None,
            "worker_run_index": None,
            "value_index": None,
            "loop_count": None,
            "is_warmup": False,
            "block_id": None,
            "variant_id": None,
            "order_in_block": None,
            "phase": None,
            "dimensions": {},
            "evidence_level": "observed",
        }
        for name, value, unit, aggregation in metrics
    ]
    frame_measurements = [
        {
            "run_id": request.run_id,
            "artifact_id": request.artifact_id,
            "frame_id": frame_id,
            "metric": metric,
            "self_value": self_value,
            "inclusive_value": inclusive_value,
            "unit": "bytes",
            "sample_count": samples,
            "thread_name": None,
            "process_name": None,
            "phase": None,
        }
        for metric, frame_id, self_value, inclusive_value, samples in projection.aggregates
    ]
    frame_rows = projection.frame_rows
    report(
        MemrayWorkerProgress(
            phase="writing_evidence",
            records_seen=(high_water_coverage.records_seen + retained_coverage.records_seen),
            records_selected=(
                high_water_coverage.records_selected + retained_coverage.records_selected
            ),
            record_bytes_seen=(
                high_water_coverage.record_bytes_seen
                + retained_coverage.record_bytes_seen
            ),
        )
    )
    files: list[WorkerOutputFile] = []
    output_bytes = 0
    for name, rows in (
        ("measurements", measurement_rows),
        ("frames", frame_rows),
        ("frame_measurements", frame_measurements),
    ):
        output = _write_table(context.job_root, name, rows, request)
        output_bytes += output.byte_length
        if output_bytes > request.limits.max_output_bytes:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Memray normalized evidence exceeds the extraction output-byte limit.",
            )
        files.append(output)
    return MemrayWorkerResult(
        reader_version=importlib.metadata.version("memray"),
        peak_memory_bytes=int(metadata.peak_memory),
        retained_end_bytes=retained_end,
        allocation_operations=(int(stats.total_num_allocations) if stats is not None else None),
        total_allocated_bytes=(int(stats.total_memory_allocated) if stats is not None else None),
        capture_records=int(metadata.total_allocations),
        has_native_traces=bool(metadata.has_native_traces),
        coverage=MemrayExtractionCoverage(
            high_watermark=high_water_coverage,
            retained_end=retained_coverage,
            frames_published=len(frame_rows),
            aggregate_rows_published=len(projection.aggregates),
            frame_contributions_dropped=projection.frame_contributions_dropped,
            frame_contribution_bytes_dropped=(
                projection.frame_contribution_bytes_dropped
            ),
            aggregate_rows_dropped=projection.aggregate_rows_dropped,
            aggregate_inclusive_bytes_dropped=(
                projection.aggregate_inclusive_bytes_dropped
            ),
            output_bytes=output_bytes,
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
