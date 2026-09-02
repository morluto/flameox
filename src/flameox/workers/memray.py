from __future__ import annotations

import hashlib
import heapq
import importlib.metadata
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from flameox.atomic import atomic_write_json
from flameox.canonical import digest_model, sha256_id
from flameox.runtime_errors import DomainError, ErrorCode
from flameox.workers.memray_contract import (
    MEMRAY_WORKER,
    MemrayExtractionCoverage,
    MemrayExtractionLimits,
    MemrayMetricCoverage,
    MemrayMetricCoverageState,
    MemrayMetricUnavailable,
    MemrayWorkerProgress,
    MemrayWorkerRequest,
    MemrayWorkerResult,
)
from flameox.workers.parquet_schemas import schema_for
from flameox.workers.protocol import (
    WorkerApplication,
    WorkerFailureKind,
    WorkerOutputFile,
    run_typed_worker,
)


def _normalize(filename: str) -> tuple[str, str | None, str]:
    if filename.startswith("<") and filename.endswith(">"):
        return filename, None, "partial"
    return Path(filename).as_posix(), None, "partial"


@dataclass(frozen=True)
class _AggregationProjection:
    frame_rows: list[dict[str, Any]]
    aggregates: list[tuple[str, str, int, int, int]]
    edge_rows: list[dict[str, Any]]
    stack_rows: list[dict[str, Any]]
    frame_contributions_dropped: int
    frame_contribution_bytes_dropped: int
    aggregate_rows_dropped: int
    aggregate_inclusive_bytes_dropped: int
    edge_rows_dropped: int
    edge_weight_bytes_dropped: int
    representative_stacks_dropped: int
    representative_stack_weight_bytes_dropped: int


class _AggregationState:
    def __init__(
        self,
        *,
        limits: MemrayExtractionLimits,
        run_id: str,
        artifact_id: str,
    ) -> None:
        self.limits = limits
        self.run_id = run_id
        self.artifact_id = artifact_id
        self.frame_cache: dict[tuple[str, str, int], str] = {}
        self.contributions = 0
        self.pending_stack: tuple[str, tuple[str, ...], int, int] | None = None
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
            CREATE TABLE edges (
                metric TEXT NOT NULL,
                parent_frame_id TEXT NOT NULL,
                child_frame_id TEXT NOT NULL,
                weight_value INTEGER NOT NULL,
                samples INTEGER NOT NULL,
                PRIMARY KEY (metric, parent_frame_id, child_frame_id)
            );
            CREATE TABLE stacks (
                stack_id TEXT PRIMARY KEY,
                metric TEXT NOT NULL,
                frame_ids_json TEXT NOT NULL,
                leaf_frame_id TEXT NOT NULL,
                weight_value INTEGER NOT NULL,
                samples INTEGER NOT NULL
            );
            CREATE TABLE stack_frames (
                stack_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                frame_id TEXT NOT NULL,
                PRIMARY KEY (stack_id, position)
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
    ) -> str:
        frame_id = self.frame_cache.get(raw_frame)
        if frame_id is None:
            normalized, frame_source_state_id, symbolization = _normalize(raw_frame[1])
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
        return frame_id

    def add_stack(
        self,
        metric: str,
        frame_ids: tuple[str, ...],
        *,
        weight_value: int,
        samples: int,
    ) -> None:
        if not frame_ids:
            return
        pending = self.pending_stack
        if pending is not None and pending[:2] == (metric, frame_ids):
            self.pending_stack = (
                metric,
                frame_ids,
                pending[2] + weight_value,
                pending[3] + samples,
            )
            return
        self._flush_stack()
        self.pending_stack = (metric, frame_ids, weight_value, samples)

    def _flush_stack(self) -> None:
        if self.pending_stack is None:
            return
        metric, frame_ids, weight_value, samples = self.pending_stack
        self.pending_stack = None
        for parent_frame_id, child_frame_id in pairwise(frame_ids):
            self.connection.execute(
                """
                INSERT INTO edges VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(metric, parent_frame_id, child_frame_id) DO UPDATE SET
                    weight_value = weight_value + excluded.weight_value,
                    samples = samples + excluded.samples
                """,
                (metric, parent_frame_id, child_frame_id, weight_value, samples),
            )
        stack_id = digest_model(
            {
                "artifact_id": self.artifact_id,
                "metric": metric,
                "frame_ids": frame_ids,
            }
        )
        self.connection.execute(
            """
            INSERT INTO stacks VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(stack_id) DO UPDATE SET
                weight_value = weight_value + excluded.weight_value,
                samples = samples + excluded.samples
            """,
            (
                stack_id,
                metric,
                json.dumps(frame_ids, separators=(",", ":")),
                frame_ids[-1],
                weight_value,
                samples,
            ),
        )
        self.connection.executemany(
            "INSERT OR IGNORE INTO stack_frames VALUES (?, ?, ?)",
            ((stack_id, position, frame_id) for position, frame_id in enumerate(frame_ids)),
        )

    def finalize(self) -> _AggregationProjection:
        self._flush_stack()
        self.connection.commit()
        self._check_budget()
        self.connection.execute(
            """
            CREATE TEMP TABLE selected_frames (frame_id TEXT PRIMARY KEY);
            CREATE TEMP TABLE selected_aggregates (
                metric TEXT NOT NULL,
                frame_id TEXT NOT NULL,
                PRIMARY KEY (metric, frame_id)
            );
            CREATE TEMP TABLE selected_edges (
                metric TEXT NOT NULL,
                parent_frame_id TEXT NOT NULL,
                child_frame_id TEXT NOT NULL,
                PRIMARY KEY (metric, parent_frame_id, child_frame_id)
            );
            CREATE TEMP TABLE selected_stacks (stack_id TEXT PRIMARY KEY);
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
        self.connection.execute(
            """
            INSERT INTO selected_edges
            SELECT e.metric, e.parent_frame_id, e.child_frame_id
            FROM edges AS e
            JOIN selected_frames AS parent ON parent.frame_id = e.parent_frame_id
            JOIN selected_frames AS child ON child.frame_id = e.child_frame_id
            ORDER BY e.weight_value DESC, e.samples DESC, e.metric,
                     e.parent_frame_id, e.child_frame_id
            LIMIT ?
            """,
            (self.limits.max_unique_edges,),
        )
        self.connection.execute(
            """
            INSERT INTO selected_stacks
            SELECT s.stack_id
            FROM stacks AS s
            WHERE NOT EXISTS (
                SELECT 1 FROM stack_frames AS sf
                LEFT JOIN selected_frames AS selected ON selected.frame_id = sf.frame_id
                WHERE sf.stack_id = s.stack_id AND selected.frame_id IS NULL
            )
            ORDER BY s.weight_value DESC, s.samples DESC, s.metric, s.stack_id
            LIMIT ?
            """,
            (self.limits.max_representative_stacks,),
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
        edge_drop = self.connection.execute(
            """
            SELECT count(*), coalesce(sum(e.weight_value), 0)
            FROM edges AS e
            LEFT JOIN selected_edges AS s
              ON s.metric = e.metric
             AND s.parent_frame_id = e.parent_frame_id
             AND s.child_frame_id = e.child_frame_id
            WHERE s.metric IS NULL
            """
        ).fetchone()
        stack_drop = self.connection.execute(
            """
            SELECT count(*), coalesce(sum(s.weight_value), 0)
            FROM stacks AS s
            LEFT JOIN selected_stacks AS selected USING (stack_id)
            WHERE selected.stack_id IS NULL
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
        selected_edges = self.connection.execute(
            """
            SELECT e.metric, e.parent_frame_id, e.child_frame_id,
                   e.weight_value, e.samples
            FROM edges AS e
            JOIN selected_edges AS s
              ON s.metric = e.metric
             AND s.parent_frame_id = e.parent_frame_id
             AND s.child_frame_id = e.child_frame_id
            ORDER BY e.metric, e.parent_frame_id, e.child_frame_id
            """
        )
        selected_edge_rows = selected_edges.fetchall()
        edge_rows = [
            {
                "run_id": self.run_id,
                "artifact_id": self.artifact_id,
                "parent_frame_id": str(parent_frame_id),
                "child_frame_id": str(child_frame_id),
                "metric": str(metric),
                "weight_value": int(weight_value),
                "unit": "bytes",
                "sample_count": int(samples),
            }
            for metric, parent_frame_id, child_frame_id, weight_value, samples in selected_edge_rows
        ]
        selected_stacks = self.connection.execute(
            """
            SELECT s.stack_id, s.metric, s.frame_ids_json, s.leaf_frame_id,
                   s.weight_value, s.samples
            FROM stacks AS s JOIN selected_stacks AS selected USING (stack_id)
            ORDER BY s.metric, s.stack_id
            """
        )
        stack_rows = [
            {
                "stack_id": str(stack_id),
                "run_id": self.run_id,
                "artifact_id": self.artifact_id,
                "frame_ids": tuple(json.loads(str(frame_ids_json))),
                "leaf_frame_id": str(leaf_frame_id),
                "metric": str(metric),
                "weight_value": int(weight_value),
                "unit": "bytes",
                "sample_count": int(samples),
                "start_ns": None,
                "track_id": None,
            }
            for (
                stack_id,
                metric,
                frame_ids_json,
                leaf_frame_id,
                weight_value,
                samples,
            ) in selected_stacks.fetchall()
        ]
        referenced = {frame_id for _metric, frame_id, *_values in aggregate_rows}
        for edge in edge_rows:
            referenced.add(cast(str, edge["parent_frame_id"]))
            referenced.add(cast(str, edge["child_frame_id"]))
        for stack in stack_rows:
            referenced.update(cast(tuple[str, ...], stack["frame_ids"]))
        selected_frames = self.connection.execute(
            "SELECT frame_id, function, file, line, source_state_id, symbolization "
            "FROM frames ORDER BY frame_id"
        )
        selected_frame_rows = selected_frames.fetchall()
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
            for (
                frame_id,
                function,
                file,
                line,
                source_state_id,
                symbolization,
            ) in selected_frame_rows
            if frame_id in referenced
        ]
        assert frame_drop is not None
        assert aggregate_drop is not None
        assert edge_drop is not None
        assert stack_drop is not None
        return _AggregationProjection(
            frame_rows=frame_rows,
            aggregates=aggregate_rows,
            edge_rows=edge_rows,
            stack_rows=stack_rows,
            frame_contributions_dropped=int(frame_drop[0]),
            frame_contribution_bytes_dropped=int(frame_drop[1]),
            aggregate_rows_dropped=int(aggregate_drop[0]),
            aggregate_inclusive_bytes_dropped=int(aggregate_drop[1]),
            edge_rows_dropped=int(edge_drop[0]),
            edge_weight_bytes_dropped=int(edge_drop[1]),
            representative_stacks_dropped=int(stack_drop[0]),
            representative_stack_weight_bytes_dropped=int(stack_drop[1]),
        )

    def close(self) -> None:
        self.connection.close()

    def _check_budget(self) -> None:
        # The isolated worker's address-space limit bounds this in-memory database.
        # Committing here would turn every 1,024 contributions into a separate
        # DuckDB transaction and dominate high-cardinality captures.
        return


def _aggregate(
    records: Iterable[Any],
    *,
    metric: Literal[
        "memory.high_watermark",
        "memory.retained_end",
        "memory.allocated",
        "memory.temporary",
    ],
    state: _AggregationState,
    progress: Callable[[MemrayWorkerProgress], None] | None = None,
) -> tuple[int, MemrayMetricCoverage]:
    total_bytes = 0
    records_seen = 0
    records_selected = 0
    record_bytes_selected = 0
    dropped_stack_frames = 0
    dropped_stack_frame_bytes = 0
    phase: Literal[
        "normalizing_high_watermark",
        "normalizing_retained_end",
        "normalizing_allocation_volume",
        "normalizing_temporary",
    ]
    if metric == "memory.high_watermark":
        phase = "normalizing_high_watermark"
    elif metric == "memory.retained_end":
        phase = "normalizing_retained_end"
    elif metric == "memory.allocated":
        phase = "normalizing_allocation_volume"
    else:
        phase = "normalizing_temporary"

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
        leaf_to_root: list[str] = []
        for index, (function, filename, line) in enumerate(record.stack_trace()):
            if index >= state.limits.max_stack_depth:
                dropped_stack_frames += 1
                dropped_stack_frame_bytes += size
                continue
            raw_frame = (str(function), str(filename), int(line))
            frame_id = state.add(
                metric,
                raw_frame,
                contribution_bytes=size,
                allocations=allocations,
                is_leaf=index == 0,
            )
            leaf_to_root.append(frame_id)
        state.add_stack(
            metric,
            tuple(reversed(leaf_to_root)),
            weight_value=size,
            samples=allocations,
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


def _coverage_progress(coverage: MemrayMetricCoverageState) -> tuple[int, int, int]:
    if isinstance(coverage, MemrayMetricUnavailable):
        return 0, 0, 0
    return coverage.records_seen, coverage.records_selected, coverage.record_bytes_seen


def _write_table(
    root: Path,
    name: str,
    rows: list[dict[str, Any]],
) -> WorkerOutputFile:
    schema = schema_for(name)
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
        progress_path = job_root / "progress.json"

        def report(progress: MemrayWorkerProgress) -> None:
            atomic_write_json(progress_path, progress.model_dump(mode="json"))

        state = _AggregationState(
            limits=request.limits,
            run_id=request.run_id,
            artifact_id=request.artifact_id,
        )
        try:
            with memray.FileReader(request.artifact_path) as reader:
                metadata = reader.metadata
                has_allocation_history = metadata.file_format == memray.FileFormat.ALL_ALLOCATIONS
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
                allocation_coverage: MemrayMetricCoverageState
                allocation_operations = 0

                def allocation_records() -> Iterable[Any]:
                    nonlocal allocation_operations
                    for record in reader.get_allocation_records():
                        if int(record.size) <= 0:
                            continue
                        allocation_operations += int(record.n_allocations)
                        yield record

                try:
                    _allocated_bytes, allocation_coverage = _aggregate(
                        allocation_records(),
                        metric="memory.allocated",
                        state=state,
                        progress=report,
                    )
                except NotImplementedError:
                    allocation_coverage = MemrayMetricUnavailable()
                temporary_coverage: MemrayMetricCoverageState
                try:
                    temporary_allocated, temporary_coverage = _aggregate(
                        reader.get_temporary_allocation_records(
                            threshold=request.limits.temporary_allocation_threshold
                        ),
                        metric="memory.temporary",
                        state=state,
                        progress=report,
                    )
                except NotImplementedError:
                    temporary_allocated = None
                    temporary_coverage = MemrayMetricUnavailable()
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

    metrics = [
        ("memory.peak", int(metadata.peak_memory), "bytes", "peak"),
        ("memory.retained_end", retained_end, "bytes", "total"),
        ("memory.capture_records", int(metadata.total_allocations), "count", "total"),
    ]
    if temporary_allocated is not None:
        metrics.append(("memory.temporary", temporary_allocated, "bytes", "total"))
    if has_allocation_history and isinstance(allocation_coverage, MemrayMetricCoverage):
        metrics.extend(
            (
                (
                    "memory.allocation_operations",
                    allocation_operations,
                    "count",
                    "total",
                ),
                ("memory.allocated_bytes", _allocated_bytes, "bytes", "total"),
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
    aggregate_projection_complete = projection.aggregate_rows_dropped == 0
    for view, coverage in (
        ("high_watermark", high_water_coverage),
        ("retained_end", retained_coverage),
        ("allocation_volume", allocation_coverage),
        ("temporary", temporary_coverage),
    ):
        if isinstance(coverage, MemrayMetricUnavailable):
            continue
        complete = coverage.complete and aggregate_projection_complete
        measurement_rows.append(
            {
                **measurement_rows[0],
                "measurement_id": digest_model(
                    {
                        "run_id": request.run_id,
                        "artifact_id": request.artifact_id,
                        "name": f"memory.frame_coverage.{view}.complete",
                    }
                ),
                "name": f"memory.frame_coverage.{view}.complete",
                "value_int": int(complete),
                "unit": "boolean",
                "aggregation": "status",
            }
        )
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
    allocation_progress = _coverage_progress(allocation_coverage)
    temporary_progress = _coverage_progress(temporary_coverage)
    report(
        MemrayWorkerProgress(
            phase="writing_evidence",
            records_seen=(
                high_water_coverage.records_seen
                + retained_coverage.records_seen
                + allocation_progress[0]
                + temporary_progress[0]
            ),
            records_selected=(
                high_water_coverage.records_selected
                + retained_coverage.records_selected
                + allocation_progress[1]
                + temporary_progress[1]
            ),
            record_bytes_seen=(
                high_water_coverage.record_bytes_seen
                + retained_coverage.record_bytes_seen
                + allocation_progress[2]
                + temporary_progress[2]
            ),
        )
    )
    files: list[WorkerOutputFile] = []
    output_bytes = 0
    for name, rows in (
        ("measurements", measurement_rows),
        ("frames", frame_rows),
        ("frame_measurements", frame_measurements),
        ("call_edges", projection.edge_rows),
        ("stacks", projection.stack_rows),
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
        temporary_allocated_bytes=temporary_allocated,
        allocation_operations=(
            allocation_operations
            if has_allocation_history and isinstance(allocation_coverage, MemrayMetricCoverage)
            else None
        ),
        total_allocated_bytes=(
            _allocated_bytes
            if has_allocation_history and isinstance(allocation_coverage, MemrayMetricCoverage)
            else None
        ),
        capture_records=int(metadata.total_allocations),
        has_native_traces=bool(metadata.has_native_traces),
        coverage=MemrayExtractionCoverage(
            high_watermark=high_water_coverage,
            retained_end=retained_coverage,
            allocation_volume=allocation_coverage,
            temporary=temporary_coverage,
            frames_published=len(frame_rows),
            aggregate_rows_published=len(projection.aggregates),
            frame_contributions_dropped=projection.frame_contributions_dropped,
            frame_contribution_bytes_dropped=(projection.frame_contribution_bytes_dropped),
            aggregate_rows_dropped=projection.aggregate_rows_dropped,
            aggregate_inclusive_bytes_dropped=(projection.aggregate_inclusive_bytes_dropped),
            edge_rows_published=len(projection.edge_rows),
            edge_rows_dropped=projection.edge_rows_dropped,
            edge_weight_bytes_dropped=projection.edge_weight_bytes_dropped,
            representative_stacks_published=len(projection.stack_rows),
            representative_stacks_dropped=projection.representative_stacks_dropped,
            representative_stack_weight_bytes_dropped=(
                projection.representative_stack_weight_bytes_dropped
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
