from __future__ import annotations

from typing import Any

from flameox.domain import DomainError, ErrorCode
from flameox.workers.perfetto_contract import (
    PERFETTO_WORKER,
    PerfettoExtractRequest,
    PerfettoExtractResult,
    PerfettoSliceRow,
    PerfettoWindowRequest,
    PerfettoWindowResult,
    PerfettoWindowRow,
    PerfettoWorkerRequest,
    PerfettoWorkerResult,
)
from flameox.workers.protocol import (
    WorkerApplication,
    WorkerFailureKind,
    run_typed_worker,
)

_SLICE_QUERY = """
    SELECT
        s.id,
        s.parent_id,
        coalesce(nullif(s.name, ''), '<unnamed>') AS name,
        s.ts,
        s.dur,
        s.track_id,
        s.category,
        coalesce(
            nullif(t.name, ''),
            nullif(tt.name, ''),
            CASE WHEN t.tid IS NOT NULL THEN printf('tid:%d', t.tid) END
        ) AS thread_name,
        coalesce(
            nullif(p.name, ''),
            nullif(pp.name, ''),
            nullif(pt.name, ''),
            CASE
                WHEN coalesce(p.pid, pp.pid) IS NOT NULL
                THEN printf('pid:%d', coalesce(p.pid, pp.pid))
            END
        ) AS process_name,
        max(CASE WHEN a.key = 'args.filename' THEN a.string_value END) AS filename,
        max(CASE WHEN a.key = 'args.line' THEN a.int_value END) AS line,
        max(CASE WHEN a.key IN ('args.Input Dims', 'args.Input Shapes')
            THEN a.string_value END) AS input_shapes,
        sum(CASE WHEN a.key IN ('args.Bytes', 'args.Allocation Bytes')
            THEN a.int_value END) AS allocation_bytes,
        max(CASE WHEN a.key IN ('args.phase', 'args.Phase')
            THEN a.string_value END) AS phase,
        max(CASE WHEN a.key IN (
            'args.correlation', 'args.correlationId', 'args.Correlation ID',
            'args.External id', 'args.external id'
        ) THEN coalesce(a.string_value, cast(a.int_value AS TEXT)) END) AS correlation_id,
        max(CASE WHEN a.key IN ('args.device', 'args.deviceId', 'args.Device')
            THEN coalesce(a.string_value, cast(a.int_value AS TEXT)) END) AS device,
        max(CASE WHEN a.key IN ('args.stream', 'args.streamId', 'args.Stream')
            THEN coalesce(a.string_value, cast(a.int_value AS TEXT)) END) AS stream
    FROM slice AS s
    LEFT JOIN args AS a ON a.arg_set_id = s.arg_set_id
    LEFT JOIN thread_track AS tt ON tt.id = s.track_id
    LEFT JOIN thread AS t ON t.utid = tt.utid
    LEFT JOIN process AS p ON p.upid = t.upid
    LEFT JOIN process_track AS pt ON pt.id = s.track_id
    LEFT JOIN process AS pp ON pp.upid = pt.upid
    WHERE s.dur > 0
    GROUP BY s.id, s.parent_id, s.name, s.ts, s.dur, s.track_id, s.category,
        t.name, tt.name, t.tid, p.name, p.pid, pp.name, pp.pid, pt.name
    ORDER BY s.ts, s.depth, s.id
"""


def _row(row: Any, names: tuple[str, ...]) -> dict[str, object]:
    return {name: getattr(row, name) for name in names}


def _query(request: PerfettoWorkerRequest) -> PerfettoWorkerResult:
    try:
        from perfetto.trace_processor import (
            TraceProcessor,
            TraceProcessorConfig,
            TraceProcessorException,
        )
    except ImportError as exc:
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "Perfetto's Python package is not installed.",
        ) from exc

    processor: Any | None = None
    try:
        processor = TraceProcessor(
            trace=request.artifact_path,
            config=TraceProcessorConfig(
                bin_path=request.binary_path,
                fetch_latest_trace_processor=False,
                load_timeout=30,
                unique_port=True,
            ),
        )
        if isinstance(request, PerfettoExtractRequest):
            max_rows = request.max_rows
            rows = list(
                processor.query(
                    f"SELECT * FROM ({_SLICE_QUERY}) AS bounded_slices LIMIT {max_rows + 1:d}"
                )
            )
            return PerfettoExtractResult(
                truncated=len(rows) > max_rows,
                rows=tuple(
                    PerfettoSliceRow.model_validate(
                        _row(
                            row,
                            (
                                "id",
                                "parent_id",
                                "name",
                                "ts",
                                "dur",
                                "track_id",
                                "category",
                                "thread_name",
                                "process_name",
                                "filename",
                                "line",
                                "input_shapes",
                                "allocation_bytes",
                                "phase",
                                "correlation_id",
                                "device",
                                "stream",
                            ),
                        )
                    )
                    for row in rows[:max_rows]
                ),
            )
        if isinstance(request, PerfettoWindowRequest):
            start_ns = request.start_ns
            end_ns = request.end_ns
            limit = request.limit
            predicate = f"ts < {end_ns:d} AND ts + dur > {start_ns:d} AND dur >= 0"
            count_rows = list(
                processor.query(f"SELECT count(*) AS total FROM slice WHERE {predicate}")
            )
            after_ts = request.after_ts
            after_id = request.after_id
            page_predicate = predicate
            if after_ts is not None and after_id is not None:
                page_predicate += (
                    f" AND (ts > {int(str(after_ts)):d} OR "
                    f"(ts = {int(str(after_ts)):d} AND id > {int(str(after_id)):d}))"
                )
            rows = list(
                processor.query(
                    "SELECT id, parent_id, "
                    "coalesce(nullif(name, ''), '<unnamed>') AS name, "
                    "category, ts, dur, track_id FROM slice WHERE "
                    f"{page_predicate} ORDER BY ts, id LIMIT {limit + 1:d}"
                )
            )
            return PerfettoWindowResult(
                total=int(count_rows[0].total) if count_rows else 0,
                rows=tuple(
                    PerfettoWindowRow.model_validate(
                        _row(
                            row,
                            ("id", "parent_id", "name", "category", "ts", "dur", "track_id"),
                        )
                    )
                    for row in rows
                ),
            )
        raise AssertionError("unreachable validated Perfetto operation")
    except (TraceProcessorException, OSError, ValueError, RuntimeError) as exc:
        raise DomainError(
            ErrorCode.ARTIFACT_PARSE_FAILED,
            f"Perfetto Trace Processor failed: {type(exc).__name__}",
        ) from exc
    finally:
        if processor is not None:
            processor.close()


def main() -> int:
    return run_typed_worker(
        WorkerApplication(
            definition=PERFETTO_WORKER,
            handler=lambda request, _job_root: _query(request),
            invalid_failure=WorkerFailureKind.INVALID_REQUEST,
            invalid_message="Perfetto worker request is invalid",
            caught=(OSError, ValueError),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
