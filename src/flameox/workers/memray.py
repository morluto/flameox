from __future__ import annotations

import hashlib
import importlib.metadata
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from flameox.domain import DomainError, ErrorCode, digest_model
from flameox.evidence.schemas import SCHEMA_MAJOR, schema_for
from flameox.workers.memray_contract import MEMRAY_WORKER, MemrayWorkerRequest, MemrayWorkerResult
from flameox.workers.protocol import (
    WorkerApplication,
    WorkerContext,
    WorkerFailureKind,
    WorkerOutputFile,
    run_typed_worker,
)


def _normalize(filename: str, project_root: Path) -> str:
    if filename.startswith("<") and filename.endswith(">"):
        return filename
    path = Path(filename).resolve()
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _aggregate(
    records: Iterable[Any],
    *,
    metric: str,
    project_root: Path,
    artifact_id: str,
    frame_rows: dict[str, dict[str, Any]],
    frame_cache: dict[tuple[str, str, int], str],
    aggregates: dict[tuple[str, str], dict[str, int]],
) -> int:
    total_bytes = 0
    for record in records:
        size = int(record.size)
        allocations = int(record.n_allocations)
        total_bytes += size
        for index, (function, filename, line) in enumerate(record.stack_trace()):
            raw_frame = (str(function), str(filename), int(line))
            frame_id = frame_cache.get(raw_frame)
            if frame_id is None:
                normalized = _normalize(raw_frame[1], project_root)
                frame_id = digest_model(
                    {
                        "language": "Python",
                        "function": raw_frame[0],
                        "file": normalized,
                        "line": raw_frame[2],
                    }
                )
                frame_cache[raw_frame] = frame_id
                frame_rows[frame_id] = {
                    "frame_id": frame_id,
                    "language": "Python",
                    "function": raw_frame[0],
                    "module": None,
                    "file": normalized,
                    "line": raw_frame[2],
                    "column": None,
                    "address": None,
                    "build_id": None,
                    "module_relative_address": None,
                    "inline_chain_id": None,
                    "source_state_id": None,
                    "artifact_id": artifact_id,
                    "inlined": False,
                    "symbolization": "complete",
                }
            values = aggregates[(metric, frame_id)]
            values["inclusive"] += size
            values["samples"] += allocations
            if index == 0:
                values["self"] += size
    return total_bytes


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
        "extractor_name": request.extractor_name,
        "extractor_version": request.extractor_version,
    }
    table = pa.Table.from_pylist([{**common, **row} for row in rows], schema=schema)
    path = root / f"{name}.parquet"
    pq.write_table(table, path, compression="zstd", version="2.6", write_statistics=True)
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
    except ImportError as error:
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "Memray reader is unavailable.",
        ) from error
    try:
        reader = memray.FileReader(request.artifact_path)
        metadata = reader.metadata
        frame_rows: dict[str, dict[str, Any]] = {}
        frame_cache: dict[tuple[str, str, int], str] = {}
        aggregates: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"self": 0, "inclusive": 0, "samples": 0}
        )
        _aggregate(
            reader.get_high_watermark_allocation_records(),
            metric="memory.high_watermark",
            project_root=Path(request.project_root),
            artifact_id=request.artifact_id,
            frame_rows=frame_rows,
            frame_cache=frame_cache,
            aggregates=aggregates,
        )
        retained_end = _aggregate(
            reader.get_leaked_allocation_records(),
            metric="memory.retained_end",
            project_root=Path(request.project_root),
            artifact_id=request.artifact_id,
            frame_rows=frame_rows,
            frame_cache=frame_cache,
            aggregates=aggregates,
        )
    except (OSError, ValueError) as error:
        diagnostic = str(error)
        raise DomainError(
            (
                ErrorCode.ADAPTER_INCOMPATIBLE
                if "incompatible" in diagnostic.casefold()
                else ErrorCode.ARTIFACT_PARSE_FAILED
            ),
            f"Memray reader rejected the capture: {diagnostic}",
        ) from error

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
        for name, value, unit, aggregation in (
            ("memory.peak", int(metadata.peak_memory), "bytes", "peak"),
            ("memory.retained_end", retained_end, "bytes", "total"),
            ("memory.total_allocations", int(metadata.total_allocations), "count", "total"),
        )
    ]
    frame_measurements = [
        {
            "run_id": request.run_id,
            "artifact_id": request.artifact_id,
            "frame_id": frame_id,
            "metric": metric,
            "self_value": values["self"],
            "inclusive_value": values["inclusive"],
            "unit": "bytes",
            "sample_count": values["samples"],
            "thread_name": None,
            "process_name": None,
            "phase": None,
        }
        for (metric, frame_id), values in sorted(aggregates.items())
    ]
    files = tuple(
        _write_table(context.job_root, name, rows, request)
        for name, rows in (
            ("measurements", measurement_rows),
            ("frames", list(frame_rows.values())),
            ("frame_measurements", frame_measurements),
        )
    )
    return MemrayWorkerResult(
        reader_version=importlib.metadata.version("memray"),
        peak_memory_bytes=int(metadata.peak_memory),
        retained_end_bytes=retained_end,
        total_allocations=int(metadata.total_allocations),
        frame_count=len(frame_rows),
        has_native_traces=bool(metadata.has_native_traces),
        files=files,
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
