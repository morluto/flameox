from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from flameox.providers.contracts import ProviderAnalysis, ProviderFailure
from flameox.runtime_errors import DomainError, ErrorCode
from flameox.workers.harness import IsolatedWorkerHarness
from flameox.workers.memray_contract import (
    MEMRAY_WORKER,
    MemrayExtractionLimits,
    MemrayMetricCoverage,
    MemrayWorkerRequest,
    MemrayWorkerResult,
)

_PROTOCOL_OVERHEAD_BYTES = 2 * 1024 * 1024
_EXPECTED_OUTPUTS = {
    "measurements": "measurements.parquet",
    "frames": "frames.parquet",
    "frame_measurements": "frame_measurements.parquet",
    "call_edges": "call_edges.parquet",
    "stacks": "stacks.parquet",
}


class MemrayProvider:
    """Analyze one explicit Memray capture without run or repository ownership."""

    def __init__(self, harness: IsolatedWorkerHarness, project_root: Path) -> None:
        self.harness = harness
        self.project_root = project_root

    def analyze(
        self,
        capability_id: str,
        path: Path,
        input_sha256: str,
        *,
        max_rows: int,
        max_input_bytes: int,
        max_output_bytes: int,
        timeout_seconds: float,
        maximum_rss_bytes: int,
    ) -> ProviderAnalysis:
        metric = (
            "memory.high_watermark" if capability_id == "memory.hotspots" else "memory.retained_end"
        )
        aggregate_limit = min(20_000_000, max(4, max_rows * 4))
        limits = MemrayExtractionLimits(
            max_input_bytes=max_input_bytes,
            max_provider_records=max(1_000, min(100_000, aggregate_limit * 8)),
            max_frames=min(10_000_000, aggregate_limit),
            max_stack_depth=256,
            max_aggregate_rows=aggregate_limit,
            max_unique_edges=min(20_000_000, aggregate_limit),
            max_representative_stacks=min(10_000_000, max_rows),
            max_output_bytes=max_output_bytes,
            wall_time_seconds=timeout_seconds,
            max_worker_memory_bytes=maximum_rss_bytes,
        )
        request = MemrayWorkerRequest(
            artifact_path=str(path),
            run_id=f"direct-{input_sha256[:32]}",
            artifact_id=f"sha256:{input_sha256}",
            workload_cwd=str(self.project_root),
            project_root=str(self.project_root),
            source_state_id=None,
            limits=limits,
        )
        try:
            with self.harness.run_typed_sync_session(
                MEMRAY_WORKER,
                request,
                timeout_seconds=timeout_seconds,
                maximum_rss_bytes=maximum_rss_bytes,
                maximum_writable_growth_bytes=max_output_bytes + _PROTOCOL_OVERHEAD_BYTES,
            ) as (result, job_root):
                files = self._validated_files(result, job_root)
                rows, observed = self._read_rows(files, metric, max_rows=max_rows)
        except DomainError as error:
            if error.code is ErrorCode.UNAVAILABLE_CAPABILITY:
                raise ProviderFailure("UNAVAILABLE_CAPABILITY", error.message) from error
            raise

        coverage = (
            result.coverage.high_watermark
            if capability_id == "memory.hotspots"
            else result.coverage.retained_end
        )
        complete = (
            isinstance(coverage, MemrayMetricCoverage)
            and coverage.complete
            and result.coverage.aggregate_rows_dropped == 0
            and result.coverage.frame_contributions_dropped == 0
            and observed <= len(rows)
        )
        limitations = [
            "Normalized callers are bounded; complete stacks remain in the native Memray profile."
        ]
        if not result.has_native_traces:
            limitations.append("The capture does not contain native stack traces.")
        if not complete:
            limitations.append(
                "Memray normalization or the requested result reached a declared bound."
            )
        return ProviderAnalysis(
            provider_id="memray",
            provider_version=result.reader_version,
            blocks=[
                {
                    "type": "metrics",
                    "values": {
                        "peak_memory_bytes": result.peak_memory_bytes,
                        "retained_end_bytes": result.retained_end_bytes,
                        "capture_records": result.capture_records,
                        "allocation_operations": result.allocation_operations,
                        "total_allocated_bytes": result.total_allocated_bytes,
                    },
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=complete,
            limitations=limitations,
        )

    def _validated_files(self, result: MemrayWorkerResult, job_root: Path) -> dict[str, Path]:
        if len(result.files) != len(_EXPECTED_OUTPUTS):
            raise ProviderFailure("DECODE_FAILURE", "Memray returned an incomplete table set")
        files: dict[str, Path] = {}
        for output in result.files:
            if (
                output.role not in _EXPECTED_OUTPUTS
                or output.relative_path != _EXPECTED_OUTPUTS[output.role]
                or output.media_type != "application/vnd.apache.parquet"
                or output.role in files
            ):
                raise ProviderFailure("DECODE_FAILURE", "Memray returned an invalid table set")
            files[output.role] = self.harness.validate_output_file(job_root, output)
        return files

    @staticmethod
    def _read_rows(
        files: dict[str, Path], metric: str, *, max_rows: int
    ) -> tuple[list[dict[str, Any]], int]:
        frames = {str(row["frame_id"]): row for row in pq.read_table(files["frames"]).to_pylist()}
        selected: list[dict[str, Any]] = []
        observed = 0
        parquet = pq.ParquetFile(files["frame_measurements"])
        for batch in parquet.iter_batches(batch_size=256):
            for measurement in batch.to_pylist():
                if measurement.get("metric") != metric:
                    continue
                observed += 1
                frame = frames.get(str(measurement.get("frame_id")), {})
                selected.append(
                    {
                        "rank": 0,
                        "function": frame.get("function"),
                        "file": frame.get("file"),
                        "line": frame.get("line"),
                        "self_bytes": measurement.get("self_value"),
                        "inclusive_bytes": measurement.get("inclusive_value"),
                        "allocation_count": measurement.get("sample_count"),
                        "frame_id": measurement.get("frame_id"),
                    }
                )
        selected.sort(
            key=lambda row: (
                -int(row["inclusive_bytes"] or 0),
                -int(row["self_bytes"] or 0),
                str(row["frame_id"]),
            )
        )
        selected = selected[:max_rows]
        for rank, row in enumerate(selected, 1):
            row["rank"] = rank
        return selected, observed
