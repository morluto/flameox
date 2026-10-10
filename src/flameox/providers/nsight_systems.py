from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from flameox.filesystem import open_binary
from flameox.providers.contracts import ProviderAnalysis, ProviderFailure

_OPERATION_TABLE_PREFIXES = (
    "CUDA_API",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_DRIVER",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    "NVTX",
    "OSRT",
    "OS_RUNTIME",
    "OPENACC",
    "OPENMP",
    "MPI",
    "VULKAN",
    "DX12",
)
_ACCELERATOR_TABLE_PREFIXES = (
    "CUDA_GPU_",
    "CUDA_KERNEL",
    "CUDA_MEM",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
)
_LIFECYCLE_TABLE_PREFIXES = (
    "PROCESS",
    "THREAD",
    "SESSION",
    "TARGET",
    "CONTEXT_SWITCH",
)


class NsightSystemsParquetProvider:
    """Bounded reads over an explicit Nsight Systems ``parquetdir`` export."""

    def analyze(
        self,
        path: Path,
        *,
        operation: str,
        max_rows: int,
        provider_version: str = "parquetdir-v1",
    ) -> ProviderAnalysis:
        if not path.is_dir():
            raise ProviderFailure(
                "UNSUPPORTED_FORMAT", "Nsight Systems Parquet evidence must be a directory"
            )
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise ProviderFailure(
                "UNSUPPORTED_FORMAT", "Nsight Systems Parquet directory contains no tables"
            )
        if operation == "inspect_gpu_launches":
            files = [
                file for file in files if file.stem.upper().startswith(_ACCELERATOR_TABLE_PREFIXES)
            ]
        elif operation == "summarize_trace_operations":
            files = [
                file for file in files if file.stem.upper().startswith(_OPERATION_TABLE_PREFIXES)
            ]
        elif operation == "summarize_trace_lifecycle":
            files = [
                file for file in files if file.stem.upper().startswith(_LIFECYCLE_TABLE_PREFIXES)
            ]
        rows: list[dict[str, Any]] = []
        observed = 0
        tables: list[str] = []
        try:
            for file in files:
                with open_binary(file) as stream:
                    parquet = pq.ParquetFile(stream)
                    tables.append(file.stem)
                    observed += parquet.metadata.num_rows
                    if len(rows) >= max_rows:
                        continue
                    for batch in parquet.iter_batches(batch_size=min(256, max_rows - len(rows))):
                        for value in batch.to_pylist():
                            normalized = json.loads(json.dumps(value, default=str))
                            normalized = (
                                {"value": normalized} if "table" in normalized else normalized
                            )
                            rows.append({"table": file.stem, **normalized})
                            if len(rows) >= max_rows:
                                break
                        if len(rows) >= max_rows:
                            break
        except (OSError, pa.ArrowException) as error:
            raise ProviderFailure(
                "DECODE_FAILURE", "Nsight Systems Parquet table is invalid"
            ) from error
        no_accelerator_activity = operation == "inspect_gpu_launches" and observed == 0
        limitations = [
            "Table schemas vary by Nsight Systems version.",
            "Cross-table temporal relationships require provider-qualified columns.",
        ]
        if no_accelerator_activity:
            limitations.append("no_accelerator_activity_observed")
        if operation in {"summarize_trace_operations", "summarize_trace_lifecycle"} and not files:
            limitations.append(f"no_{operation.removeprefix('summarize_trace_')}_activity_observed")
        metrics: dict[str, Any] = {"table_count": len(tables), "row_count": observed}
        if operation == "inspect_gpu_launches":
            metrics["accelerator_activity_observed"] = not no_accelerator_activity
        elif operation == "summarize_trace_operations":
            metrics["operation_row_count"] = observed
        elif operation == "summarize_trace_lifecycle":
            metrics["lifecycle_row_count"] = observed
        return ProviderAnalysis(
            provider_id="nsight-systems-parquetdir",
            provider_version=provider_version,
            blocks=[
                {
                    "type": "metrics",
                    "values": metrics,
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=len(rows) >= observed,
            limitations=limitations,
        )
