from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from flameox.providers.contracts import ProviderAnalysis, ProviderFailure

_OPERATION_TABLE_PREFIXES = (
    "CUDA_API",
    "NVTX",
    "OS_RUNTIME",
    "OPENACC",
    "OPENMP",
    "MPI",
    "VULKAN",
    "DX12",
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
        capability_id: str,
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
        if capability_id == "gpu.launches":
            files = [file for file in files if file.stem.upper().startswith(("CUDA_", "CUPTI_"))]
        elif capability_id == "trace.operations":
            files = [
                file for file in files if file.stem.upper().startswith(_OPERATION_TABLE_PREFIXES)
            ]
        elif capability_id == "trace.lifecycle":
            files = [
                file for file in files if file.stem.upper().startswith(_LIFECYCLE_TABLE_PREFIXES)
            ]
        rows: list[dict[str, Any]] = []
        observed = 0
        tables: list[str] = []
        for file in files:
            parquet = pq.ParquetFile(file)
            tables.append(file.stem)
            observed += parquet.metadata.num_rows
            if len(rows) >= max_rows:
                continue
            for batch in parquet.iter_batches(batch_size=min(256, max_rows - len(rows))):
                for value in batch.to_pylist():
                    normalized = json.loads(json.dumps(value, default=str))
                    rows.append({"table": file.stem, **normalized})
                    if len(rows) >= max_rows:
                        break
                if len(rows) >= max_rows:
                    break
        no_accelerator_activity = capability_id == "gpu.launches" and observed == 0
        limitations = [
            "Table schemas vary by Nsight Systems version.",
            "Cross-table temporal relationships require provider-qualified columns.",
        ]
        if no_accelerator_activity:
            limitations.append("no_accelerator_activity_observed")
        if capability_id in {"trace.operations", "trace.lifecycle"} and not files:
            limitations.append(f"no_{capability_id.removeprefix('trace.')}_activity_observed")
        metrics: dict[str, Any] = {"table_count": len(tables), "row_count": observed}
        if capability_id == "gpu.launches":
            metrics["accelerator_activity_observed"] = not no_accelerator_activity
        elif capability_id == "trace.operations":
            metrics["operation_row_count"] = observed
        elif capability_id == "trace.lifecycle":
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
