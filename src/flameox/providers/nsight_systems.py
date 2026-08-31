from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from flameox.providers.contracts import ProviderAnalysis, ProviderFailure


class NsightSystemsParquetProvider:
    """Bounded reads over an explicit Nsight Systems ``parquetdir`` export."""

    def analyze(
        self, path: Path, *, max_rows: int, provider_version: str = "parquetdir-v1"
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
        return ProviderAnalysis(
            provider_id="nsight-systems-parquetdir",
            provider_version=provider_version,
            blocks=[
                {
                    "type": "metrics",
                    "values": {"table_count": len(tables), "row_count": observed},
                },
                {"type": "table", "rows": rows},
            ],
            rows_observed=observed,
            complete=len(rows) >= observed,
            limitations=[
                "Table schemas vary by Nsight Systems version.",
                "Cross-table temporal relationships require provider-qualified columns.",
            ],
        )
