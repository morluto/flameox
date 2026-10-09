from __future__ import annotations

import pyarrow as pa

SCHEMAS = {
    "frames": pa.schema(
        (
            pa.field("frame_id", pa.string(), nullable=False),
            pa.field("function", pa.string()),
            pa.field("file", pa.string()),
            pa.field("line", pa.int32()),
        )
    ),
    "frame_measurements": pa.schema(
        (
            pa.field("frame_id", pa.string(), nullable=False),
            pa.field("metric", pa.string(), nullable=False),
            pa.field("self_value", pa.int64()),
            pa.field("inclusive_value", pa.int64()),
            pa.field("sample_count", pa.uint64()),
        )
    ),
}
