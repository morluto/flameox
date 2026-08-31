from __future__ import annotations

from typing import Any

import pyarrow as pa


def _schema(name: str, fields: tuple[Any, ...]) -> pa.Schema:
    return pa.schema(fields, metadata={b"flameox.table": name.encode()})


SCHEMAS = {
    "measurements": _schema(
        "measurements",
        (
            pa.field("measurement_id", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("artifact_id", pa.string()),
            pa.field("name", pa.string(), nullable=False),
            pa.field("value_int", pa.int64()),
            pa.field("value_float", pa.float64()),
            pa.field("value_uint", pa.uint64()),
            pa.field("value_kind", pa.string()),
            pa.field("unit", pa.string(), nullable=False),
            pa.field("aggregation", pa.string(), nullable=False),
            pa.field("scope", pa.string(), nullable=False),
            pa.field("trial_id", pa.string()),
            pa.field("worker_id", pa.string()),
            pa.field("worker_run_index", pa.int32()),
            pa.field("value_index", pa.int32()),
            pa.field("loop_count", pa.uint64()),
            pa.field("is_warmup", pa.bool_(), nullable=False),
            pa.field("block_id", pa.string()),
            pa.field("variant_id", pa.string()),
            pa.field("order_in_block", pa.int32()),
            pa.field("phase", pa.string()),
            pa.field("dimensions", pa.map_(pa.string(), pa.string())),
            pa.field("evidence_level", pa.string(), nullable=False),
        ),
    ),
    "frames": _schema(
        "frames",
        (
            pa.field("frame_id", pa.string(), nullable=False),
            pa.field("language", pa.string()),
            pa.field("function", pa.string()),
            pa.field("module", pa.string()),
            pa.field("file", pa.string()),
            pa.field("line", pa.int32()),
            pa.field("column", pa.int32()),
            pa.field("address", pa.uint64()),
            pa.field("build_id", pa.string()),
            pa.field("module_relative_address", pa.uint64()),
            pa.field("inline_chain_id", pa.string()),
            pa.field("source_state_id", pa.string()),
            pa.field("artifact_id", pa.string()),
            pa.field("inlined", pa.bool_()),
            pa.field("symbolization", pa.string(), nullable=False),
        ),
    ),
    "frame_measurements": _schema(
        "frame_measurements",
        (
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("artifact_id", pa.string(), nullable=False),
            pa.field("frame_id", pa.string(), nullable=False),
            pa.field("metric", pa.string(), nullable=False),
            pa.field("self_value", pa.int64()),
            pa.field("inclusive_value", pa.int64()),
            pa.field("unit", pa.string(), nullable=False),
            pa.field("sample_count", pa.uint64()),
            pa.field("thread_name", pa.string()),
            pa.field("process_name", pa.string()),
            pa.field("phase", pa.string()),
        ),
    ),
    "call_edges": _schema(
        "call_edges",
        (
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("artifact_id", pa.string(), nullable=False),
            pa.field("parent_frame_id", pa.string(), nullable=False),
            pa.field("child_frame_id", pa.string(), nullable=False),
            pa.field("metric", pa.string(), nullable=False),
            pa.field("weight_value", pa.int64(), nullable=False),
            pa.field("unit", pa.string(), nullable=False),
            pa.field("sample_count", pa.uint64(), nullable=False),
        ),
    ),
    "stacks": _schema(
        "stacks",
        (
            pa.field("stack_id", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("artifact_id", pa.string(), nullable=False),
            pa.field("frame_ids", pa.list_(pa.string()), nullable=False),
            pa.field("leaf_frame_id", pa.string(), nullable=False),
            pa.field("metric", pa.string(), nullable=False),
            pa.field("weight_value", pa.int64(), nullable=False),
            pa.field("unit", pa.string(), nullable=False),
            pa.field("sample_count", pa.uint64(), nullable=False),
            pa.field("start_ns", pa.int64()),
            pa.field("track_id", pa.uint64()),
        ),
    ),
}


def schema_for(name: str) -> pa.Schema:
    return SCHEMAS[name]
