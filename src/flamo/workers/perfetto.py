from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

_SLICE_QUERY = """
    SELECT
        s.id,
        s.parent_id,
        coalesce(nullif(s.name, ''), '<unnamed>') AS name,
        s.ts,
        s.dur,
        s.track_id,
        s.category,
        max(CASE WHEN a.key = 'args.filename' THEN a.string_value END) AS filename,
        max(CASE WHEN a.key = 'args.line' THEN a.int_value END) AS line
    FROM slice AS s
    LEFT JOIN args AS a ON a.arg_set_id = s.arg_set_id
    WHERE s.dur > 0
    GROUP BY s.id, s.parent_id, s.name, s.ts, s.dur, s.track_id, s.category
    ORDER BY s.ts, s.depth, s.id
"""


def _row(row: Any, names: tuple[str, ...]) -> dict[str, object]:
    return {name: getattr(row, name) for name in names}


def _query(request: dict[str, object]) -> dict[str, object]:
    try:
        from perfetto.trace_processor import (
            TraceProcessor,
            TraceProcessorConfig,
            TraceProcessorException,
        )
    except ImportError:
        return {
            "ok": False,
            "code": "CAPABILITY_UNAVAILABLE",
            "message": "Perfetto's Python package is not installed.",
        }

    processor: Any | None = None
    try:
        processor = TraceProcessor(
            trace=str(request["artifact_path"]),
            config=TraceProcessorConfig(
                bin_path=str(request["binary_path"]),
                fetch_latest_trace_processor=False,
                load_timeout=30,
                unique_port=True,
            ),
        )
        operation = request["operation"]
        if operation == "extract":
            rows = list(processor.query(_SLICE_QUERY))
            return {
                "ok": True,
                "rows": [
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
                            "filename",
                            "line",
                        ),
                    )
                    for row in rows
                ],
            }
        if operation == "window":
            start_ns = int(str(request["start_ns"]))
            end_ns = int(str(request["end_ns"]))
            limit = int(str(request["limit"]))
            predicate = f"ts < {end_ns:d} AND ts + dur > {start_ns:d} AND dur >= 0"
            count_rows = list(
                processor.query(f"SELECT count(*) AS total FROM slice WHERE {predicate}")
            )
            after_ts = request.get("after_ts")
            after_id = request.get("after_id")
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
            return {
                "ok": True,
                "total": int(count_rows[0].total) if count_rows else 0,
                "rows": [
                    _row(
                        row,
                        ("id", "parent_id", "name", "category", "ts", "dur", "track_id"),
                    )
                    for row in rows
                ],
            }
        return {
            "ok": False,
            "code": "WORKSPACE_INVALID",
            "message": "Unknown curated Perfetto worker operation.",
        }
    except (TraceProcessorException, OSError, ValueError, RuntimeError) as exc:
        return {
            "ok": False,
            "code": "ARTIFACT_PARSE_FAILED",
            "message": f"Perfetto Trace Processor failed: {exc}",
        }
    finally:
        if processor is not None:
            processor.close()


def _write_response(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        request = json.loads(arguments.request.read_text())
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        response = _query(request)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        response = {
            "ok": False,
            "code": "WORKSPACE_INVALID",
            "message": f"Perfetto worker request is invalid: {exc}",
        }
    _write_response(arguments.response, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
