from __future__ import annotations

import pstats
import sys
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from flameox.workers.protocol import WorkerApplication, WorkerFailureKind, run_typed_worker
from flameox.workers.pstats_contract import PSTATS_WORKER, PstatsWorkerRequest, PstatsWorkerResult


def _handle(request: PstatsWorkerRequest, _job_root: Path) -> PstatsWorkerResult:
    stats = cast(Any, pstats.Stats(request.artifact_path)).stats
    if request.projection == "call_graph":
        return _call_graph(request, stats)
    rows: list[dict[str, Any]] = []
    for (filename, line, function), values in stats.items():
        primitive_calls, total_calls, self_time, cumulative_time, _callers = values
        rows.append(
            {
                "function": function,
                "file": filename,
                "line": line,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_time_seconds": self_time,
                "cumulative_time_seconds": cumulative_time,
            }
        )
    rows.sort(key=lambda row: (-float(row[request.metric]), str(row["file"]), int(row["line"])))
    return PstatsWorkerResult(
        reader_version=".".join(map(str, sys.version_info[:3])),
        metric=request.metric,
        function_count=len(rows),
        edge_count=0,
        rows=cast(tuple[dict[str, JsonValue], ...], tuple(rows[: request.max_rows])),
        truncated=len(rows) > request.max_rows,
        limitations=(
            "cProfile is deterministic instrumentation; call counts are not sampling frequency.",
            "pstats files have no compatibility guarantee across Python versions or profilers.",
            "Profile observations rank work but do not establish causal optimization impact.",
        ),
    )


def _call_graph(request: PstatsWorkerRequest, stats: dict[Any, Any]) -> PstatsWorkerResult:
    rows: list[dict[str, Any]] = []
    for callee_key, values in stats.items():
        callee = _identity(callee_key)
        callers = values[4]
        if not isinstance(callers, dict):
            continue
        for caller_key, raw_edge in callers.items():
            caller = _identity(caller_key)
            if not _matches_edge(request, caller, callee):
                continue
            primitive_calls, total_calls, self_time, cumulative_time = _edge_values(raw_edge)
            rows.append(
                {
                    "caller_function": caller["function"],
                    "caller_file": caller["file"],
                    "caller_line": caller["line"],
                    "callee_function": callee["function"],
                    "callee_file": callee["file"],
                    "callee_line": callee["line"],
                    "primitive_calls": primitive_calls,
                    "total_calls": total_calls,
                    "self_time_seconds": self_time,
                    "cumulative_time_seconds": cumulative_time,
                }
            )
    rows.sort(
        key=lambda row: (
            -float(row["cumulative_time_seconds"] or 0),
            str(row["caller_file"]),
            int(row["caller_line"]),
            str(row["callee_file"]),
            int(row["callee_line"]),
        )
    )
    return PstatsWorkerResult(
        reader_version=".".join(map(str, sys.version_info[:3])),
        metric=request.metric,
        function_count=len(stats),
        edge_count=len(rows),
        rows=cast(tuple[dict[str, JsonValue], ...], tuple(rows[: request.max_rows])),
        truncated=len(rows) > request.max_rows,
        limitations=(
            "pstats caller edges are deterministic instrumentation, not sampled stack frequency.",
            "pstats files have no compatibility guarantee across Python versions or profilers.",
            "Caller relationships show observed attribution and do not prove causal dependence.",
        ),
    )


def _identity(key: Any) -> dict[str, Any]:
    filename, line, function = key
    return {"function": str(function), "file": str(filename), "line": int(line)}


def _matches_edge(
    request: PstatsWorkerRequest,
    caller: dict[str, Any],
    callee: dict[str, Any],
) -> bool:
    if request.function is None:
        return True
    needle = request.function.casefold()
    caller_matches = needle in f"{caller['file']}:{caller['line']}:{caller['function']}".casefold()
    callee_matches = needle in f"{callee['file']}:{callee['line']}:{callee['function']}".casefold()
    if request.direction == "callers":
        return callee_matches
    if request.direction == "callees":
        return caller_matches
    return caller_matches or callee_matches


def _edge_values(value: Any) -> tuple[int, int, float | None, float | None]:
    if isinstance(value, tuple) and len(value) == 4:
        primitive_calls, total_calls, self_time, cumulative_time = value
        return int(primitive_calls), int(total_calls), float(self_time), float(cumulative_time)
    calls = int(value)
    return calls, calls, None, None


def main() -> int:
    return run_typed_worker(
        WorkerApplication(
            definition=PSTATS_WORKER,
            handler=_handle,
            invalid_failure=WorkerFailureKind.INPUT_MALFORMED,
            invalid_message="pstats profile is unsupported or invalid",
            caught=(OSError, EOFError, ValueError, TypeError),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
