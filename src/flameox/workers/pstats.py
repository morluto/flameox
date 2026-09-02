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
        rows=cast(tuple[dict[str, JsonValue], ...], tuple(rows[: request.max_rows])),
        truncated=len(rows) > request.max_rows,
        limitations=(
            "cProfile is deterministic instrumentation; call counts are not sampling frequency.",
            "pstats files have no compatibility guarantee across Python versions or profilers.",
            "Profile observations rank work but do not establish causal optimization impact.",
        ),
    )


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
