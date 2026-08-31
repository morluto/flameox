from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from coverage import CoverageData
from coverage.exceptions import DataError
from pydantic import JsonValue

from flameox.workers.coverage_contract import (
    COVERAGE_WORKER,
    CoverageWorkerRequest,
    CoverageWorkerResult,
)
from flameox.workers.protocol import WorkerApplication, WorkerFailureKind, run_typed_worker


def _relative_path(filename: str, project_root: Path) -> str | None:
    try:
        return Path(filename).resolve().relative_to(project_root).as_posix()
    except (OSError, ValueError):
        return None


def _handle(request: CoverageWorkerRequest, _job_root: Path) -> CoverageWorkerResult:
    project_root = Path(request.project_root).resolve(strict=True)
    data = CoverageData(basename=request.artifact_path)
    data.read()
    measured_files = sorted(data.measured_files())
    rows: list[dict[str, Any]] = []
    limitations: list[str] = []
    line_count = 0
    arc_count = 0
    for filename in measured_files:
        relative = _relative_path(filename, project_root)
        if relative is None:
            limitations.append(f"Skipped coverage outside the project root: {filename}")
            continue
        contexts = data.contexts_by_lineno(filename)
        for line in sorted(data.lines(filename) or ()):
            line_contexts = sorted(contexts.get(line) or ("",))
            for context in line_contexts:
                line_count += 1
                if len(rows) < request.max_rows:
                    rows.append(
                        {
                            "kind": "line_hit",
                            "file": relative,
                            "line_from": line,
                            "line_to": None,
                            "context": context or None,
                        }
                    )
        if data.has_arcs():
            for line_from, line_to in sorted(data.arcs(filename) or ()):
                arc_count += 1
                if len(rows) < request.max_rows:
                    rows.append(
                        {
                            "kind": "branch_arc",
                            "file": relative,
                            "line_from": line_from,
                            "line_to": line_to,
                            "context": None,
                        }
                    )
    observed = line_count + arc_count
    if observed > len(rows):
        limitations.append(f"Coverage rows were truncated to {request.max_rows} entries.")
    return CoverageWorkerResult(
        reader_version=version("coverage"),
        file_count=len(measured_files),
        line_count=line_count,
        arc_count=arc_count,
        rows=cast(tuple[dict[str, JsonValue], ...], tuple(rows)),
        truncated=observed > len(rows),
        limitations=tuple(dict.fromkeys(limitations)),
    )


def main() -> int:
    return run_typed_worker(
        WorkerApplication(
            definition=COVERAGE_WORKER,
            handler=_handle,
            invalid_failure=WorkerFailureKind.INPUT_MALFORMED,
            invalid_message="coverage.py data is unsupported or invalid",
            caught=(DataError, OSError, ValueError),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
