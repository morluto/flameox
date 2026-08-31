from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from flameox.workers.otlp_contract import OTLP_WORKER, OtlpWorkerRequest, OtlpWorkerResult
from flameox.workers.otlp_parser import OtlpRowLimitExceeded, ParsedOtlp, parse_otlp
from flameox.workers.protocol import (
    WorkerApplication,
    WorkerFailureKind,
    run_typed_worker,
)


def _handle(request: OtlpWorkerRequest, _job_root: Path) -> OtlpWorkerResult:
    try:
        parsed = parse_otlp(
            Path(request.artifact_path),
            request.media_type,
            row_limit=request.row_limit,
            start_ns=request.start_ns,
            end_ns=request.end_ns,
        )
        return _result(parsed)
    except OtlpRowLimitExceeded as exc:
        return _result(exc.parsed, row_limit_exceeded=True, counts=exc.counts)


def _result(
    parsed: ParsedOtlp,
    *,
    row_limit_exceeded: bool = False,
    counts: dict[str, int] | None = None,
) -> OtlpWorkerResult:
    return OtlpWorkerResult(
        row_limit_exceeded=row_limit_exceeded,
        resources=cast(tuple[dict[str, JsonValue], ...], tuple(parsed.resources)),
        scopes=cast(tuple[dict[str, JsonValue], ...], tuple(parsed.scopes)),
        spans=cast(tuple[dict[str, JsonValue], ...], tuple(parsed.spans)),
        events=cast(tuple[dict[str, JsonValue], ...], tuple(parsed.events)),
        links=cast(tuple[dict[str, JsonValue], ...], tuple(parsed.links)),
        counts=cast(Any, counts or {}),
        limitations=parsed.limitations,
    )


def main() -> int:
    return run_typed_worker(
        WorkerApplication(
            definition=OTLP_WORKER,
            handler=_handle,
            invalid_failure=WorkerFailureKind.INPUT_MALFORMED,
            invalid_message="OTLP artifact is invalid",
            caught=(OSError, TypeError, ValueError),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
