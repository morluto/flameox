from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic import JsonValue

from flameox.application.otlp import _OtlpRowLimitExceeded, _parse_otlp
from flameox.workers.otlp_contract import OTLP_WORKER, OtlpWorkerRequest, OtlpWorkerResult
from flameox.workers.protocol import (
    WorkerApplication,
    WorkerFailureKind,
    run_typed_worker,
)


def _handle(request: OtlpWorkerRequest, _job_root: Path) -> OtlpWorkerResult:
    try:
        parsed = _parse_otlp(
            Path(request.artifact_path),
            request.media_type,
            row_limit=request.row_limit,
        )
        return OtlpWorkerResult(
            resources=cast(tuple[dict[str, JsonValue], ...], tuple(parsed.resources)),
            scopes=cast(tuple[dict[str, JsonValue], ...], tuple(parsed.scopes)),
            spans=cast(tuple[dict[str, JsonValue], ...], tuple(parsed.spans)),
            events=cast(tuple[dict[str, JsonValue], ...], tuple(parsed.events)),
            links=cast(tuple[dict[str, JsonValue], ...], tuple(parsed.links)),
            limitations=parsed.limitations,
        )
    except _OtlpRowLimitExceeded as exc:
        return OtlpWorkerResult(
            row_limit_exceeded=True,
            counts=exc.counts,  # type: ignore[arg-type]
            limitations=exc.limitations,
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
