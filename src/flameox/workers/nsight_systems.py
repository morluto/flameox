"""Typed rejection for the removed Nsight Systems SQLite export reader."""

from __future__ import annotations

from pathlib import Path

from flameox.runtime_errors import DomainError, ErrorCode
from flameox.workers.nsight_systems_contract import (
    NSIGHT_SYSTEMS_WORKER,
    NsightSystemsWorkerRequest,
    NsightSystemsWorkerResult,
)
from flameox.workers.protocol import WorkerApplication, WorkerFailureKind, run_typed_worker


def _handle(
    _request: NsightSystemsWorkerRequest,
    _job_root: Path,
) -> NsightSystemsWorkerResult:
    raise DomainError(
        ErrorCode.UNSUPPORTED_FORMAT,
        "Nsight Systems SQLite exports are unsupported; export parquetdir evidence instead.",
    )


def main() -> int:
    return run_typed_worker(
        WorkerApplication(
            definition=NSIGHT_SYSTEMS_WORKER,
            handler=_handle,
            invalid_failure=WorkerFailureKind.INPUT_FORMAT_UNSUPPORTED,
            invalid_message="Nsight Systems requires parquetdir evidence",
            caught=(OSError, ValueError, KeyError, TypeError),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
