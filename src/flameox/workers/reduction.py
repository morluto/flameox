from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from pydantic import ValidationError

from flameox.application.native_reducer import (
    NativeDdminReducer,
    NativePredicateClassification,
)
from flameox.application.reduction_worker import NativeReductionWorkerRequest
from flameox.domain import DomainError, ErrorCode
from flameox.execution import ExecutionRequest, ResourcePolicy, SubprocessBroker


def _write_response(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    )
    os.replace(temporary, path)


def _reduce(request: NativeReductionWorkerRequest, job_root: Path) -> dict[str, object]:
    original = request.artifact_path.read_bytes()
    command = request.predicate_command
    limits = request.limits
    predicate_timeout = request.predicate_timeout_seconds
    wall_deadline = time.monotonic() + limits.wall_time_seconds
    candidate = job_root / "candidate"
    latest_output: tuple[bytes, bytes] | None = None
    failure: str | None = None
    accepted_paths: list[str] = []
    broker = SubprocessBroker()

    def predicate(payload: bytes) -> NativePredicateClassification:
        nonlocal failure, latest_output
        candidate.write_bytes(payload)
        remaining = wall_deadline - time.monotonic()
        if remaining <= 0:
            failure = "reduction_wall_time"
            return "unresolved"
        try:
            outcome = broker.run_sync(
                ExecutionRequest(
                    argv=command.argv,
                    cwd=Path(command.cwd),
                    environment_allowlist=("PATH",),
                    environment_overrides={
                        **command.env_overrides,
                        "FLAMEOX_REDUCTION_CANDIDATE": str(candidate),
                    },
                    allowed_working_roots=(request.project_root,),
                    timeout_seconds=min(predicate_timeout, remaining),
                    max_output_bytes=request.max_output_bytes,
                    resource_policy=ResourcePolicy(
                        filesystem_path=request.workspace_root,
                        staging_root=request.staging_root,
                        writable_roots=(job_root,),
                        minimum_free_bytes=request.minimum_free_bytes,
                        maximum_rss_bytes=request.maximum_rss_bytes,
                        sampling_interval_ms=request.sampling_interval_ms,
                        max_observed_files=request.max_observed_files,
                    ),
                )
            )
        except DomainError as exc:
            failure = exc.code.value
            return "unresolved"
        failure = None
        latest_output = (outcome.stdout, outcome.stderr)
        return "interesting" if outcome.process.exit_code == 0 else "not_interesting"

    def preserve_best(payload: bytes) -> None:
        path = job_root / f"best-{len(accepted_paths):08d}"
        path.write_bytes(payload)
        accepted_paths.append(path.name)

    reducer = NativeDdminReducer(
        request.partitioning,
        limits=limits,
    )
    native = reducer.reduce(
        original,
        predicate,
        failure_detail=lambda: failure,
        on_best=preserve_best,
    )
    final_path = job_root / "final-candidate"
    final_path.write_bytes(native.final_payload if native.final_payload is not None else original)
    stdout_path = stderr_path = None
    if latest_output is not None:
        stdout_path = job_root / "predicate.stdout"
        stderr_path = job_root / "predicate.stderr"
        stdout_path.write_bytes(latest_output[0])
        stderr_path.write_bytes(latest_output[1])
    return {
        "ok": True,
        "result": native.model_dump(mode="json"),
        "accepted_paths": accepted_paths,
        "final_path": final_path.name,
        "stdout_path": stdout_path.name if stdout_path is not None else None,
        "stderr_path": stderr_path.name if stderr_path is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        request = NativeReductionWorkerRequest.model_validate_json(arguments.request.read_text())
        response = _reduce(request, arguments.request.parent)
    except DomainError as exc:
        response = {"ok": False, "code": exc.code.value, "message": exc.message}
    except (OSError, ValidationError, ValueError) as exc:
        response = {
            "ok": False,
            "code": ErrorCode.ARTIFACT_PARSE_FAILED.value,
            "message": f"Reduction worker request is invalid: {exc}",
        }
    _write_response(arguments.response, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
