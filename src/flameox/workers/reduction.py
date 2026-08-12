from __future__ import annotations

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
from flameox.workers.protocol import run_worker


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
            return NativePredicateClassification.UNRESOLVED
        try:
            outcome = broker.run_sync(
                ExecutionRequest(
                    argv=command.argv,
                    executable_binding=request.predicate_executable_binding,
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
            return NativePredicateClassification.UNRESOLVED
        failure = None
        latest_output = (outcome.stdout, outcome.stderr)
        return (
            NativePredicateClassification.INTERESTING
            if outcome.process.exit_code == 0
            else NativePredicateClassification.NOT_INTERESTING
        )

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
    return run_worker(
        lambda request, request_path: _reduce(
            NativeReductionWorkerRequest.model_validate(request), request_path.parent
        ),
        invalid_code=ErrorCode.ARTIFACT_PARSE_FAILED,
        invalid_message="Reduction worker request is invalid",
        caught=(OSError, ValidationError, ValueError),
    )


if __name__ == "__main__":
    raise SystemExit(main())
