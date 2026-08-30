from __future__ import annotations

import hashlib
import shlex
import shutil
import time
import zipfile
from pathlib import Path

from flameox.application.reduction_contracts import (
    PredicateClassification,
    ReductionAttemptReceipt,
    ReductionDisposition,
)
from flameox.atomic import atomic_write_bytes, atomic_write_json
from flameox.domain import DomainError, ErrorCode, process_exit_code
from flameox.execution import ExecutionRequest, ResourcePolicy, SubprocessBroker
from flameox.filesystem import BoundedFileSystem
from flameox.workers.protocol import (
    WorkerApplication,
    WorkerFailureKind,
    WorkerOutputFile,
    run_typed_worker,
)
from flameox.workers.reduction_contract import (
    SHRINKRAY_PROFILE,
    SHRINKRAY_VERSION,
    SHRINKRAY_WORKER,
    ShrinkRayWorkerRequest,
    ShrinkRayWorkerResult,
)

_PREDICATE_CONFIG_ENV = "FLAMEOX_REDUCTION_PREDICATE_CONFIG"


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _output(path: Path, root: Path, *, role: str, media_type: str) -> WorkerOutputFile:
    digest, size = _sha256(path)
    return WorkerOutputFile(
        role=role,
        relative_path=path.relative_to(root).as_posix(),
        media_type=media_type,
        byte_length=size,
        sha256=digest,
    )


def _read_attempts(root: Path, *, maximum: int) -> tuple[ReductionAttemptReceipt, ...]:
    attempts: list[ReductionAttemptReceipt] = []
    for path in sorted(root.glob("attempt-*.json")):
        if len(attempts) >= maximum:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Reduction emitted more predicate receipts than its attempt budget.",
            )
        payload = BoundedFileSystem((root,)).read_bytes(
            path,
            max_bytes=64 * 1024,
            require_single_link=True,
        )
        receipt = ReductionAttemptReceipt.model_validate_json(payload)
        expected = f"attempt-{len(attempts):08d}"
        if receipt.attempt_id != expected or path.name != f"{expected}.json":
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Reduction predicate receipt sequence is not contiguous.",
            )
        attempts.append(receipt)
    return tuple(attempts)


def _write_attempt_bundle(
    attempts: tuple[ReductionAttemptReceipt, ...],
    destination: Path,
) -> None:
    payload = b"".join(item.model_dump_json().encode() + b"\n" for item in attempts)
    atomic_write_bytes(destination, payload)


def _archive_history(source: Path, destination: Path, *, max_files: int) -> bool:
    if not source.is_dir():
        return False
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if len(files) > max_files:
        raise DomainError(
            ErrorCode.QUERY_BUDGET_EXCEEDED,
            "ShrinkRay history exceeded its file-count budget.",
        )
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(source)
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "ShrinkRay history contains a symbolic link.",
                )
            archive.write(path, relative.as_posix())
    return True


def _require_tree_budget(root: Path, *, max_files: int, max_bytes: int) -> None:
    files = 0
    byte_count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        files += 1
        byte_count += path.stat().st_size
        if files > max_files or byte_count > max_bytes:
            raise DomainError(
                ErrorCode.STORAGE_QUOTA_EXCEEDED,
                "ShrinkRay staging output exceeded its declared operation budget.",
                details={
                    "observed_files": files,
                    "observed_bytes": byte_count,
                    "max_files": max_files,
                    "max_bytes": max_bytes,
                },
            )


def _handle(request: ShrinkRayWorkerRequest, job_root: Path) -> ShrinkRayWorkerResult:
    root = job_root
    receipts = root / "predicate-receipts"
    receipts.mkdir(mode=0o700)
    target = root / "input.flameox-data"
    shutil.copyfile(request.artifact_path, target)
    original_sha256, original_size = _sha256(target)
    config_path = root / "predicate-config.json"
    config = request.predicate_config.validated_copy(
        update={
            "operation_root": str(root),
            "receipt_root": str(receipts),
            "counter_path": str(root / "predicate-counter"),
            "deadline_monotonic": time.monotonic() + request.wall_time_seconds,
        }
    )
    atomic_write_json(config_path, config.model_dump(mode="json"))
    stdout_path = root / "shrinkray.stdout"
    stderr_path = root / "shrinkray.stderr"
    history_root = root / "history"
    argv = (
        request.shrinkray_executable,
        "--backup",
        str(root / "original.backup"),
        "--timeout",
        str(config.predicate_timeout_seconds),
        "--memory-limit",
        str(config.maximum_rss_bytes),
        "--seed",
        "0",
        "--volume",
        "normal",
        "--in-place",
        "--input-type",
        "arg",
        "--parallelism",
        "1",
        "--ui",
        "basic",
        "--formatter",
        "none",
        "--trivial-is-not-error",
        "--history",
        "--also-interesting",
        "101",
        "--no-python-reducer",
        "--no-restart",
        "--no-llm",
        shlex.quote(request.predicate_bridge_executable),
        str(target),
    )
    exit_code: int | None = None
    limitation: str | None = None
    try:
        outcome = SubprocessBroker().run_sync(
            ExecutionRequest(
                argv=argv,
                executable_binding=request.shrinkray_executable_binding,
                cwd=root,
                environment_allowlist=(),
                environment_overrides={
                    _PREDICATE_CONFIG_ENV: str(config_path),
                    "SHRINKRAY_DIRECTORY": str(history_root),
                    "SHRINKRAY_LLM": "0",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
                allowed_working_roots=(root,),
                timeout_seconds=request.wall_time_seconds,
                max_output_bytes=config.max_output_bytes,
                resource_policy=ResourcePolicy(
                    filesystem_path=Path(config.workspace_root),
                    staging_root=Path(config.staging_root),
                    writable_roots=(root,),
                    minimum_free_bytes=config.minimum_free_bytes,
                    maximum_rss_bytes=config.maximum_rss_bytes,
                    sampling_interval_ms=config.sampling_interval_ms,
                    max_observed_files=min(config.max_observed_files, request.max_staging_files),
                    maximum_writable_growth_bytes=request.max_staging_bytes,
                ),
            )
        )
        exit_code = process_exit_code(outcome.process.termination)
        atomic_write_bytes(stdout_path, outcome.stdout)
        atomic_write_bytes(stderr_path, outcome.stderr)
    except DomainError as error:
        limitation = f"shrinkray_execution:{error.code.value}"
        atomic_write_bytes(stdout_path, b"")
        atomic_write_bytes(stderr_path, error.message.encode()[: config.max_output_bytes])

    try:
        SubprocessBroker().run_sync(
            ExecutionRequest(
                argv=(request.predicate_bridge_executable, str(target)),
                executable_binding=request.predicate_bridge_binding,
                cwd=root,
                environment_allowlist=(),
                environment_overrides={_PREDICATE_CONFIG_ENV: str(config_path)},
                allowed_working_roots=(root,),
                timeout_seconds=min(
                    config.predicate_timeout_seconds * config.predicate_repetitions + 5,
                    request.wall_time_seconds,
                ),
                max_output_bytes=config.max_output_bytes,
                resource_policy=ResourcePolicy(
                    filesystem_path=Path(config.workspace_root),
                    staging_root=Path(config.staging_root),
                    writable_roots=(root,),
                    minimum_free_bytes=config.minimum_free_bytes,
                    maximum_rss_bytes=config.maximum_rss_bytes,
                    sampling_interval_ms=config.sampling_interval_ms,
                    max_observed_files=min(config.max_observed_files, request.max_staging_files),
                    maximum_writable_growth_bytes=request.max_staging_bytes,
                ),
            )
        )
    except DomainError as error:
        limitation = limitation or f"final_bridge_revalidation:{error.code.value}"

    attempts = _read_attempts(receipts, maximum=config.max_attempts)
    attempts_path = root / "predicate-attempts.jsonl"
    _write_attempt_bundle(attempts, attempts_path)
    final_sha256, final_size = _sha256(target)
    final_classification = (
        attempts[-1].classification if attempts else PredicateClassification.UNRESOLVED
    )
    if exit_code == 0 and final_classification is PredicateClassification.INTERESTING:
        disposition = (
            ReductionDisposition.UNCHANGED
            if final_sha256 == original_sha256
            else ReductionDisposition.SUCCEEDED
        )
    elif attempts and attempts[0].classification is PredicateClassification.NOT_INTERESTING:
        disposition = ReductionDisposition.ORIGINAL_NOT_INTERESTING
    else:
        disposition = ReductionDisposition.INCONCLUSIVE

    history_path = root / "shrinkray-history.zip"
    history_output = None
    if _archive_history(
        history_root / ".shrinkray",
        history_path,
        max_files=request.max_staging_files,
    ):
        history_output = _output(
            history_path,
            root,
            role="shrinkray_history",
            media_type="application/zip",
        )
    limitations = [
        "ShrinkRay completion does not establish global or one-minimality.",
        "The offline profile disables LLM, Python, formatter, restart, and "
        "language-named input passes.",
    ]
    if any(item.classification is PredicateClassification.UNRESOLVED for item in attempts):
        limitations.append("One or more candidate predicate evaluations were unresolved.")
    if limitation is not None:
        limitations.append(limitation)
    _require_tree_budget(
        root,
        max_files=request.max_staging_files,
        max_bytes=request.max_staging_bytes,
    )
    return ShrinkRayWorkerResult(
        disposition=disposition,
        tool_completed=exit_code == 0,
        final_classification=final_classification,
        final_candidate=_output(
            target,
            root,
            role="final_candidate",
            media_type="application/octet-stream",
        ),
        attempt_receipts=_output(
            attempts_path,
            root,
            role="predicate_attempts",
            media_type="application/x-ndjson",
        ),
        attempted=len(attempts),
        passed=sum(item.classification is PredicateClassification.INTERESTING for item in attempts),
        failed=sum(
            item.classification is PredicateClassification.NOT_INTERESTING for item in attempts
        ),
        unresolved=sum(
            item.classification is PredicateClassification.UNRESOLVED for item in attempts
        ),
        contradictory=sum(
            len({observation.classification for observation in item.observations}) > 1
            for item in attempts
        ),
        timed_out=sum(
            any(
                observation.failure_category in {"process_timeout", "operation_deadline"}
                for observation in item.observations
            )
            for item in attempts
        ),
        history=history_output,
        stdout=_output(stdout_path, root, role="shrinkray_stdout", media_type="text/plain"),
        stderr=_output(stderr_path, root, role="shrinkray_stderr", media_type="text/plain"),
        original_sha256=original_sha256,
        original_size_bytes=original_size,
        final_size_bytes=final_size,
        budget_exhausted=(len(attempts) >= config.max_attempts or limitation is not None),
        shrinkray_version=SHRINKRAY_VERSION,
        profile=SHRINKRAY_PROFILE,
        limitations=tuple(limitations),
    )


def main() -> int:
    return run_typed_worker(
        WorkerApplication(
            definition=SHRINKRAY_WORKER,
            handler=_handle,
            invalid_failure=WorkerFailureKind.INPUT_MALFORMED,
            invalid_message="The ShrinkRay reduction request or output is invalid",
            caught=(OSError, ValueError),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
