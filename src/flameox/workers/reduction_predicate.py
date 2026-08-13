from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import portalocker
from pydantic import ValidationError

from flameox.application.reduction_contracts import (
    PredicateClassification,
    PredicateObservation,
    ReductionAttemptReceipt,
    collapse_predicate_observations,
)
from flameox.atomic import atomic_write_json
from flameox.domain import DomainError, ErrorCode
from flameox.domain.models import utc_now
from flameox.execution import ExecutionRequest, ResourcePolicy, SubprocessBroker
from flameox.filesystem import BoundedFileSystem
from flameox.workers.reduction_contract import (
    UNRESOLVED_EXIT_CODE,
    ReductionPredicateConfig,
)

_CONFIG_ENV = "FLAMEOX_REDUCTION_PREDICATE_CONFIG"
_CONFIG_LIMIT = 1024 * 1024


def _next_attempt(config: ReductionPredicateConfig) -> int | None:
    counter_path = Path(config.counter_path)
    with portalocker.Lock(counter_path, mode="a+", timeout=30) as stream:
        stream.seek(0)
        raw = stream.read().strip()
        current = int(raw) if raw else 0
        if current >= config.max_attempts:
            return None
        stream.seek(0)
        stream.truncate()
        stream.write(str(current + 1))
        stream.flush()
        os.fsync(stream.fileno())
        return current


def _hash_candidate(path: Path, config: ReductionPredicateConfig) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with BoundedFileSystem((Path(config.operation_root),)).open_regular(
        path,
        max_bytes=config.max_candidate_bytes,
        require_single_link=True,
    ) as descriptor:
        metadata = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    if size != metadata.st_size:
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            "Reduction candidate changed while it was read.",
        )
    return "sha256:" + digest.hexdigest(), size


def _observe(
    candidate: Path,
    config: ReductionPredicateConfig,
    repetition: int,
) -> PredicateObservation:
    started = time.monotonic()
    remaining = config.deadline_monotonic - started
    if remaining <= 0:
        return PredicateObservation(
            repetition=repetition,
            classification=PredicateClassification.UNRESOLVED,
            failure_category="operation_deadline",
            duration_ms=0,
        )
    try:
        outcome = SubprocessBroker().run_sync(
            ExecutionRequest(
                argv=config.predicate_command.argv,
                executable_binding=config.predicate_executable_binding,
                cwd=Path(config.predicate_command.cwd),
                environment_allowlist=("PATH",),
                environment_overrides={
                    **config.predicate_command.env_overrides,
                    "FLAMEOX_REDUCTION_CANDIDATE": str(candidate),
                },
                allowed_working_roots=(Path(config.project_root),),
                timeout_seconds=min(config.predicate_timeout_seconds, remaining),
                max_output_bytes=config.max_output_bytes,
                resource_policy=ResourcePolicy(
                    filesystem_path=Path(config.workspace_root),
                    staging_root=Path(config.staging_root),
                    writable_roots=(Path(config.operation_root),),
                    minimum_free_bytes=config.minimum_free_bytes,
                    maximum_rss_bytes=config.maximum_rss_bytes,
                    sampling_interval_ms=config.sampling_interval_ms,
                    max_observed_files=config.max_observed_files,
                    maximum_writable_growth_bytes=config.max_candidate_bytes,
                ),
            )
        )
    except DomainError as error:
        return PredicateObservation(
            repetition=repetition,
            classification=PredicateClassification.UNRESOLVED,
            failure_category=error.code.value,
            duration_ms=(time.monotonic() - started) * 1_000,
        )
    return PredicateObservation(
        repetition=repetition,
        classification=(
            PredicateClassification.INTERESTING
            if outcome.process.exit_code == 0
            else PredicateClassification.NOT_INTERESTING
        ),
        exit_code=outcome.process.exit_code,
        duration_ms=(time.monotonic() - started) * 1_000,
    )


def _load_config() -> ReductionPredicateConfig:
    raw_path = os.environ.get(_CONFIG_ENV)
    if raw_path is None:
        raise ValueError("predicate configuration is missing")
    path = Path(raw_path)
    payload = BoundedFileSystem((path.parent,)).read_bytes(
        path,
        max_bytes=_CONFIG_LIMIT,
        require_single_link=True,
    )
    return ReductionPredicateConfig.model_validate_json(payload)


def main() -> int:
    try:
        config = _load_config()
        if len(sys.argv) != 2:
            raise ValueError("exactly one candidate path is required")
        candidate = Path(sys.argv[1])
        attempt_index = _next_attempt(config)
        if attempt_index is None:
            return UNRESOLVED_EXIT_CODE
        candidate_sha256, candidate_size = _hash_candidate(candidate, config)
        observations = tuple(
            _observe(candidate, config, repetition)
            for repetition in range(config.predicate_repetitions)
        )
        classification = collapse_predicate_observations(
            tuple(item.classification for item in observations)
        )
        receipt = ReductionAttemptReceipt(
            attempt_id=f"attempt-{attempt_index:08d}",
            candidate_sha256=candidate_sha256,
            candidate_size_bytes=candidate_size,
            observations=observations,
            classification=classification,
            recorded_at=utc_now().isoformat(),
        )
        atomic_write_json(
            Path(config.receipt_root) / f"{receipt.attempt_id}.json",
            receipt.model_dump(mode="json"),
        )
        if classification is PredicateClassification.INTERESTING:
            return 0
        if classification is PredicateClassification.NOT_INTERESTING:
            return 1
        return UNRESOLVED_EXIT_CODE
    except (DomainError, OSError, ValueError, ValidationError, json.JSONDecodeError):
        return UNRESOLVED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
