from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    CaptureStatus,
    CommandSpec,
    ExecutionStatus,
    RunManifest,
    RunType,
    Sensitivity,
    ValidationStatus,
    effective_sensitivity,
)

DIGEST = "sha256:" + ("a" * 64)


def test_sensitivity_is_monotonic_across_registrations() -> None:
    assert (
        effective_sensitivity([Sensitivity.NORMAL, Sensitivity.SENSITIVE, Sensitivity.INTERNAL])
        is Sensitivity.SENSITIVE
    )


def test_core_dump_has_mandatory_sensitivity_floor() -> None:
    with pytest.raises(ValidationError):
        ArtifactRegistration(
            registration_id="registration",
            run_id="run",
            artifact_id=DIGEST,
            display_name="core",
            media_type="application/x-core",
            kind=ArtifactKind.CORE_DUMP,
            role="primary",
            sensitivity=Sensitivity.INTERNAL,
        )


def test_import_run_cannot_claim_execution() -> None:
    with pytest.raises(ValidationError):
        RunManifest(
            run_id="run",
            run_type=RunType.IMPORT,
            execution_status=ExecutionStatus.SUCCEEDED,
            capture_status=CaptureStatus.REGISTERED,
            validation_status=ValidationStatus.NOT_REQUESTED,
            environment_id=DIGEST,
        )


def test_command_rejects_empty_argv_and_nul() -> None:
    with pytest.raises(ValidationError):
        CommandSpec(argv=(), cwd=".")
    with pytest.raises(ValidationError):
        CommandSpec(argv=("python", "bad\x00arg"), cwd=".")


def test_contracts_require_timezone_aware_datetimes() -> None:
    with pytest.raises(ValidationError):
        RunManifest(
            run_id="run",
            run_type=RunType.IMPORT,
            created_at=datetime(2026, 1, 1),
            execution_status=ExecutionStatus.NOT_APPLICABLE,
            capture_status=CaptureStatus.REGISTERED,
            validation_status=ValidationStatus.NOT_REQUESTED,
            environment_id=DIGEST,
        )

    manifest = RunManifest(
        run_id="run",
        run_type=RunType.IMPORT,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        execution_status=ExecutionStatus.NOT_APPLICABLE,
        capture_status=CaptureStatus.REGISTERED,
        validation_status=ValidationStatus.NOT_REQUESTED,
        environment_id=DIGEST,
    )
    assert manifest.created_at.tzinfo is UTC
