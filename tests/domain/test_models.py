from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from flameox.application.oracle_receipts import parse_oracle_receipt
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    CaptureStatus,
    CommandSpec,
    DomainError,
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


def test_inference_request_trace_has_mandatory_sensitivity_floor() -> None:
    with pytest.raises(ValidationError):
        ArtifactRegistration(
            registration_id="registration",
            run_id="run",
            artifact_id=DIGEST,
            display_name="requests.jsonl",
            media_type="application/x-ndjson",
            kind=ArtifactKind.INFERENCE_REQUEST_TRACE,
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


def test_oracle_receipt_parser_is_strict_and_preserves_typed_mismatch() -> None:
    payload = b"""{
      "schema_version":"flameox.oracle-receipt.v1",
      "status":"fail",
      "reason":"contract_mismatch",
      "output_field":"backward",
      "coordinate":[2,"x"],
      "expected":{"kind":"scalar","value":1.0},
      "observed":{"kind":"digest","value":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
      "absolute_error":0.25,
      "tolerance":{"absolute":0.01,"relative":0.001},
      "diagnostic_roles":["primary"]
    }"""

    receipt = parse_oracle_receipt(payload)

    assert receipt.status == "fail"
    assert receipt.output_field == "backward"
    assert receipt.coordinate == (2, "x")
    assert receipt.observed is not None and receipt.observed.kind == "digest"


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":"flameox.oracle-receipt.v1","status":"pass","reason":"ok","reason":"again"}',
        b'{"schema_version":"flameox.oracle-receipt.v1","status":"pass","reason":"ok"} {}',
        b'{"schema_version":"flameox.oracle-receipt.v1","status":"pass","reason":"ok","extra":1}',
        b'{"schema_version":"flameox.oracle-receipt.v2","status":"pass","reason":"ok"}',
        b'{"schema_version":"flameox.oracle-receipt.v1","status":"pass","reason":"ok","absolute_error":NaN}',
    ],
)
def test_oracle_receipt_parser_rejects_ambiguous_or_unsupported_json(payload: bytes) -> None:
    with pytest.raises(DomainError):
        parse_oracle_receipt(payload)
