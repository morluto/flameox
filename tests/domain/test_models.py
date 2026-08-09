from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from flameox.application.oracle_receipts import parse_oracle_receipt
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    CaptureStatus,
    CommandSpec,
    DomainError,
    ExecutionStatus,
    OracleReceiptValue,
    RunManifest,
    Sensitivity,
    Trial,
    TrialOutcome,
    ValidationStatus,
    effective_sensitivity,
)
from flameox.domain.models import (
    DigestOracleReceiptValue,
    ExecutionRunManifest,
    ImportRunManifest,
    ScalarOracleReceiptValue,
    SucceededTrial,
)
from flameox.domain.scalars import FloatingValue, IntegerValue, NumericValue, parse_numeric_value

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
        ImportRunManifest.model_validate(
            {
                "run_id": "run",
                "execution_status": ExecutionStatus.SUCCEEDED,
                "capture_status": CaptureStatus.REGISTERED,
                "validation_status": ValidationStatus.NOT_REQUESTED,
                "environment_id": DIGEST,
            }
        )


def test_run_manifest_parser_routes_schema_one_json_to_legal_variants() -> None:
    adapter: TypeAdapter[RunManifest] = TypeAdapter(RunManifest)
    common = {
        "schema_version": 1,
        "revision": 0,
        "run_id": "run",
        "created_at": "2026-01-01T00:00:00Z",
        "capture_status": "pending",
        "validation_status": "not_requested",
        "environment_id": DIGEST,
    }

    imported = adapter.validate_json(
        json.dumps(
            {
                **common,
                "run_type": "import",
                "execution_status": "not_applicable",
            }
        )
    )
    execution = adapter.validate_json(
        json.dumps(
            {
                **common,
                "run_type": "execution",
                "execution_status": "planned",
            }
        )
    )

    assert isinstance(imported, ImportRunManifest)
    assert isinstance(execution, ExecutionRunManifest)
    assert isinstance(imported.model_copy(update={"revision": 1}), ImportRunManifest)

    with pytest.raises(ValidationError, match="execution_status"):
        adapter.validate_python(
            {
                **common,
                "run_type": "execution",
                "execution_status": "not_applicable",
            }
        )


def test_import_run_rejects_execution_only_identity() -> None:
    with pytest.raises(ValidationError, match="workload_definition_id"):
        ImportRunManifest.model_validate(
            {
                "run_id": "run",
                "capture_status": CaptureStatus.PENDING,
                "validation_status": ValidationStatus.NOT_REQUESTED,
                "environment_id": DIGEST,
                "workload_definition_id": DIGEST,
            }
        )


@pytest.mark.parametrize("argv", [(), ("python", "bad\x00arg"), ("",)])
def test_command_rejects_empty_or_unsafe_arguments(argv: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        CommandSpec(argv=argv, cwd=".")


def test_contracts_require_timezone_aware_datetimes() -> None:
    with pytest.raises(ValidationError):
        ImportRunManifest(
            run_id="run",
            created_at=datetime(2026, 1, 1),
            execution_status=ExecutionStatus.NOT_APPLICABLE,
            capture_status=CaptureStatus.REGISTERED,
            validation_status=ValidationStatus.NOT_REQUESTED,
            environment_id=DIGEST,
        )

    manifest = ImportRunManifest(
        run_id="run",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        execution_status=ExecutionStatus.NOT_APPLICABLE,
        capture_status=CaptureStatus.REGISTERED,
        validation_status=ValidationStatus.NOT_REQUESTED,
        environment_id=DIGEST,
    )
    assert manifest.created_at.tzinfo is UTC


def test_numeric_value_variants_preserve_their_kind_through_json() -> None:
    adapter: TypeAdapter[NumericValue] = TypeAdapter(NumericValue)

    integer = adapter.validate_json('{"kind":"integer","value":42}')
    floating = adapter.validate_json('{"kind":"floating","value":42.0}')

    assert integer == IntegerValue(value=42)
    assert floating == FloatingValue(value=42.0)


@pytest.mark.parametrize(
    ("value_type", "value"),
    [
        (IntegerValue, True),
        (IntegerValue, 2**63),
        (FloatingValue, True),
        (FloatingValue, float("inf")),
    ],
)
def test_numeric_value_variants_reject_unrepresentable_values(
    value_type: type[IntegerValue] | type[FloatingValue],
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        value_type.model_validate({"value": value})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42", IntegerValue(value=42)),
        ("42.5", FloatingValue(value=42.5)),
        (True, None),
        (str(2**63), None),
        (float("nan"), None),
    ],
)
def test_numeric_factor_parser_returns_only_representable_values(
    raw: object,
    expected: IntegerValue | FloatingValue | None,
) -> None:
    assert parse_numeric_value(raw) == expected


def test_trial_exposes_one_tagged_parameter_value() -> None:
    trial = SucceededTrial(
        trial_id="trial",
        experiment_id="experiment",
        variant_id="variant",
        combination_id=DIGEST,
        parameter_name="workers",
        parameter_value=IntegerValue(value=4),
        outcome=TrialOutcome.SUCCEEDED,
        validation_status=ValidationStatus.PASSED,
    )

    adapter: TypeAdapter[Trial] = TypeAdapter(Trial)
    assert adapter.validate_json(trial.model_dump_json()) == trial
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                **trial.model_dump(mode="python"),
                "parameter_value_int": 4,
                "parameter_value_float": 4.0,
            }
        )


@pytest.mark.parametrize(
    ("outcome", "failure_class", "exclusion_reason"),
    [
        ("succeeded", "none", "cannot exclude success"),
        ("failed", "none", "failed"),
        ("timed_out", "process_failure", "timed out"),
        ("oom", "infrastructure_failure", "out of memory"),
    ],
)
def test_trial_parser_rejects_outcome_failure_combinations_that_cannot_occur(
    outcome: str,
    failure_class: str,
    exclusion_reason: str,
) -> None:
    adapter: TypeAdapter[Trial] = TypeAdapter(Trial)

    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "trial_id": "trial",
                "experiment_id": "experiment",
                "variant_id": "variant",
                "combination_id": DIGEST,
                "outcome": outcome,
                "failure_class": failure_class,
                "exclusion_reason": exclusion_reason,
                "validation_status": "not_requested",
            }
        )


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
    assert isinstance(receipt.expected, ScalarOracleReceiptValue)
    assert isinstance(receipt.observed, DigestOracleReceiptValue)


@pytest.mark.parametrize(
    "value",
    [
        {"kind": "digest", "value": 42},
        {"kind": "digest", "value": "not-a-digest"},
        {"kind": "scalar", "value": "x" * 501},
        {"kind": "scalar", "value": 10**18 + 1},
        {"kind": "scalar", "value": float("inf")},
    ],
)
def test_oracle_receipt_value_rejects_kind_value_mismatches(
    value: dict[str, object],
) -> None:
    adapter: TypeAdapter[OracleReceiptValue] = TypeAdapter(OracleReceiptValue)

    with pytest.raises(ValidationError):
        adapter.validate_python(value)


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
