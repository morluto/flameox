from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path

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
    ExitedProcessTermination,
    OracleReceiptValue,
    ProcessCancellationCause,
    ProcessResult,
    ProcessTerminationKind,
    RunManifest,
    RunSemantics,
    RuntimeResourceSummary,
    Sensitivity,
    Trial,
    TrialOutcome,
    ValidationStatus,
    Variant,
    VariantIdentityQuality,
    digest_model,
    effective_sensitivity,
)
from flameox.domain.executables import ResolvedExecutable
from flameox.domain.models import (
    ActiveCapturePlan,
    DigestOracleReceiptValue,
    ExecutionRunManifest,
    ImportRunManifest,
    ScalarOracleReceiptValue,
    SucceededTrial,
    parse_capture_plan,
)
from flameox.domain.scalars import FloatingValue, IntegerValue, NumericValue, parse_numeric_value

pytestmark = pytest.mark.unit

DIGEST = "sha256:" + ("a" * 64)
SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "flameox"


def _capture_plan_payload(**overrides: object) -> dict[str, object]:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    command = CommandSpec(argv=("python",), cwd=".")
    binding = {
        "requested_token": "python",
        "invocation_path": "/usr/bin/python",
        "canonical_target": "/usr/bin/python",
        "origin": "path_search",
        "matched_path_entry": "/usr/bin",
        "identity": {
            "sha256": "a" * 64,
            "size": 1,
            "mode": 0o755,
            "device": 1,
            "inode": 1,
            "modified_ns": 1,
        },
        "policy_decision": {
            "policy": "trusted_host_tool",
            "allowed": True,
        },
    }
    instance_content = {
        "workload_definition_id": DIGEST,
        "command": command.model_dump(mode="json"),
        "executable_binding": ResolvedExecutable.model_validate(binding).model_dump(mode="json"),
        "parameters": {},
    }
    execution_identity_content = {
        "quality": "not_applicable",
        "inputs": [],
        "missing_inputs": [],
    }
    payload: dict[str, object] = {
        "plan_token": "token",
        "plan_id": DIGEST,
        "run_id": "run",
        "workspace_id": "workspace",
        "workload_name": "workload",
        "workload_definition_id": DIGEST,
        "workload_instance": {
            "workload_instance_id": digest_model(instance_content),
            **instance_content,
        },
        "semantics": {"origin": "capture", "adapter": "adapter"},
        "execution_policy": "trusted_local",
        "collector_argv": ["python"],
        "collector_executable_binding": binding,
        "expected_artifact_kinds": [],
        "expected_overhead": "low",
        "containment": "active",
        "network_contained": True,
        "systemd_scope_unit": "flameox-plan.scope",
        "preflight": {
            "preflight_id": DIGEST,
            "mode": "passive",
            "disposition": "ready",
            "requirements": [],
        },
        "planned_execution_identity": {
            "identity_id": digest_model(execution_identity_content),
            **execution_identity_content,
        },
        "execution_limits": {
            "child_environment_allowlist": ["PATH"],
            "max_output_bytes": 1024,
            "max_artifact_bytes": 4096,
            "minimum_free_bytes": 0,
            "resource_sampling_interval_ms": 250,
            "max_resource_observed_files": 100,
            "max_rows_per_generation": 1000,
        },
        "created_at": created_at,
        "expires_at": datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
    }
    payload.update(overrides)
    return payload


def test_sensitivity_is_monotonic_across_registrations() -> None:
    assert (
        effective_sensitivity([Sensitivity.NORMAL, Sensitivity.SENSITIVE, Sensitivity.INTERNAL])
        is Sensitivity.SENSITIVE
    )


@pytest.mark.parametrize("name", ["", "x" * 200])
def test_variant_name_allows_empty_labels_with_a_bounded_length(name: str) -> None:
    assert (
        Variant(
            variant_id="variant",
            experiment_id="experiment",
            name=name,
            identity_quality=VariantIdentityQuality.INCOMPLETE,
        ).name
        == name
    )

    with pytest.raises(ValidationError, match="at most 200 characters"):
        Variant(
            variant_id="variant",
            experiment_id="experiment",
            name="x" * 201,
            identity_quality=VariantIdentityQuality.INCOMPLETE,
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


def test_workload_instance_rejects_tampered_bound_content() -> None:
    command = CommandSpec(argv=("python",), cwd=".")
    content = {
        "workload_definition_id": DIGEST,
        "command": command.model_dump(mode="json"),
        "parameters": {"workers": 1},
    }
    payload = {
        "workload_instance_id": digest_model(content),
        **content,
    }
    payload["parameters"] = {"workers": 2}

    with pytest.raises(ValidationError):
        parse_capture_plan(
            _capture_plan_payload(
                workload_instance=payload,
            )
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


def test_run_manifest_parser_routes_json_to_legal_variants() -> None:
    adapter: TypeAdapter[RunManifest] = TypeAdapter(RunManifest)
    common = {
        "revision": 0,
        "run_id": "run",
        "created_at": "2026-01-01T00:00:00Z",
        "capture_status": "pending",
        "validation_status": "not_requested",
        "environment_id": DIGEST,
        "semantics": {"origin": "internal", "unavailable_fields": ["scope"]},
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


def test_validated_copy_reparses_updates_into_a_valid_contract() -> None:
    imported = ImportRunManifest(
        run_id="run",
        capture_status=CaptureStatus.PENDING,
        validation_status=ValidationStatus.NOT_REQUESTED,
        environment_id=DIGEST,
        semantics=RunSemantics.unavailable(origin="import", adapter="import"),
    )

    updated = imported.validated_copy(
        update={"revision": 1, "capture_status": CaptureStatus.REGISTERED.value}
    )

    assert updated.revision == 1
    assert updated.capture_status is CaptureStatus.REGISTERED


def test_validated_copy_rejects_invalid_timestamp_and_lifecycle_updates() -> None:
    planned = ExecutionRunManifest(
        run_id="run",
        execution_status=ExecutionStatus.PLANNED,
        capture_status=CaptureStatus.PENDING,
        validation_status=ValidationStatus.NOT_REQUESTED,
        environment_id=DIGEST,
        semantics=RunSemantics.unavailable(origin="internal", adapter=None),
    )

    with pytest.raises(ValidationError, match="timezone"):
        planned.validated_copy(update={"created_at": datetime(2026, 1, 1)})

    with pytest.raises(ValidationError, match="running execution"):
        planned.validated_copy(update={"execution_status": ExecutionStatus.RUNNING})


def test_production_contract_updates_cannot_bypass_validation() -> None:
    violations: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "model_copy"
                and any(keyword.arg == "update" for keyword in node.keywords)
            ):
                violations.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}")

    assert violations == [], (
        "production contract updates must use validated_copy() or a named validated transition: "
        + ", ".join(violations)
    )


@pytest.mark.parametrize(
    "updates",
    (
        {"execution_status": "planned", "capture_status": "running"},
        {
            "execution_status": "running",
            "capture_status": "running",
            "started_at": None,
        },
        {
            "execution_status": "failed",
            "capture_status": "failed",
            "finished_at": None,
        },
    ),
)
def test_execution_run_rejects_contradictory_lifecycle(updates: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "run_id": "run",
        "environment_id": DIGEST,
        "validation_status": "not_requested",
        "created_at": "2026-01-01T00:00:00Z",
        "started_at": "2026-01-01T00:00:01Z",
        "finished_at": "2026-01-01T00:00:02Z",
        "execution_status": "failed",
        "capture_status": "failed",
    }
    payload.update(updates)

    with pytest.raises(ValidationError):
        ExecutionRunManifest.model_validate(payload)


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


def test_capture_plan_parses_containment_into_a_legal_variant() -> None:
    plan = parse_capture_plan(_capture_plan_payload())

    assert isinstance(plan, ActiveCapturePlan)
    assert parse_capture_plan(plan.model_dump(mode="json")) == plan


@pytest.mark.parametrize(
    "overrides",
    (
        {"containment": "active", "systemd_scope_unit": None},
        {"containment": "degraded", "systemd_scope_unit": "unexpected.scope"},
        {"containment": "uncontained", "network_contained": True},
        {"containment": "unavailable", "network_contained": True},
    ),
)
def test_capture_plan_rejects_contradictory_containment(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        parse_capture_plan(_capture_plan_payload(**overrides))


@pytest.mark.parametrize(
    "payload",
    (
        {"timed_out": True, "cancellation_cause": "caller_cancelled"},
        {"timed_out": False, "cancellation_cause": "timeout"},
        {
            "peak_rss_bytes": 10,
            "resources": RuntimeResourceSummary(
                sampling_interval_ms=10,
                peak_rss_bytes=20,
            ),
        },
    ),
)
def test_process_result_rejects_incoherent_termination(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ProcessResult.model_validate(payload)


def test_process_result_rejects_the_derived_timeout_projection_as_input() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProcessResult.model_validate({"timed_out": True})

    result = ProcessResult(cancellation_cause=ProcessCancellationCause.TIMEOUT)
    assert result.cancellation_cause is ProcessCancellationCause.TIMEOUT
    assert result.timed_out is True
    assert result.validated_copy() == result


def test_process_result_requires_a_nested_typed_termination() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProcessResult.model_validate({"exit_code": 0})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProcessResult.model_validate({"terminating_signal": 15})

    exited = ProcessResult(termination=ExitedProcessTermination(exit_code=0))

    assert exited.termination.kind is ProcessTerminationKind.EXITED
    assert exited.termination == ExitedProcessTermination(exit_code=0)
    assert exited.model_dump(mode="json") == {
        "termination": {"kind": "exited", "exit_code": 0},
        "wall_time_ns": None,
        "peak_rss_bytes": None,
        "cancellation_cause": None,
        "cleanup_complete": None,
        "resources": None,
        "stdout": None,
        "stderr": None,
        "timed_out": False,
    }
    assert exited.validated_copy() == exited


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
            semantics=RunSemantics.unavailable(origin="import", adapter="import"),
        )

    manifest = ImportRunManifest(
        run_id="run",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        execution_status=ExecutionStatus.NOT_APPLICABLE,
        capture_status=CaptureStatus.REGISTERED,
        validation_status=ValidationStatus.NOT_REQUESTED,
        environment_id=DIGEST,
        semantics=RunSemantics.unavailable(origin="import", adapter="import"),
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


def test_oracle_receipt_parser_preserves_pair_binding() -> None:
    receipt = parse_oracle_receipt(
        json.dumps(
            {
                "schema_version": "flameox.oracle-receipt.v1",
                "status": "pass",
                "reason": "pair_match",
                "binding": {
                    "pair_id": "sha256:" + "1" * 64,
                    "treatment": "candidate",
                    "input_identity": "sha256:" + "2" * 64,
                    "workload_identity": "sha256:" + "3" * 64,
                    "output_identity": "sha256:" + "4" * 64,
                    "compared_property": "forward",
                    "oracle_identity": "sha256:" + "5" * 64,
                    "tolerance": {"absolute": 0, "relative": 0, "equal_nan": False},
                },
            }
        ).encode()
    )

    assert receipt.binding is not None
    assert receipt.binding.treatment == "candidate"
    assert receipt.binding.input_identity == "sha256:" + "2" * 64


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
