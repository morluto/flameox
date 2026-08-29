from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)

from flameox.action_graph import ManualAction, NextAction, ToolAction
from flameox.domain.executables import ResolvedExecutable
from flameox.domain.identity import canonical_json_bytes, digest_model
from flameox.domain.scalars import NumericValue
from flameox.models import ContractModel

Identifier = Annotated[str, StringConstraints(min_length=1, max_length=200)]
VariantName = Annotated[str, StringConstraints(max_length=200)]
Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=500)]
MAX_RUN_SET_MEMBERS = 1_000
MAX_RUN_SET_SELECTION_BYTES = 16 * 1024


def validate_run_set_selection(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    count = 0

    def visit(item: JsonValue, depth: int) -> None:
        nonlocal count
        count += 1
        if count > 256:
            raise ValueError("run-set selection is limited to 256 JSON values")
        if depth > 4:
            raise ValueError("run-set selection is limited to 4 nested levels")
        if isinstance(item, str) and len(item) > 500:
            raise ValueError("run-set selection strings are limited to 500 characters")
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
        elif isinstance(item, dict):
            for key, child in item.items():
                if len(key) > 100:
                    raise ValueError("run-set selection keys are limited to 100 characters")
                visit(child, depth + 1)

    visit(value, 0)
    if len(canonical_json_bytes(value)) > MAX_RUN_SET_SELECTION_BYTES:
        raise ValueError("run-set selection exceeds its 16 KiB serialized budget")
    return value


class LimitationSource(StrEnum):
    ADAPTER = "adapter"
    CONTAINMENT = "containment"
    PREFLIGHT = "preflight"
    COLLECTOR = "collector"
    ARTIFACT = "artifact"
    RESOURCE = "resource"
    VALIDATION = "validation"


class LimitationDetail(ContractModel):
    """A bounded, machine-readable explanation of an evidence limitation."""

    source: LimitationSource
    code: Annotated[
        str,
        StringConstraints(
            min_length=1,
            max_length=100,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        ),
    ]
    message: Annotated[str, StringConstraints(min_length=1, max_length=500)]


def utc_now() -> datetime:
    return datetime.now(UTC)


class EvidenceLevel(StrEnum):
    OBSERVED = "observed"
    DERIVED = "derived"
    INFERRED = "inferred"


class ResourceAvailability(StrEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class IdentityQuality(StrEnum):
    CLEAN = "clean"
    EXACT = "exact"
    PARTIAL = "partial"


class Sensitivity(StrEnum):
    NORMAL = "normal"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"


_SENSITIVITY_ORDER = {
    Sensitivity.NORMAL: 0,
    Sensitivity.INTERNAL: 1,
    Sensitivity.SENSITIVE: 2,
}


def effective_sensitivity(
    values: tuple[Sensitivity, ...] | list[Sensitivity],
) -> Sensitivity:
    if not values:
        return Sensitivity.NORMAL
    return max(values, key=_SENSITIVITY_ORDER.__getitem__)


class ArtifactKind(StrEnum):
    EXECUTION_TRACE = "execution_trace"
    OTLP_TRACE = "otlp_trace"
    PYTHON_STARTUP = "python_startup"
    TEST_EXECUTION = "test_execution"
    SAMPLE_PROFILE = "sample_profile"
    MEMORY_PROFILE = "memory_profile"
    BENCHMARK_SAMPLES = "benchmark_samples"
    EXECUTION_COVERAGE = "execution_coverage"
    SEMANTIC_OBSERVATIONS = "semantic_observations"
    PROCESS_OUTPUT = "process_output"
    VALIDATION_OUTPUT = "validation_output"
    CORE_DUMP = "core_dump"
    SANITIZER_REPORT = "sanitizer_report"
    SOURCE_SNAPSHOT = "source_snapshot"
    COLLECTOR_METADATA = "collector_metadata"
    ANALYSIS_RESULT = "analysis_result"
    PROCESS_TREE_SNAPSHOT = "process_tree_snapshot"
    EXPERIMENT_CONFIGURATION = "experiment_configuration"
    INFERENCE_REQUEST_TRACE = "inference_request_trace"
    INFERENCE_RESULT = "inference_result"
    KERNEL_BUILD = "kernel_build"
    KERNEL_PROFILE = "kernel_profile"
    METAL_TRACE = "metal_trace"


class RunType(StrEnum):
    EXECUTION = "execution"
    IMPORT = "import"


class ExecutionStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class CaptureStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    REGISTERED = "registered"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    CANCELLED = "cancelled"


class ValidationStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"
    CANCELLED = "cancelled"


class GenerationStatus(StrEnum):
    STAGED = "staged"
    PUBLISHED = "published"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    QUARANTINED = "quarantined"


class ExperimentOutcomeMethod(StrEnum):
    FIXED_ATTEMPTS_V1 = "fixed_attempts_v1"
    ABSENCE_OF_FAILURE_FIXED_ATTEMPTS_V1 = "absence_of_failure_fixed_attempts_v1"


class ExperimentOutcomeGoal(StrEnum):
    ABSENCE_OF_FAILURE = "absence_of_failure"


class ExperimentOutcomeDisposition(StrEnum):
    ALL_CLEAN = "all_clean"
    BASE_ONLY_FAILURE = "base_only_failure"
    CANDIDATE_ONLY_FAILURE = "candidate_only_failure"
    MIXED = "mixed"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class TrialOutcome(StrEnum):
    UNATTEMPTED = "unattempted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    UNSUPPORTED = "unsupported"
    RESOURCE_POLICY = "resource_policy"
    ORACLE_FAILED = "oracle_failed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    INVALID = "invalid"


class TrialFailureClass(StrEnum):
    NONE = "none"
    UNATTEMPTED = "unattempted"
    PROCESS_FAILURE = "process_failure"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    ORACLE_UNSUPPORTED = "oracle_unsupported"
    UNSUPPORTED_ENVIRONMENT = "unsupported_environment"
    RESOURCE_POLICY = "resource_policy"
    ORACLE_FAILURE = "oracle_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    ORACLE_INCONCLUSIVE = "oracle_inconclusive"
    ORACLE_RECEIPT_ERROR = "oracle_receipt_error"


class ExperimentRole(StrEnum):
    EXPLORATORY = "exploratory"
    CONFIRMATORY = "confirmatory"


class OracleStrength(StrEnum):
    EXECUTION_CHECK = "execution_check"
    CONTRACT_CHECK = "contract_check"
    CROSS_TREATMENT_EQUIVALENCE = "cross_treatment_equivalence"


class OracleStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"


OracleScalar = None | bool | int | float | str


class ScalarOracleReceiptValue(ContractModel):
    kind: Literal["scalar"]
    value: OracleScalar

    @field_validator("value")
    @classmethod
    def bounded_value(cls, value: OracleScalar) -> OracleScalar:
        if isinstance(value, str) and len(value) > 500:
            raise ValueError("scalar receipt strings are limited to 500 characters")
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 10**18:
            raise ValueError("scalar receipt integers are limited to 18 digits")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("receipt scalar numbers must be finite")
        return value


class DigestOracleReceiptValue(ContractModel):
    kind: Literal["digest"]
    value: Digest


type OracleReceiptValue = Annotated[
    ScalarOracleReceiptValue | DigestOracleReceiptValue,
    Field(discriminator="kind"),
]


class OracleTolerance(ContractModel):
    absolute: Annotated[float, Field(ge=0)] | None = None
    relative: Annotated[float, Field(ge=0)] | None = None
    equal_nan: bool = False

    @field_validator("absolute", "relative")
    @classmethod
    def finite_tolerance(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("oracle tolerances must be finite")
        return value


class OracleReceiptV1(ContractModel):
    schema_version: Literal["flameox.oracle-receipt.v1"]
    status: OracleStatus
    reason: Annotated[
        str,
        StringConstraints(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._:-]+$"),
    ]
    case_id: Identifier | None = None
    output_field: Identifier | None = None
    coordinate: Annotated[tuple[str | int, ...], Field(max_length=8)] = ()
    expected: OracleReceiptValue | None = None
    observed: OracleReceiptValue | None = None
    absolute_error: Annotated[float, Field(ge=0)] | None = None
    relative_error: Annotated[float, Field(ge=0)] | None = None
    tolerance: OracleTolerance | None = None
    diagnostic_roles: Annotated[tuple[Identifier, ...], Field(max_length=8)] = ()
    limitations: Annotated[
        tuple[Annotated[str, StringConstraints(max_length=500)], ...],
        Field(max_length=16),
    ] = ()

    @field_validator("absolute_error", "relative_error")
    @classmethod
    def finite_error(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("oracle errors must be finite")
        return value

    @field_validator("coordinate")
    @classmethod
    def bounded_coordinate(cls, value: tuple[str | int, ...]) -> tuple[str | int, ...]:
        if any(
            (isinstance(item, bool))
            or (isinstance(item, str) and (not item or len(item) > 200))
            or (isinstance(item, int) and abs(item) > 10**9)
            for item in value
        ):
            raise ValueError("coordinate components exceed their bounds")
        return value


class OracleReceiptRecord(ContractModel):
    receipt: OracleReceiptV1
    receipt_artifact_id: Digest
    validation_stdout_artifact_id: Digest | None = None
    validation_stderr_artifact_id: Digest | None = None
    diagnostic_artifact_ids: Annotated[tuple[Digest, ...], Field(max_length=8)] = ()
    parsing_limitations: Annotated[
        tuple[Annotated[str, StringConstraints(max_length=500)], ...],
        Field(max_length=16),
    ] = ()


class InvestigationStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    ARCHIVED = "archived"


class FindingAssessment(StrEnum):
    UNASSESSED = "unassessed"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class FindingConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class FindingLifecycle(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class ComparisonDecision(StrEnum):
    MEANINGFUL_IMPROVEMENT = "meaningful_improvement"
    MEANINGFUL_REGRESSION = "meaningful_regression"
    NO_MEANINGFUL_DIFFERENCE = "no_meaningful_difference"
    INCONCLUSIVE = "inconclusive"
    DESCRIPTIVE_ONLY = "descriptive_only"


class ComparisonValidity(StrEnum):
    VALID = "valid"
    EXPLORATORY = "exploratory"
    INVALID = "invalid"


class EvidenceReferenceType(StrEnum):
    ANALYSIS = "analysis"
    ARTIFACT = "artifact"
    COMPARISON = "comparison"
    GENERATION = "generation"
    OBSERVATION = "observation"
    RUN = "run"
    RUN_SET = "run_set"
    TRIAL = "trial"


class EvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXT = "context"
    VALIDATES = "validates"


class MetricPolarity(StrEnum):
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"
    NEUTRAL = "neutral"


class MetricSource(StrEnum):
    MEASUREMENT = "measurement"
    RUNTIME_RESOURCE = "runtime_resource"
    KERNEL_VALIDATION = "kernel_validation"


class MetricValueDomain(StrEnum):
    STRICTLY_POSITIVE = "strictly_positive"


class MetricZeroPolicy(StrEnum):
    REJECT = "reject"


class MeasurementSeriesSelector(ContractModel):
    """The semantic identity of one normalized measurement population."""

    scope: Identifier
    aggregation: Identifier
    phase: Identifier | None = None
    loop_count: Annotated[int, Field(ge=1)] | None = None
    dimensions: dict[str, str] = Field(default_factory=dict, max_length=64)


class ComparisonMetricContract(ContractModel):
    """A closed contract for the population and transform used by a comparison."""

    schema_version: Literal[1] = 1
    source: MetricSource
    metric: Identifier
    unit: Identifier
    polarity: MetricPolarity
    estimand: Literal["median_paired_log_ratio", "difference_in_median_logs"]
    value_domain: Literal[MetricValueDomain.STRICTLY_POSITIVE] = MetricValueDomain.STRICTLY_POSITIVE
    zero_policy: Literal[MetricZeroPolicy.REJECT] = MetricZeroPolicy.REJECT
    series: MeasurementSeriesSelector | None = None

    @computed_field(return_type=str)  # type: ignore[prop-decorator]
    @property
    def contract_id(self) -> str:
        payload = self.model_dump(mode="json", exclude={"contract_id"})
        return digest_model(payload)

    @model_validator(mode="after")
    def source_semantics_are_coherent(self) -> ComparisonMetricContract:
        if self.source is MetricSource.RUNTIME_RESOURCE:
            if self.unit != "bytes":
                raise ValueError("runtime-resource metric contracts require byte units")
            if self.series is not None:
                raise ValueError("runtime-resource metric contracts cannot select a series")
        return self


class ConfidenceInterval(ContractModel):
    low: Annotated[float, Field(allow_inf_nan=False)]
    high: Annotated[float, Field(allow_inf_nan=False)]
    level: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]

    @model_validator(mode="after")
    def lower_bound_does_not_exceed_upper_bound(self) -> ConfidenceInterval:
        if self.low > self.high:
            raise ValueError("confidence interval lower bound exceeds its upper bound")
        return self


_CONFIDENCE_PROJECTION_FIELDS = ("confidence_low", "confidence_high", "confidence_level")


def _advertise_confidence_interval_projections(schema: dict[str, Any]) -> None:
    properties = schema.setdefault("properties", {})
    assert isinstance(properties, dict)
    properties.pop("confidence_interval", None)
    for field_name in _CONFIDENCE_PROJECTION_FIELDS:
        properties[field_name] = {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "default": None,
            "readOnly": True,
            "title": field_name.replace("_", " ").title(),
        }


class ConfidenceIntervalFields(ContractModel):
    """Optional confidence interval with flattened compatibility projections."""

    model_config = ConfigDict(json_schema_extra=_advertise_confidence_interval_projections)

    confidence_interval: ConfidenceInterval | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def parse_flat_confidence_interval(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        supplied = tuple(name for name in _CONFIDENCE_PROJECTION_FIELDS if name in value)
        if not supplied:
            return value
        if "confidence_interval" in value:
            raise ValueError(
                "use either confidence_interval or flattened confidence fields, not both"
            )
        parsed = dict(value)
        low = parsed.pop("confidence_low", None)
        high = parsed.pop("confidence_high", None)
        level = parsed.pop("confidence_level", None)
        observed = (low, high, level)
        if all(item is None for item in observed):
            parsed["confidence_interval"] = None
        elif any(item is None for item in observed):
            raise ValueError("confidence bounds and level must appear together")
        else:
            parsed["confidence_interval"] = {"low": low, "high": high, "level": level}
        return parsed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence_low(self) -> float | None:
        return self.confidence_interval.low if self.confidence_interval is not None else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence_high(self) -> float | None:
        return self.confidence_interval.high if self.confidence_interval is not None else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confidence_level(self) -> float | None:
        return self.confidence_interval.level if self.confidence_interval is not None else None


class Integrity(ContractModel):
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    hashed_at: datetime = Field(default_factory=utc_now)


class ArtifactContent(ContractModel):
    schema_version: Literal[1] = 1
    artifact_id: Digest
    byte_length: Annotated[int, Field(ge=0)]
    payload_name: Identifier
    integrity: Integrity

    @model_validator(mode="after")
    def content_id_matches_integrity(self) -> ArtifactContent:
        if self.artifact_id != f"sha256:{self.integrity.sha256}":
            raise ValueError("artifact id must match its integrity digest")
        return self


class ArtifactRegistration(ContractModel):
    schema_version: Literal[1] = 1
    registration_id: Identifier
    run_id: Identifier
    artifact_id: Digest
    display_name: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    media_type: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    kind: ArtifactKind
    role: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    producer: str | None = None
    producer_version: str | None = None
    sensitivity: Sensitivity
    registered_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def enforce_sensitivity_floor(self) -> ArtifactRegistration:
        if (
            self.kind
            in {
                ArtifactKind.CORE_DUMP,
                ArtifactKind.SOURCE_SNAPSHOT,
                ArtifactKind.INFERENCE_REQUEST_TRACE,
            }
            and self.sensitivity is not Sensitivity.SENSITIVE
        ):
            raise ValueError(f"{self.kind.value} artifacts must be sensitive")
        return self


CommandArgument = Annotated[
    str,
    StringConstraints(min_length=1, max_length=32_768, pattern=r"^[^\x00]*$"),
]


class CommandSpec(ContractModel):
    argv: Annotated[tuple[CommandArgument, ...], Field(min_length=1)]
    cwd: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    env_overrides: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: Annotated[float, Field(gt=0, le=86_400)] = 300


class EnvironmentRecord(ContractModel):
    schema_version: Literal[1] = 1
    environment_id: Digest
    observed_at: datetime = Field(default_factory=utc_now)
    identity_quality: IdentityQuality
    fields: dict[str, JsonValue]
    missing_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def identity_matches_content(self) -> EnvironmentRecord:
        expected = digest_model(
            {"identity_quality": self.identity_quality.value, "fields": self.fields}
        )
        if self.environment_id != expected:
            raise ValueError("environment id must match identity fields")
        return self


class AcceleratorMigMode(StrEnum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


class AcceleratorLinkKind(StrEnum):
    NVLINK = "nvlink"
    PCIE = "pcie"
    HOST_BRIDGE = "host_bridge"
    NUMA = "numa"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class AcceleratorIdentityStatus(StrEnum):
    AVAILABLE = "available"
    MISSING = "missing"
    PERMISSION_DENIED = "permission_denied"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class AcceleratorDevice(ContractModel):
    index: Annotated[int, Field(ge=0)]
    stable_id: str | None = None
    pci_bus_id: str | None = None
    model: str | None = None
    compute_capability: str | None = None
    memory_bytes: Annotated[int, Field(ge=0)] | None = None
    memory_mib: Annotated[int, Field(ge=0)] | None = None
    gpu_core_count: Annotated[int, Field(gt=0)] | None = None
    mig_mode: AcceleratorMigMode = AcceleratorMigMode.UNKNOWN


class AcceleratorLink(ContractModel):
    left: Annotated[int, Field(ge=0)]
    right: Annotated[int, Field(ge=0)]
    kind: AcceleratorLinkKind
    width: Annotated[int, Field(gt=0)] | None = None


class AcceleratorIdentityFacet(ContractModel):
    provider: Literal["cuda", "metal"]
    status: AcceleratorIdentityStatus
    identity_quality: IdentityQuality
    driver_version: str | None = None
    runtime_version: str | None = None
    provider_version: str | None = None
    metal_support: str | None = None
    unified_memory_bytes: Annotated[int, Field(ge=0)] | None = None
    macos_product_version: str | None = None
    macos_build: str | None = None
    devices: tuple[AcceleratorDevice, ...] = ()
    links: tuple[AcceleratorLink, ...] = ()
    missing_fields: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class SourceState(ContractModel):
    schema_version: Literal[1] = 1
    source_state_id: Digest
    identity_quality: IdentityQuality
    repository_root: str | None = None
    head_commit: str | None = None
    diff_digest: Digest | None = None
    executable_digest: Digest | None = None
    build_id: str | None = None
    fields: dict[str, JsonValue] = Field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def identity_matches_content(self) -> SourceState:
        content: dict[str, JsonValue] = {
            "identity_quality": self.identity_quality.value,
            "fields": self.fields,
            "missing_fields": list(self.missing_fields),
        }
        for name, value in (
            ("repository_root", self.repository_root),
            ("head_commit", self.head_commit),
            ("diff_digest", self.diff_digest),
            ("executable_digest", self.executable_digest),
            ("build_id", self.build_id),
        ):
            if value is not None:
                content[name] = value
        if self.source_state_id != digest_model(content):
            raise ValueError("source-state id must match identity fields")
        return self


class WorkloadDefinition(ContractModel):
    schema_version: Literal[1] = 1
    workload_definition_id: Digest
    name: Identifier
    command_template: tuple[str, ...]
    parameter_names: tuple[str, ...] = ()
    validation_spec_id: Digest | None = None


class WorkloadInstance(ContractModel):
    schema_version: Literal[1] = 1
    workload_instance_id: Digest
    workload_definition_id: Digest
    command: CommandSpec
    executable_binding: ResolvedExecutable
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def identity_matches_content(self) -> WorkloadInstance:
        content: dict[str, JsonValue] = {
            "workload_definition_id": self.workload_definition_id,
            "command": self.command.model_dump(mode="json"),
            "parameters": self.parameters,
        }
        content["executable_binding"] = self.executable_binding.model_dump(mode="json")
        expected = digest_model(content)
        if self.workload_instance_id != expected:
            raise ValueError("workload instance id must match its bound command")
        return self


class CapabilityStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    PERMISSION_REQUIRED = "permission_required"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    UNKNOWN = "unknown"


class CapabilityPermissionStatus(StrEnum):
    UNKNOWN_UNTIL_ACTIVE_PROBE = "unknown_until_active_probe"
    NOT_EXERCISED = "not_exercised"
    GRANTED = "granted"
    DENIED = "denied"
    UNKNOWN = "unknown"


class ProbeKind(StrEnum):
    PASSIVE = "passive"
    ACTIVE = "active"


class PreflightMode(StrEnum):
    AUTO = "auto"
    PASSIVE = "passive"
    ACTIVE = "active"


class PreflightDisposition(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"
    EXPLORATORY = "exploratory"


class RequirementKind(StrEnum):
    EXECUTABLE = "executable"
    PYTHON_DISTRIBUTION = "python_distribution"
    CAPABILITY = "capability"


class RequirementStatus(StrEnum):
    AVAILABLE = "available"
    ABSENT = "absent"
    PERMISSION_DENIED = "permission_denied"
    ENVIRONMENT_BLOCKED = "environment_blocked"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
    PROBE_FAILED = "probe_failed"


class CapabilitySetupVerification(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    PASSIVE = "passive"
    ACTIVE = "active"


class CapabilityProvisioning(StrEnum):
    """Who can make a capability usable in the current environment."""

    BUNDLED = "bundled"
    MANAGED_RUNTIME = "managed_runtime"
    WORKLOAD_ENVIRONMENT = "workload_environment"
    HOST = "host"
    THIRD_PARTY_APPROVAL = "third_party_approval"
    UNSUPPORTED = "unsupported"


class CapabilityExtra(StrEnum):
    CPU = "cpu"
    EXECUTION = "execution"
    HARDWARE = "hardware"
    INFERENCE = "inference"
    MEMORY = "memory"
    REDUCTION = "reduction"
    TEST = "test"
    TRACE = "trace"
    TORCH = "torch"
    TOXIPROXY = "toxiproxy"


_MANAGED_RUNTIME_EXTRAS = frozenset(CapabilityExtra) - {CapabilityExtra.TOXIPROXY}


def parse_managed_runtime_extras(value: object) -> tuple[CapabilityExtra, ...]:
    """Parse the package extras that may be carried into a versioned runtime."""

    if not isinstance(value, (list, tuple)):
        return ()
    extras: set[CapabilityExtra] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        try:
            extra = CapabilityExtra(item)
        except ValueError:
            continue
        if extra in _MANAGED_RUNTIME_EXTRAS:
            extras.add(extra)
    return tuple(sorted(extras, key=str))


class CapabilitySetup(ContractModel):
    """The bounded setup action FlameOx can take for one capability."""

    extra: CapabilityExtra
    requirement: str | None = None
    next_action: ManualAction
    verification_action: ToolAction


class AdapterSetup(ContractModel):
    """The agent-owned setup action for an installed third-party adapter."""

    adapter: Identifier
    distribution: Identifier
    package_identity: str
    next_action: ToolAction
    verification_action: ToolAction


class CapabilityReport(ContractModel):
    schema_version: Literal[1] = 1
    adapter: Identifier
    status: CapabilityStatus
    provisioning: CapabilityProvisioning = CapabilityProvisioning.HOST
    executable: str | None = None
    import_location: str | None = None
    version: str | None = None
    supported_modes: tuple[str, ...] = ()
    supported_formats: tuple[str, ...] = ()
    platform: str | None = None
    architecture: str | None = None
    permissions: tuple[str, ...] = ()
    permission_status: CapabilityPermissionStatus | None = None
    restrictions: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    remediation: tuple[str, ...] = ()
    setup: CapabilitySetup | AdapterSetup | None = None
    setup_verification: CapabilitySetupVerification = CapabilitySetupVerification.NOT_REQUIRED
    probe_kind: ProbeKind = ProbeKind.PASSIVE
    probed_at: datetime | None = None


class RequirementResult(ContractModel):
    requirement: str
    kind: RequirementKind
    required: bool
    probe_kind: ProbeKind
    status: RequirementStatus
    identity: str | None = None
    evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    remediation: tuple[str, ...] = ()
    next_action: NextAction | None = None


class PreflightReport(ContractModel):
    schema_version: Literal[1] = 1
    preflight_id: Digest
    mode: ProbeKind
    disposition: PreflightDisposition
    requirements: tuple[RequirementResult, ...]
    limitations: tuple[str, ...] = ()


class WritableRootBinding(ContractModel):
    target_path: str
    storage_path: str
    target_identity: Digest


class ExternalExecutionContext(ContractModel):
    orchestrator: Annotated[
        str,
        StringConstraints(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._:/-]+$"),
    ]
    provider: Annotated[
        str,
        StringConstraints(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._:/-]+$"),
    ]
    lease_id: Annotated[
        str,
        StringConstraints(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:/-]+$"),
    ]
    worker_id: Annotated[
        str,
        StringConstraints(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:/-]+$"),
    ]
    orchestration_run_id: Annotated[
        str,
        StringConstraints(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:/-]+$"),
    ]
    sensitivity: Literal[Sensitivity.INTERNAL, Sensitivity.SENSITIVE] = Sensitivity.INTERNAL


class ExecutionIdentityInputKind(StrEnum):
    PYTHON_MODULE = "python_module"
    NATIVE_FILE = "native_file"


class ExecutionIdentityInputStatus(StrEnum):
    EXACT = "exact"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    RESOLUTION_FAILED = "resolution_failed"
    NOT_OBSERVED = "not_observed"


class ExecutionIdentityBasis(StrEnum):
    PROJECT_SOURCE = "project_source"
    DISTRIBUTION_METADATA = "distribution_metadata"
    INTERPRETER_STDLIB = "interpreter_stdlib"
    EXPLICIT_FILE = "explicit_file"


class ExecutionIdentityQuality(StrEnum):
    EXACT = "exact"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ExecutionIdentityInput(ContractModel):
    kind: ExecutionIdentityInputKind
    requested: str
    identity_basis: ExecutionIdentityBasis | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    configured_path: str | None = None
    resolved_path: str | None = None
    loaded_path: str | None = None
    distribution: str | None = None
    version: str | None = None
    content_digest: Digest | None = None
    build_id: str | None = None
    status: ExecutionIdentityInputStatus
    limitations: tuple[str, ...] = ()


class WorkloadExecutionIdentity(ContractModel):
    schema_version: Literal[1] = 1
    identity_id: Digest
    quality: ExecutionIdentityQuality
    inputs: tuple[ExecutionIdentityInput, ...]
    missing_inputs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def identity_matches_content(self) -> WorkloadExecutionIdentity:
        expected = digest_model(
            {
                "quality": self.quality,
                "inputs": [item.model_dump(mode="json") for item in self.inputs],
                "missing_inputs": list(self.missing_inputs),
            }
        )
        if self.identity_id != expected:
            raise ValueError("execution identity id must match its inputs")
        return self


type CaptureContainment = Literal["active", "degraded", "uncontained", "unavailable"]


class SemanticOption(ContractModel):
    name: Identifier
    value: JsonValue


class CaptureScope(ContractModel):
    """Property-defining capture scope projected from validated adapter options."""

    mode: SemanticOption | None = None
    process_scope: SemanticOption | None = None
    bounds: dict[Identifier, JsonValue] = Field(default_factory=dict, max_length=32)
    filters: dict[Identifier, JsonValue] = Field(default_factory=dict, max_length=32)


class RunSemantics(ContractModel):
    """Durable authority for what a run attempted, independent of produced bytes."""

    origin: Literal["capture", "import", "internal"]
    adapter: Identifier | None = None
    adapter_version: Annotated[str, StringConstraints(max_length=200)] | None = None
    configuration: dict[Identifier, JsonValue] = Field(default_factory=dict, max_length=64)
    scope: CaptureScope = Field(default_factory=CaptureScope)
    unavailable_fields: Annotated[tuple[Identifier, ...], Field(max_length=32)] = ()

    @property
    def semantic_id(self) -> str:
        return digest_model(self.model_dump(mode="json"))

    @property
    def effective_options(self) -> dict[str, JsonValue]:
        options = dict(self.configuration)
        if self.scope.mode is not None:
            options[self.scope.mode.name] = self.scope.mode.value
        if self.scope.process_scope is not None:
            options[self.scope.process_scope.name] = self.scope.process_scope.value
        options.update(self.scope.bounds)
        options.update(self.scope.filters)
        return options

    @model_validator(mode="after")
    def semantic_fields_have_one_owner(self) -> RunSemantics:
        if self.origin == "capture" and self.adapter is None:
            raise ValueError("captured run semantics require an adapter")
        owned_names = set(self.configuration) | set(self.scope.bounds) | set(self.scope.filters)
        if self.scope.mode is not None:
            owned_names.add(self.scope.mode.name)
        if self.scope.process_scope is not None:
            owned_names.add(self.scope.process_scope.name)
        if len(owned_names) != (
            len(self.configuration)
            + len(self.scope.bounds)
            + len(self.scope.filters)
            + int(self.scope.mode is not None)
            + int(self.scope.process_scope is not None)
        ):
            raise ValueError("run semantic fields must have exactly one owner")
        if len(set(self.unavailable_fields)) != len(self.unavailable_fields):
            raise ValueError("unavailable semantic fields must be unique")
        return self

    @classmethod
    def unavailable(
        cls,
        *,
        origin: Literal["import", "internal"],
        adapter: str | None,
        adapter_version: str | None = None,
        fields: tuple[str, ...] = ("configuration", "scope"),
    ) -> RunSemantics:
        return cls(
            origin=origin,
            adapter=adapter,
            adapter_version=adapter_version,
            unavailable_fields=fields,
        )


class _CapturePlan(ContractModel):
    plan_token: Identifier
    plan_id: Identifier
    run_id: Identifier
    request_digest: Digest
    workspace_id: Identifier
    workload_name: Identifier
    workload_definition_id: Digest
    workload_instance: WorkloadInstance
    semantics: RunSemantics
    dynamic_parameters: tuple[Identifier, ...] = ()
    execution_policy: Literal["trusted_local", "approved_agent"]
    adapter_execution_plan: dict[str, JsonValue] | None = None
    collector_argv: tuple[str, ...]
    collector_executable_binding: ResolvedExecutable
    oracle_argv: tuple[str, ...] | None = None
    oracle_executable_binding: ResolvedExecutable | None = None
    oracle_launch_executable_binding: ResolvedExecutable | None = None
    oracle_containment: CaptureContainment | None = None
    oracle_network_contained: bool | None = None
    oracle_systemd_scope_unit: str | None = None
    collector_environment: dict[str, str] = Field(default_factory=dict)
    expected_artifact_kinds: tuple[ArtifactKind, ...]
    expected_overhead: str
    permissions: tuple[str, ...] = ()
    preflight: PreflightReport
    writable_roots: tuple[WritableRootBinding, ...] = ()
    external_context: ExternalExecutionContext | None = None
    planned_execution_identity: WorkloadExecutionIdentity
    adapter_capability: CapabilityReport | None = None
    bound_identities: dict[str, JsonValue] = Field(default_factory=dict)
    limits: dict[str, JsonValue] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    limitation_details: tuple[LimitationDetail, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime

    @model_validator(mode="after")
    def expiry_follows_creation(self) -> _CapturePlan:
        if self.expires_at <= self.created_at:
            raise ValueError("capture plan expiry must follow creation")
        return self

    @model_validator(mode="after")
    def oracle_authority_is_complete(self) -> _CapturePlan:
        authority = (
            self.oracle_executable_binding,
            self.oracle_launch_executable_binding,
            self.oracle_containment,
            self.oracle_network_contained,
        )
        if self.oracle_argv is None and any(item is not None for item in authority):
            raise ValueError("oracle authority requires planned argv")
        if self.oracle_argv is not None and any(item is None for item in authority):
            raise ValueError("planned oracle requires complete execution authority")
        return self

    @property
    def adapter(self) -> str:
        if self.semantics.adapter is None:
            raise ValueError("capture plan semantics require an adapter")
        return self.semantics.adapter

    @property
    def adapter_version(self) -> str | None:
        return self.semantics.adapter_version

    @property
    def adapter_options(self) -> dict[str, JsonValue]:
        return self.semantics.effective_options


class ActiveCapturePlan(_CapturePlan):
    containment: Literal["active"]
    network_contained: bool
    systemd_scope_unit: str


class DegradedCapturePlan(_CapturePlan):
    containment: Literal["degraded"]
    network_contained: bool
    systemd_scope_unit: None = None


class UncontainedCapturePlan(_CapturePlan):
    containment: Literal["uncontained"]
    network_contained: Literal[False] = False
    systemd_scope_unit: None = None


class UnavailableCapturePlan(_CapturePlan):
    containment: Literal["unavailable"]
    network_contained: Literal[False] = False
    systemd_scope_unit: None = None


type CapturePlan = Annotated[
    ActiveCapturePlan | DegradedCapturePlan | UncontainedCapturePlan | UnavailableCapturePlan,
    Field(discriminator="containment"),
]

_CAPTURE_PLAN_ADAPTER: TypeAdapter[CapturePlan] = TypeAdapter(CapturePlan)


def parse_capture_plan(value: object) -> CapturePlan:
    return _CAPTURE_PLAN_ADAPTER.validate_python(value)


class ProcessCancellationCause(StrEnum):
    TIMEOUT = "timeout"
    CALLER_CANCELLED = "caller_cancelled"
    OUTPUT_LIMIT = "output_limit"
    IO_FAILURE = "io_failure"
    STORAGE_RESERVE_EXCEEDED = "storage_reserve_exceeded"
    WRITABLE_LIMIT_EXCEEDED = "writable_limit_exceeded"
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
    PROCESS_ERROR = "process_error"
    CRASH_RECOVERY = "crash_recovery"


class ProcessTerminationKind(StrEnum):
    UNREPORTED = "unreported"
    EXITED = "exited"
    SIGNALLED = "signalled"


class UnreportedProcessTermination(ContractModel):
    kind: Literal[ProcessTerminationKind.UNREPORTED] = ProcessTerminationKind.UNREPORTED


class ExitedProcessTermination(ContractModel):
    kind: Literal[ProcessTerminationKind.EXITED] = ProcessTerminationKind.EXITED
    exit_code: Annotated[int, Field(ge=0)]


class SignalledProcessTermination(ContractModel):
    kind: Literal[ProcessTerminationKind.SIGNALLED] = ProcessTerminationKind.SIGNALLED
    signal: Annotated[int, Field(gt=0)]


type ProcessTermination = Annotated[
    UnreportedProcessTermination | ExitedProcessTermination | SignalledProcessTermination,
    Field(discriminator="kind"),
]


def process_termination_from_returncode(returncode: int | None) -> ProcessTermination:
    if returncode is None:
        return UnreportedProcessTermination()
    if returncode >= 0:
        return ExitedProcessTermination(exit_code=returncode)
    return SignalledProcessTermination(signal=-returncode)


def _advertise_process_termination_projections(schema: dict[str, Any]) -> None:
    properties = schema.setdefault("properties", {})
    assert isinstance(properties, dict)
    properties.pop("termination", None)
    properties.update(
        {
            "exit_code": {
                "anyOf": [{"minimum": 0, "type": "integer"}, {"type": "null"}],
                "default": None,
                "title": "Exit Code",
            },
            "terminating_signal": {
                "anyOf": [{"exclusiveMinimum": 0, "type": "integer"}, {"type": "null"}],
                "default": None,
                "title": "Terminating Signal",
            },
        }
    )


class ProcessTerminationFields(ContractModel):
    """Canonical process termination with flattened compatibility projections."""

    model_config = ConfigDict(json_schema_extra=_advertise_process_termination_projections)

    termination: ProcessTermination = Field(
        default_factory=UnreportedProcessTermination,
        exclude=True,
    )

    @model_validator(mode="before")
    @classmethod
    def parse_flat_termination(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        has_exit_code = "exit_code" in value
        has_signal = "terminating_signal" in value
        if not has_exit_code and not has_signal:
            return value
        if "termination" in value:
            raise ValueError("use either termination or flattened termination fields, not both")
        parsed = dict(value)
        exit_code = parsed.pop("exit_code", None)
        signal = parsed.pop("terminating_signal", None)
        if exit_code is not None and signal is not None:
            raise ValueError("a process cannot have both an exit code and a terminating signal")
        if exit_code is not None:
            termination: dict[str, object] = {
                "kind": ProcessTerminationKind.EXITED,
                "exit_code": exit_code,
            }
        elif signal is not None:
            termination = {
                "kind": ProcessTerminationKind.SIGNALLED,
                "signal": signal,
            }
        else:
            termination = {"kind": ProcessTerminationKind.UNREPORTED}
        parsed["termination"] = termination
        return parsed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exit_code(self) -> int | None:
        if isinstance(self.termination, ExitedProcessTermination):
            return self.termination.exit_code
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def terminating_signal(self) -> int | None:
        if isinstance(self.termination, SignalledProcessTermination):
            return self.termination.signal
        return None


type ResourcePolicyCancellationCause = Literal[
    ProcessCancellationCause.STORAGE_RESERVE_EXCEEDED,
    ProcessCancellationCause.WRITABLE_LIMIT_EXCEEDED,
    ProcessCancellationCause.MEMORY_LIMIT_EXCEEDED,
]


class ProcessResult(ProcessTerminationFields):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    wall_time_ns: Annotated[int, Field(ge=0)] | None = None
    peak_rss_bytes: Annotated[int, Field(ge=0)] | None = None
    cancellation_cause: ProcessCancellationCause | None = None
    cleanup_complete: bool | None = None
    resources: RuntimeResourceSummary | None = None
    stdout: str | None = None
    stderr: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def timed_out(self) -> bool:
        return self.cancellation_cause is ProcessCancellationCause.TIMEOUT

    @model_validator(mode="after")
    def resource_summary_is_coherent(self) -> ProcessResult:
        if self.resources is not None and self.peak_rss_bytes != self.resources.peak_rss_bytes:
            raise ValueError("process and resource-summary peak RSS must agree")
        return self


class RuntimeResourceSummary(ContractModel):
    sampling_interval_ms: Annotated[int, Field(gt=0)]
    minimum_free_bytes: Annotated[int, Field(ge=0)] | None = None
    staging_growth_bytes: Annotated[int, Field(ge=0)] | None = None
    writable_root_growth_bytes: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    peak_rss_bytes: Annotated[int, Field(ge=0)] | None = None
    # This identifies the observation method, not a guarantee of the process
    # tree's lifetime maximum.
    peak_rss_backend: str | None = None
    unavailable_metrics: tuple[str, ...] = ()
    policy_termination: ResourcePolicyCancellationCause | None = None


class CaptureLease(ContractModel):
    process_id: Annotated[int, Field(gt=0)]
    process_start_identity: Identifier
    boot_id: Identifier
    heartbeat_monotonic_ns: Annotated[int, Field(ge=0)]
    observed_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def expiry_follows_observation(self) -> CaptureLease:
        if self.expires_at <= self.observed_at:
            raise ValueError("lease expiry must follow observation")
        return self


class _RunManifest(ContractModel):
    schema_version: Literal[2] = 2
    revision: Annotated[int, Field(ge=0)] = 0
    run_id: Identifier
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    capture_status: CaptureStatus
    validation_status: ValidationStatus
    workload_definition_id: Digest | None = None
    workload_instance_id: Digest | None = None
    measurement_protocol_id: Digest | None = None
    source_measurement_run_id: Identifier | None = None
    environment_id: Digest
    source_state_id: Digest | None = None
    semantics: RunSemantics
    command: CommandSpec | None = None
    process: ProcessResult | None = None
    lease: CaptureLease | None = None
    artifacts: tuple[ArtifactRegistration, ...] = ()
    limitations: tuple[str, ...] = ()
    limitation_details: tuple[LimitationDetail, ...] = ()
    preflight: PreflightReport | None = None
    writable_roots: tuple[WritableRootBinding, ...] = ()
    external_context: ExternalExecutionContext | None = None
    execution_identity: WorkloadExecutionIdentity | None = None
    oracle_receipt: OracleReceiptRecord | None = None
    inference_protocol_identity_id: Digest | None = None
    inference_protocol_identity_json: str | None = None

    @model_validator(mode="after")
    def finish_follows_start(self) -> _RunManifest:
        if (
            self.finished_at is not None
            and self.started_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at cannot precede started_at")
        return self


class ImportRunManifest(_RunManifest):
    run_type: Literal[RunType.IMPORT] = RunType.IMPORT
    execution_status: Literal[ExecutionStatus.NOT_APPLICABLE] = ExecutionStatus.NOT_APPLICABLE
    started_at: Literal[None] = None
    finished_at: Literal[None] = None
    workload_definition_id: Literal[None] = None
    workload_instance_id: Literal[None] = None
    measurement_protocol_id: Literal[None] = None
    source_measurement_run_id: Literal[None] = None
    command: Literal[None] = None
    process: Literal[None] = None
    lease: Literal[None] = None
    preflight: Literal[None] = None
    writable_roots: tuple[()] = ()
    external_context: Literal[None] = None
    execution_identity: Literal[None] = None
    oracle_receipt: Literal[None] = None


class ExecutionRunManifest(_RunManifest):
    run_type: Literal[RunType.EXECUTION] = RunType.EXECUTION
    execution_status: Literal[
        ExecutionStatus.PLANNED,
        ExecutionStatus.RUNNING,
        ExecutionStatus.SUCCEEDED,
        ExecutionStatus.FAILED,
        ExecutionStatus.TIMED_OUT,
        ExecutionStatus.CANCELLED,
    ]

    @model_validator(mode="after")
    def lifecycle_is_coherent(self) -> ExecutionRunManifest:
        if self.execution_status is ExecutionStatus.PLANNED:
            if self.capture_status is not CaptureStatus.PENDING:
                raise ValueError("a planned execution requires pending capture state")
            if (
                self.started_at is not None
                or self.finished_at is not None
                or self.process is not None
            ):
                raise ValueError("a planned execution cannot carry runtime observations")
            return self
        if self.execution_status is ExecutionStatus.RUNNING:
            if self.capture_status is not CaptureStatus.RUNNING:
                raise ValueError("a running execution requires running capture state")
            if self.started_at is None:
                raise ValueError("a running execution requires a start timestamp")
            if self.finished_at is not None or self.process is not None:
                raise ValueError("a running execution cannot carry terminal observations")
            return self
        if self.capture_status in {CaptureStatus.PENDING, CaptureStatus.RUNNING}:
            raise ValueError("a terminal execution requires a terminal capture state")
        if self.finished_at is None:
            raise ValueError("a terminal execution requires a finish timestamp")
        return self


type RunManifest = Annotated[
    ImportRunManifest | ExecutionRunManifest,
    Field(discriminator="run_type"),
]

_RUN_MANIFEST_ADAPTER: TypeAdapter[RunManifest] = TypeAdapter(RunManifest)


def parse_run_manifest(value: object) -> RunManifest:
    return _RUN_MANIFEST_ADAPTER.validate_python(value)


def parse_run_manifest_json(value: str | bytes) -> RunManifest:
    return _RUN_MANIFEST_ADAPTER.validate_json(value)


class Investigation(ContractModel):
    schema_version: Literal[1] = 1
    investigation_id: Identifier
    question: ShortText
    symptom: str | None = None
    project_root: str
    status: InvestigationStatus = InvestigationStatus.OPEN
    parent_investigation_id: Identifier | None = None
    created_at: datetime = Field(default_factory=utc_now)


class Hypothesis(ContractModel):
    schema_version: Literal[1] = 1
    hypothesis_id: Identifier
    investigation_id: Identifier
    revision: Annotated[int, Field(ge=1)] = 1
    claim: ShortText
    prediction: ShortText
    discriminating_condition: ShortText
    assessment: FindingAssessment = FindingAssessment.UNASSESSED
    lifecycle: FindingLifecycle = FindingLifecycle.ACTIVE
    created_at: datetime = Field(default_factory=utc_now)


class Experiment(ContractModel):
    schema_version: Literal[1, 2] = 2
    experiment_id: Identifier
    investigation_id: Identifier
    hypothesis_id: Identifier | None = None
    recipe: Identifier
    recipe_version: Identifier
    workload_definition_id: Digest
    experiment_design_id: Digest
    measurement_protocol_id: Digest
    validation_spec_id: Digest | None = None
    primary_metric: Identifier
    metric_source: MetricSource | None = None
    primary_metric_unit: Identifier | None = None
    measurement_series: MeasurementSeriesSelector | None = None
    value_domain: MetricValueDomain | None = None
    zero_policy: MetricZeroPolicy | None = None
    polarity: MetricPolarity
    estimand: Identifier
    practical_threshold: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    confidence_level: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)]
    stopping_rule: dict[str, JsonValue]
    random_seed: Annotated[int, Field(ge=0)]
    role: ExperimentRole
    created_at: datetime = Field(default_factory=utc_now)


class VariantIdentityQuality(StrEnum):
    EXACT_UNIFORM = "exact_uniform"
    HETEROGENEOUS = "heterogeneous"
    INCOMPLETE = "incomplete"
    LEGACY_REPRESENTATIVE = "legacy_representative"


class Variant(ContractModel):
    schema_version: Literal[1, 2] = 2
    variant_id: Identifier
    experiment_id: Identifier
    name: VariantName
    treatment_factor: Identifier | None = None
    treatment_value: JsonValue = None
    treatment_identity_id: Digest | None = None
    identity_quality: VariantIdentityQuality
    source_state_id: Digest | None = None
    workload_instance_id: Digest | None = None
    environment_id: Digest | None = None
    source_state_ids: Annotated[tuple[Digest, ...], Field(max_length=1_000)] = ()
    workload_instance_ids: Annotated[tuple[Digest, ...], Field(max_length=1_000)] = ()
    environment_ids: Annotated[tuple[Digest, ...], Field(max_length=1_000)] = ()
    combination_ids: Annotated[tuple[Digest, ...], Field(max_length=10_000)] = ()
    environment_requirements: dict[str, JsonValue] = Field(default_factory=dict)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    varying_factors: dict[str, tuple[JsonValue, ...]] = Field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_representative(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and value.get("schema_version", 2) == 1:
            migrated = dict(value)
            migrated.setdefault("identity_quality", VariantIdentityQuality.LEGACY_REPRESENTATIVE)
            source_state_id = migrated.get("source_state_id")
            workload_instance_id = migrated.get("workload_instance_id")
            migrated.setdefault(
                "source_state_ids", (source_state_id,) if source_state_id is not None else ()
            )
            migrated.setdefault(
                "workload_instance_ids",
                (workload_instance_id,) if workload_instance_id is not None else (),
            )
            migrated.setdefault(
                "limitations",
                (
                    "schema-v1 variant identities are representative-only and do not prove "
                    "cohort uniformity",
                ),
            )
            return migrated
        return value

    @model_validator(mode="after")
    def cohort_identity_is_coherent(self) -> Variant:
        for field_name in (
            "source_state_ids",
            "workload_instance_ids",
            "environment_ids",
            "combination_ids",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must contain unique identities")
        for singular_name, plural_name in (
            ("source_state_id", "source_state_ids"),
            ("workload_instance_id", "workload_instance_ids"),
            ("environment_id", "environment_ids"),
        ):
            singular = getattr(self, singular_name)
            plural = getattr(self, plural_name)
            if singular is not None and plural != (singular,):
                raise ValueError(
                    f"singular {singular_name} requires exactly one matching cohort identity"
                )
        if self.treatment_factor is None:
            if self.treatment_identity_id is not None:
                raise ValueError("treatment identity requires a treatment factor")
        else:
            if not isinstance(self.treatment_value, str | int | float | bool):
                raise ValueError("treatment values must be scalar JSON values")
            value_type = (
                "bool"
                if type(self.treatment_value) is bool
                else "int"
                if type(self.treatment_value) is int
                else "float"
                if type(self.treatment_value) is float
                else "string"
            )
            expected = digest_model(
                {
                    "factor": self.treatment_factor,
                    "value_type": value_type,
                    "value": self.treatment_value,
                }
            )
            if self.treatment_identity_id != expected:
                raise ValueError("treatment identity must match its typed factor value")
        heterogeneous = bool(self.varying_factors) or any(
            len(values) > 1
            for values in (
                self.source_state_ids,
                self.workload_instance_ids,
                self.environment_ids,
            )
        )
        if self.identity_quality is VariantIdentityQuality.EXACT_UNIFORM and heterogeneous:
            raise ValueError("an exact-uniform variant cannot contain heterogeneous identities")
        if self.identity_quality is VariantIdentityQuality.HETEROGENEOUS and not heterogeneous:
            raise ValueError("a heterogeneous variant must expose its varying cohort identity")
        if self.identity_quality is VariantIdentityQuality.LEGACY_REPRESENTATIVE:
            if self.schema_version != 1:
                raise ValueError("legacy representative quality is reserved for schema v1")
        elif self.schema_version != 2:
            raise ValueError("schema-v1 variants must be marked legacy representative")
        return self


class _Trial(ContractModel):
    schema_version: Literal[2] = 2
    trial_id: Identifier
    experiment_id: Identifier
    variant_id: Identifier
    run_id: Identifier | None = None
    combination_id: Digest
    factors: dict[str, JsonValue] = Field(default_factory=dict, max_length=8)
    block_id: Identifier | None = None
    order_in_block: Annotated[int, Field(ge=0)] | None = None
    parameter_name: str | None = None
    parameter_value: NumericValue | None = None
    attempt: Annotated[int, Field(ge=1)] = 1
    validation_status: ValidationStatus
    oracle_receipt: OracleReceiptV1 | None = None
    oracle_receipt_artifact_id: Digest | None = None


class SucceededTrial(_Trial):
    outcome: Literal[TrialOutcome.SUCCEEDED] = TrialOutcome.SUCCEEDED
    exclusion_reason: Literal[None] = None
    failure_class: Literal[TrialFailureClass.NONE] = TrialFailureClass.NONE


class UnattemptedTrial(_Trial):
    outcome: Literal[TrialOutcome.UNATTEMPTED] = TrialOutcome.UNATTEMPTED
    exclusion_reason: str
    failure_class: Literal[TrialFailureClass.UNATTEMPTED] = TrialFailureClass.UNATTEMPTED


class FailedTrial(_Trial):
    outcome: Literal[TrialOutcome.FAILED] = TrialOutcome.FAILED
    exclusion_reason: str
    failure_class: Literal[TrialFailureClass.PROCESS_FAILURE] = TrialFailureClass.PROCESS_FAILURE


class TimedOutTrial(_Trial):
    outcome: Literal[TrialOutcome.TIMED_OUT] = TrialOutcome.TIMED_OUT
    exclusion_reason: str
    failure_class: Literal[TrialFailureClass.TIMEOUT] = TrialFailureClass.TIMEOUT


class CancelledTrial(_Trial):
    outcome: Literal[TrialOutcome.CANCELLED] = TrialOutcome.CANCELLED
    exclusion_reason: str
    failure_class: Literal[TrialFailureClass.CANCELLATION] = TrialFailureClass.CANCELLATION


class UnsupportedTrial(_Trial):
    outcome: Literal[TrialOutcome.UNSUPPORTED] = TrialOutcome.UNSUPPORTED
    exclusion_reason: str
    failure_class: Literal[
        TrialFailureClass.ORACLE_UNSUPPORTED,
        TrialFailureClass.UNSUPPORTED_ENVIRONMENT,
    ]


class ResourcePolicyTrial(_Trial):
    outcome: Literal[TrialOutcome.RESOURCE_POLICY] = TrialOutcome.RESOURCE_POLICY
    exclusion_reason: str
    failure_class: Literal[TrialFailureClass.RESOURCE_POLICY] = TrialFailureClass.RESOURCE_POLICY


class OracleFailedTrial(_Trial):
    outcome: Literal[TrialOutcome.ORACLE_FAILED] = TrialOutcome.ORACLE_FAILED
    exclusion_reason: str
    failure_class: Literal[TrialFailureClass.ORACLE_FAILURE] = TrialFailureClass.ORACLE_FAILURE


class InfrastructureFailedTrial(_Trial):
    outcome: Literal[TrialOutcome.INFRASTRUCTURE_FAILED] = TrialOutcome.INFRASTRUCTURE_FAILED
    exclusion_reason: str
    failure_class: Literal[TrialFailureClass.INFRASTRUCTURE_FAILURE] = (
        TrialFailureClass.INFRASTRUCTURE_FAILURE
    )


class InvalidTrial(_Trial):
    outcome: Literal[TrialOutcome.INVALID] = TrialOutcome.INVALID
    exclusion_reason: str
    failure_class: Literal[
        TrialFailureClass.ORACLE_INCONCLUSIVE,
        TrialFailureClass.ORACLE_RECEIPT_ERROR,
    ]


type Trial = Annotated[
    SucceededTrial
    | UnattemptedTrial
    | FailedTrial
    | TimedOutTrial
    | CancelledTrial
    | UnsupportedTrial
    | ResourcePolicyTrial
    | OracleFailedTrial
    | InfrastructureFailedTrial
    | InvalidTrial,
    Field(discriminator="outcome"),
]

_TRIAL_ADAPTER: TypeAdapter[Trial] = TypeAdapter(Trial)


def parse_trial(value: object) -> Trial:
    return _TRIAL_ADAPTER.validate_python(value)


def parse_trial_json(value: str | bytes) -> Trial:
    return _TRIAL_ADAPTER.validate_json(value)


class _RunSetMember(ContractModel):
    run_id: Identifier
    trial_id: Identifier | None = None
    order: Annotated[int, Field(ge=0)]


class IncludedRunSetMember(_RunSetMember):
    included: Literal[True] = True
    reason: Literal[None] = None


class ExcludedRunSetMember(_RunSetMember):
    included: Literal[False] = False
    reason: ShortText


type RunSetMember = Annotated[
    IncludedRunSetMember | ExcludedRunSetMember,
    Field(discriminator="included"),
]


class RunSet(ContractModel):
    schema_version: Literal[1] = 1
    run_set_id: Digest
    corpus_commit_id: Digest
    created_at: datetime = Field(default_factory=utc_now)
    selection: Annotated[dict[str, JsonValue], Field(max_length=16)]
    members: Annotated[
        tuple[RunSetMember, ...],
        Field(min_length=1, max_length=MAX_RUN_SET_MEMBERS),
    ]
    membership_digest: Digest

    @field_validator("selection")
    @classmethod
    def selection_is_bounded(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return validate_run_set_selection(value)

    @model_validator(mode="after")
    def identity_matches_membership(self) -> RunSet:
        identities = [(member.run_id, member.trial_id) for member in self.members]
        if len(set(identities)) != len(identities):
            raise ValueError("run-set member identities must be unique")
        run_ids = [member.run_id for member in self.members]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("a run can appear in a run set only once")
        membership = [member.model_dump(mode="json") for member in self.members]
        if self.membership_digest != digest_model(membership):
            raise ValueError("run-set membership digest must match its members")
        expected_id = digest_model(
            {
                "corpus_commit_id": self.corpus_commit_id,
                "selection": self.selection,
                "members": membership,
            }
        )
        if self.run_set_id != expected_id:
            raise ValueError("run-set id must match its snapshot and membership")
        return self


class AnalysisRecord(ContractModel):
    schema_version: Literal[1] = 1
    analysis_id: Identifier
    recipe: Identifier
    recipe_version: Identifier
    parameters: dict[str, JsonValue]
    parameters_digest: Digest
    corpus_commit_id: Digest
    input_generation_ids: tuple[Identifier, ...] = ()
    input_run_ids: tuple[Identifier, ...] = ()
    input_artifact_ids: tuple[Digest, ...] = ()
    result_digest: Digest
    result_artifact_id: Digest | None = None
    coverage: dict[str, JsonValue] = Field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    started_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def provenance_is_coherent(self) -> AnalysisRecord:
        if self.parameters_digest != digest_model(self.parameters):
            raise ValueError("analysis parameter digest must match its parameters")
        if self.completed_at < self.started_at:
            raise ValueError("analysis completion cannot precede its start")
        return self


class EvidenceReference(ContractModel):
    schema_version: Literal[2] = 2
    owner_type: Literal["analysis", "finding", "hypothesis"]
    owner_id: Identifier
    owner_revision: Annotated[int, Field(ge=1)] | None = None
    ref_type: EvidenceReferenceType
    ref_id: Identifier
    relation: EvidenceRelation

    @model_validator(mode="after")
    def revisioned_owners_are_exact(self) -> EvidenceReference:
        revisioned = self.owner_type in {"finding", "hypothesis"}
        if revisioned != (self.owner_revision is not None):
            raise ValueError("revisioned evidence owners require an exact owner revision")
        return self


class Comparison(ConfidenceIntervalFields):
    # This is the public result envelope, not the durable comparison format. Persisted
    # comparisons retain the separate schema-v1 Parquet projection in evidence/schemas.py.
    schema_version: Literal[2] = 2
    comparison_id: Identifier
    experiment_id: Identifier | None = None
    baseline_run_set_id: Digest
    candidate_run_set_id: Digest
    metric: Identifier
    unit: Identifier
    metric_source: MetricSource = MetricSource.MEASUREMENT
    metric_contract_id: Digest
    measurement_series_id: Digest | None = None
    protocol_identity_id: Digest | None = None
    value_domain: MetricValueDomain = MetricValueDomain.STRICTLY_POSITIVE
    zero_policy: MetricZeroPolicy = MetricZeroPolicy.REJECT
    polarity: MetricPolarity
    estimand: Identifier
    practical_threshold: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    baseline_value: NumericValue | None = None
    candidate_value: NumericValue | None = None
    absolute_change: NumericValue | None = None
    relative_change: Annotated[float, Field(allow_inf_nan=False)] | None = None
    # ``effect_size`` holds the relative median effect (exp(median(log(
    # candidate/baseline))) - 1), a dimensionless ratio, not a standardized
    # mean difference such as Cohen's d. See ``estimand`` for the exact
    # quantity this value estimates.
    effect_size: Annotated[float, Field(allow_inf_nan=False)] | None = None
    method: Identifier
    random_seed: Annotated[int, Field(ge=0)] | None = None
    independent_unit: Identifier
    baseline_attempted_n: Annotated[int, Field(ge=0)]
    baseline_eligible_n: Annotated[int, Field(ge=0)]
    baseline_failed_n: Annotated[int, Field(ge=0)] = 0
    baseline_excluded_n: Annotated[int, Field(ge=0)] = 0
    baseline_missing_n: Annotated[int, Field(ge=0)] = 0
    baseline_out_of_domain_n: Annotated[int, Field(ge=0)] = 0
    candidate_attempted_n: Annotated[int, Field(ge=0)]
    candidate_eligible_n: Annotated[int, Field(ge=0)]
    candidate_failed_n: Annotated[int, Field(ge=0)] = 0
    candidate_excluded_n: Annotated[int, Field(ge=0)] = 0
    candidate_missing_n: Annotated[int, Field(ge=0)] = 0
    candidate_out_of_domain_n: Annotated[int, Field(ge=0)] = 0
    complete_pair_n: Annotated[int, Field(ge=0)] | None = None
    multiplicity: dict[str, JsonValue] | None = None
    decision: ComparisonDecision
    validity: ComparisonValidity
    mismatches: tuple[str, ...] = ()

    @computed_field(return_type=bool)  # type: ignore[prop-decorator]
    @property
    def paired(self) -> bool:
        return self.complete_pair_n is not None

    @model_validator(mode="after")
    def statistical_state_is_coherent(self) -> Comparison:
        if self.complete_pair_n is not None and self.complete_pair_n > min(
            self.baseline_eligible_n,
            self.candidate_eligible_n,
        ):
            raise ValueError("complete pairs cannot exceed eligible samples")
        if self.validity is ComparisonValidity.VALID:
            if self.mismatches:
                raise ValueError("valid comparisons cannot retain compatibility mismatches")
            if self.confidence_interval is None:
                raise ValueError("valid comparisons require a finite confidence interval")
            if self.paired and self.complete_pair_n != self.baseline_eligible_n:
                raise ValueError("valid paired comparisons require complete baseline coverage")
            if self.paired and self.complete_pair_n != self.candidate_eligible_n:
                raise ValueError("valid paired comparisons require complete candidate coverage")
        elif self.decision not in {
            ComparisonDecision.INCONCLUSIVE,
            ComparisonDecision.DESCRIPTIVE_ONLY,
        }:
            raise ValueError("non-valid comparisons cannot make a confirmatory decision")
        return self


class Finding(ContractModel):
    schema_version: Literal[1] = 1
    finding_id: Identifier
    revision: Annotated[int, Field(ge=1)]
    created_at: datetime = Field(default_factory=utc_now)
    kind: Identifier
    title: ShortText
    claim: ShortText
    evidence_level: EvidenceLevel
    confidence: FindingConfidence
    assessment: FindingAssessment
    lifecycle: FindingLifecycle
    limitations: tuple[str, ...] = ()
    next_experiments: tuple[dict[str, JsonValue], ...] = ()
