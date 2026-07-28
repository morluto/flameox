from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from flameox.models import ContractModel

Identifier = Annotated[str, StringConstraints(min_length=1, max_length=200)]
Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=500)]


def utc_now() -> datetime:
    return datetime.now(UTC)


class EvidenceLevel(StrEnum):
    OBSERVED = "observed"
    DERIVED = "derived"
    INFERRED = "inferred"


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
    CANCELLED = "cancelled"


class GenerationStatus(StrEnum):
    STAGED = "staged"
    PUBLISHED = "published"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    QUARANTINED = "quarantined"


class TrialOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    OOM = "oom"
    INVALID = "invalid"


class OracleStrength(StrEnum):
    EXECUTION_CHECK = "execution_check"
    CONTRACT_CHECK = "contract_check"
    CROSS_TREATMENT_EQUIVALENCE = "cross_treatment_equivalence"


class InvestigationStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    ARCHIVED = "archived"


class FindingAssessment(StrEnum):
    UNASSESSED = "unassessed"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


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


class Integrity(ContractModel):
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    hashed_at: datetime = Field(default_factory=utc_now)


class ArtifactContent(ContractModel):
    schema_version: Literal[1] = 1
    artifact_id: Digest
    byte_length: Annotated[int, Field(ge=0)]
    payload_name: Identifier
    integrity: Integrity


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
            self.kind in {ArtifactKind.CORE_DUMP, ArtifactKind.SOURCE_SNAPSHOT}
            and self.sensitivity is not Sensitivity.SENSITIVE
        ):
            raise ValueError(f"{self.kind.value} artifacts must be sensitive")
        return self


class CommandSpec(ContractModel):
    argv: tuple[Annotated[str, StringConstraints(min_length=1, max_length=32_768)], ...]
    cwd: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    env_overrides: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: Annotated[float, Field(gt=0, le=86_400)] = 300

    @field_validator("argv")
    @classmethod
    def require_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("argv must contain an executable")
        if any("\x00" in item for item in value):
            raise ValueError("argv cannot contain NUL bytes")
        return value


class EnvironmentRecord(ContractModel):
    schema_version: Literal[1] = 1
    environment_id: Digest
    observed_at: datetime = Field(default_factory=utc_now)
    identity_quality: IdentityQuality
    fields: dict[str, JsonValue]
    missing_fields: tuple[str, ...] = ()


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


class WorkloadDefinition(ContractModel):
    schema_version: Literal[1] = 1
    workload_definition_id: Digest
    name: Identifier
    command_template: tuple[str, ...]
    parameter_names: tuple[str, ...] = ()
    validation_spec_id: Digest | None = None
    approved_definition_digest: Digest | None = None


class WorkloadInstance(ContractModel):
    schema_version: Literal[1] = 1
    workload_instance_id: Digest
    workload_definition_id: Digest
    command: CommandSpec
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class CapabilityStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    PERMISSION_REQUIRED = "permission_required"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    UNKNOWN = "unknown"


class CapabilityReport(ContractModel):
    schema_version: Literal[1] = 1
    adapter: Identifier
    status: CapabilityStatus
    executable: str | None = None
    import_location: str | None = None
    version: str | None = None
    supported_modes: tuple[str, ...] = ()
    supported_formats: tuple[str, ...] = ()
    platform: str | None = None
    architecture: str | None = None
    permissions: tuple[str, ...] = ()
    permission_status: str | None = None
    restrictions: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    remediation: tuple[str, ...] = ()
    probe_kind: Literal["passive", "active"] = "passive"
    probed_at: datetime | None = None


class CapturePlan(ContractModel):
    schema_version: Literal[1] = 1
    plan_id: Identifier
    run_id: Identifier
    request_digest: Digest
    workspace_id: Identifier
    workload_name: Identifier
    workload_definition_id: Digest
    approval_digest: Digest
    workload_instance: WorkloadInstance
    adapter: Identifier
    execution_policy: Literal["trusted_local", "approved_agent"]
    adapter_version: str | None = None
    collector_argv: tuple[str, ...]
    collector_environment: dict[str, str] = Field(default_factory=dict)
    expected_artifact_kinds: tuple[ArtifactKind, ...]
    expected_overhead: str
    containment: Literal["active", "degraded", "uncontained", "unavailable"]
    network_contained: bool
    systemd_scope_unit: str | None = None
    permissions: tuple[str, ...] = ()
    bound_identities: dict[str, JsonValue] = Field(default_factory=dict)
    limits: dict[str, JsonValue] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime

    @model_validator(mode="after")
    def expiry_follows_creation(self) -> CapturePlan:
        if self.expires_at <= self.created_at:
            raise ValueError("capture plan expiry must follow creation")
        return self


class ProcessResult(ContractModel):
    exit_code: int | None = None
    terminating_signal: int | None = None
    wall_time_ns: Annotated[int, Field(ge=0)] | None = None
    peak_rss_bytes: Annotated[int, Field(ge=0)] | None = None
    timed_out: bool = False
    cancellation_cause: str | None = None
    cleanup_complete: bool | None = None


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


class RunManifest(ContractModel):
    schema_version: Literal[1] = 1
    revision: Annotated[int, Field(ge=0)] = 0
    run_id: Identifier
    run_type: RunType
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    execution_status: ExecutionStatus
    capture_status: CaptureStatus
    validation_status: ValidationStatus
    workload_definition_id: Digest | None = None
    workload_instance_id: Digest | None = None
    measurement_protocol_id: Digest | None = None
    environment_id: Digest
    source_state_id: Digest | None = None
    collector: str | None = None
    collector_version: str | None = None
    command: CommandSpec | None = None
    process: ProcessResult | None = None
    lease: CaptureLease | None = None
    artifacts: tuple[ArtifactRegistration, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_run_type(self) -> RunManifest:
        if self.run_type is RunType.IMPORT:
            if self.execution_status is not ExecutionStatus.NOT_APPLICABLE:
                raise ValueError("import runs do not execute a workload")
            if self.command is not None:
                raise ValueError("import runs cannot contain a command")
        if (
            self.finished_at is not None
            and self.started_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at cannot precede started_at")
        return self


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
    schema_version: Literal[1] = 1
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
    polarity: Literal["lower_is_better", "higher_is_better", "neutral"]
    estimand: Identifier
    practical_threshold: Annotated[float, Field(ge=0)]
    confidence_level: Annotated[float, Field(gt=0, lt=1)]
    stopping_rule: dict[str, JsonValue]
    random_seed: Annotated[int, Field(ge=0)]
    role: Literal["exploratory", "confirmatory"]
    created_at: datetime = Field(default_factory=utc_now)


class Variant(ContractModel):
    schema_version: Literal[1] = 1
    variant_id: Identifier
    experiment_id: Identifier
    name: Identifier
    source_state_id: Digest
    workload_instance_id: Digest
    environment_requirements: dict[str, JsonValue] = Field(default_factory=dict)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class Trial(ContractModel):
    schema_version: Literal[1] = 1
    trial_id: Identifier
    experiment_id: Identifier
    variant_id: Identifier
    run_id: Identifier
    block_id: Identifier | None = None
    order_in_block: Annotated[int, Field(ge=0)] | None = None
    parameter_name: str | None = None
    parameter_value_int: int | None = None
    parameter_value_float: float | None = None
    attempt: Annotated[int, Field(ge=1)] = 1
    outcome: TrialOutcome
    exclusion_reason: str | None = None
    validation_status: ValidationStatus

    @model_validator(mode="after")
    def only_one_parameter_value(self) -> Trial:
        if self.parameter_value_int is not None and self.parameter_value_float is not None:
            raise ValueError("only one parameter value representation may be set")
        return self


class RunSetMember(ContractModel):
    run_id: Identifier
    trial_id: Identifier | None = None
    included: bool = True
    reason: str | None = None
    order: Annotated[int, Field(ge=0)]


class RunSet(ContractModel):
    schema_version: Literal[1] = 1
    run_set_id: Digest
    corpus_commit_id: Digest
    created_at: datetime = Field(default_factory=utc_now)
    selection: dict[str, JsonValue]
    members: tuple[RunSetMember, ...]
    membership_digest: Digest


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


class EvidenceReference(ContractModel):
    schema_version: Literal[1] = 1
    owner_type: Literal["analysis", "finding", "hypothesis"]
    owner_id: Identifier
    ref_type: Literal[
        "analysis",
        "artifact",
        "comparison",
        "generation",
        "observation",
        "run",
        "run_set",
        "trial",
    ]
    ref_id: Identifier
    relation: Literal["supports", "contradicts", "context", "validates"]


class Comparison(ContractModel):
    schema_version: Literal[1] = 1
    comparison_id: Identifier
    experiment_id: Identifier | None = None
    baseline_run_set_id: Digest
    candidate_run_set_id: Digest
    metric: Identifier
    unit: Identifier
    polarity: Literal["lower_is_better", "higher_is_better", "neutral"]
    estimand: Identifier
    practical_threshold: Annotated[float, Field(ge=0)]
    baseline_value_int: int | None = None
    baseline_value_float: float | None = None
    candidate_value_int: int | None = None
    candidate_value_float: float | None = None
    absolute_change_int: int | None = None
    absolute_change_float: float | None = None
    relative_change: float | None = None
    # ``effect_size`` holds the relative median effect (exp(median(log(
    # candidate/baseline))) - 1), a dimensionless ratio, not a standardized
    # mean difference such as Cohen's d. See ``estimand`` for the exact
    # quantity this value estimates.
    effect_size: float | None = None
    confidence_low: float | None = None
    confidence_high: float | None = None
    confidence_level: float | None = None
    method: Identifier
    random_seed: Annotated[int, Field(ge=0)] | None = None
    independent_unit: Identifier
    paired: bool
    baseline_attempted_n: Annotated[int, Field(ge=0)]
    baseline_eligible_n: Annotated[int, Field(ge=0)]
    baseline_failed_n: Annotated[int, Field(ge=0)] = 0
    baseline_excluded_n: Annotated[int, Field(ge=0)] = 0
    candidate_attempted_n: Annotated[int, Field(ge=0)]
    candidate_eligible_n: Annotated[int, Field(ge=0)]
    candidate_failed_n: Annotated[int, Field(ge=0)] = 0
    candidate_excluded_n: Annotated[int, Field(ge=0)] = 0
    complete_pair_n: Annotated[int, Field(ge=0)] | None = None
    multiplicity: dict[str, JsonValue] | None = None
    decision: ComparisonDecision
    validity: ComparisonValidity
    mismatches: tuple[str, ...] = ()


class Finding(ContractModel):
    schema_version: Literal[1] = 1
    finding_id: Identifier
    revision: Annotated[int, Field(ge=1)]
    created_at: datetime = Field(default_factory=utc_now)
    kind: Identifier
    title: ShortText
    claim: ShortText
    evidence_level: EvidenceLevel
    confidence: Literal["high", "medium", "low", "unknown"]
    assessment: FindingAssessment
    lifecycle: FindingLifecycle
    limitations: tuple[str, ...] = ()
    next_experiments: tuple[dict[str, JsonValue], ...] = ()
