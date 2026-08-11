from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal

from pydantic import ConfigDict, Field, TypeAdapter, computed_field, model_validator

from flameox.domain import (
    CaptureStatus,
    ConfidenceIntervalFields,
    EvidenceLevel,
    ExecutionStatus,
    ProcessCancellationCause,
    ResourcePolicyCancellationCause,
    ValidationStatus,
)
from flameox.domain.scalars import NumericValue
from flameox.evidence_status import (
    EvidenceAvailability,
    EvidenceStatus,
    available_availability,
    parse_evidence_availability,
)
from flameox.models import ContractModel
from flameox.pagination import BoundedCollectionContract


class Hotspot(ContractModel):
    frame_id: str
    function: str | None
    file: str | None
    line: int | None
    metric: str
    self_value: int | None
    inclusive_value: int | None
    unit: str
    sample_count: int | None


class HotspotResult(BoundedCollectionContract):
    page_items_field: ClassVar[str] = "hotspots"

    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    hotspots: tuple[Hotspot, ...]
    total: int
    coverage: dict[str, int]
    limitations: tuple[str, ...]
    evidence: EvidenceAvailability = Field(default_factory=available_availability)

    @model_validator(mode="before")
    @classmethod
    def parse_legacy_evidence_projection(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        parsed = dict(value)
        status_present = "evidence_status" in parsed
        reason_present = "unavailable_reason" in parsed
        legacy_status = parsed.pop("evidence_status", "available")
        legacy_reason = parsed.pop("unavailable_reason", None)
        raw_evidence = parsed.get("evidence")
        if raw_evidence is None:
            parsed["evidence"] = parse_evidence_availability(
                {
                    "status": legacy_status,
                    "reason": (
                        legacy_reason
                        if legacy_status == "unavailable" and legacy_reason is not None
                        else f"legacy_{legacy_status}"
                    ),
                }
            )
            return parsed
        evidence = parse_evidence_availability(raw_evidence)
        if status_present and legacy_status != evidence.status:
            raise ValueError("evidence status fields must agree")
        expected_reason = evidence.reason if evidence.status == "unavailable" else None
        if reason_present and legacy_reason != expected_reason:
            raise ValueError("unavailable reason must agree with structured evidence")
        parsed["evidence"] = evidence
        return parsed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def evidence_status(self) -> EvidenceStatus:
        return self.evidence.status

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unavailable_reason(self) -> str | None:
        return self.evidence.reason if self.evidence.status == "unavailable" else None


class MeasurementSummary(ContractModel):
    name: str
    value: NumericValue | None
    unit: str
    aggregation: str
    scope: str


class MemoryAnalysisResult(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    schema_version: Literal[2] = 2
    corpus_commit_id: str
    input_id: str
    measurements: tuple[MeasurementSummary, ...]
    hotspots: tuple[Hotspot, ...]
    phase_growth: tuple[MemoryPhaseGrowth, ...] = ()
    limitations: tuple[str, ...]
    runtime_resources: tuple[RuntimeResourceObservation, ...] = ()
    runtime_resource_totals: RuntimeResourceTotals | None = None
    writable_root_observations: tuple[WritableRootObservation, ...] = ()
    evidence: EvidenceAvailability = Field(default_factory=available_availability)

    @model_validator(mode="before")
    @classmethod
    def parse_legacy_truncated(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        parsed = dict(value)
        supplied = tuple(
            parsed.pop(name)
            for name in ("truncated", "runtime_resources_truncated")
            if name in parsed
        )
        if not supplied:
            return parsed
        if len(supplied) == 2 and supplied[0] != supplied[1]:
            raise ValueError("truncation fields must agree")
        totals = parsed.get("runtime_resource_totals")
        run_count = (
            totals.get("run_count")
            if isinstance(totals, dict)
            else getattr(totals, "run_count", None)
        )
        resources = parsed.get("runtime_resources", ())
        if (
            isinstance(run_count, int)
            and isinstance(resources, (list, tuple))
            and supplied[0] != (run_count > len(resources))
        ):
            raise ValueError("truncation fields must agree with the runtime-resource total")
        return parsed

    @model_validator(mode="after")
    def runtime_resource_count_is_coherent(self) -> MemoryAnalysisResult:
        if (
            self.runtime_resource_totals is not None
            and self.runtime_resource_totals.run_count < len(self.runtime_resources)
        ):
            raise ValueError("runtime-resource total cannot be smaller than returned resources")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def runtime_resources_truncated(self) -> bool:
        return (
            self.runtime_resource_totals is not None
            and self.runtime_resource_totals.run_count > len(self.runtime_resources)
        )

    @model_validator(mode="before")
    @classmethod
    def parse_legacy_resource_projection(cls, value: object) -> object:
        if not isinstance(value, dict) or not {
            "policy_termination",
            "unavailable_metrics",
        }.intersection(value):
            return value
        parsed = dict(value)
        resources = tuple(
            item
            if isinstance(item, RuntimeResourceObservation)
            else RuntimeResourceObservation.model_validate(item)
            for item in parsed.get("runtime_resources", ())
        )
        expected_policy = next(
            (item.policy_termination for item in resources if item.policy_termination is not None),
            None,
        )
        expected_metrics = tuple(
            sorted({metric for item in resources for metric in item.unavailable_metrics})
        )
        if "policy_termination" in parsed and parsed["policy_termination"] != expected_policy:
            raise ValueError("policy termination must derive from runtime resources")
        if "unavailable_metrics" in parsed:
            raw_metrics = parsed["unavailable_metrics"]
            if not isinstance(raw_metrics, list | tuple) or tuple(raw_metrics) != expected_metrics:
                raise ValueError("unavailable metrics must derive from runtime resources")
        parsed.pop("policy_termination", None)
        parsed.pop("unavailable_metrics", None)
        parsed["runtime_resources"] = resources
        return parsed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def truncated(self) -> bool:
        return self.runtime_resources_truncated

    @computed_field  # type: ignore[prop-decorator]
    @property
    def policy_termination(self) -> ProcessCancellationCause | None:
        return next(
            (
                item.policy_termination
                for item in self.runtime_resources
                if item.policy_termination is not None
            ),
            None,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unavailable_metrics(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {metric for item in self.runtime_resources for metric in item.unavailable_metrics}
            )
        )


class RuntimeResourceObservation(ContractModel):
    run_id: str
    sampling_interval_ms: int
    minimum_free_bytes: int | None
    staging_growth_bytes: int | None
    peak_rss_bytes: int | None
    policy_termination: ResourcePolicyCancellationCause | None
    unavailable_metrics: tuple[str, ...] = ()


class RuntimeResourceTotals(ContractModel):
    run_count: int = 0
    minimum_free_bytes: int | None = None
    total_staging_growth_bytes: int | None = None
    maximum_peak_rss_bytes: int | None = None


class _WritableRootObservation(ContractModel):
    run_id: str
    writable_root_identity: str
    target_path: str


class AvailableWritableRootObservation(_WritableRootObservation):
    available: Literal[True] = True
    growth_bytes: int
    unavailable_reason: Literal[None] = None


class UnavailableWritableRootObservation(_WritableRootObservation):
    available: Literal[False] = False
    growth_bytes: Literal[None] = None
    unavailable_reason: str


type WritableRootObservation = Annotated[
    AvailableWritableRootObservation | UnavailableWritableRootObservation,
    Field(discriminator="available"),
]

_WRITABLE_ROOT_OBSERVATION_ADAPTER: TypeAdapter[WritableRootObservation] = TypeAdapter(
    WritableRootObservation
)


def parse_writable_root_observation(value: object) -> WritableRootObservation:
    return _WRITABLE_ROOT_OBSERVATION_ADAPTER.validate_python(value)


class MemoryPhaseGrowth(ContractModel):
    phase: str
    metric: str
    value: float
    previous_value: float | None
    delta: float | None
    unit: str
    sample_count: int


class ExecutionObservation(ContractModel):
    observation_id: str
    kind: str
    name: str
    value_json: str
    file: str | None
    line_from: int | None
    line_to: int | None
    context: str | None
    evidence_level: EvidenceLevel


class ExecutionAnalysisResult(BoundedCollectionContract):
    page_items_field: ClassVar[str] = "observations"

    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    observations: tuple[ExecutionObservation, ...]
    comparison_input_id: str | None = None
    added: tuple[ExecutionObservation, ...] = ()
    removed: tuple[ExecutionObservation, ...] = ()
    changed: tuple[ExecutionObservationChange, ...] = ()
    total: int
    limitations: tuple[str, ...]
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class ExecutionObservationChange(ContractModel):
    kind: str
    name: str
    file: str | None
    line_from: int | None
    line_to: int | None
    context: str | None
    baseline_value_json: str
    candidate_value_json: str


class OperatorSummary(ContractModel):
    frame_id: str
    operator: str
    category: str | None
    self_cpu_ns: int | None
    total_cpu_ns: int | None
    device_ns: int | None
    inclusive_ns: int
    event_count: int
    input_shapes: tuple[str, ...] = ()
    allocation_bytes: int | None = None
    synchronization: bool = False
    warmup: bool | None = None


class PyTorchAnalysisResult(BoundedCollectionContract):
    page_items_field: ClassVar[str] = "operators"

    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    operators: tuple[OperatorSummary, ...]
    total: int
    coverage: dict[str, bool]
    repeated_small_operations: tuple[OperatorSummary, ...] = ()
    synchronization_time_ns: int = 0
    compilation_time_ns: int = 0
    warmup_time_ns: int = 0
    allocation_bytes: int | None = None
    limitations: tuple[str, ...]
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class KernelNameCount(ContractModel):
    name: str
    count: int


class AcceleratorStreamSummary(ContractModel):
    identity: str
    device: str | None = None
    context: str | None = None
    stream: str | None = None
    track_id: str | None = None
    kernel_count: int
    kernel_duration_ns: int
    idle_gap_count: int
    idle_gap_total_ns: int
    idle_gap_max_ns: int


def _advertise_stream_truncation(schema: dict[str, Any]) -> None:
    properties = schema.setdefault("properties", {})
    assert isinstance(properties, dict)
    properties["streams_truncated"] = {"type": "boolean", "readOnly": True}
    required = schema.setdefault("required", [])
    assert isinstance(required, list)
    if "streams_truncated" not in required:
        required.append("streams_truncated")


class AcceleratorLaunchRegion(ContractModel):
    model_config = ConfigDict(json_schema_extra=_advertise_stream_truncation)

    region: str
    region_start_ns: int
    region_end_ns: int
    region_duration_ns: int
    selection_rule: str
    direct_launch_count: int
    direct_launch_duration_ns: int
    graph_launch_count: int
    graph_launch_duration_ns: int
    kernel_count: int
    kernel_duration_ns: int
    kernel_names: tuple[KernelNameCount, ...]
    kernel_names_truncated: bool
    correlated_kernel_count: int
    runtime_launch_gap_count: int
    runtime_launch_gap_total_ns: int
    runtime_launch_gap_max_ns: int
    idle_gap_count: int
    idle_gap_total_ns: int
    idle_gap_max_ns: int
    stream_count: int
    streams: tuple[AcceleratorStreamSummary, ...]

    @model_validator(mode="before")
    @classmethod
    def parse_stream_truncation_projection(cls, value: object) -> object:
        if not isinstance(value, dict) or "streams_truncated" not in value:
            return value
        parsed = dict(value)
        supplied = parsed.pop("streams_truncated")
        stream_count = parsed.get("stream_count")
        streams = parsed.get("streams")
        if (
            isinstance(stream_count, int)
            and isinstance(streams, (list, tuple))
            and supplied != (stream_count > len(streams))
        ):
            raise ValueError("stream truncation must agree with the stream total")
        return parsed

    @model_validator(mode="after")
    def stream_count_is_coherent(self) -> AcceleratorLaunchRegion:
        if self.stream_count < len(self.streams):
            raise ValueError("stream total cannot be smaller than returned streams")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def streams_truncated(self) -> bool:
        return self.stream_count > len(self.streams)


class AcceleratorLaunchComparison(ContractModel):
    region: str
    direct_launch_count_delta: int
    graph_launch_count_delta: int
    kernel_count_delta: int
    kernel_duration_delta_ns: int
    runtime_launch_gap_total_delta_ns: int
    idle_gap_total_delta_ns: int


class AcceleratorLaunchAnalysisResult(BoundedCollectionContract):
    page_items_field: ClassVar[str] = "regions"

    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    comparison_input_id: str | None = None
    phase_filter: str | None = None
    regions: tuple[AcceleratorLaunchRegion, ...]
    comparison_regions: tuple[AcceleratorLaunchRegion, ...] = ()
    comparisons: tuple[AcceleratorLaunchComparison, ...] = ()
    total: int
    coverage: dict[str, bool]
    comparison_coverage: dict[str, bool] | None = None
    limitations: tuple[str, ...]
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class FailureCluster(ContractModel):
    collector: str | None
    execution_status: ExecutionStatus
    capture_status: CaptureStatus
    validation_status: ValidationStatus
    exit_code: int | None
    workload_definition_id: str | None
    environment_id: str
    source_state_id: str | None
    run_count: int
    first_seen: str
    last_seen: str
    representative_artifact_ids: tuple[str, ...] = ()


class FailureChangePoint(ContractModel):
    observed_date: str
    run_count: int
    previous_run_count: int | None


class FailurePopulationStatus(StrEnum):
    OBSERVED = "observed"
    EMPTY = "empty"
    FILTERED_EMPTY = "filtered_empty"


class FailureAnalysisResult(BoundedCollectionContract):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    page_items_field: ClassVar[str] = "failures"
    total_items_field: ClassVar[str] = "total_clusters"

    schema_version: int = 1
    corpus_commit_id: str
    cohort_id: str
    filters_applied: tuple[str, ...]
    eligible_runs: int = 0
    failed_runs: int = 0
    population_status: FailurePopulationStatus = FailurePopulationStatus.EMPTY
    failures: tuple[FailureCluster, ...]
    total_clusters: int
    change_points: tuple[FailureChangePoint, ...]
    coverage: dict[str, float]
    competing_hypotheses: tuple[str, ...]
    limitations: tuple[str, ...] = (
        "Clusters use lifecycle status and exit code; native crash signatures "
        "require a specialized extractor.",
    )
    evidence: EvidenceAvailability = Field(default_factory=available_availability)

    @model_validator(mode="before")
    @classmethod
    def parse_legacy_empty_reason(cls, value: object) -> object:
        if not isinstance(value, dict) or "empty_reason" not in value:
            return value
        status = FailurePopulationStatus(value.get("population_status", "empty"))
        failed_runs = value.get("failed_runs", 0)
        expected = (
            "no_runs"
            if status is FailurePopulationStatus.EMPTY
            else (
                "no_matching_runs"
                if status is FailurePopulationStatus.FILTERED_EMPTY
                else "no_failures"
                if failed_runs == 0
                else None
            )
        )
        if value["empty_reason"] != expected:
            raise ValueError("empty reason must derive from the failure population")
        parsed = dict(value)
        del parsed["empty_reason"]
        return parsed

    @model_validator(mode="after")
    def population_is_coherent(self) -> FailureAnalysisResult:
        if self.failed_runs > self.eligible_runs:
            raise ValueError("failed runs cannot exceed eligible runs")
        if (self.population_status is FailurePopulationStatus.OBSERVED) != (self.eligible_runs > 0):
            raise ValueError("observed population status must match eligible runs")
        if self.failed_runs == 0 and (self.failures or self.total_clusters):
            raise ValueError("an empty failure population cannot contain clusters")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def empty_reason(self) -> Literal["no_runs", "no_matching_runs", "no_failures"] | None:
        if self.population_status is FailurePopulationStatus.EMPTY:
            return "no_runs"
        if self.population_status is FailurePopulationStatus.FILTERED_EMPTY:
            return "no_matching_runs"
        return "no_failures" if self.failed_runs == 0 else None


class ScalingPoint(ConfidenceIntervalFields):
    variant: str
    block_id: str | None
    input_value: float | None
    value: float
    dispersion: float
    unit: str
    sample_count: int
    raw_sample_count: int = 0
    environment_count: int = Field(ge=1)


class ScalingTrialSummary(ContractModel):
    trial_id: str
    variant: str
    block_id: str | None
    input_value: float | None
    median: float
    dispersion: float
    unit: str
    raw_sample_count: int
    environment_id: str


class ScalingCorrelatedHotspot(ContractModel):
    variant: str
    frame_id: str
    function: str | None
    file: str | None
    line: int | None
    metric: str
    unit: str
    spearman_rho: float
    p_value: float
    adjusted_p_value: float
    multiplicity_method: str
    # ``None`` while no hypotheses have been tested yet; the multiplicity
    # method string must not be combined with a zero count.
    tested_hypothesis_count: int | None = None
    independent_trial_count: int
    supported_min: float
    supported_max: float


class ScalingFit(ContractModel):
    model: str
    variant: str
    coefficients: tuple[float, ...]
    coefficient_standard_errors: tuple[float, ...]
    coefficient_confidence_intervals: tuple[tuple[float, float], ...]
    residual_rms: float
    r_squared: float | None
    aicc: float | None
    condition_number: float
    durbin_watson: float | None
    observation_count: int
    supported_min: float
    supported_max: float


def _advertise_environment_stability(schema: dict[str, Any]) -> None:
    properties = schema.setdefault("properties", {})
    assert isinstance(properties, dict)
    properties["environment_stable"] = {"type": "boolean", "readOnly": True}
    required = schema.setdefault("required", [])
    assert isinstance(required, list)
    if "environment_stable" not in required:
        required.append("environment_stable")


class ScalingAnalysisResult(ContractModel):
    model_config = ConfigDict(json_schema_extra=_advertise_environment_stability)

    schema_version: int = 1
    corpus_commit_id: str
    experiment_id: str
    metric: str
    points: tuple[ScalingPoint, ...]
    trials: tuple[ScalingTrialSummary, ...]
    attempted_trials: int
    succeeded_trials: int
    failed_trials: int
    complete_blocks: int
    fits: tuple[ScalingFit, ...]
    correlated_hotspots: tuple[ScalingCorrelatedHotspot, ...]
    conclusion: str
    warnings: tuple[str, ...]
    limitations: tuple[str, ...] = (
        "Points are per-trial medians; statistical decisions belong to frozen run-set comparisons.",
    )
    evidence: EvidenceAvailability = Field(default_factory=available_availability)

    @model_validator(mode="before")
    @classmethod
    def parse_environment_stability_projection(cls, value: object) -> object:
        if not isinstance(value, dict) or "environment_stable" not in value:
            return value
        parsed = dict(value)
        supplied = parsed.pop("environment_stable")
        points = parsed.get("points")
        if isinstance(points, (list, tuple)):
            counts = tuple(
                point.get("environment_count")
                if isinstance(point, dict)
                else getattr(point, "environment_count", None)
                for point in points
            )
            if all(isinstance(count, int) for count in counts) and supplied != all(
                count == 1 for count in counts
            ):
                raise ValueError("environment stability must derive from scaling points")
        return parsed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def environment_stable(self) -> bool:
        return all(point.environment_count == 1 for point in self.points)
