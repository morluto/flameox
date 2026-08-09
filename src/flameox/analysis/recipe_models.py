from __future__ import annotations

from typing import Literal

from pydantic import Field

from flameox.domain import NumericValue
from flameox.evidence_status import EvidenceAvailability, available_availability
from flameox.models import ContractModel


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


class HotspotResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    hotspots: tuple[Hotspot, ...]
    total: int
    returned: int
    truncated: bool
    coverage: dict[str, int]
    limitations: tuple[str, ...]
    evidence_status: Literal["available", "empty", "unavailable", "partial", "unknown"] = (
        "available"
    )
    unavailable_reason: str | None = None
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class MeasurementSummary(ContractModel):
    name: str
    value: NumericValue | None
    unit: str
    aggregation: str
    scope: str


class MemoryAnalysisResult(ContractModel):
    schema_version: Literal[2] = 2
    corpus_commit_id: str
    input_id: str
    measurements: tuple[MeasurementSummary, ...]
    hotspots: tuple[Hotspot, ...]
    phase_growth: tuple[MemoryPhaseGrowth, ...] = ()
    limitations: tuple[str, ...]
    runtime_resources: tuple[RuntimeResourceObservation, ...] = ()
    runtime_resource_totals: RuntimeResourceTotals | None = None
    runtime_resources_truncated: bool = False
    truncated: bool = False
    writable_root_observations: tuple[WritableRootObservation, ...] = ()
    policy_termination: str | None = None
    unavailable_metrics: tuple[str, ...] = ()
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class RuntimeResourceObservation(ContractModel):
    run_id: str
    sampling_interval_ms: int
    minimum_free_bytes: int | None
    staging_growth_bytes: int | None
    peak_rss_bytes: int | None
    policy_termination: str | None
    unavailable_metrics: tuple[str, ...] = ()


class RuntimeResourceTotals(ContractModel):
    run_count: int = 0
    minimum_free_bytes: int | None = None
    total_staging_growth_bytes: int | None = None
    maximum_peak_rss_bytes: int | None = None


class WritableRootObservation(ContractModel):
    run_id: str
    writable_root_identity: str
    target_path: str
    growth_bytes: int | None
    available: bool
    unavailable_reason: str | None = None


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
    evidence_level: str


class ExecutionAnalysisResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    observations: tuple[ExecutionObservation, ...]
    comparison_input_id: str | None = None
    added: tuple[ExecutionObservation, ...] = ()
    removed: tuple[ExecutionObservation, ...] = ()
    changed: tuple[ExecutionObservationChange, ...] = ()
    total: int
    returned: int
    truncated: bool
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


class PyTorchAnalysisResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    operators: tuple[OperatorSummary, ...]
    total: int
    returned: int
    truncated: bool
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


class AcceleratorLaunchRegion(ContractModel):
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
    streams_truncated: bool


class AcceleratorLaunchComparison(ContractModel):
    region: str
    direct_launch_count_delta: int
    graph_launch_count_delta: int
    kernel_count_delta: int
    kernel_duration_delta_ns: int
    runtime_launch_gap_total_delta_ns: int
    idle_gap_total_delta_ns: int


class AcceleratorLaunchAnalysisResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    comparison_input_id: str | None = None
    phase_filter: str | None = None
    regions: tuple[AcceleratorLaunchRegion, ...]
    comparison_regions: tuple[AcceleratorLaunchRegion, ...] = ()
    comparisons: tuple[AcceleratorLaunchComparison, ...] = ()
    total: int
    returned: int
    truncated: bool
    coverage: dict[str, bool]
    comparison_coverage: dict[str, bool] | None = None
    limitations: tuple[str, ...]
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class FailureCluster(ContractModel):
    collector: str | None
    execution_status: str
    capture_status: str
    validation_status: str
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


class FailureAnalysisResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    cohort_id: str
    filters_applied: tuple[str, ...]
    eligible_runs: int = 0
    failed_runs: int = 0
    population_status: Literal["observed", "empty", "filtered_empty"] = "empty"
    empty_reason: Literal["no_runs", "no_matching_runs", "no_failures"] | None = "no_runs"
    failures: tuple[FailureCluster, ...]
    total_clusters: int
    returned: int
    truncated: bool
    change_points: tuple[FailureChangePoint, ...]
    coverage: dict[str, float]
    competing_hypotheses: tuple[str, ...]
    limitations: tuple[str, ...] = (
        "Clusters use lifecycle status and exit code; native crash signatures "
        "require a specialized extractor.",
    )
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class ScalingPoint(ContractModel):
    variant: str
    block_id: str | None
    input_value: float | None
    value: float
    dispersion: float
    confidence_low: float | None = None
    confidence_high: float | None = None
    confidence_level: float | None = None
    unit: str
    sample_count: int
    raw_sample_count: int = 0
    environment_count: int


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


class ScalingAnalysisResult(ContractModel):
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
    environment_stable: bool
    warnings: tuple[str, ...]
    limitations: tuple[str, ...] = (
        "Points are per-trial medians; statistical decisions belong to frozen run-set comparisons.",
    )
    evidence: EvidenceAvailability = Field(default_factory=available_availability)
