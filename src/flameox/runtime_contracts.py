from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from flameox.environment_policy import blocked_environment_override

MAX_INPUTS = 32
MAX_ROWS = 1_000
MAX_RESULT_BYTES = 256 * 1024
LOWERCASE_SHA256_PATTERN = r"^[0-9a-f]{64}$"
SEMANTIC_ORACLE_STDOUT_ENV = "FLAMEOX_CAPTURE_STDOUT"
SEMANTIC_ORACLE_STDERR_ENV = "FLAMEOX_CAPTURE_STDERR"


class RuntimeFailure(RuntimeError):
    def __init__(
        self, code: str, message: str, *, details: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code, self.message, self.details = code, message, dict(details or {})


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathSource(StrictModel):
    kind: Literal["path"] = "path"
    path: str = Field(
        description="Path to the native artifact or artifact directory.",
        min_length=1,
        max_length=4096,
    )
    format: str | None = Field(
        default=None,
        description="Explicit artifact format; omit to detect it from the artifact.",
        min_length=1,
        max_length=80,
    )
    producer: str | None = Field(
        default=None,
        description="Optional producer identity used to interpret otherwise ambiguous formats.",
        min_length=1,
        max_length=80,
    )
    expected_sha256: str | None = Field(
        default=None,
        description="Expected lowercase SHA-256 digest; analysis fails if the artifact differs.",
        pattern=LOWERCASE_SHA256_PATTERN,
    )


class EvidenceSource(StrictModel):
    kind: Literal["evidence"]
    evidence_id: str = Field(
        description="Identifier of previously preserved immutable evidence.",
        pattern=LOWERCASE_SHA256_PATTERN,
    )
    artifact_role: str | None = Field(
        default=None,
        description=(
            "Select one artifact role when preserved evidence contains multiple artifacts."
        ),
        min_length=1,
        max_length=80,
    )


Source = Annotated[PathSource | EvidenceSource, Field(discriminator="kind")]


class EmptyArguments(StrictModel):
    pass


class PreviewArguments(StrictModel):
    offset: int = Field(
        default=0, description="Zero-based byte offset at which to begin the preview.", ge=0
    )


class SummaryArguments(StrictModel):
    metric: str | None = Field(
        default=None,
        description="Provider-supported metric to summarize; omit for its default metric.",
        max_length=120,
    )
    group_by: str | None = Field(
        default=None,
        description="Provider-supported dimension used to group summary rows.",
        max_length=120,
    )


class StaticArguments(StrictModel):
    include_paths: list[str] = Field(
        default_factory=list,
        description="Path patterns eligible for static analysis.",
        max_length=128,
    )
    exclude_paths: list[str] = Field(
        default_factory=list,
        description="Path patterns excluded after includes are applied.",
        max_length=128,
    )

    @field_validator("include_paths", "exclude_paths")
    @classmethod
    def bounded_patterns(cls, value: list[str]) -> list[str]:
        if any(not pattern or len(pattern) > 256 or "\x00" in pattern for pattern in value):
            raise ValueError("path patterns must be non-empty, bounded, and contain no NUL")
        return value


class WindowArguments(StrictModel):
    start_ns: int = Field(description="Inclusive trace-window start in nanoseconds.", ge=0)
    end_ns: int = Field(description="Exclusive trace-window end in nanoseconds.", gt=0)

    @model_validator(mode="after")
    def ordered(self) -> WindowArguments:
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be greater than start_ns")
        return self


class CompareArguments(StrictModel):
    metric: str | None = Field(
        default=None,
        description="Metric to compare; omit for the capability's default metric.",
        max_length=120,
    )
    baseline_index: int = Field(
        default=0,
        description="Zero-based source index used as the comparison baseline.",
        ge=0,
        le=31,
    )


ArgumentModel = type[
    EmptyArguments
    | PreviewArguments
    | SummaryArguments
    | StaticArguments
    | WindowArguments
    | CompareArguments
]


class RequestLimits(StrictModel):
    max_rows: int = Field(
        default=100, description="Maximum evidence rows returned on this page.", ge=1, le=MAX_ROWS
    )
    max_result_bytes: int = Field(
        default=MAX_RESULT_BYTES,
        description="Maximum serialized structured result size in bytes.",
        ge=1024,
        le=MAX_RESULT_BYTES,
    )
    max_input_bytes: int = Field(
        default=1024**3,
        description="Maximum total bytes read from source artifacts.",
        ge=1024,
        le=1024**3,
    )
    max_input_files: int = Field(
        default=4096, description="Maximum files traversed across source artifacts.", ge=1, le=4096
    )
    timeout_seconds: float = Field(
        default=300,
        description="Maximum time for each bounded capture, conversion, or worker in seconds.",
        gt=0,
        le=3600,
    )
    max_output_bytes: int = Field(
        default=16 * 1024 * 1024,
        description="Maximum output bytes from each bounded process or generated artifact.",
        ge=1024,
        le=64 * 1024 * 1024,
    )
    max_memory_bytes: int = Field(
        default=1024**3,
        description="Maximum resident memory for each bounded capture or worker process tree.",
        ge=16 * 1024 * 1024,
        le=16 * 1024**3,
    )
    max_provenance_bytes: int = Field(
        default=16 * 1024 * 1024,
        description="Maximum captured execution metadata retained for preservation, in bytes.",
        ge=4 * 1024,
        le=64 * 1024 * 1024,
    )

    def lowered_against(self, startup: RequestLimits) -> RequestLimits:
        effective = startup.model_dump()
        for name in self.model_fields_set:
            requested = getattr(self, name)
            if requested > getattr(startup, name):
                raise RuntimeFailure("LIMIT_EXCEEDED", f"{name} cannot raise the server limit")
            effective[name] = requested
        return RequestLimits.model_validate(effective)


class DirectTarget(StrictModel):
    argv: list[str] = Field(
        description="Executable and arguments passed directly without a shell.",
        min_length=1,
        max_length=256,
    )
    cwd: str = Field(
        description="Existing absolute directory in which to execute the target.",
        min_length=1,
        max_length=4_096,
    )
    environment: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Variables added to Flameox's minimal allowlisted environment (normally PATH); "
            "sensitive loader overrides are rejected."
        ),
        max_length=32,
    )

    @field_validator("argv")
    @classmethod
    def valid_argv(cls, value: list[str]) -> list[str]:
        return _valid_argv(value)

    @field_validator("cwd")
    @classmethod
    def absolute_cwd(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("cwd must be an absolute path")
        return value

    @field_validator("environment")
    @classmethod
    def valid_environment(cls, value: dict[str, str]) -> dict[str, str]:
        return _valid_environment(value)


class CaptureTarget(DirectTarget):
    provider_id: str = Field(min_length=1, max_length=80)
    capture_arguments: dict[str, Any] = Field(default_factory=dict)
    analysis_arguments: dict[str, Any] = Field(default_factory=dict)


class PyperfCaptureArguments(StrictModel):
    processes: int = Field(
        default=1, description="Number of worker processes used by pyperf.", ge=1, le=32
    )
    values: int = Field(
        default=3, description="Measured values collected per pyperf process.", ge=1, le=100
    )
    warmups: int = Field(
        default=1, description="Warmup values collected per pyperf process.", ge=0, le=100
    )
    loops: int = Field(
        default=0,
        description="Loops per value; zero lets pyperf calibrate automatically.",
        ge=0,
        le=1_000_000_000,
    )
    min_time: float = Field(
        default=0.1, description="Minimum duration of each pyperf value in seconds.", gt=0, le=60
    )
    name: str = Field(
        default="command",
        description="Benchmark name recorded in the pyperf suite.",
        min_length=1,
        max_length=120,
    )


class PySpyCaptureArguments(StrictModel):
    rate: int = Field(
        default=100, description="Sampling frequency in samples per second.", ge=1, le=1_000
    )
    gil: bool = Field(
        default=False, description="Include only threads currently holding the Python GIL."
    )
    native: bool = Field(
        default=False, description="Include native extension frames when supported."
    )


class PerfCaptureArguments(StrictModel):
    frequency: int = Field(
        default=99, description="Sampling frequency in samples per second.", ge=1, le=1_000
    )
    call_graph: Literal["dwarf", "fp"] = Field(
        default="dwarf", description="Call-graph unwinding mode: DWARF metadata or frame pointers."
    )


class MemrayCaptureArguments(StrictModel):
    native: bool = Field(
        default=False, description="Track native allocations and include native stack frames."
    )


class ComputeSanitizerCaptureArguments(StrictModel):
    tool: Literal["memcheck", "racecheck", "initcheck", "synccheck"] = Field(
        default="memcheck", description="Compute Sanitizer analysis tool to run."
    )


def _default_nsys_traces() -> list[Literal["cuda", "nvtx", "osrt", "cublas", "cudnn"]]:
    return ["cuda", "nvtx", "osrt"]


class NsightSystemsCaptureArguments(StrictModel):
    trace: list[Literal["cuda", "nvtx", "osrt", "cublas", "cudnn"]] = Field(
        default_factory=_default_nsys_traces,
        description="Nsight Systems trace domains to collect.",
        min_length=1,
        max_length=5,
    )


class NsightComputeCaptureArguments(StrictModel):
    replay_mode: Literal["kernel", "application", "range"] = Field(
        default="kernel", description="Nsight Compute replay mode."
    )
    launch_skip: int = Field(
        default=0,
        description="Matching kernel launches to skip before profiling.",
        ge=0,
        le=1_000_000,
    )
    launch_count: int = Field(
        default=1, description="Matching kernel launches to profile.", ge=1, le=1_000_000
    )
    section: list[str] = Field(
        default_factory=list,
        description="Nsight Compute section identifiers; empty uses its default section set.",
        max_length=32,
    )

    @field_validator("section")
    @classmethod
    def bounded_sections(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 200 or "\x00" in item for item in value):
            raise ValueError("Nsight Compute sections must be non-empty and bounded")
        return value


class RocprofCaptureArguments(StrictModel):
    hip_trace: bool = Field(default=False, description="Collect HIP API events.")
    kernel_trace: bool = Field(default=True, description="Collect kernel dispatch events.")
    memory_copy_trace: bool = Field(default=False, description="Collect memory-copy events.")
    memory_allocation_trace: bool = Field(
        default=False, description="Collect memory-allocation events."
    )
    scratch_memory_trace: bool = Field(default=False, description="Collect scratch-memory events.")
    marker_trace: bool = Field(default=False, description="Collect marker events.")

    @model_validator(mode="after")
    def at_least_one_domain(self) -> RocprofCaptureArguments:
        if not any(
            (
                self.hip_trace,
                self.kernel_trace,
                self.memory_copy_trace,
                self.memory_allocation_trace,
                self.scratch_memory_trace,
                self.marker_trace,
            )
        ):
            raise ValueError("at least one ROCprof trace domain must be enabled")
        return self


class XctraceCaptureArguments(StrictModel):
    template: Literal["Metal System Trace", "Time Profiler"] = Field(
        default="Metal System Trace", description="Instruments recording template."
    )


def _default_torch_activities() -> list[Literal["cpu", "cuda", "cuda_if_available"]]:
    return ["cpu"]


class TorchProfilerCaptureArguments(StrictModel):
    activities: list[Literal["cpu", "cuda", "cuda_if_available"]] = Field(
        default_factory=_default_torch_activities,
        description="Profiler activities to record; cuda_if_available falls back without error.",
        min_length=1,
        max_length=2,
    )
    wait: int = Field(
        default=0, description="Profiler schedule steps to wait per cycle.", ge=0, le=10_000
    )
    warmup: int = Field(
        default=1, description="Profiler schedule warmup steps per cycle.", ge=0, le=10_000
    )
    active: int = Field(
        default=1, description="Profiler schedule recording steps per cycle.", ge=1, le=10_000
    )
    skip_first: int = Field(
        default=0, description="Initial target steps skipped before the schedule.", ge=0, le=10_000
    )
    record_shapes: bool = Field(default=False, description="Record operator input shapes.")
    profile_memory: bool = Field(
        default=False, description="Record tensor allocation and deallocation events."
    )
    with_stack: bool = Field(default=False, description="Record source stacks when supported.")
    with_flops: bool = Field(default=False, description="Estimate supported operator FLOPs.")
    with_modules: bool = Field(default=False, description="Record module hierarchy when supported.")


class CoverageCaptureArguments(StrictModel):
    branch: bool = Field(default=False, description="Measure branch as well as statement coverage.")
    source: list[str] = Field(
        default_factory=list, description="coverage.py source selectors to measure.", max_length=64
    )
    include: list[str] = Field(
        default_factory=list, description="coverage.py file patterns to include.", max_length=64
    )
    omit: list[str] = Field(
        default_factory=list, description="coverage.py file patterns to omit.", max_length=64
    )

    @field_validator("source", "include", "omit")
    @classmethod
    def bounded_values(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 256 or "\x00" in item for item in value):
            raise ValueError("coverage selectors must be non-empty and bounded")
        return value


class TorchBenchmarkRuntimeArguments(StrictModel):
    min_run_time_seconds: float = Field(
        default=0.2,
        description="Minimum adaptive timing duration per sample in seconds.",
        gt=0,
        le=60,
    )
    max_samples: int = Field(
        default=100, description="Maximum raw timing samples to retain.", ge=1, le=1_000
    )
    num_threads: int = Field(
        default=1, description="PyTorch intra-op threads used by the benchmark.", ge=1, le=256
    )
    cuda_event_timing: bool = Field(
        default=False, description="Use synchronized CUDA events instead of host wall-clock timing."
    )


class BenchmarkSamplesCaptureArguments(StrictModel):
    torch_benchmark: TorchBenchmarkRuntimeArguments | None = Field(
        default=None, description="Run the target through the torch.utils.benchmark timing harness."
    )


type CaptureArguments = (
    EmptyArguments
    | PyperfCaptureArguments
    | PySpyCaptureArguments
    | PerfCaptureArguments
    | MemrayCaptureArguments
    | ComputeSanitizerCaptureArguments
    | NsightSystemsCaptureArguments
    | NsightComputeCaptureArguments
    | RocprofCaptureArguments
    | XctraceCaptureArguments
    | TorchProfilerCaptureArguments
    | CoverageCaptureArguments
    | BenchmarkSamplesCaptureArguments
)


@dataclass(frozen=True, slots=True)
class CaptureArtifactContract:
    role: str
    format: str


@dataclass(frozen=True, slots=True)
class CaptureProviderContract:
    id: str
    argument_model: type[StrictModel]
    artifacts: tuple[CaptureArtifactContract, ...]
    artifact_description: str


def _capture_provider(
    provider_id: str,
    argument_model: type[StrictModel],
    artifacts: tuple[tuple[str, str], ...],
    description: str,
) -> CaptureProviderContract:
    return CaptureProviderContract(
        provider_id,
        argument_model,
        tuple(CaptureArtifactContract(role, format_name) for role, format_name in artifacts),
        description,
    )


CAPTURE_PROVIDER_CONTRACTS = {
    item.id: item
    for item in (
        _capture_provider(
            "direct",
            EmptyArguments,
            (("stdout", "text"), ("stderr", "text")),
            "stdout and stderr text",
        ),
        _capture_provider(
            "pyperf", PyperfCaptureArguments, (("benchmark", "pyperf"),), "native pyperf JSON suite"
        ),
        _capture_provider(
            "py-spy",
            PySpyCaptureArguments,
            (("cpu-profile", "py-spy"),),
            "py-spy Speedscope profile",
        ),
        _capture_provider(
            "perf",
            PerfCaptureArguments,
            (("cpu-profile", "perf-data"),),
            "native perf.data profile",
        ),
        _capture_provider(
            "node-cpu-profile",
            EmptyArguments,
            (("cpu-profile", "cpuprofile"),),
            "native V8 CPU profile",
        ),
        _capture_provider(
            "memray",
            MemrayCaptureArguments,
            (("memory", "memray"),),
            "native Memray allocation file",
        ),
        _capture_provider(
            "torch-profiler",
            TorchProfilerCaptureArguments,
            (("trace", "pytorch"),),
            "native PyTorch Chrome trace",
        ),
        _capture_provider(
            "benchmark-samples",
            BenchmarkSamplesCaptureArguments,
            (("benchmark", "samples"),),
            "native Flameox benchmark samples",
        ),
        _capture_provider(
            "nvbench",
            EmptyArguments,
            (("benchmark", "nvbench"),),
            "native NVBench JSON-bin directory",
        ),
        _capture_provider(
            "compute-sanitizer",
            ComputeSanitizerCaptureArguments,
            (("sanitizer", "compute-sanitizer"),),
            "native Compute Sanitizer log",
        ),
        _capture_provider(
            "nsight-systems",
            NsightSystemsCaptureArguments,
            (("trace", "nsys-rep"),),
            "native Nsight Systems report",
        ),
        _capture_provider(
            "nsight-compute",
            NsightComputeCaptureArguments,
            (("report", "nsight-compute"),),
            "native Nsight Compute report",
        ),
        _capture_provider(
            "rocprofv3",
            RocprofCaptureArguments,
            (("trace", "rocprof-pftrace"),),
            "native ROCprof PFTrace",
        ),
        _capture_provider(
            "xctrace",
            XctraceCaptureArguments,
            (("trace", "xctrace"),),
            "native Apple .trace bundle",
        ),
        _capture_provider(
            "coverage",
            CoverageCaptureArguments,
            (("coverage", "coverage"),),
            "native coverage.py data file",
        ),
        _capture_provider(
            "observations",
            EmptyArguments,
            (("observations", "observations"),),
            "native Flameox observation stream",
        ),
        _capture_provider(
            "pytest",
            EmptyArguments,
            (("reliability", "pytest"),),
            "bounded native pytest event stream",
        ),
    )
}


class ExperimentCase(StrictModel):
    name: str = Field(
        description="Stable case label used in comparison evidence.", min_length=1, max_length=80
    )
    argv: list[str] | None = Field(
        default=None,
        description="Case-specific executable and arguments; omit to inherit target.argv.",
        min_length=1,
        max_length=256,
    )
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Case variables merged over target.environment.",
        max_length=32,
    )

    @field_validator("argv")
    @classmethod
    def valid_argv(cls, value: list[str] | None) -> list[str] | None:
        return _valid_argv(value) if value is not None else None

    @field_validator("environment")
    @classmethod
    def valid_environment(cls, value: dict[str, str]) -> dict[str, str]:
        return _valid_environment(value)


class ExperimentDesign(StrictModel):
    cases: list[ExperimentCase] = Field(
        description="Compared cases in declaration order; the first case is the baseline.",
        min_length=2,
        max_length=16,
    )
    blocks: int = Field(
        description="Paired repetitions; case order is randomized independently within each block.",
        ge=1,
        le=100,
    )
    seed: int = Field(
        description="Seed for reproducible case ordering and confidence-interval resampling."
    )
    metric: Literal["wall_time_ns"] = Field(
        description="Per-execution metric used for paired differences."
    )
    estimand: Literal["median_difference", "mean_difference"] = Field(
        description="Summary statistic of candidate-minus-baseline paired differences."
    )
    practical_threshold: float = Field(
        description=(
            "Absolute difference considered practically equivalent, in metric units (nanoseconds)."
        ),
        ge=0,
    )
    semantic_oracle: list[str] | None = Field(
        default=None,
        description=(
            "Separate executable argv run after each successful capture; element zero is "
            f"resolved without a shell. {SEMANTIC_ORACLE_STDOUT_ENV} and "
            f"{SEMANTIC_ORACLE_STDERR_ENV} name the captured files. A nonzero exit excludes "
            "that case-block pair."
        ),
        min_length=1,
        max_length=256,
    )

    @model_validator(mode="after")
    def case_names_are_unique(self) -> ExperimentDesign:
        names = [case.name for case in self.cases]
        if len(names) != len(set(names)):
            raise ValueError("experiment case names must be unique")
        return self

    @field_validator("semantic_oracle")
    @classmethod
    def valid_semantic_oracle(cls, value: list[str] | None) -> list[str] | None:
        return _valid_argv(value) if value is not None else None


def _valid_argv(value: list[str]) -> list[str]:
    if any(not item or "\x00" in item or len(item) > 16_384 for item in value):
        raise ValueError("argv entries must be non-empty, bounded, and contain no NUL")
    return value


def _valid_environment(value: dict[str, str]) -> dict[str, str]:
    if any(
        not key or len(key) > 256 or "\x00" in key + item or "=" in key or len(item) > 8192
        for key, item in value.items()
    ):
        raise ValueError("environment overrides must be bounded names and values")
    blocked = blocked_environment_override(value)
    if blocked is not None:
        raise ValueError(f"environment override {blocked!r} is blocked by policy")
    return value


class InputIdentity(StrictModel):
    path: str
    sha256: str = Field(pattern=LOWERCASE_SHA256_PATTERN)
    size_bytes: int = Field(ge=0)
    format: str
    producer: str | None
    role: str


class ProviderIdentity(StrictModel):
    id: str
    version: str


class MetricsBlock(StrictModel):
    type: Literal["metrics"]
    values: dict[str, JsonValue]


class TableBlock(StrictModel):
    type: Literal["table"]
    rows: list[dict[str, JsonValue]]


EvidenceBlock = Annotated[MetricsBlock | TableBlock, Field(discriminator="type")]


class Coverage(StrictModel):
    rows_returned: int = Field(description="Evidence rows included on this page.", ge=0)
    rows_observed: int = Field(
        description="Evidence rows observed within the provider's readable projection.", ge=0
    )
    complete: bool = Field(
        description="Whether all available evidence was returned without truncation."
    )


class Truncation(StrictModel):
    reason: Literal["row_limit", "result_bytes", "provider_limit"] = Field(
        description="Bound that prevented the complete result from being returned."
    )
    next_offset: int = Field(description="Zero-based offset of the next unread row.", ge=0)


class AnalysisFailure(StrictModel):
    code: str
    message: str
    details: dict[str, JsonValue]


class AnalysisResult(StrictModel):
    analysis_id: str = Field(pattern=LOWERCASE_SHA256_PATTERN)
    capability_id: str
    provider: ProviderIdentity
    inputs: list[InputIdentity]
    blocks: list[EvidenceBlock]
    coverage: Coverage
    truncation: Truncation | None
    limitations: list[str]
    continuation: str | None
    capture: dict[str, JsonValue] | None = None
    analysis_failure: AnalysisFailure | None = None


@dataclass(frozen=True, slots=True)
class Capability:
    id: str
    summary: str
    formats: tuple[str, ...]
    model: ArgumentModel
    limitation: str = "Observed artifact contents do not by themselves prove causality."


def _caps(
    ids: Iterable[str], summary: str, formats: tuple[str, ...], model: ArgumentModel
) -> list[Capability]:
    return [Capability(item, summary, formats, model) for item in ids]


CAPABILITIES = tuple(
    _caps(
        ("trace.summary",),
        "Summarize bounded execution-trace evidence.",
        (
            "perfetto",
            "chrome-trace",
            "otlp",
            "nsys-rep",
            "nsys-parquet",
            "pytorch",
            "rocprof-pftrace",
            "xctrace",
        ),
        SummaryArguments,
    )
    + _caps(
        ("trace.call_graph",),
        "Project bounded caller-callee edges from Chrome, Perfetto, or PyTorch traces.",
        ("perfetto", "chrome-trace", "pytorch"),
        SummaryArguments,
    )
    + _caps(
        ("trace.pytorch",),
        "Summarize bounded PyTorch operator and event evidence.",
        ("perfetto", "chrome-trace", "pytorch"),
        SummaryArguments,
    )
    + _caps(
        ("trace.operations",),
        "Summarize bounded operation timing from OTLP or Nsight Systems traces.",
        ("otlp", "nsys-rep", "nsys-parquet"),
        SummaryArguments,
    )
    + _caps(
        ("trace.lifecycle",),
        "Summarize bounded execution lifecycle events from OTLP or Nsight Systems traces.",
        ("otlp", "nsys-rep", "nsys-parquet"),
        SummaryArguments,
    )
    + _caps(
        ("trace.window",),
        "Read a bounded trace time window.",
        ("perfetto", "chrome-trace", "pytorch", "otlp"),
        WindowArguments,
    )
    + _caps(
        ("cpu.hotspots",),
        "Rank bounded CPU evidence.",
        ("cpuprofile", "pstats", "py-spy", "perf", "perf-data"),
        SummaryArguments,
    )
    + _caps(
        ("memory.hotspots",),
        "Rank bounded allocation hotspots from a Memray capture.",
        ("memray",),
        SummaryArguments,
    )
    + _caps(
        ("memory.retained",),
        "Rank bounded retained-memory evidence from a Memray capture.",
        ("memray",),
        SummaryArguments,
    )
    + _caps(
        ("benchmark.summary",),
        "Summarize bounded benchmark latency measurements.",
        ("pyperf", "samples", "nvbench"),
        SummaryArguments,
    )
    + _caps(
        ("benchmark.scaling",),
        "Summarize how benchmark measurements scale across declared parameters.",
        ("pyperf", "samples", "nvbench"),
        SummaryArguments,
    )
    + _caps(
        ("benchmark.compare",),
        "Compare compatible benchmark artifacts.",
        ("pyperf", "samples", "nvbench"),
        CompareArguments,
    )
    + _caps(
        ("inference.summary",),
        "Summarize bounded inference-export measurements.",
        ("aiperf", "vllm-benchmark", "sglang-benchmark", "mooncake-trace"),
        SummaryArguments,
    )
    + _caps(
        ("inference.compare",),
        "Compare compatible prompt-free inference measurements.",
        ("aiperf", "vllm-benchmark", "sglang-benchmark", "mooncake-trace"),
        CompareArguments,
    )
    + _caps(
        ("gpu.launches",),
        "Summarize GPU launch evidence.",
        ("nsys-rep", "nsys-parquet"),
        SummaryArguments,
    )
    + _caps(
        ("gpu.kernel_metrics",),
        "Summarize GPU kernel metrics.",
        ("nsight-compute",),
        SummaryArguments,
    )
    + _caps(
        ("triton.autotune",),
        "Summarize Triton autotune evidence.",
        ("triton",),
        SummaryArguments,
    )
    + _caps(
        ("sanitizer.failures",),
        "Summarize Compute Sanitizer failures.",
        ("compute-sanitizer",),
        SummaryArguments,
    )
    + _caps(
        ("kernel.validation",),
        "Summarize semantic kernel validation results.",
        ("kernel-validation",),
        SummaryArguments,
    )
    + _caps(
        ("kernel.compare",),
        "Compare compatible kernel evidence.",
        ("kernel-validation",),
        CompareArguments,
    )
    + _caps(
        ("failures.summary",),
        "Summarize bounded reliability evidence.",
        ("pytest", "observations"),
        SummaryArguments,
    )
    + _caps(
        ("coverage.summary",),
        "Summarize native coverage.py evidence.",
        ("coverage",),
        SummaryArguments,
    )
    + _caps(
        ("static.performance_candidates",),
        "Normalize bounded source-analysis candidates.",
        ("sarif",),
        StaticArguments,
    )
    + _caps(
        ("artifact.preview",),
        "Preview a bounded artifact without mutation.",
        ("json", "jsonl", "csv", "text", "parquet"),
        PreviewArguments,
    )
)
CAPABILITY_BY_ID = {item.id: item for item in CAPABILITIES}


def compatible_capture_providers(capability: Capability) -> tuple[CaptureProviderContract, ...]:
    """Return capture providers whose artifacts can feed a capability."""

    if capability.id.endswith(".compare"):
        return ()
    return tuple(
        contract
        for contract in CAPTURE_PROVIDER_CONTRACTS.values()
        if set(capability.formats).intersection(artifact.format for artifact in contract.artifacts)
    )
