from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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
MAX_SNIFF_INPUTS = 16


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
    path: str = Field(min_length=1, max_length=4096)
    format: str | None = Field(default=None, min_length=1, max_length=80)
    producer: str | None = Field(default=None, min_length=1, max_length=80)
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class EvidenceSource(StrictModel):
    kind: Literal["evidence"]
    evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_role: str | None = Field(default=None, min_length=1, max_length=80)


Source = Annotated[PathSource | EvidenceSource, Field(discriminator="kind")]


class EmptyArguments(StrictModel):
    pass


class PreviewArguments(StrictModel):
    offset: int = Field(default=0, ge=0)


class SummaryArguments(StrictModel):
    metric: str | None = Field(default=None, max_length=120)
    group_by: str | None = Field(default=None, max_length=120)


class StaticArguments(StrictModel):
    include_paths: list[str] = Field(default_factory=list, max_length=128)
    exclude_paths: list[str] = Field(default_factory=list, max_length=128)

    @field_validator("include_paths", "exclude_paths")
    @classmethod
    def bounded_patterns(cls, value: list[str]) -> list[str]:
        if any(not pattern or len(pattern) > 256 or "\x00" in pattern for pattern in value):
            raise ValueError("path patterns must be non-empty, bounded, and contain no NUL")
        return value


class WindowArguments(StrictModel):
    start_ns: int = Field(ge=0)
    end_ns: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> WindowArguments:
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be greater than start_ns")
        return self


class CompareArguments(StrictModel):
    metric: str | None = Field(default=None, max_length=120)
    baseline_index: int = Field(default=0, ge=0, le=31)


ArgumentModel = type[
    EmptyArguments
    | PreviewArguments
    | SummaryArguments
    | StaticArguments
    | WindowArguments
    | CompareArguments
]


class RequestLimits(StrictModel):
    max_rows: int = Field(default=100, ge=1, le=MAX_ROWS)
    max_result_bytes: int = Field(default=MAX_RESULT_BYTES, ge=1024, le=MAX_RESULT_BYTES)
    max_input_bytes: int = Field(default=1024**3, ge=1024, le=1024**3)
    max_input_files: int = Field(default=4096, ge=1, le=4096)
    timeout_seconds: float = Field(default=300, gt=0, le=3600)
    max_output_bytes: int = Field(default=16 * 1024 * 1024, ge=1024, le=64 * 1024 * 1024)
    max_memory_bytes: int = Field(default=1024**3, ge=16 * 1024 * 1024, le=16 * 1024**3)
    max_provenance_bytes: int = Field(default=16 * 1024 * 1024, ge=4 * 1024, le=64 * 1024 * 1024)

    def lowered_against(self, startup: RequestLimits) -> RequestLimits:
        effective = startup.model_dump()
        for name in self.model_fields_set:
            requested = getattr(self, name)
            if requested > getattr(startup, name):
                raise RuntimeFailure("LIMIT_EXCEEDED", f"{name} cannot raise the server limit")
            effective[name] = requested
        return RequestLimits.model_validate(effective)


class CaptureTarget(StrictModel):
    argv: list[str] = Field(min_length=1, max_length=256)
    cwd: str = "."
    environment: dict[str, str] = Field(default_factory=dict, max_length=32)
    provider_id: str = Field(min_length=1, max_length=80)
    capture_arguments: dict[str, Any] = Field(default_factory=dict)
    analysis_arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("argv")
    @classmethod
    def valid_argv(cls, value: list[str]) -> list[str]:
        return _valid_argv(value)

    @field_validator("environment")
    @classmethod
    def valid_environment(cls, value: dict[str, str]) -> dict[str, str]:
        return _valid_environment(value)


class PyperfCaptureArguments(StrictModel):
    processes: int = Field(default=1, ge=1, le=32)
    values: int = Field(default=3, ge=1, le=100)
    warmups: int = Field(default=1, ge=0, le=100)
    loops: int = Field(default=0, ge=0, le=1_000_000_000)
    min_time: float = Field(default=0.1, gt=0, le=60)
    name: str = Field(default="command", min_length=1, max_length=120)


class PySpyCaptureArguments(StrictModel):
    rate: int = Field(default=100, ge=1, le=1_000)
    gil: bool = False
    native: bool = False


class PerfCaptureArguments(StrictModel):
    frequency: int = Field(default=99, ge=1, le=1_000)
    call_graph: Literal["dwarf", "fp"] = "dwarf"


class MemrayCaptureArguments(StrictModel):
    native: bool = False


class ComputeSanitizerCaptureArguments(StrictModel):
    tool: Literal["memcheck", "racecheck", "initcheck", "synccheck"] = "memcheck"


def _default_nsys_traces() -> list[Literal["cuda", "nvtx", "osrt", "cublas", "cudnn"]]:
    return ["cuda", "nvtx", "osrt"]


class NsightSystemsCaptureArguments(StrictModel):
    trace: list[Literal["cuda", "nvtx", "osrt", "cublas", "cudnn"]] = Field(
        default_factory=_default_nsys_traces, min_length=1, max_length=5
    )


class NsightComputeCaptureArguments(StrictModel):
    replay_mode: Literal["kernel", "application", "range"] = "kernel"
    launch_skip: int = Field(default=0, ge=0, le=1_000_000)
    launch_count: int = Field(default=1, ge=1, le=1_000_000)
    section: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("section")
    @classmethod
    def bounded_sections(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 200 or "\x00" in item for item in value):
            raise ValueError("Nsight Compute sections must be non-empty and bounded")
        return value


class RocprofCaptureArguments(StrictModel):
    hip_trace: bool = False
    kernel_trace: bool = True
    memory_copy_trace: bool = False
    memory_allocation_trace: bool = False
    scratch_memory_trace: bool = False
    marker_trace: bool = False

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
    template: Literal["Metal System Trace", "Time Profiler"] = "Metal System Trace"


def _default_torch_activities() -> list[Literal["cpu", "cuda", "cuda_if_available"]]:
    return ["cpu"]


class TorchProfilerCaptureArguments(StrictModel):
    activities: list[Literal["cpu", "cuda", "cuda_if_available"]] = Field(
        default_factory=_default_torch_activities, min_length=1, max_length=2
    )
    wait: int = Field(default=0, ge=0, le=10_000)
    warmup: int = Field(default=1, ge=0, le=10_000)
    active: int = Field(default=1, ge=1, le=10_000)
    skip_first: int = Field(default=0, ge=0, le=10_000)
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = False
    with_modules: bool = False


class CoverageCaptureArguments(StrictModel):
    branch: bool = False
    source: list[str] = Field(default_factory=list, max_length=64)
    include: list[str] = Field(default_factory=list, max_length=64)
    omit: list[str] = Field(default_factory=list, max_length=64)

    @field_validator("source", "include", "omit")
    @classmethod
    def bounded_values(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 256 or "\x00" in item for item in value):
            raise ValueError("coverage selectors must be non-empty and bounded")
        return value


class TorchBenchmarkRuntimeArguments(StrictModel):
    min_run_time_seconds: float = Field(default=0.2, gt=0, le=60)
    max_samples: int = Field(default=100, ge=1, le=1_000)
    num_threads: int = Field(default=1, ge=1, le=256)
    cuda_event_timing: bool = False


class BenchmarkSamplesCaptureArguments(StrictModel):
    torch_benchmark: TorchBenchmarkRuntimeArguments | None = None


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
    preview_streams: bool = False


def _capture_provider(
    provider_id: str,
    argument_model: type[StrictModel],
    artifacts: tuple[tuple[str, str], ...],
    description: str,
    *,
    preview_streams: bool = False,
) -> CaptureProviderContract:
    return CaptureProviderContract(
        provider_id,
        argument_model,
        tuple(CaptureArtifactContract(role, format_name) for role, format_name in artifacts),
        description,
        preview_streams,
    )


CAPTURE_PROVIDER_CONTRACTS = {
    item.id: item
    for item in (
        _capture_provider(
            "direct", EmptyArguments, (), "stdout and stderr text", preview_streams=True
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
    name: str = Field(min_length=1, max_length=80)
    argv: list[str] | None = Field(default=None, min_length=1, max_length=256)
    environment: dict[str, str] = Field(default_factory=dict, max_length=32)

    @field_validator("argv")
    @classmethod
    def valid_argv(cls, value: list[str] | None) -> list[str] | None:
        return _valid_argv(value) if value is not None else None

    @field_validator("environment")
    @classmethod
    def valid_environment(cls, value: dict[str, str]) -> dict[str, str]:
        return _valid_environment(value)


class ExperimentDesign(StrictModel):
    cases: list[ExperimentCase] = Field(min_length=2, max_length=16)
    blocks: int = Field(ge=1, le=100)
    seed: int
    metric: Literal["wall_time_ns"]
    estimand: Literal["median_difference", "mean_difference"]
    practical_threshold: float = Field(ge=0)
    semantic_oracle: list[str] | None = Field(default=None, min_length=1, max_length=256)

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
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
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
    rows_returned: int = Field(ge=0)
    rows_observed: int = Field(ge=0)
    complete: bool


class Truncation(StrictModel):
    reason: Literal["row_limit", "result_bytes", "provider_limit"]
    next_offset: int = Field(ge=0)


class AnalysisFailure(StrictModel):
    code: str
    message: str
    details: dict[str, JsonValue]


class AnalysisResult(StrictModel):
    analysis_id: str = Field(pattern=r"^[0-9a-f]{64}$")
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
class ProviderProbe:
    id: str
    formats: tuple[str, ...]
    module: str | None = None
    distribution: str | None = None
    supported_versions: str | None = None
    executable: str | None = None
    configured_path_environment: str | None = None
    platforms: tuple[str, ...] = ()
    setup_provider: str | None = None


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
        ("trace.call_graph", "trace.pytorch"),
        "Project bounded call-graph or PyTorch trace evidence.",
        ("perfetto", "chrome-trace", "pytorch"),
        SummaryArguments,
    )
    + _caps(
        ("trace.operations", "trace.lifecycle"),
        "Summarize bounded operation and lifecycle evidence.",
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
        ("memory.hotspots", "memory.retained"),
        "Rank bounded allocation evidence.",
        ("memray",),
        SummaryArguments,
    )
    + _caps(
        ("benchmark.summary", "benchmark.scaling"),
        "Summarize benchmark measurements.",
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
FORMAT_PROBES = (
    ProviderProbe("flameox-preview", ("json", "jsonl", "csv", "text", "parquet")),
    ProviderProbe("v8-cpu-profile", ("cpuprofile",)),
    ProviderProbe("py-spy-speedscope", ("py-spy",)),
    ProviderProbe("perf-collapsed", ("perf",)),
    ProviderProbe("perf-data", ("perf-data",), executable="perf", platforms=("linux",)),
    ProviderProbe(
        "memray",
        ("memray",),
        module="memray",
        distribution="memray",
        supported_versions=">=1.17",
        setup_provider="memray",
    ),
    ProviderProbe(
        "pyperf",
        ("pyperf",),
        module="pyperf",
        distribution="pyperf",
        supported_versions=">=2.10,<2.11",
    ),
    ProviderProbe("benchmark-samples", ("samples",)),
    ProviderProbe(
        "aiperf",
        ("aiperf",),
        module="aiperf",
        distribution="aiperf",
        supported_versions=">=0.12,<0.13",
        setup_provider="aiperf",
    ),
    ProviderProbe("vllm-benchmark", ("vllm-benchmark",)),
    ProviderProbe("sglang-benchmark", ("sglang-benchmark",)),
    ProviderProbe("mooncake-trace", ("mooncake-trace",)),
    ProviderProbe("nvbench-jsonbin", ("nvbench",)),
    ProviderProbe(
        "perfetto",
        ("perfetto", "chrome-trace", "pytorch", "rocprof-pftrace"),
        module="perfetto.trace_processor",
        distribution="perfetto",
        supported_versions=">=0.57,<0.58",
        executable="trace_processor_shell",
        configured_path_environment="FLAMEOX_TRACE_PROCESSOR",
        setup_provider="perfetto",
    ),
    ProviderProbe(
        "otlp",
        ("otlp",),
        module="opentelemetry.proto",
        distribution="opentelemetry-proto",
        supported_versions=">=1.44,<1.45",
        setup_provider="otlp",
    ),
    ProviderProbe(
        "nsight-systems",
        ("nsys-rep",),
        executable="nsys",
        platforms=("linux", "win32"),
    ),
    ProviderProbe(
        "nsight-systems-parquetdir",
        ("nsys-parquet",),
        module="pyarrow",
        distribution="pyarrow",
        supported_versions=">=20,<26",
    ),
    ProviderProbe(
        "nsight-compute",
        ("nsight-compute",),
        executable="ncu",
        platforms=("linux", "win32"),
    ),
    ProviderProbe("xctrace", ("xctrace",), executable="xcrun", platforms=("darwin",)),
    ProviderProbe("compute-sanitizer-xml", ("compute-sanitizer",)),
    ProviderProbe("triton-autotune-jsonl", ("triton",)),
    ProviderProbe("kernel-validation", ("kernel-validation",)),
    ProviderProbe("pytest", ("pytest",)),
    ProviderProbe(
        "coverage.py",
        ("coverage",),
        module="coverage",
        distribution="coverage",
        supported_versions=">=7.14,<8",
    ),
    ProviderProbe("flameox-observations", ("observations",)),
    ProviderProbe("sarif", ("sarif",)),
)

CAPTURE_PROBES = {
    "direct": ProviderProbe("direct", ()),
    "pyperf": ProviderProbe(
        "pyperf",
        (),
        module="pyperf",
        distribution="pyperf",
        supported_versions=">=2.10,<2.11",
    ),
    "py-spy": ProviderProbe("py-spy", (), executable="py-spy", setup_provider="py-spy"),
    "perf": ProviderProbe("perf", (), executable="perf", platforms=("linux",)),
    "node-cpu-profile": ProviderProbe("node-cpu-profile", ()),
    "memray": ProviderProbe(
        "memray",
        (),
        module="memray",
        distribution="memray",
        supported_versions=">=1.17",
        setup_provider="memray",
    ),
    "torch-profiler": ProviderProbe(
        "torch-profiler", (), module="torch", distribution="torch", setup_provider="torch"
    ),
    "nvbench": ProviderProbe("nvbench", (), platforms=("linux", "win32")),
    "compute-sanitizer": ProviderProbe(
        "compute-sanitizer",
        (),
        executable="compute-sanitizer",
        platforms=("linux", "win32"),
    ),
    "nsight-systems": ProviderProbe(
        "nsight-systems", (), executable="nsys", platforms=("linux", "win32")
    ),
    "nsight-compute": ProviderProbe(
        "nsight-compute", (), executable="ncu", platforms=("linux", "win32")
    ),
    "rocprofv3": ProviderProbe("rocprofv3", (), executable="rocprofv3", platforms=("linux",)),
    "xctrace": ProviderProbe("xctrace", (), executable="xcrun", platforms=("darwin",)),
    "coverage": ProviderProbe(
        "coverage",
        (),
        module="coverage",
        distribution="coverage",
        supported_versions=">=7.14,<8",
    ),
    "benchmark-samples": ProviderProbe("benchmark-samples", ()),
    "observations": ProviderProbe("observations", ()),
    "pytest": ProviderProbe("pytest", ()),
}
