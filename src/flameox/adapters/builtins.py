from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from packaging.version import InvalidVersion, Version

from flameox.adapters.memray_options import memray_capture_options
from flameox.adapters.options import (
    adapter_accepts_options,
    compute_sanitizer_options,
    compute_sanitizer_suppression_path,
    cute_compiler_options,
    nsight_compute_options,
    nsight_systems_options,
    nvbench_options,
    rocprofv3_options,
    triton_compiler_options,
)
from flameox.adapters.torch_benchmark import torch_benchmark_options
from flameox.adapters.torch_profiler import SdkTorchProfilerOptions, torch_profiler_options
from flameox.domain import ArtifactKind, CapabilityExtra, DomainError, ErrorCode
from flameox.startup_profile import PYTHON_STARTUP_PROFILE


class AdapterDependencyKind(StrEnum):
    INTERNAL = "internal"
    EXECUTABLE = "executable"
    PACKAGE = "package"


@dataclass(frozen=True, slots=True)
class BuiltinAdapter:
    name: str
    dependency_kind: AdapterDependencyKind
    dependency: str | None
    supported_modes: tuple[str, ...]
    supported_formats: tuple[str, ...]
    features: tuple[str, ...] = ()
    remediation: tuple[str, ...] = ()
    version_args: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    restriction_path: Path | None = None
    supported_platforms: tuple[str, ...] | None = None
    output_filename: str | None = None
    artifact_kinds: tuple[ArtifactKind, ...] = ()
    expected_overhead: str | None = None
    capture_limitations: tuple[str, ...] = ()
    preserve_artifact_on_nonzero: bool = False
    managed_extra: CapabilityExtra | None = None
    managed_requirement: str | None = None


@dataclass(frozen=True, slots=True)
class CaptureInvocation:
    argv: tuple[str, ...]
    artifact_kinds: tuple[ArtifactKind, ...]
    expected_overhead: str
    limitations: tuple[str, ...]
    environment: dict[str, str]
    implementation_id: str | None = None


def node_version_is_supported(value: str | None) -> bool:
    """Return whether Node exposes the stable V8 profile flags used by these adapters."""
    if value is None:
        return False
    match = re.search(r"\d+(?:\.\d+)+", value)
    if match is None:
        return False
    try:
        version = Version(match.group())
    except InvalidVersion:
        return False
    return (
        (version.major == 20 and version >= Version("20.16"))
        or (version.major == 22 and version >= Version("22.4"))
        or version.major > 22
    )


BUILTIN_ADAPTERS = {
    adapter.name: adapter
    for adapter in (
        BuiltinAdapter(
            name="command",
            dependency_kind=AdapterDependencyKind.INTERNAL,
            dependency=None,
            supported_modes=("named_workload",),
            supported_formats=("process-output",),
            output_filename="capture.bin",
            artifact_kinds=(ArtifactKind.PROCESS_OUTPUT,),
            expected_overhead="No profiler overhead; process output only.",
            capture_limitations=("No sampled stack or operator evidence is collected.",),
        ),
        BuiltinAdapter(
            name="node-cpu-prof",
            dependency_kind=AdapterDependencyKind.EXECUTABLE,
            dependency="node",
            supported_modes=("record",),
            supported_formats=("v8-cpuprofile",),
            features=("sampled_stacks", "javascript_symbols"),
            remediation=("Install Node.js 20.16+ or 22.4+ which expose stable --cpu-prof flags.",),
            version_args=("--version",),
            output_filename="cpu.cpuprofile",
            artifact_kinds=(ArtifactKind.SAMPLE_PROFILE,),
            expected_overhead=(
                "V8 CPU sampling overhead; exact rate depends on --cpu-prof-interval."
            ),
            capture_limitations=(
                "Only the main Node.js thread is profiled; worker threads are not sampled.",
                "The CPU profile contains sampled stack locations, not wall-clock or "
                "allocation evidence.",
            ),
            preserve_artifact_on_nonzero=True,
        ),
        BuiltinAdapter(
            name="node-heap-prof",
            dependency_kind=AdapterDependencyKind.EXECUTABLE,
            dependency="node",
            supported_modes=("record",),
            supported_formats=("v8-sampling-heap-profile",),
            features=("allocations", "sampled_allocations", "stacks"),
            remediation=("Install Node.js 20.16+ or 22.4+ which expose stable --heap-prof flags.",),
            version_args=("--version",),
            output_filename="heap.heapprofile",
            artifact_kinds=(ArtifactKind.MEMORY_PROFILE,),
            expected_overhead=(
                "V8 heap sampling overhead; exact rate depends on --heap-prof-interval."
            ),
            capture_limitations=(
                "Sampled allocation bytes are an estimate, not the exact retained heap or "
                "process RSS.",
                "Only allocations sampled by V8 are reported; small or short-lived "
                "allocations may be underrepresented.",
            ),
            preserve_artifact_on_nonzero=True,
        ),
        BuiltinAdapter(
            name="benchmark-samples",
            dependency_kind=AdapterDependencyKind.INTERNAL,
            dependency=None,
            supported_modes=("named_workload",),
            supported_formats=("flameox.benchmark-samples.v1",),
            features=("structured_benchmark_samples", "worker_hierarchy", "warmups"),
            output_filename="benchmark-samples.json",
            artifact_kinds=(ArtifactKind.BENCHMARK_SAMPLES,),
            expected_overhead="Workload-defined structured benchmark collection.",
            capture_limitations=(
                "The workload owns measurement clock, synchronization, warm-up, and sample "
                "semantics and must declare them in the structured document.",
            ),
        ),
        BuiltinAdapter(
            name="py-spy",
            dependency_kind=AdapterDependencyKind.EXECUTABLE,
            dependency="py-spy",
            supported_modes=("record", "attach", "chrome"),
            supported_formats=("speedscope", "chrome-trace", "raw"),
            features=("sampled_stacks", "python_symbols"),
            remediation=("Install py-spy and grant ptrace access for attach mode.",),
            version_args=("--version",),
            permissions=("ptrace",),
            restriction_path=Path("/proc/sys/kernel/yama/ptrace_scope"),
            output_filename="profile.json",
            artifact_kinds=(ArtifactKind.SAMPLE_PROFILE,),
            expected_overhead=(
                "Sampling overhead; exact rate and native-frame scope are collector defaults."
            ),
            capture_limitations=(
                "GIL, native-frame, idle-thread, and subprocess modes are disabled.",
            ),
            managed_extra=CapabilityExtra.CPU,
            managed_requirement="py-spy>=0.4.2,<0.5",
        ),
        BuiltinAdapter(
            name="aiperf",
            dependency_kind=AdapterDependencyKind.EXECUTABLE,
            dependency="aiperf",
            supported_modes=("profile",),
            supported_formats=("aiperf-profile-export-jsonl",),
            features=("inference_benchmark", "per_request_metrics", "fixed_schedule"),
            remediation=(
                "Call start_capability_setup with adapter='aiperf' to create a verified "
                "provider environment.",
            ),
            version_args=("--version",),
            managed_extra=CapabilityExtra.INFERENCE,
            managed_requirement="aiperf>=0.12,<0.13",
        ),
        BuiltinAdapter(
            name="perf",
            dependency_kind=AdapterDependencyKind.EXECUTABLE,
            dependency="perf",
            supported_modes=("record", "stat", "sched"),
            supported_formats=("perf.data",),
            features=("sampled_stacks", "native_symbols"),
            remediation=("Install Linux perf matching the running kernel.",),
            version_args=("--version",),
            restriction_path=Path("/proc/sys/kernel/perf_event_paranoid"),
            supported_platforms=("linux",),
            output_filename="perf.data",
            artifact_kinds=(ArtifactKind.SAMPLE_PROFILE,),
            expected_overhead=(
                "Kernel sampling overhead; symbol coverage depends on build identities."
            ),
        ),
        BuiltinAdapter(
            name="perfetto",
            dependency_kind=AdapterDependencyKind.EXECUTABLE,
            dependency="trace_processor_shell",
            supported_modes=("import", "query"),
            supported_formats=("perfetto", "chrome-trace", "pprof", "perf.data"),
            features=("trace_sql", "temporal_slices"),
            remediation=(
                "Call start_capability_setup with adapter='perfetto' to stage the user-space "
                "Trace Processor, or configure analysis.trace_processor_path explicitly.",
            ),
            version_args=("--version",),
            managed_extra=CapabilityExtra.TRACE,
            managed_requirement="perfetto>=0.57,<0.58",
        ),
        BuiltinAdapter(
            name="pyperf",
            dependency_kind=AdapterDependencyKind.PACKAGE,
            dependency="pyperf",
            supported_modes=("import", "benchmark"),
            supported_formats=("pyperf-json",),
            features=("worker_hierarchy", "warmups", "raw_samples"),
            output_filename="benchmark.json",
            artifact_kinds=(ArtifactKind.BENCHMARK_SAMPLES,),
            expected_overhead=(
                "Repeated calibrated process execution; three workers, three values, "
                "and one warm-up per worker."
            ),
            capture_limitations=(
                "Every measured value is a fresh interpreter process and includes Python "
                "startup, imports, framework initialization, and workload setup.",
                "Use torch.benchmark for an SDK-marked in-process PyTorch operation; use "
                "benchmark-samples for another explicit synchronized producer.",
                "Experiment-level treatment randomization is separate from pyperf's worker "
                "hierarchy.",
            ),
        ),
        BuiltinAdapter(
            name="torch.benchmark",
            dependency_kind=AdapterDependencyKind.PACKAGE,
            dependency="torch",
            supported_modes=("sdk_callable",),
            supported_formats=("flameox.benchmark-samples.v1",),
            features=(
                "in_process_operation",
                "timer_wall_time",
                "optional_cuda_event_time",
                "raw_samples",
            ),
            output_filename="benchmark-samples.json",
            artifact_kinds=(ArtifactKind.BENCHMARK_SAMPLES,),
            expected_overhead=(
                "torch.utils.benchmark.Timer controls warm-up, threadpool size, CUDA "
                "synchronization, and host wall-time sampling for SDK-marked callables."
            ),
            capture_limitations=(
                "The workload constructs inputs, completes lazy initialization, and runs "
                "correctness checks outside flameox.sdk.torch_benchmark().",
                "CUDA-event timing is opt-in and emitted as a separate device-time metric; "
                "it is never substituted for Timer host wall time.",
            ),
            managed_extra=CapabilityExtra.TORCH,
            managed_requirement="torch>=2.7",
        ),
        BuiltinAdapter(
            name="python-startup",
            dependency_kind=AdapterDependencyKind.PACKAGE,
            dependency="pyperf",
            supported_modes=("repeated_process",),
            supported_formats=("pyperf-json", "python-importtime"),
            features=("wall_time", "import_cost", "module_count", "peak_rss"),
            output_filename=PYTHON_STARTUP_PROFILE.wall_output_name,
            artifact_kinds=(ArtifactKind.BENCHMARK_SAMPLES, ArtifactKind.PYTHON_STARTUP),
            expected_overhead=(
                "Five one-loop pyperf command workers plus one separately instrumented "
                "import-time execution."
            ),
            capture_limitations=(
                "The initial OS file-cache state is observed but not controlled; flameox does "
                "not drop caches.",
                "Import timing instruments imports and therefore adds measurement overhead.",
                "Peak RSS is available only when pyperf records command_max_rss for the host.",
            ),
            preserve_artifact_on_nonzero=True,
        ),
        BuiltinAdapter(
            name="pytest",
            dependency_kind=AdapterDependencyKind.PACKAGE,
            dependency="pytest",
            supported_modes=("serial", "xdist"),
            supported_formats=("pytest-reportlog-jsonl",),
            features=(
                "test_phases",
                "fixture_setup",
                "failure_latency",
                "worker_lifecycle",
            ),
            output_filename="pytest-reportlog.jsonl",
            artifact_kinds=(ArtifactKind.TEST_EXECUTION,),
            expected_overhead=(
                "pytest-reportlog serialization plus namespaced fixture timing annotations."
            ),
            capture_limitations=(
                "Controller receipt timestamps include pytest and xdist hook scheduling delay.",
                "A terminated worker can only contribute reports already delivered to pytest.",
            ),
            preserve_artifact_on_nonzero=True,
            managed_extra=CapabilityExtra.TEST,
            managed_requirement="pytest-reportlog>=1,<2",
        ),
        BuiltinAdapter(
            name="torch.profiler",
            dependency_kind=AdapterDependencyKind.PACKAGE,
            dependency="torch",
            supported_modes=("whole_entrypoint", "sdk"),
            supported_formats=("chrome-trace",),
            features=(
                "operators",
                "input_shapes",
                "allocations",
                "stacks",
                "bounded_schedules",
                "multi_cycle_exports",
            ),
            output_filename="torch-trace.json",
            artifact_kinds=(ArtifactKind.EXECUTION_TRACE,),
            expected_overhead=(
                "Operator timing is conservative by default; shapes, memory, stacks, FLOPs, "
                "and modules add collection overhead when explicitly enabled."
            ),
            capture_limitations=(
                "Evidence coverage and overhead depend on the feature flags bound into the "
                "capture plan.",
                "Whole-entrypoint capture cannot identify setup, compilation, warm-up, or "
                "steady state in an unmodified workload.",
                "Scheduled SDK mode requires explicit workload step boundaries.",
            ),
            managed_extra=CapabilityExtra.TORCH,
            managed_requirement="torch>=2.7",
        ),
        BuiltinAdapter(
            name="memray",
            dependency_kind=AdapterDependencyKind.PACKAGE,
            dependency="memray",
            supported_modes=("import", "run"),
            supported_formats=("memray",),
            features=("allocations", "retained_memory", "stacks"),
            output_filename="memory.bin",
            managed_extra=CapabilityExtra.MEMORY,
            managed_requirement="memray>=1.17",
            artifact_kinds=(ArtifactKind.MEMORY_PROFILE,),
            expected_overhead=(
                "Allocation tracing overhead; native traces are disabled by default."
            ),
            capture_limitations=(
                "The capture records the main process unless follow-fork is "
                "explicitly added in a future plan mode.",
            ),
        ),
        BuiltinAdapter(
            name="compute-sanitizer",
            dependency_kind=AdapterDependencyKind.EXECUTABLE,
            dependency="compute-sanitizer",
            supported_modes=("memcheck", "racecheck", "initcheck", "synccheck"),
            supported_formats=("compute-sanitizer-xml",),
            features=("memory_safety", "race_detection", "initialization", "synchronization"),
            remediation=(
                "Install NVIDIA Compute Sanitizer from the CUDA toolkit and verify GPU access.",
            ),
            version_args=("--version",),
            supported_platforms=("linux", "windows"),
            output_filename="compute-sanitizer.xml",
            artifact_kinds=(ArtifactKind.SANITIZER_REPORT,),
            expected_overhead="GPU instrumentation overhead; exact cost depends on sanitizer tool.",
            capture_limitations=(
                "A clean report covers only the selected tool, launches, processes, and filters.",
                "Compute Sanitizer XML has no published stable XSD; extraction is version-bounded.",
            ),
            preserve_artifact_on_nonzero=True,
        ),
        BuiltinAdapter(
            name="nvbench",
            dependency_kind=AdapterDependencyKind.INTERNAL,
            dependency=None,
            supported_modes=("benchmark",),
            supported_formats=("nvbench-json", "nvbench-jsonbin"),
            features=("gpu_benchmark", "sample_times", "sample_freqs"),
            remediation=(
                "Build the benchmark executable with NVBench linked "
                "(nvbench::main or NVBENCH_MAIN); verify CUDA toolkit and GPU access.",
            ),
            version_args=("--version",),
            supported_platforms=("linux", "windows"),
            output_filename="nvbench.json",
            artifact_kinds=(ArtifactKind.BENCHMARK_SAMPLES,),
            expected_overhead=(
                "GPU benchmark with configurable stopping criterion and sample count."
            ),
            capture_limitations=(
                "NVBench is linked into each benchmark executable; "
                "the workload argv[0] is the benchmark binary, not a wrapper.",
                "The adapter injects --jsonbin (or --json) into the declared argv; "
                "pre-existing --json or --jsonbin flags are rejected as conflicts.",
                "NVBench JSON schema is versioned; extraction is bounded to documented fields.",
                "Binary sidecars store sample data as little-endian float32.",
                "Warm-up samples are not separated in the jsonbin sidecar data.",
            ),
            preserve_artifact_on_nonzero=True,
        ),
        BuiltinAdapter(
            name="rocprofv3",
            dependency_kind=AdapterDependencyKind.EXECUTABLE,
            dependency="rocprofv3",
            supported_modes=("pftrace",),
            supported_formats=("pftrace",),
            features=(
                "hip_api",
                "kernel_dispatch",
                "memory_copy",
                "memory_allocation",
                "scratch_memory",
                "marker_ranges",
            ),
            remediation=(
                "Install rocprofiler-sdk with rocprofv3 from ROCm and verify AMD GPU access.",
            ),
            version_args=("--version",),
            supported_platforms=("linux",),
            output_filename="rocprofv3_results.pftrace",
            artifact_kinds=(ArtifactKind.EXECUTION_TRACE,),
            expected_overhead=(
                "ROCm tracing overhead depends on the selected API, dispatch, memory, scratch, "
                "and marker domains."
            ),
            capture_limitations=(
                "PFTrace contains only the explicitly selected rocprofv3 trace domains.",
                "Counter collection and the raw rocprofiler SDK are not enabled.",
            ),
            preserve_artifact_on_nonzero=True,
        ),
        BuiltinAdapter(
            name="nsight.systems",
            dependency_kind=AdapterDependencyKind.EXECUTABLE,
            dependency="nsys",
            supported_modes=("profile",),
            supported_formats=("nsys-rep", "sqlite"),
            features=("cuda_runtime", "kernels", "nvtx", "process_tree", "trace_windows"),
            remediation=("Install NVIDIA Nsight Systems and verify CUDA tracing access.",),
            version_args=("--version",),
            supported_platforms=("linux", "windows"),
            output_filename="nsight-systems.nsys-rep",
            artifact_kinds=(ArtifactKind.EXECUTION_TRACE,),
            expected_overhead=(
                "System tracing overhead depends on selected domains and process scope."
            ),
            capture_limitations=(
                "CPU sampling and context-switch sampling are disabled.",
                "SQLite is a provider export; the native .nsys-rep remains authoritative.",
                "Symbol downloads are disabled for unattended finalization.",
            ),
            preserve_artifact_on_nonzero=True,
        ),
        BuiltinAdapter(
            name="nsight.compute",
            dependency_kind=AdapterDependencyKind.EXECUTABLE,
            dependency="ncu",
            supported_modes=("profile",),
            supported_formats=("ncu-rep", "ncu-repz"),
            features=("gpu_metrics", "report_sections", "rules", "source_correlation"),
            remediation=(
                "Install NVIDIA Nsight Compute and grant access to NVIDIA GPU performance "
                "counters.",
            ),
            version_args=("--version",),
            permissions=("nvidia_gpu_performance_counters",),
            supported_platforms=("linux", "windows"),
            output_filename="nsight-compute.ncu-rep",
            artifact_kinds=(ArtifactKind.KERNEL_PROFILE,),
            expected_overhead=(
                "Kernel replay and metric collection can substantially change execution time."
            ),
            capture_limitations=(
                "Only the selected set or explicit sections and bounded launches are profiled.",
                "Counter availability depends on GPU, driver, and system permissions.",
                "Roofline evidence is exposed only when present in the official report.",
            ),
            preserve_artifact_on_nonzero=True,
        ),
        BuiltinAdapter(
            name="triton.compiler",
            dependency_kind=AdapterDependencyKind.PACKAGE,
            dependency="triton",
            supported_modes=("env_dump", "autotune_listener"),
            supported_formats=("ttir", "ttgir", "llir", "ptx", "cubin"),
            features=("compiler_ir", "ptx", "cubin", "autotune_selection"),
            remediation=(
                "Install Triton and ensure the workload invokes triton.compile or "
                "@triton.jit kernels.",
            ),
            output_filename="kernel-build.json",
            artifact_kinds=(ArtifactKind.KERNEL_BUILD,),
            expected_overhead=(
                "Env-var dump adds per-kernel IR emission overhead; a root-interpreter "
                "autotune listener records bounded provider selection facts."
            ),
            capture_limitations=(
                "Triton compiler capture requires a declared Python script or python -m module; "
                "the listener runs in that same interpreter.",
                "Only multi-configuration autotune decisions in the root Python interpreter are "
                "observed.",
                "Triton's listener does not identify a compiler dump group or cache file, so "
                "selection facts are not linked to an artifact pipeline.",
                "Only allowlisted native extensions are inventoried; unknown files are ignored.",
                "The manifest is emitted after success or compiler nonzero exit.",
            ),
            preserve_artifact_on_nonzero=True,
        ),
        BuiltinAdapter(
            name="cute.compiler",
            dependency_kind=AdapterDependencyKind.INTERNAL,
            dependency=None,
            supported_modes=("env_dump",),
            supported_formats=("cute_dsl_ir", "ptx", "cubin"),
            features=("compiler_ir", "ptx", "cubin"),
            remediation=(
                "Install CuTe DSL and ensure the workload invokes cute.compiler kernels.",
            ),
            output_filename="kernel-build.json",
            artifact_kinds=(ArtifactKind.KERNEL_BUILD,),
            expected_overhead=(
                "Env-var dump adds per-kernel IR emission overhead; no separate process is "
                "launched."
            ),
            capture_limitations=(
                "Only CUTE_DSL_DUMP_DIR and CUTE_DSL_KEEP are set; the broker runs the workload "
                "directly.",
                "Only allowlisted native extensions matching CUTE_DSL_KEEP are inventoried.",
                "The manifest is emitted after success or compiler nonzero exit.",
            ),
            preserve_artifact_on_nonzero=True,
        ),
        BuiltinAdapter(
            name="coverage",
            dependency_kind=AdapterDependencyKind.PACKAGE,
            dependency="coverage",
            supported_modes=("import", "run"),
            supported_formats=("coverage-data",),
            features=("lines", "branches"),
            output_filename=".coverage",
            managed_extra=CapabilityExtra.EXECUTION,
            managed_requirement="coverage>=7.14,<8",
            artifact_kinds=(ArtifactKind.EXECUTION_COVERAGE,),
            expected_overhead="Tracing overhead; branch collection is enabled.",
            capture_limitations=("Coverage records execution, not values or control-flow causes.",),
        ),
    )
}


def builtin_adapter(name: str) -> BuiltinAdapter | None:
    return BUILTIN_ADAPTERS.get(name)


def build_capture_invocation(  # noqa: C901 - provider routing is intentionally explicit
    adapter_name: str,
    workload_argv: tuple[str, ...],
    output_root: Path,
    *,
    executable: str | None,
    timeout_seconds: float = 300,
    options: dict[str, object] | None = None,
    project_root: Path | None = None,
    workload_executable: str | None = None,
) -> CaptureInvocation:
    adapter = BUILTIN_ADAPTERS.get(adapter_name)
    if (
        adapter is None
        or not adapter.artifact_kinds
        or adapter.output_filename is None
        or adapter.expected_overhead is None
    ):
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            f"Adapter {adapter_name!r} does not support capture planning.",
        )
    if options and not adapter_accepts_options(adapter_name):
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            f"Adapter {adapter_name!r} does not accept capture options.",
        )
    output = str(output_root / adapter.output_filename)
    environment: dict[str, str] = {}
    implementation_id: str | None = None
    limitations = adapter.capture_limitations
    if adapter_name == "command":
        argv = workload_argv
    elif adapter_name == "benchmark-samples":
        argv = workload_argv
        environment["FLAMEOX_BENCHMARK_OUTPUT"] = output
    elif adapter_name == "py-spy":
        argv = (
            _require_executable(adapter_name, executable),
            "record",
            "--format",
            "chrometrace",
            "--output",
            output,
            "--",
            *workload_argv,
        )
    elif adapter_name == "perf":
        argv = (
            _require_executable(adapter_name, executable),
            "record",
            "-o",
            output,
            "--",
            *workload_argv,
        )
    elif adapter_name == "coverage":
        python, target = _python_target(workload_argv)
        argv = (
            python,
            "-m",
            "coverage",
            "run",
            "--branch",
            f"--data-file={output}",
            *target,
        )
    elif adapter_name == "memray":
        return _memray_capture_invocation(adapter, workload_argv, output, options=options)
    elif adapter_name in {"node-cpu-prof", "node-heap-prof"}:
        return _node_v8_capture_invocation(
            adapter_name,
            adapter,
            workload_argv,
            output,
            workload_executable=workload_executable,
        )
    elif adapter_name == "torch.profiler":
        return _torch_capture_invocation(
            adapter,
            workload_argv,
            output_root,
            output,
            options=options,
        )
    elif adapter_name == "torch.benchmark":
        return _torch_benchmark_capture_invocation(
            adapter,
            workload_argv,
            output,
            options=options,
        )
    elif adapter_name == "compute-sanitizer":
        return _compute_sanitizer_capture_invocation(
            adapter,
            workload_argv,
            output,
            executable=executable,
            options=options,
            project_root=project_root,
        )
    elif adapter_name == "nvbench":
        return _nvbench_capture_invocation(
            adapter,
            workload_argv,
            output,
            options=options,
        )
    elif adapter_name == "rocprofv3":
        return _rocprofv3_capture_invocation(
            adapter,
            workload_argv,
            output_root,
            executable=executable,
            options=options,
        )
    elif adapter_name == "nsight.compute":
        return _nsight_compute_capture_invocation(
            adapter,
            workload_argv,
            output,
            executable=executable,
            options=options,
        )
    elif adapter_name == "nsight.systems":
        return _nsight_systems_capture_invocation(
            adapter,
            workload_argv,
            output_root,
            executable=executable,
            options=options,
        )
    elif adapter_name == "triton.compiler":
        return _triton_compiler_capture_invocation(
            adapter,
            workload_argv,
            output_root,
            options=options,
        )
    elif adapter_name == "cute.compiler":
        return _cute_compiler_capture_invocation(
            adapter,
            workload_argv,
            output_root,
            options=options,
        )
    elif adapter_name == "pyperf":
        python, _ = _python_target(workload_argv)
        argv = (
            python,
            "-m",
            "pyperf",
            "command",
            "--output",
            output,
            "--processes",
            "3",
            "--values",
            "3",
            "--warmups",
            "1",
            "--name",
            "workload",
            "--",
            *workload_argv,
        )
    elif adapter_name == "python-startup":
        python, _ = _python_target(workload_argv)
        launcher, implementation_id = _collector_source(adapter_name)
        argv = (
            python,
            "-c",
            launcher,
            "--benchmark-output",
            str(output_root / PYTHON_STARTUP_PROFILE.wall_output_name),
            "--import-trace-output",
            str(output_root / PYTHON_STARTUP_PROFILE.import_trace_output_name),
            "--timeout-seconds",
            str(timeout_seconds),
            "--",
            *workload_argv,
        )
    elif adapter_name == "pytest":
        launcher, implementation_id = _collector_source(adapter_name)
        argv = (
            _python_executable_for_launcher(workload_argv),
            "-c",
            launcher,
            "--output",
            output,
            "--",
            *workload_argv,
        )
    else:
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            f"Adapter {adapter_name!r} does not support capture planning.",
        )
    return CaptureInvocation(
        argv=argv,
        artifact_kinds=adapter.artifact_kinds,
        expected_overhead=adapter.expected_overhead,
        limitations=limitations,
        environment=environment,
        implementation_id=implementation_id,
    )


def _memray_capture_invocation(
    adapter: BuiltinAdapter,
    workload_argv: tuple[str, ...],
    output: str,
    *,
    options: dict[str, object] | None,
) -> CaptureInvocation:
    python, target = _python_target(workload_argv)
    selected = memray_capture_options(options)
    assert adapter.expected_overhead is not None
    if selected.mode == "sdk":
        return CaptureInvocation(
            argv=workload_argv,
            artifact_kinds=adapter.artifact_kinds,
            expected_overhead=(
                "Allocation tracing is active only during the declared SDK region."
                + (" Native stacks are enabled." if selected.native_traces else "")
                + (
                    " Python allocator events are enabled."
                    if selected.trace_python_allocators
                    else ""
                )
            ),
            limitations=(
                "The Memray tracker records every thread in the workload process while the "
                "declared region is active; it is not a thread filter.",
                "The workload must perform setup and warm-up before entering exactly one "
                "flameox.sdk.memray_region() context.",
                "Nested, concurrent, or repeated Memray SDK regions are rejected. Forked "
                "children are not tracked.",
            ),
            environment={
                "FLAMEOX_MEMRAY_CONFIG": json.dumps(
                    selected.model_dump(mode="json"),
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "FLAMEOX_MEMRAY_OUTPUT": output,
            },
        )
    return CaptureInvocation(
        argv=(
            python,
            "-m",
            "memray",
            "run",
            *(("--native",) if selected.native_traces else ()),
            *(("--trace-python-allocators",) if selected.trace_python_allocators else ()),
            "--output",
            output,
            *target,
        ),
        artifact_kinds=adapter.artifact_kinds,
        expected_overhead=(
            adapter.expected_overhead
            + (" Native stacks are enabled." if selected.native_traces else "")
            + (" Python allocator events are enabled." if selected.trace_python_allocators else "")
        ),
        limitations=(
            *adapter.capture_limitations,
            "Whole-entrypoint mode includes setup, imports, and warm-up. Use the SDK mode "
            "for one declared operation region.",
        ),
        environment={},
    )


def _torch_capture_invocation(
    adapter: BuiltinAdapter,
    workload_argv: tuple[str, ...],
    output_root: Path,
    output: str,
    *,
    options: dict[str, object] | None,
) -> CaptureInvocation:
    python, target = _torch_python_target(workload_argv)
    selected = torch_profiler_options(options)
    config = selected.model_dump(mode="json")
    encoded_config = json.dumps(
        config,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    feature_limitations = tuple(
        message
        for enabled, message in (
            (selected.record_shapes, "Input shapes were not collected."),
            (selected.profile_memory, "Profiler memory events were not collected."),
            (selected.with_stack, "Python stacks were not collected."),
            (selected.with_flops, "Operator-limited FLOP estimates were not collected."),
            (selected.with_modules, "Module hierarchy was not collected."),
        )
        if not enabled
    )
    feature_limitations = (
        *feature_limitations,
        *(
            ("FLOP estimates cover only operators supported by PyTorch's estimator.",)
            if selected.with_flops
            else ()
        ),
        *(
            ("Module hierarchy coverage is limited by the active PyTorch execution mode.",)
            if selected.with_modules
            else ()
        ),
    )
    expensive_features = tuple(
        name
        for name, enabled in (
            ("shapes", selected.record_shapes),
            ("memory", selected.profile_memory),
            ("stacks", selected.with_stack),
            ("FLOPs", selected.with_flops),
            ("modules", selected.with_modules),
        )
        if enabled
    )
    selected_activities = ", ".join(selected.activities)
    expected_overhead = (
        "Selected activities: "
        f"{selected_activities}. High-cardinality options: {', '.join(expensive_features)}."
        if expensive_features
        else (f"Selected activities: {selected_activities}. High-cardinality options: none.")
    )
    if isinstance(selected, SdkTorchProfilerOptions):
        return CaptureInvocation(
            argv=workload_argv,
            artifact_kinds=adapter.artifact_kinds,
            expected_overhead=expected_overhead,
            limitations=(
                "The approved workload owns profiler context entry and must call "
                "flameox.sdk.torch_profiler().step() at each declared step boundary.",
                "The workload environment must be able to import flameox.sdk.",
                "Schedule state is driven only by explicit workload step calls; Flameox does "
                "not infer iteration boundaries.",
                f"Selected activities: {selected_activities}.",
                *feature_limitations,
            ),
            environment={
                "FLAMEOX_TORCH_PROFILER_CONFIG": encoded_config,
                "FLAMEOX_TORCH_PROFILER_OUTPUT_ROOT": str(output_root),
            },
        )
    if target[0] == "-c":
        launcher_target = (f"--inline-code={target[1]}", "--", *target[2:])
    elif target[0] == "-m":
        if len(target) < 2:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The Python module name is missing.",
            )
        launcher_target = ("--module", target[1], "--", *target[2:])
    else:
        launcher_target = ("--script", target[0], "--", *target[1:])
    return CaptureInvocation(
        argv=(
            python,
            _launcher_path("torch_launcher.py"),
            "--output",
            output,
            "--config",
            encoded_config,
            *launcher_target,
        ),
        artifact_kinds=adapter.artifact_kinds,
        expected_overhead=expected_overhead,
        limitations=(
            "Whole-entrypoint mode captures the complete Python entrypoint. Flameox cannot "
            "separate setup, compilation, warm-up, or steady-state work in an unmodified "
            "workload.",
            "Tighter operation windows require workload instrumentation with "
            "flameox.sdk.torch_profiler() and explicit step boundaries.",
            f"Selected activities: {selected_activities}.",
            *feature_limitations,
        ),
        environment={},
    )


def _torch_benchmark_capture_invocation(
    adapter: BuiltinAdapter,
    workload_argv: tuple[str, ...],
    output: str,
    *,
    options: dict[str, object] | None,
) -> CaptureInvocation:
    _torch_python_target(workload_argv)
    selected = torch_benchmark_options(options)
    encoded_config = json.dumps(
        selected.model_dump(mode="json"),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    event_clause = (
        "CUDA event timing is enabled as a separate device metric."
        if selected.cuda_event_timing
        else "CUDA event timing is disabled."
    )
    return CaptureInvocation(
        argv=workload_argv,
        artifact_kinds=adapter.artifact_kinds,
        expected_overhead=(
            "Torch Timer minimum measurement window: "
            f"{selected.min_run_time_seconds:g} seconds; retained samples: at most "
            f"{selected.max_samples}; Torch threads: {selected.num_threads}. {event_clause}"
        ),
        limitations=(
            "The workload must call flameox.sdk.torch_benchmark() around an already "
            "constructed callable.",
            "Timer records host-observed operation time after provider-owned warm-up; it does "
            "not measure interpreter startup or input construction.",
            *adapter.capture_limitations,
        ),
        environment={
            "FLAMEOX_BENCHMARK_OUTPUT": output,
            "FLAMEOX_TORCH_BENCHMARK_CONFIG": encoded_config,
        },
    )


def _node_v8_capture_invocation(
    adapter_name: str,
    adapter: BuiltinAdapter,
    workload_argv: tuple[str, ...],
    output: str,
    *,
    workload_executable: str | None,
) -> CaptureInvocation:
    """Inject Node.js --cpu-prof or --heap-prof flags into a declared Node workload.

    Node.js exposes stable V8 profiling through CLI flags (Node 20.16+ / 22.4+).
    The declared workload argv already starts with the Node executable, so the
    adapter inserts the profiling flags immediately after argv[0] and before the
    user script and its arguments.  The output directory and file name are bound
    explicitly so Flameox owns the artifact path and can preserve it.
    """
    if not workload_argv:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "A declared Node.js workload command is required for V8 profiling.",
        )
    node_executable = workload_executable or workload_argv[0]
    node_name = Path(node_executable).name
    if not (
        node_name == "node" or node_name.startswith("node") or node_executable.endswith("node")
    ):
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            f"V8 profiling requires a Node.js workload; the declared "
            f"executable is {node_executable!r}.",
            remediation=("Declare a Node.js command (e.g. `node script.js`) as the workload.",),
        )
    if len(workload_argv) < 2:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "A declared Node.js script or module is required after the node executable.",
        )
    # Compute the directory and file name for the V8 profile output.
    output_path = Path(output)
    prof_dir = str(output_path.parent)
    prof_name = output_path.name
    if adapter_name == "node-cpu-prof":
        prof_flags = (
            "--cpu-prof",
            "--cpu-prof-dir=" + prof_dir,
            "--cpu-prof-name=" + prof_name,
        )
    else:
        prof_flags = (
            "--heap-prof",
            "--heap-prof-dir=" + prof_dir,
            "--heap-prof-name=" + prof_name,
        )
    argv = (node_executable, *prof_flags, *workload_argv[1:])
    return CaptureInvocation(
        argv=argv,
        artifact_kinds=adapter.artifact_kinds,
        expected_overhead=adapter.expected_overhead or "",
        limitations=adapter.capture_limitations,
        environment={},
    )


def _compute_sanitizer_capture_invocation(
    adapter: BuiltinAdapter,
    workload_argv: tuple[str, ...],
    output: str,
    *,
    executable: str | None,
    options: dict[str, object] | None,
    project_root: Path | None,
) -> CaptureInvocation:
    selected = compute_sanitizer_options(options)
    limitations = [*adapter.capture_limitations]
    if selected.launch_count == 0:
        limitations.append(
            "launch_count=0 is an explicit unlimited capture and may instrument every CUDA "
            "launch until the workload exits or times out."
        )
    limitations.append(
        "The launch bound counts setup and reference CUDA work before the target operation; "
        "isolate a target-only workload or apply a kernel filter when those launches dominate."
    )
    argv_parts: list[str] = [
        _require_executable(adapter.name, executable),
        "--tool",
        selected.tool,
        "--xml",
        "--save",
        output,
        "--error-exitcode",
        str(selected.finding_exit_code),
        "--launch-skip",
        str(selected.launch_skip),
        "--launch-count",
        str(selected.launch_count),
        "--target-processes",
        selected.target_processes,
        "--demangle",
        selected.demangle,
    ]
    if selected.target_processes_filter is not None:
        argv_parts.extend(("--target-processes-filter", selected.target_processes_filter))
    if selected.kernel_name is not None:
        argv_parts.extend(("--kernel-name", selected.kernel_name))
    suppression = compute_sanitizer_suppression_path(
        selected,
        project_root=project_root or Path.cwd(),
    )
    if suppression is not None:
        argv_parts.extend(("--suppressions", str(suppression)))
    argv = (*argv_parts, *workload_argv)
    return CaptureInvocation(
        argv=argv,
        artifact_kinds=adapter.artifact_kinds,
        expected_overhead=adapter.expected_overhead or "",
        limitations=tuple(limitations),
        environment={},
    )


def replace_compute_sanitizer_suppression(
    argv: tuple[str, ...],
    staged_path: Path,
    *,
    workload_argv: tuple[str, ...],
) -> tuple[str, ...]:
    """Point a planned Compute Sanitizer invocation at its verified staged input."""
    if len(workload_argv) > len(argv) or argv[-len(workload_argv) :] != workload_argv:
        raise DomainError(
            ErrorCode.INTERNAL_ERROR,
            "Planned Compute Sanitizer workload arguments do not match the collector suffix.",
        )
    collector_length = len(argv) - len(workload_argv)
    indexes = [
        index for index, value in enumerate(argv[:collector_length]) if value == "--suppressions"
    ]
    if len(indexes) != 1:
        raise DomainError(
            ErrorCode.INTERNAL_ERROR,
            "Planned Compute Sanitizer suppression argument is missing or ambiguous.",
        )
    index = indexes[0] + 1
    return (*argv[:index], str(staged_path), *argv[index + 1 :])


def _nvbench_capture_invocation(
    adapter: BuiltinAdapter,
    workload_argv: tuple[str, ...],
    output: str,
    *,
    options: dict[str, object] | None,
) -> CaptureInvocation:
    """Inject the official output flag into the declared benchmark argv.

    NVBench is linked into each benchmark executable via ``NVBENCH_MAIN``
    (``nvbench/main.cuh``); there is no universal ``nvbench`` wrapper that
    runs a workload after ``--``.  The declared ``workload_argv[0]`` IS the
    benchmark binary.  This function injects the official output flag
    (verified from ``nvbench/option_parser.cu`` at commit c184889):

    ``--json`` and ``--jsonbin`` are **alternative** output modes, not
    complementary:

    - ``--jsonbin <path>`` (default, ``enable_jsonbin=True``): creates a
      ``json_printer`` with ``enable_binary_output=true`` that writes the
      JSON document **and** binary sidecar directories ``<path>-bin/`` and
      ``<path>-freqs-bin/``.
    - ``--json <path>`` (``enable_jsonbin=False``): creates a
      ``json_printer`` with ``enable_binary_output=false`` that writes only
      the JSON document, with no binary sidecars.

    Emitting both would create two competing JSON printers writing to the
    same file, so exactly one is injected.

    Pre-existing ``--json`` or ``--jsonbin`` flags (in either
    space-separated ``--json <path>`` or equals-separated ``--json=<path>``
    form) in ``workload_argv`` are rejected as conflicts.
    Optional NVBench flags (``--stopping-criterion``, ``--min-samples``,
    ``--timeout``, ``-d``) are injected before the workload's own arguments.
    """
    if not workload_argv:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "NVBench capture requires a benchmark executable as workload argv[0].",
        )
    for arg in workload_argv:
        if arg == "--json" or arg == "--jsonbin" or arg.startswith(("--json=", "--jsonbin=")):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                f"Workload argv already contains an NVBench output flag "
                f"({arg!r}); the managed nvbench adapter injects its own "
                "and conflicts must be removed.",
            )
    selected = nvbench_options(options)
    benchmark_exe = workload_argv[0]
    rest_argv = workload_argv[1:]
    output_flag = "--jsonbin" if selected.enable_jsonbin else "--json"
    injected: list[str] = [benchmark_exe, output_flag, output]
    if selected.stopping_criterion is not None:
        injected.extend(("--stopping-criterion", selected.stopping_criterion))
    if selected.min_samples is not None:
        injected.extend(("--min-samples", str(selected.min_samples)))
    if selected.timeout is not None:
        injected.extend(("--timeout", str(selected.timeout)))
    if selected.devices is not None:
        injected.extend(("-d", selected.devices))
    argv = (*injected, *rest_argv)
    return CaptureInvocation(
        argv=argv,
        artifact_kinds=adapter.artifact_kinds,
        expected_overhead=adapter.expected_overhead or "",
        limitations=adapter.capture_limitations,
        environment={},
    )


def _rocprofv3_capture_invocation(
    adapter: BuiltinAdapter,
    workload_argv: tuple[str, ...],
    output_root: Path,
    *,
    executable: str | None,
    options: dict[str, object] | None,
) -> CaptureInvocation:
    selected = rocprofv3_options(options)
    argv_parts: list[str] = [
        _require_executable(adapter.name, executable),
        "--output-format",
        "pftrace",
        "-o",
        "rocprofv3",
        "-d",
        str(output_root),
    ]
    for enabled, flag in (
        (selected.hip_trace, "--hip-trace"),
        (selected.kernel_trace, "--kernel-trace"),
        (selected.memory_copy_trace, "--memory-copy-trace"),
        (selected.memory_allocation_trace, "--memory-allocation-trace"),
        (selected.scratch_memory_trace, "--scratch-memory-trace"),
        (selected.marker_trace, "--marker-trace"),
    ):
        if enabled:
            argv_parts.append(flag)
    argv = (*argv_parts, "--", *workload_argv)
    return CaptureInvocation(
        argv=argv,
        artifact_kinds=adapter.artifact_kinds,
        expected_overhead=adapter.expected_overhead or "",
        limitations=adapter.capture_limitations,
        environment={},
    )


def _nsight_compute_capture_invocation(
    adapter: BuiltinAdapter,
    workload_argv: tuple[str, ...],
    output: str,
    *,
    executable: str | None,
    options: dict[str, object] | None,
) -> CaptureInvocation:
    selected = nsight_compute_options(options)
    argv_parts: list[str] = [
        _require_executable(adapter.name, executable),
        "--export",
        output,
        "--force-overwrite",
        "--replay-mode",
        selected.replay_mode,
        "--launch-skip",
        str(selected.launch_skip),
        "--launch-count",
        str(selected.launch_count),
    ]
    if selected.set is not None:
        argv_parts.extend(("--set", selected.set))
    else:
        for section in selected.sections or ():
            argv_parts.extend(("--section", section))
    if selected.kernel_name is not None:
        argv_parts.extend(
            ("--kernel-name-base", "demangled", "--kernel-name", selected.kernel_name)
        )
    argv = (*argv_parts, *workload_argv)
    return CaptureInvocation(
        argv=argv,
        artifact_kinds=adapter.artifact_kinds,
        expected_overhead=adapter.expected_overhead or "",
        limitations=adapter.capture_limitations,
        environment={},
    )


def _nsight_systems_capture_invocation(
    adapter: BuiltinAdapter,
    workload_argv: tuple[str, ...],
    output_root: Path,
    *,
    executable: str | None,
    options: dict[str, object] | None,
) -> CaptureInvocation:
    selected = nsight_systems_options(options)
    output_stem = output_root / "nsight-systems"
    argv_parts = [
        _require_executable(adapter.name, executable),
        "profile",
        f"--trace={','.join(selected.trace)}",
        "--sample=none",
        "--cpuctxsw=none",
        "--resolve-symbols=false",
        "--export=sqlite",
        "--force-overwrite=true",
        "--trace-fork-before-exec="
        + ("true" if selected.include_pre_exec_fork_interval else "false"),
        f"--cuda-trace-scope={selected.cuda_trace_scope}",
        f"--cuda-graph-trace={selected.cuda_graph_trace}",
        f"--capture-range={selected.capture_range}",
    ]
    if selected.capture_range != "none":
        assert selected.capture_range_end is not None
        argv_parts.append(f"--capture-range-end={selected.capture_range_end}")
    if selected.nvtx_capture is not None:
        argv_parts.append(f"--nvtx-capture={selected.nvtx_capture}")
    argv = (*argv_parts, "--output", str(output_stem), *workload_argv)
    return CaptureInvocation(
        argv=argv,
        artifact_kinds=adapter.artifact_kinds,
        expected_overhead=adapter.expected_overhead or "",
        limitations=adapter.capture_limitations,
        environment={},
    )


def _triton_compiler_capture_invocation(
    adapter: BuiltinAdapter,
    workload_argv: tuple[str, ...],
    output_root: Path,
    *,
    options: dict[str, object] | None,
) -> CaptureInvocation:
    """Capture native dumps and provider-owned autotune decisions in one interpreter."""
    selected = triton_compiler_options(options)
    python, target = _triton_python_target(workload_argv)
    launcher, implementation_id = _collector_source("triton.compiler")
    dump_dir = output_root / selected.dump_subdir
    environment: dict[str, str] = {
        "TRITON_DUMP_DIR": str(dump_dir),
        "TRITON_KERNEL_DUMP": "1" if selected.kernel_dump else "0",
    }
    if selected.reproducer_filename is not None:
        environment["TRITON_REPRODUCER_PATH"] = str(output_root / selected.reproducer_filename)
    return CaptureInvocation(
        argv=(
            python,
            "-c",
            launcher,
            "--events",
            str(output_root / "triton-autotune.jsonl"),
            "--compiler-events",
            str(output_root / "triton-compiler.jsonl"),
            "--",
            *target,
        ),
        artifact_kinds=adapter.artifact_kinds,
        expected_overhead=adapter.expected_overhead or "",
        limitations=adapter.capture_limitations,
        environment=environment,
        implementation_id=implementation_id,
    )


def _cute_compiler_capture_invocation(
    adapter: BuiltinAdapter,
    workload_argv: tuple[str, ...],
    output_root: Path,
    *,
    options: dict[str, object] | None,
) -> CaptureInvocation:
    """Set CuTe DSL env-var dump controls; the broker runs the workload directly."""
    selected = cute_compiler_options(options)
    dump_dir = output_root / selected.dump_subdir
    environment: dict[str, str] = {
        "CUTE_DSL_DUMP_DIR": str(dump_dir),
        "CUTE_DSL_KEEP": ",".join(selected.keep_allowlist),
    }
    return CaptureInvocation(
        argv=workload_argv,
        artifact_kinds=adapter.artifact_kinds,
        expected_overhead=adapter.expected_overhead or "",
        limitations=adapter.capture_limitations,
        environment=environment,
    )


def _python_target(
    workload_argv: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    executable = Path(workload_argv[0]).name
    if not executable.startswith("python"):
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "This adapter can launch only a declared Python script or module.",
        )
    arguments = workload_argv[1:]
    if not arguments or arguments[0] == "-c":
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "Inline Python commands cannot be wrapped by this adapter.",
            remediation=("Declare a script or `python -m module` workload.",),
        )
    return workload_argv[0], arguments


def _triton_python_target(workload_argv: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    python, arguments = _python_target(workload_argv)
    if arguments[0] == "-m" and len(arguments) >= 2:
        return python, arguments
    if arguments[0].startswith("-"):
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "Triton compiler capture supports only a Python script or `python -m module`.",
            remediation=("Declare a script or `python -m module` workload.",),
        )
    return python, arguments


def _torch_python_target(
    workload_argv: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    """Resolve Python targets supported by the torch whole-entrypoint launcher."""
    if not workload_argv:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "The declared Python workload command is missing.",
        )
    executable = Path(workload_argv[0]).name
    if not executable.startswith("python"):
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "This adapter can launch only a declared Python script, module, or inline command.",
        )
    arguments = workload_argv[1:]
    if not arguments:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "The declared Python workload target is missing.",
        )
    if arguments[0] == "-c" and len(arguments) < 2:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "The inline Python program is missing.",
        )
    if arguments[0] != "-c":
        return _python_target(workload_argv)
    return workload_argv[0], arguments


def _require_executable(adapter: str, executable: str | None) -> str:
    if executable is None:
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            f"Adapter {adapter!r} has no resolved executable.",
        )
    return executable


def _python_executable_for_launcher(workload_argv: tuple[str, ...]) -> str:
    if not workload_argv:
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "The pytest workload command is missing.",
            remediation=("Declare `pytest` or `python -m pytest` as the workload command.",),
        )
    executable = Path(workload_argv[0]).name
    if executable.startswith("python"):
        if len(workload_argv) < 3 or workload_argv[1:3] != ("-m", "pytest"):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The pytest adapter requires `pytest` or `python -m pytest`.",
                remediation=("Declare `pytest` or `python -m pytest` as the workload command.",),
            )
    elif not executable.startswith("pytest"):
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "The pytest adapter requires `pytest` or `python -m pytest`.",
            remediation=("Declare `pytest` or `python -m pytest` as the workload command.",),
        )
    if executable.startswith("python"):
        return workload_argv[0]
    return sys.executable


def _launcher_path(name: str) -> str:
    """Return a standalone launcher path usable by an unrelated workload venv."""
    return str((Path(__file__).parent.parent / "collectors" / name).resolve())


_COLLECTOR_FILES = {
    "python-startup": ("python_startup.py",),
    "pytest": (
        "pytest_launcher.py",
        "pytest_plugin.py",
    ),
    "triton.compiler": ("triton_autotune.py",),
}


def collector_implementation_id(adapter: str) -> str | None:
    files = _COLLECTOR_FILES.get(adapter)
    if files is None:
        return None
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update(b"\x00")
        digest.update(Path(_launcher_path(name)).read_bytes())
        digest.update(b"\x00")
    return "sha256:" + digest.hexdigest()


def _collector_source(adapter: str) -> tuple[str, str]:
    launcher_name = _COLLECTOR_FILES[adapter][0]
    launcher = Path(_launcher_path(launcher_name)).read_text(encoding="utf-8")
    if adapter == "pytest":
        plugin = Path(_launcher_path("pytest_plugin.py")).read_text(encoding="utf-8")
        launcher = launcher.replace('"__FLAMEOX_PLUGIN_SOURCE__"', repr(plugin))
    implementation_id = collector_implementation_id(adapter)
    assert implementation_id is not None
    return launcher, implementation_id
