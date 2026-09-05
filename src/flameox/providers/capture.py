"""Provider-owned construction of native capture commands and artifacts."""

from __future__ import annotations

import base64
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from flameox.runtime_contracts import (
    BenchmarkSamplesCaptureArguments,
    CaptureArguments,
    ComputeSanitizerCaptureArguments,
    CoverageCaptureArguments,
    EmptyArguments,
    MemrayCaptureArguments,
    NsightComputeCaptureArguments,
    NsightSystemsCaptureArguments,
    PerfCaptureArguments,
    PyperfCaptureArguments,
    PySpyCaptureArguments,
    RocprofCaptureArguments,
    RuntimeFailure,
    TorchProfilerCaptureArguments,
    XctraceCaptureArguments,
)


@dataclass(frozen=True, slots=True)
class CaptureInvocation:
    argv: tuple[str, ...]
    environment: dict[str, str]
    artifacts: tuple[tuple[Path, str, str], ...]
    # Wrapper processes cannot prove the workload's exit without a separate receipt.
    returncode_scope: Literal["workload", "collector"] = "collector"


@dataclass(frozen=True, slots=True)
class CaptureBuildRequest:
    target_argv: list[str]
    environment: dict[str, str]
    arguments: CaptureArguments
    directory: Path


type ManagedExecutable = Callable[[str, str], str]
type CaptureBuilder = Callable[[CaptureBuildRequest, ManagedExecutable], CaptureInvocation]


def _arguments(request: CaptureBuildRequest, expected: type[CaptureArguments]) -> CaptureArguments:
    if not isinstance(request.arguments, expected):
        raise RuntimeFailure("INVALID_INPUT", "Capture provider arguments do not match its schema")
    return request.arguments


def _direct(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    _arguments(request, EmptyArguments)
    return CaptureInvocation(tuple(request.target_argv), request.environment, (), "workload")


def _pyperf(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    arguments = cast(PyperfCaptureArguments, _arguments(request, PyperfCaptureArguments))
    output = request.directory / "benchmark.json"
    target = tuple(request.target_argv)
    if any("\n" in item or "\r" in item for item in target):
        encoded = base64.urlsafe_b64encode(
            json.dumps(request.target_argv, ensure_ascii=False).encode()
        ).decode("ascii")
        target = (sys.executable, "-m", "flameox.workers.pyperf_target", encoded)
    argv = (
        sys.executable,
        "-m",
        "pyperf",
        "command",
        "--quiet",
        "--output",
        str(output),
        "--processes",
        str(arguments.processes),
        "--values",
        str(arguments.values),
        "--warmups",
        str(arguments.warmups),
        "--loops",
        str(arguments.loops),
        "--min-time",
        str(arguments.min_time),
        "--name",
        arguments.name,
        *target,
    )
    return CaptureInvocation(argv, request.environment, ((output, "pyperf", "benchmark"),))


def _py_spy(request: CaptureBuildRequest, managed: ManagedExecutable) -> CaptureInvocation:
    arguments = cast(PySpyCaptureArguments, _arguments(request, PySpyCaptureArguments))
    output = request.directory / "profile.speedscope.json"
    options = ["--rate", str(arguments.rate)]
    if arguments.gil:
        options.append("--gil")
    if arguments.native:
        options.append("--native")
    if arguments.subprocesses:
        options.append("--subprocesses")
    argv = (
        managed("py-spy", "py-spy"),
        "record",
        "--format",
        "speedscope",
        "--output",
        str(output),
        *options,
        "--",
        *request.target_argv,
    )
    return CaptureInvocation(argv, request.environment, ((output, "py-spy", "cpu-profile"),))


def _memray(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    arguments = cast(MemrayCaptureArguments, _arguments(request, MemrayCaptureArguments))
    if len(request.target_argv) < 2:
        raise RuntimeFailure(
            "INVALID_INPUT",
            "memray capture requires a Python interpreter followed by a script, -m, or -c",
        )
    output = request.directory / "memory.bin"
    options = ["--output", str(output)]
    if arguments.native:
        options.append("--native")
    argv = (request.target_argv[0], "-m", "memray", "run", *options, *request.target_argv[1:])
    return CaptureInvocation(argv, request.environment, ((output, "memray", "memory"),))


def _node_cpu(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    _arguments(request, EmptyArguments)
    output = request.directory / "profile.cpuprofile"
    argv = (
        request.target_argv[0],
        "--cpu-prof",
        f"--cpu-prof-dir={request.directory}",
        f"--cpu-prof-name={output.name}",
        *request.target_argv[1:],
    )
    return CaptureInvocation(
        argv, request.environment, ((output, "cpuprofile", "cpu-profile"),), "workload"
    )


def _node_heap(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    _arguments(request, EmptyArguments)
    output = request.directory / "profile.heapprofile"
    argv = (
        request.target_argv[0],
        "--heap-prof",
        f"--heap-prof-dir={request.directory}",
        f"--heap-prof-name={output.name}",
        *request.target_argv[1:],
    )
    return CaptureInvocation(
        argv, request.environment, ((output, "heapprofile", "memory-profile"),), "workload"
    )


def _benchmark_samples(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    arguments = cast(
        BenchmarkSamplesCaptureArguments,
        _arguments(request, BenchmarkSamplesCaptureArguments),
    )
    output = request.directory / "benchmark-samples.json"
    environment = {**request.environment, "FLAMEOX_BENCHMARK_OUTPUT": str(output)}
    if arguments.torch_benchmark is not None:
        environment["FLAMEOX_TORCH_BENCHMARK_CONFIG"] = json.dumps(
            arguments.torch_benchmark.model_dump(mode="json"),
            separators=(",", ":"),
            sort_keys=True,
        )
    return CaptureInvocation(
        tuple(request.target_argv), environment, ((output, "samples", "benchmark"),), "workload"
    )


def _nvbench(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    _arguments(request, EmptyArguments)
    output_root = request.directory / "nvbench"
    output = output_root / "results.json"
    return CaptureInvocation(
        (*request.target_argv, "--jsonbin", str(output)),
        request.environment,
        ((output_root, "nvbench", "benchmark"),),
        "workload",
    )


def _torch_profiler(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    arguments = cast(
        TorchProfilerCaptureArguments, _arguments(request, TorchProfilerCaptureArguments)
    )
    output_root = request.directory / "torch-profiler"
    output = output_root / "torch-trace-cycle-0000.json"
    config = {
        "mode": "sdk",
        "activities": arguments.activities,
        "schedule": {
            "wait": arguments.wait,
            "warmup": arguments.warmup,
            "active": arguments.active,
            "repeat": 1,
            "skip_first": arguments.skip_first,
        },
        "record_shapes": arguments.record_shapes,
        "profile_memory": arguments.profile_memory,
        "with_stack": arguments.with_stack,
        "with_flops": arguments.with_flops,
        "with_modules": arguments.with_modules,
    }
    environment = {
        **request.environment,
        "FLAMEOX_TORCH_PROFILER_CONFIG": json.dumps(config, separators=(",", ":"), sort_keys=True),
        "FLAMEOX_TORCH_PROFILER_OUTPUT_ROOT": str(output_root),
    }
    return CaptureInvocation(
        tuple(request.target_argv), environment, ((output, "pytorch", "trace"),), "workload"
    )


def _compute_sanitizer(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    arguments = cast(
        ComputeSanitizerCaptureArguments,
        _arguments(request, ComputeSanitizerCaptureArguments),
    )
    output = request.directory / "compute-sanitizer.log"
    argv = (
        "compute-sanitizer",
        "--tool",
        arguments.tool,
        "--xml",
        "--save",
        str(output),
        "--error-exitcode",
        "99",
        *request.target_argv,
    )
    return CaptureInvocation(
        argv, request.environment, ((output, "compute-sanitizer", "sanitizer"),)
    )


def _perf(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    arguments = cast(PerfCaptureArguments, _arguments(request, PerfCaptureArguments))
    output = request.directory / "perf.data"
    argv = (
        "perf",
        "record",
        "--freq",
        str(arguments.frequency),
        "--call-graph",
        arguments.call_graph,
        "--output",
        str(output),
        "--",
        *request.target_argv,
    )
    return CaptureInvocation(argv, request.environment, ((output, "perf-data", "cpu-profile"),))


def _nsight_systems(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    arguments = cast(
        NsightSystemsCaptureArguments, _arguments(request, NsightSystemsCaptureArguments)
    )
    output_stem = request.directory / "nsight-systems"
    output = output_stem.with_suffix(".nsys-rep")
    argv = (
        "nsys",
        "profile",
        f"--trace={','.join(arguments.trace)}",
        "--sample=none",
        "--cpuctxsw=none",
        "--resolve-symbols=false",
        "--force-overwrite=true",
        "--output",
        str(output_stem),
        *request.target_argv,
    )
    return CaptureInvocation(argv, request.environment, ((output, "nsys-rep", "trace"),))


def _nsight_compute(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    arguments = cast(
        NsightComputeCaptureArguments, _arguments(request, NsightComputeCaptureArguments)
    )
    output = request.directory / "nsight-compute.ncu-rep"
    options = [
        "--export",
        str(output),
        "--force-overwrite",
        "--replay-mode",
        arguments.replay_mode,
        "--launch-skip",
        str(arguments.launch_skip),
        "--launch-count",
        str(arguments.launch_count),
    ]
    for section in arguments.section:
        options.extend(("--section", section))
    return CaptureInvocation(
        ("ncu", *options, *request.target_argv),
        request.environment,
        ((output, "nsight-compute", "kernel-metrics"),),
    )


def _rocprof(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    arguments = cast(RocprofCaptureArguments, _arguments(request, RocprofCaptureArguments))
    output_root = request.directory / "rocprof"
    output = output_root / "rocprofv3_results.pftrace"
    options = [
        flag
        for enabled, flag in (
            (arguments.hip_trace, "--hip-trace"),
            (arguments.kernel_trace, "--kernel-trace"),
            (arguments.memory_copy_trace, "--memory-copy-trace"),
            (arguments.memory_allocation_trace, "--memory-allocation-trace"),
            (arguments.scratch_memory_trace, "--scratch-memory-trace"),
            (arguments.marker_trace, "--marker-trace"),
        )
        if enabled
    ]
    argv = (
        "rocprofv3",
        "--output-format",
        "pftrace",
        "-o",
        "rocprofv3",
        "-d",
        str(output_root),
        *options,
        "--",
        *request.target_argv,
    )
    return CaptureInvocation(argv, request.environment, ((output, "rocprof-pftrace", "trace"),))


def _xctrace(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    arguments = cast(XctraceCaptureArguments, _arguments(request, XctraceCaptureArguments))
    output = request.directory / "capture.trace"
    argv = (
        "xcrun",
        "xctrace",
        "record",
        "--template",
        arguments.template,
        "--output",
        str(output),
        "--launch",
        "--",
        *request.target_argv,
    )
    return CaptureInvocation(argv, request.environment, ((output, "xctrace", "trace"),))


def _coverage(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    arguments = cast(CoverageCaptureArguments, _arguments(request, CoverageCaptureArguments))
    if len(request.target_argv) < 2:
        raise RuntimeFailure(
            "INVALID_INPUT",
            "coverage capture requires a Python interpreter followed by a script or -m",
        )
    output = request.directory / ".coverage"
    options = ["--data-file", str(output), "--rcfile", str(request.directory / "coverage.ini")]
    if arguments.branch:
        options.append("--branch")
    for name, values in (
        ("source", arguments.source),
        ("include", arguments.include),
        ("omit", arguments.omit),
    ):
        if values:
            options.extend((f"--{name}", ",".join(values)))
    argv = (
        request.target_argv[0],
        "-m",
        "coverage",
        "run",
        *options,
        *request.target_argv[1:],
    )
    return CaptureInvocation(argv, request.environment, ((output, "coverage", "coverage"),))


def _observations(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    _arguments(request, EmptyArguments)
    output = request.directory / "observations.jsonl"
    environment = {**request.environment, "FLAMEOX_OBSERVATIONS_PATH": str(output)}
    return CaptureInvocation(
        tuple(request.target_argv),
        environment,
        ((output, "observations", "observations"),),
        "workload",
    )


def _pytest(request: CaptureBuildRequest, _: ManagedExecutable) -> CaptureInvocation:
    _arguments(request, EmptyArguments)
    executable = Path(request.target_argv[0]).name
    if not executable.startswith("python") or request.target_argv[1:3] != ["-m", "pytest"]:
        raise RuntimeFailure("INVALID_INPUT", "pytest capture requires a python -m pytest argv")
    output = request.directory / "pytest.jsonl"
    runner = Path(__file__).parents[1] / "pytest_runner.py"
    argv = (
        request.target_argv[0],
        "-c",
        "import runpy,sys; runner=sys.argv.pop(1); runpy.run_path(runner,run_name='__main__')",
        str(runner.resolve()),
        "--output",
        str(output),
        "--",
        *request.target_argv[3:],
    )
    return CaptureInvocation(argv, request.environment, ((output, "pytest", "reliability"),))


CAPTURE_BUILDERS: dict[str, CaptureBuilder] = {
    "benchmark-samples": _benchmark_samples,
    "compute-sanitizer": _compute_sanitizer,
    "coverage": _coverage,
    "direct": _direct,
    "memray": _memray,
    "node-cpu-profile": _node_cpu,
    "node-heap-profile": _node_heap,
    "nsight-compute": _nsight_compute,
    "nsight-systems": _nsight_systems,
    "nvbench": _nvbench,
    "observations": _observations,
    "perf": _perf,
    "py-spy": _py_spy,
    "pyperf": _pyperf,
    "pytest": _pytest,
    "rocprofv3": _rocprof,
    "torch-profiler": _torch_profiler,
    "xctrace": _xctrace,
}


def build_capture_invocation(
    provider_id: str,
    target_argv: list[str],
    environment: dict[str, str],
    arguments: CaptureArguments,
    directory: Path,
    managed_executable: ManagedExecutable,
) -> CaptureInvocation:
    try:
        builder = CAPTURE_BUILDERS[provider_id]
    except KeyError as error:
        raise RuntimeFailure(
            "UNKNOWN_CAPABILITY", f"Unknown capture provider: {provider_id}"
        ) from error
    return builder(
        CaptureBuildRequest(target_argv, environment, arguments, directory), managed_executable
    )


def materialize_capture_support(provider_id: str, directory: Path) -> None:
    child_directory = {
        "nvbench": "nvbench",
        "torch-profiler": "torch-profiler",
        "rocprofv3": "rocprof",
    }.get(provider_id)
    if child_directory is not None:
        (directory / child_directory).mkdir()
    if provider_id == "coverage":
        (directory / "coverage.ini").write_text("[run]\n")
