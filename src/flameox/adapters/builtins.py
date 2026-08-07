from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from flameox.adapters.torch_profiler import torch_profiler_options
from flameox.domain import ArtifactKind, DomainError, ErrorCode

DependencyKind = Literal["internal", "executable", "package"]


@dataclass(frozen=True, slots=True)
class BuiltinAdapter:
    name: str
    dependency_kind: DependencyKind
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
    managed_extra: Literal["cpu", "execution", "memory", "test", "trace", "torch"] | None = None
    managed_requirement: str | None = None


@dataclass(frozen=True, slots=True)
class CaptureInvocation:
    argv: tuple[str, ...]
    artifact_kinds: tuple[ArtifactKind, ...]
    expected_overhead: str
    limitations: tuple[str, ...]
    environment: dict[str, str]


BUILTIN_ADAPTERS = {
    adapter.name: adapter
    for adapter in (
        BuiltinAdapter(
            name="command",
            dependency_kind="internal",
            dependency=None,
            supported_modes=("named_workload",),
            supported_formats=("process-output",),
            output_filename="capture.bin",
            artifact_kinds=(ArtifactKind.PROCESS_OUTPUT,),
            expected_overhead="No profiler overhead; process output only.",
            capture_limitations=("No sampled stack or operator evidence is collected.",),
        ),
        BuiltinAdapter(
            name="py-spy",
            dependency_kind="executable",
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
            managed_extra="cpu",
            managed_requirement="py-spy>=0.4.2,<0.5",
        ),
        BuiltinAdapter(
            name="perf",
            dependency_kind="executable",
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
            dependency_kind="executable",
            dependency="trace_processor_shell",
            supported_modes=("import", "query"),
            supported_formats=("perfetto", "chrome-trace", "pprof", "perf.data"),
            features=("trace_sql", "temporal_slices"),
            remediation=(
                "Call start_capability_setup with adapter='perfetto' to stage the user-space "
                "Trace Processor, or configure analysis.trace_processor_path explicitly.",
            ),
            version_args=("--version",),
            managed_extra="trace",
            managed_requirement="perfetto>=0.57,<0.58",
        ),
        BuiltinAdapter(
            name="pyperf",
            dependency_kind="package",
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
                "Experiment-level treatment randomization is separate from "
                "pyperf's worker hierarchy.",
            ),
        ),
        BuiltinAdapter(
            name="python-startup",
            dependency_kind="internal",
            dependency=None,
            supported_modes=("repeated_process",),
            supported_formats=("flameox-python-startup-json", "python-importtime"),
            features=("wall_time", "import_cost", "module_count", "peak_rss"),
            output_filename="python-startup.json",
            artifact_kinds=(ArtifactKind.PYTHON_STARTUP,),
            expected_overhead=(
                "Fresh interpreter launches with import timing enabled; target output is "
                "captured by the launcher and replayed after each sample."
            ),
            capture_limitations=(
                "The initial OS file-cache state is observed but not controlled; flameox does "
                "not drop caches.",
                "Import timing instruments imports and therefore adds measurement overhead.",
            ),
        ),
        BuiltinAdapter(
            name="pytest",
            dependency_kind="package",
            dependency="pytest",
            supported_modes=("serial", "xdist"),
            supported_formats=("flameox-pytest-events-jsonl",),
            features=(
                "test_phases",
                "fixture_setup",
                "failure_latency",
                "worker_lifecycle",
            ),
            output_filename="pytest-events.jsonl",
            artifact_kinds=(ArtifactKind.TEST_EXECUTION,),
            expected_overhead=(
                "Pytest hook timestamps and JSONL event recording; fixture setup events are "
                "written to bounded per-worker sidecars under xdist."
            ),
            capture_limitations=(
                "Public xdist hooks expose scheduler strategy, worker lifecycle, and execution "
                "start, but not an exact per-test controller queue timestamp.",
                "If the entire pytest controller is forcibly terminated, sidecars not yet "
                "recovered into the primary artifact may be unavailable.",
            ),
            preserve_artifact_on_nonzero=True,
            managed_extra="test",
            managed_requirement="pytest>=8.3",
        ),
        BuiltinAdapter(
            name="torch.profiler",
            dependency_kind="package",
            dependency="torch",
            supported_modes=("trace_import", "whole_entrypoint", "sdk"),
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
                "Operator tracing with shapes, memory, and Python stacks has substantial overhead."
            ),
            capture_limitations=(
                "Evidence coverage and overhead depend on the feature flags bound into the "
                "capture plan.",
                "Scheduled SDK mode requires explicit workload step boundaries.",
            ),
            managed_extra="torch",
            managed_requirement="torch>=2.7",
        ),
        BuiltinAdapter(
            name="memray",
            dependency_kind="package",
            dependency="memray",
            supported_modes=("import", "run"),
            supported_formats=("memray",),
            features=("allocations", "retained_memory", "stacks"),
            output_filename="memory.bin",
            managed_extra="memory",
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
            name="coverage",
            dependency_kind="package",
            dependency="coverage",
            supported_modes=("import", "run"),
            supported_formats=("coverage-data",),
            features=("lines", "branches"),
            output_filename=".coverage",
            managed_extra="execution",
            managed_requirement="coverage>=7.14,<8",
            artifact_kinds=(ArtifactKind.EXECUTION_COVERAGE,),
            expected_overhead="Tracing overhead; branch collection is enabled.",
            capture_limitations=("Coverage records execution, not values or control-flow causes.",),
        ),
    )
}


def builtin_adapter(name: str) -> BuiltinAdapter | None:
    return BUILTIN_ADAPTERS.get(name)


def build_capture_invocation(
    adapter_name: str,
    workload_argv: tuple[str, ...],
    output_root: Path,
    *,
    executable: str | None,
    timeout_seconds: float = 300,
    options: dict[str, object] | None = None,
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
    if options and adapter_name != "torch.profiler":
        raise DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            f"Adapter {adapter_name!r} does not accept capture options.",
        )
    output = str(output_root / adapter.output_filename)
    environment: dict[str, str] = {}
    limitations = adapter.capture_limitations
    if adapter_name == "command":
        argv = workload_argv
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
    elif adapter_name in {"coverage", "memray"}:
        python, target = _python_target(workload_argv)
        if adapter_name == "coverage":
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
            argv = (
                python,
                "-m",
                "memray",
                "run",
                "--output",
                output,
                *target,
            )
    elif adapter_name == "torch.profiler":
        return _torch_capture_invocation(
            adapter,
            workload_argv,
            output_root,
            output,
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
        argv = (
            python,
            "-m",
            "flameox.collectors.python_startup",
            "--output",
            output,
            "--samples",
            "5",
            "--timeout-seconds",
            str(timeout_seconds),
            "--",
            *workload_argv,
        )
    elif adapter_name == "pytest":
        argv = (
            _python_executable_for_launcher(workload_argv),
            "-m",
            "flameox.collectors.pytest_launcher",
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
    config["expected_cycles"] = selected.expected_cycles
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
    expected_overhead = (
        "Operator tracing with enabled expensive features: " + ", ".join(expensive_features) + "."
        if expensive_features
        else "Operator tracing without shapes, memory, stacks, FLOPs, or modules."
    )
    if selected.mode == "sdk":
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
        launcher_target = ("--module", target[1], *target[2:])
    else:
        launcher_target = ("--script", target[0], *target[1:])
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
            "Whole-entrypoint mode cannot distinguish application-specific warm-up and "
            "steady-state phases.",
            *feature_limitations,
        ),
        environment={},
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
