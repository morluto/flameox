from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

from flameox.command_binding import ExecutableResolver
from flameox.execution import ExecutionRequest, SubprocessBroker

_IMPORT_TIME = re.compile(
    r"^import time:\s+(?P<self>\d+)\s+\|\s+(?P<cumulative>\d+)\s+\|\s+(?P<module>.+)$"
)


def _handle_termination(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("workload", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.workload[:1] == ["--"]:
        arguments.workload = arguments.workload[1:]
    if not arguments.workload or arguments.samples < 1 or arguments.timeout_seconds <= 0:
        parser.error("a workload, at least one sample, and a positive timeout are required")
    return arguments


def _execute(
    command: tuple[str, ...],
    *,
    timeout_seconds: float,
) -> tuple[bytes, bytes, int | None, str, int, int]:
    cwd = Path.cwd().resolve()
    outcome = SubprocessBroker().run_sync(
        ExecutionRequest(
            argv=command,
            executable_binding=ExecutableResolver().require_host_tool(command[0], cwd=cwd),
            cwd=cwd,
            allowed_working_roots=(cwd,),
            environment_allowlist=tuple(os.environ),
            observation="child_peak_rss",
            timeout_seconds=timeout_seconds,
        )
    )
    if outcome.peak_rss_backend is None:
        raise RuntimeError("child peak-RSS observation did not report a backend")
    if outcome.process.wall_time_ns is None:
        raise RuntimeError("observed child process did not report a duration")
    if outcome.process.cleanup_complete is not True:
        raise RuntimeError("observed child process cleanup was incomplete")
    sample_exit_code = outcome.process.exit_code
    if sample_exit_code is None:
        if outcome.process.terminating_signal is None:
            raise RuntimeError("child process did not report an exit status")
        sample_exit_code = -outcome.process.terminating_signal
    return (
        outcome.stdout,
        outcome.stderr,
        outcome.process.peak_rss_bytes,
        outcome.peak_rss_backend,
        outcome.process.wall_time_ns,
        sample_exit_code,
    )


def _group_imports(stderr: str) -> tuple[str, list[dict[str, Any]], int]:
    raw_lines: list[str] = []
    modules: dict[str, tuple[int, int]] = {}
    ignored_lines = 0
    for line in stderr.splitlines():
        if not line.startswith("import time:"):
            continue
        raw_lines.append(line)
        match = _IMPORT_TIME.match(line)
        if match is None:
            if "self [us]" not in line:
                ignored_lines += 1
            continue
        module = match.group("module").strip()
        self_us = int(match.group("self"))
        cumulative_us = int(match.group("cumulative"))
        previous_self, previous_cumulative = modules.get(module, (0, 0))
        modules[module] = (
            previous_self + self_us,
            max(previous_cumulative, cumulative_us),
        )

    grouped: dict[str, dict[str, int]] = {}
    for module, (self_us, cumulative_us) in modules.items():
        normalized = module.lstrip(".")
        package = normalized.split(".", 1)[0] or module
        group = grouped.setdefault(
            package,
            {"module_count": 0, "self_us": 0, "max_cumulative_us": 0},
        )
        group["module_count"] += 1
        group["self_us"] += self_us
        group["max_cumulative_us"] = max(group["max_cumulative_us"], cumulative_us)
    return (
        "\n".join(raw_lines) + ("\n" if raw_lines else ""),
        [{"package": package, **values} for package, values in sorted(grouped.items())],
        ignored_lines,
    )


def main() -> int:
    if os.name == "posix":
        signal.signal(signal.SIGTERM, _handle_termination)
    arguments = _arguments()
    workload = tuple(str(item) for item in arguments.workload)
    python_name = Path(workload[0]).name
    if not python_name.startswith("python"):
        raise SystemExit("python-startup requires a Python script or module workload")

    started_at_ns = time.time_ns()
    samples: list[dict[str, Any]] = []
    exit_code = 0
    for index in range(arguments.samples):
        (
            stdout,
            stderr,
            peak_rss,
            peak_rss_backend,
            duration_ns,
            sample_exit_code,
        ) = _execute(workload, timeout_seconds=arguments.timeout_seconds)
        if stdout:
            sys.stdout.buffer.write(stdout)
        if stderr:
            sys.stderr.buffer.write(stderr)
        samples.append(
            {
                "index": index,
                "cache_semantics": (
                    "uncontrolled_initial" if index == 0 else "warm_process_restart"
                ),
                "fresh_interpreter": True,
                "duration_ns": duration_ns,
                "peak_rss_bytes": peak_rss,
                "peak_rss_backend": peak_rss_backend,
                "exit_code": sample_exit_code,
            }
        )
        if sample_exit_code:
            exit_code = sample_exit_code

    import_command = (workload[0], "-X", "importtime", *workload[1:])
    trace_stdout, trace_stderr_bytes, _, _, _, trace_exit_code = _execute(
        import_command,
        timeout_seconds=arguments.timeout_seconds,
    )
    trace_stderr = trace_stderr_bytes.decode("utf-8", errors="replace")
    raw_importtime, packages, ignored_lines = _group_imports(trace_stderr)
    non_import_stderr = "\n".join(
        line for line in trace_stderr.splitlines() if not line.startswith("import time:")
    )
    if trace_stdout:
        sys.stdout.buffer.write(trace_stdout)
    if non_import_stderr:
        sys.stderr.write(non_import_stderr + "\n")
    if trace_exit_code:
        exit_code = trace_exit_code

    payload = {
        "schema": "flameox.python-startup.v1",
        "started_at_ns": started_at_ns,
        "finished_at_ns": time.time_ns(),
        "python_executable": workload[0],
        "python_version": sys.version,
        "workload_argv": list(workload),
        "semantics": {
            "interpreter": "fresh process per wall sample and import trace",
            "initial_cache": "uncontrolled; no OS caches were dropped",
            "later_cache": "warm process restart; filesystem cache may have been populated",
            "wall_time": "uninstrumented workload execution",
            "import_trace": "separate instrumented workload execution",
        },
        "samples": samples,
        "import_trace": {
            "exit_code": trace_exit_code,
            "raw_importtime": raw_importtime,
            "packages": packages,
            "unparsed_importtime_lines": ignored_lines,
        },
    }
    arguments.output.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
