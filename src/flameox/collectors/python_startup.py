from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

from flameox.atomic import atomic_write_bytes
from flameox.command_binding import ExecutableResolver
from flameox.execution import ExecutionOutcome, ExecutionRequest, SubprocessBroker
from flameox.startup_profile import PYTHON_STARTUP_PROFILE


def _handle_termination(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--import-trace-output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("workload", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.workload[:1] == ["--"]:
        arguments.workload = arguments.workload[1:]
    if not arguments.workload or arguments.timeout_seconds <= 0:
        parser.error("a workload and a positive timeout are required")
    if arguments.benchmark_output == arguments.import_trace_output:
        parser.error("benchmark and import-trace outputs must be distinct")
    return arguments


def _execute(command: tuple[str, ...], *, timeout_seconds: float) -> ExecutionOutcome:
    cwd = Path.cwd().resolve()
    return SubprocessBroker().run_sync(
        ExecutionRequest(
            argv=command,
            executable_binding=ExecutableResolver().require_host_tool(command[0], cwd=cwd),
            cwd=cwd,
            allowed_working_roots=(cwd,),
            environment_allowlist=tuple(os.environ),
            timeout_seconds=timeout_seconds,
        )
    )


def _exit_code(outcome: ExecutionOutcome) -> int:
    if outcome.process.exit_code is not None:
        return outcome.process.exit_code
    if outcome.process.terminating_signal is not None:
        return 128 + outcome.process.terminating_signal
    raise RuntimeError("child process did not report an exit status")


def _replay(outcome: ExecutionOutcome) -> None:
    if outcome.stdout:
        sys.stdout.buffer.write(outcome.stdout)
    if outcome.stderr:
        sys.stderr.buffer.write(outcome.stderr)


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Python startup capture exhausted its operation deadline")
    return remaining


def main() -> int:
    if os.name == "posix":
        signal.signal(signal.SIGTERM, _handle_termination)
    arguments = _arguments()
    workload = tuple(str(item) for item in arguments.workload)
    if not Path(workload[0]).name.startswith("python"):
        raise SystemExit("python-startup requires a Python script or module workload")

    deadline = time.monotonic() + arguments.timeout_seconds
    benchmark = _execute(
        PYTHON_STARTUP_PROFILE.pyperf_argv(
            python=sys.executable,
            output=arguments.benchmark_output,
            timeout_seconds=arguments.timeout_seconds,
            workload=workload,
        ),
        timeout_seconds=_remaining(deadline),
    )
    _replay(benchmark)
    benchmark_exit = _exit_code(benchmark)
    if benchmark_exit:
        # pyperf intentionally suppresses command output. Reproduce a failure once as
        # explicitly non-measured diagnostic evidence for the outer broker to preserve.
        diagnostic = _execute(workload, timeout_seconds=_remaining(deadline))
        _replay(diagnostic)
        return benchmark_exit

    import_command = (workload[0], "-X", "importtime", *workload[1:])
    import_trace = _execute(import_command, timeout_seconds=_remaining(deadline))
    atomic_write_bytes(arguments.import_trace_output, import_trace.stderr)
    if import_trace.stdout:
        sys.stdout.buffer.write(import_trace.stdout)
    non_import_stderr = b"\n".join(
        line for line in import_trace.stderr.splitlines() if not line.startswith(b"import time:")
    )
    if non_import_stderr:
        sys.stderr.buffer.write(non_import_stderr + b"\n")
    return _exit_code(import_trace)


if __name__ == "__main__":
    raise SystemExit(main())
