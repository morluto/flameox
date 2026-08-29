from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
from pathlib import Path


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


def _execute(
    command: tuple[str, ...],
    *,
    stdout: int | None = None,
    stderr: int | None = None,
) -> int:
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env=os.environ.copy(),
        stdout=stdout,
        stderr=stderr,
    )
    return process.wait()


def _exit_code(returncode: int) -> int:
    return returncode if returncode >= 0 else 128 - returncode


def main() -> int:
    if os.name == "posix":
        signal.signal(signal.SIGTERM, _handle_termination)
    arguments = _arguments()
    workload = tuple(str(item) for item in arguments.workload)
    if not Path(workload[0]).name.startswith("python"):
        raise SystemExit("python-startup requires a Python script or module workload")

    benchmark = _execute(
        (
            sys.executable,
            "-m",
            "pyperf",
            "command",
            "--output",
            str(arguments.benchmark_output),
            "--processes",
            "5",
            "--values",
            "1",
            "--loops",
            "1",
            "--warmups",
            "0",
            "--timeout",
            str(math.ceil(arguments.timeout_seconds)),
            "--copy-env",
            "--name",
            "flameox.python_startup.wall_time",
            "--",
            *workload,
        ),
    )
    benchmark_exit = _exit_code(benchmark)
    if benchmark_exit:
        # pyperf intentionally suppresses command output. Reproduce a failure once as
        # explicitly non-measured diagnostic evidence for the outer broker to preserve.
        _execute(workload)
        return benchmark_exit

    import_command = (workload[0], "-X", "importtime", *workload[1:])
    with arguments.import_trace_output.open("wb") as trace_stream:
        import_returncode = _execute(
            import_command,
            stderr=trace_stream.fileno(),
        )
    with arguments.import_trace_output.open("rb") as trace_stream:
        for line in trace_stream:
            if not line.startswith(b"import time:"):
                sys.stderr.buffer.write(line)
    return _exit_code(import_returncode)


if __name__ == "__main__":
    raise SystemExit(main())
