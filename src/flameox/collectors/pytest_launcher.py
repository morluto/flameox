from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("workload", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    workload = arguments.workload[1:] if arguments.workload[:1] == ["--"] else arguments.workload
    if not workload:
        parser.error("a pytest workload is required")

    executable = Path(workload[0]).name
    if executable.startswith("python"):
        if len(workload) < 3 or workload[1:3] != ["-m", "pytest"]:
            parser.error("Python workloads must invoke `python -m pytest`")
        command = [*workload[:3], "-p", "flameox.collectors.pytest_plugin", *workload[3:]]
    elif executable.startswith("pytest"):
        command = [workload[0], "-p", "flameox.collectors.pytest_plugin", *workload[1:]]
    else:
        parser.error("the pytest adapter requires `pytest` or `python -m pytest`")

    environment = os.environ.copy()
    environment["FLAMEOX_PYTEST_EVIDENCE_PATH"] = str(arguments.output)
    environment["FLAMEOX_PYTEST_RUN_STARTED_NS"] = str(time.time_ns())
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
