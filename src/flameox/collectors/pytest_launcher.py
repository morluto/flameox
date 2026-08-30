from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import types
from pathlib import Path

import pytest

_PLUGIN_SOURCE = "__FLAMEOX_PLUGIN_SOURCE__"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("workload", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    workload = parsed.workload[1:] if parsed.workload[:1] == ["--"] else parsed.workload
    if not workload:
        parser.error("a pytest workload is required")

    executable = os.path.basename(workload[0])
    if executable.startswith("python"):
        if len(workload) < 3 or workload[1:3] != ["-m", "pytest"]:
            parser.error("Python workloads must invoke `python -m pytest`")
        pytest_arguments = workload[3:]
    elif executable.startswith("pytest"):
        pytest_arguments = workload[1:]
    else:
        parser.error("the pytest adapter requires `pytest` or `python -m pytest`")

    os.environ["FLAMEOX_PYTEST_RUN_STARTED_NS"] = str(time.time_ns())
    os.environ["PYTHONSAFEPATH"] = "1"
    plugin_name = (
        "_flameox_bound_pytest_" + hashlib.sha256(_PLUGIN_SOURCE.encode()).hexdigest()[:16]
    )
    plugin = types.ModuleType(plugin_name)
    exec(compile(_PLUGIN_SOURCE, "<flameox-pytest-plugin>", "exec"), plugin.__dict__)
    sys.modules[plugin.__name__] = plugin
    plugin_path = Path(parsed.output).parent / f"{plugin.__name__}.py"
    plugin_path.write_text(_PLUGIN_SOURCE, encoding="utf-8")
    plugin_path.chmod(0o400)
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        str(plugin_path.parent)
        if not existing_pythonpath
        else os.pathsep.join((str(plugin_path.parent), existing_pythonpath))
    )
    sys.path.insert(0, str(plugin_path.parent))
    try:
        return pytest.main(
            [
                "-p",
                plugin.__name__,
                *pytest_arguments,
                f"--report-log={parsed.output}",
            ],
            plugins=[plugin],
        )
    finally:
        plugin_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
