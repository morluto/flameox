"""Execute pytest with the adjacent request-bound Flameox capture plugin."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

pytest = importlib.import_module("pytest")


def _capture_plugin() -> ModuleType:
    path = Path(__file__).with_name("pytest_capture.py")
    spec = importlib.util.spec_from_file_location("_flameox_pytest_capture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Flameox pytest capture plugin could not be loaded")
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    return plugin


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("pytest_arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    arguments = (
        parsed.pytest_arguments[1:]
        if parsed.pytest_arguments[:1] == ["--"]
        else parsed.pytest_arguments
    )
    os.environ["FLAMEOX_PYTEST_OUTPUT"] = parsed.output
    return int(pytest.main(arguments, plugins=[_capture_plugin()]))


if __name__ == "__main__":
    raise SystemExit(main())
