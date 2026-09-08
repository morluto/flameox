"""Execute pytest with the adjacent request-bound Flameox capture plugin."""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

pytest = importlib.import_module("pytest")


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
    # Local xdist workers inherit sys.path. Expose only the owned plugin, without
    # importing or replacing a Flameox installation in the workload interpreter.
    with TemporaryDirectory(prefix="pytest-plugin-", dir=Path(parsed.output).parent) as directory:
        shutil.copyfile(
            Path(__file__).with_name("pytest_capture.py"),
            Path(directory) / "_flameox_pytest_capture.py",
        )
        sys.path.insert(0, directory)
        try:
            return int(
                pytest.main(
                    ["-p", "_flameox_pytest_capture", "--flameox-output", parsed.output, *arguments]
                )
            )
        finally:
            sys.path.remove(directory)


if __name__ == "__main__":
    raise SystemExit(main())
