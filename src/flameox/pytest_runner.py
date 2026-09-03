"""Execute pytest with the adjacent request-bound Flameox capture plugin."""

from __future__ import annotations

import argparse
import importlib
import os

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
    os.environ["FLAMEOX_PYTEST_OUTPUT"] = parsed.output
    existing_plugins = os.environ.get("PYTEST_PLUGINS")
    capture_plugin = "flameox.pytest_capture"
    os.environ["PYTEST_PLUGINS"] = (
        f"{existing_plugins},{capture_plugin}" if existing_plugins else capture_plugin
    )
    return int(pytest.main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
