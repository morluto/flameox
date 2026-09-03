"""Execute pytest with the adjacent request-bound Flameox capture plugin."""

from __future__ import annotations

import argparse
import importlib

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
    return int(
        pytest.main(
            ["-p", "flameox.pytest_capture", "--flameox-output", parsed.output, *arguments],
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
