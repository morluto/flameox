from __future__ import annotations

import json
import tomllib
from importlib.metadata import requires
from pathlib import Path

from packaging.requirements import Requirement

from flameox import __version__


def test_npm_bootstrap_matches_python_release_version() -> None:
    root = Path(__file__).parents[1]
    package = json.loads((root / "npm" / "package.json").read_text())
    package_lock = json.loads((root / "npm" / "package-lock.json").read_text())
    uv_lock = tomllib.loads((root / "uv.lock").read_text())
    locked_python = next(package for package in uv_lock["package"] if package["name"] == "flameox")

    assert package["version"] == __version__
    assert package_lock["version"] == __version__
    assert package_lock["packages"][""]["version"] == __version__
    assert locked_python["version"] == __version__


def test_python_release_installs_pyperf_for_the_eager_adapter_import() -> None:
    requirements = [Requirement(value) for value in requires("flameox") or []]

    assert any(
        requirement.name == "pyperf" and requirement.marker is None for requirement in requirements
    )
