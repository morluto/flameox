from __future__ import annotations

import json
from importlib.metadata import requires
from pathlib import Path

from packaging.requirements import Requirement

from flameox import __version__


def test_npm_bootstrap_matches_python_release_version() -> None:
    package = json.loads((Path(__file__).parents[1] / "npm" / "package.json").read_text())

    assert package["version"] == __version__


def test_python_release_installs_pyperf_for_the_eager_adapter_import() -> None:
    requirements = [Requirement(value) for value in requires("flameox") or []]

    assert any(
        requirement.name == "pyperf" and requirement.marker is None for requirement in requirements
    )
