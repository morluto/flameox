from __future__ import annotations

import json
import tomllib
from importlib.metadata import requires
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from flameox import __version__

pytestmark = pytest.mark.unit


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


def test_mcp_registry_metadata_launches_the_python_distribution() -> None:
    root = Path(__file__).parents[1]
    server = json.loads((root / "server.json").read_text())
    readme = (root / "README.md").read_text()

    assert server["name"] == "io.github.morluto/flameox"
    assert server["version"] == __version__
    assert server["repository"] == {
        "url": "https://github.com/morluto/flameox",
        "source": "github",
    }
    assert server["packages"] == [
        {
            "registryType": "pypi",
            "registryBaseUrl": "https://pypi.org",
            "identifier": "flameox",
            "version": __version__,
            "runtimeHint": "uvx",
            "transport": {"type": "stdio"},
            "packageArguments": [
                {"type": "positional", "value": "mcp"},
                {"type": "positional", "value": "serve"},
                {"type": "positional", "value": "--project-root"},
                {"type": "positional", "value": "."},
            ],
        }
    ]
    assert "<!-- mcp-name: io.github.morluto/flameox -->" in readme


def test_python_release_installs_pyperf_for_the_eager_adapter_import() -> None:
    requirements = [Requirement(value) for value in requires("flameox") or []]

    assert any(
        requirement.name == "pyperf" and requirement.marker is None for requirement in requirements
    )


def test_python_release_installs_duckdb_only_for_ephemeral_aggregation() -> None:
    requirements = [Requirement(value) for value in requires("flameox") or []]

    assert any(
        requirement.name == "duckdb" and requirement.marker is None for requirement in requirements
    )


def test_python_release_installs_duckdb_timezone_support() -> None:
    requirements = [Requirement(value) for value in requires("flameox") or []]

    assert any(
        requirement.name == "pytz" and requirement.marker is None for requirement in requirements
    )
