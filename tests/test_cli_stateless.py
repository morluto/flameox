from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flameox.cli import app

pytestmark = pytest.mark.integration


def test_help_exposes_only_stateless_command_families() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert all(
        command in result.output
        for command in ("setup", "analyze", "capture", "capabilities", "mcp", "evidence")
    )
    assert all(
        removed not in result.output
        for removed in ("workspace", "catalog", "runs", "investigations", "detached")
    )


def test_analyze_preserves_only_when_requested(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1}]')
    runner = CliRunner()

    inline = runner.invoke(
        app,
        [
            "analyze",
            "artifact.preview",
            str(artifact),
            "--project-root",
            str(tmp_path),
        ],
    )
    assert inline.exit_code == 0, inline.output
    assert not (tmp_path / ".flameox").exists()

    preserved = runner.invoke(
        app,
        [
            "analyze",
            "artifact.preview",
            str(artifact),
            "--preserve",
            "--project-root",
            str(tmp_path),
        ],
    )
    assert preserved.exit_code == 0, preserved.output
    assert json.loads(preserved.output)["preserved"]["evidence_id"]
    assert (tmp_path / ".flameox" / "repository.json").is_file()


@pytest.mark.process
def test_capture_accepts_argv_after_separator(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "direct",
            "--capture-arguments",
            "{}",
            "--project-root",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "print('cli-capture')",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["blocks"][1]["rows"][0]["text"] == "cli-capture"
    assert not (tmp_path / ".flameox").exists()


def test_setup_prints_configuration_without_creating_repository(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["setup", "--project-root", str(tmp_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["args"] == ["mcp", "serve", "--project-root", str(tmp_path)]
    assert payload["repository_created"] is False
    assert not (tmp_path / ".flameox").exists()


def test_setup_installs_only_explicit_python_providers_and_guides_system_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected: list[list[str]] = []

    def install(providers: list[str]) -> list[str]:
        selected.append(providers)
        return ["uv", "tool", "install", "flameox[memory]==0.2.0"]

    monkeypatch.setattr("flameox.cli.install_providers", install)
    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--project-root",
            str(tmp_path),
            "--provider",
            "memray",
            "--provider",
            "nsight-compute",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert selected == [["memray", "nsight-compute"]]
    assert payload["providers"] == ["memray", "nsight-compute"]
    assert "extras/python" in payload["external_guidance"][0]
    assert not (tmp_path / ".flameox").exists()
