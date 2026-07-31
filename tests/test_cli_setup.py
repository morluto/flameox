from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flameox.cli import app


def test_setup_dry_run_emits_machine_readable_exact_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLAMEOX_SETUP_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FLAMEOX_SETUP_DATA_ROOT", str(tmp_path / "data"))

    result = CliRunner().invoke(app, ["setup", "--claude", "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["operation"] == "configure"
    assert payload["clients"][0]["client"] == "claude"
    assert payload["clients"][0]["action"] == "create"
    assert not (tmp_path / "home" / ".claude.json").exists()


def test_setup_yes_requires_an_explicit_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLAMEOX_SETUP_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FLAMEOX_SETUP_DATA_ROOT", str(tmp_path / "data"))

    result = CliRunner().invoke(app, ["setup", "--yes"])

    assert result.exit_code == 9
    assert "--yes requires explicit clients" in result.output


def test_setup_verify_dry_run_reports_configured_launcher_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    data = tmp_path / "data"
    home.mkdir()
    data.mkdir()
    executable = data / "runtimes" / "0.1.0" / "bin" / "flameox"
    (home / ".claude.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "flameox": {
                        "command": str(executable),
                        "args": ["mcp", "serve", "--project-root", "."],
                    }
                }
            }
        )
        + "\n"
    )
    (data / "install.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active_version": "0.1.0",
                "executable": str(executable),
            }
        )
        + "\n"
    )
    monkeypatch.setenv("FLAMEOX_SETUP_HOME", str(home))
    monkeypatch.setenv("FLAMEOX_SETUP_DATA_ROOT", str(data))

    result = CliRunner().invoke(app, ["setup", "--verify", "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["operation"] == "verify"
    assert payload["clients"] == [
        {
            "client": "claude",
            "display_name": "Claude Code",
            "path": str(home / ".claude.json"),
            "action": "already_current",
            "detected": True,
        }
    ]


def test_setup_reports_corrupt_install_metadata_as_a_domain_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "install.json").write_text("{broken")
    monkeypatch.setenv("FLAMEOX_SETUP_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FLAMEOX_SETUP_DATA_ROOT", str(data))

    result = CliRunner().invoke(app, ["setup", "--verify", "--yes"])

    assert result.exit_code == 5
    assert "ARTIFACT_INTEGRITY_FAILED" in result.output
    assert "Traceback" not in result.output
    assert result.exception is not None


def test_npm_bootstrap_prints_runtime_to_wizard_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLAMEOX_NPM_BOOTSTRAP", "1")
    result = CliRunner().invoke(app, ["setup", "--claude", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Managed runtime ready. Starting flameox setup..." in result.output
    assert result.output.index("Managed runtime ready") < result.output.index("flameox setup")
