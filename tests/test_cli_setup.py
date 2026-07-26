from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flamo.cli import app


def test_setup_dry_run_emits_machine_readable_exact_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLAMO_SETUP_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FLAMO_SETUP_DATA_ROOT", str(tmp_path / "data"))

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
    monkeypatch.setenv("FLAMO_SETUP_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FLAMO_SETUP_DATA_ROOT", str(tmp_path / "data"))

    result = CliRunner().invoke(app, ["setup", "--yes"])

    assert result.exit_code == 9
    assert "--yes requires explicit clients" in result.output


def test_setup_reports_corrupt_install_metadata_as_a_domain_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "install.json").write_text("{broken")
    monkeypatch.setenv("FLAMO_SETUP_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FLAMO_SETUP_DATA_ROOT", str(data))

    result = CliRunner().invoke(app, ["setup", "--verify", "--yes"])

    assert result.exit_code == 5
    assert "ARTIFACT_INTEGRITY_FAILED" in result.output
    assert "Traceback" not in result.output
    assert result.exception is not None
