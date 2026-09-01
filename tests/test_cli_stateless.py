from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flameox import __version__
from flameox.cli import app
from flameox.setup import ProviderInstallation

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


def test_analyze_consumes_continuation_from_a_previous_cli_invocation(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text(json.dumps([{"value": value} for value in range(104)]))
    runner = CliRunner()
    arguments = [
        "analyze",
        "artifact.preview",
        str(artifact),
        "--project-root",
        str(tmp_path),
    ]

    first = runner.invoke(app, arguments)
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.output)

    second = runner.invoke(app, [*arguments, "--continuation", first_payload["continuation"]])
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.output)

    assert [row["value"] for row in first_payload["blocks"][1]["rows"]] == list(range(100))
    assert [row["value"] for row in second_payload["blocks"][1]["rows"]] == list(range(100, 104))


def test_analyze_rejects_a_malformed_continuation_without_a_traceback(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")

    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "artifact.preview",
            str(artifact),
            "--continuation",
            "not-a-token",
            "--project-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert '"code": "INVALID_INPUT"' in result.stderr
    assert "Traceback" not in result.output


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


@pytest.mark.process
def test_capture_returns_nonzero_for_failed_target(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "direct",
            "--project-root",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
    )

    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["capture"]["executions"][0]["returncode"] == 7


def test_analyze_projects_runtime_errors_without_traceback(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "unknown.capability",
            str(tmp_path / "missing.json"),
            "--project-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert '"code": "UNKNOWN_CAPABILITY"' in result.stderr
    assert "Traceback" not in result.output


def test_setup_prints_configuration_without_creating_repository(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["setup", "--project-root", str(tmp_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["command"] == "uvx"
    assert payload["args"] == [
        "--python",
        "3.12",
        "--from",
        f"flameox=={__version__}",
        "flameox",
        "mcp",
        "serve",
        "--project-root",
        str(tmp_path),
    ]
    assert payload["resolved_version"] == __version__
    assert payload["client_registration_changed"] is False
    assert payload["repository_created"] is False
    assert not (tmp_path / ".flameox").exists()


def test_setup_installs_only_explicit_python_providers_and_guides_system_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected: list[list[str]] = []

    def install(providers: list[str]) -> ProviderInstallation:
        selected.append(providers)
        return ProviderInstallation(
            ["uv", "tool", "install", f"flameox[memory]=={__version__}"],
            ["memray", "nsight-compute", "py-spy"],
        )

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
    assert payload["providers"] == ["memray", "nsight-compute", "py-spy"]
    assert f"flameox[cpu,memory]=={__version__}" in payload["args"]
    assert "extras/python" in payload["external_guidance"][0]
    assert not (tmp_path / ".flameox").exists()
