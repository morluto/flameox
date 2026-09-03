from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

from flameox import __version__
from flameox.cli import app
from flameox.setup import CliVersionAdvisory, ExternalRequirement, ProviderPreparation, SetupClient

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def isolated_data_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLAMEOX_DATA_DIR", str(tmp_path / "flameox-data"))


def test_help_exposes_current_command_families() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert all(
        command in result.output
        for command in (
            "setup",
            "analyze",
            "capture",
            "mcp",
            "evidence",
        )
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
        ],
    )
    assert inline.exit_code == 0, inline.output
    assert not (tmp_path / "flameox-data").exists()

    preserved = runner.invoke(
        app,
        [
            "analyze",
            "artifact.preview",
            str(artifact),
            "--preserve",
        ],
    )
    assert preserved.exit_code == 0, preserved.output
    assert json.loads(preserved.output)["preserved"]["evidence_id"]
    assert (tmp_path / "flameox-data" / "repository.json").is_file()


def test_analyze_consumes_continuation_from_a_previous_cli_invocation(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text(json.dumps([{"value": value} for value in range(104)]))
    runner = CliRunner()
    arguments = [
        "analyze",
        "artifact.preview",
        str(artifact),
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
            "--cwd",
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
            "--cwd",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
    )

    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["capture"]["executions"][0]["returncode"] == 7


@pytest.mark.process
def test_capture_accepts_the_runtime_experiment_contract(tmp_path: Path) -> None:
    experiment = {
        "cases": [
            {"name": "baseline"},
            {
                "name": "candidate",
                "argv": [sys.executable, "-c", "print('candidate')"],
            },
        ],
        "blocks": 2,
        "seed": 7,
        "metric": "wall_time_ns",
        "estimand": "median_difference",
        "practical_threshold": 0,
    }
    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "direct",
            "--cwd",
            str(tmp_path),
            "--experiment",
            json.dumps(experiment),
            "--",
            sys.executable,
            "-c",
            "print('baseline')",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    executions = payload["capture"]["executions"]
    assert len(executions) == 4
    assert {(item["case"], item["block"]) for item in executions} == {
        ("baseline", 1),
        ("candidate", 1),
        ("baseline", 2),
        ("candidate", 2),
    }
    assert payload["blocks"][-1]["rows"][0]["baseline_case"] == "baseline"


def test_capture_rejects_an_invalid_experiment_before_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "direct",
            "--cwd",
            str(tmp_path),
            "--experiment",
            '{"cases": []}',
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
    )

    assert result.exit_code == 1
    assert '"code": "INVALID_INPUT"' in result.stderr
    assert not marker.exists()


def test_analyze_projects_runtime_errors_without_traceback(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "unknown.capability",
            str(tmp_path / "missing.json"),
        ],
    )

    assert result.exit_code == 1
    assert '"code": "UNKNOWN_CAPABILITY"' in result.stderr
    assert "Traceback" not in result.output


def test_evidence_query_rejects_a_malformed_input_digest() -> None:
    result = CliRunner().invoke(app, ["evidence", "query", "--input-sha256", "not-a-digest"])

    assert result.exit_code == 1
    assert '"code": "INVALID_INPUT"' in result.stderr


def test_setup_without_a_tty_requires_an_explicit_client(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["setup", "--json"])

    assert result.exit_code == 2
    assert "No MCP client selected" in result.output
    assert "--client codex --yes" in " ".join(unstyle(result.output).split())
    assert not (tmp_path / "flameox-data").exists()


def test_setup_prepares_exact_python_providers_and_guides_system_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected: list[list[str]] = []
    monkeypatch.setenv("HOME", str(tmp_path))

    def prepare(providers: list[str], timeout_seconds: int) -> ProviderPreparation:
        assert timeout_seconds == 1_800
        selected.append(providers)
        return ProviderPreparation(
            providers,
            ["memray"],
            [
                ExternalRequirement(
                    "nsight-compute",
                    "Install NVIDIA Nsight Compute with its extras/python interface.",
                )
            ],
            ["/usr/bin/uvx", "--from", f"flameox[memory]=={__version__}", "--version"],
            "uvx",
            [
                "--python",
                "3.12",
                "--from",
                f"flameox[cpu,memory]=={__version__}",
                "flameox",
                "mcp",
                "serve",
            ],
        )

    monkeypatch.setattr("flameox.cli.prepare_providers", prepare)
    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--provider",
            "memray",
            "--provider",
            "nsight-compute",
            "--client",
            "codex",
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert selected == [["memray", "nsight-compute"]]
    assert payload["providers"] == ["memray", "nsight-compute"]
    assert payload["preparation_command"][0] == "/usr/bin/uvx"
    assert f"flameox[cpu,memory]=={__version__}" in payload["args"]
    assert "extras/python" in payload["external_guidance"][0]
    assert not (tmp_path / "flameox-data").exists()


def test_setup_prints_external_provider_guidance_for_humans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "flameox.cli.prepare_providers",
        lambda *_args, **_kwargs: ProviderPreparation(
            requested_providers=["perf"],
            launcher_command="uvx",
            launcher_args=["flameox", "mcp", "serve"],
            prepared_managed_providers=[],
            external_requirements=[ExternalRequirement("perf", "Install perf externally.")],
            preparation_command=[],
        ),
    )

    result = CliRunner().invoke(app, ["setup", "--client", "codex", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Install perf externally." in result.output


def test_setup_configures_explicit_global_clients_and_reports_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(app, ["setup", "--client", "codex", "--yes"])

    assert result.exit_code == 0, result.output
    config = tmp_path / ".codex" / "config.toml"
    assert config.is_file()
    assert f"flameox=={__version__}" in config.read_text()
    assert "Codex configured" in result.output
    assert "Restart or reconnect Codex" in result.output


def test_setup_json_returns_typed_reconnect_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(app, ["setup", "--client", "gemini", "--yes", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["restart_required"] is True
    assert payload["next_action"] == {
        "kind": "reconnect_mcp",
        "clients": ["Gemini CLI"],
        "message": "Restart or reconnect Gemini CLI to load Flameox.",
    }


def test_setup_reports_a_different_path_cli_without_changing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "flameox.cli.path_cli_version_advisory",
        lambda: CliVersionAdvisory("/tools/flameox", "0.1.0", __version__),
    )

    structured = CliRunner().invoke(app, ["setup", "--client", "codex", "--yes", "--json"])
    human = CliRunner().invoke(app, ["setup", "--client", "codex", "--yes"])

    assert structured.exit_code == 0, structured.output
    assert json.loads(structured.output)["advisories"] == [
        {
            "kind": "path_cli_version_mismatch",
            "executable": "/tools/flameox",
            "cli_version": "0.1.0",
            "mcp_version": __version__,
            "message": (
                f"Direct CLI commands use Flameox 0.1.0 at /tools/flameox, while the "
                f"configured MCP launcher uses {__version__}. Manage that CLI separately if "
                "you want the versions aligned."
            ),
        }
    ]
    assert "Direct CLI commands use Flameox 0.1.0" in human.output


def test_setup_dry_run_does_not_prepare_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "flameox.cli.prepare_providers",
        lambda *_args, **_kwargs: pytest.fail("dry-run prepared providers"),
    )

    result = CliRunner().invoke(
        app,
        ["setup", "--client", "cursor", "--provider", "memray", "--dry-run", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["plan"][0]["action"] == "create"
    assert f"flameox[memory]=={__version__}" in payload["args"]
    assert not (tmp_path / ".cursor").exists()


def test_setup_dry_run_validates_providers_and_reports_external_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    unknown = CliRunner().invoke(
        app, ["setup", "--client", "codex", "--provider", "mystery", "--dry-run"]
    )
    external = CliRunner().invoke(
        app, ["setup", "--client", "codex", "--provider", "perf", "--dry-run", "--json"]
    )

    assert unknown.exit_code == 2
    assert "Unknown provider" in unknown.output
    assert external.exit_code == 0, external.output
    assert json.loads(external.output)["external_guidance"]


def test_setup_yes_requires_explicit_client_selection() -> None:
    result = CliRunner().invoke(app, ["setup", "--yes"])

    assert result.exit_code == 2
    assert "detection is not consent" in result.output


@pytest.mark.parametrize("arguments", [["--dry-run"], ["--client", "codex", "--json"]])
def test_setup_automation_requires_explicit_selection_and_consent(arguments: list[str]) -> None:
    result = CliRunner().invoke(app, ["setup", *arguments])

    assert result.exit_code == 2
    assert "requires" in result.output


def test_setup_interactive_flow_asks_only_for_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("flameox.cli._is_interactive", lambda: True)
    selections: list[list[SetupClient]] = []

    def select(detected: list[SetupClient]) -> list[SetupClient]:
        selections.append(detected)
        return [SetupClient.CODEX]

    monkeypatch.setattr("flameox.cli._select_setup_clients", select)

    result = CliRunner().invoke(app, ["setup"])

    assert result.exit_code == 0, result.output
    assert selections == [[]]
    assert (tmp_path / ".codex" / "config.toml").is_file()


def test_setup_interactive_dry_run_uses_the_client_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("flameox.cli._is_interactive", lambda: True)
    monkeypatch.setattr("flameox.cli._select_setup_clients", lambda _detected: [SetupClient.CODEX])

    result = CliRunner().invoke(app, ["setup", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "No changes were made" in result.output
    assert not (tmp_path / ".codex").exists()


def test_setup_interactive_cancellation_makes_no_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("flameox.cli._is_interactive", lambda: True)
    monkeypatch.setattr("flameox.cli._select_setup_clients", lambda _detected: [])

    result = CliRunner().invoke(app, ["setup"])

    assert result.exit_code == 0, result.output
    assert "cancelled. No changes were made" in result.output
    assert not (tmp_path / ".codex").exists()
