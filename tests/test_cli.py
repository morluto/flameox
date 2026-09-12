from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click import unstyle
from typer.testing import CliRunner

from flameox import __version__
from flameox.cli import app
from flameox.repository import EvidenceRepository
from flameox.runtime_contracts import RequestLimits
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


def test_mcp_inspect_supports_compact_discovery_and_exact_tool_drill_down() -> None:
    runner = CliRunner()
    summary_result = runner.invoke(app, ["mcp", "inspect"])
    exact_result = runner.invoke(app, ["mcp", "inspect", "--tool", "capture_and_analyze"])

    assert summary_result.exit_code == 0, summary_result.output
    summary = json.loads(summary_result.output)
    assert summary["tool_count"] == 6
    assert all(
        "input_schema" not in tool and "output_schema" not in tool for tool in summary["tools"]
    )
    hotspot_capability = next(
        item for item in summary["capabilities"] if item["id"] == "cpu.hotspots"
    )
    assert hotspot_capability["capture_providers"] == ["py-spy", "perf", "node-cpu-profile"]
    capture = next(tool for tool in summary["tools"] if tool["name"] == "capture_and_analyze")
    assert capture["annotations"]["destructive_hint"] is True
    assert capture["required_inputs"] == ["request"]

    assert exact_result.exit_code == 0, exact_result.output
    exact = json.loads(exact_result.output)
    assert [tool["name"] for tool in exact["tools"]] == ["capture_and_analyze"]
    request = exact["tools"][0]["input_schema"]["properties"]["request"]
    assert request["discriminator"]["propertyName"] == "capability_id"


def test_mcp_inspect_unknown_tool_returns_bounded_recovery() -> None:
    result = CliRunner().invoke(app, ["mcp", "inspect", "--tool", "missing_tool"])

    assert result.exit_code == 1
    failure = json.loads(result.stderr)
    assert failure["code"] == "UNKNOWN_CAPABILITY"
    assert failure["details"]["requested_tool"] == "missing_tool"
    assert "capture_and_analyze" in failure["details"]["available_tools"]
    assert failure["details"]["recovery"] == "Run `flameox mcp inspect` to select a tool."


def test_mcp_inspect_rejects_conflicting_detail_modes() -> None:
    result = CliRunner().invoke(
        app,
        ["mcp", "inspect", "--tool", "capture_and_analyze", "--full"],
    )

    assert result.exit_code == 1
    assert "either --tool" in result.stderr


def test_mcp_startup_limits_reach_the_shared_server(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[RequestLimits | None] = []

    def serve(*, limits: RequestLimits | None = None) -> None:
        received.append(limits)

    monkeypatch.setattr("flameox.cli.run_server", serve)
    result = CliRunner().invoke(
        app, ["mcp", "serve", "--limits", '{"max_memory_bytes":8589934592}']
    )
    assert result.exit_code == 0, result.output
    assert received == [RequestLimits(max_memory_bytes=8 * 1024**3)]


@pytest.mark.parametrize("value", ["[]", "not-json", '{"max_memory_bytes":0}', '{"typo":1}'])
def test_mcp_startup_rejects_invalid_limits(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*, limits: RequestLimits | None = None) -> None:
        pytest.fail("invalid limits must not start the server")

    monkeypatch.setattr("flameox.cli.run_server", unexpected)
    result = CliRunner().invoke(app, ["mcp", "serve", "--limits", value])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_analyze_enforces_cli_startup_input_limit(tmp_path: Path) -> None:
    artifact = tmp_path / "oversized.txt"
    artifact.write_text("x" * 2048)
    result = CliRunner().invoke(
        app, ["analyze", "artifact.preview", str(artifact), "--limits", '{"max_input_bytes":1024}']
    )
    assert result.exit_code == 1
    assert json.loads(result.stderr)["code"] == "LIMIT_EXCEEDED"


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

    assert "analysis_id" not in first_payload
    assert first_payload["next_page"]["command"] == "flameox"
    second = runner.invoke(app, first_payload["next_page"]["argv"])
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.output)

    assert [row["value"] for row in first_payload["blocks"][1]["rows"]] == list(range(100))
    assert [row["value"] for row in second_payload["blocks"][1]["rows"]] == list(range(100, 104))


def test_analyze_reads_ordered_sources_from_preserved_evidence(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value": 1}, {"value": 2}]')
    runner = CliRunner()
    first = runner.invoke(
        app,
        [
            "analyze",
            "artifact.preview",
            str(artifact),
            "--limits",
            '{"max_rows":1}',
            "--preserve",
        ],
    )
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.output)

    next_page = first_payload["next_page"]
    assert first_payload["preserved"]["evidence_id"] in next_page["argv"]
    second = runner.invoke(app, next_page["argv"])

    assert second.exit_code == 0, second.output
    assert json.loads(second.output)["blocks"][1]["rows"][0]["value"] == 2


def test_analyze_requires_exactly_one_source_mode(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runner = CliRunner()
    missing = runner.invoke(app, ["analyze", "artifact.preview"])
    conflicting = runner.invoke(
        app,
        [
            "analyze",
            "artifact.preview",
            str(artifact),
            "--evidence",
            "a" * 64,
        ],
    )

    assert missing.exit_code == 1
    assert "Provide artifact paths or --evidence" in missing.stderr
    assert conflicting.exit_code == 1
    assert "either artifact paths or --evidence" in conflicting.stderr


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
def test_capture_does_not_offer_an_unusable_scratch_continuation(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "direct",
            "--cwd",
            str(tmp_path),
            "--limits",
            '{"max_rows":1}',
            "--",
            sys.executable,
            "-c",
            "print('first'); print('second')",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["coverage"]["complete"] is False
    assert payload["continuation"] is None
    assert any("--preserve or --rescue-to" in item for item in payload["limitations"])


@pytest.mark.process
def test_preserved_capture_returns_an_executable_cross_process_next_page(tmp_path: Path) -> None:
    runner = CliRunner()
    first = runner.invoke(
        app,
        [
            "capture",
            "--provider",
            "direct",
            "--cwd",
            str(tmp_path),
            "--limits",
            '{"max_rows":1}',
            "--preserve",
            "--",
            sys.executable,
            "-c",
            "print('first'); print('second')",
        ],
    )
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.output)
    next_page = first_payload["next_page"]
    assert next_page["command"] == "flameox"
    assert first_payload["preserved"]["evidence_id"] in next_page["argv"]
    assert "analysis_id" not in first_payload

    second = runner.invoke(app, next_page["argv"])

    assert second.exit_code == 0, second.output
    assert json.loads(second.output)["blocks"][1]["rows"][0]["text"] == "second"


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
@pytest.mark.parametrize("retention", ["diagnostics", "full"])
def test_capture_console_retention_option_reaches_native_capture(
    tmp_path: Path, retention: str
) -> None:
    workload = tmp_path / "workload.py"
    workload.write_text("print('console-evidence')\n")
    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "coverage",
            "--capability",
            "coverage.summary",
            "--cwd",
            str(tmp_path),
            "--console-output",
            retention,
            "--",
            sys.executable,
            str(workload),
        ],
    )
    assert result.exit_code == 0, result.output
    execution = json.loads(result.stdout)["capture"]["executions"][0]
    if retention == "diagnostics":
        assert execution["console_diagnostics"]["stdout"] == "console-evidence\n"
        assert execution.get("output_streams") is None
    else:
        assert execution["output_streams"]["stdout_bytes"] == len(b"console-evidence\n")
        assert execution.get("console_diagnostics") is None


def test_capture_rejects_unknown_console_retention_before_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "direct",
            "--cwd",
            str(tmp_path),
            "--console-output",
            "automatic",
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
    )
    assert result.exit_code != 0
    assert not marker.exists()
    assert "Traceback" not in result.output


@pytest.mark.process
def test_capture_workload_budget_is_independent_of_decoder_limits(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "direct",
            "--cwd",
            str(tmp_path),
            "--limits",
            '{"timeout_seconds":0.01}',
            "--workload-budget",
            '{"timeout_seconds":0.2}',
            "--",
            sys.executable,
            "-c",
            "import time; print('started', flush=True); time.sleep(5)",
        ],
    )
    assert result.exit_code == 1, result.output
    execution = json.loads(result.stdout)["capture"]["executions"][0]
    assert execution["limit"]["kind"] == "timeout"
    assert execution["limit"]["configured"] == 0.2


@pytest.mark.parametrize(
    "budget", ["[]", '{"timeout_seconds":0}', '{"max_memory_bytes":-1}', '{"unknown":1}']
)
def test_capture_rejects_invalid_workload_budget_before_execution(
    tmp_path: Path, budget: str
) -> None:
    marker = tmp_path / "executed"
    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "direct",
            "--cwd",
            str(tmp_path),
            "--workload-budget",
            budget,
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
    )
    assert result.exit_code != 0
    assert not marker.exists()
    assert "Traceback" not in result.output


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
    failure = json.loads(result.stderr)
    assert failure["details"]["requested_capability"] == "unknown.capability"
    assert "cpu.hotspots" in failure["details"]["available_capabilities"]
    assert "flameox mcp inspect" in failure["details"]["recovery"]
    assert "Traceback" not in result.output


def test_capture_unknown_provider_returns_choices_before_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "missing-provider",
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
    )

    assert result.exit_code == 1
    failure = json.loads(result.stderr)
    assert failure["details"]["requested_provider"] == "missing-provider"
    assert "direct" in failure["details"]["available_capture_providers"]
    assert failure["details"]["recovery"].endswith(
        "`flameox mcp inspect --tool capture_and_analyze`."
    )
    assert not marker.exists()


def test_analyze_unsupported_format_returns_accepted_formats(tmp_path: Path) -> None:
    artifact = tmp_path / "profile.json"
    artifact.write_text("{}")
    result = CliRunner().invoke(
        app,
        ["analyze", "cpu.hotspots", str(artifact), "--format", "json"],
    )

    assert result.exit_code == 1
    failure = json.loads(result.stderr)
    assert failure["code"] == "UNSUPPORTED_FORMAT"
    assert failure["details"]["received_format"] == "json"
    assert failure["details"]["accepted_formats"] == [
        "cpuprofile",
        "pstats",
        "py-spy",
        "perf",
        "perf-data",
    ]
    assert failure["details"]["recovery"].endswith("`flameox mcp inspect --tool analyze`.")


def test_analyze_can_rescue_evidence_when_configured_store_is_corrupt(tmp_path: Path) -> None:
    configured = tmp_path / "flameox-data"
    configured.mkdir()
    (configured / "unexpected").write_text("owned")
    artifact = tmp_path / "rows.json"
    artifact.write_text('[{"value": 1}, {"value": 2}]')
    rescue = tmp_path / "rescue"

    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "artifact.preview",
            str(artifact),
            "--limits",
            '{"max_rows":1}',
            "--rescue-to",
            str(rescue),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    rescued = payload["rescued"]
    assert rescued["rescue_destination"] == str(rescue)
    assert rescued["next_action"]["environment"] == {"FLAMEOX_DATA_DIR": str(rescue)}
    assert payload["next_page"]["environment"] == {"FLAMEOX_DATA_DIR": str(rescue)}
    assert rescued["evidence_id"] in payload["next_page"]["argv"]
    assert (configured / "unexpected").read_text() == "owned"
    manifest = EvidenceRepository(rescue, "cli-test").read(rescued["evidence_id"])
    assert manifest["body"]["capability_id"] == "artifact.preview"


def test_capture_rejects_invalid_rescue_destination_before_workload(tmp_path: Path) -> None:
    destination = tmp_path / "nonempty"
    destination.mkdir()
    (destination / "owned").write_text("keep")
    marker = tmp_path / "executed"

    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "direct",
            "--rescue-to",
            str(destination),
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.stderr)["message"] == (
        "Rescue destination must be a new path that does not exist"
    )
    assert not marker.exists()
    assert (destination / "owned").read_text() == "keep"


def test_capture_rejects_conflicting_evidence_destinations_before_workload(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "executed"
    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "direct",
            "--preserve",
            "--rescue-to",
            str(tmp_path / "rescue"),
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.stderr)["message"] == (
        "Use either --preserve or --rescue-to, not both."
    )
    assert not marker.exists()


def test_evidence_query_rejects_a_malformed_input_digest() -> None:
    result = CliRunner().invoke(app, ["evidence", "query", "--input-sha256", "not-a-digest"])

    assert result.exit_code == 1
    assert '"code": "INVALID_INPUT"' in result.stderr


def test_evidence_query_returns_an_executable_next_page(tmp_path: Path) -> None:
    runner = CliRunner()
    for index in range(2):
        artifact = tmp_path / f"evidence-{index}.json"
        artifact.write_text(json.dumps([{"value": index}]))
        preserved = runner.invoke(
            app,
            ["analyze", "artifact.preview", str(artifact), "--preserve"],
        )
        assert preserved.exit_code == 0, preserved.output

    first = runner.invoke(
        app,
        ["evidence", "query", "--capability", "artifact.preview", "--limit", "1"],
    )
    assert first.exit_code == 0, first.output
    payload = json.loads(first.output)
    next_page = payload["next_page"]
    second = runner.invoke(app, next_page["argv"])

    assert next_page["command"] == "flameox"
    assert "artifact.preview" in next_page["argv"]
    assert second.exit_code == 0, second.output
    assert json.loads(second.output).get("next_page") is None


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


@pytest.mark.parametrize("path_kind", ["missing", "file", "loop"])
def test_cli_capture_reports_invalid_working_directory_without_traceback(
    tmp_path: Path, path_kind: str
) -> None:
    cwd = tmp_path / "cwd"
    if path_kind == "file":
        cwd.write_text("not a directory")
    elif path_kind == "loop":
        cwd.symlink_to(cwd.name)
    result = CliRunner().invoke(
        app,
        ["capture", "--provider", "direct", "--cwd", str(cwd), "--", sys.executable, "-c", "pass"],
    )
    assert result.exit_code == 1
    assert json.loads(result.stderr)["code"] == "INVALID_INPUT"
    assert "Traceback" not in result.output


@pytest.mark.parametrize("input_kind", ["parquet", "parquet_body", "loop"])
def test_cli_preview_reports_unreadable_artifacts_without_traceback(
    tmp_path: Path, input_kind: str
) -> None:
    source = tmp_path / ("input.parquet" if input_kind.startswith("parquet") else "input.loop")
    if input_kind == "parquet":
        source.write_bytes(b"PAR1bad")
    elif input_kind == "parquet_body":
        pq.write_table(pa.table({"value": list(range(1_000))}), source, compression="snappy")
        offset = pq.ParquetFile(source).metadata.row_group(0).column(0).data_page_offset
        with source.open("r+b") as stream:
            stream.seek(offset + 20)
            stream.write(b"\xff" * 100)
    else:
        source.symlink_to(source.name)
    result = CliRunner().invoke(app, ["analyze", "artifact.preview", str(source)])
    assert result.exit_code == 1
    assert json.loads(result.stderr)["code"] == (
        "MISSING_OR_CHANGED_INPUT" if input_kind == "loop" else "DECODE_FAILURE"
    )
    assert "Traceback" not in result.output


def test_cli_rejects_misspelled_capture_option_before_resolving_the_workload() -> None:
    result = CliRunner().invoke(
        app, ["capture", "--provider", "direct", "--presrve", "--", sys.executable, "-c", "pass"]
    )
    assert result.exit_code == 2
    message = " ".join(unstyle(result.stderr).split())
    assert "No such option" in message
    assert "--presrve" in message
    assert "Executable" not in message


def test_cli_rejects_format_override_for_preserved_evidence_before_reading_store() -> None:
    result = CliRunner().invoke(
        app, ["analyze", "artifact.preview", "--evidence", "a" * 64, "--format", "text"]
    )
    assert result.exit_code == 1
    failure = json.loads(result.stderr)
    assert failure["code"] == "INVALID_INPUT"
    assert "--format" in failure["message"]


@pytest.mark.process
def test_cli_forwards_target_options_and_empty_arguments_after_separator(tmp_path: Path) -> None:
    arguments = ["--provider", "target-owned", "--help", "--preserve", "", "two words", "--"]
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
            "import json,sys; print(json.dumps(sys.argv[1:]))",
            *arguments,
        ],
    )
    assert result.exit_code == 0, result.output
    value = json.loads(result.stdout)
    assert json.loads(value["blocks"][1]["rows"][0]["text"]) == arguments
    assert value["capture"]["executions"][0]["argv"][3:] == arguments
