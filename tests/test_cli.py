from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

from flameox import __version__
from flameox.cli import app
from flameox.repository import EvidenceRepository

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def isolated_data_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLAMEOX_DATA_DIR", str(tmp_path / "flameox-data"))


def test_mcp_inspect_exposes_flat_named_tool_discovery() -> None:
    runner = CliRunner()
    summary = runner.invoke(app, ["mcp", "inspect"])
    exact = runner.invoke(app, ["mcp", "inspect", "--tool", "capture_cpu_hotspots"])

    assert summary.exit_code == 0, summary.output
    catalog = json.loads(summary.output)
    tool = next(item for item in catalog["tools"] if item["name"] == "capture_cpu_hotspots")
    assert tool["required_inputs"] == ["target", "provider"]
    assert "input_schema" not in tool

    assert exact.exit_code == 0, exact.output
    schema = json.loads(exact.output)["tools"][0]["input_schema"]
    assert "metric" in schema["properties"]
    assert "request" not in schema["properties"]
    assert set(schema["properties"]["provider"]["discriminator"]["mapping"]) == {
        "py-spy",
        "perf",
        "node-cpu-profile",
    }


def test_analyze_resumes_pagination_from_its_emitted_command(tmp_path: Path) -> None:
    artifact = tmp_path / "rows.json"
    artifact.write_text(json.dumps([{"value": value} for value in range(104)]))
    runner = CliRunner()

    first = runner.invoke(
        app,
        ["analyze", "artifact.preview", str(artifact), "--limits", '{"max_rows":100}'],
    )
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.output)
    assert "analysis_id" not in first_payload
    assert first_payload["next_page"]["command"] == "flameox"

    second = runner.invoke(app, first_payload["next_page"]["argv"])
    assert second.exit_code == 0, second.output
    assert [row["value"] for row in json.loads(second.output)["blocks"][1]["rows"]] == [
        100,
        101,
        102,
        103,
    ]


def test_analyze_can_rescue_a_page_and_continue_from_the_alternate_store(tmp_path: Path) -> None:
    configured = tmp_path / "flameox-data"
    configured.mkdir()
    (configured / "unexpected").write_text("keep")
    artifact = tmp_path / "rows.json"
    artifact.write_text('[{"value":1},{"value":2}]')
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
    assert payload["rescued"]["rescue_destination"] == str(rescue)
    assert payload["next_page"]["environment"] == {"FLAMEOX_DATA_DIR": str(rescue)}
    assert payload["rescued"]["evidence_id"] in payload["next_page"]["argv"]
    assert (configured / "unexpected").read_text() == "keep"
    manifest = EvidenceRepository(rescue, "cli-test").read(payload["rescued"]["evidence_id"])
    assert manifest["body"]["capability_id"] == "artifact.preview"


def test_capture_rejects_two_evidence_destinations_before_running_target(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    result = CliRunner().invoke(
        app,
        [
            "capture",
            "--provider",
            "direct",
            "--cwd",
            str(tmp_path),
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
    assert json.loads(result.stderr)["code"] == "INVALID_INPUT"
    assert not marker.exists()


def test_setup_requires_an_explicit_client_without_a_tty(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["setup", "--json"])

    assert result.exit_code == 2
    assert "No MCP client selected" in result.output
    assert "--client codex --yes" in unstyle(result.output)
    assert not (tmp_path / "flameox-data").exists()


def test_setup_dry_run_describes_install_without_preparing_or_writing(
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


@pytest.mark.process
def test_capture_forwards_arguments_verbatim_after_separator(tmp_path: Path) -> None:
    forwarded = ["--provider", "target-owned", "--help", "--preserve", "", "two words", "--"]
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
            *forwarded,
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert json.loads(payload["blocks"][1]["rows"][0]["text"]) == forwarded
    assert payload["capture"]["executions"][0]["argv"][3:] == forwarded
