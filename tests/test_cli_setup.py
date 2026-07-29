from __future__ import annotations

import json
import os
import pty
import selectors
import subprocess
import sys
import time
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


@pytest.mark.skipif(os.name != "posix", reason="PTY smoke test requires POSIX")
def test_npm_bootstrap_pty_reaches_first_python_prompt(tmp_path: Path) -> None:
    fake_uvx = tmp_path / "uvx"
    flameox = Path(sys.executable).with_name("flameox")
    fake_uvx.write_text(
        f"""#!{sys.executable}
import os, sys
arguments = sys.argv[1:]
handoff = arguments.index("flameox")
os.execv({str(flameox)!r}, [{str(flameox)!r}, *arguments[handoff + 1:]])
"""
    )
    fake_uvx.chmod(0o700)
    bootstrap = Path(__file__).parents[1] / "npm" / "bin" / "flameox.cjs"
    master, slave = pty.openpty()
    process = subprocess.Popen(
        ["node", str(bootstrap), "setup"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env={
            **os.environ,
            "FLAMEOX_UV_EXECUTABLE": str(fake_uvx),
            "FLAMEOX_SETUP_HOME": str(tmp_path / "home"),
            "FLAMEOX_SETUP_DATA_ROOT": str(tmp_path / "data"),
        },
        close_fds=True,
    )
    os.close(slave)
    selector = selectors.DefaultSelector()
    selector.register(master, selectors.EVENT_READ)
    transcript = bytearray()
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline and b"Select MCP clients to connect" not in transcript:
            if selector.select(timeout=0.25):
                transcript.extend(os.read(master, 4_096))
    finally:
        process.terminate()
        process.wait(timeout=5)
        selector.close()
        os.close(master)

    output = transcript.decode(errors="replace")
    assert "cached managed Python runtime" in output
    assert "Managed runtime ready. Starting flameox setup..." in output
    assert "Select MCP clients to connect" in output
