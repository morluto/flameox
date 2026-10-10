from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest
from click import unstyle

from flameox import __version__
from flameox.setup import SetupClient, apply_client_setup, detect_setup_clients, plan_client_setup

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.process,
    pytest.mark.skipif(os.name != "posix", reason="Installer shim requires POSIX"),
]


@pytest.mark.skipif(
    os.name == "posix" and os.getuid() == 0, reason="Root bypasses directory permissions"
)
@pytest.mark.parametrize("client", ["codex", "opencode"])
@pytest.mark.parametrize("command", ["setup", "update"])
def test_inaccessible_profiles_have_a_cli_diagnostic(
    tmp_path: Path, client: str, command: str
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    environment = {**os.environ, "HOME": str(tmp_path)}
    environment["CODEX_HOME" if client == "codex" else "OPENCODE_CONFIG_DIR"] = str(profile)
    arguments = ["--dry-run"] if command == "setup" else ["--check", "--version", __version__]
    profile.chmod(0)
    try:
        result = subprocess.run(
            [
                str(Path(sys.executable).with_name("flameox")),
                command,
                "--client",
                client,
                *arguments,
                "--json",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 2
        assert "configuration" in result.stderr
        assert "Traceback" not in result.stderr
        assert "PermissionError" not in result.stderr
    finally:
        profile.chmod(0o700)
    assert list(profile.iterdir()) == []


@pytest.mark.parametrize("redirect_stderr", [False, True])
def test_interactive_setup_reviews_changes_and_cancellation_never_prepares(
    tmp_path: Path, redirect_stderr: bool
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "prepared"
    uvx = bin_dir / "uvx"
    uvx.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n")
    uvx.chmod(0o755)
    environment = {**os.environ, "HOME": str(tmp_path), "PATH": str(bin_dir), "NO_COLOR": "1"}
    master, slave = os.openpty()
    output = b""
    process = subprocess.Popen(
        [str(Path(sys.executable).with_name("flameox")), "setup", "--client", "cursor"],
        env=environment,
        stdin=slave,
        stdout=slave,
        stderr=subprocess.PIPE if redirect_stderr else slave,
    )
    os.close(slave)
    try:
        deadline = time.monotonic() + 10
        answered = False
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output += chunk
                if b"Apply these changes?" in output and not answered:
                    assert b"setup plan" in output
                    assert str(tmp_path / ".cursor" / "mcp.json").encode() in output
                    assert b"uvx" in output
                    assert not marker.exists()
                    os.write(master, b"n\n")
                    answered = True
            elif process.poll() is not None:
                break
        process.wait(timeout=2)
        if redirect_stderr:
            assert process.returncode == 2
            assert process.stderr is not None
            assert b"Non-interactive setup requires" in process.stderr.read()
            assert not answered
        else:
            assert process.returncode == 0, output.decode(errors="replace")
            assert answered
            assert b"cancelled" in output
        assert not marker.exists()
        assert not (tmp_path / ".cursor").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if process.stderr is not None:
            process.stderr.close()
        os.close(master)


@pytest.mark.parametrize(
    ("client", "overrides", "relative_path"),
    [
        ("claude", {"CLAUDE_CONFIG_DIR": "profile"}, "profile/.claude.json"),
        ("codex", {"CODEX_HOME": "profile"}, "profile/config.toml"),
        ("gemini", {"GEMINI_CLI_HOME": "profile"}, "profile/.gemini/settings.json"),
        ("opencode", {"XDG_CONFIG_HOME": "xdg"}, "xdg/opencode/opencode.jsonc"),
        ("opencode", {"OPENCODE_CONFIG": "custom.jsonc"}, "custom.jsonc"),
        (
            "opencode",
            {"OPENCODE_CONFIG_DIR": "profile", "OPENCODE_CONFIG": "other.jsonc"},
            "profile/opencode.jsonc",
        ),
        ("codex", {"CODEX_HOME": ""}, ".codex/config.toml"),
    ],
)
def test_setup_and_update_use_the_selected_client_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: str,
    overrides: dict[str, str],
    relative_path: str,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    for name, value in overrides.items():
        monkeypatch.setenv(name, str(tmp_path / value) if value else "")
    environment = dict(os.environ)
    executable = str(Path(sys.executable).with_name("flameox"))

    def invoke(*arguments: str) -> dict[str, Any]:
        result = subprocess.run(
            [executable, *arguments, "--client", client, "--json"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        return cast(dict[str, Any], json.loads(result.stdout))

    expected = tmp_path / relative_path
    preview = invoke("setup", "--dry-run")
    assert preview["plan"][0]["path"] == str(expected)
    assert not expected.exists()
    plans = plan_client_setup([SetupClient(client)], [])
    apply_client_setup(plans)
    original = expected.read_text()
    assert SetupClient(client) in detect_setup_clients()
    repeated = invoke("setup", "--dry-run")
    assert repeated["plan"][0]["action"] == "already_current"
    update = invoke("update", "--version", __version__, "--check")
    assert update["clients"][0]["path"] == str(expected)
    assert expected.read_text() == original


@pytest.mark.parametrize("failure", [None, "missing", "version", "resolver", "output", "timeout"])
@pytest.mark.parametrize("provider", [None, "memray"])
def test_setup_verifies_release_before_publishing_and_settles_installer_children(
    tmp_path: Path, failure: str | None, provider: str | None
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    child_started = tmp_path / "child-started"
    child_finished = tmp_path / "child-finished"
    release_child = tmp_path / "release-child"
    configuration = tmp_path / ".cursor" / "mcp.json"
    configuration.parent.mkdir()
    original = json.dumps(
        {
            "mcpServers": {
                "flameox": {
                    "command": "custom",
                    "env": {"UV_DEFAULT_INDEX": "https://custom.invalid/simple"},
                }
            }
        }
    )
    configuration.write_text(original)
    if failure != "missing":
        uvx = bin_dir / "uvx"
        uvx.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys, time\n"
            "from pathlib import Path\n"
            f"failure = {failure!r}\n"
            "assert os.environ['UV_DEFAULT_INDEX'] == 'https://custom.invalid/simple'\n"
            "assert sys.argv[1:3] == ['--no-config', '--no-sources']\n"
            "if failure == 'timeout':\n"
            "    if os.fork() == 0:\n"
            f"        Path({str(child_started)!r}).touch()\n"
            f"        while not Path({str(release_child)!r}).exists():\n"
            "            time.sleep(0.01)\n"
            f"        Path({str(child_finished)!r}).touch()\n"
            "        sys.exit(0)\n"
            "    time.sleep(30)\n"
            "if failure == 'output':\n"
            "    sys.stderr.buffer.write(b'x' * (2 * 1024 * 1024))\n"
            "    sys.exit(0)\n"
            "if failure == 'resolver':\n"
            "    print('resolver diagnostic', file=sys.stderr)\n"
            "    sys.exit(23)\n"
            "if sys.argv[-2] == '-c':\n"
            f"    version = '0.0.0' if failure == 'version' else {__version__!r}\n"
            "    print(json.dumps({'version': version, 'extras': ['memory']}))\n"
            "else:\n"
            "    print(json.dumps({'tools': [{'name': 'cpu_hotspots'}]}))\n"
        )
        uvx.chmod(0o755)
    environment = {name: value for name, value in os.environ.items() if not name.startswith("UV_")}
    environment.update(
        {"HOME": str(tmp_path), "PATH": str(bin_dir), "FORCE_COLOR": "1", "COLUMNS": "80"}
    )
    command = [
        str(Path(sys.executable).with_name("flameox")),
        "setup",
        "--client",
        "cursor",
        "--yes",
        "--json",
    ]
    if provider:
        command.extend(["--provider", provider])
    if failure == "timeout":
        command.extend(["--timeout-seconds", "5"])
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=25)
    if failure is None:
        assert result.returncode == 0, result.stderr
        assert json.loads(configuration.read_text())["mcpServers"]["flameox"]["command"] == "uvx"
    else:
        assert result.returncode != 0, result.stdout
        assert configuration.read_text() == original
        assert len(unstyle(result.stderr)) < 1100 * 1024
        if failure == "resolver":
            assert "resolver diagnostic" in result.stderr
        if failure == "timeout":
            assert child_started.exists()
            release_child.touch()
            time.sleep(1)
            assert not child_finished.exists()
