from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import json5
import pytest
import tomlkit

from flameox import __version__
from flameox.providers.environment import SetupFailure
from flameox.setup import SetupClient, apply_client_setup, plan_client_setup


@pytest.mark.process
@pytest.mark.skipif(os.name != "posix", reason="FIFOs require POSIX")
@pytest.mark.parametrize("client", ["claude", "codex"])
@pytest.mark.parametrize("operation", ["plan", "apply"])
def test_setup_rejects_special_configuration_files_without_blocking(
    tmp_path: Path, client: str, operation: str
) -> None:
    script = """
import os, sys
from pathlib import Path
from flameox.setup import SetupClient, plan_client_setup, apply_client_setup
from flameox.providers.environment import SetupFailure
root = Path(sys.argv[1])
client = SetupClient(sys.argv[2])
path = client.config_path(root)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text('' if client is SetupClient.CODEX else '{}')
plans = plan_client_setup([client], [], home=root)
path.unlink()
os.mkfifo(path)
try:
    if sys.argv[3] == 'plan':
        plan_client_setup([client], [], home=root)
    else:
        apply_client_setup(plans)
except SetupFailure:
    pass
else:
    raise AssertionError('special configuration accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), client, operation],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("client", [SetupClient.CLAUDE, SetupClient.CURSOR, SetupClient.GEMINI])
def test_json_clients_get_a_pinned_global_launcher(client: SetupClient, tmp_path: Path) -> None:
    plan = plan_client_setup([client], [], home=tmp_path)[0]
    result = apply_client_setup([plan])[0]

    entry = json.loads(plan.path.read_text())["mcpServers"]["flameox"]
    assert result.action == "created"
    assert entry["command"] == "uvx"
    assert entry["args"][-3:] == ["flameox", "mcp", "serve"]
    assert f"flameox=={__version__}" in entry["args"]


def test_setup_preserves_existing_client_configuration_and_is_idempotent(
    tmp_path: Path,
) -> None:
    config = tmp_path / ".cursor" / "mcp.json"
    config.parent.mkdir(parents=True)
    original = '{"theme":"dark","mcpServers":{"other":{"command":"other"}}}'
    config.write_text(original)

    plan = plan_client_setup([SetupClient.CURSOR], [], home=tmp_path)[0]
    assert apply_client_setup([plan])[0].action == "created"
    updated = json.loads(config.read_text())
    assert updated["theme"] == "dark"
    assert updated["mcpServers"]["other"] == {"command": "other"}
    assert "flameox" in updated["mcpServers"]

    custom_format = json.dumps(updated, separators=(",", ":"))
    config.write_text(custom_format)
    repeated = plan_client_setup([SetupClient.CURSOR], [], home=tmp_path)[0]
    assert repeated.content == custom_format
    assert apply_client_setup([repeated])[0].action == "already_current"
    assert config.read_text() == custom_format


@pytest.mark.parametrize("prefix", ["", "\ufeff"])
def test_opencode_setup_edits_active_jsonc_without_losing_comments(
    tmp_path: Path, prefix: str
) -> None:
    config = tmp_path / ".config" / "opencode" / "opencode.jsonc"
    config.parent.mkdir(parents=True)
    config.write_text(
        prefix + '{\n  // keep this comment\n  "theme": "dark",\n'
        '  "enabled": true// keep the scalar comment\n}\n'
    )

    plan = plan_client_setup([SetupClient.OPENCODE], [], home=tmp_path)[0]
    apply_client_setup([plan])

    content = config.read_text()
    document = json5.loads(content)
    assert content.startswith(prefix + "{")
    assert "keep this comment" in content
    assert "true,// keep the scalar comment" in content
    assert document["enabled"] is True
    assert document["theme"] == "dark"
    assert document["mcp"]["flameox"]["type"] == "local"
    assert document["mcp"]["flameox"]["command"][-3:] == ["flameox", "mcp", "serve"]


@pytest.mark.parametrize("extension", ["json", "jsonc"])
def test_opencode_setup_rejects_explicit_null_mcp_sections_consistently(
    tmp_path: Path, extension: str
) -> None:
    config = tmp_path / ".config" / "opencode" / f"opencode.{extension}"
    config.parent.mkdir(parents=True)
    original = '{"theme":"dark","mcp":null}'
    config.write_text(original)
    with pytest.raises(SetupFailure, match="'mcp' must be an object"):
        plan_client_setup([SetupClient.OPENCODE], [], home=tmp_path)
    assert config.read_text() == original


def test_codex_toml_setup_preserves_comments_and_unrelated_settings(tmp_path: Path) -> None:
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('# keep this comment\nmodel = "gpt"\n')

    plan = plan_client_setup([SetupClient.CODEX], [], home=tmp_path)[0]
    apply_client_setup([plan])

    document = tomlkit.parse(config.read_text())
    assert document["model"] == "gpt"
    assert document["mcp_servers"]["flameox"]["command"] == "uvx"
    assert "# keep this comment" in config.read_text()


@pytest.mark.parametrize("client", list(SetupClient))
@pytest.mark.parametrize(
    "environment", [{"UV_DEFAULT_INDEX": "https://custom.invalid/simple"}, {"UV_OFFLINE": False}]
)
def test_setup_prepares_preserved_client_environments_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: SetupClient,
    environment: dict[str, object],
) -> None:
    from flameox.setup import prepare_providers

    initial = plan_client_setup([client], [], home=tmp_path)[0]
    apply_client_setup([initial])
    document = (
        tomlkit.parse(initial.path.read_text())
        if client is SetupClient.CODEX
        else json5.loads(initial.path.read_text())
    )
    key = "environment" if client is SetupClient.OPENCODE else "env"
    document[client.server_section]["flameox"][key] = environment
    original = tomlkit.dumps(document) if client is SetupClient.CODEX else json.dumps(document)
    initial.path.write_text(original)
    plans = plan_client_setup([client], ["memray"], home=tmp_path)
    verified: list[tuple[str, dict[str, str]]] = []

    def verify(releases: list[tuple[str, dict[str, str]]], *, timeout_seconds: float) -> None:
        assert timeout_seconds == 10
        verified.extend(releases)

    monkeypatch.setattr("flameox.setup.verify_releases", verify)
    if isinstance(next(iter(environment.values())), str):
        prepare_providers(plans, ["memray"], 10)
        assert verified == [(f"flameox[memory]=={__version__}", environment)]
    else:
        with pytest.raises(SetupFailure, match="Invalid Flameox environment"):
            prepare_providers(plans, ["memray"], 10)
        assert not verified
    assert initial.path.read_text() == original


@pytest.mark.process
@pytest.mark.skipif(os.name != "posix", reason="Version probe shim requires POSIX")
@pytest.mark.parametrize("mode", ["different", "matching", "output", "timeout"])
def test_path_cli_advisory_is_bounded_nonfatal_and_settles_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import time

    from flameox.setup import path_cli_version_advisory

    started = tmp_path / "started"
    release = tmp_path / "release"
    finished = tmp_path / "finished"
    executable = tmp_path / "flameox"
    executable.write_text(
        f"#!{sys.executable}\nimport os, time\nfrom pathlib import Path\n"
        f"mode = {mode!r}\n"
        "if mode == 'timeout':\n"
        "    if os.fork() == 0:\n"
        f"        Path({str(started)!r}).touch()\n"
        f"        while not Path({str(release)!r}).exists():\n"
        "            time.sleep(0.01)\n"
        f"        Path({str(finished)!r}).touch()\n"
        "        os._exit(0)\n"
        "    time.sleep(30)\n"
        "elif mode == 'output':\n    print('x' * 8192)\n"
        f"else:\n    print({__version__!r} if mode == 'matching' else '0.0.1')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    result = path_cli_version_advisory()
    if mode == "different":
        assert result is not None
        assert result.cli_version == "0.0.1"
    else:
        assert result is None
    if mode == "timeout":
        assert started.exists()
        release.touch()
        time.sleep(1)
        assert not finished.exists()
