from __future__ import annotations

import json
from pathlib import Path

import json5
import pytest
import tomlkit

from flameox import __version__
from flameox.setup import (
    SetupClient,
    SetupFailure,
    apply_client_setup,
    plan_client_setup,
)


@pytest.mark.parametrize(
    "client",
    [
        SetupClient.CLAUDE,
        SetupClient.CURSOR,
        SetupClient.GEMINI,
        SetupClient.ANTIGRAVITY,
    ],
)
def test_standard_json_clients_receive_command_and_args(
    client: SetupClient, tmp_path: Path
) -> None:
    plan = plan_client_setup([client], [], home=tmp_path)[0]
    apply_client_setup([plan])

    entry = json.loads(client.config_path(tmp_path).read_text())["mcpServers"]["flameox"]
    assert entry["command"] == "uvx"
    assert entry["args"][-3:] == ["flameox", "mcp", "serve"]


def test_opencode_receives_its_local_command_shape(tmp_path: Path) -> None:
    plan = plan_client_setup([SetupClient.OPENCODE], [], home=tmp_path)[0]
    apply_client_setup([plan])

    entry = json.loads(SetupClient.OPENCODE.config_path(tmp_path).read_text())["mcp"]["flameox"]
    assert entry == {
        "type": "local",
        "command": [
            "uvx",
            "--python",
            "3.12",
            "--from",
            f"flameox=={__version__}",
            "flameox",
            "mcp",
            "serve",
        ],
        "enabled": True,
    }


def test_opencode_jsonc_preserves_comments_when_adding_flameox(tmp_path: Path) -> None:
    jsonc = tmp_path / ".config" / "opencode" / "opencode.jsonc"
    jsonc.parent.mkdir(parents=True)
    jsonc.write_text(
        '{\n  // keep this comment\n  "theme": "dark",\n  "mcp": {\n'
        '    "other": {"type": "local", "command": ["other"]},\n  },\n}\n'
    )

    plan = plan_client_setup([SetupClient.OPENCODE], [], home=tmp_path)[0]
    apply_client_setup([plan])

    content = jsonc.read_text()
    assert "keep this comment" in content
    document = json5.loads(content)
    assert document["theme"] == "dark"
    assert document["mcp"]["other"]["command"] == ["other"]
    assert document["mcp"]["flameox"]["type"] == "local"


def test_opencode_jsonc_replaces_its_flameox_entry_in_place(tmp_path: Path) -> None:
    jsonc = tmp_path / ".config" / "opencode" / "opencode.jsonc"
    jsonc.parent.mkdir(parents=True)
    jsonc.write_text(
        '{\n  "mcp": {\n    // Flameox notes\n'
        '    "flameox": {"type": "local", "command": ["custom"], "cwd": "/work"},\n'
        "  },\n}\n"
    )

    plan = plan_client_setup([SetupClient.OPENCODE], [], home=tmp_path)[0]
    apply_client_setup([plan])

    document = json5.loads(jsonc.read_text())
    assert "Flameox notes" in jsonc.read_text()
    assert document["mcp"]["flameox"] == {
        "type": "local",
        "command": [
            "uvx",
            "--python",
            "3.12",
            "--from",
            f"flameox=={__version__}",
            "flameox",
            "mcp",
            "serve",
        ],
        "enabled": True,
        "cwd": "/work",
    }


@pytest.mark.parametrize("filename", ["opencode.json", "config.json"])
def test_opencode_updates_the_active_global_json_layer(filename: str, tmp_path: Path) -> None:
    config = tmp_path / ".config" / "opencode" / filename
    config.parent.mkdir(parents=True)
    config.write_text('{"theme":"dark"}')

    plan = plan_client_setup([SetupClient.OPENCODE], [], home=tmp_path)[0]
    apply_client_setup([plan])

    assert plan.path == config
    assert json.loads(config.read_text())["mcp"]["flameox"]["type"] == "local"


def test_opencode_updates_higher_precedence_jsonc_when_json_also_exists(tmp_path: Path) -> None:
    directory = tmp_path / ".config" / "opencode"
    directory.mkdir(parents=True)
    jsonc = directory / "opencode.jsonc"
    jsonc.write_text("{// keep\n}")
    (directory / "opencode.json").write_text("{}")

    plan = plan_client_setup([SetupClient.OPENCODE], [], home=tmp_path)[0]
    apply_client_setup([plan])

    assert json5.loads(jsonc.read_text())["mcp"]["flameox"]["type"] == "local"
    assert json.loads((directory / "opencode.json").read_text()) == {}


def test_setup_preserves_unrelated_json_and_toml_configuration(tmp_path: Path) -> None:
    cursor = tmp_path / ".cursor" / "mcp.json"
    cursor.parent.mkdir(parents=True)
    cursor.write_text('{"theme":"dark","mcpServers":{"other":{"command":"other"}}}')
    codex = tmp_path / ".codex" / "config.toml"
    codex.parent.mkdir(parents=True)
    codex.write_text('# keep this comment\nmodel = "gpt"\n')

    plans = plan_client_setup([SetupClient.CURSOR, SetupClient.CODEX], [], home=tmp_path)
    results = apply_client_setup(plans)

    cursor_config = json.loads(cursor.read_text())
    assert cursor_config["theme"] == "dark"
    assert cursor_config["mcpServers"]["other"] == {"command": "other"}
    assert cursor_config["mcpServers"]["flameox"]["args"][-3:] == [
        "flameox",
        "mcp",
        "serve",
    ]
    assert "# keep this comment" in codex.read_text()
    assert "[mcp_servers.flameox]" in codex.read_text()
    assert [result.action for result in results] == ["created", "created"]

    repeated = plan_client_setup([SetupClient.CURSOR, SetupClient.CODEX], [], home=tmp_path)
    assert [plan.action for plan in repeated] == ["already_current", "already_current"]


def test_setup_refuses_a_symlinked_client_configuration(tmp_path: Path) -> None:
    target = tmp_path / "managed-elsewhere.json"
    target.write_text("{}")
    config = SetupClient.CURSOR.config_path(tmp_path)
    config.parent.mkdir(parents=True)
    config.symlink_to(target)

    with pytest.raises(SetupFailure, match="symbolic-link"):
        plan_client_setup([SetupClient.CURSOR], [], home=tmp_path)

    assert target.read_text() == "{}"


def test_setup_refuses_configuration_changed_after_planning(tmp_path: Path) -> None:
    plan = plan_client_setup([SetupClient.CLAUDE], [], home=tmp_path)[0]
    plan.path.write_text('{"changed":true}')

    with pytest.raises(SetupFailure, match="changed during setup"):
        apply_client_setup([plan])

    assert json.loads(plan.path.read_text()) == {"changed": True}


def test_setup_updates_a_previous_version_pinned_launcher(tmp_path: Path) -> None:
    config = tmp_path / ".claude.json"
    config.write_text(
        '{"mcpServers":{"flameox":{"command":"uvx","args":'
        '["--python","3.12","--from","flameox==0.1.0","flameox","mcp","serve"]}}}'
    )

    plans = plan_client_setup([SetupClient.CLAUDE], [], home=tmp_path)

    assert plans[0].action == "update"
    assert f"flameox=={__version__}" in plans[0].content


def test_setup_migrates_the_legacy_project_bound_launcher(tmp_path: Path) -> None:
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        '[mcp_servers.flameox]\ncommand = "/opt/uv/bin/uvx"\n'
        'args = ["--python", "3.12", "--from", "flameox==0.2.2", '
        '"flameox", "mcp", "serve", "--project-root", "/work/old"]\n'
    )

    plan = plan_client_setup([SetupClient.CODEX], [], home=tmp_path)[0]
    apply_client_setup([plan])

    entry = tomlkit.parse(config.read_text())["mcp_servers"]["flameox"]
    assert entry["command"] == "uvx"
    assert entry["args"][-3:] == ["flameox", "mcp", "serve"]
    assert "--project-root" not in entry["args"]


def test_setup_preserves_custom_fields_inside_managed_entries(tmp_path: Path) -> None:
    cursor = tmp_path / ".cursor" / "mcp.json"
    cursor.parent.mkdir(parents=True)
    cursor.write_text(
        '{"mcpServers":{"flameox":{"command":"uvx","args":'
        '["--from","flameox==0.1.0","flameox","mcp","serve"],'
        '"env":{"TOKEN":"from-environment"}}}}'
    )
    codex = tmp_path / ".codex" / "config.toml"
    codex.parent.mkdir(parents=True)
    codex.write_text(
        '[mcp_servers.flameox]\ncommand = "uvx"\n'
        'args = ["--from", "flameox==0.1.0", "flameox", "mcp", "serve"]\n'
        'env = { TOKEN = "from-environment" }\n'
    )

    plans = plan_client_setup([SetupClient.CURSOR, SetupClient.CODEX], [], home=tmp_path)
    apply_client_setup(plans)

    assert json.loads(cursor.read_text())["mcpServers"]["flameox"]["env"] == {
        "TOKEN": "from-environment"
    }
    assert 'env = { TOKEN = "from-environment" }' in codex.read_text()


def test_setup_supports_an_inline_codex_server_table(tmp_path: Path) -> None:
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text("mcp_servers = {}\n")

    plan = plan_client_setup([SetupClient.CODEX], [], home=tmp_path)[0]
    apply_client_setup([plan])

    assert "flameox" in tomlkit.parse(config.read_text())["mcp_servers"]


def test_setup_replaces_a_non_table_codex_flameox_entry(tmp_path: Path) -> None:
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('mcp_servers = { flameox = "custom" }\n')

    plan = plan_client_setup([SetupClient.CODEX], [], home=tmp_path)[0]
    apply_client_setup([plan])

    entry = tomlkit.parse(config.read_text())["mcp_servers"]["flameox"]
    assert entry["command"] == "uvx"
