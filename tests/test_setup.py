from __future__ import annotations

import json
from pathlib import Path

import json5
import pytest
import tomlkit

from flameox import __version__
from flameox.setup import SetupClient, apply_client_setup, plan_client_setup


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


def test_opencode_setup_edits_active_jsonc_without_losing_comments(tmp_path: Path) -> None:
    config = tmp_path / ".config" / "opencode" / "opencode.jsonc"
    config.parent.mkdir(parents=True)
    config.write_text(
        '{\n  // keep this comment\n  "theme": "dark",\n'
        '  "enabled": true// keep the scalar comment\n}\n'
    )

    plan = plan_client_setup([SetupClient.OPENCODE], [], home=tmp_path)[0]
    apply_client_setup([plan])

    content = config.read_text()
    document = json5.loads(content)
    assert "keep this comment" in content
    assert "true,// keep the scalar comment" in content
    assert document["enabled"] is True
    assert document["theme"] == "dark"
    assert document["mcp"]["flameox"]["type"] == "local"
    assert document["mcp"]["flameox"]["command"][-3:] == ["flameox", "mcp", "serve"]


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
