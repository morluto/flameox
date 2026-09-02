from __future__ import annotations

import json
import subprocess
from pathlib import Path
from subprocess import CompletedProcess

import pytest
import tomlkit

from flameox import __version__
from flameox.setup import (
    DEFAULT_PREPARATION_TIMEOUT_SECONDS,
    SETUP_CLIENTS,
    SetupClient,
    SetupFailure,
    apply_client_setup,
    detect_setup_clients,
    mcp_launcher,
    parse_setup_clients,
    path_cli_version_advisory,
    plan_client_setup,
    prepare_providers,
)


def test_path_cli_version_advisory_reports_only_a_different_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/tools/flameox")
    monkeypatch.setattr(
        "flameox.setup.subprocess.run",
        lambda command, **_kwargs: CompletedProcess(command, 0, stdout=b"0.1.0\n"),
    )

    advisory = path_cli_version_advisory()

    assert advisory is not None
    assert advisory.executable == "/tools/flameox"
    assert advisory.cli_version == "0.1.0"
    assert advisory.mcp_version == __version__
    assert "direct cli commands" in advisory.message.lower()


def test_path_cli_version_advisory_is_nonfatal_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/tools/flameox")
    monkeypatch.setattr(
        "flameox.setup.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    assert path_cli_version_advisory() is None


def test_path_cli_version_advisory_ignores_an_absent_or_aligned_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: None)
    assert path_cli_version_advisory() is None

    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/tools/flameox")
    monkeypatch.setattr(
        "flameox.setup.subprocess.run",
        lambda command, **_kwargs: CompletedProcess(command, 0, stdout=f"{__version__}\n".encode()),
    )
    assert path_cli_version_advisory() is None


def test_mcp_launcher_pins_python_release_and_provider_extras() -> None:
    command, args = mcp_launcher(["memray", "nsight-compute", "py-spy"])

    assert command == "uvx"
    assert args == [
        "--python",
        "3.12",
        "--from",
        f"flameox[cpu,memory]=={__version__}",
        "flameox",
    ]


def test_preparation_runs_and_returns_the_same_exact_uvx_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    options: list[dict[str, object]] = []

    def run(command: list[str], **kwargs: object) -> CompletedProcess[bytes]:
        calls.append(command)
        options.append(kwargs)
        return CompletedProcess(command, 0, stderr=b"")

    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/usr/bin/uvx")
    monkeypatch.setattr("flameox.setup.subprocess.run", run)

    prepared = prepare_providers(["memray", "py-spy", "memray"])

    requirement = f"flameox[cpu,memory]=={__version__}"
    assert prepared.requested_providers == ["memray", "py-spy"]
    assert prepared.prepared_managed_providers == ["memray", "py-spy"]
    assert prepared.preparation_status == "prepared"
    assert prepared.restart_required is True
    assert calls == [
        [
            "/usr/bin/uvx",
            "--python",
            "3.12",
            "--from",
            requirement,
            "flameox",
            "--version",
        ]
    ]
    assert options == [
        {
            "check": False,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.PIPE,
            "timeout": DEFAULT_PREPARATION_TIMEOUT_SECONDS,
        }
    ]
    assert prepared.launcher_command == "uvx"
    assert prepared.launcher_args == [
        "--python",
        "3.12",
        "--from",
        requirement,
        "flameox",
        "mcp",
        "serve",
    ]


def test_host_only_preparation_returns_guidance_without_requiring_uvx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: None)

    prepared = prepare_providers(["nsight-compute"])

    assert prepared.prepared_managed_providers == []
    assert [item.provider_id for item in prepared.external_requirements] == ["nsight-compute"]
    assert prepared.preparation_command == []
    assert prepared.preparation_status == "not_applicable"
    assert prepared.restart_required is False


def test_uvx_is_required_before_managed_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: None)

    with pytest.raises(SetupFailure, match="requires uvx"):
        prepare_providers(["memray"])


def test_uvx_failure_is_normalized_as_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/usr/bin/uvx")
    monkeypatch.setattr(
        "flameox.setup.subprocess.run",
        lambda command, **_kwargs: CompletedProcess(
            command,
            2,
            stderr=b"No solution found: dependency conflict.\xff",
        ),
    )

    with pytest.raises(SetupFailure) as raised:
        prepare_providers(["memray"])

    assert "status 2" in str(raised.value)
    assert "No solution found: dependency conflict.\ufffd" in str(raised.value)


def test_preparation_timeout_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> CompletedProcess[bytes]:
        observed.update(kwargs)
        return CompletedProcess(command, 0, stderr=b"")

    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/usr/bin/uvx")
    monkeypatch.setattr("flameox.setup.subprocess.run", run)

    prepare_providers(["torch"], timeout_seconds=2_400)

    assert observed["timeout"] == 2_400


def test_preparation_timeout_preserves_uvx_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(*_args: object, **_kwargs: object) -> CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired("uvx", 2_400, stderr=b"download stalled\xff")

    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/usr/bin/uvx")
    monkeypatch.setattr("flameox.setup.subprocess.run", run)

    with pytest.raises(SetupFailure) as raised:
        prepare_providers(["torch"], timeout_seconds=2_400)

    assert "exceeded 2400 seconds" in str(raised.value)
    assert "download stalled\ufffd" in str(raised.value)


@pytest.mark.parametrize("timeout_seconds", [0, 3_601])
def test_preparation_timeout_is_bounded(timeout_seconds: int) -> None:
    with pytest.raises(SetupFailure, match="between 1 and 3600"):
        prepare_providers(["torch"], timeout_seconds=timeout_seconds)


def test_uvx_start_failure_is_normalized_as_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> CompletedProcess[str]:
        raise PermissionError("denied")

    monkeypatch.setattr("flameox.setup.shutil.which", lambda _name: "/usr/bin/uvx")
    monkeypatch.setattr("flameox.setup.subprocess.run", fail)

    with pytest.raises(SetupFailure, match="could not prepare"):
        prepare_providers(["memray"])


def test_unknown_provider_is_rejected_before_preparation() -> None:
    with pytest.raises(SetupFailure, match="Unknown provider"):
        prepare_providers(["mystery"])


def test_setup_clients_have_distinct_global_configuration_paths(tmp_path: Path) -> None:
    paths = {client: client.config_path(tmp_path) for client in SETUP_CLIENTS}

    assert paths == {
        SetupClient.CLAUDE: tmp_path / ".claude.json",
        SetupClient.CURSOR: tmp_path / ".cursor" / "mcp.json",
        SetupClient.OPENCODE: tmp_path / ".config" / "opencode" / "opencode.jsonc",
        SetupClient.CODEX: tmp_path / ".codex" / "config.toml",
        SetupClient.GEMINI: tmp_path / ".gemini" / "settings.json",
        SetupClient.ANTIGRAVITY: tmp_path / ".gemini" / "config" / "mcp_config.json",
    }
    assert len(set(paths.values())) == len(SETUP_CLIENTS)


def test_client_detection_reports_presence_without_selecting_clients(tmp_path: Path) -> None:
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".agent").mkdir()

    assert detect_setup_clients(tmp_path) == [SetupClient.CODEX, SetupClient.ANTIGRAVITY]


def test_antigravity_directory_does_not_imply_gemini_cli(tmp_path: Path) -> None:
    (tmp_path / ".gemini" / "antigravity").mkdir(parents=True)

    assert detect_setup_clients(tmp_path) == [SetupClient.ANTIGRAVITY]


def test_claude_global_config_is_a_detection_marker(tmp_path: Path) -> None:
    (tmp_path / ".claude.json").write_text("{}")

    assert detect_setup_clients(tmp_path) == [SetupClient.CLAUDE]


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


def test_existing_opencode_jsonc_is_not_silently_rewritten(tmp_path: Path) -> None:
    jsonc = tmp_path / ".config" / "opencode" / "opencode.jsonc"
    jsonc.parent.mkdir(parents=True)
    jsonc.write_text('{\n  // keep this comment\n  "theme": "dark",\n}\n')

    with pytest.raises(SetupFailure, match="will not rewrite it"):
        plan_client_setup([SetupClient.OPENCODE], [], home=tmp_path)

    assert "keep this comment" in jsonc.read_text()


@pytest.mark.parametrize("filename", ["opencode.json", "config.json"])
def test_opencode_updates_the_active_global_json_layer(filename: str, tmp_path: Path) -> None:
    config = tmp_path / ".config" / "opencode" / filename
    config.parent.mkdir(parents=True)
    config.write_text('{"theme":"dark"}')

    plan = plan_client_setup([SetupClient.OPENCODE], [], home=tmp_path)[0]
    apply_client_setup([plan])

    assert plan.path == config
    assert json.loads(config.read_text())["mcp"]["flameox"]["type"] == "local"


def test_opencode_refuses_higher_precedence_jsonc_when_json_also_exists(tmp_path: Path) -> None:
    directory = tmp_path / ".config" / "opencode"
    directory.mkdir(parents=True)
    jsonc = directory / "opencode.jsonc"
    jsonc.write_text("{// keep\n}")
    (directory / "opencode.json").write_text("{}")

    with pytest.raises(SetupFailure, match="will not rewrite"):
        plan_client_setup([SetupClient.OPENCODE], [], home=tmp_path)


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


def test_setup_rejects_unmanaged_entries_before_writing_any_client(tmp_path: Path) -> None:
    cursor = tmp_path / ".cursor" / "mcp.json"
    cursor.parent.mkdir(parents=True)
    cursor.write_text('{"mcpServers":{"flameox":{"command":"custom","args":[]}}}')

    with pytest.raises(SetupFailure, match="unmanaged Flameox entry"):
        plan_client_setup([SetupClient.CODEX, SetupClient.CURSOR], [], home=tmp_path)

    assert not (tmp_path / ".codex" / "config.toml").exists()


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


def test_already_current_setup_rejects_a_concurrent_change(tmp_path: Path) -> None:
    first = plan_client_setup([SetupClient.CLAUDE], [], home=tmp_path)[0]
    apply_client_setup([first])
    current = plan_client_setup([SetupClient.CLAUDE], [], home=tmp_path)[0]
    current.path.unlink()

    with pytest.raises(SetupFailure, match="changed during setup"):
        apply_client_setup([current])


def test_setup_does_not_take_ownership_of_a_package_name_prefix(tmp_path: Path) -> None:
    config = tmp_path / ".claude.json"
    config.write_text(
        '{"mcpServers":{"flameox":{"command":"uvx","args":'
        '["--from","flameox-custom==1.0","flameox","mcp","serve"]}}}'
    )

    with pytest.raises(SetupFailure, match="unmanaged Flameox entry"):
        plan_client_setup([SetupClient.CLAUDE], [], home=tmp_path)


def test_client_selection_is_deduplicated_and_rejects_unknown_values() -> None:
    assert parse_setup_clients(["codex", "cursor", "codex"]) == [
        SetupClient.CODEX,
        SetupClient.CURSOR,
    ]
    with pytest.raises(SetupFailure, match="Unknown client"):
        parse_setup_clients(["missing"])
