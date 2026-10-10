from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.error import URLError

import json5
import pytest
import tomlkit
from typer.testing import CliRunner

from flameox.cli import app
from flameox.providers.environment import SetupFailure
from flameox.setup import (
    SETUP_CLIENTS,
    ClientUpdatePlan,
    SetupClient,
    apply_client_setup,
    plan_client_setup,
    read_client_installations,
)
from flameox.updates import latest_release_version, prepare_updated_releases


def plan_client_update(
    clients: list[SetupClient], version: str, *, home: Path
) -> list[ClientUpdatePlan]:
    return [item.plan_update(version) for item in read_client_installations(clients, home=home)]


@pytest.mark.parametrize("client", SETUP_CLIENTS)
def test_update_preserves_providers_client_settings_and_disabled_state(
    tmp_path: Path, client: SetupClient
) -> None:
    initial = plan_client_setup([client], ["memray", "py-spy"], home=tmp_path)[0]
    apply_client_setup([initial])
    source = initial.path.read_text()
    if client is SetupClient.CODEX:
        document = tomlkit.parse(source)
        section = document["mcp_servers"]
    else:
        document = json5.loads(source)
        section = document["mcp" if client is SetupClient.OPENCODE else "mcpServers"]
    entry = section["flameox"]
    entry["env"] = {"KEEP": "value"}
    if client is SetupClient.OPENCODE:
        entry["enabled"] = False
        entry["command"].extend(["--limits", '{"max_rows":10}'])
    else:
        entry["args"].extend(["--limits", '{"max_rows":10}'])
    source = tomlkit.dumps(document) if client is SetupClient.CODEX else json.dumps(document)
    if client in {SetupClient.CODEX, SetupClient.OPENCODE}:
        source = (
            "# keep comment\n" if client is SetupClient.CODEX else "// keep comment\n"
        ) + source
    initial.path.write_text(source)

    planned = plan_client_update([client], "9.8.7", home=tmp_path)
    assert planned[0].requirement == "flameox[cpu,memory]==9.8.7"
    assert initial.path.read_text() == source
    apply_client_setup([plan.setup for plan in planned])
    content = initial.path.read_text()
    updated = tomlkit.parse(content) if client is SetupClient.CODEX else json5.loads(content)
    updated_section = updated[
        "mcp_servers"
        if client is SetupClient.CODEX
        else "mcp"
        if client is SetupClient.OPENCODE
        else "mcpServers"
    ]
    expected = entry.copy()
    if client is SetupClient.OPENCODE:
        expected["command"][6] = "flameox[cpu,memory]==9.8.7"
    else:
        expected["args"][5] = "flameox[cpu,memory]==9.8.7"
    assert updated_section["flameox"] == expected
    if client in {SetupClient.CODEX, SetupClient.OPENCODE}:
        assert "keep comment" in content
    repeated = plan_client_update([client], "9.8.7", home=tmp_path)
    assert repeated[0].setup.action == "already_current"
    assert repeated[0].setup.content == content


@pytest.mark.parametrize("option", ["--check", "--dry-run"])
def test_update_preview_preserves_newer_releases_and_does_not_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = plan_client_setup([SetupClient.CURSOR, SetupClient.CODEX], [], home=tmp_path)
    apply_client_setup(plans)
    apply_client_setup([plan_client_update([SetupClient.CODEX], "9.8.7", home=tmp_path)[0].setup])
    originals = {plan.path: plan.path.read_text() for plan in plans}
    monkeypatch.setattr("flameox.cli.latest_release_version", lambda: "1.2.3")
    monkeypatch.setattr(
        "flameox.cli.prepare_updated_releases",
        lambda *_args: pytest.fail("preview prepared release"),
    )

    result = CliRunner().invoke(app, ["update", option, "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["update_available"] is True
    assert payload["restart_required"] is False
    assert {client["id"]: client["status"] for client in payload["clients"]} == {
        "cursor": "update_available",
        "codex": "already_current",
    }
    assert {path: path.read_text() for path in originals} == originals


def test_update_prepares_all_environments_before_writing_and_allows_explicit_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plans = plan_client_setup([SetupClient.CURSOR], ["memray"], home=tmp_path)
    plans += plan_client_setup([SetupClient.CODEX], ["py-spy"], home=tmp_path)
    apply_client_setup(plans)
    originals = {plan.path: plan.path.read_text() for plan in plans}
    monkeypatch.setattr(
        "flameox.cli.latest_release_version", lambda: pytest.fail("explicit release contacted PyPI")
    )

    def fail_preparation(plans: list[ClientUpdatePlan], timeout: int) -> None:
        assert {plan.requirement for plan in plans} == {
            "flameox[memory]==0.1.0",
            "flameox[cpu]==0.1.0",
        }
        assert {path: path.read_text() for path in originals} == originals
        raise SetupFailure("second environment failed")

    monkeypatch.setattr("flameox.cli.prepare_updated_releases", fail_preparation)
    runner = CliRunner()
    failure = runner.invoke(app, ["update", "--version", "0.1.0", "--json"])
    assert failure.exit_code == 2, failure.output
    assert "second environment failed" in failure.output
    assert {path: path.read_text() for path in originals} == originals

    def edit_during_preparation(*_args: object) -> None:
        plans[-1].path.write_text(originals[plans[-1].path] + "\n# user edit\n")

    monkeypatch.setattr("flameox.cli.prepare_updated_releases", edit_during_preparation)
    conflict = runner.invoke(app, ["update", "--version", "0.1.0", "--json"])
    assert conflict.exit_code == 2, conflict.output
    assert "configuration changed" in conflict.output
    assert plans[0].path.read_text() == originals[plans[0].path]
    assert plans[-1].path.read_text().endswith("# user edit\n")
    plans[-1].path.write_text(originals[plans[-1].path])

    monkeypatch.setattr("flameox.cli.prepare_updated_releases", lambda *_args: None)
    success = runner.invoke(app, ["update", "--version", "0.1.0", "--json"])
    assert success.exit_code == 0, success.output
    payload = json.loads(success.stdout)
    assert payload["restart_required"] is True
    assert all(client["status"] == "updated" for client in payload["clients"])
    assert all("==0.1.0" in path.read_text() for path in originals)
    assert not SetupClient.CLAUDE.config_path(tmp_path).exists()


@pytest.mark.parametrize("kind", ["custom", "duplicate", "symlink", "missing"])
def test_update_rejects_unrecognized_registration_before_release_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = SetupClient.CURSOR.config_path(tmp_path)
    config.parent.mkdir(parents=True)
    if kind == "custom":
        config.write_text('{"mcpServers":{"flameox":{"command":"custom","args":[]}}}')
    elif kind == "duplicate":
        config.write_text('{"mcpServers":{},"mcpServers":{}}')
    elif kind == "symlink":
        target = tmp_path / "config.json"
        target.write_text("{}")
        config.symlink_to(target)
    monkeypatch.setattr(
        "flameox.cli.latest_release_version",
        lambda: pytest.fail("invalid registration contacted PyPI"),
    )
    result = CliRunner().invoke(app, ["update", "--client", "cursor", "--check"])
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "metadata",
    [
        {"info": {"name": "flameox", "version": "1.2.3"}},
        {"info": {"name": "other", "version": "1.2.3"}},
        {"info": {"name": "flameox", "version": "1.2.3rc1"}},
        {"info": {"name": "flameox", "version": None}},
        [],
    ],
)
def test_release_check_validates_pypi_response(
    monkeypatch: pytest.MonkeyPatch, metadata: object
) -> None:
    monkeypatch.setattr(
        "flameox.updates.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(json.dumps(metadata).encode()),
    )
    if metadata == {"info": {"name": "flameox", "version": "1.2.3"}}:
        assert latest_release_version() == "1.2.3"
    else:
        with pytest.raises(SetupFailure):
            latest_release_version()


def test_release_check_reports_offline_failure_without_changing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def offline(*_args: object, **_kwargs: object) -> None:
        raise URLError("offline")

    monkeypatch.setattr("flameox.updates.urlopen", offline)
    with pytest.raises(SetupFailure, match="--version"):
        latest_release_version()


@pytest.mark.parametrize("client", [SetupClient.CODEX, SetupClient.OPENCODE])
def test_update_migrates_legacy_launchers_without_removing_internal_comments(
    tmp_path: Path, client: SetupClient
) -> None:
    config = client.config_path(tmp_path)
    config.parent.mkdir(parents=True)
    if client is SetupClient.CODEX:
        source = """[mcp_servers.flameox]
command = "uvx"
args = [
  "--python", "3.12", "--from",
  "flameox[cpu]==0.1.0", # provider rationale
  "flameox", "mcp", "serve",
]
"""
    else:
        source = """{
  "mcp": {
    "flameox": {
      // disabled until next experiment
      "enabled": false,
      "type": "local",
      "command": [
        "uvx", "--python", "3.12", "--from",
        "flameox[cpu]==0.1.0", // provider rationale
        "flameox", "mcp", "serve"
      ]
    }
  }
}"""
    config.write_text(source)
    installation = read_client_installations([client], home=tmp_path)[0]
    planned = installation.plan_update("1.2.3")
    assert "provider rationale" in planned.setup.content
    assert "--no-config" in planned.setup.content
    assert "flameox[cpu]==1.2.3" in planned.setup.content
    if client is SetupClient.OPENCODE:
        assert "disabled until next experiment" in planned.setup.content
    # Planning is repeatable from the same immutable source snapshot.
    assert installation.plan_update("1.2.3").setup.content == planned.setup.content
    apply_client_setup([planned.setup])
    assert read_client_installations([client], home=tmp_path)[0].version == "1.2.3"


def test_update_checks_from_development_build_and_rejects_intervening_config_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("flameox.cli.__version__", "1.2.3+local")
    plans = plan_client_setup([SetupClient.CURSOR], [], home=tmp_path)
    apply_client_setup(plans)
    config = plans[0].path

    def lookup() -> str:
        config.unlink()
        return "9.8.7"

    monkeypatch.setattr("flameox.cli.latest_release_version", lookup)
    monkeypatch.setattr("flameox.cli.prepare_updated_releases", lambda *_args: None)
    result = CliRunner().invoke(app, ["update", "--client", "cursor", "--json"])
    assert result.exit_code == 2, result.output
    assert "configuration changed" in result.output
    assert "Traceback" not in result.output
    assert not config.exists()


def test_release_check_bounds_metadata_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flameox.updates.urlopen", lambda *_args, **_kwargs: io.BytesIO(b" " * (1024 * 1024 + 1))
    )
    with pytest.raises(SetupFailure, match="size limit"):
        latest_release_version()


@pytest.mark.parametrize("source", ["client", "inherited"])
def test_update_rejects_unforwardable_index_credentials_without_disclosing_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    plans = plan_client_setup([SetupClient.CURSOR], [], home=tmp_path)
    apply_client_setup(plans)
    if source == "client":
        document = json.loads(plans[0].path.read_text())
        document["mcpServers"]["flameox"]["env"] = {"UV_INDEX_PRIVATE_PASSWORD": "PRIVATE-VALUE"}
        plans[0].path.write_text(json.dumps(document))
    else:
        monkeypatch.setenv("UV_INDEX_PRIVATE_PASSWORD", "PRIVATE-VALUE")
    planned = plan_client_update([SetupClient.CURSOR], "1.2.3", home=tmp_path)
    original = plans[0].path.read_text()
    with pytest.raises(SetupFailure, match="UV_INDEX_PRIVATE_PASSWORD") as failure:
        prepare_updated_releases(planned, 10)
    assert "PRIVATE-VALUE" not in str(failure.value)
    assert plans[0].path.read_text() == original


def test_update_reports_partial_publication_if_later_file_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flameox.atomic import atomic_write_text

    monkeypatch.setenv("HOME", str(tmp_path))
    plans = plan_client_setup([SetupClient.CURSOR, SetupClient.CODEX], [], home=tmp_path)
    apply_client_setup(plans)
    original_codex = plans[-1].path.read_text()
    monkeypatch.setattr("flameox.cli.prepare_updated_releases", lambda *_args: None)

    def fail_codex(path: Path, content: str) -> None:
        if path == plans[-1].path:
            raise OSError("disk write failed")
        atomic_write_text(path, content)

    monkeypatch.setattr("flameox.setup.atomic_write_text", fail_codex)
    result = CliRunner().invoke(app, ["update", "--version", "9.8.7"])
    assert result.exit_code == 2, result.output
    assert "restart or reconnect" in result.output
    assert "==9.8.7" in plans[0].path.read_text()
    assert plans[-1].path.read_text() == original_codex
