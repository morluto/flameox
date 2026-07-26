from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from flamo.adapters import RuntimeInstallation, SetupClient
from flamo.application import SetupOperation, SetupService
from flamo.domain import DomainError, ErrorCode
from flamo.storage.atomic import atomic_write_bytes, atomic_write_json


class FakeRuntime:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.versions: set[str] = set()
        self.verified: list[Path] = []

    def executable(self, version: str) -> Path:
        return self.root / "runtimes" / version / "bin" / "flamo"

    def installed_versions(self) -> tuple[str, ...]:
        return tuple(sorted(self.versions, reverse=True))

    async def install(self, version: str) -> RuntimeInstallation:
        executable = self.executable(version)
        installed = version not in self.versions
        self.versions.add(version)
        self.verified.append(executable)
        return RuntimeInstallation(version, executable, installed)

    async def verify(self, executable: Path, version: str) -> None:
        assert executable == self.executable(version)
        self.verified.append(executable)


def make_service(tmp_path: Path) -> tuple[SetupService, FakeRuntime, Path]:
    home = tmp_path / "home"
    data = tmp_path / "data"
    home.mkdir()
    runtime = FakeRuntime(data)
    return SetupService(home=home, data_root=data, runtime=runtime), runtime, home


@pytest.mark.anyio
async def test_setup_connects_only_explicitly_selected_clients(tmp_path: Path) -> None:
    service, runtime, home = make_service(tmp_path)
    (home / ".claude").mkdir()
    (home / ".cursor").mkdir()

    plan = service.plan(
        operation=SetupOperation.CONFIGURE,
        clients=(SetupClient.CLAUDE,),
        version="0.1.0",
    )
    report = await service.apply(plan)

    configured = json.loads((home / ".claude.json").read_text())
    assert configured["mcpServers"]["flamo"] == {
        "command": str(runtime.executable("0.1.0")),
        "args": ["mcp", "serve", "--project-root", "."],
    }
    assert not (home / ".cursor" / "mcp.json").exists()
    assert report.changed_clients == (SetupClient.CLAUDE,)


@pytest.mark.anyio
async def test_setup_is_idempotent_and_preserves_unrelated_json(tmp_path: Path) -> None:
    service, _, home = make_service(tmp_path)
    config = home / ".gemini" / "settings.json"
    config.parent.mkdir()
    config.write_text('{"theme": "dark", "mcpServers": {"other": {"command": "x"}}}\n')

    first = service.plan(
        operation=SetupOperation.CONFIGURE,
        clients=(SetupClient.GEMINI,),
        version="0.1.0",
    )
    await service.apply(first)
    second = service.plan(
        operation=SetupOperation.CONFIGURE,
        clients=(SetupClient.GEMINI,),
        version="0.1.0",
    )
    report = await service.apply(second)

    configured = json.loads(config.read_text())
    assert configured["theme"] == "dark"
    assert configured["mcpServers"]["other"] == {"command": "x"}
    assert report.changed_clients == ()
    assert report.unchanged_clients == (SetupClient.GEMINI,)


@pytest.mark.anyio
async def test_setup_refuses_config_changed_after_preview(tmp_path: Path) -> None:
    service, _, home = make_service(tmp_path)
    config = home / ".claude.json"
    config.write_text("{}\n")
    plan = service.plan(
        operation=SetupOperation.CONFIGURE,
        clients=(SetupClient.CLAUDE,),
        version="0.1.0",
    )
    config.write_text('{"new": true}\n')

    with pytest.raises(DomainError) as caught:
        await service.apply(plan)

    assert caught.value.code is ErrorCode.REVISION_CONFLICT
    assert json.loads(config.read_text()) == {"new": True}


@pytest.mark.anyio
async def test_setup_refuses_stale_plan_after_active_runtime_changes(
    tmp_path: Path,
) -> None:
    service, runtime, home = make_service(tmp_path)
    competing = SetupService(
        home=home,
        data_root=service.data_root,
        runtime=runtime,
    )
    first = service.plan(
        operation=SetupOperation.CONFIGURE,
        clients=(SetupClient.CLAUDE,),
        version="0.1.0",
    )
    stale = competing.plan(
        operation=SetupOperation.CONFIGURE,
        clients=(SetupClient.GEMINI,),
        version="0.2.0",
    )

    await service.apply(first)
    with pytest.raises(DomainError) as caught:
        await competing.apply(stale)

    assert caught.value.code is ErrorCode.REVISION_CONFLICT
    assert service.inspect().active_version == "0.1.0"
    assert "0.2.0" not in runtime.versions
    assert not (home / ".gemini" / "settings.json").exists()


@pytest.mark.anyio
async def test_setup_recovers_interrupted_config_transaction(tmp_path: Path) -> None:
    service, _, home = make_service(tmp_path)
    config = home / ".claude.json"
    config.write_text('{"before": true}\n')
    original = config.read_bytes()
    config.write_text('{"partial": true}\n')
    service.data_root.mkdir(parents=True)
    atomic_write_json(
        service.journal_path,
        {
            "schema_version": 1,
            "mutations": [
                {
                    "path": str(config),
                    "original": "eyJiZWZvcmUiOiB0cnVlfQo=",
                    "original_sha256": hashlib.sha256(original).hexdigest(),
                    "updated_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "mode": 0o600,
                }
            ],
        },
    )

    plan = service.plan(
        operation=SetupOperation.VERIFY,
        clients=(),
        version=None,
    )
    with pytest.raises(DomainError) as caught:
        await service.apply(plan)

    assert caught.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert config.read_bytes() == original
    assert not service.journal_path.exists()


def test_setup_recovers_interrupted_transaction_before_planning(tmp_path: Path) -> None:
    service, runtime, home = make_service(tmp_path)
    config = home / ".claude.json"
    original = b'{"mcpServers": {}}\n'
    staged = (
        json.dumps(
            {
                "mcpServers": {
                    "flamo": {
                        "command": str(runtime.executable("0.1.0")),
                        "args": ["mcp", "serve", "--project-root", "."],
                    }
                }
            }
        )
        + "\n"
    ).encode()
    config.write_bytes(staged)
    service.data_root.mkdir(parents=True)
    atomic_write_json(
        service.journal_path,
        {
            "schema_version": 1,
            "mutations": [
                {
                    "path": str(config),
                    "original": base64.b64encode(original).decode(),
                    "original_sha256": hashlib.sha256(original).hexdigest(),
                    "updated_sha256": hashlib.sha256(staged).hexdigest(),
                    "mode": 0o600,
                }
            ],
        },
    )

    plan = service.plan(
        operation=SetupOperation.CONFIGURE,
        clients=(SetupClient.CLAUDE,),
        version="0.1.0",
    )

    assert plan.edits[0].action.value == "update"
    assert plan.edits[0].original == original
    assert config.read_bytes() == original
    assert not service.journal_path.exists()


def test_codex_toml_edit_preserves_comments(tmp_path: Path) -> None:
    service, _, home = make_service(tmp_path)
    config = home / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text('# keep this note\nmodel = "gpt-5"\n')

    plan = service.plan(
        operation=SetupOperation.CONFIGURE,
        clients=(SetupClient.CODEX,),
        version="0.1.0",
    )

    assert plan.edits[0].updated is not None
    updated = plan.edits[0].updated.decode()
    assert "# keep this note" in updated
    assert "[mcp_servers.flamo]" in updated


def test_opencode_uses_its_native_local_server_shape(tmp_path: Path) -> None:
    service, runtime, home = make_service(tmp_path)
    (home / ".config" / "opencode").mkdir(parents=True)

    plan = service.plan(
        operation=SetupOperation.CONFIGURE,
        clients=(SetupClient.OPENCODE,),
        version="0.1.0",
    )

    assert plan.edits[0].updated is not None
    configured = json.loads(plan.edits[0].updated)
    assert configured["mcp"]["flamo"] == {
        "type": "local",
        "command": [
            str(runtime.executable("0.1.0")),
            "mcp",
            "serve",
            "--project-root",
            ".",
        ],
        "cwd": ".",
        "enabled": True,
    }


def test_direct_python_setup_refuses_to_overwrite_jsonc_without_helper(
    tmp_path: Path,
) -> None:
    service, _, home = make_service(tmp_path)
    config = home / ".config" / "opencode" / "opencode.jsonc"
    config.parent.mkdir(parents=True)
    config.write_text("{\n  // user note\n  \"mcp\": {}\n}\n")

    with pytest.raises(DomainError) as caught:
        service.plan(
            operation=SetupOperation.CONFIGURE,
            clients=(SetupClient.OPENCODE,),
            version="0.1.0",
        )

    assert caught.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert config.read_text().startswith("{\n  // user note")


def test_jsonc_helper_does_not_relax_standard_json_clients(tmp_path: Path) -> None:
    home = tmp_path / "home"
    data = tmp_path / "data"
    home.mkdir()
    config = home / ".claude.json"
    config.write_text('{\n  // invalid for Claude\n  "mcpServers": {}\n}\n')
    service = SetupService(
        home=home,
        data_root=data,
        jsonc_helper=tmp_path / "helper-that-must-not-run.cjs",
        runtime=FakeRuntime(data),
    )

    with pytest.raises(DomainError) as caught:
        service.plan(
            operation=SetupOperation.CONFIGURE,
            clients=(SetupClient.CLAUDE,),
            version="0.1.0",
        )

    assert caught.value.code is ErrorCode.EXECUTION_REFUSED
    assert config.read_text().startswith("{\n  // invalid for Claude")


@pytest.mark.anyio
async def test_setup_restores_every_config_after_partial_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _, home = make_service(tmp_path)
    claude = home / ".claude.json"
    gemini = home / ".gemini" / "settings.json"
    claude.write_text('{"before": "claude"}\n')
    gemini.parent.mkdir()
    gemini.write_text('{"before": "gemini"}\n')
    originals = {claude: claude.read_bytes(), gemini: gemini.read_bytes()}
    plan = service.plan(
        operation=SetupOperation.CONFIGURE,
        clients=(SetupClient.CLAUDE, SetupClient.GEMINI),
        version="0.1.0",
    )
    real_atomic_write = atomic_write_bytes
    calls = 0

    def fail_second_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        real_atomic_write(path, payload, mode=mode)

    monkeypatch.setattr("flamo.application.setup.atomic_write_bytes", fail_second_write)

    with pytest.raises(OSError, match="injected write failure"):
        await service.apply(plan)

    assert claude.read_bytes() == originals[claude]
    assert gemini.read_bytes() == originals[gemini]
    assert not service.install_manifest.exists()
    assert not service.journal_path.exists()


@pytest.mark.anyio
async def test_remove_preserves_other_mcp_servers(tmp_path: Path) -> None:
    service, runtime, home = make_service(tmp_path)
    config = home / ".claude.json"
    config.write_text(
        '{"mcpServers":{"other":{"command":"other"},"flamo":{"command":"old"}}}\n'
    )
    plan = service.plan(
        operation=SetupOperation.REMOVE,
        clients=(SetupClient.CLAUDE,),
        version=None,
    )

    report = await service.apply(plan)

    assert json.loads(config.read_text()) == {
        "mcpServers": {"other": {"command": "other"}}
    }
    assert report.changed_clients == (SetupClient.CLAUDE,)
    assert runtime.verified == []
    assert not service.install_manifest.exists()


@pytest.mark.anyio
async def test_rollback_repoints_clients_to_an_installed_runtime(tmp_path: Path) -> None:
    service, runtime, home = make_service(tmp_path)
    runtime.versions.add("0.0.9")
    config = home / ".claude.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "flamo": {
                        "command": str(runtime.executable("0.1.0")),
                        "args": ["mcp", "serve", "--project-root", "."],
                    }
                }
            }
        )
    )
    plan = service.plan(
        operation=SetupOperation.ROLLBACK,
        clients=(SetupClient.CLAUDE,),
        version="0.0.9",
    )

    await service.apply(plan)

    configured = json.loads(config.read_text())
    assert configured["mcpServers"]["flamo"]["command"] == str(
        runtime.executable("0.0.9")
    )
    assert service.inspect().active_version == "0.0.9"


def test_recovery_refuses_to_overwrite_a_post_crash_user_edit(
    tmp_path: Path,
) -> None:
    service, _, home = make_service(tmp_path)
    config = home / ".claude.json"
    original = b'{"state": "original"}\n'
    staged = b'{"state": "staged"}\n'
    user_edit = b'{"state": "user-edit"}\n'
    config.write_bytes(user_edit)
    service.data_root.mkdir(parents=True)
    atomic_write_json(
        service.journal_path,
        {
            "schema_version": 1,
            "mutations": [
                {
                    "path": str(config),
                    "original": base64.b64encode(original).decode(),
                    "original_sha256": hashlib.sha256(original).hexdigest(),
                    "updated_sha256": hashlib.sha256(staged).hexdigest(),
                    "mode": 0o600,
                }
            ],
        },
    )
    with pytest.raises(DomainError) as caught:
        service.plan(
            operation=SetupOperation.VERIFY,
            clients=(),
            version=None,
        )

    assert caught.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert config.read_bytes() == user_edit
    assert service.journal_path.exists()
