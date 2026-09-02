from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import MutableMapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import tomlkit
from tomlkit.exceptions import TOMLKitError
from tomlkit.items import InlineTable

from flameox import __version__
from flameox.atomic import atomic_write_text
from flameox.providers.availability import (
    MANAGED_PROVIDER_EXTRAS,
    SYSTEM_PROVIDER_GUIDANCE,
)

DEFAULT_PREPARATION_TIMEOUT_SECONDS = 1_800
MAX_PREPARATION_TIMEOUT_SECONDS = 3_600


class SetupFailure(RuntimeError):
    pass


class ProviderSelectionFailure(SetupFailure):
    pass


class SetupClient(StrEnum):
    CLAUDE = "claude"
    CURSOR = "cursor"
    OPENCODE = "opencode"
    CODEX = "codex"
    GEMINI = "gemini"
    ANTIGRAVITY = "antigravity"

    @property
    def display_name(self) -> str:
        return {
            SetupClient.CLAUDE: "Claude Code",
            SetupClient.CURSOR: "Cursor",
            SetupClient.OPENCODE: "OpenCode",
            SetupClient.CODEX: "Codex",
            SetupClient.GEMINI: "Gemini CLI",
            SetupClient.ANTIGRAVITY: "Google Antigravity",
        }[self]

    def config_path(self, home: Path) -> Path:
        return {
            SetupClient.CLAUDE: home / ".claude.json",
            SetupClient.CURSOR: home / ".cursor" / "mcp.json",
            SetupClient.OPENCODE: home / ".config" / "opencode" / "opencode.jsonc",
            SetupClient.CODEX: home / ".codex" / "config.toml",
            SetupClient.GEMINI: home / ".gemini" / "settings.json",
            SetupClient.ANTIGRAVITY: home / ".gemini" / "config" / "mcp_config.json",
        }[self]

    def is_detected(self, home: Path) -> bool:
        if self is SetupClient.ANTIGRAVITY:
            return any(
                path.exists()
                for path in (
                    home / ".agent",
                    home / ".gemini" / "antigravity",
                    self.config_path(home),
                )
            )
        marker = {
            SetupClient.CLAUDE: home / ".claude",
            SetupClient.CURSOR: home / ".cursor",
            SetupClient.OPENCODE: home / ".config" / "opencode",
            SetupClient.CODEX: home / ".codex",
            SetupClient.GEMINI: self.config_path(home),
        }[self]
        return marker.exists() or self.config_path(home).exists()


SETUP_CLIENTS = tuple(SetupClient)


@dataclass(frozen=True, slots=True)
class ClientSetupPlan:
    client: SetupClient
    path: Path
    action: Literal["create", "update", "already_current"]
    detected: bool
    original: str | None
    content: str


@dataclass(frozen=True, slots=True)
class ClientSetupResult:
    client: SetupClient
    path: Path
    action: Literal["created", "updated", "already_current"]


def parse_setup_clients(values: list[str]) -> list[SetupClient]:
    clients: list[SetupClient] = []
    for value in values:
        try:
            client = SetupClient(value)
        except ValueError as error:
            supported = ", ".join(item.value for item in SETUP_CLIENTS)
            raise SetupFailure(f"Unknown client {value!r}; choose one of: {supported}") from error
        if client not in clients:
            clients.append(client)
    return clients


def detect_setup_clients(home: Path | None = None) -> list[SetupClient]:
    root = home or Path.home()
    return [client for client in SETUP_CLIENTS if client.is_detected(root)]


def _launcher_is_managed(command: object, args: object) -> bool:
    if (
        not isinstance(command, str)
        or Path(command).name != "uvx"
        or not isinstance(args, list)
        or not all(isinstance(item, str) for item in args)
    ):
        return False
    try:
        requirement = args[args.index("--from") + 1]
    except (ValueError, IndexError):
        return False
    try:
        serve_index = args.index("flameox")
    except ValueError:
        return False
    trailing = args[serve_index + 3 :]
    legacy_project_root = (
        len(trailing) == 2 and trailing[0] == "--project-root" and bool(trailing[1])
    )
    return bool(
        re.fullmatch(r"flameox(?:\[[a-z0-9,-]+\])?==[^=]+", requirement)
        and args[serve_index : serve_index + 3] == ["flameox", "mcp", "serve"]
        and (not trailing or legacy_project_root)
    )


def _json_entry(client: SetupClient, command: str, args: list[str]) -> dict[str, object]:
    if client is SetupClient.OPENCODE:
        return {"type": "local", "command": [command, *args], "enabled": True}
    return {"command": command, "args": args}


def _json_plan(
    client: SetupClient,
    path: Path,
    command: str,
    args: list[str],
) -> tuple[str | None, str, Literal["create", "update", "already_current"]]:
    source: str | None = None
    if path.exists():
        try:
            source = path.read_text()
            document = json.loads(source)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SetupFailure(
                f"Could not read {client.display_name} configuration: {path}"
            ) from error
        if not isinstance(document, dict):
            raise SetupFailure(
                f"{client.display_name} configuration must contain a JSON object: {path}"
            )
    else:
        document = {}
    section_name = "mcp" if client is SetupClient.OPENCODE else "mcpServers"
    section = document.get(section_name)
    if section is None:
        section = {}
        document[section_name] = section
    if not isinstance(section, dict):
        raise SetupFailure(
            f"{client.display_name} configuration {section_name!r} must be an object: {path}"
        )
    entry = _json_entry(client, command, args)
    existing = section.get("flameox")
    if isinstance(existing, dict) and all(
        existing.get(key) == value for key, value in entry.items()
    ):
        action: Literal["create", "update", "already_current"] = "already_current"
    else:
        if existing is not None:
            if client is SetupClient.OPENCODE:
                managed = isinstance(existing, dict) and _launcher_is_managed(
                    existing.get("command", [None])[0]
                    if isinstance(existing.get("command"), list) and existing["command"]
                    else None,
                    existing.get("command", [])[1:]
                    if isinstance(existing.get("command"), list)
                    else None,
                )
            else:
                managed = isinstance(existing, dict) and _launcher_is_managed(
                    existing.get("command"), existing.get("args")
                )
            if not managed:
                raise SetupFailure(f"Refusing to replace an unmanaged Flameox entry in {path}.")
        action = "update" if existing is not None else "create"
        section["flameox"] = {**existing, **entry} if isinstance(existing, dict) else entry
    content = f"{json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)}\n"
    return source, content, action


def _codex_plan(
    path: Path,
    command: str,
    args: list[str],
) -> tuple[str | None, str, Literal["create", "update", "already_current"]]:
    source: str | None = None
    if path.exists():
        try:
            source = path.read_text()
            document = tomlkit.parse(source)
        except (OSError, UnicodeError, TOMLKitError) as error:
            raise SetupFailure(f"Could not read Codex configuration: {path}") from error
    else:
        document = tomlkit.document()
    servers = document.get("mcp_servers")
    if servers is None:
        servers = tomlkit.table()
        document["mcp_servers"] = servers
    if not isinstance(servers, MutableMapping):
        raise SetupFailure(f"Codex configuration 'mcp_servers' must be a table: {path}")
    entry = servers.get("flameox")
    expected = {"command": command, "args": args}
    if entry is not None:
        if not isinstance(entry, MutableMapping) or not _launcher_is_managed(
            entry.get("command"), entry.get("args")
        ):
            raise SetupFailure(f"Refusing to replace an unmanaged Flameox entry in {path}.")
        if all(entry.get(key) == value for key, value in expected.items()):
            return source, source or "", "already_current"
        action: Literal["create", "update", "already_current"] = "update"
    else:
        action = "create"
    if entry is None:
        entry = tomlkit.inline_table() if isinstance(servers, InlineTable) else tomlkit.table()
        servers["flameox"] = entry
    entry["command"] = command
    entry["args"] = args
    return source, tomlkit.dumps(document), action


def plan_client_setup(
    clients: list[SetupClient],
    providers: list[str],
    *,
    home: Path | None = None,
) -> list[ClientSetupPlan]:
    root = home or Path.home()
    command, launcher_args = mcp_launcher(providers)
    args = [*launcher_args, "mcp", "serve"]
    plans: list[ClientSetupPlan] = []
    for client in clients:
        path = _client_config_path(client, root)
        if path.is_symlink():
            raise SetupFailure(f"Refusing to replace symbolic-link client configuration: {path}")
        if client is SetupClient.OPENCODE and path.suffix == ".jsonc" and path.exists():
            raise SetupFailure(
                "OpenCode uses a comment-bearing opencode.jsonc configuration. "
                f"Flameox will not rewrite it; add the Flameox MCP entry to {path} manually."
            )
        if client is SetupClient.CODEX:
            original, content, action = _codex_plan(path, command, args)
        else:
            original, content, action = _json_plan(client, path, command, args)
        plans.append(
            ClientSetupPlan(client, path, action, client.is_detected(root), original, content)
        )
    return plans


def _client_config_path(client: SetupClient, home: Path) -> Path:
    if client is not SetupClient.OPENCODE:
        return client.config_path(home)
    directory = home / ".config" / "opencode"
    candidates = [directory / name for name in ("opencode.jsonc", "opencode.json", "config.json")]
    return next((path for path in candidates if path.exists()), candidates[0])


def apply_client_setup(plans: list[ClientSetupPlan]) -> list[ClientSetupResult]:
    results: list[ClientSetupResult] = []
    for plan in plans:
        try:
            if plan.path.is_symlink():
                raise SetupFailure(
                    f"Refusing to replace symbolic-link client configuration: {plan.path}"
                )
            current = plan.path.read_text() if plan.path.exists() else None
            if current != plan.original:
                raise SetupFailure(
                    f"{plan.client.display_name} configuration changed during setup: {plan.path}"
                )
            if plan.action != "already_current":
                atomic_write_text(plan.path, plan.content)
        except SetupFailure:
            raise
        except (OSError, UnicodeError) as error:
            raise SetupFailure(
                f"Could not update {plan.client.display_name} configuration: {plan.path}. "
                "Earlier selected clients may already have been configured."
            ) from error
        if plan.action == "already_current":
            action: Literal["created", "updated", "already_current"] = "already_current"
        else:
            action = "created" if plan.action == "create" else "updated"
        results.append(ClientSetupResult(plan.client, plan.path, action))
    return results


@dataclass(frozen=True, slots=True)
class ExternalRequirement:
    provider_id: str
    guidance: str


@dataclass(frozen=True, slots=True)
class ProviderPreparation:
    requested_providers: list[str]
    prepared_managed_providers: list[str]
    external_requirements: list[ExternalRequirement]
    preparation_command: list[str]
    launcher_command: str
    launcher_args: list[str]

    @property
    def preparation_status(self) -> Literal["prepared", "not_applicable"]:
        return "prepared" if self.prepared_managed_providers else "not_applicable"

    @property
    def restart_required(self) -> bool:
        return bool(self.prepared_managed_providers)


def _validate_providers(providers: list[str]) -> None:
    unknown = sorted(
        set(providers).difference(MANAGED_PROVIDER_EXTRAS).difference(SYSTEM_PROVIDER_GUIDANCE)
    )
    if unknown:
        supported = ", ".join(sorted(MANAGED_PROVIDER_EXTRAS | SYSTEM_PROVIDER_GUIDANCE))
        raise ProviderSelectionFailure(
            f"Unknown provider {unknown[0]!r}; choose one of: {supported}"
        )


def mcp_launcher(providers: list[str]) -> tuple[str, list[str]]:
    """Return a version-bound MCP launcher for client configuration."""

    _validate_providers(providers)
    extras = sorted(
        {
            MANAGED_PROVIDER_EXTRAS[provider]
            for provider in providers
            if provider in MANAGED_PROVIDER_EXTRAS
        }
    )
    extras_suffix = f"[{','.join(extras)}]" if extras else ""
    requirement = f"flameox{extras_suffix}=={__version__}"
    return (
        "uvx",
        ["--python", "3.12", "--from", requirement, "flameox"],
    )


def external_provider_requirements(providers: list[str]) -> list[ExternalRequirement]:
    _validate_providers(providers)
    return [
        ExternalRequirement(provider, SYSTEM_PROVIDER_GUIDANCE[provider])
        for provider in dict.fromkeys(providers)
        if provider in SYSTEM_PROVIDER_GUIDANCE
    ]


def _decode_stderr(stderr: bytes | str | None) -> str:
    if not stderr:
        return ""
    return stderr.strip() if isinstance(stderr, str) else stderr.decode(errors="replace").strip()


def _failure_message(message: str, stderr: bytes | str | None) -> str:
    diagnostic = _decode_stderr(stderr)
    return f"{message}\n\nuvx stderr:\n{diagnostic}" if diagnostic else message


def prepare_providers(
    providers: list[str],
    timeout_seconds: int = DEFAULT_PREPARATION_TIMEOUT_SECONDS,
) -> ProviderPreparation:
    if not 1 <= timeout_seconds <= MAX_PREPARATION_TIMEOUT_SECONDS:
        raise SetupFailure(
            f"timeout_seconds must be between 1 and {MAX_PREPARATION_TIMEOUT_SECONDS}"
        )
    requested = list(dict.fromkeys(providers))
    _validate_providers(requested)
    managed = [item for item in requested if item in MANAGED_PROVIDER_EXTRAS]
    external = [
        ExternalRequirement(item, SYSTEM_PROVIDER_GUIDANCE[item])
        for item in requested
        if item in SYSTEM_PROVIDER_GUIDANCE
    ]
    launcher_command, launcher_args = mcp_launcher(managed)
    server_args = [*launcher_args, "mcp", "serve"]
    preparation_command: list[str] = []

    if managed:
        uvx = shutil.which(launcher_command)
        if uvx is None:
            raise SetupFailure("Provider preparation requires uvx on PATH.")
        preparation_command = [uvx, *launcher_args, "--version"]
        try:
            completed = subprocess.run(
                preparation_command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
        except OSError as error:
            raise SetupFailure("uvx could not prepare the provider environment.") from error
        except subprocess.TimeoutExpired as error:
            raise SetupFailure(
                _failure_message(
                    f"uvx provider preparation exceeded {timeout_seconds} seconds.", error.stderr
                )
            ) from error
        if completed.returncode != 0:
            raise SetupFailure(
                _failure_message(
                    f"uvx provider preparation exited with status {completed.returncode}.",
                    completed.stderr,
                )
            )

    return ProviderPreparation(
        requested,
        managed,
        external,
        preparation_command,
        launcher_command,
        server_args,
    )
