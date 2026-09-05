from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import MutableMapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import json5
import tomlkit
from tomlkit.exceptions import TOMLKitError
from tomlkit.items import InlineTable

from flameox import __version__
from flameox.atomic import atomic_write_text
from flameox.providers.availability import (
    MANAGED_PROVIDER_EXTRAS,
    SYSTEM_PROVIDER_GUIDANCE,
)
from flameox.providers.environment import (
    DEFAULT_PREPARATION_TIMEOUT_SECONDS as DEFAULT_PREPARATION_TIMEOUT_SECONDS,
)
from flameox.providers.environment import (
    MAX_PREPARATION_TIMEOUT_SECONDS as MAX_PREPARATION_TIMEOUT_SECONDS,
)
from flameox.providers.environment import (
    ExternalRequirement as ExternalRequirement,
)
from flameox.providers.environment import (
    ProviderPreparation as ProviderPreparation,
)
from flameox.providers.environment import (
    ProviderSelectionFailure as ProviderSelectionFailure,
)
from flameox.providers.environment import (
    SetupFailure as SetupFailure,
)
from flameox.providers.environment import (
    _validate_providers,
)
from flameox.providers.environment import (
    active_provider_status as active_provider_status,
)
from flameox.providers.environment import (
    external_provider_requirements as external_provider_requirements,
)
from flameox.providers.environment import (
    mcp_launcher as mcp_launcher,
)

PATH_CLI_PROBE_TIMEOUT_SECONDS = 5


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

    def active_config_path(self, home: Path) -> Path:
        return _client_config_path(self, home)


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


@dataclass(frozen=True, slots=True)
class CliVersionAdvisory:
    executable: str
    cli_version: str
    mcp_version: str

    @property
    def message(self) -> str:
        return (
            f"Direct CLI commands use Flameox {self.cli_version} at {self.executable}, while the "
            f"configured MCP launcher uses {self.mcp_version}. Manage that CLI separately if "
            "you want the versions aligned."
        )


def path_cli_version_advisory() -> CliVersionAdvisory | None:
    """Return a non-fatal advisory when the PATH CLI differs from this release."""

    executable = shutil.which("flameox")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=PATH_CLI_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout
    version = (
        output.decode(errors="replace").strip() if isinstance(output, bytes) else output.strip()
    )
    if not version or "\n" in version or version == __version__:
        return None
    return CliVersionAdvisory(executable, version, __version__)


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


def _json_entry(client: SetupClient, command: str, args: list[str]) -> dict[str, object]:
    if client is SetupClient.OPENCODE:
        return {"type": "local", "command": [command, *args], "enabled": True}
    return {"command": command, "args": args}


@dataclass(frozen=True, slots=True)
class _JsoncProperty:
    key: str
    key_start: int
    value_start: int
    value_end: int
    has_trailing_comma: bool


def _jsonc_skip_trivia(source: str, index: int) -> int:
    while True:
        while index < len(source) and source[index].isspace():
            index += 1
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline == -1 else newline + 1
            continue
        if source.startswith("/*", index):
            comment_end = source.find("*/", index + 2)
            if comment_end == -1:
                raise SetupFailure("OpenCode configuration contains an unterminated comment.")
            index = comment_end + 2
            continue
        return index


def _jsonc_string_end(source: str, index: int) -> int:
    quote = source[index]
    index += 1
    while index < len(source):
        character = source[index]
        if character == "\\":
            index += 2
        elif character == quote:
            return index + 1
        else:
            index += 1
    raise SetupFailure("OpenCode configuration contains an unterminated string.")


def _jsonc_value_end(source: str, index: int) -> int:
    index = _jsonc_skip_trivia(source, index)
    if index == len(source):
        raise SetupFailure("OpenCode configuration ends before a value.")
    if source[index] in "\"'":
        return _jsonc_string_end(source, index)
    if source[index] not in "[{":
        while index < len(source) and source[index] not in ",}]" and not source[index].isspace():
            index += 1
        return index

    closing = {"{": "}", "[": "]"}
    stack = [closing[source[index]]]
    index += 1
    while stack:
        index = _jsonc_skip_trivia(source, index)
        if index == len(source):
            raise SetupFailure("OpenCode configuration ends before a value is complete.")
        character = source[index]
        if character in "\"'":
            index = _jsonc_string_end(source, index)
        elif character in closing:
            stack.append(closing[character])
            index += 1
        elif character == stack[-1]:
            stack.pop()
            index += 1
        else:
            index += 1
    return index


def _jsonc_object_properties(source: str, object_start: int) -> tuple[list[_JsoncProperty], int]:
    if object_start == len(source) or source[object_start] != "{":
        raise SetupFailure("OpenCode configuration must contain a JSON object.")
    properties: list[_JsoncProperty] = []
    index = _jsonc_skip_trivia(source, object_start + 1)
    while index < len(source) and source[index] != "}":
        key_start = index
        if source[index] in "\"'":
            key_end = _jsonc_string_end(source, index)
            key = json5.loads(source[index:key_end])
            if not isinstance(key, str):
                raise SetupFailure("OpenCode configuration contains an invalid property name.")
        else:
            key_end = index
            while key_end < len(source) and (source[key_end].isalnum() or source[key_end] in "_$"):
                key_end += 1
            key = source[index:key_end]
            if not key:
                raise SetupFailure("OpenCode configuration contains an invalid property name.")
        index = _jsonc_skip_trivia(source, key_end)
        if index == len(source) or source[index] != ":":
            raise SetupFailure("OpenCode configuration is missing a property separator.")
        value_start = _jsonc_skip_trivia(source, index + 1)
        value_end = _jsonc_value_end(source, value_start)
        index = _jsonc_skip_trivia(source, value_end)
        has_trailing_comma = index < len(source) and source[index] == ","
        properties.append(
            _JsoncProperty(key, key_start, value_start, value_end, has_trailing_comma)
        )
        if has_trailing_comma:
            index = _jsonc_skip_trivia(source, index + 1)
        if index == len(source) or source[index] == "}":
            break
    if index == len(source) or source[index] != "}":
        raise SetupFailure("OpenCode configuration contains an incomplete JSON object.")
    return properties, index


def _jsonc_line_indent(source: str, index: int, fallback: str) -> str:
    line_start = source.rfind("\n", 0, index) + 1
    indentation = source[line_start:index]
    return indentation if indentation.strip() == "" else fallback


def _jsonc_property_text(name: str, value: object, indent: str) -> str:
    rendered_value = json.dumps(value, ensure_ascii=False, indent=2)
    return f"{indent}{json.dumps(name)}: {rendered_value.replace(chr(10), chr(10) + indent)}"


def _jsonc_insert_property(
    source: str,
    object_end: int,
    properties: list[_JsoncProperty],
    property_text: str,
    fallback_indent: str,
) -> str:
    closing_indent = _jsonc_line_indent(source, object_end, fallback_indent)
    if properties and not properties[-1].has_trailing_comma:
        last_value_end = properties[-1].value_end
        source = f"{source[:last_value_end]},{source[last_value_end:]}"
        object_end += 1
    return f"{source[:object_end]}\n{property_text}\n{closing_indent}{source[object_end:]}"


def _jsonc_update_mcp_entry(source: str, section_name: str, entry: object) -> str:
    root_start = _jsonc_skip_trivia(source, 0)
    root_properties, root_end = _jsonc_object_properties(source, root_start)
    root_indent = (
        _jsonc_line_indent(source, root_properties[0].key_start, "  ") if root_properties else "  "
    )
    section = next((item for item in root_properties if item.key == section_name), None)
    if section is None:
        entry_indent = f"{root_indent}  "
        section_value = (
            f"{{\n{_jsonc_property_text('flameox', entry, entry_indent)}\n{root_indent}}}"
        )
        return _jsonc_insert_property(
            source,
            root_end,
            root_properties,
            f"{root_indent}{json.dumps(section_name)}: {section_value}",
            "",
        )

    section_start = _jsonc_skip_trivia(source, section.value_start)
    section_properties, section_end = _jsonc_object_properties(source, section_start)
    entry_indent = (
        _jsonc_line_indent(source, section_properties[0].key_start, f"{root_indent}  ")
        if section_properties
        else f"{root_indent}  "
    )
    existing = next((item for item in section_properties if item.key == "flameox"), None)
    if existing is not None:
        rendered_entry = json.dumps(entry, ensure_ascii=False, indent=2)
        rendered_entry = rendered_entry.replace(chr(10), chr(10) + entry_indent)
        return f"{source[: existing.value_start]}{rendered_entry}{source[existing.value_end :]}"
    return _jsonc_insert_property(
        source,
        section_end,
        section_properties,
        _jsonc_property_text("flameox", entry, entry_indent),
        root_indent,
    )


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
            document = json5.loads(source) if path.suffix == ".jsonc" else json.loads(source)
        except (OSError, UnicodeError, ValueError) as error:
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
        action = "update" if existing is not None else "create"
        section["flameox"] = {**existing, **entry} if isinstance(existing, dict) else entry
    if path.suffix == ".jsonc" and source is not None and action != "already_current":
        content = _jsonc_update_mcp_entry(source, section_name, section["flameox"])
    else:
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
        if isinstance(entry, MutableMapping) and all(
            entry.get(key) == value for key, value in expected.items()
        ):
            return source, source or "", "already_current"
        action: Literal["create", "update", "already_current"] = "update"
    else:
        action = "create"
    if not isinstance(entry, MutableMapping):
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
        active_provider_status(managed) if managed else "not_applicable",
    )
