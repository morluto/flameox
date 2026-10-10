from __future__ import annotations

import json
import os
import tempfile
from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import json5
import tomlkit
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version
from tomlkit.exceptions import TOMLKitError
from tomlkit.items import InlineTable

from flameox import __version__
from flameox.atomic import atomic_write_text
from flameox.command_binding import ExecutableResolver
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.filesystem import BoundedFileSystem
from flameox.installation import verify_releases
from flameox.providers.availability import MANAGED_PROVIDER_EXTRAS
from flameox.providers.environment import (
    DEFAULT_PREPARATION_TIMEOUT_SECONDS,
    MAX_PREPARATION_TIMEOUT_SECONDS,
    ProviderPreparation,
    SetupFailure,
    active_provider_status,
    external_provider_requirements,
    mcp_launcher,
)
from flameox.runtime_errors import DomainError

PATH_CLI_PROBE_TIMEOUT_SECONDS = 5


class SetupClient(StrEnum):
    CLAUDE = "claude"
    CURSOR = "cursor"
    OPENCODE = "opencode"
    CODEX = "codex"
    GEMINI = "gemini"
    ANTIGRAVITY = "antigravity"

    @property
    def server_section(self) -> str:
        if self is SetupClient.CODEX:
            return "mcp_servers"
        return "mcp" if self is SetupClient.OPENCODE else "mcpServers"

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
        claude_home = Path(os.environ.get("CLAUDE_CONFIG_DIR") or home)
        codex_home = Path(os.environ.get("CODEX_HOME") or home / ".codex")
        gemini_home = Path(os.environ.get("GEMINI_CLI_HOME") or home)
        opencode_home = Path(
            os.environ.get("OPENCODE_CONFIG_DIR")
            or Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config") / "opencode"
        )
        return {
            SetupClient.CLAUDE: claude_home / ".claude.json",
            SetupClient.CURSOR: home / ".cursor" / "mcp.json",
            SetupClient.OPENCODE: opencode_home / "opencode.jsonc",
            SetupClient.CODEX: codex_home / "config.toml",
            SetupClient.GEMINI: gemini_home / ".gemini" / "settings.json",
            SetupClient.ANTIGRAVITY: home / ".gemini" / "config" / "mcp_config.json",
        }[self]

    def is_detected(self, home: Path) -> bool:
        try:
            if self is SetupClient.ANTIGRAVITY:
                return any(
                    path.exists()
                    for path in (
                        home / ".agent",
                        home / ".gemini" / "antigravity",
                        self.config_path(home),
                    )
                )
            path = self.active_config_path(home)
            marker = {
                SetupClient.CLAUDE: Path(os.environ.get("CLAUDE_CONFIG_DIR") or home / ".claude"),
                SetupClient.CURSOR: home / ".cursor",
                SetupClient.OPENCODE: path.parent,
                SetupClient.CODEX: path.parent,
                SetupClient.GEMINI: path.parent,
            }[self]
            return marker.exists() or path.exists()
        except OSError as error:
            raise SetupFailure(
                f"Could not inspect {self.display_name} configuration: {error.filename or home}"
            ) from error

    def active_config_path(self, home: Path) -> Path:
        try:
            path = self.config_path(home).absolute()
            if self is not SetupClient.OPENCODE:
                return path
            if not os.environ.get("OPENCODE_CONFIG_DIR") and os.environ.get("OPENCODE_CONFIG"):
                return Path(os.environ["OPENCODE_CONFIG"]).absolute()
            names: tuple[str, ...] = ("opencode.jsonc", "opencode.json")
            if not os.environ.get("OPENCODE_CONFIG_DIR"):
                names += ("config.json",)
            candidates = [path.with_name(name) for name in names]
            return next((item for item in candidates if item.exists() or item.is_symlink()), path)
        except OSError as error:
            raise SetupFailure(
                f"Could not inspect {self.display_name} configuration: {error.filename or home}"
            ) from error


SETUP_CLIENTS = tuple(SetupClient)


@dataclass(frozen=True, slots=True)
class ClientSetupPlan:
    client: SetupClient
    path: Path
    action: Literal["create", "update", "already_current"]
    detected: bool
    original: str | None
    content: str

    @property
    def environment(self) -> dict[str, str]:
        document = _parse_client_configuration(self.client, self.path, self.content)
        entry = document[self.client.server_section]["flameox"]
        return _client_environment(self.client, self.path, entry)


@dataclass(frozen=True, slots=True)
class ClientSetupResult:
    client: SetupClient
    path: Path
    action: Literal["created", "updated", "already_current"]


@dataclass(frozen=True, slots=True)
class ClientUpdatePlan:
    setup: ClientSetupPlan
    previous_version: str
    requirement: str
    environment: dict[str, str]


@dataclass(frozen=True, slots=True)
class ClientInstallation:
    client: SetupClient
    path: Path
    original: str
    document: MutableMapping[str, Any]
    command_line: list[str]
    requirement: Requirement
    environment: dict[str, str]

    @property
    def version(self) -> str:
        return str(Version(next(iter(self.requirement.specifier)).version))

    def plan_update(self, version: str) -> ClientUpdatePlan:
        document = deepcopy(self.document)
        entry = document[self.client.server_section]["flameox"]
        extras = f"[{','.join(sorted(self.requirement.extras))}]" if self.requirement.extras else ""
        requirement = f"flameox{extras}=={Version(version)}"
        updated = [*self.command_line]
        updated[6] = requirement
        action: Literal["update", "already_current"] = (
            "already_current" if updated == self.command_line else "update"
        )
        if action == "already_current":
            content = self.original
        elif self.client is SetupClient.CODEX:
            args = entry["args"]
            args[5] = requirement
            content = tomlkit.dumps(document)
        elif self.client is SetupClient.OPENCODE:
            content = _jsonc_update_launcher(self.original, requirement)
        else:
            entry["args"] = updated[1:]
            content = f"{json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)}\n"
        return ClientUpdatePlan(
            ClientSetupPlan(self.client, self.path, action, True, self.original, content),
            self.version,
            requirement,
            self.environment,
        )


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

    try:
        binding = ExecutableResolver().require_host_tool(
            "flameox", cwd=Path.cwd(), environment=dict(os.environ)
        )
        with tempfile.TemporaryDirectory(prefix="flameox-version-") as directory:
            scratch = Path(directory).resolve()
            result = SubprocessBroker().run_sync(
                ExecutionRequest(
                    argv=(str(binding.invocation_path), "--version"),
                    executable_binding=binding,
                    cwd=scratch,
                    allowed_working_roots=(scratch,),
                    timeout_seconds=PATH_CLI_PROBE_TIMEOUT_SECONDS,
                    max_output_bytes=4096,
                )
            )
    except (OSError, DomainError):
        return None
    if getattr(result.process.termination, "exit_code", None) != 0:
        return None
    version = result.stdout.decode(errors="replace").strip()
    if not version or "\n" in version or version == __version__:
        return None
    return CliVersionAdvisory(str(binding.invocation_path), version, __version__)


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
    if index == 0 and source.startswith("\ufeff"):
        index += 1
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
            if source.startswith(("//", "/*"), index):
                break
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


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _read_configuration(path: Path) -> str:
    with (
        BoundedFileSystem((path.parent,)).open_regular(path) as descriptor,
        os.fdopen(descriptor, "r", closefd=False) as stream,
    ):
        return stream.read()


def _read_client_configuration(
    client: SetupClient, path: Path
) -> tuple[str | None, MutableMapping[str, Any]]:
    try:
        if path.is_symlink():
            raise SetupFailure(f"Refusing to replace symbolic-link client configuration: {path}")
        if not path.exists():
            return None, tomlkit.document() if client is SetupClient.CODEX else {}
        source = _read_configuration(path)
    except (OSError, UnicodeError, DomainError) as error:
        raise SetupFailure(f"Could not read {client.display_name} configuration: {path}") from error
    return source, _parse_client_configuration(client, path, source)


def _parse_client_configuration(
    client: SetupClient, path: Path, source: str
) -> MutableMapping[str, Any]:
    try:
        document = (
            tomlkit.parse(source)
            if client is SetupClient.CODEX
            else json5.loads(source, allow_duplicate_keys=False)
            if client is SetupClient.OPENCODE
            else json.loads(source, object_pairs_hook=_unique_json_object)
        )
    except (ValueError, RecursionError, TOMLKitError) as error:
        raise SetupFailure(f"Could not read {client.display_name} configuration: {path}") from error
    if not isinstance(document, MutableMapping):
        raise SetupFailure(
            f"{client.display_name} configuration must contain a JSON object: {path}"
        )
    return document


def _client_environment(client: SetupClient, path: Path, entry: Any) -> dict[str, str]:
    environment_key = "environment" if client is SetupClient.OPENCODE else "env"
    environment = entry.get(environment_key, {})
    if not isinstance(environment, MutableMapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in environment.items()
    ):
        raise SetupFailure(f"Invalid Flameox environment in {path}; repair it before preparation.")
    return dict(environment)


def _json_plan(
    client: SetupClient,
    path: Path,
    command: str,
    args: list[str],
) -> tuple[str | None, str, Literal["create", "update", "already_current"]]:
    source, document = _read_client_configuration(client, path)
    section_name = client.server_section
    section = document.setdefault(section_name, {})
    if not isinstance(section, dict):
        raise SetupFailure(
            f"{client.display_name} configuration {section_name!r} must be an object: {path}"
        )
    entry = _json_entry(client, command, args)
    existing = section.get("flameox")
    if isinstance(existing, dict) and all(
        existing.get(key) == value for key, value in entry.items()
    ):
        return source, source or "", "already_current"
    action: Literal["create", "update"] = "update" if existing is not None else "create"
    section["flameox"] = {**existing, **entry} if isinstance(existing, dict) else entry
    if client is SetupClient.OPENCODE and source is not None:
        content = _jsonc_update_mcp_entry(source, section_name, section["flameox"])
    else:
        content = f"{json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)}\n"
    return source, content, action


def _codex_plan(
    path: Path,
    command: str,
    args: list[str],
) -> tuple[str | None, str, Literal["create", "update", "already_current"]]:
    source, document = _read_client_configuration(SetupClient.CODEX, path)
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
        path = client.active_config_path(root)
        if client is SetupClient.CODEX:
            original, content, action = _codex_plan(path, command, args)
        else:
            original, content, action = _json_plan(client, path, command, args)
        plans.append(
            ClientSetupPlan(client, path, action, client.is_detected(root), original, content)
        )
    return plans


def read_client_installations(
    clients: list[SetupClient], *, home: Path | None = None
) -> list[ClientInstallation]:
    """Read recognized launchers once; publication later checks the same snapshots."""
    root = home or Path.home()
    installations: list[ClientInstallation] = []
    for client in clients:
        path = client.active_config_path(root)
        original, document = _read_client_configuration(client, path)
        section_name = client.server_section
        section = document.get(section_name, {})
        if not isinstance(section, MutableMapping):
            raise SetupFailure(
                f"{client.display_name} configuration {section_name!r} must be a table."
            )
        entry = section.get("flameox")
        if entry is None:
            continue
        if not isinstance(entry, MutableMapping):
            raise SetupFailure(f"Unrecognized Flameox launcher in {path}; rerun setup.")
        if client is SetupClient.OPENCODE:
            command_line = entry.get("command")
        else:
            args = entry.get("args")
            command_line = [entry.get("command"), *args] if isinstance(args, list) else None
        requirement = _installed_launcher_requirement(command_line, path)
        environment = _client_environment(client, path, entry)
        assert isinstance(command_line, list) and original is not None
        installations.append(
            ClientInstallation(
                client,
                path,
                original,
                document,
                command_line,
                requirement,
                environment,
            )
        )
    return installations


def _installed_launcher_requirement(command_line: object, path: Path) -> Requirement:
    message = f"Unrecognized Flameox launcher in {path}; update it manually or rerun setup."
    if not isinstance(command_line, list) or not all(isinstance(arg, str) for arg in command_line):
        raise SetupFailure(message)
    if (
        len(command_line) < 10
        or command_line[0] != "uvx"
        or command_line[1:6] != ["--no-config", "--no-sources", "--python", "3.12", "--from"]
        or command_line[7:10] != ["flameox", "mcp", "serve"]
    ):
        raise SetupFailure(message)
    try:
        requirement = Requirement(command_line[6])
        pins = list(requirement.specifier)
        if (
            requirement.name != "flameox"
            or requirement.url
            or requirement.marker
            or len(pins) != 1
            or pins[0].operator != "=="
        ):
            raise SetupFailure(message)
        Version(pins[0].version)
    except (InvalidRequirement, InvalidVersion) as error:
        raise SetupFailure(message) from error
    return requirement


def _jsonc_update_launcher(source: str, requirement: str) -> str:
    # Replace the one string token, retaining comments inside the launcher and its array.
    start = _jsonc_skip_trivia(source, 0)
    for name in ("mcp", "flameox", "command"):
        properties, _ = _jsonc_object_properties(source, start)
        property_ = next(item for item in properties if item.key == name)
        start = _jsonc_skip_trivia(source, property_.value_start)
    index = _jsonc_skip_trivia(source, start + 1)
    for position in range(7):
        end = _jsonc_value_end(source, index)
        if position == 6:
            result = source[:index] + json.dumps(requirement) + source[end:]
            return result
        index = _jsonc_skip_trivia(source, end)
        index = _jsonc_skip_trivia(source, index + 1)
    raise SetupFailure("Could not locate the Flameox release pin in the JSONC launcher.")


def _verify_client_original(plan: ClientSetupPlan) -> None:
    try:
        if plan.path.is_symlink():
            raise SetupFailure(
                f"Refusing to replace symbolic-link client configuration: {plan.path}"
            )
        current = _read_configuration(plan.path) if plan.path.exists() else None
    except (OSError, UnicodeError, DomainError) as error:
        raise SetupFailure(
            f"Could not read {plan.client.display_name} configuration: {plan.path}"
        ) from error
    if current != plan.original:
        raise SetupFailure(
            f"{plan.client.display_name} configuration changed during setup: {plan.path}"
        )


def apply_client_setup(plans: list[ClientSetupPlan]) -> list[ClientSetupResult]:
    # Detect edits made during preparation before publishing the first client.
    for plan in plans:
        _verify_client_original(plan)
    results: list[ClientSetupResult] = []
    for plan in plans:
        try:
            _verify_client_original(plan)
            if plan.action != "already_current":
                atomic_write_text(plan.path, plan.content)
        except SetupFailure as error:
            if any(result.action != "already_current" for result in results):
                raise SetupFailure(
                    f"{error}. Earlier selected clients may already have been updated; "
                    "restart or reconnect changed clients and retry the remaining update."
                ) from error
            raise
        except (OSError, UnicodeError, DomainError) as error:
            raise SetupFailure(
                f"Could not update {plan.client.display_name} configuration: {plan.path}. "
                "Selected clients may already have been configured; restart or reconnect "
                "changed clients and retry the remaining update."
            ) from error
        if plan.action == "already_current":
            action: Literal["created", "updated", "already_current"] = "already_current"
        else:
            action = "created" if plan.action == "create" else "updated"
        results.append(ClientSetupResult(plan.client, plan.path, action))
    return results


def prepare_providers(
    plans: list[ClientSetupPlan],
    providers: list[str],
    timeout_seconds: int = DEFAULT_PREPARATION_TIMEOUT_SECONDS,
) -> ProviderPreparation:
    if not 1 <= timeout_seconds <= MAX_PREPARATION_TIMEOUT_SECONDS:
        raise SetupFailure(
            f"timeout_seconds must be between 1 and {MAX_PREPARATION_TIMEOUT_SECONDS}"
        )
    requested = list(dict.fromkeys(providers))
    external = external_provider_requirements(requested)
    managed = [item for item in requested if item in MANAGED_PROVIDER_EXTRAS]
    launcher_command, launcher_args = mcp_launcher(managed)
    server_args = [*launcher_args, "mcp", "serve"]
    preparation_command = [launcher_command, *launcher_args, "mcp", "inspect"]
    requirement = launcher_args[launcher_args.index("--from") + 1]
    environments = [(requirement, plan.environment) for plan in plans]
    verify_releases(environments, timeout_seconds=timeout_seconds)

    return ProviderPreparation(
        requested,
        managed,
        external,
        preparation_command,
        launcher_command,
        server_args,
        active_provider_status(managed) if managed else "not_applicable",
    )
