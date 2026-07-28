from __future__ import annotations

import json
import stat
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import tomlkit

from flameox.domain import DomainError, ErrorCode


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
            SetupClient.ANTIGRAVITY: "Antigravity",
        }[self]


ALL_SETUP_CLIENTS = tuple(SetupClient)


class ClientPlanAction(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    ALREADY_CURRENT = "already_current"
    REMOVE = "remove"
    NOT_CONFIGURED = "not_configured"


@dataclass(frozen=True, slots=True)
class Launcher:
    command: str
    args: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClientConfigEdit:
    client: SetupClient
    path: Path
    action: ClientPlanAction
    detected: bool
    original: bytes | None
    updated: bytes | None
    mode: int


@dataclass(frozen=True, slots=True)
class _ClientDefinition:
    path: Path
    format: str
    config_key: str
    detected: bool


class ClientConfigRegistry:
    """Resolve narrow, source-preserving edits to supported MCP client configs."""

    def __init__(
        self,
        *,
        home: Path,
        jsonc_helper: Path | None = None,
        node_executable: str = "node",
    ) -> None:
        self.home = home
        self.jsonc_helper = jsonc_helper
        self.node_executable = node_executable

    def definition(self, client: SetupClient) -> _ClientDefinition:
        if client is SetupClient.CLAUDE:
            path = self.home / ".claude.json"
            return _ClientDefinition(
                path,
                "json",
                "mcpServers",
                (self.home / ".claude").exists() or path.exists(),
            )
        if client is SetupClient.CURSOR:
            path = self.home / ".cursor" / "mcp.json"
            return _ClientDefinition(path, "json", "mcpServers", path.parent.exists())
        if client is SetupClient.OPENCODE:
            root = self.home / ".config" / "opencode"
            candidates = (
                root / "opencode.json",
                root / "opencode.jsonc",
                root / ".opencode.json",
                root / ".opencode.jsonc",
            )
            path = next(
                (candidate for candidate in candidates if candidate.exists()),
                candidates[0],
            )
            format_name = "jsonc" if path.suffix == ".jsonc" else "json"
            return _ClientDefinition(path, format_name, "mcp", root.exists())
        if client is SetupClient.CODEX:
            path = self.home / ".codex" / "config.toml"
            return _ClientDefinition(path, "toml", "mcp_servers", path.parent.exists())
        if client is SetupClient.GEMINI:
            path = self.home / ".gemini" / "settings.json"
            return _ClientDefinition(path, "json", "mcpServers", path.parent.exists())
        path = self.home / ".gemini" / "config" / "mcp_config.json"
        detected = (self.home / ".gemini" / "antigravity").exists() or (
            self.home / ".agent"
        ).exists()
        return _ClientDefinition(path, "json", "mcpServers", detected)

    def detected_clients(self) -> tuple[SetupClient, ...]:
        return tuple(client for client in ALL_SETUP_CLIENTS if self.definition(client).detected)

    def configured_clients(self, *, strict: bool = False) -> tuple[SetupClient, ...]:
        configured: list[SetupClient] = []
        for client in ALL_SETUP_CLIENTS:
            definition = self.definition(client)
            if not definition.path.exists():
                continue
            try:
                value = self._read_entry(definition)
            except DomainError:
                if strict:
                    raise
                continue
            if value is not None:
                configured.append(client)
        return tuple(configured)

    def allowed_config_paths(self) -> frozenset[Path]:
        opencode_root = self.home / ".config" / "opencode"
        return frozenset(
            {
                self.home / ".claude.json",
                self.home / ".cursor" / "mcp.json",
                opencode_root / "opencode.json",
                opencode_root / "opencode.jsonc",
                opencode_root / ".opencode.json",
                opencode_root / ".opencode.jsonc",
                self.home / ".codex" / "config.toml",
                self.home / ".gemini" / "settings.json",
                self.home / ".gemini" / "config" / "mcp_config.json",
            }
        )

    def plan(
        self,
        client: SetupClient,
        launcher: Launcher,
        *,
        remove: bool,
    ) -> ClientConfigEdit:
        definition = self.definition(client)
        try:
            metadata = definition.path.stat()
        except FileNotFoundError:
            original = None
            mode = 0o600
        except OSError as exc:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Could not inspect MCP client configuration {definition.path}.",
                details={"error": str(exc)},
            ) from exc
        else:
            original = self._read_bytes(definition.path)
            mode = stat.S_IMODE(metadata.st_mode)
        text = self._decode(definition.path, original) if original is not None else ""
        current = self._read_entry(definition, text=text)

        if remove:
            if current is None:
                return ClientConfigEdit(
                    client,
                    definition.path,
                    ClientPlanAction.NOT_CONFIGURED,
                    definition.detected,
                    original,
                    original,
                    mode,
                )
            updated = self._updated_content(definition, original, value=None, remove=True)
            return ClientConfigEdit(
                client,
                definition.path,
                ClientPlanAction.REMOVE,
                definition.detected,
                original,
                updated,
                mode,
            )

        wanted = self._entry(client, launcher)
        if current == wanted:
            return ClientConfigEdit(
                client,
                definition.path,
                ClientPlanAction.ALREADY_CURRENT,
                definition.detected,
                original,
                original,
                mode,
            )
        updated = self._updated_content(definition, original, value=wanted, remove=False)
        action = ClientPlanAction.CREATE if original is None else ClientPlanAction.UPDATE
        return ClientConfigEdit(
            client,
            definition.path,
            action,
            definition.detected,
            original,
            updated,
            mode,
        )

    def _entry(self, client: SetupClient, launcher: Launcher) -> dict[str, Any]:
        if client is SetupClient.OPENCODE:
            return {
                "type": "local",
                "command": [launcher.command, *launcher.args],
                "cwd": ".",
                "enabled": True,
            }
        return {"command": launcher.command, "args": list(launcher.args)}

    def _read_entry(
        self,
        definition: _ClientDefinition,
        *,
        text: str | None = None,
    ) -> dict[str, Any] | None:
        if not definition.path.exists():
            return None
        text = self._read_text(definition.path) if text is None else text
        if definition.format == "toml":
            try:
                toml_document = tomlkit.parse(text)
            except (ValueError, TypeError) as exc:
                raise self._invalid_config(definition.path, exc) from exc
            section = toml_document.get(definition.config_key)
            if section is None:
                return None
            try:
                value = section.get("flameox")
            except AttributeError as exc:
                raise self._invalid_config(
                    definition.path, f"{definition.config_key} is not a table"
                ) from exc
            plain = _plain_mapping(value)
            if value is not None and plain is None:
                raise self._invalid_config(definition.path, "the flameox entry is not a table")
            return plain

        if definition.format == "jsonc" and self.jsonc_helper is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Editing {definition.path} requires the flameox npm JSONC helper.",
                remediation=("Run setup through `npx flameox setup`.",),
            )
        json_document = self._parse_json(definition, text)
        section = json_document.get(definition.config_key)
        if section is None:
            return None
        if not isinstance(section, dict):
            raise self._invalid_config(definition.path, f"{definition.config_key} is not an object")
        value = section.get("flameox")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise self._invalid_config(definition.path, "the flameox entry is not an object")
        return value

    def _updated_content(
        self,
        definition: _ClientDefinition,
        original: bytes | None,
        *,
        value: dict[str, Any] | None,
        remove: bool,
    ) -> bytes:
        text = self._decode(definition.path, original) if original is not None else ""
        if definition.format == "toml":
            return self._edit_toml(definition, text, value=value, remove=remove).encode()
        if definition.format == "jsonc" and self.jsonc_helper is not None:
            return self._edit_jsonc(
                definition,
                text,
                value=value,
                remove=remove,
            ).encode()
        if definition.format == "jsonc":
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Editing {definition.path} requires the flameox npm JSONC helper.",
                remediation=("Run setup through `npx flameox setup`.",),
            )
        document = self._parse_json(definition, text) if text else {}
        section = document.setdefault(definition.config_key, {})
        if not isinstance(section, dict):
            raise self._invalid_config(definition.path, f"{definition.config_key} is not an object")
        if remove:
            section.pop("flameox", None)
        else:
            section["flameox"] = value
        return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode()

    def _edit_toml(
        self,
        definition: _ClientDefinition,
        text: str,
        *,
        value: dict[str, Any] | None,
        remove: bool,
    ) -> str:
        try:
            document = tomlkit.parse(text) if text else tomlkit.document()
        except (ValueError, TypeError) as exc:
            raise self._invalid_config(definition.path, exc) from exc
        section = document.get(definition.config_key)
        if section is None:
            section = tomlkit.table()
            document[definition.config_key] = section
        if not hasattr(section, "get") or not hasattr(section, "__setitem__"):
            raise self._invalid_config(definition.path, f"{definition.config_key} is not a table")
        if remove:
            section.pop("flameox", None)
        else:
            table = tomlkit.table()
            assert value is not None
            table.add("command", value["command"])
            table.add("args", value["args"])
            section["flameox"] = table
        return tomlkit.dumps(document)

    def _parse_json(self, definition: _ClientDefinition, text: str) -> dict[str, Any]:
        if not text.strip():
            return {}
        if definition.format == "jsonc":
            if self.jsonc_helper is None:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    f"Editing {definition.path} requires the flameox npm JSONC helper.",
                    remediation=("Run setup through `npx flameox setup`.",),
                )
            result = self._run_jsonc_helper(
                {"operation": "parse", "text": text},
                definition.path,
            )
            value = result.get("value")
        else:
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise self._invalid_config(definition.path, exc) from exc
        if not isinstance(value, dict):
            raise self._invalid_config(
                definition.path,
                "the document root is not an object",
            )
        return value

    def _edit_jsonc(
        self,
        definition: _ClientDefinition,
        text: str,
        *,
        value: dict[str, Any] | None,
        remove: bool,
    ) -> str:
        result = self._run_jsonc_helper(
            {
                "operation": "modify",
                "text": text or "{}\n",
                "path": [definition.config_key, "flameox"],
                "remove": remove,
                "value": value,
            },
            definition.path,
        )
        updated = result.get("text")
        if not isinstance(updated, str):
            raise self._invalid_config(definition.path, "JSONC helper returned no text")
        return updated

    def _run_jsonc_helper(self, request: dict[str, Any], path: Path) -> dict[str, Any]:
        assert self.jsonc_helper is not None
        try:
            completed = subprocess.run(
                [self.node_executable, str(self.jsonc_helper)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "The flameox JSONC editor could not be started.",
                details={"path": str(path), "error": str(exc)},
                remediation=("Run setup through `npx flameox setup` with Node.js available.",),
            ) from exc
        if completed.returncode != 0:
            message = completed.stderr.strip() or "unknown JSONC parser failure"
            raise self._invalid_config(path, message)
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise self._invalid_config(path, "JSONC helper returned invalid output") from exc
        if not isinstance(result, dict):
            raise self._invalid_config(path, "JSONC helper returned a non-object")
        return result

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Could not read MCP client configuration {path}.",
                details={"error": str(exc)},
            ) from exc

    @staticmethod
    def _read_bytes(path: Path) -> bytes:
        try:
            return path.read_bytes()
        except OSError as exc:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Could not read MCP client configuration {path}.",
                details={"error": str(exc)},
            ) from exc

    @staticmethod
    def _decode(path: Path, value: bytes) -> str:
        try:
            return value.decode("utf-8")
        except UnicodeError as exc:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"MCP client configuration is not UTF-8: {path}.",
            ) from exc

    @staticmethod
    def _invalid_config(path: Path, error: object) -> DomainError:
        return DomainError(
            ErrorCode.EXECUTION_REFUSED,
            f"Refusing to overwrite malformed client configuration {path}.",
            details={"error": str(error)},
            remediation=("Repair the configuration and run setup again.",),
        )


def _plain_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "unwrap"):
        value = value.unwrap()
    if not isinstance(value, dict):
        return None
    return {str(key): _plain_value(item) for key, item in value.items()}


def _plain_value(value: Any) -> Any:
    if hasattr(value, "unwrap"):
        return _plain_value(value.unwrap())
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value
