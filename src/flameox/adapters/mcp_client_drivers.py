from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Protocol

from packaging.version import InvalidVersion, Version

from flameox.adapters.client_setup import ClientPlanAction, Launcher, SetupClient
from flameox.command_binding import ExecutableResolver
from flameox.domain import DomainError, ErrorCode
from flameox.domain.executables import ResolvedExecutable
from flameox.execution import ExecutionOutcome, ExecutionRequest, SubprocessBroker


class ClientManagementMechanism(StrEnum):
    OFFICIAL_CLI = "official_cli"
    QUALIFIED_CONFIG_FILE = "qualified_config_file"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class ClientCommandPlan:
    client: SetupClient
    mechanism: ClientManagementMechanism
    action: ClientPlanAction
    executable: ResolvedExecutable
    client_version: str
    inspect_argv: tuple[str, ...]
    mutation_argv: tuple[tuple[str, ...], ...]
    launcher: Launcher | None
    previous_launcher: Launcher | None
    scope: str


class ClientCommandDriver(Protocol):
    client: SetupClient

    def probe(self) -> tuple[ResolvedExecutable, str] | None: ...

    def plan(self, launcher: Launcher, *, remove: bool) -> ClientCommandPlan: ...

    async def apply(self, plan: ClientCommandPlan) -> None: ...


class OfficialCliDriver:
    """One qualified, bounded command profile owned by an MCP client."""

    _EXECUTABLES: ClassVar[dict[SetupClient, str]] = {
        SetupClient.CLAUDE: "claude",
        SetupClient.CODEX: "codex",
        SetupClient.GEMINI: "gemini",
    }

    def __init__(
        self,
        client: SetupClient,
        *,
        broker: SubprocessBroker,
        resolver: ExecutableResolver | None = None,
        cwd: Path | None = None,
    ) -> None:
        if client not in self._EXECUTABLES:
            raise ValueError(f"{client.value} has no qualified non-interactive CLI profile")
        self.client = client
        self.broker = broker
        self.resolver = resolver or ExecutableResolver()
        self.cwd = (cwd or Path.cwd()).resolve()

    def probe(self) -> tuple[ResolvedExecutable, str] | None:
        executable = self.resolver.resolve_host_tool(
            self._EXECUTABLES[self.client], cwd=self.cwd
        )
        if executable is None:
            return None
        outcome = self.broker.run_sync(
            self._request(executable, (executable.requested_token, "--version"))
        )
        if outcome.process.exit_code != 0:
            return None
        version = _extract_version(outcome.stdout.decode("utf-8", errors="replace"))
        return (executable, version) if version is not None else None

    def plan(self, launcher: Launcher, *, remove: bool) -> ClientCommandPlan:
        capability = self.probe()
        if capability is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"A qualified {self.client.display_name} MCP management CLI was not found.",
            )
        executable, version = capability
        inspect_argv = self._inspect_argv(executable)
        inspected = self.broker.run_sync(self._request(executable, inspect_argv))
        current = self._current_launcher(inspected)
        if remove:
            action = (
                ClientPlanAction.REMOVE
                if current is not None
                else ClientPlanAction.NOT_CONFIGURED
            )
            mutations = () if current is None else (self._remove_argv(executable),)
            expected: Launcher | None = None
        elif current == launcher:
            action = ClientPlanAction.ALREADY_CURRENT
            mutations = ()
            expected = launcher
        else:
            action = ClientPlanAction.CREATE if current is None else ClientPlanAction.UPDATE
            remove_argv = () if current is None else (self._remove_argv(executable),)
            mutations = (*remove_argv, self._add_argv(executable, launcher))
            expected = launcher
        return ClientCommandPlan(
            client=self.client,
            mechanism=ClientManagementMechanism.OFFICIAL_CLI,
            action=action,
            executable=executable,
            client_version=version,
            inspect_argv=inspect_argv,
            mutation_argv=mutations,
            launcher=expected,
            previous_launcher=current,
            scope="user",
        )

    async def apply(self, plan: ClientCommandPlan) -> None:
        before = await self.broker.run(self._request(plan.executable, plan.inspect_argv))
        if self._current_launcher(before) != plan.previous_launcher:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                f"{self.client.display_name} MCP state changed after setup planning.",
                retryable=True,
            )
        for argv in plan.mutation_argv:
            outcome = await self.broker.run(self._request(plan.executable, argv))
            self._require_success(outcome, "mutation")
        inspected = await self.broker.run(
            self._request(plan.executable, plan.inspect_argv)
        )
        current = self._current_launcher(inspected)
        if current != plan.launcher:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                f"{self.client.display_name} did not round-trip the requested MCP registration.",
                details={"client": self.client.value, "version": plan.client_version},
                remediation=("Inspect the client MCP configuration and retry setup.",),
            )

    def _inspect_argv(self, executable: ResolvedExecutable) -> tuple[str, ...]:
        token = executable.requested_token
        if self.client is SetupClient.CODEX:
            return (token, "mcp", "get", "flameox", "--json")
        return (token, "mcp", "get", "flameox") if self.client is SetupClient.CLAUDE else (
            token,
            "mcp",
            "list",
        )

    def _add_argv(
        self, executable: ResolvedExecutable, launcher: Launcher
    ) -> tuple[str, ...]:
        token = executable.requested_token
        if self.client is SetupClient.CLAUDE:
            payload = json.dumps(
                {"type": "stdio", "command": launcher.command, "args": list(launcher.args)},
                separators=(",", ":"),
            )
            return (token, "mcp", "add-json", "--scope", "user", "flameox", payload)
        if self.client is SetupClient.CODEX:
            return (token, "mcp", "add", "flameox", "--", launcher.command, *launcher.args)
        return (
            token,
            "mcp",
            "add",
            "--scope",
            "user",
            "flameox",
            launcher.command,
            *launcher.args,
        )

    def _remove_argv(self, executable: ResolvedExecutable) -> tuple[str, ...]:
        token = executable.requested_token
        if self.client is SetupClient.CODEX:
            return (token, "mcp", "remove", "flameox")
        return (token, "mcp", "remove", "--scope", "user", "flameox")

    def _current_launcher(self, outcome: ExecutionOutcome) -> Launcher | None:
        if outcome.process.exit_code != 0:
            return None
        text = outcome.stdout.decode("utf-8", errors="replace")
        if self.client is SetupClient.CODEX:
            try:
                value = json.loads(text)
                transport = value["transport"]
                if transport["type"] != "stdio":
                    return None
                return Launcher(
                    command=str(transport["command"]),
                    args=tuple(str(item) for item in transport.get("args") or ()),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "Codex returned an invalid structured MCP inspection result.",
                ) from exc
        # Claude and Gemini currently expose human inspection output. Their
        # qualified profile verifies the exact command and each argument token.
        if "flameox" not in text:
            return None
        command = _field(text, "command")
        args = _field(text, "args")
        if command is None:
            return None
        parsed_args = tuple(json.loads(args)) if args and args.startswith("[") else ()
        return Launcher(command=command, args=parsed_args)

    def _request(
        self, executable: ResolvedExecutable, argv: tuple[str, ...]
    ) -> ExecutionRequest:
        return ExecutionRequest(
            argv=argv,
            executable_binding=executable,
            cwd=self.cwd,
            environment_allowlist=("PATH", "HOME", "CODEX_HOME"),
            environment_overrides={"HOME": str(Path.home())},
            allowed_working_roots=(self.cwd,),
            timeout_seconds=15,
            max_output_bytes=1024 * 1024,
        )

    def _require_success(self, outcome: ExecutionOutcome, operation: str) -> None:
        if outcome.process.exit_code == 0:
            return
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            f"{self.client.display_name} MCP {operation} failed.",
            details={
                "client": self.client.value,
                "exit_code": outcome.process.exit_code,
                "diagnostic": outcome.stderr.decode("utf-8", errors="replace")[:4096],
            },
        )


def _extract_version(text: str) -> str | None:
    for token in text.replace(",", " ").split():
        candidate = token.removeprefix("v")
        try:
            return str(Version(candidate))
        except InvalidVersion:
            continue
    return None


def _field(text: str, name: str) -> str | None:
    prefix = f"{name}:"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            return stripped[len(prefix) :].strip()
    return None
