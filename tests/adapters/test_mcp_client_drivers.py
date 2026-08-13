from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters.client_setup import ClientPlanAction, Launcher, SetupClient
from flameox.adapters.mcp_client_drivers import (
    ClientManagementMechanism,
    OfficialCliDriver,
)
from flameox.command_binding import ExecutableResolver
from flameox.domain import ProcessResult, process_termination_from_returncode
from flameox.domain.executables import ResolvedExecutable
from flameox.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    ProcessContainment,
    SubprocessBroker,
)

pytestmark = pytest.mark.unit


class _Resolver(ExecutableResolver):
    def __init__(self, executable: ResolvedExecutable) -> None:
        self.executable = executable

    def resolve_host_tool(
        self,
        token: str,
        *,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> ResolvedExecutable | None:
        del token, cwd, environment
        return self.executable


class _CodexBroker(SubprocessBroker):
    def __init__(self, executable: ResolvedExecutable) -> None:
        self.executable = executable
        self.launcher: Launcher | None = None
        self.requests: list[tuple[str, ...]] = []

    def run_sync(self, request: ExecutionRequest, **_: object) -> ExecutionOutcome:
        return self._run(request)

    async def run(self, request: ExecutionRequest, **_: object) -> ExecutionOutcome:
        return self._run(request)

    def _run(self, request: ExecutionRequest) -> ExecutionOutcome:
        argv = request.argv
        self.requests.append(argv)
        if argv[1:] == ("--version",):
            return self._outcome(b"codex-cli 0.75.0\n")
        if argv[1:4] == ("mcp", "get", "flameox"):
            if self.launcher is None:
                return self._outcome(b"", exit_code=1)
            return self._outcome(
                json.dumps(
                    {
                        "name": "flameox",
                        "transport": {
                            "type": "stdio",
                            "command": self.launcher.command,
                            "args": list(self.launcher.args),
                        },
                    }
                ).encode()
            )
        if argv[1:4] == ("mcp", "add", "flameox"):
            separator = argv.index("--")
            self.launcher = Launcher(argv[separator + 1], argv[separator + 2 :])
            return self._outcome(b"")
        if argv[1:4] == ("mcp", "remove", "flameox"):
            self.launcher = None
            return self._outcome(b"")
        raise AssertionError(argv)

    def _outcome(self, stdout: bytes, *, exit_code: int = 0) -> ExecutionOutcome:
        return ExecutionOutcome(
            process=ProcessResult(
                termination=process_termination_from_returncode(exit_code),
                cleanup_complete=True,
            ),
            stdout=stdout,
            stderr=b"",
            resolved_executable=self.executable.canonical_target,
            executable_binding=self.executable,
            containment=ProcessContainment.PROCESS_GROUP,
        )


def _driver(tmp_path: Path) -> tuple[OfficialCliDriver, _CodexBroker]:
    executable = ExecutableResolver().require_host_tool("python", cwd=tmp_path)
    broker = _CodexBroker(executable)
    return (
        OfficialCliDriver(
            SetupClient.CODEX,
            broker=broker,
            resolver=_Resolver(executable),
            cwd=tmp_path,
        ),
        broker,
    )


@pytest.mark.anyio
async def test_codex_driver_uses_structured_official_cli_and_verifies_round_trip(
    tmp_path: Path,
) -> None:
    driver, broker = _driver(tmp_path)
    launcher = Launcher("/opt/flameox", ("mcp", "serve", "--project-root", "."))

    plan = driver.plan(launcher, remove=False)
    await driver.apply(plan)

    assert plan.mechanism is ClientManagementMechanism.OFFICIAL_CLI
    assert plan.action is ClientPlanAction.CREATE
    assert plan.client_version == "0.75.0"
    assert broker.launcher == launcher
    assert any(argv[1:4] == ("mcp", "add", "flameox") for argv in broker.requests)


@pytest.mark.anyio
async def test_codex_driver_rejects_state_changed_after_preview(tmp_path: Path) -> None:
    driver, broker = _driver(tmp_path)
    wanted = Launcher("/opt/flameox", ("mcp", "serve"))
    plan = driver.plan(wanted, remove=False)
    broker.launcher = Launcher("/other", ())

    with pytest.raises(Exception, match="changed after setup planning"):
        await driver.apply(plan)

    assert broker.launcher == Launcher("/other", ())
