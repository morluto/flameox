"""Request-owned dependency preparation and session-local collector activation."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import suppress
from pathlib import Path

import anyio

from flameox import __version__
from flameox.command_binding import ExecutableResolver
from flameox.executable_models import ResolvedExecutable
from flameox.execution import (
    INSTALLER_ENVIRONMENT_ALLOWLIST,
    ExecutionRequest,
    ProcessExecutionError,
    SubprocessBroker,
)
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
from flameox.runtime_contracts import RuntimeFailure
from flameox.runtime_errors import DomainError

# Keep the standalone collector aligned with the tested version in uv.lock.
PY_SPY_VERSION = "0.4.2"


class ProviderDependencies:
    """Own verified executable bindings, never an installed-provider inventory."""

    def __init__(self, broker: SubprocessBroker, scratch: Path) -> None:
        self.broker = broker
        self.scratch = scratch
        self._py_spy: ResolvedExecutable | None = None

    def py_spy_executable(self) -> str | None:
        if self._py_spy is None:
            return None
        try:
            current = self._bind(str(self._py_spy.invocation_path))
        except (DomainError, OSError) as error:
            raise self._expired_collector() from error
        if current.identity != self._py_spy.identity:
            raise self._expired_collector()
        return str(current.invocation_path)

    @staticmethod
    def _expired_collector() -> RuntimeFailure:
        return RuntimeFailure(
            "UNAVAILABLE_CAPABILITY",
            "The prepared collector is missing or changed; prepare py-spy again before capture.",
            details={"preparation_tool": "prepare_providers", "provider_ids": ["py-spy"]},
        )

    def verify_capture_binding(self, provider_id: str, binding: ResolvedExecutable) -> None:
        if (
            provider_id == "py-spy"
            and self._py_spy is not None
            and (
                binding.identity != self._py_spy.identity
                or binding.invocation_path != self._py_spy.invocation_path
            )
        ):
            raise self._expired_collector()

    def _bind(self, executable: str) -> ResolvedExecutable:
        return ExecutableResolver().require_host_tool(
            executable, cwd=self.scratch, environment=dict(os.environ)
        )

    async def _run(self, argv: list[str], timeout: int) -> bytes:
        try:
            request = ExecutionRequest(
                argv=tuple(argv),
                executable_binding=self._bind(argv[0]),
                cwd=self.scratch,
                allowed_working_roots=(self.scratch,),
                environment_allowlist=INSTALLER_ENVIRONMENT_ALLOWLIST,
                timeout_seconds=timeout,
                max_output_bytes=256 * 1024,
            )
            task = asyncio.create_task(self.broker.run(request))
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                task.cancel()
                # AnyIO cancellation is level-triggered. Let the broker settle its
                # readers and descendants under one cancellation before unwinding.
                with anyio.CancelScope(shield=True), suppress(asyncio.CancelledError):
                    await task
                raise
        except (DomainError, ProcessExecutionError, OSError) as error:
            raise SetupFailure(
                "Provider preparation could not complete; verify uvx availability, package "
                "index access, platform support, and the preparation timeout."
            ) from error
        if getattr(result.process.termination, "exit_code", None) != 0:
            raise SetupFailure(
                "Provider preparation failed; verify package availability and platform support."
            )
        return result.stdout

    async def _prepare_py_spy(self, timeout: int) -> tuple[ResolvedExecutable, list[str]]:
        try:
            if self.py_spy_executable() is not None:
                assert self._py_spy is not None
                return self._py_spy, []
        except RuntimeFailure:
            pass  # This explicit preparation request authorizes resolving a new binding.
        adjacent = Path(sys.executable).with_name("py-spy.exe" if os.name == "nt" else "py-spy")
        if adjacent.is_file():
            output = await self._run([str(adjacent), "--version"], min(timeout, 10))
            if output.decode().strip() == f"py-spy {PY_SPY_VERSION}":
                return self._bind(str(adjacent)), []
        probe = (
            "import importlib.metadata,json,os,sysconfig; "
            "from pathlib import Path; "
            "print(json.dumps({'version':importlib.metadata.version('py-spy'),"
            "'executable':str(Path(sysconfig.get_path('scripts')) / "
            "('py-spy.exe' if os.name == 'nt' else 'py-spy'))}))"
        )
        command = [
            "uvx",
            "--isolated",
            "--python",
            "3.12",
            "--from",
            f"py-spy=={PY_SPY_VERSION}",
            "python",
            "-c",
            probe,
        ]
        output = await self._run(command, timeout)
        try:
            receipt = json.loads(output)
            if receipt["version"] != PY_SPY_VERSION:
                raise ValueError("version mismatch")
            path = Path(receipt["executable"])
            if not path.is_absolute():
                raise ValueError("non-absolute collector")
            binding = self._bind(str(path))
            version = await self._run([str(path), "--version"], min(timeout, 10))
            if version.decode().strip() != f"py-spy {PY_SPY_VERSION}":
                raise ValueError("collector version mismatch")
        except (KeyError, TypeError, ValueError, DomainError, OSError) as error:
            raise SetupFailure("Prepared collector identity could not be verified.") from error
        return binding, command

    async def prepare(
        self,
        providers: list[str],
        timeout_seconds: int = DEFAULT_PREPARATION_TIMEOUT_SECONDS,
    ) -> ProviderPreparation:
        if not 1 <= timeout_seconds <= MAX_PREPARATION_TIMEOUT_SECONDS:
            raise SetupFailure("Invalid provider preparation timeout.")
        try:
            with anyio.fail_after(timeout_seconds):
                return await self._prepare(providers, timeout_seconds)
        except TimeoutError as error:
            raise SetupFailure("Provider preparation exceeded its overall timeout.") from error

    async def _prepare(self, providers: list[str], timeout_seconds: int) -> ProviderPreparation:
        requested = list(dict.fromkeys(providers))
        launcher_command, launcher_args = mcp_launcher(requested)
        managed = [item for item in requested if item in MANAGED_PROVIDER_EXTRAS]
        command: list[str] = []
        collector: ResolvedExecutable | None = None
        if "py-spy" in managed:
            collector, command = await self._prepare_py_spy(timeout_seconds)
        server_providers = [item for item in managed if item != "py-spy"]
        activation = active_provider_status(server_providers) if server_providers else "ready"
        if server_providers and activation != "ready":
            # The launcher still names the complete desired server environment. Preparing
            # it neither changes active imports nor discards unrelated collector bindings.
            command = [launcher_command, *launcher_args, "--version"]
            output = await self._run(command, timeout_seconds)
            if output.decode().strip() != __version__:
                raise SetupFailure("Prepared Flameox release does not match the running release.")
        preparation = ProviderPreparation(
            requested,
            managed,
            external_provider_requirements(requested),
            command,
            launcher_command,
            [*launcher_args, "mcp", "serve"],
            activation if managed else "not_applicable",
        )
        # Publish session state only after every requested preparation succeeds.
        # No await follows this commit, and failed requests never roll back another
        # concurrent request's successful binding.
        if collector is not None:
            self._py_spy = collector
        return preparation
