from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from packaging.version import InvalidVersion, Version

from flameox.atomic import atomic_write_json, atomic_write_text
from flameox.domain import DomainError, ErrorCode
from flameox.execution import (
    INSTALLER_ENVIRONMENT_ALLOWLIST,
    ExecutionOutcome,
    ExecutionRequest,
    SubprocessBroker,
)
from flameox.storage import Workspace


@dataclass(frozen=True, slots=True)
class RuntimeInstallation:
    version: str
    executable: Path
    installed: bool


@dataclass(frozen=True, slots=True)
class TraceProcessorInstallation:
    version: str
    executable: Path
    installed: bool


class ManagedRuntime:
    """Install and verify version-addressed flameox tool environments."""

    distribution = "flameox"

    def __init__(
        self,
        root: Path,
        *,
        uv_executable: str = "uv",
        broker: SubprocessBroker | None = None,
    ) -> None:
        self.root = root
        self.uv_executable = uv_executable
        self.broker = broker or SubprocessBroker()

    def executable(self, version: str) -> Path:
        self._validated_version(version)
        suffix = ".exe" if os.name == "nt" else ""
        return self.root / "runtimes" / version / "bin" / f"flameox{suffix}"

    def installed_versions(self) -> tuple[str, ...]:
        runtimes = self.root / "runtimes"
        if not runtimes.exists():
            return ()
        versions: list[str] = []
        for path in runtimes.iterdir():
            try:
                parsed = Version(path.name)
            except InvalidVersion:
                continue
            if str(parsed) != path.name:
                continue
            executable = self.executable(path.name)
            if (
                path.is_dir()
                and executable.is_file()
                and self._manifest_matches(path / "runtime.json", path.name, executable)
            ):
                versions.append(path.name)
        return tuple(sorted(versions, key=Version, reverse=True))

    async def install(self, version: str) -> RuntimeInstallation:
        version = self._validated_version(version)
        executable = self.executable(version)
        manifest = executable.parent.parent / "runtime.json"
        if executable.is_file() and manifest.is_file():
            if not self._manifest_matches(manifest, version, executable):
                raise self._verification_error(executable, "runtime metadata is invalid")
            await self.verify(executable, version)
            return RuntimeInstallation(version, executable, False)

        runtime_root = executable.parent.parent
        runtime_root.mkdir(parents=True, exist_ok=True)
        try:
            extras = self._managed_extras()
            distribution = (
                f"{self.distribution}[{','.join(extras)}]" if extras else self.distribution
            )
            outcome = await self.broker.run(
                ExecutionRequest(
                    argv=(
                        self.uv_executable,
                        "tool",
                        "install",
                        "--force",
                        "--no-config",
                        "--no-sources",
                        "--prerelease",
                        "allow",
                        "--python",
                        "3.12",
                        f"{distribution}=={version}",
                    ),
                    cwd=Path.cwd(),
                    environment_allowlist=INSTALLER_ENVIRONMENT_ALLOWLIST,
                    environment_overrides={
                        "UV_TOOL_DIR": str(runtime_root / "tools"),
                        "UV_TOOL_BIN_DIR": str(runtime_root / "bin"),
                    },
                    allowed_working_roots=(Path.cwd(),),
                    timeout_seconds=600,
                    max_output_bytes=16 * 1024 * 1024,
                )
            )
        except (DomainError, OSError) as exc:
            raise self._installation_error(version, str(exc)) from exc
        if outcome.process.exit_code != 0 or not executable.is_file():
            detail = _output_detail(outcome)
            raise self._installation_error(version, detail or "uv produced no launcher")

        try:
            await self.verify(executable, version)
        except BaseException:
            shutil.rmtree(runtime_root, ignore_errors=True)
            raise
        atomic_write_json(
            manifest,
            {
                "schema_version": 1,
                "distribution": self.distribution,
                "version": version,
                "executable": str(executable),
            },
        )
        return RuntimeInstallation(version, executable, True)

    def _managed_extras(self) -> tuple[str, ...]:
        """Carry agent-prepared providers into the next versioned runtime."""
        path = self.root / "capabilities.json"
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return ()
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return ()
        extras = payload.get("extras")
        if not isinstance(extras, list):
            return ()
        allowed = {"cpu", "execution", "memory", "test", "trace", "torch"}
        return tuple(
            sorted(value for value in extras if isinstance(value, str) and value in allowed)
        )

    async def verify(self, executable: Path, version: str) -> None:
        try:
            outcome = await self.broker.run(
                ExecutionRequest(
                    argv=(str(executable), "--version"),
                    cwd=Path.cwd(),
                    environment_allowlist=("PATH",),
                    allowed_working_roots=(Path.cwd(),),
                    timeout_seconds=30,
                    max_output_bytes=1024 * 1024,
                )
            )
        except (DomainError, OSError) as exc:
            raise self._verification_error(executable, str(exc)) from exc
        if outcome.process.exit_code != 0:
            raise self._verification_error(
                executable,
                _output_detail(outcome) or "the version command failed",
            )
        stdout = outcome.stdout.decode("utf-8", errors="replace").strip()
        if stdout != version:
            raise self._verification_error(
                executable,
                f"expected version {version}, got {stdout!r}",
            )

        parameters = StdioServerParameters(
            command=str(executable),
            args=["mcp", "serve", "--project-root", "."],
            cwd=Path.cwd(),
        )
        try:
            with anyio.fail_after(60):
                async with Client(stdio_client(parameters), raise_exceptions=True) as client:
                    tools = await client.list_tools()
        except Exception as exc:
            raise self._verification_error(executable, str(exc)) from exc
        if not tools.tools:
            raise self._verification_error(executable, "the MCP server exposed no tools")

    @staticmethod
    def _validated_version(version: str) -> str:
        try:
            parsed = Version(version)
        except InvalidVersion as exc:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Invalid flameox runtime version: {version}",
            ) from exc
        if str(parsed) != version:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Runtime version must be canonical: {version}",
            )
        return version

    def _installation_error(self, version: str, detail: str) -> DomainError:
        return DomainError(
            ErrorCode.PROCESS_FAILED,
            f"Could not install flameox {version} with uv.",
            details={"error": detail, "distribution": self.distribution},
            remediation=(
                "Install uv, check network access, then run `npx flameox@latest setup` again.",
            ),
        )

    @staticmethod
    def _manifest_matches(path: Path, version: str, executable: Path) -> bool:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(value, dict):
            return False
        return value == {
            "schema_version": 1,
            "distribution": ManagedRuntime.distribution,
            "version": version,
            "executable": str(executable),
        }

    @staticmethod
    def _verification_error(executable: Path, detail: str) -> DomainError:
        return DomainError(
            ErrorCode.PROCESS_FAILED,
            "The staged flameox runtime failed verification.",
            details={"executable": str(executable), "error": detail},
            remediation=("The previous configured runtime remains active.",),
        )


TRACE_PROCESSOR_VERSION = "v55.1"


def install_trace_processor(
    workspace: Workspace,
    *,
    cancel_event: threading.Event | None = None,
    broker: SubprocessBroker | None = None,
) -> TraceProcessorInstallation:
    """Stage the pinned user-space Trace Processor without requiring host privileges."""
    temporary: Path | None = None
    try:
        platform_key = {
            ("linux", "x86_64"): "linux-amd64",
            ("linux", "amd64"): "linux-amd64",
            ("linux", "aarch64"): "linux-arm64",
            ("darwin", "x86_64"): "mac-amd64",
            ("darwin", "arm64"): "mac-arm64",
        }.get((sys.platform, _machine()))
        if platform_key is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "FlameOx has no managed Trace Processor binary for this platform.",
                details={"next_tool": "list_capabilities"},
                remediation=(
                    "Install the official Perfetto Trace Processor for this platform or set "
                    "analysis.trace_processor_path in the workspace policy.",
                ),
            )

        target = workspace.paths.root / "tools" / "trace_processor_shell"
        if target.is_file() and os.access(target, os.X_OK):
            _check_staging_cancelled(cancel_event)
            _verify_trace_processor(target, cancel_event=cancel_event, broker=broker)
            return TraceProcessorInstallation(TRACE_PROCESSOR_VERSION, target, False)

        url = (
            "https://commondatastorage.googleapis.com/perfetto-luci-artifacts/"
            f"{TRACE_PROCESSOR_VERSION}/{platform_key}/trace_processor_shell"
        )
        staging = workspace.paths.staging
        staging.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=staging,
            prefix="trace-processor-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(url, timeout=120) as response:
                total = 0
                while chunk := response.read(1024 * 1024):
                    _check_staging_cancelled(cancel_event)
                    total += len(chunk)
                    if total > 512 * 1024 * 1024:
                        raise DomainError(
                            ErrorCode.ARTIFACT_TOO_LARGE,
                            "The managed Trace Processor download exceeded 512 MiB.",
                            details={"next_tool": "start_capability_setup", "adapter": "perfetto"},
                        )
                    stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o755)
        _check_staging_cancelled(cancel_event)
        _verify_trace_processor(temporary, cancel_event=cancel_event, broker=broker)
        target.parent.mkdir(parents=True, exist_ok=True)
        with workspace.write_locked():
            _check_staging_cancelled(cancel_event)
            os.replace(temporary, target)
            config = workspace.config
            updated = config.model_copy(
                update={
                    "analysis": config.analysis.model_copy(
                        update={"trace_processor_path": str(target)}
                    )
                }
            )
            atomic_write_text(workspace.paths.config, updated.to_toml())
        temporary = None
    except DomainError as exc:
        raise _annotate_staging_error(exc) from exc
    except (OSError, urllib.error.URLError, ValueError) as exc:
        category = _staging_failure_category(exc)
        detail = _bounded_staging_detail(exc)
        raise DomainError(
            ErrorCode.PROCESS_FAILED,
            "FlameOx could not stage the managed Trace Processor.",
            retryable=True,
            details={
                "next_tool": "start_capability_setup",
                "adapter": "perfetto",
                "failure_category": category,
                "failure_detail": detail,
            },
            remediation=(
                "Retry start_capability_setup; if the download remains unavailable, install "
                "the official user-space binary or configure analysis.trace_processor_path.",
            ),
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return TraceProcessorInstallation(TRACE_PROCESSOR_VERSION, target, True)


def _annotate_staging_error(error: DomainError) -> DomainError:
    """Retain a bounded cause when a staging helper already raised a domain error."""
    details = dict(error.details)
    details.setdefault("failure_category", _domain_failure_category(error))
    details.setdefault(
        "failure_detail",
        _bounded_staging_detail(details.get("error") or error.message),
    )
    details.setdefault("phase", "staging_trace_processor")
    return DomainError(
        error.code,
        error.message,
        retryable=error.retryable,
        details=details,
        remediation=error.remediation,
        run_id=error.run_id,
    )


def _domain_failure_category(error: DomainError) -> str:
    if error.code is ErrorCode.PROCESS_CANCELLED:
        return "cancelled"
    if error.code is ErrorCode.PROCESS_TIMEOUT:
        return "timeout"
    if error.code is ErrorCode.ARTIFACT_TOO_LARGE:
        return "download_limit"
    if error.code is ErrorCode.CAPABILITY_UNAVAILABLE:
        return "unsupported_platform"
    return "verification"


def _staging_failure_category(error: BaseException) -> str:
    if isinstance(error, urllib.error.URLError):
        return "network"
    if isinstance(error, OSError):
        return "filesystem"
    return "verification"


def _bounded_staging_detail(error: object) -> str:
    detail = " ".join(str(error).split())
    return detail[:500] or "The staging operation returned no diagnostic detail."


def _verify_trace_processor(
    executable: Path,
    *,
    cancel_event: threading.Event | None = None,
    broker: SubprocessBroker | None = None,
) -> None:
    execution_broker = broker or SubprocessBroker()
    try:
        outcome = _run_brokered_sync(
            execution_broker,
            ExecutionRequest(
                argv=(str(executable), "--version"),
                cwd=Path.cwd(),
                environment_allowlist=(),
                allowed_working_roots=(Path.cwd(),),
                timeout_seconds=30,
                max_output_bytes=1024 * 1024,
            ),
            cancel_event=cancel_event,
            cancellation_message="Trace Processor staging was cancelled before publication.",
            cancellation_details={"next_tool": "start_capability_setup", "adapter": "perfetto"},
        )
    except (DomainError, OSError) as exc:
        if isinstance(exc, DomainError) and exc.code in {
            ErrorCode.PROCESS_CANCELLED,
            ErrorCode.PROCESS_TIMEOUT,
        }:
            raise
        raise DomainError(
            ErrorCode.PROCESS_FAILED,
            "The staged Trace Processor failed its bounded version check.",
            details={"next_tool": "start_capability_setup", "adapter": "perfetto"},
        ) from exc
    _validate_trace_processor_result(
        outcome.process.exit_code,
        outcome.stdout.decode("utf-8", errors="replace"),
        outcome.stderr.decode("utf-8", errors="replace"),
    )


async def _run_brokered(
    broker: SubprocessBroker,
    request: ExecutionRequest,
    *,
    cancel_event: threading.Event | None,
    cancellation_message: str,
    cancellation_details: dict[str, str],
) -> ExecutionOutcome:
    execution = asyncio.create_task(broker.run(request))
    if cancel_event is None:
        return await execution

    cancellation = asyncio.create_task(_wait_for_cancellation(cancel_event))
    done, _ = await asyncio.wait(
        (execution, cancellation),
        return_when=asyncio.FIRST_COMPLETED,
    )
    if cancellation in done and execution not in done:
        execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)
        raise DomainError(
            ErrorCode.PROCESS_CANCELLED,
            cancellation_message,
            retryable=True,
            details=cancellation_details,
        )
    cancellation.cancel()
    await asyncio.gather(cancellation, return_exceptions=True)
    return await execution


async def _wait_for_cancellation(cancel_event: threading.Event) -> None:
    while not cancel_event.is_set():
        await asyncio.sleep(0.05)


def _run_brokered_sync(
    broker: SubprocessBroker,
    request: ExecutionRequest,
    *,
    cancel_event: threading.Event | None,
    cancellation_message: str,
    cancellation_details: dict[str, str],
) -> ExecutionOutcome:
    coroutine = _run_brokered(
        broker,
        request,
        cancel_event=cancel_event,
        cancellation_message=cancellation_message,
        cancellation_details=cancellation_details,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result: list[ExecutionOutcome] = []
    error: list[BaseException] = []

    def run() -> None:
        try:
            result.append(asyncio.run(coroutine))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=run, name="flameox-broker-bridge")
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def _output_detail(outcome: ExecutionOutcome) -> str:
    return (
        outcome.stderr.decode("utf-8", errors="replace").strip()
        or outcome.stdout.decode("utf-8", errors="replace").strip()
    )


def _validate_trace_processor_result(
    returncode: int | None,
    stdout: str,
    stderr: str,
) -> None:
    output = f"{stdout}\n{stderr}"
    if returncode != 0 or TRACE_PROCESSOR_VERSION.removeprefix("v") not in output:
        raise DomainError(
            ErrorCode.PROCESS_FAILED,
            "The staged Trace Processor failed its bounded version check.",
            details={"next_tool": "start_capability_setup", "adapter": "perfetto"},
            remediation=(stderr.strip()[:500] or "Retry the managed setup.",),
        )


def _check_staging_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise DomainError(
            ErrorCode.PROCESS_CANCELLED,
            "Trace Processor staging was cancelled before publication.",
            retryable=True,
            details={"next_tool": "start_capability_setup", "adapter": "perfetto"},
        )


def _machine() -> str:
    return os.uname().machine.lower() if hasattr(os, "uname") else ""
