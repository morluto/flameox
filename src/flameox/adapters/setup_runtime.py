from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
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

    def __init__(self, root: Path, *, uv_executable: str = "uv") -> None:
        self.root = root
        self.uv_executable = uv_executable

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
        environment = os.environ.copy()
        environment["UV_TOOL_DIR"] = str(runtime_root / "tools")
        environment["UV_TOOL_BIN_DIR"] = str(runtime_root / "bin")
        try:
            extras = self._managed_extras()
            distribution = (
                f"{self.distribution}[{','.join(extras)}]" if extras else self.distribution
            )
            completed = subprocess.run(
                [
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
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise self._installation_error(version, str(exc)) from exc
        if completed.returncode != 0 or not executable.is_file():
            detail = completed.stderr.strip() or completed.stdout.strip()
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
            completed = subprocess.run(
                [str(executable), "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise self._verification_error(executable, str(exc)) from exc
        if completed.returncode != 0:
            raise self._verification_error(
                executable,
                completed.stderr.strip() or "the version command failed",
            )
        if completed.stdout.strip() != version:
            raise self._verification_error(
                executable,
                f"expected version {version}, got {completed.stdout.strip()!r}",
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
            remediation=("Install uv, check network access, and run setup again.",),
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


def install_trace_processor(workspace: Workspace) -> TraceProcessorInstallation:
    """Stage the pinned user-space Trace Processor without requiring host privileges."""
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
        _verify_trace_processor(target)
        return TraceProcessorInstallation(TRACE_PROCESSOR_VERSION, target, False)

    url = (
        "https://commondatastorage.googleapis.com/perfetto-luci-artifacts/"
        f"{TRACE_PROCESSOR_VERSION}/{platform_key}/trace_processor_shell"
    )
    staging = workspace.paths.staging
    staging.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=staging,
            prefix="trace-processor-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(url, timeout=120) as response:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 512 * 1024 * 1024:
                        raise DomainError(
                            ErrorCode.ARTIFACT_TOO_LARGE,
                            "The managed Trace Processor download exceeded 512 MiB.",
                            details={"next_tool": "prepare_capabilities", "adapter": "perfetto"},
                        )
                    stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o755)
        _verify_trace_processor(temporary)
        target.parent.mkdir(parents=True, exist_ok=True)
        with workspace.write_locked():
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
    except DomainError:
        raise
    except (OSError, urllib.error.URLError, ValueError) as exc:
        raise DomainError(
            ErrorCode.PROCESS_FAILED,
            "FlameOx could not stage the managed Trace Processor.",
            retryable=True,
            details={"next_tool": "prepare_capabilities", "adapter": "perfetto"},
            remediation=(
                "Retry prepare_capabilities; if the download remains unavailable, install "
                "the official user-space binary or configure analysis.trace_processor_path.",
            ),
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return TraceProcessorInstallation(TRACE_PROCESSOR_VERSION, target, True)


def _verify_trace_processor(executable: Path) -> None:
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DomainError(
            ErrorCode.PROCESS_FAILED,
            "The staged Trace Processor failed its bounded version check.",
            details={"next_tool": "prepare_capabilities", "adapter": "perfetto"},
        ) from exc
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0 or TRACE_PROCESSOR_VERSION.removeprefix("v") not in output:
        raise DomainError(
            ErrorCode.PROCESS_FAILED,
            "The staged Trace Processor failed its bounded version check.",
            details={"next_tool": "prepare_capabilities", "adapter": "perfetto"},
            remediation=(completed.stderr.strip()[:500] or "Retry the managed setup.",),
        )


def _machine() -> str:
    return os.uname().machine.lower() if hasattr(os, "uname") else ""
