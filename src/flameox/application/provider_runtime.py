from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import portalocker
from packaging.requirements import Requirement
from pydantic import Field

from flameox import __version__
from flameox.atomic import atomic_write_json
from flameox.command_binding import ExecutableResolver
from flameox.domain import CapabilityExtra, DomainError, ErrorCode, digest_model
from flameox.domain.executables import ResolvedExecutable
from flameox.execution import (
    INSTALLER_ENVIRONMENT_ALLOWLIST,
    ExecutionOutcome,
    ExecutionRequest,
    SubprocessBroker,
)
from flameox.models import ContractModel

_PROBE = """
import importlib.metadata
import json
import sys

names = sys.argv[1:]
print(json.dumps({
    "executable": sys.executable,
    "prefix": sys.prefix,
    "versions": {name: importlib.metadata.version(name) for name in names},
}, allow_nan=False, separators=(",", ":"), sort_keys=True))
""".strip()


class ProviderRuntimeReceipt(ContractModel):
    """Identity and verification evidence for one immutable provider environment."""

    schema_version: Literal[3] = 3
    installation_profile: Literal["core-plus-provider-v1"] = "core-plus-provider-v1"
    environment_id: str
    flameox_version: str
    extra: CapabilityExtra
    requirement: str
    python_requirement: str
    platform: str
    architecture: str
    uv_version: str
    uv_sha256: str
    python_relative_path: str
    python_sha256: str
    distributions: dict[str, str]
    executable_relative_path: str | None = None
    executable_sha256: str | None = None
    environment_tree_sha256: str = "sha256:" + "0" * 64
    limitations: tuple[str, ...] = Field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ProviderRuntime:
    root: Path
    receipt: ProviderRuntimeReceipt

    @property
    def python(self) -> Path:
        return self.root / self.receipt.python_relative_path

    @property
    def executable(self) -> Path | None:
        relative = self.receipt.executable_relative_path
        return self.root / relative if relative is not None else None


class ProviderRuntimeManager:
    """Build version-addressed provider environments without changing the control runtime."""

    def __init__(
        self,
        root: Path,
        *,
        broker: SubprocessBroker | None = None,
        uv_executable: str = "uv",
    ) -> None:
        self.root = root
        self.broker = broker or SubprocessBroker()
        self.uv_executable = uv_executable

    def prepare(
        self,
        *,
        extra: CapabilityExtra,
        requirement: str,
        executable_name: str | None = None,
        request_runner: Callable[[ExecutionRequest], ExecutionOutcome] | None = None,
    ) -> ProviderRuntime:
        parsed = Requirement(requirement)
        uv_binding = ExecutableResolver().require_host_tool(self.uv_executable, cwd=Path.cwd())
        run = request_runner or self.broker.run_sync
        uv_version = self._uv_version(uv_binding, run=run)
        python_requirement = f"{sys.version_info.major}.{sys.version_info.minor}"
        host_platform = platform.system().lower()
        architecture = platform.machine().lower()
        identity = {
            "schema_version": 3,
            "installation_profile": "core-plus-provider-v1",
            "flameox_version": __version__,
            "extra": extra.value,
            "requirement": str(parsed),
            "python_requirement": python_requirement,
            "platform": host_platform,
            "architecture": architecture,
            "uv_version": uv_version,
            "uv_sha256": uv_binding.identity.sha256,
        }
        environment_id = digest_model(identity)
        directory_name = self._directory_name(environment_id)
        destination = self.root / directory_name
        lock_path = self.root / f"{directory_name}.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(lock_path, mode="a", timeout=30):
            existing = self._read_verified(destination, environment_id=environment_id)
            if existing is not None:
                return existing
            staging = self.root / f".{directory_name}.{secrets.token_hex(12)}"
            staging.mkdir(mode=0o700)
            try:
                python_relative = self._python_relative_path()
                python = staging / python_relative
                self._run(
                    uv_binding,
                    (
                        str(uv_binding.invocation_path),
                        "venv",
                        "--python",
                        sys.executable,
                        "--no-project",
                        str(staging),
                    ),
                    writable_root=staging,
                    run=run,
                )
                self._run(
                    uv_binding,
                    (
                        str(uv_binding.invocation_path),
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "--no-config",
                        "--no-sources",
                        f"flameox=={__version__}",
                        str(parsed),
                    ),
                    writable_root=staging,
                    timeout_seconds=1_800,
                    run=run,
                )
                distributions = self._verify_python(
                    python,
                    names=("flameox", parsed.name),
                    writable_root=staging,
                    run=run,
                )
                if distributions["flameox"] != __version__:
                    raise self._invalid("provider environment contains another Flameox version")
                if not parsed.specifier.contains(distributions[parsed.name], prereleases=True):
                    raise self._invalid("provider environment does not satisfy its requirement")
                executable_relative: str | None = None
                executable_sha256: str | None = None
                if executable_name is not None:
                    executable = staging / self._scripts_relative_path() / executable_name
                    if os.name == "nt" and executable.suffix.casefold() != ".exe":
                        executable = executable.with_suffix(".exe")
                    if not executable.is_file() or not os.access(executable, os.X_OK):
                        raise self._invalid("provider environment did not expose its executable")
                    executable_relative = executable.relative_to(staging).as_posix()
                    executable_sha256 = self._sha256(executable)
                receipt = ProviderRuntimeReceipt(
                    environment_id=environment_id,
                    flameox_version=__version__,
                    extra=extra,
                    requirement=str(parsed),
                    python_requirement=python_requirement,
                    platform=host_platform,
                    architecture=architecture,
                    uv_version=uv_version,
                    uv_sha256=uv_binding.identity.sha256,
                    python_relative_path=python_relative.as_posix(),
                    python_sha256=self._sha256(python.resolve()),
                    distributions=distributions,
                    executable_relative_path=executable_relative,
                    executable_sha256=executable_sha256,
                    environment_tree_sha256=self._tree_sha256(staging),
                    limitations=(
                        "Provider isolation limits dependency and crash propagation; it is not "
                        "a sandbox against a malicious provider.",
                    ),
                )
                atomic_write_json(
                    staging / "provider-runtime.json",
                    receipt.model_dump(mode="json"),
                )
                os.replace(staging, destination)
                return ProviderRuntime(destination, receipt)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise

    def find(
        self,
        *,
        extra: CapabilityExtra,
        requirement: str,
    ) -> ProviderRuntime | None:
        parsed = Requirement(requirement)
        uv_binding = ExecutableResolver().resolve_host_tool(self.uv_executable, cwd=Path.cwd())
        if uv_binding is None:
            return None
        expected = {
            "schema_version": 3,
            "installation_profile": "core-plus-provider-v1",
            "flameox_version": __version__,
            "extra": extra,
            "requirement": str(parsed),
            "python_requirement": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": platform.system().lower(),
            "architecture": platform.machine().lower(),
            "uv_sha256": uv_binding.identity.sha256,
        }
        try:
            candidates = tuple(
                path
                for path in self.root.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            )[:64]
        except OSError:
            return None
        for candidate in candidates:
            runtime = self._read_verified(
                candidate,
                environment_id=f"sha256:{candidate.name}",
            )
            if runtime is not None and all(
                getattr(runtime.receipt, name) == value for name, value in expected.items()
            ):
                return runtime
        return None

    def get(self, environment_id: str) -> ProviderRuntime | None:
        """Resolve one exact receipt identity without substituting a compatible runtime."""
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", environment_id):
            return None
        return self._read_verified(
            self.root / self._directory_name(environment_id),
            environment_id=environment_id,
        )

    @staticmethod
    def _directory_name(environment_id: str) -> str:
        return environment_id.removeprefix("sha256:")

    def _read_verified(self, root: Path, *, environment_id: str) -> ProviderRuntime | None:
        try:
            receipt = ProviderRuntimeReceipt.model_validate_json(
                (root / "provider-runtime.json").read_bytes()
            )
            python = root / receipt.python_relative_path
            if (
                receipt.environment_id != environment_id
                or self._sha256(python.resolve()) != receipt.python_sha256
            ):
                return None
            executable = (
                root / receipt.executable_relative_path
                if receipt.executable_relative_path is not None
                else None
            )
            if executable is not None and self._sha256(executable) != receipt.executable_sha256:
                return None
            if self._tree_sha256(root) != receipt.environment_tree_sha256:
                return None
        except (OSError, ValueError):
            return None
        return ProviderRuntime(root, receipt)

    def _run(
        self,
        uv_binding: ResolvedExecutable,
        argv: tuple[str, ...],
        *,
        writable_root: Path,
        timeout_seconds: float = 300,
        run: Callable[[ExecutionRequest], ExecutionOutcome],
    ) -> None:
        outcome = run(
            ExecutionRequest(
                argv=argv,
                executable_binding=uv_binding,
                cwd=Path.cwd(),
                environment_allowlist=INSTALLER_ENVIRONMENT_ALLOWLIST,
                environment_overrides={"UV_NO_PROGRESS": "1"},
                allowed_working_roots=(Path.cwd(), writable_root),
                timeout_seconds=timeout_seconds,
                max_output_bytes=16 * 1024 * 1024,
            )
        )
        if outcome.process.exit_code != 0:
            detail = (
                outcome.stderr.decode(errors="replace").strip()
                or outcome.stdout.decode(errors="replace").strip()
            )[:500]
            raise DomainError(
                ErrorCode.PROCESS_FAILED,
                "The managed provider environment could not be constructed.",
                retryable=True,
                details={"error": detail},
            )

    @classmethod
    def _tree_sha256(cls, root: Path) -> str:
        """Identify immutable provider bytes while excluding runtime bytecode caches."""
        digest = hashlib.sha256()
        ordered = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        for path in ordered:
            relative = path.relative_to(root)
            if (
                path.name == "provider-runtime.json"
                or "__pycache__" in relative.parts
                or path.suffix in {".pyc", ".pyo"}
            ):
                continue
            metadata = path.lstat()
            encoded_path = relative.as_posix().encode()
            if path.is_symlink():
                digest.update(b"L\0" + encoded_path + b"\0" + os.readlink(path).encode() + b"\0")
            elif path.is_file():
                digest.update(
                    b"F\0" + encoded_path + b"\0" + str(metadata.st_mode & 0o777).encode() + b"\0"
                )
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
            elif path.is_dir():
                digest.update(b"D\0" + encoded_path + b"\0")
        return "sha256:" + digest.hexdigest()

    def _verify_python(
        self,
        python: Path,
        *,
        names: tuple[str, ...],
        writable_root: Path,
        run: Callable[[ExecutionRequest], ExecutionOutcome],
    ) -> dict[str, str]:
        binding = ExecutableResolver().require_host_tool(str(python), cwd=Path.cwd())
        outcome = run(
            ExecutionRequest(
                argv=(str(python), "-I", "-c", _PROBE, *names),
                executable_binding=binding,
                cwd=Path.cwd(),
                environment_allowlist=(),
                allowed_working_roots=(Path.cwd(), writable_root),
                timeout_seconds=30,
                max_output_bytes=64 * 1024,
            )
        )
        if outcome.process.exit_code != 0:
            raise self._invalid("provider environment verification failed")
        try:
            payload = json.loads(outcome.stdout)
            versions = payload["versions"]
            if not isinstance(versions, dict) or set(versions) != set(names):
                raise ValueError("unexpected distribution set")
            result = {name: versions[name] for name in names}
            if not all(isinstance(item, str) and item for item in result.values()):
                raise ValueError("invalid distribution version")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise self._invalid(
                "provider environment returned invalid verification evidence"
            ) from exc
        return result

    def _uv_version(
        self,
        binding: ResolvedExecutable,
        *,
        run: Callable[[ExecutionRequest], ExecutionOutcome],
    ) -> str:
        outcome = run(
            ExecutionRequest(
                argv=(str(binding.invocation_path), "--version"),
                executable_binding=binding,
                cwd=Path.cwd(),
                environment_allowlist=(),
                allowed_working_roots=(Path.cwd(),),
                timeout_seconds=10,
                max_output_bytes=4 * 1024,
            )
        )
        if outcome.process.exit_code != 0:
            raise self._invalid("uv version probe failed")
        value = outcome.stdout.decode("utf-8", errors="replace").strip()
        if not value.startswith("uv ") or len(value) > 100:
            raise self._invalid("uv version probe returned invalid output")
        return value.removeprefix("uv ").strip()

    @staticmethod
    def _python_relative_path() -> Path:
        return Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")

    @staticmethod
    def _scripts_relative_path() -> Path:
        return Path("Scripts") if os.name == "nt" else Path("bin")

    @staticmethod
    def _sha256(path: Path) -> str:
        with path.open("rb") as stream:
            return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()

    @staticmethod
    def _invalid(reason: str) -> DomainError:
        return DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "The managed provider environment failed verification.",
            details={"reason": reason},
        )
