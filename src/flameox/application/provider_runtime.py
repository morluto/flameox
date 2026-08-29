from __future__ import annotations

import hashlib
import importlib.metadata
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
from urllib.parse import unquote, urlparse

import portalocker
from packaging.requirements import Requirement
from pydantic import Field

from flameox import __version__
from flameox.atomic import atomic_write_json
from flameox.command_binding import ExecutableResolver
from flameox.domain import CapabilityExtra, DomainError, ErrorCode, digest_model, is_digest
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


class ProviderRuntimeIdentity(ContractModel):
    """Inputs that determine one managed provider environment."""

    flameox_version: str
    flameox_package_source: Literal["index", "local_wheel"]
    flameox_package_sha256: str | None = None
    installation_lock_sha256: str | None = None
    flameox_source_tree_sha256: str | None = None
    flameox_source_revision: str | None = None
    flameox_source_dirty: bool | None = None
    extra: CapabilityExtra
    requirement: str
    python_requirement: str
    platform: str
    architecture: str
    uv_version: str
    uv_sha256: str


class ProviderRuntimeReceipt(ProviderRuntimeIdentity):
    """Identity and verification evidence for one immutable provider environment."""

    environment_id: str
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


@dataclass(frozen=True, slots=True)
class _FlameoxPackage:
    source: Literal["index", "local_wheel"]
    install_arguments: tuple[str, ...]
    package_sha256: str | None
    installation_lock_sha256: str | None
    source_tree_sha256: str | None
    source_revision: str | None
    source_dirty: bool | None


class ProviderRuntimeManager:
    """Build version-addressed provider environments without changing the control runtime."""

    def __init__(
        self,
        root: Path,
        *,
        broker: SubprocessBroker | None = None,
        uv_executable: str = "uv",
        package_source: Literal["auto", "index", "source"] = "auto",
        flameox_source_root: Path | None = None,
    ) -> None:
        self.root = root
        self.broker = broker or SubprocessBroker()
        self.uv_executable = uv_executable
        detected_source = (
            flameox_source_root.resolve()
            if flameox_source_root is not None
            else self._active_source_root()
        )
        if package_source == "source" and detected_source is None:
            raise ValueError("source package mode requires a Flameox source checkout")
        self.flameox_source_root = None if package_source == "index" else detected_source

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
        self.root.mkdir(parents=True, exist_ok=True)
        flameox_package = self._resolve_flameox_package(
            uv_binding,
            provider_requirement=parsed,
            run=run,
        )
        identity = ProviderRuntimeIdentity(
            flameox_version=__version__,
            flameox_package_source=flameox_package.source,
            flameox_package_sha256=flameox_package.package_sha256,
            installation_lock_sha256=flameox_package.installation_lock_sha256,
            flameox_source_tree_sha256=flameox_package.source_tree_sha256,
            flameox_source_revision=flameox_package.source_revision,
            flameox_source_dirty=flameox_package.source_dirty,
            extra=extra,
            requirement=str(parsed),
            python_requirement=python_requirement,
            platform=host_platform,
            architecture=architecture,
            uv_version=uv_version,
            uv_sha256=uv_binding.identity.sha256,
        )
        environment_id = digest_model(identity)
        directory_name = self._directory_name(environment_id)
        destination = self.root / directory_name
        lock_path = self.root / f"{directory_name}.lock"
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
                        "--relocatable",
                        "--python",
                        sys.executable,
                        "--no-project",
                        str(staging),
                    ),
                    writable_root=staging,
                    phase="create_environment",
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
                        *flameox_package.install_arguments,
                    ),
                    writable_root=staging,
                    timeout_seconds=1_800,
                    phase="install_provider",
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
                    **identity.model_dump(),
                    environment_id=environment_id,
                    python_relative_path=python_relative.as_posix(),
                    python_sha256=self._sha256(python.resolve()),
                    distributions=distributions,
                    executable_relative_path=executable_relative,
                    executable_sha256=executable_sha256,
                    environment_tree_sha256=self._tree_sha256(staging),
                    limitations=(
                        "Provider isolation limits dependency and crash propagation; it is not "
                        "a sandbox against a malicious provider.",
                        *(
                            (
                                "The local Flameox source checkout has no observable Git "
                                "revision or dirty-state metadata; wheel and source-tree digests "
                                "remain authoritative.",
                            )
                            if flameox_package.source == "local_wheel"
                            and flameox_package.source_revision is None
                            else ()
                        ),
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
        try:
            source_tree_sha256 = (
                self._source_tree_sha256(self.flameox_source_root)
                if self.flameox_source_root is not None
                else None
            )
        except (DomainError, OSError):
            return None
        expected = {
            "flameox_version": __version__,
            "flameox_package_source": (
                "local_wheel" if self.flameox_source_root is not None else "index"
            ),
            "flameox_source_tree_sha256": source_tree_sha256,
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
                if path.is_dir()
                and not path.name.startswith(".")
                and path.name not in {"locks", "wheels"}
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
    def _active_source_root() -> Path | None:
        try:
            direct_url = importlib.metadata.distribution("flameox").read_text("direct_url.json")
            payload = json.loads(direct_url) if direct_url is not None else {}
            url = payload.get("url")
            directory = payload.get("dir_info")
            if (
                not isinstance(url, str)
                or not isinstance(directory, dict)
                or directory.get("editable") is not True
            ):
                return None
            parsed = urlparse(url)
            if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
                return None
            path = Path(unquote(parsed.path)).resolve()
            return path if (path / "pyproject.toml").is_file() else None
        except (OSError, ValueError, json.JSONDecodeError, importlib.metadata.PackageNotFoundError):
            return None

    def _resolve_flameox_package(
        self,
        uv_binding: ResolvedExecutable,
        *,
        provider_requirement: Requirement,
        run: Callable[[ExecutionRequest], ExecutionOutcome],
    ) -> _FlameoxPackage:
        source_root = self.flameox_source_root
        if source_root is None:
            return self._resolve_index_lock(
                uv_binding,
                provider_requirement=provider_requirement,
                run=run,
            )
        source_tree_sha256 = self._source_tree_sha256(source_root)
        revision, dirty = self._git_facts(source_root, run=run)
        build_root = self.root / f".flameox-wheel.{secrets.token_hex(12)}"
        build_root.mkdir(mode=0o700)
        try:
            self._run(
                uv_binding,
                (
                    str(uv_binding.invocation_path),
                    "build",
                    "--wheel",
                    "--out-dir",
                    str(build_root),
                    "--no-config",
                    str(source_root),
                ),
                writable_root=build_root,
                cwd=source_root,
                timeout_seconds=1_800,
                phase="build_flameox_wheel",
                run=run,
            )
            wheels = tuple(build_root.glob("*.whl"))
            if len(wheels) != 1:
                raise self._invalid("source build did not produce exactly one Flameox wheel")
            wheel = wheels[0]
            package_sha256 = self._sha256(wheel)
            if self._source_tree_sha256(source_root) != source_tree_sha256:
                raise self._invalid("Flameox source changed while its wheel was being built")
            if self._git_facts(source_root, run=run) != (revision, dirty):
                raise self._invalid("Flameox revision changed while its wheel was being built")
            wheels_root = self.root / "wheels"
            wheels_root.mkdir(mode=0o700, exist_ok=True)
            wheel_identity_root = wheels_root / package_sha256.removeprefix("sha256:")
            wheel_identity_root.mkdir(mode=0o700, exist_ok=True)
            preserved = wheel_identity_root / wheel.name
            if preserved.exists():
                if self._sha256(preserved) != package_sha256:
                    raise self._invalid("cached Flameox wheel contradicts its content identity")
            else:
                os.replace(wheel, preserved)
            return _FlameoxPackage(
                source="local_wheel",
                install_arguments=(str(preserved), str(provider_requirement)),
                package_sha256=package_sha256,
                installation_lock_sha256=None,
                source_tree_sha256=source_tree_sha256,
                source_revision=revision,
                source_dirty=dirty,
            )
        finally:
            shutil.rmtree(build_root, ignore_errors=True)

    def _resolve_index_lock(
        self,
        uv_binding: ResolvedExecutable,
        *,
        provider_requirement: Requirement,
        run: Callable[[ExecutionRequest], ExecutionOutcome],
    ) -> _FlameoxPackage:
        staging = self.root / f".flameox-lock.{secrets.token_hex(12)}"
        staging.mkdir(mode=0o700)
        requirements = staging / "requirements.in"
        lock = staging / "requirements.txt"
        requirements.write_text(
            f"flameox=={__version__}\n{provider_requirement}\n",
            encoding="utf-8",
        )
        try:
            self._run(
                uv_binding,
                (
                    str(uv_binding.invocation_path),
                    "pip",
                    "compile",
                    "--generate-hashes",
                    "--no-header",
                    "--no-config",
                    "--no-sources",
                    "--output-file",
                    str(lock),
                    str(requirements),
                ),
                writable_root=staging,
                timeout_seconds=1_800,
                phase="resolve_provider_lock",
                run=run,
            )
            lock_sha256 = self._sha256(lock)
            locks_root = self.root / "locks" / lock_sha256.removeprefix("sha256:")
            locks_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            preserved = locks_root / "requirements.txt"
            if preserved.exists():
                if self._sha256(preserved) != lock_sha256:
                    raise self._invalid("cached provider lock contradicts its content identity")
            else:
                os.replace(lock, preserved)
            return _FlameoxPackage(
                source="index",
                install_arguments=("--require-hashes", "-r", str(preserved)),
                package_sha256=None,
                installation_lock_sha256=lock_sha256,
                source_tree_sha256=None,
                source_revision=None,
                source_dirty=None,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _source_tree_sha256(source_root: Path) -> str:
        digest = hashlib.sha256()
        roots = (
            source_root / "pyproject.toml",
            source_root / "README.md",
            source_root / "LICENSE",
            source_root / "src" / "flameox",
        )
        files = sorted(
            (
                path
                for root in roots
                for path in ((root,) if root.is_file() else root.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts
            ),
            key=lambda path: path.relative_to(source_root).as_posix(),
        )
        if not files:
            raise ProviderRuntimeManager._invalid("Flameox source tree contains no build inputs")
        if len(files) > 10_000:
            raise ProviderRuntimeManager._invalid("Flameox source tree exceeds the file budget")
        total_bytes = 0
        for path in files:
            relative = path.relative_to(source_root).as_posix().encode()
            digest.update(relative + b"\0")
            if path.is_symlink():
                digest.update(b"L\0" + os.readlink(path).encode())
            else:
                size = path.stat().st_size
                total_bytes += size
                if total_bytes > 512 * 1024 * 1024:
                    raise ProviderRuntimeManager._invalid(
                        "Flameox source tree exceeds the byte budget"
                    )
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
            digest.update(b"\0")
        return "sha256:" + digest.hexdigest()

    def _git_facts(
        self,
        source_root: Path,
        *,
        run: Callable[[ExecutionRequest], ExecutionOutcome],
    ) -> tuple[str | None, bool | None]:
        git = ExecutableResolver().resolve_host_tool("git", cwd=source_root)
        if git is None or not (source_root / ".git").exists():
            return None, None

        def invoke(*arguments: str) -> ExecutionOutcome:
            return run(
                ExecutionRequest(
                    argv=(str(git.invocation_path), *arguments),
                    executable_binding=git,
                    cwd=source_root,
                    environment_allowlist=("PATH",),
                    allowed_working_roots=(source_root,),
                    timeout_seconds=30,
                    max_output_bytes=1024 * 1024,
                )
            )

        head = invoke("rev-parse", "HEAD")
        status = invoke("status", "--porcelain", "--untracked-files=normal")
        if head.process.exit_code != 0 or status.process.exit_code != 0:
            return None, None
        revision = head.stdout.decode("utf-8", errors="replace").strip()
        return (revision or None), bool(status.stdout.strip())

    @staticmethod
    def _directory_name(environment_id: str) -> str:
        return environment_id.removeprefix("sha256:")

    def _read_verified(self, root: Path, *, environment_id: str) -> ProviderRuntime | None:
        try:
            receipt = ProviderRuntimeReceipt.model_validate_json(
                (root / "provider-runtime.json").read_bytes()
            )
            source_fields = (
                receipt.flameox_source_tree_sha256,
                receipt.flameox_source_revision,
                receipt.flameox_source_dirty,
            )
            if (
                receipt.flameox_package_source == "local_wheel"
                and (
                    receipt.flameox_package_sha256 is None
                    or receipt.installation_lock_sha256 is not None
                    or receipt.flameox_source_tree_sha256 is None
                )
            ) or (
                receipt.flameox_package_source == "index"
                and (
                    receipt.flameox_package_sha256 is not None
                    or receipt.installation_lock_sha256 is None
                    or any(value is not None for value in source_fields)
                )
            ):
                return None
            python = self._contained_receipt_path(
                root,
                receipt.python_relative_path,
                allow_final_symlink=True,
            )
            if (
                receipt.environment_id != environment_id
                or digest_model(self._identity(receipt)) != environment_id
                or self._sha256(python.resolve()) != receipt.python_sha256
            ):
                return None
            executable = (
                self._contained_receipt_path(root, receipt.executable_relative_path)
                if receipt.executable_relative_path is not None
                else None
            )
            if executable is not None and self._sha256(executable) != receipt.executable_sha256:
                return None
            if receipt.flameox_package_source == "local_wheel":
                if receipt.flameox_package_sha256 is None or not is_digest(
                    receipt.flameox_package_sha256
                ):
                    return None
                wheels_root = self.root / "wheels"
                wheel_root = wheels_root / receipt.flameox_package_sha256.removeprefix("sha256:")
                if wheel_root.is_symlink() or not wheel_root.resolve().is_relative_to(
                    wheels_root.resolve()
                ):
                    return None
                wheels = tuple(wheel_root.glob("*.whl"))
                if (
                    len(wheels) != 1
                    or wheels[0].is_symlink()
                    or not wheels[0].is_file()
                    or self._sha256(wheels[0]) != receipt.flameox_package_sha256
                ):
                    return None
            elif receipt.installation_lock_sha256 is not None:
                if not is_digest(receipt.installation_lock_sha256):
                    return None
                locks_root = self.root / "locks"
                lock_root = locks_root / receipt.installation_lock_sha256.removeprefix("sha256:")
                lock = (
                    lock_root / "requirements.txt"
                )
                if (
                    lock_root.is_symlink()
                    or not lock_root.resolve().is_relative_to(locks_root.resolve())
                    or lock.is_symlink()
                    or not lock.is_file()
                    or self._sha256(lock) != receipt.installation_lock_sha256
                ):
                    return None
            else:
                return None
            if self._tree_sha256(root) != receipt.environment_tree_sha256:
                return None
        except (OSError, ValueError):
            return None
        return ProviderRuntime(root, receipt)

    @staticmethod
    def _identity(receipt: ProviderRuntimeReceipt) -> ProviderRuntimeIdentity:
        return ProviderRuntimeIdentity.model_validate(
            receipt.model_dump(include=set(ProviderRuntimeIdentity.model_fields))
        )

    @staticmethod
    def _contained_receipt_path(
        root: Path,
        value: str,
        *,
        allow_final_symlink: bool = False,
    ) -> Path:
        relative = Path(value)
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != value
        ):
            raise ValueError("provider receipt path is not normalized and relative")
        resolved_root = root.resolve()
        candidate = resolved_root / relative
        if not candidate.parent.resolve().is_relative_to(resolved_root):
            raise ValueError("provider receipt path escapes its runtime")
        if allow_final_symlink and candidate.is_symlink():
            return candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise ValueError("provider receipt path escapes its runtime")
        return resolved

    def _run(
        self,
        uv_binding: ResolvedExecutable,
        argv: tuple[str, ...],
        *,
        writable_root: Path,
        cwd: Path | None = None,
        timeout_seconds: float = 300,
        phase: str,
        run: Callable[[ExecutionRequest], ExecutionOutcome],
    ) -> None:
        outcome = run(
            ExecutionRequest(
                argv=argv,
                executable_binding=uv_binding,
                cwd=cwd or Path.cwd(),
                environment_allowlist=INSTALLER_ENVIRONMENT_ALLOWLIST,
                environment_overrides={"UV_NO_PROGRESS": "1"},
                allowed_working_roots=(cwd or Path.cwd(), writable_root),
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
                retryable="no solution found" not in detail.casefold(),
                details={"phase": phase, "error": detail},
                remediation=(
                    "Inspect the bounded setup output, repair the package source or provider "
                    "requirement, and start setup again.",
                ),
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
