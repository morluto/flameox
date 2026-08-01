from __future__ import annotations

import base64
import hashlib
import json
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

import portalocker
from platformdirs import user_data_path
from pydantic import Field

from flameox.adapters.client_setup import (
    ALL_SETUP_CLIENTS,
    ClientConfigEdit,
    ClientConfigRegistry,
    ClientPlanAction,
    Launcher,
    SetupClient,
)
from flameox.adapters.setup_runtime import ManagedRuntime, RuntimeInstallation
from flameox.domain import DomainError, ErrorCode
from flameox.models import ContractModel
from flameox.storage.atomic import atomic_write_bytes, atomic_write_json, fsync_directory


class RuntimeAction(StrEnum):
    INSTALL = "install"
    REUSE = "reuse"
    NOT_REQUIRED = "not_required"


class SetupOperation(StrEnum):
    CONFIGURE = "configure"
    REMOVE = "remove"
    ROLLBACK = "rollback"
    VERIFY = "verify"


class ClientSetupPlan(ContractModel):
    client: SetupClient
    display_name: str
    path: Path
    action: ClientPlanAction
    detected: bool


class SetupPlan(ContractModel):
    schema_version: Literal[1] = 1
    operation: SetupOperation
    version: str | None
    runtime_action: RuntimeAction
    runtime_executable: Path | None
    clients: tuple[ClientSetupPlan, ...]
    warnings: tuple[str, ...] = ()


class SetupReport(ContractModel):
    schema_version: Literal[1] = 1
    operation: SetupOperation
    version: str | None
    runtime_installed: bool
    changed_clients: tuple[SetupClient, ...]
    unchanged_clients: tuple[SetupClient, ...]
    verified: bool


class SetupInspection(ContractModel):
    schema_version: Literal[1] = 1
    active_version: str | None
    active_executable: Path | None
    configured_clients: tuple[SetupClient, ...]
    detected_clients: tuple[SetupClient, ...]
    installed_versions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResolvedSetupPlan:
    public: SetupPlan
    edits: tuple[ClientConfigEdit, ...]
    install_manifest_original: bytes | None
    install_manifest_mode: int


@dataclass(frozen=True, slots=True)
class _FileMutation:
    path: Path
    original: bytes | None
    updated: bytes
    mode: int


class _InstallManifest(ContractModel):
    schema_version: Literal[1] = 1
    active_version: str
    executable: Path


class _JournalMutation(ContractModel):
    path: Path
    original: str | None
    original_sha256: str | None
    updated_sha256: str
    mode: int = Field(ge=0, le=0o777)


class _SetupJournal(ContractModel):
    schema_version: Literal[1] = 1
    mutations: tuple[_JournalMutation, ...]


class RuntimeManager(Protocol):
    def executable(self, version: str) -> Path: ...

    def installed_versions(self) -> tuple[str, ...]: ...

    async def install(self, version: str) -> RuntimeInstallation: ...

    async def verify(self, executable: Path, version: str) -> None: ...


class SetupService:
    """Plan and atomically activate one managed flameox runtime for MCP clients."""

    def __init__(
        self,
        *,
        home: Path | None = None,
        data_root: Path | None = None,
        jsonc_helper: Path | None = None,
        node_executable: str = "node",
        uv_executable: str = "uv",
        runtime: RuntimeManager | None = None,
    ) -> None:
        resolved_home = home or Path.home()
        self.data_root = data_root or Path(user_data_path("flameox", appauthor=False))
        self.registry = ClientConfigRegistry(
            home=resolved_home,
            jsonc_helper=jsonc_helper,
            node_executable=node_executable,
        )
        self.runtime = runtime or ManagedRuntime(self.data_root, uv_executable=uv_executable)
        self.install_manifest = self.data_root / "install.json"
        self.journal_path = self.data_root / "setup-journal.json"
        self.lock_path = self.data_root / "setup.lock"

    def inspect(self) -> SetupInspection:
        self._recover_interrupted()
        manifest = self._read_install_manifest()
        return SetupInspection(
            active_version=manifest.active_version if manifest else None,
            active_executable=manifest.executable if manifest else None,
            configured_clients=self.registry.configured_clients(),
            detected_clients=self.registry.detected_clients(),
            installed_versions=self.runtime.installed_versions(),
        )

    def plan(
        self,
        *,
        operation: SetupOperation,
        clients: tuple[SetupClient, ...],
        version: str | None,
    ) -> ResolvedSetupPlan:
        self._recover_interrupted()
        manifest_original, manifest_mode = self._snapshot_file(self.install_manifest)
        if operation is SetupOperation.VERIFY:
            manifest = self._read_install_manifest()
            edits: tuple[ClientConfigEdit, ...] = ()
            if manifest is not None:
                launcher = Launcher(
                    command=str(manifest.executable),
                    args=("mcp", "serve", "--project-root", "."),
                )
                edits = tuple(
                    self.registry.plan(client, launcher, remove=False)
                    for client in self.registry.configured_clients(strict=True)
                )
            public = SetupPlan(
                operation=operation,
                version=manifest.active_version if manifest else None,
                runtime_action=RuntimeAction.NOT_REQUIRED,
                runtime_executable=manifest.executable if manifest else None,
                clients=tuple(
                    ClientSetupPlan(
                        client=edit.client,
                        display_name=edit.client.display_name,
                        path=edit.path,
                        action=edit.action,
                        detected=edit.detected,
                    )
                    for edit in edits
                ),
            )
            return ResolvedSetupPlan(
                public,
                edits,
                manifest_original,
                manifest_mode,
            )
        if not clients:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Select at least one MCP client.",
            )

        remove = operation is SetupOperation.REMOVE
        if remove:
            launcher = Launcher(command="", args=())
            runtime_action = RuntimeAction.NOT_REQUIRED
            executable = None
            version = None
        else:
            if version is None:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "A target runtime version is required.",
                )
            installed_versions = self.runtime.installed_versions()
            if operation is SetupOperation.ROLLBACK and version not in installed_versions:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    f"flameox {version} is not installed locally.",
                )
            executable = self.runtime.executable(version)
            launcher = Launcher(
                command=str(executable),
                args=("mcp", "serve", "--project-root", "."),
            )
            runtime_action = (
                RuntimeAction.REUSE if version in installed_versions else RuntimeAction.INSTALL
            )

        edits = tuple(
            self.registry.plan(client, launcher, remove=remove)
            for client in _unique_clients(clients)
        )
        public = SetupPlan(
            operation=operation,
            version=version,
            runtime_action=runtime_action,
            runtime_executable=executable,
            clients=tuple(
                ClientSetupPlan(
                    client=edit.client,
                    display_name=edit.client.display_name,
                    path=edit.path,
                    action=edit.action,
                    detected=edit.detected,
                )
                for edit in edits
            ),
            warnings=(
                ("Client detection is informational; only selected clients will change.",)
                if any(not edit.detected for edit in edits)
                else ()
            ),
        )
        return ResolvedSetupPlan(
            public,
            edits,
            manifest_original,
            manifest_mode,
        )

    async def apply(self, plan: ResolvedSetupPlan) -> SetupReport:
        self.data_root.mkdir(parents=True, exist_ok=True)
        try:
            lock = portalocker.Lock(self.lock_path, mode="a", timeout=10)
            with lock:
                self._recover_locked()
                return await self._apply_locked(plan)
        except portalocker.exceptions.LockException as exc:
            raise DomainError(
                ErrorCode.WRITE_LOCK_TIMEOUT,
                "Another flameox setup operation holds the setup lock.",
                retryable=True,
            ) from exc

    def _recover_interrupted(self) -> None:
        if not self.journal_path.exists():
            return
        self.data_root.mkdir(parents=True, exist_ok=True)
        try:
            lock = portalocker.Lock(self.lock_path, mode="a", timeout=10)
            with lock:
                self._recover_locked()
        except portalocker.exceptions.LockException as exc:
            raise DomainError(
                ErrorCode.WRITE_LOCK_TIMEOUT,
                "Another flameox setup operation holds the setup lock.",
                retryable=True,
            ) from exc

    async def _apply_locked(self, plan: ResolvedSetupPlan) -> SetupReport:
        public = plan.public
        self._check_preflight(plan)
        if public.operation is SetupOperation.VERIFY:
            executable = public.runtime_executable
            if executable is None:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "No active flameox runtime is configured.",
                    remediation=("Run `npx flameox@latest setup` first.",),
                )
            if public.version is None:
                raise DomainError(
                    ErrorCode.INTERNAL_ERROR,
                    "Resolved runtime has no version after installation.",
                )
            mismatched = tuple(
                edit.client
                for edit in plan.edits
                if edit.action is not ClientPlanAction.ALREADY_CURRENT
            )
            if mismatched:
                names = ", ".join(client.display_name for client in mismatched)
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"Configured MCP launchers do not match the active runtime: {names}.",
                    details={"clients": [client.value for client in mismatched]},
                    remediation=(
                        "Run `npx flameox@latest setup` and choose Connect or update MCP clients.",
                    ),
                )
            await self.runtime.verify(executable, public.version)
            return SetupReport(
                operation=public.operation,
                version=public.version,
                runtime_installed=False,
                changed_clients=(),
                unchanged_clients=tuple(edit.client for edit in plan.edits),
                verified=True,
            )

        installation: RuntimeInstallation | None = None
        if public.version is not None:
            if (
                public.operation is SetupOperation.ROLLBACK
                and public.version not in self.runtime.installed_versions()
            ):
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"Rollback runtime {public.version} is no longer installed.",
                    retryable=True,
                    remediation=("Review installed versions and run setup again.",),
                )
            installation = await self.runtime.install(public.version)

        changed = tuple(
            edit
            for edit in plan.edits
            if edit.action
            not in (ClientPlanAction.ALREADY_CURRENT, ClientPlanAction.NOT_CONFIGURED)
        )
        unchanged = tuple(edit.client for edit in plan.edits if edit not in changed)
        manifest_original = plan.install_manifest_original
        manifest_updated = (
            manifest_original
            if public.operation is SetupOperation.REMOVE
            else self._updated_install_manifest(public)
        )
        mutations = [
            _FileMutation(edit.path, edit.original, edit.updated, edit.mode)
            for edit in changed
            if edit.updated is not None
        ]
        if (
            manifest_updated is not None
            and manifest_updated != manifest_original
            and (manifest_original is not None or public.operation is not SetupOperation.REMOVE)
        ):
            mutations.append(
                _FileMutation(
                    self.install_manifest,
                    manifest_original,
                    manifest_updated,
                    plan.install_manifest_mode,
                )
            )
        if not mutations:
            return SetupReport(
                operation=public.operation,
                version=public.version,
                runtime_installed=installation.installed if installation else False,
                changed_clients=(),
                unchanged_clients=unchanged,
                verified=installation is not None,
            )

        self._write_journal(mutations)
        try:
            for mutation in mutations:
                current = mutation.path.read_bytes() if mutation.path.exists() else None
                if current != mutation.original:
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        f"Configuration changed during setup: {mutation.path}",
                        retryable=True,
                        remediation=("Review the file and run setup again.",),
                    )
                atomic_write_bytes(
                    mutation.path,
                    mutation.updated,
                    mode=mutation.mode,
                )
        except BaseException:
            self._restore_mutations(mutations)
            self._remove_journal()
            raise
        self._remove_journal()
        return SetupReport(
            operation=public.operation,
            version=public.version,
            runtime_installed=installation.installed if installation else False,
            changed_clients=tuple(edit.client for edit in changed),
            unchanged_clients=unchanged,
            verified=installation is not None,
        )

    def _updated_install_manifest(
        self,
        plan: SetupPlan,
    ) -> bytes:
        if plan.version is None or plan.runtime_executable is None:
            return b""
        manifest = _InstallManifest(
            active_version=plan.version,
            executable=plan.runtime_executable,
        )
        return (
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        ).encode()

    def _check_preflight(self, plan: ResolvedSetupPlan) -> None:
        for edit in plan.edits:
            current = edit.path.read_bytes() if edit.path.exists() else None
            current_mode = stat.S_IMODE(edit.path.stat().st_mode) if edit.path.exists() else 0o600
            if current != edit.original or current_mode != edit.mode:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"Client configuration changed after the setup preview: {edit.path}",
                    retryable=True,
                    remediation=("Review the new configuration and run setup again.",),
                )
        manifest, manifest_mode = self._snapshot_file(self.install_manifest)
        if (
            manifest != plan.install_manifest_original
            or manifest_mode != plan.install_manifest_mode
        ):
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "flameox setup metadata changed after the setup preview.",
                retryable=True,
                remediation=("Review the active runtime and run setup again.",),
            )

    @staticmethod
    def _snapshot_file(path: Path) -> tuple[bytes | None, int]:
        try:
            metadata = path.stat()
            return path.read_bytes(), stat.S_IMODE(metadata.st_mode)
        except FileNotFoundError:
            return None, 0o600
        except OSError as exc:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Could not inspect setup state: {path}",
                details={"error": str(exc)},
            ) from exc

    def _write_journal(self, edits: list[_FileMutation]) -> None:
        journal = _SetupJournal(
            mutations=tuple(
                _JournalMutation(
                    path=edit.path,
                    original=(
                        base64.b64encode(edit.original).decode()
                        if edit.original is not None
                        else None
                    ),
                    original_sha256=_digest(edit.original),
                    updated_sha256=hashlib.sha256(edit.updated).hexdigest(),
                    mode=edit.mode,
                )
                for edit in edits
            )
        )
        atomic_write_json(self.journal_path, journal.model_dump(mode="json"))

    def _recover_locked(self) -> None:
        if not self.journal_path.exists():
            return
        try:
            journal = _SetupJournal.model_validate_json(self.journal_path.read_text())
            allowed_paths = {
                *self.registry.allowed_config_paths(),
                self.install_manifest,
            }
            decoded: dict[Path, bytes | None] = {}
            for mutation in journal.mutations:
                path = mutation.path
                if path not in allowed_paths:
                    raise ValueError(f"journal contains an unexpected path: {path}")
                original = (
                    base64.b64decode(mutation.original, validate=True)
                    if mutation.original is not None
                    else None
                )
                if _digest(original) != mutation.original_sha256:
                    raise ValueError(f"journal digest mismatch: {path}")
                decoded[path] = original
            for mutation in reversed(journal.mutations):
                path = mutation.path
                current = path.read_bytes() if path.exists() else None
                original_digest = mutation.original_sha256
                current_digest = _digest(current)
                if current_digest == original_digest:
                    continue
                if current_digest != mutation.updated_sha256:
                    raise ValueError(f"configuration changed after interruption: {path}")
                original = decoded[path]
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    atomic_write_bytes(path, original, mode=mutation.mode)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "The interrupted setup journal could not be recovered safely.",
                details={"path": str(self.journal_path), "error": str(exc)},
                remediation=("Preserve the journal and repair the affected client configs.",),
            ) from exc
        self._remove_journal()

    @staticmethod
    def _restore_mutations(edits: list[_FileMutation]) -> None:
        conflicts: list[Path] = []
        for edit in reversed(edits):
            current = edit.path.read_bytes() if edit.path.exists() else None
            if current == edit.original:
                continue
            if current != edit.updated:
                conflicts.append(edit.path)
                continue
            if edit.original is None:
                edit.path.unlink(missing_ok=True)
            else:
                atomic_write_bytes(edit.path, edit.original, mode=edit.mode)
        if conflicts:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "One or more configs changed during setup rollback.",
                details={"paths": [str(path) for path in conflicts]},
                remediation=("Preserve the setup journal and review the files manually.",),
            )

    def _remove_journal(self) -> None:
        self.journal_path.unlink(missing_ok=True)
        fsync_directory(self.journal_path.parent)

    def _read_install_manifest(self) -> _InstallManifest | None:
        if not self.install_manifest.exists():
            return None
        try:
            manifest = _InstallManifest.model_validate_json(self.install_manifest.read_text())
            expected = self.runtime.executable(manifest.active_version)
            if manifest.executable != expected:
                raise ValueError("managed runtime path does not match its version")
        except DomainError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                f"flameox setup metadata is invalid: {self.install_manifest}",
                details={"error": exc.message},
            ) from exc
        except (OSError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                f"flameox setup metadata is invalid: {self.install_manifest}",
                details={"error": str(exc)},
            ) from exc
        return manifest


def _unique_clients(clients: tuple[SetupClient, ...]) -> tuple[SetupClient, ...]:
    selected = set(clients)
    return tuple(client for client in ALL_SETUP_CLIENTS if client in selected)


def _digest(value: bytes | None) -> str | None:
    return hashlib.sha256(value).hexdigest() if value is not None else None
