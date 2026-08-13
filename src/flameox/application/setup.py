from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

import portalocker
from packaging.version import InvalidVersion, Version
from platformdirs import user_data_path
from pydantic import ConfigDict, computed_field

from flameox.adapters.client_setup import (
    ALL_SETUP_CLIENTS,
    ClientConfigEdit,
    ClientPlanAction,
    Launcher,
    QualifiedClientConfigFallbacks,
    SetupClient,
)
from flameox.adapters.mcp_client_drivers import (
    ClientCommandPlan,
    ClientManagementMechanism,
    OfficialCliDriver,
)
from flameox.adapters.setup_runtime import ManagedRuntime, RuntimeInstallation
from flameox.atomic import atomic_write_bytes
from flameox.domain import DomainError, ErrorCode
from flameox.execution import SubprocessBroker
from flameox.models import ContractModel


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
    mechanism: ClientManagementMechanism = ClientManagementMechanism.QUALIFIED_CONFIG_FILE
    client_version: str | None = None


class SetupPlan(ContractModel):
    schema_version: Literal[1] = 1
    operation: SetupOperation
    version: str | None
    runtime_action: RuntimeAction
    runtime_executable: Path | None
    clients: tuple[ClientSetupPlan, ...]
    warnings: tuple[str, ...] = ()


class SetupReport(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    schema_version: Literal[1] = 1
    operation: SetupOperation
    version: str | None
    runtime_installed: bool
    changed_clients: tuple[SetupClient, ...]
    unchanged_clients: tuple[SetupClient, ...]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def verified(self) -> bool:
        return self.operation is SetupOperation.VERIFY or self.version is not None


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
    commands: tuple[ClientCommandPlan, ...] = ()


class _InstallManifest(ContractModel):
    schema_version: Literal[1] = 1
    active_version: str
    executable: Path


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
        broker: SubprocessBroker | None = None,
        prefer_official_clients: bool | None = None,
    ) -> None:
        resolved_home = home or Path.home()
        self.data_root = data_root or Path(user_data_path("flameox", appauthor=False))
        self.broker = broker or SubprocessBroker()
        self.registry = QualifiedClientConfigFallbacks(
            home=resolved_home,
            jsonc_helper=jsonc_helper,
            node_executable=node_executable,
            broker=self.broker,
        )
        use_official = home is None if prefer_official_clients is None else prefer_official_clients
        self.official_drivers = (
            {
                client: OfficialCliDriver(client, broker=self.broker)
                for client in (SetupClient.CLAUDE, SetupClient.CODEX, SetupClient.GEMINI)
            }
            if use_official
            else {}
        )
        self.runtime = runtime or ManagedRuntime(
            self.data_root,
            uv_executable=uv_executable,
            broker=self.broker,
        )
        self.install_manifest = self.data_root / "install.json"
        self.lock_path = self.data_root / "setup.lock"

    def inspect(self) -> SetupInspection:
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

        warnings: list[str] = []
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
            active_manifest = self._read_install_manifest()
            if (
                operation is SetupOperation.CONFIGURE
                and active_manifest is not None
                and _is_older_version(version, active_manifest.active_version)
            ):
                warnings.append(
                    f"Requested flameox {version} is older than the active runtime "
                    f"{active_manifest.active_version}; keeping the active runtime."
                )
                version = active_manifest.active_version
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

        planned_edits: list[ClientConfigEdit] = []
        commands: list[ClientCommandPlan] = []
        for client in _unique_clients(clients):
            driver = self.official_drivers.get(client)
            if driver is not None:
                commands.append(driver.plan(launcher, remove=remove))
            else:
                planned_edits.append(self.registry.plan(client, launcher, remove=remove))
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
                for edit in planned_edits
            )
            + tuple(
                ClientSetupPlan(
                    client=command.client,
                    display_name=command.client.display_name,
                    path=command.executable.canonical_target,
                    action=command.action,
                    detected=True,
                    mechanism=command.mechanism,
                    client_version=command.client_version,
                )
                for command in commands
            ),
            warnings=tuple(warnings)
            + (
                ("Client detection is informational; only selected clients will change.",)
                if any(not edit.detected for edit in planned_edits)
                else ()
            ),
        )
        return ResolvedSetupPlan(
            public,
            tuple(planned_edits),
            manifest_original,
            manifest_mode,
            tuple(commands),
        )

    async def apply(self, plan: ResolvedSetupPlan) -> SetupReport:
        self.data_root.mkdir(parents=True, exist_ok=True)
        try:
            lock = portalocker.Lock(self.lock_path, mode="a", timeout=10)
            with lock:
                return await self._apply_locked(plan)
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
                    remediation=(
                        "Review installed versions, then run `npx flameox@latest setup` again.",
                    ),
                )
            installation = await self.runtime.install(public.version)

        changed = tuple(
            edit
            for edit in plan.edits
            if edit.action
            not in (ClientPlanAction.ALREADY_CURRENT, ClientPlanAction.NOT_CONFIGURED)
        )
        unchanged = tuple(edit.client for edit in plan.edits if edit not in changed)
        command_changed = tuple(
            command
            for command in plan.commands
            if command.action
            not in (ClientPlanAction.ALREADY_CURRENT, ClientPlanAction.NOT_CONFIGURED)
        )
        command_unchanged = tuple(
            command.client for command in plan.commands if command not in command_changed
        )
        manifest_original = plan.install_manifest_original
        manifest_updated = (
            manifest_original
            if public.operation is SetupOperation.REMOVE
            else self._updated_install_manifest(public)
        )
        if not changed and not command_changed and manifest_updated == manifest_original:
            return SetupReport(
                operation=public.operation,
                version=public.version,
                runtime_installed=installation.installed if installation else False,
                changed_clients=(),
                unchanged_clients=unchanged + command_unchanged,
            )

        # Client applications are independent authorities. Apply and verify each
        # selected mutation on its own; never fabricate a distributed rollback.
        for edit in changed:
            current = edit.path.read_bytes() if edit.path.exists() else None
            if current != edit.original:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"Configuration changed during setup: {edit.path}",
                    retryable=True,
                    remediation=("Review the file, then run `npx flameox@latest setup` again.",),
                )
            if edit.updated is not None:
                atomic_write_bytes(edit.path, edit.updated, mode=edit.mode)
        for command in command_changed:
            driver = self.official_drivers.get(command.client)
            if driver is None:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"The planned {command.client.display_name} driver is no longer available.",
                    retryable=True,
                )
            await driver.apply(command)
        if (
            manifest_updated is not None
            and manifest_updated != manifest_original
            and (manifest_original is not None or public.operation is not SetupOperation.REMOVE)
        ):
            atomic_write_bytes(
                self.install_manifest,
                manifest_updated,
                mode=plan.install_manifest_mode,
            )
        return SetupReport(
            operation=public.operation,
            version=public.version,
            runtime_installed=installation.installed if installation else False,
            changed_clients=tuple(edit.client for edit in changed)
            + tuple(command.client for command in command_changed),
            unchanged_clients=unchanged + command_unchanged,
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
                    remediation=(
                        "Review the new configuration, then run `npx flameox@latest setup` again.",
                    ),
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
                remediation=(
                    "Review the active runtime, then run `npx flameox@latest setup` again.",
                ),
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


def _is_older_version(candidate: str, current: str) -> bool:
    try:
        return Version(candidate) < Version(current)
    except InvalidVersion:
        return False
