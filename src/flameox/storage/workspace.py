from __future__ import annotations

import errno
import json
import logging
import os
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import portalocker
import tomlkit
from pydantic import Field

from flameox import __version__
from flameox.atomic import atomic_write_json, atomic_write_text
from flameox.config import WorkspaceConfig
from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.models import utc_now
from flameox.models import ContractModel
from flameox.storage.corpus import CorpusStore

logger = logging.getLogger(__name__)
_REMOVED_EXECUTION_SETTING = "allow_mcp_ad_hoc_commands"
_READ_ONLY_ERRNOS = {errno.EACCES, errno.EPERM, errno.EROFS}


def _has_removed_execution_setting(payload: dict[str, object]) -> bool:
    execution = payload.get("execution")
    return isinstance(execution, dict) and _REMOVED_EXECUTION_SETTING in execution


def _migrated_config_payload(payload: dict[str, object]) -> dict[str, object]:
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        return payload
    return {
        **payload,
        "execution": {
            key: value for key, value in execution.items() if key != _REMOVED_EXECUTION_SETTING
        },
    }


class WorkspaceIdentity(ContractModel):
    schema_version: Literal[1] = 1
    workspace_id: str = Field(min_length=1)
    created_at: datetime
    project_root: str
    flameox_version: str


def _workspace_initialization_error(error: OSError) -> DomainError:
    details: dict[str, int | str] = {"operation": "workspace_initialization"}
    if error.errno is not None:
        details["errno"] = error.errno

    quota_errors = {errno.ENOSPC}
    if hasattr(errno, "EDQUOT"):
        quota_errors.add(errno.EDQUOT)
    if error.errno in quota_errors:
        return DomainError(
            ErrorCode.STORAGE_QUOTA_EXCEEDED,
            "Workspace initialization needs more local storage.",
            details=details,
            remediation=(
                "Free local storage or increase the filesystem quota, then retry initialization.",
            ),
        )
    if error.errno in {errno.EACCES, errno.EPERM, errno.EROFS}:
        return DomainError(
            ErrorCode.WORKSPACE_INVALID,
            "The selected project root is not writable for workspace initialization.",
            details=details,
            remediation=(
                "Choose a writable project root or repair its permissions, then retry "
                "initialization.",
            ),
        )
    return DomainError(
        ErrorCode.INTERNAL_ERROR,
        "Workspace initialization failed because the local filesystem rejected a write.",
        details=details,
        remediation=("Check filesystem health and permissions, then retry initialization.",),
    )


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    root: Path

    @property
    def identity(self) -> Path:
        return self.root / "workspace.json"

    @property
    def config(self) -> Path:
        return self.root / "config.toml"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts" / "sha256"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def records(self) -> Path:
        return self.root / "records"

    @property
    def evidence(self) -> Path:
        return self.root / "evidence"

    @property
    def generations(self) -> Path:
        return self.root / "generations"

    @property
    def staging(self) -> Path:
        return self.root / "staging"

    @property
    def quarantine(self) -> Path:
        return self.root / "quarantine"

    @property
    def trash(self) -> Path:
        return self.root / "trash"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def operation_log(self) -> Path:
        return self.logs / "operations.jsonl"

    @property
    def operation_log_lock(self) -> Path:
        return self.logs / "operations.lock"

    @property
    def write_lock(self) -> Path:
        return self.root / "write.lock"

    @property
    def catalog_lock(self) -> Path:
        return self.root / "catalog.lock"

    @property
    def retention_lock(self) -> Path:
        return self.root / "retention.lock"

    @property
    def catalog(self) -> Path:
        return self.root / "catalog.duckdb"


class Workspace:
    DIRECTORY_NAMES = (
        "artifacts/sha256",
        "corpus/commits",
        "evidence",
        "generations",
        "logs",
        "quarantine",
        "records",
        "runs",
        "staging",
        "trash",
    )

    def __init__(self, root: Path) -> None:
        self.paths = WorkspacePaths(root.resolve())
        self.corpus = CorpusStore(self.paths.root)

    @property
    def project_root(self) -> Path:
        return (self.paths.root / self.identity.project_root).resolve()

    @property
    def identity(self) -> WorkspaceIdentity:
        try:
            payload = json.loads(self.paths.identity.read_text())
            return WorkspaceIdentity.model_validate(payload)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Invalid workspace identity at {self.paths.identity}.",
            ) from exc

    @property
    def config(self) -> WorkspaceConfig:
        try:
            return self._load_config()
        except (FileNotFoundError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Invalid workspace configuration at {self.paths.config}.",
            ) from exc

    def _load_config(self) -> WorkspaceConfig:
        payload = self._read_config_payload()
        if not _has_removed_execution_setting(payload):
            return WorkspaceConfig.model_validate(payload)
        try:
            with self.write_locked():
                return self._load_config_locked()
        except OSError as exc:
            if exc.errno not in _READ_ONLY_ERRNOS:
                raise
            logger.warning(
                "Could not persist removal of deprecated execution.%s from %s; "
                "using the validated migrated configuration without writing.",
                _REMOVED_EXECUTION_SETTING,
                self.paths.config,
            )
            return WorkspaceConfig.model_validate(_migrated_config_payload(payload))

    def _load_config_locked(self) -> WorkspaceConfig:
        payload = self._read_config_payload()
        execution = payload.get("execution")
        if not isinstance(execution, dict) or _REMOVED_EXECUTION_SETTING not in execution:
            return WorkspaceConfig.model_validate(payload)

        migrated_payload = _migrated_config_payload(payload)
        config = WorkspaceConfig.model_validate(migrated_payload)
        document = tomlkit.parse(self.paths.config.read_text(encoding="utf-8"))
        document_execution = document.get("execution")
        if document_execution is None or _REMOVED_EXECUTION_SETTING not in document_execution:
            return config
        del document_execution[_REMOVED_EXECUTION_SETTING]
        mode = self.paths.config.stat().st_mode & 0o777
        atomic_write_text(
            self.paths.config,
            tomlkit.dumps(document),
            mode=mode,
        )
        logger.warning(
            "Removed deprecated execution.%s from %s. "
            "MCP ad-hoc commands are no longer supported; declare a named workload "
            "before planning a capture.",
            _REMOVED_EXECUTION_SETTING,
            self.paths.config,
        )
        return config

    def _read_config_payload(self) -> dict[str, object]:
        with self.paths.config.open("rb") as stream:
            payload = tomllib.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("Workspace configuration must be a TOML table.")
        return payload

    @classmethod
    def initialize(
        cls,
        project_root: Path,
        *,
        workspace_root: Path | None = None,
    ) -> Workspace:
        try:
            return cls._initialize(project_root, workspace_root=workspace_root)
        except OSError as error:
            raise _workspace_initialization_error(error) from error

    @classmethod
    def _initialize(
        cls,
        project_root: Path,
        *,
        workspace_root: Path | None = None,
    ) -> Workspace:
        project_root = project_root.resolve()
        root = (workspace_root or project_root / ".diagnostics").resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with suppress(OSError):
            os.chmod(root, 0o700)
        workspace = cls(root)
        for directory in cls.DIRECTORY_NAMES:
            (root / directory).mkdir(parents=True, exist_ok=True)
        for lock_path in (
            workspace.paths.write_lock,
            workspace.paths.catalog_lock,
            workspace.paths.retention_lock,
            workspace.paths.operation_log_lock,
        ):
            lock_path.touch(mode=0o600, exist_ok=True)

        with workspace.write_locked():
            if workspace.paths.identity.exists():
                cls._validate_project_binding(workspace, project_root)
            else:
                relative_project_root = os.path.relpath(project_root, root)
                identity = WorkspaceIdentity(
                    workspace_id=str(uuid4()),
                    created_at=utc_now(),
                    project_root=relative_project_root,
                    flameox_version=__version__,
                )
                atomic_write_json(
                    workspace.paths.identity,
                    identity.model_dump(mode="json"),
                )
            if not workspace.paths.config.exists():
                atomic_write_text(workspace.paths.config, WorkspaceConfig().to_toml())
            _ = workspace._load_config_locked()
            workspace.corpus.initialize()

        if root == project_root / ".diagnostics":
            cls._exclude_local_workspace(project_root)
        if not workspace.paths.catalog.exists():
            from flameox.catalog import Catalog

            Catalog(workspace).rebuild()
        return workspace

    @classmethod
    def discover(
        cls,
        start: Path,
        *,
        explicit: Path | None = None,
        project_root: Path | None = None,
    ) -> Workspace:
        if explicit is not None:
            workspace = cls(explicit)
            if project_root is not None:
                cls._validate_project_binding(workspace, project_root)
            return workspace
        current = start.resolve()
        for directory in (current, *current.parents):
            candidate = directory / ".diagnostics"
            if candidate.is_dir():
                workspace = cls(candidate)
                _ = workspace.identity
                return workspace
        raise DomainError(
            ErrorCode.WORKSPACE_NOT_FOUND,
            "No .diagnostics workspace was found.",
            remediation=("Run `flameox init` from the project root.",),
        )

    @staticmethod
    def _validate_project_binding(workspace: Workspace, project_root: Path) -> None:
        expected = project_root.resolve()
        actual = workspace.project_root
        if actual == expected:
            return
        raise DomainError(
            ErrorCode.WORKSPACE_INVALID,
            "The selected workspace is bound to a different project root.",
            details={
                "workspace_root": str(workspace.paths.root),
                "bound_project_root": str(actual),
                "requested_project_root": str(expected),
            },
            remediation=(
                "Select the workspace that belongs to this project, or initialize a new "
                "explicit workspace root for it.",
            ),
        )

    @contextmanager
    def write_locked(
        self,
        *,
        timeout: float = 30,
    ) -> Iterator[object]:
        yield from self._locked(self.paths.write_lock, timeout=timeout)

    @contextmanager
    def retention_locked(
        self,
        *,
        shared: bool,
        timeout: float = 30,
    ) -> Iterator[object]:
        flag = portalocker.LOCK_SH if shared else portalocker.LOCK_EX
        yield from self._locked(
            self.paths.retention_lock,
            timeout=timeout,
            flags=flag | portalocker.LOCK_NB,
        )

    @contextmanager
    def catalog_locked(
        self,
        *,
        shared: bool,
        timeout: float = 30,
    ) -> Iterator[object]:
        flag = portalocker.LOCK_SH if shared else portalocker.LOCK_EX
        yield from self._locked(
            self.paths.catalog_lock,
            timeout=timeout,
            flags=flag | portalocker.LOCK_NB,
        )

    def _locked(
        self,
        path: Path,
        *,
        timeout: float,
        flags: portalocker.LockFlags | None = None,
    ) -> Iterator[object]:
        lock = (
            portalocker.Lock(path, mode="a", timeout=timeout)
            if flags is None
            else portalocker.Lock(
                path,
                mode="a",
                timeout=timeout,
                flags=flags,
            )
        )
        try:
            with lock as stream:
                yield stream
        except portalocker.exceptions.LockException as exc:
            raise DomainError(
                ErrorCode.WRITE_LOCK_TIMEOUT,
                f"Timed out waiting for workspace lock {path.name!r}.",
                retryable=True,
                details={"lock": path.name, "timeout_seconds": timeout},
            ) from exc

    @staticmethod
    def _exclude_local_workspace(project_root: Path) -> None:
        git_directory = project_root / ".git"
        if not git_directory.is_dir():
            return
        exclude = git_directory / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude.read_text() if exclude.exists() else ""
        lines = {line.strip() for line in existing.splitlines()}
        if ".diagnostics/" in lines:
            return
        prefix = existing
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        atomic_write_text(exclude, f"{prefix}.diagnostics/\n", mode=0o644)
