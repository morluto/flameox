from __future__ import annotations

import errno
import json
import os
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import Field

from flameox.atomic import atomic_write_json, atomic_write_text
from flameox.config import WorkspaceConfig
from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.models import utc_now
from flameox.models import ContractModel
from flameox.storage.corpus import CorpusStore
from flameox.storage.locks import (
    CATALOG_EXCLUSIVE,
    CATALOG_SHARED,
    RETENTION_EXCLUSIVE,
    RETENTION_SHARED,
    WRITE_EXCLUSIVE,
    WorkspaceLockIntent,
    WorkspaceLockManager,
    WorkspaceLockResource,
)

if TYPE_CHECKING:
    from flameox.storage.cursors import CursorStore


class WorkspaceIdentity(ContractModel):
    workspace_id: str = Field(min_length=1)
    created_at: datetime
    project_root: str


def _workspace_initialization_error(error: OSError | ValueError) -> DomainError:
    details: dict[str, int | str] = {"operation": "workspace_initialization"}
    if isinstance(error, ValueError):
        return DomainError(
            ErrorCode.WORKSPACE_INVALID,
            "The project root and workspace root must share a filesystem path root.",
            details=details,
            remediation=(
                "Choose a workspace root on the same filesystem path root as the project, "
                "then retry initialization.",
            ),
        )
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

    @property
    def control_plane(self) -> Path:
        return self.root / "control-plane.sqlite3"


class Workspace:
    DIRECTORY_NAMES = (
        "artifacts/sha256",
        "corpus/commits",
        "evidence",
        "generations",
        "logs",
        "quarantine",
        "records",
        "staging",
        "trash",
    )

    def __init__(self, root: Path) -> None:
        self.paths = WorkspacePaths(root.resolve())
        self.corpus = CorpusStore(self.paths.root)
        self.lock_manager = WorkspaceLockManager(
            self.paths.root,
            {
                WorkspaceLockResource.WRITE: self.paths.write_lock,
                WorkspaceLockResource.RETENTION: self.paths.retention_lock,
                WorkspaceLockResource.CATALOG: self.paths.catalog_lock,
            },
        )

    @property
    def project_root(self) -> Path:
        return (self.paths.root / self.identity.project_root).resolve()

    @cached_property
    def cursors(self) -> CursorStore:
        from flameox.storage.cursors import CursorStore

        return CursorStore(self)

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
        return WorkspaceConfig.model_validate(self._read_config_payload())

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
        except (OSError, ValueError) as error:
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
                )
                atomic_write_json(
                    workspace.paths.identity,
                    identity.model_dump(mode="json"),
                )
            if not workspace.paths.config.exists():
                atomic_write_text(workspace.paths.config, WorkspaceConfig().to_toml())
            _ = workspace._load_config()
            workspace.corpus.initialize()
            from flameox.storage.control_plane import ControlPlane

            ControlPlane(workspace).initialize()

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
            workspace._validate_open_format()
            if project_root is not None:
                cls._validate_project_binding(workspace, project_root)
            return workspace
        current = start.resolve()
        for directory in (current, *current.parents):
            candidate = directory / ".diagnostics"
            if candidate.is_dir():
                workspace = cls(candidate)
                workspace._validate_open_format()
                return workspace
        raise DomainError(
            ErrorCode.WORKSPACE_NOT_FOUND,
            "No .diagnostics workspace was found.",
            remediation=("Run `flameox init` from the project root.",),
        )

    def _validate_open_format(self) -> None:
        _ = self.identity
        from flameox.storage.control_plane import ControlPlane

        ControlPlane(self).validate_existing()

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
        with self.locked(
            WRITE_EXCLUSIVE,
            timeout=timeout,
            phase="workspace write",
        ) as streams:
            yield streams[0]

    @contextmanager
    def retention_locked(
        self,
        *,
        shared: bool,
        timeout: float = 30,
    ) -> Iterator[object]:
        with self.locked(
            RETENTION_SHARED if shared else RETENTION_EXCLUSIVE,
            timeout=timeout,
            phase="workspace retention",
        ) as streams:
            yield streams[0]

    @contextmanager
    def catalog_locked(
        self,
        *,
        shared: bool,
        timeout: float = 30,
    ) -> Iterator[object]:
        with self.locked(
            CATALOG_SHARED if shared else CATALOG_EXCLUSIVE,
            timeout=timeout,
            phase="workspace catalog",
        ) as streams:
            yield streams[0]

    @contextmanager
    def locked(
        self,
        *intents: WorkspaceLockIntent,
        timeout: float = 30,
        phase: str = "workspace mutation",
    ) -> Iterator[tuple[object, ...]]:
        with self.lock_manager.acquire(*intents, timeout=timeout, phase=phase) as streams:
            yield streams

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
