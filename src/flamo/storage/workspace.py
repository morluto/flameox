from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import portalocker
from pydantic import BaseModel, ConfigDict, Field

from flamo import __version__
from flamo.config import WorkspaceConfig
from flamo.domain.errors import DomainError, ErrorCode
from flamo.domain.models import utc_now
from flamo.storage.atomic import atomic_write_json, atomic_write_text
from flamo.storage.corpus import CorpusStore


class WorkspaceIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    workspace_id: str = Field(min_length=1)
    created_at: datetime
    project_root: str
    flamo_version: str


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
            return WorkspaceConfig.from_path(self.paths.config)
        except (FileNotFoundError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Invalid workspace configuration at {self.paths.config}.",
            ) from exc

    @classmethod
    def initialize(
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
                _ = workspace.identity
            else:
                relative_project_root = os.path.relpath(project_root, root)
                identity = WorkspaceIdentity(
                    workspace_id=str(uuid4()),
                    created_at=utc_now(),
                    project_root=relative_project_root,
                    flamo_version=__version__,
                )
                atomic_write_json(
                    workspace.paths.identity,
                    identity.model_dump(mode="json"),
                )
            if not workspace.paths.config.exists():
                atomic_write_text(workspace.paths.config, WorkspaceConfig().to_toml())
            _ = workspace.config
            workspace.corpus.initialize()

        if root == project_root / ".diagnostics":
            cls._exclude_local_workspace(project_root)
        if not workspace.paths.catalog.exists():
            from flamo.catalog import Catalog

            Catalog(workspace).rebuild()
        return workspace

    @classmethod
    def discover(cls, start: Path, *, explicit: Path | None = None) -> Workspace:
        if explicit is not None:
            workspace = cls(explicit)
            _ = workspace.identity
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
            remediation=("Run `flamo init` from the project root.",),
        )

    def write_locked(
        self,
        *,
        timeout: float = 30,
    ) -> portalocker.Lock:
        return portalocker.Lock(
            self.paths.write_lock,
            mode="a",
            timeout=timeout,
        )

    def retention_locked(
        self,
        *,
        shared: bool,
        timeout: float = 30,
    ) -> portalocker.Lock:
        flag = portalocker.LOCK_SH if shared else portalocker.LOCK_EX
        return portalocker.Lock(
            self.paths.retention_lock,
            mode="a",
            timeout=timeout,
            flags=flag | portalocker.LOCK_NB,
        )

    def catalog_locked(
        self,
        *,
        shared: bool,
        timeout: float = 30,
    ) -> portalocker.Lock:
        flag = portalocker.LOCK_SH if shared else portalocker.LOCK_EX
        return portalocker.Lock(
            self.paths.catalog_lock,
            mode="a",
            timeout=timeout,
            flags=flag | portalocker.LOCK_NB,
        )

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
