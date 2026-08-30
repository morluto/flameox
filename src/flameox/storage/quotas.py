from __future__ import annotations

import shutil
import stat
from pathlib import Path

from flameox.domain import DomainError, ErrorCode
from flameox.storage.workspace import Workspace


class StorageQuota:
    """Checks configured local storage budgets at mutation boundaries."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def require_capacity(
        self,
        *,
        additional_bytes: int = 0,
        staging: bool = False,
    ) -> None:
        config = self.workspace.config.storage
        # ``trash`` and ``quarantine`` hold data that has already been
        # logically deleted (or isolated) and is pending physical purge by
        # retention. Counting them against the live workspace quota would
        # refuse new writes after a deletion until retention catches up.
        workspace_bytes = (
            tree_bytes(self.workspace.paths.root)
            - tree_bytes(self.workspace.paths.trash)
            - tree_bytes(self.workspace.paths.quarantine)
        )
        if workspace_bytes + additional_bytes > config.max_workspace_bytes:
            raise DomainError(
                ErrorCode.STORAGE_QUOTA_EXCEEDED,
                "The operation would exceed the configured workspace storage quota.",
                details={
                    "current_bytes": workspace_bytes,
                    "additional_bytes": additional_bytes,
                    "limit_bytes": config.max_workspace_bytes,
                },
            )
        if staging:
            staging_bytes = tree_bytes(self.workspace.paths.staging)
            if staging_bytes + additional_bytes > config.max_staging_bytes:
                raise DomainError(
                    ErrorCode.STORAGE_QUOTA_EXCEEDED,
                    "The operation would exceed the configured staging storage quota.",
                    details={
                        "current_bytes": staging_bytes,
                        "additional_bytes": additional_bytes,
                        "limit_bytes": config.max_staging_bytes,
                    },
                )
        free_bytes = shutil.disk_usage(self.workspace.paths.root).free
        if free_bytes - additional_bytes < config.min_free_bytes:
            raise DomainError(
                ErrorCode.STORAGE_QUOTA_EXCEEDED,
                "The operation would violate the configured minimum free-space reserve.",
                details={
                    "free_bytes": free_bytes,
                    "additional_bytes": additional_bytes,
                    "minimum_free_bytes": config.min_free_bytes,
                },
            )

    def require_generation_row_count(self, row_count: int) -> None:
        limit = self.workspace.config.storage.max_rows_per_generation
        if row_count > limit:
            raise DomainError(
                ErrorCode.STORAGE_QUOTA_EXCEEDED,
                "The generation exceeds the configured evidence-row quota.",
                details={"row_count": row_count, "limit": limit},
            )


def tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(metadata.st_mode):
            total += metadata.st_size
    return total
