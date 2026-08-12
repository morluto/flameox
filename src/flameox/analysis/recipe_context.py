from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from flameox.catalog import Catalog, Snapshot, SnapshotHandle
from flameox.domain import DomainError, ErrorCode
from flameox.storage import Workspace


class RecipeContext:
    """Shared snapshot and query-budget policy for analysis recipe families."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        snapshot: Snapshot | None = None,
        snapshot_handle: SnapshotHandle | None = None,
    ) -> None:
        if snapshot is not None and snapshot_handle is not None:
            raise ValueError("provide a snapshot or snapshot handle, not both")
        self.workspace = workspace
        self.snapshot = snapshot
        self.snapshot_handle = snapshot.handle if snapshot is not None else snapshot_handle

    def _limit(self, value: int | None) -> int:
        if value is None:
            return self.workspace.config.analysis.default_row_limit
        if value < 1 or value > self.workspace.config.analysis.max_row_limit:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Limit must be between 1 and {self.workspace.config.analysis.max_row_limit}.",
            )
        return value

    def _pinned_commit_id(self, value: str | None) -> str:
        if value is not None:
            if self.snapshot_handle is not None and self.snapshot_handle.commit.commit_id != value:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "Recipe snapshot does not match the requested corpus commit.",
                )
            return value
        if self.snapshot_handle is not None:
            return self.snapshot_handle.commit.commit_id
        return Catalog(self.workspace).pin().commit.commit_id

    @contextmanager
    def _open_snapshot(self, corpus_commit_id: str) -> Iterator[Snapshot]:
        if self.snapshot is not None:
            if self.snapshot.commit.commit_id != corpus_commit_id:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "Recipe attempted to cross its pinned corpus snapshot.",
                )
            yield self.snapshot
            return
        catalog = Catalog(self.workspace)
        if (
            self.snapshot_handle is not None
            and self.snapshot_handle.commit.commit_id != corpus_commit_id
        ):
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Recipe attempted to cross its pinned corpus snapshot.",
            )
        handle = self.snapshot_handle or catalog.pin(corpus_commit_id)
        with catalog.open_snapshot(handle) as snapshot:
            yield snapshot
