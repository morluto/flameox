from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from flameox.catalog import Catalog, Snapshot
from flameox.domain import DomainError, ErrorCode
from flameox.storage import Workspace


class RecipeContext:
    """Shared snapshot and query-budget policy for analysis recipe families."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        snapshot: Snapshot | None = None,
    ) -> None:
        self.workspace = workspace
        self.snapshot = snapshot

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
            if self.snapshot is not None and self.snapshot.commit.commit_id != value:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "Recipe snapshot does not match the requested corpus commit.",
                )
            return value
        if self.snapshot is not None:
            return self.snapshot.commit.commit_id
        return self.workspace.corpus.read_head().commit_id

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
        with Catalog(self.workspace).open_snapshot(corpus_commit_id) as snapshot:
            yield snapshot
