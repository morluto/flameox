from __future__ import annotations

import json
from typing import cast

from pydantic import Field

from flameox.adapters.triton_autotune import TritonAutotuneCandidate
from flameox.catalog import Catalog
from flameox.domain import CursorNamespace, DomainError, ErrorCode, digest_model
from flameox.evidence_status import (
    EvidenceAvailability,
    available_availability,
    empty_availability,
    unavailable_availability,
)
from flameox.models import ContractModel
from flameox.pagination import CursorPageContract
from flameox.storage import RunStore, Workspace

_MAX_SELECTION_PAGE_SIZE = 64


class TritonAutotuneSelectionView(ContractModel):
    selection_id: str
    run_id: str
    function_name: str
    key_digest: str
    cache_hit: bool
    duration_ms: float | None
    winner_config_id: str
    candidate_count: int
    candidates_truncated: bool
    candidates: tuple[TritonAutotuneCandidate, ...] = Field(max_length=32)
    limitations: tuple[str, ...]


class TritonAutotuneSelectionQueryResult(CursorPageContract):
    page_items_field = "selections"

    corpus_commit_id: str
    selections: tuple[TritonAutotuneSelectionView, ...]
    total: int
    evidence: EvidenceAvailability


class TritonAutotuneService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)

    def selections(
        self,
        *,
        run_id: str,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> TritonAutotuneSelectionQueryResult:
        run = self.runs.read(run_id)
        bounded = (
            limit
            if limit is not None
            else min(self.workspace.config.analysis.default_row_limit, _MAX_SELECTION_PAGE_SIZE)
        )
        if bounded < 1 or bounded > min(
            self.workspace.config.analysis.max_row_limit,
            _MAX_SELECTION_PAGE_SIZE,
        ):
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Limit must be between 1 and {_MAX_SELECTION_PAGE_SIZE}.",
            )
        head = self.workspace.corpus.read_head()
        scope_digest = digest_model({"run_id": run_id})
        after = (
            cast(
                tuple[str],
                self.workspace.cursors.resolve(
                    cursor,
                    namespace=CursorNamespace.TRITON_AUTOTUNE_SELECTIONS,
                    snapshot_id=head.commit_id,
                    scope_digest=scope_digest,
                ),
            )[0]
            if cursor is not None
            else None
        )
        where = "run_id = ?"
        parameters: list[object] = [run_id]
        if after is not None:
            where += " AND selection_id > ?"
            parameters.append(after)
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            count = snapshot.execute(
                "SELECT count(*) FROM triton_autotune_selections WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            assert count is not None
            rows = snapshot.execute(
                "SELECT selection_id, run_id, function_name, key_digest, cache_hit, duration_ms, "
                "winner_config_id, candidate_count, candidates_truncated, candidates_json, "
                "limitations FROM triton_autotune_selections WHERE "
                + where
                + " ORDER BY selection_id LIMIT ?",
                (*parameters, bounded + 1),
            ).fetchall()
        selections = tuple(_selection(row) for row in rows[:bounded])
        has_more = len(rows) > bounded
        return TritonAutotuneSelectionQueryResult(
            corpus_commit_id=head.commit_id,
            selections=selections,
            total=int(count[0]),
            next_cursor=(
                self.workspace.cursors.issue(
                    namespace=CursorNamespace.TRITON_AUTOTUNE_SELECTIONS,
                    snapshot_id=head.commit_id,
                    scope_digest=scope_digest,
                    position=(selections[-1].selection_id,),
                )
                if has_more and selections
                else None
            ),
            evidence=(
                available_availability("triton_autotune_selections_present")
                if selections
                else (
                    empty_availability("no_multi_config_autotune_decisions")
                    if run.semantics.adapter == "triton.compiler"
                    else unavailable_availability("triton_autotune_not_captured")
                )
            ),
        )


def _selection(row: tuple[object, ...]) -> TritonAutotuneSelectionView:
    candidates_value = json.loads(cast(str, row[9]))
    if not isinstance(candidates_value, list):
        raise DomainError(
            ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
            "Triton autotune candidates are not a JSON array.",
        )
    try:
        candidates = tuple(
            TritonAutotuneCandidate.model_validate(item) for item in candidates_value
        )
    except ValueError as error:
        raise DomainError(
            ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
            "Triton autotune candidates are invalid.",
        ) from error
    return TritonAutotuneSelectionView(
        selection_id=cast(str, row[0]),
        run_id=cast(str, row[1]),
        function_name=cast(str, row[2]),
        key_digest=cast(str, row[3]),
        cache_hit=cast(bool, row[4]),
        duration_ms=cast(float | None, row[5]),
        winner_config_id=cast(str, row[6]),
        candidate_count=cast(int, row[7]),
        candidates_truncated=cast(bool, row[8]),
        candidates=candidates,
        limitations=tuple(cast(list[str], row[10])),
    )
