from __future__ import annotations

from flameox.analysis.recipe_context import RecipeContext
from flameox.analysis.recipe_models import (
    ExecutionAnalysisResult,
    ExecutionCollection,
    ExecutionCollectionTotals,
    ExecutionObservation,
    ExecutionObservationChange,
    ExecutionObservationFilter,
)
from flameox.catalog import Snapshot
from flameox.domain import CursorNamespace, DomainError, ErrorCode, digest_model
from flameox.domain.models import EvidenceLevel
from flameox.evidence_scope import resolve_evidence_scope
from flameox.evidence_status import available_availability, empty_availability


def execution_query_scope_digest(
    input_id: str,
    comparison_input_id: str | None,
    collection: ExecutionCollection,
    filters: ExecutionObservationFilter,
) -> str:
    return digest_model(
        {
            "input_id": input_id,
            "comparison_input_id": comparison_input_id,
            "collection": collection,
            "filters": filters.model_dump(mode="json"),
        }
    )


def _observation_key(
    item: ExecutionObservation,
) -> tuple[str, str, str | None, int | None, int | None, str | None]:
    return (
        item.kind,
        item.name,
        item.file,
        item.line_from,
        item.line_to,
        item.context,
    )


def _observations_by_key(
    observations: tuple[ExecutionObservation, ...],
    *,
    input_id: str,
) -> dict[tuple[str, str, str | None, int | None, int | None, str | None], ExecutionObservation]:
    keyed: dict[
        tuple[str, str, str | None, int | None, int | None, str | None],
        ExecutionObservation,
    ] = {}
    for observation in observations:
        key = _observation_key(observation)
        if key in keyed:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                "Execution comparison requires one observation per semantic location.",
                details={"input_id": input_id, "duplicate_observation_key": list(key)},
                remediation=(
                    "Normalize repeated observations into one value before comparing runs.",
                ),
            )
        keyed[key] = observation
    return keyed


class ExecutionRecipes(RecipeContext):
    def execution(
        self,
        input_id: str,
        *,
        comparison_input_id: str | None = None,
        collection: ExecutionCollection = "observations",
        filters: ExecutionObservationFilter | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        corpus_commit_id: str | None = None,
    ) -> ExecutionAnalysisResult:
        selected_filters = filters or ExecutionObservationFilter()
        if comparison_input_id is None and collection != "observations":
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "Comparison execution collections require comparison_run_or_artifact.",
            )
        scope_digest = execution_query_scope_digest(
            input_id,
            comparison_input_id,
            collection,
            selected_filters,
        )
        offset = 0
        if cursor is not None:
            cursor_commit_id, position = self.workspace.cursors.resolve_bound(
                cursor,
                namespace=CursorNamespace.EXECUTION_ANALYSIS,
                scope_digest=scope_digest,
            )
            if corpus_commit_id is not None and corpus_commit_id != cursor_commit_id:
                raise DomainError(
                    ErrorCode.STALE_CURSOR,
                    "Cursor belongs to a different immutable corpus snapshot.",
                )
            corpus_commit_id = cursor_commit_id
            offset = int(position[0])
        corpus_commit_id = self._pinned_commit_id(corpus_commit_id)
        bounded = self._limit(limit)
        with self._open_snapshot(corpus_commit_id) as snapshot:
            all_observations, total = self._execution_observations(
                snapshot,
                input_id,
                filters=selected_filters,
                limit=bounded if comparison_input_id is None else None,
                offset=offset if comparison_input_id is None else 0,
            )
            compared = (
                self._execution_observations(
                    snapshot,
                    comparison_input_id,
                    filters=selected_filters,
                    limit=None,
                )[0]
                if comparison_input_id is not None
                else ()
            )
            snapshot_commit_id = snapshot.commit.commit_id

        if comparison_input_id is None:
            added: tuple[ExecutionObservation, ...] = ()
            removed: tuple[ExecutionObservation, ...] = ()
            changed: tuple[ExecutionObservationChange, ...] = ()
        else:
            baseline_by_key = _observations_by_key(all_observations, input_id=input_id)
            assert comparison_input_id is not None
            candidate_by_key = _observations_by_key(
                compared,
                input_id=comparison_input_id,
            )
            added = tuple(
                candidate_by_key[item_key]
                for item_key in sorted(
                    candidate_by_key.keys() - baseline_by_key.keys(),
                    key=repr,
                )
            )
            removed = tuple(
                baseline_by_key[item_key]
                for item_key in sorted(
                    baseline_by_key.keys() - candidate_by_key.keys(),
                    key=repr,
                )
            )
            changed = tuple(
                ExecutionObservationChange(
                    kind=item_key[0],
                    name=item_key[1],
                    file=item_key[2],
                    line_from=item_key[3],
                    line_to=item_key[4],
                    context=item_key[5],
                    baseline_value_json=baseline_by_key[item_key].value_json,
                    candidate_value_json=candidate_by_key[item_key].value_json,
                )
                for item_key in sorted(
                    baseline_by_key.keys() & candidate_by_key.keys(),
                    key=repr,
                )
                if baseline_by_key[item_key].value_json
                != candidate_by_key[item_key].value_json
            )
        collections: dict[
            str,
            tuple[ExecutionObservation | ExecutionObservationChange, ...],
        ] = {
            "observations": all_observations,
            "added": added,
            "removed": removed,
            "changed": changed,
        }
        selected = collections[collection]
        page = selected if comparison_input_id is None else selected[offset : offset + bounded]
        next_offset = offset + len(page)
        selected_total = total if comparison_input_id is None else len(selected)
        next_cursor = (
            self.workspace.cursors.issue(
                namespace=CursorNamespace.EXECUTION_ANALYSIS,
                snapshot_id=snapshot_commit_id,
                scope_digest=scope_digest,
                position=(next_offset,),
            )
            if next_offset < selected_total
            else None
        )
        totals = ExecutionCollectionTotals(
            observations=total,
            added=len(added),
            removed=len(removed),
            changed=len(changed),
        )
        limitations = [
            "Coverage proves that a path executed, not why it executed or which "
            "values controlled it."
        ]
        if comparison_input_id is not None:
            limitations.append(
                "Execution-path differences report observed path or value changes; "
                "they do not establish causality."
            )
        return ExecutionAnalysisResult(
            corpus_commit_id=snapshot_commit_id,
            input_id=input_id,
            comparison_input_id=comparison_input_id,
            collection=collection,
            filters=selected_filters,
            items=page,
            total=selected_total,
            totals=totals,
            next_cursor=next_cursor,
            limitations=tuple(limitations),
            evidence=(
                empty_availability("no_execution_observations")
                if not any((totals.observations, totals.added, totals.removed, totals.changed))
                else available_availability()
            ),
        )

    def _execution_observations(
        self,
        snapshot: Snapshot,
        input_id: str,
        *,
        filters: ExecutionObservationFilter,
        limit: int | None,
        offset: int = 0,
    ) -> tuple[tuple[ExecutionObservation, ...], int]:
        scope = resolve_evidence_scope(snapshot, input_id)
        where, parameters = scope.predicate(
            run_column="run_id",
            artifact_column="artifact_id",
        )
        predicates = [where]
        query_parameters = list(parameters)
        if filters.file_prefix is not None:
            escaped = (
                filters.file_prefix.replace("^", "^^").replace("%", "^%").replace("_", "^_")
            )
            predicates.append("file LIKE ? ESCAPE '^'")
            query_parameters.append(f"{escaped}%")
        if filters.kind is not None:
            predicates.append("kind = ?")
            query_parameters.append(filters.kind)
        if filters.name is not None:
            predicates.append("name = ?")
            query_parameters.append(filters.name)
        if filters.line_from is not None:
            predicates.append("line_to >= ?")
            query_parameters.append(filters.line_from)
        if filters.line_to is not None:
            predicates.append("line_from <= ?")
            query_parameters.append(filters.line_to)
        filtered_where = " AND ".join(predicates)
        count_row = snapshot.execute(
            "SELECT count(*) FROM observations WHERE " + filtered_where,
            tuple(query_parameters),
        ).fetchone()
        assert count_row is not None
        query = (
            "SELECT observation_id, kind, name, value_json, file, line_from, "
            "line_to, context, evidence_level FROM observations WHERE "
            + filtered_where
            + " ORDER BY file, line_from, line_to, observation_id"
        )
        rows = snapshot.execute(
            query + (" LIMIT ? OFFSET ?" if limit is not None else ""),
            (*query_parameters, limit, offset) if limit is not None else tuple(query_parameters),
        ).fetchall()
        return (
            tuple(
                ExecutionObservation(
                    observation_id=str(row[0]),
                    kind=str(row[1]),
                    name=str(row[2]),
                    value_json=str(row[3]),
                    file=str(row[4]) if row[4] is not None else None,
                    line_from=int(row[5]) if row[5] is not None else None,
                    line_to=int(row[6]) if row[6] is not None else None,
                    context=str(row[7]) if row[7] is not None else None,
                    evidence_level=EvidenceLevel(str(row[8])),
                )
                for row in rows
            ),
            int(count_row[0]),
        )
