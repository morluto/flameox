from __future__ import annotations

from flameox.analysis.recipe_context import RecipeContext
from flameox.analysis.recipe_models import (
    ExecutionAnalysisResult,
    ExecutionObservation,
    ExecutionObservationChange,
)
from flameox.catalog import Snapshot
from flameox.evidence_scope import resolve_evidence_scope
from flameox.evidence_status import available_availability, empty_availability


class ExecutionRecipes(RecipeContext):
    def execution(
        self,
        input_id: str,
        *,
        comparison_input_id: str | None = None,
        limit: int | None = None,
        corpus_commit_id: str | None = None,
    ) -> ExecutionAnalysisResult:
        corpus_commit_id = self._pinned_commit_id(corpus_commit_id)
        bounded = self._limit(limit)
        with self._open_snapshot(corpus_commit_id) as snapshot:
            all_observations, total = self._execution_observations(
                snapshot,
                input_id,
                limit=None if comparison_input_id is not None else bounded,
            )
            compared = (
                self._execution_observations(
                    snapshot,
                    comparison_input_id,
                    limit=None,
                )[0]
                if comparison_input_id is not None
                else ()
            )
        observations = all_observations[:bounded]

        def key(
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

        baseline_by_key = {key(item): item for item in all_observations}
        candidate_by_key = {key(item): item for item in compared}
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
            if baseline_by_key[item_key].value_json != candidate_by_key[item_key].value_json
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
            corpus_commit_id=snapshot.commit.commit_id,
            input_id=input_id,
            observations=observations,
            comparison_input_id=comparison_input_id,
            added=added[:bounded],
            removed=removed[:bounded],
            changed=changed[:bounded],
            total=total,
            returned=len(observations),
            truncated=total > len(observations),
            limitations=tuple(limitations),
            evidence=(
                empty_availability("no_execution_observations")
                if not (total or added or removed or changed)
                else available_availability()
            ),
        )

    def _execution_observations(
        self,
        snapshot: Snapshot,
        input_id: str,
        *,
        limit: int | None,
    ) -> tuple[tuple[ExecutionObservation, ...], int]:
        scope = resolve_evidence_scope(snapshot, input_id)
        where, parameters = scope.predicate(
            run_column="run_id",
            artifact_column="artifact_id",
        )
        count_row = snapshot.execute(
            "SELECT count(*) FROM observations WHERE " + where,
            parameters,
        ).fetchone()
        assert count_row is not None
        query = (
            "SELECT observation_id, kind, name, value_json, file, line_from, "
            "line_to, context, evidence_level FROM observations WHERE "
            + where
            + " ORDER BY file, line_from, line_to, observation_id"
        )
        rows = snapshot.execute(
            query + (" LIMIT ?" if limit is not None else ""),
            (*parameters, limit) if limit is not None else parameters,
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
                    evidence_level=str(row[8]),
                )
                for row in rows
            ),
            int(count_row[0]),
        )
