from __future__ import annotations

from enum import StrEnum
from typing import cast

from pydantic import Field

from flameox.action_graph import ActionId, ToolAction, tool_action
from flameox.catalog import Catalog, Snapshot
from flameox.domain import CursorNamespace, DomainError, ErrorCode, digest_model
from flameox.evidence_scope import EvidenceScope, resolve_evidence_scope
from flameox.evidence_status import (
    EvidenceAvailability,
    available_availability,
    empty_availability,
)
from flameox.models import ContractModel
from flameox.pagination import CursorPageContract
from flameox.storage import Workspace


class CallDirection(StrEnum):
    CALLERS = "callers"
    CALLEES = "callees"


class FrameDetail(ContractModel):
    frame_id: str
    function: str | None
    module: str | None
    file: str | None
    line: int | None


class CallEdgeDetail(FrameDetail):
    metric: str
    weight_value: int
    unit: str
    sample_count: int


class CallEdgeResult(CursorPageContract):
    page_items_field = "frames"

    corpus_commit_id: str
    input_id: str
    frame_id: str
    metric: str | None
    direction: CallDirection
    frames: tuple[CallEdgeDetail, ...]
    total: int
    coverage: float
    limitations: tuple[str, ...] = (
        "Edges represent captured stack adjacency, not causal dependence.",
    )
    evidence: EvidenceAvailability = Field(default_factory=available_availability)
    recovery: ToolAction | None = None


class StackExample(ContractModel):
    stack_id: str
    metric: str
    weight_value: int
    unit: str
    sample_count: int
    start_ns: int | None
    track_id: int | None
    frames: tuple[FrameDetail, ...]


class StackExamplesResult(CursorPageContract):
    page_items_field = "examples"

    corpus_commit_id: str
    input_id: str
    frame_id: str
    metric: str | None
    examples: tuple[StackExample, ...]
    total: int
    coverage: float
    limitations: tuple[str, ...] = (
        "Examples are representative leaf stacks retained during extraction.",
    )
    evidence: EvidenceAvailability = Field(default_factory=available_availability)
    recovery: ToolAction | None = None


class DrilldownService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def callers(
        self,
        input_id: str,
        frame_id: str,
        *,
        metric: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CallEdgeResult:
        return self._edges(
            input_id,
            frame_id,
            direction=CallDirection.CALLERS,
            metric=metric,
            limit=limit,
            cursor=cursor,
        )

    def callees(
        self,
        input_id: str,
        frame_id: str,
        *,
        metric: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CallEdgeResult:
        return self._edges(
            input_id,
            frame_id,
            direction=CallDirection.CALLEES,
            metric=metric,
            limit=limit,
            cursor=cursor,
        )

    def examples(
        self,
        input_id: str,
        frame_id: str,
        *,
        metric: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> StackExamplesResult:
        limit = self._limit(limit)
        head = self.workspace.corpus.read_head()
        scope_digest = digest_model({"input_id": input_id, "frame_id": frame_id, "metric": metric})
        after_weight: int | None = None
        after_metric: str | None = None
        after_unit: str | None = None
        after_stack_id: str | None = None
        if cursor is not None:
            position = cast(
                tuple[int, str, str, str],
                self.workspace.cursors.resolve(
                    cursor,
                    namespace=CursorNamespace.STACK_EXAMPLES,
                    snapshot_id=head.commit_id,
                    scope_digest=scope_digest,
                ),
            )
            weight_value, metric_value, unit_value, stack_value = position
            assert isinstance(weight_value, int)
            assert isinstance(metric_value, str)
            assert isinstance(unit_value, str)
            assert isinstance(stack_value, str)
            after_weight, after_metric, after_unit, after_stack_id = (
                weight_value,
                metric_value,
                unit_value,
                stack_value,
            )
        page_where = ""
        page_parameters: tuple[object, ...] = ()
        if (
            after_weight is not None
            and after_metric is not None
            and after_unit is not None
            and after_stack_id is not None
        ):
            page_where = (
                " WHERE weight_value < ? OR "
                "(weight_value = ? AND metric > ?) OR "
                "(weight_value = ? AND metric = ? AND unit > ?) OR "
                "(weight_value = ? AND metric = ? AND unit = ? AND stack_id > ?)"
            )
            page_parameters = (
                after_weight,
                after_weight,
                after_metric,
                after_weight,
                after_metric,
                after_unit,
                after_weight,
                after_metric,
                after_unit,
                after_stack_id,
            )
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            scope = resolve_evidence_scope(snapshot, input_id)
            where, parameters = scope.predicate(
                run_column="run_id",
                artifact_column="artifact_id",
            )
            metric_where = " AND metric = ?" if metric is not None else ""
            scoped = (
                "SELECT DISTINCT stack_id, metric, weight_value, unit, sample_count, "
                "start_ns, track_id, frame_ids FROM stacks WHERE "
                + where
                + " AND list_contains(frame_ids, ?)"
                + metric_where
            )
            scoped_parameters = (*parameters, frame_id, *((metric,) if metric is not None else ()))
            rows = snapshot.execute(
                "WITH scoped AS ("
                + scoped
                + ") SELECT * FROM scoped"
                + page_where
                + " ORDER BY weight_value DESC, metric, unit, stack_id LIMIT ?",
                (*scoped_parameters, *page_parameters, limit + 1),
            ).fetchall()
            count_row = snapshot.execute(
                "WITH scoped AS (" + scoped + ") SELECT count(*) FROM scoped",
                scoped_parameters,
            ).fetchone()
            if count_row is None:
                raise DomainError(
                    ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                    "Stack count query returned no row.",
                )
            total = int(count_row[0])
            selected = rows[:limit]
            recovery = (
                self._memory_analysis_recovery(input_id, metric, limit)
                or self._native_viewer_recovery(snapshot, scope)
                if not selected
                else None
            )
            frame_ids = {str(item) for row in selected for item in row[7]}
            frames = self._frames(snapshot, frame_ids)
            self._require_frames(frames, frame_ids)
        examples = tuple(
            StackExample(
                stack_id=str(row[0]),
                metric=str(row[1]),
                weight_value=int(row[2]),
                unit=str(row[3]),
                sample_count=int(row[4]),
                start_ns=int(row[5]) if row[5] is not None else None,
                track_id=int(row[6]) if row[6] is not None else None,
                frames=tuple(frames[str(item)] for item in row[7]),
            )
            for row in selected
        )
        has_more = len(rows) > limit
        next_cursor = (
            self.workspace.cursors.issue(
                namespace=CursorNamespace.STACK_EXAMPLES,
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
                position=(
                    examples[-1].weight_value,
                    examples[-1].metric,
                    examples[-1].unit,
                    examples[-1].stack_id,
                ),
            )
            if has_more and examples
            else None
        )
        return StackExamplesResult(
            corpus_commit_id=head.commit_id,
            input_id=input_id,
            frame_id=frame_id,
            metric=metric,
            examples=examples,
            total=total,
            coverage=(len(examples) / total if total else 0.0),
            evidence=(
                empty_availability("no_matching_stacks")
                if not examples
                else available_availability("stacks_present")
            ),
            recovery=recovery,
            next_cursor=next_cursor,
        )

    def _edges(
        self,
        input_id: str,
        frame_id: str,
        *,
        direction: CallDirection,
        metric: str | None,
        limit: int,
        cursor: str | None,
    ) -> CallEdgeResult:
        limit = self._limit(limit)
        head = self.workspace.corpus.read_head()
        selected_column = (
            "parent_frame_id" if direction is CallDirection.CALLERS else "child_frame_id"
        )
        match_column = "child_frame_id" if direction is CallDirection.CALLERS else "parent_frame_id"
        scope_digest = digest_model(
            {
                "input_id": input_id,
                "frame_id": frame_id,
                "direction": direction,
                "metric": metric,
            }
        )
        after_weight: int | None = None
        after_metric: str | None = None
        after_unit: str | None = None
        after_frame_id: str | None = None
        if cursor is not None:
            position = cast(
                tuple[int, str, str, str],
                self.workspace.cursors.resolve(
                    cursor,
                    namespace=CursorNamespace.CALL_EDGES,
                    snapshot_id=head.commit_id,
                    scope_digest=scope_digest,
                ),
            )
            weight_value, metric_value, unit_value, frame_value = position
            assert isinstance(weight_value, int)
            assert isinstance(metric_value, str)
            assert isinstance(unit_value, str)
            assert isinstance(frame_value, str)
            after_weight, after_metric, after_unit, after_frame_id = (
                weight_value,
                metric_value,
                unit_value,
                frame_value,
            )
        grouped = (
            f"SELECT {selected_column} AS frame_id, metric, unit, "
            "sum(sample_count) AS sample_count, sum(weight_value) AS weight_value "
            f"FROM scoped GROUP BY {selected_column}, metric, unit"
        )
        page_where = ""
        page_parameters: tuple[object, ...] = ()
        if (
            after_weight is not None
            and after_metric is not None
            and after_unit is not None
            and after_frame_id is not None
        ):
            page_where = (
                " WHERE weight_value < ? OR "
                "(weight_value = ? AND metric > ?) OR "
                "(weight_value = ? AND metric = ? AND unit > ?) OR "
                "(weight_value = ? AND metric = ? AND unit = ? AND frame_id > ?)"
            )
            page_parameters = (
                after_weight,
                after_weight,
                after_metric,
                after_weight,
                after_metric,
                after_unit,
                after_weight,
                after_metric,
                after_unit,
                after_frame_id,
            )
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            scope = resolve_evidence_scope(snapshot, input_id)
            where, parameters = scope.predicate(
                run_column="run_id",
                artifact_column="artifact_id",
            )
            metric_where = " AND metric = ?" if metric is not None else ""
            base = (
                "SELECT DISTINCT run_id, artifact_id, parent_frame_id, "
                "child_frame_id, metric, weight_value, unit, sample_count "
                "FROM call_edges WHERE " + where + f" AND {match_column} = ?" + metric_where
            )
            base_parameters = (*parameters, frame_id, *((metric,) if metric is not None else ()))
            rows = snapshot.execute(
                f"WITH scoped AS ({base}), grouped AS ({grouped}) "
                "SELECT frame_id, metric, unit, sample_count, weight_value FROM grouped"
                + page_where
                + " ORDER BY weight_value DESC, metric, unit, frame_id LIMIT ?",
                (*base_parameters, *page_parameters, limit + 1),
            ).fetchall()
            count_row = snapshot.execute(
                f"WITH scoped AS ({base}), grouped AS ({grouped}) SELECT count(*) FROM grouped",
                base_parameters,
            ).fetchone()
            if count_row is None:
                raise DomainError(
                    ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                    "Call-edge count query returned no row.",
                )
            total = int(count_row[0])
            selected = rows[:limit]
            recovery = (
                self._memory_analysis_recovery(input_id, metric, limit)
                or self._native_viewer_recovery(snapshot, scope)
                if not selected
                else None
            )
            metadata = self._frames(
                snapshot,
                {str(row[0]) for row in selected},
            )
            self._require_frames(
                metadata,
                {str(row[0]) for row in selected},
            )
        frames = tuple(
            CallEdgeDetail(
                **metadata[str(row[0])].model_dump(),
                metric=str(row[1]),
                unit=str(row[2]),
                sample_count=int(row[3]),
                weight_value=int(row[4]),
            )
            for row in selected
        )
        has_more = len(rows) > limit
        next_cursor = (
            self.workspace.cursors.issue(
                namespace=CursorNamespace.CALL_EDGES,
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
                position=(
                    frames[-1].weight_value,
                    frames[-1].metric,
                    frames[-1].unit,
                    frames[-1].frame_id,
                ),
            )
            if has_more and frames
            else None
        )
        return CallEdgeResult(
            corpus_commit_id=head.commit_id,
            input_id=input_id,
            frame_id=frame_id,
            metric=metric,
            direction=direction,
            frames=frames,
            total=total,
            coverage=(len(frames) / total if total else 0.0),
            evidence=(
                empty_availability("no_matching_edges")
                if not frames
                else available_availability("edges_present")
            ),
            recovery=recovery,
            next_cursor=next_cursor,
        )

    @staticmethod
    def _memory_analysis_recovery(
        input_id: str,
        metric: str | None,
        limit: int,
    ) -> ToolAction | None:
        if metric is None:
            return None
        view = {
            "memory.high_watermark": "high_watermark",
            "memory.retained_end": "retained_end",
            "memory.allocated": "allocation_volume",
            "memory.temporary": "temporary",
        }.get(metric)
        if view is None:
            return None
        return tool_action(
            ActionId.ANALYZE_MEMORY,
            run_or_artifact=input_id,
            limit=limit,
            query={"view": view, "ranking": "inclusive", "project_only": True},
        )

    @staticmethod
    def _native_viewer_recovery(
        snapshot: Snapshot,
        scope: EvidenceScope,
    ) -> ToolAction | None:
        if len(scope.artifact_ids) == 1:
            artifact_id = scope.artifact_ids[0]
        elif scope.run_ids:
            placeholders = ", ".join("?" for _ in scope.run_ids)
            rows = snapshot.execute(
                "SELECT DISTINCT artifact_id FROM artifact_registrations "
                f"WHERE run_id IN ({placeholders}) AND kind = ? "
                "ORDER BY artifact_id LIMIT 2",
                (*scope.run_ids, "memory_profile"),
            ).fetchall()
            if len(rows) != 1:
                return None
            artifact_id = str(rows[0][0])
        else:
            return None
        return tool_action(ActionId.GET_NATIVE_VIEWER_PLAN, artifact_id=artifact_id)

    def _frames(
        self,
        snapshot: Snapshot,
        frame_ids: set[str],
    ) -> dict[str, FrameDetail]:
        if not frame_ids:
            return {}
        placeholders = ", ".join("?" for _ in frame_ids)
        rows = snapshot.execute(
            "SELECT DISTINCT frame_id, function, module, file, line FROM frames "
            f"WHERE frame_id IN ({placeholders})",
            tuple(sorted(frame_ids)),
        ).fetchall()
        return {
            str(row[0]): FrameDetail(
                frame_id=str(row[0]),
                function=str(row[1]) if row[1] is not None else None,
                module=str(row[2]) if row[2] is not None else None,
                file=str(row[3]) if row[3] is not None else None,
                line=int(row[4]) if row[4] is not None else None,
            )
            for row in rows
        }

    def _require_frames(
        self,
        frames: dict[str, FrameDetail],
        expected: set[str],
    ) -> None:
        missing = expected - set(frames)
        if missing:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                "Stack evidence references missing frame rows.",
                details={"missing_frame_ids": sorted(missing)},
            )

    def _limit(self, value: int) -> int:
        maximum = self.workspace.config.analysis.max_row_limit
        if value < 1 or value > maximum:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Limit must be between 1 and {maximum}.",
            )
        return value
