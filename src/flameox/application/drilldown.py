from __future__ import annotations

from enum import StrEnum
from typing import cast

from pydantic import Field

from flameox.catalog import Catalog, Snapshot
from flameox.domain import CursorNamespace, DomainError, ErrorCode, digest_model
from flameox.evidence_scope import resolve_evidence_scope
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
    sample_count: int
    duration_ns: int


class CallEdgeResult(CursorPageContract):
    page_items_field = "frames"

    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    frame_id: str
    direction: CallDirection
    frames: tuple[FrameDetail, ...]
    total: int
    coverage: float
    limitations: tuple[str, ...] = (
        "Edges represent syntactic nesting in the captured trace, not causal dependence.",
    )
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class StackExample(ContractModel):
    stack_id: str
    start_ns: int
    duration_ns: int
    track_id: int
    frames: tuple[FrameDetail, ...]


class StackExamplesResult(CursorPageContract):
    page_items_field = "examples"

    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    frame_id: str
    examples: tuple[StackExample, ...]
    total: int
    coverage: float
    limitations: tuple[str, ...] = (
        "Examples are representative leaf stacks retained during extraction.",
    )
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class DrilldownService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def callers(
        self,
        input_id: str,
        frame_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CallEdgeResult:
        return self._edges(
            input_id,
            frame_id,
            direction=CallDirection.CALLERS,
            limit=limit,
            cursor=cursor,
        )

    def callees(
        self,
        input_id: str,
        frame_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CallEdgeResult:
        return self._edges(
            input_id,
            frame_id,
            direction=CallDirection.CALLEES,
            limit=limit,
            cursor=cursor,
        )

    def examples(
        self,
        input_id: str,
        frame_id: str,
        *,
        limit: int = 20,
        cursor: str | None = None,
    ) -> StackExamplesResult:
        limit = self._limit(limit)
        head = self.workspace.corpus.read_head()
        scope_digest = digest_model({"input_id": input_id, "frame_id": frame_id})
        after_duration: int | None = None
        after_stack_id: str | None = None
        if cursor is not None:
            position = cast(
                tuple[int, str],
                self.workspace.cursors.resolve(
                    cursor,
                    namespace=CursorNamespace.STACK_EXAMPLES,
                    snapshot_id=head.commit_id,
                    scope_digest=scope_digest,
                ),
            )
            duration_value, stack_value = position
            assert isinstance(duration_value, int)
            assert isinstance(stack_value, str)
            after_duration, after_stack_id = duration_value, stack_value
        page_where = ""
        page_parameters: tuple[object, ...] = ()
        if after_duration is not None and after_stack_id is not None:
            page_where = " WHERE duration_ns < ? OR (duration_ns = ? AND stack_id > ?)"
            page_parameters = (after_duration, after_duration, after_stack_id)
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            scope = resolve_evidence_scope(snapshot, input_id)
            where, parameters = scope.predicate(
                run_column="run_id",
                artifact_column="artifact_id",
            )
            scoped = (
                "SELECT DISTINCT stack_id, start_ns, duration_ns, track_id, frame_ids "
                "FROM stacks WHERE " + where + " AND list_contains(frame_ids, ?)"
            )
            rows = snapshot.execute(
                "WITH scoped AS ("
                + scoped
                + ") SELECT * FROM scoped"
                + page_where
                + " ORDER BY duration_ns DESC, stack_id LIMIT ?",
                (*parameters, frame_id, *page_parameters, limit + 1),
            ).fetchall()
            count_row = snapshot.execute(
                "WITH scoped AS (" + scoped + ") SELECT count(*) FROM scoped",
                (*parameters, frame_id),
            ).fetchone()
            if count_row is None:
                raise DomainError(
                    ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                    "Stack count query returned no row.",
                )
            total = int(count_row[0])
            selected = rows[:limit]
            frame_ids = {str(item) for row in selected for item in row[4]}
            frames = self._frames(snapshot, frame_ids)
            self._require_frames(frames, frame_ids)
        examples = tuple(
            StackExample(
                stack_id=str(row[0]),
                start_ns=int(row[1]),
                duration_ns=int(row[2]),
                track_id=int(row[3]),
                frames=tuple(frames[str(item)] for item in row[4]),
            )
            for row in selected
        )
        has_more = len(rows) > limit
        next_cursor = (
            self.workspace.cursors.issue(
                namespace=CursorNamespace.STACK_EXAMPLES,
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
                position=(examples[-1].duration_ns, examples[-1].stack_id),
            )
            if has_more and examples
            else None
        )
        return StackExamplesResult(
            corpus_commit_id=head.commit_id,
            input_id=input_id,
            frame_id=frame_id,
            examples=examples,
            total=total,
            coverage=(len(examples) / total if total else 0.0),
            evidence=(
                empty_availability("no_matching_stacks")
                if not examples
                else available_availability("stacks_present")
            ),
            next_cursor=next_cursor,
        )

    def _edges(
        self,
        input_id: str,
        frame_id: str,
        *,
        direction: CallDirection,
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
            }
        )
        after_duration: int | None = None
        after_frame_id: str | None = None
        if cursor is not None:
            position = cast(
                tuple[int, str],
                self.workspace.cursors.resolve(
                    cursor,
                    namespace=CursorNamespace.CALL_EDGES,
                    snapshot_id=head.commit_id,
                    scope_digest=scope_digest,
                ),
            )
            duration_value, frame_value = position
            assert isinstance(duration_value, int)
            assert isinstance(frame_value, str)
            after_duration, after_frame_id = duration_value, frame_value
        grouped = (
            f"SELECT {selected_column} AS frame_id, "
            "sum(sample_count) AS sample_count, sum(duration_ns) AS duration_ns "
            f"FROM scoped GROUP BY {selected_column}"
        )
        page_where = ""
        page_parameters: tuple[object, ...] = ()
        if after_duration is not None and after_frame_id is not None:
            page_where = " WHERE duration_ns < ? OR (duration_ns = ? AND frame_id > ?)"
            page_parameters = (after_duration, after_duration, after_frame_id)
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            scope = resolve_evidence_scope(snapshot, input_id)
            where, parameters = scope.predicate(
                run_column="run_id",
                artifact_column="artifact_id",
            )
            base = (
                "SELECT DISTINCT run_id, artifact_id, parent_frame_id, "
                "child_frame_id, sample_count, duration_ns FROM call_edges WHERE "
                + where
                + f" AND {match_column} = ?"
            )
            rows = snapshot.execute(
                f"WITH scoped AS ({base}), grouped AS ({grouped}) "
                "SELECT frame_id, sample_count, duration_ns FROM grouped"
                + page_where
                + " ORDER BY duration_ns DESC, frame_id LIMIT ?",
                (*parameters, frame_id, *page_parameters, limit + 1),
            ).fetchall()
            count_row = snapshot.execute(
                f"WITH scoped AS ({base}), grouped AS ({grouped}) SELECT count(*) FROM grouped",
                (*parameters, frame_id),
            ).fetchone()
            if count_row is None:
                raise DomainError(
                    ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                    "Call-edge count query returned no row.",
                )
            total = int(count_row[0])
            selected = rows[:limit]
            metadata = self._frames(
                snapshot,
                {str(row[0]) for row in selected},
            )
            self._require_frames(
                metadata,
                {str(row[0]) for row in selected},
            )
        frames = tuple(
            metadata[str(row[0])].validated_copy(
                update={
                    "sample_count": int(row[1]),
                    "duration_ns": int(row[2]),
                }
            )
            for row in selected
        )
        has_more = len(rows) > limit
        next_cursor = (
            self.workspace.cursors.issue(
                namespace=CursorNamespace.CALL_EDGES,
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
                position=(frames[-1].duration_ns, frames[-1].frame_id),
            )
            if has_more and frames
            else None
        )
        return CallEdgeResult(
            corpus_commit_id=head.commit_id,
            input_id=input_id,
            frame_id=frame_id,
            direction=direction,
            frames=frames,
            total=total,
            coverage=(len(frames) / total if total else 0.0),
            evidence=(
                empty_availability("no_matching_edges")
                if not frames
                else available_availability("edges_present")
            ),
            next_cursor=next_cursor,
        )

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
                sample_count=0,
                duration_ns=0,
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
