from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from flamo.catalog import Catalog, Snapshot
from flamo.domain import CursorCodec, DomainError, ErrorCode, digest_model
from flamo.storage import ArtifactStore, RunStore, Workspace


class FrameDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: str
    function: str | None
    module: str | None
    file: str | None
    line: int | None
    sample_count: int
    duration_ns: int


class CallEdgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    frame_id: str
    direction: Literal["callers", "callees"]
    frames: tuple[FrameDetail, ...]
    total: int
    returned: int
    truncated: bool
    coverage: float
    next_cursor: str | None
    limitations: tuple[str, ...] = (
        "Edges represent syntactic nesting in the captured trace, not causal dependence.",
    )


class StackExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stack_id: str
    start_ns: int
    duration_ns: int
    track_id: int
    frames: tuple[FrameDetail, ...]


class StackExamplesResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    frame_id: str
    examples: tuple[StackExample, ...]
    total: int
    returned: int
    truncated: bool
    coverage: float
    next_cursor: str | None
    limitations: tuple[str, ...] = (
        "Examples are representative leaf stacks retained during extraction.",
    )


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
            direction="callers",
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
            direction="callees",
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
        run_ids, artifact_ids = self._scope(input_id)
        where, parameters = self._scope_where(
            run_ids,
            artifact_ids,
            run_column="run_id",
            artifact_column="artifact_id",
        )
        scope_digest = digest_model({"input_id": input_id, "frame_id": frame_id})
        after_duration: int | None = None
        after_stack_id: str | None = None
        if cursor is not None:
            position = CursorCodec.decode(
                cursor,
                namespace="stack_examples",
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
            )
            if (
                len(position) != 2
                or not isinstance(position[0], int)
                or not isinstance(position[1], str)
            ):
                raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.")
            duration_value, stack_value = position
            assert isinstance(duration_value, int)
            assert isinstance(stack_value, str)
            after_duration, after_stack_id = duration_value, stack_value
        scoped = (
            "SELECT DISTINCT stack_id, start_ns, duration_ns, track_id, frame_ids "
            "FROM stacks WHERE "
            + where
            + " AND list_contains(frame_ids, ?)"
        )
        page_where = ""
        page_parameters: tuple[object, ...] = ()
        if after_duration is not None and after_stack_id is not None:
            page_where = (
                " WHERE duration_ns < ? OR "
                "(duration_ns = ? AND stack_id > ?)"
            )
            page_parameters = (after_duration, after_duration, after_stack_id)
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            rows = snapshot.execute(
                "WITH scoped AS (" + scoped + ") SELECT * FROM scoped"
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
            CursorCodec.encode(
                namespace="stack_examples",
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
            returned=len(examples),
            truncated=has_more,
            coverage=(len(examples) / total if total else 1.0),
            next_cursor=next_cursor,
        )

    def _edges(
        self,
        input_id: str,
        frame_id: str,
        *,
        direction: Literal["callers", "callees"],
        limit: int,
        cursor: str | None,
    ) -> CallEdgeResult:
        limit = self._limit(limit)
        head = self.workspace.corpus.read_head()
        run_ids, artifact_ids = self._scope(input_id)
        where, parameters = self._scope_where(
            run_ids,
            artifact_ids,
            run_column="run_id",
            artifact_column="artifact_id",
        )
        selected_column = "parent_frame_id" if direction == "callers" else "child_frame_id"
        match_column = "child_frame_id" if direction == "callers" else "parent_frame_id"
        base = (
            "SELECT DISTINCT run_id, artifact_id, parent_frame_id, "
            "child_frame_id, sample_count, duration_ns FROM call_edges WHERE "
            + where
            + f" AND {match_column} = ?"
        )
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
            position = CursorCodec.decode(
                cursor,
                namespace="call_edges",
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
            )
            if (
                len(position) != 2
                or not isinstance(position[0], int)
                or not isinstance(position[1], str)
            ):
                raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.")
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
            page_where = (
                " WHERE duration_ns < ? OR "
                "(duration_ns = ? AND frame_id > ?)"
            )
            page_parameters = (after_duration, after_duration, after_frame_id)
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            rows = snapshot.execute(
                f"WITH scoped AS ({base}), grouped AS ({grouped}) "
                "SELECT frame_id, sample_count, duration_ns FROM grouped"
                + page_where
                + " ORDER BY duration_ns DESC, frame_id LIMIT ?",
                (*parameters, frame_id, *page_parameters, limit + 1),
            ).fetchall()
            count_row = snapshot.execute(
                f"WITH scoped AS ({base}), grouped AS ({grouped}) "
                "SELECT count(*) FROM grouped",
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
            metadata[str(row[0])].model_copy(
                update={
                    "sample_count": int(row[1]),
                    "duration_ns": int(row[2]),
                }
            )
            for row in selected
        )
        has_more = len(rows) > limit
        next_cursor = (
            CursorCodec.encode(
                namespace="call_edges",
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
            returned=len(frames),
            truncated=has_more,
            coverage=(len(frames) / total if total else 1.0),
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

    def _scope(self, input_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        try:
            run = RunStore(self.workspace).read(input_id)
        except DomainError:
            ArtifactStore(self.workspace).get(input_id)
            return (), (input_id,)
        return (run.run_id,), tuple(item.artifact_id for item in run.artifacts)

    def _scope_where(
        self,
        run_ids: tuple[str, ...],
        artifact_ids: tuple[str, ...],
        *,
        run_column: str,
        artifact_column: str,
    ) -> tuple[str, tuple[object, ...]]:
        if run_ids:
            placeholders = ", ".join("?" for _ in run_ids)
            return f"{run_column} IN ({placeholders})", tuple(run_ids)
        placeholders = ", ".join("?" for _ in artifact_ids)
        return f"{artifact_column} IN ({placeholders})", tuple(artifact_ids)

    def _limit(self, value: int) -> int:
        maximum = self.workspace.config.analysis.max_row_limit
        if value < 1 or value > maximum:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Limit must be between 1 and {maximum}.",
            )
        return value
