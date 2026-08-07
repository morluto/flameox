from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from flameox.catalog import Catalog
from flameox.domain import CursorCodec, DomainError, ErrorCode, digest_model
from flameox.models import ContractModel
from flameox.storage import Workspace


class LifecycleItem(ContractModel):
    kind: Literal["span", "event", "transition", "gap", "process"]
    run_id: str | None = None
    artifact_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    name: str | None = None
    source_ordinal: int | None = None
    start_time_unix_nano: int | None = None
    end_time_unix_nano: int | None = None
    time_unix_nano: int | None = None
    duration_ns: int | None = None
    depth: int | None = None
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class LifecycleQueryResult(ContractModel):
    schema_version: int = 1
    operation: str
    corpus_commit_id: str
    evidence_level: Literal["derived"] = "derived"
    run_id: str | None = None
    artifact_id: str | None = None
    query_bounds: dict[str, Any] = Field(default_factory=dict)
    items: tuple[LifecycleItem, ...]
    total: int
    returned: int
    truncated: bool
    next_cursor: str | None = None
    limitations: tuple[str, ...] = ()


class LifecycleEvidenceService:
    """Expose reviewed, bounded lifecycle queries over normalized evidence."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def get_operation_window(
        self,
        *,
        artifact_id: str,
        start_ns: int,
        end_ns: int,
        trace_id: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> LifecycleQueryResult:
        bounded = self._limit(limit)
        if end_ns <= start_ns:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED, "end_ns must be greater than start_ns."
            )
        return self._query_spans(
            operation="get_operation_window",
            artifact_id=artifact_id,
            limit=bounded,
            cursor=cursor,
            parameters={"start_ns": start_ns, "end_ns": end_ns, "trace_id": trace_id},
            where=(
                "artifact_id = ? AND start_time_unix_nano > 0 AND end_time_unix_nano > 0 "
                "AND start_time_unix_nano < ? AND end_time_unix_nano > ?"
            ),
            values=(artifact_id, end_ns, start_ns),
            select="kind, run_id, artifact_id, trace_id, span_id, parent_span_id, name, "
            "source_ordinal, start_time_unix_nano, end_time_unix_nano, duration_ns",
            mapper=lambda row: LifecycleItem(
                kind="span",
                run_id=row[1],
                artifact_id=row[2],
                trace_id=row[3],
                span_id=row[4],
                parent_span_id=row[5],
                name=row[6],
                source_ordinal=row[7],
                start_time_unix_nano=row[8],
                end_time_unix_nano=row[9],
                duration_ns=row[10],
            ),
            trace_id=trace_id,
            limitations=("spans_with_missing_timestamps_are_excluded",),
        )

    def get_operation_transitions(
        self,
        *,
        artifact_id: str,
        trace_id: str | None = None,
        max_depth: int = 8,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> LifecycleQueryResult:
        bounded = self._limit(limit)
        if max_depth < 0 or max_depth > 32:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED, "max_depth must be between 0 and 32."
            )
        head = self.workspace.corpus.read_head()
        scope = {"artifact_id": artifact_id, "trace_id": trace_id, "max_depth": max_depth}
        digest = digest_model(scope)
        after = self._cursor(cursor, "operation_transitions", head.commit_id, digest)
        where = "s.artifact_id = ?"
        parameters: list[object] = [artifact_id]
        if trace_id is not None:
            where += " AND s.trace_id = ?"
            parameters.append(trace_id)
        after_sql = ""
        if after is not None:
            after_sql = " AND s.source_ordinal > ?"
        query = f"""
            WITH RECURSIVE ancestry AS (
                SELECT s.trace_id, s.span_id, s.parent_span_id, s.name, s.source_ordinal,
                       0 AS depth, s.run_id, s.artifact_id
                FROM otel_spans s
                WHERE {where} AND (s.parent_span_id IS NULL OR s.parent_span_id = '')
                UNION ALL
                SELECT child.trace_id, child.span_id, child.parent_span_id, child.name,
                       child.source_ordinal, parent.depth + 1, child.run_id, child.artifact_id
                FROM otel_spans child
                JOIN ancestry parent ON parent.trace_id = child.trace_id
                    AND parent.span_id = child.parent_span_id
                WHERE child.artifact_id = ? AND parent.depth < ?
            )
            SELECT 'transition' AS kind, run_id, artifact_id, trace_id, span_id,
                   parent_span_id, name, source_ordinal, NULL, NULL, NULL, depth
            FROM ancestry
            WHERE 1 = 1 {after_sql}
            ORDER BY source_ordinal
            LIMIT ?
        """
        parameters.extend([artifact_id, max_depth])
        if after_sql:
            # The outer predicate follows the recursive CTE's child and depth
            # parameters in SQL placeholder order.
            parameters.append(after)
        parameters.append(bounded + 1)
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            rows = snapshot.execute(query, tuple(parameters)).fetchall()
            orphan_rows = snapshot.execute(
                """SELECT count(*) FROM otel_spans child
                   LEFT JOIN otel_spans parent ON parent.artifact_id = child.artifact_id
                     AND parent.trace_id = child.trace_id AND parent.span_id = child.parent_span_id
                   WHERE child.artifact_id = ? AND child.parent_span_id IS NOT NULL
                     AND parent.span_id IS NULL""",
                (artifact_id,),
            ).fetchone()
        selected = rows[:bounded]
        items = tuple(
            LifecycleItem(
                kind="transition",
                run_id=row[1],
                artifact_id=row[2],
                trace_id=row[3],
                span_id=row[4],
                parent_span_id=row[5],
                name=row[6],
                source_ordinal=row[7],
                depth=row[11],
            )
            for row in selected
        )
        return self._result(
            "get_operation_transitions",
            head.commit_id,
            artifact_id,
            scope,
            items,
            len(rows),
            len(rows) > bounded,
            (items[-1].source_ordinal if len(rows) > bounded else None),
            cursor_namespace="operation_transitions",
            digest=digest,
            limitations=("missing_parent_references_are_coverage_gaps",)
            if orphan_rows and orphan_rows[0]
            else (),
        )

    def find_repeated_operation_sequences(
        self,
        *,
        artifact_id: str,
        minimum_repetitions: int = 2,
        limit: int | None = None,
    ) -> LifecycleQueryResult:
        bounded = self._limit(limit)
        if minimum_repetitions < 2 or minimum_repetitions > 100:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED, "minimum_repetitions must be 2..100."
            )
        head = self.workspace.corpus.read_head()
        query = """
            WITH normalized AS (
                SELECT s.*,
                       coalesce(
                           json_extract_string(r.attributes_json, '$."service.name"'), ''
                       ) AS service_identity,
                       coalesce(sc.name, '') AS scope_name
                FROM otel_spans s
                LEFT JOIN otel_resources r
                  ON r.artifact_id = s.artifact_id
                 AND r.resource_ordinal = s.resource_ordinal
                LEFT JOIN otel_scopes sc
                  ON sc.artifact_id = s.artifact_id
                 AND sc.resource_ordinal = s.resource_ordinal
                 AND sc.scope_ordinal = s.scope_ordinal
                WHERE s.artifact_id = ?
            ), signatures AS (
                SELECT *,
                  lag(service_identity) OVER (PARTITION BY trace_id ORDER BY source_ordinal)
                      AS previous_service_identity,
                  lag(scope_name) OVER (PARTITION BY trace_id ORDER BY source_ordinal)
                      AS previous_scope_name,
                  lag(name) OVER (PARTITION BY trace_id ORDER BY source_ordinal) AS previous_name,
                  lag(kind) OVER (PARTITION BY trace_id ORDER BY source_ordinal) AS previous_kind
                FROM normalized
            ), runs AS (
                SELECT *, sum(
                    CASE WHEN service_identity = previous_service_identity
                              AND scope_name = previous_scope_name
                              AND name = previous_name AND kind = previous_kind
                         THEN 0 ELSE 1 END
                ) OVER (PARTITION BY trace_id ORDER BY source_ordinal) AS sequence_group
                FROM signatures
            ), repeated AS (
                SELECT *, count(*) OVER (PARTITION BY trace_id, sequence_group) AS repetition_count
                FROM runs
            )
            SELECT 'span', run_id, artifact_id, trace_id, span_id, parent_span_id, name,
                   source_ordinal, start_time_unix_nano, end_time_unix_nano, duration_ns,
                   repetition_count
            FROM repeated
            WHERE repetition_count >= ?
            ORDER BY source_ordinal LIMIT ?
        """
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            rows = snapshot.execute(
                query, (artifact_id, minimum_repetitions, bounded + 1)
            ).fetchall()
        selected = rows[:bounded]
        items = tuple(
            LifecycleItem(
                kind="span",
                run_id=row[1],
                artifact_id=row[2],
                trace_id=row[3],
                span_id=row[4],
                parent_span_id=row[5],
                name=row[6],
                source_ordinal=row[7],
                start_time_unix_nano=row[8],
                end_time_unix_nano=row[9],
                duration_ns=row[10],
                details={"repetition_count": row[11]},
            )
            for row in selected
        )
        return self._result(
            "find_repeated_operation_sequences",
            head.commit_id,
            artifact_id,
            {"minimum_repetitions": minimum_repetitions, "limit": bounded},
            items,
            len(selected),
            len(rows) > bounded,
            None,
            limitations=("repetition_is_evidence_not_a_loop_verdict",),
        )

    def get_lifecycle_gaps(
        self, *, artifact_id: str, limit: int | None = None
    ) -> LifecycleQueryResult:
        bounded = self._limit(limit)
        head = self.workspace.corpus.read_head()
        query = """
            SELECT 'gap', s.run_id, s.artifact_id, s.trace_id, s.span_id, s.parent_span_id,
                   s.name, s.source_ordinal, s.start_time_unix_nano, s.end_time_unix_nano,
                   s.duration_ns,
                   CASE WHEN s.start_time_unix_nano = 0 OR s.end_time_unix_nano = 0
                        THEN 'missing_timestamp' ELSE 'end_before_start' END
            FROM otel_spans s WHERE s.artifact_id = ?
              AND (s.start_time_unix_nano = 0 OR s.end_time_unix_nano = 0
                   OR s.end_time_unix_nano < s.start_time_unix_nano)
            UNION ALL
            SELECT 'gap', child.run_id, child.artifact_id, child.trace_id, child.span_id,
                   child.parent_span_id, child.name, child.source_ordinal,
                   child.start_time_unix_nano, child.end_time_unix_nano, child.duration_ns,
                   'missing_parent'
            FROM otel_spans child LEFT JOIN otel_spans parent
              ON parent.artifact_id = child.artifact_id AND parent.trace_id = child.trace_id
             AND parent.span_id = child.parent_span_id
            WHERE child.artifact_id = ? AND child.parent_span_id IS NOT NULL
              AND parent.span_id IS NULL
            UNION ALL
            SELECT 'gap', s.run_id, s.artifact_id, s.trace_id, s.span_id, s.parent_span_id,
                   s.name, s.source_ordinal, s.start_time_unix_nano, s.end_time_unix_nano,
                   s.duration_ns, 'duplicate_identity'
            FROM otel_spans s
            JOIN (
                SELECT artifact_id, trace_id, span_id
                FROM otel_spans
                WHERE artifact_id = ?
                GROUP BY artifact_id, trace_id, span_id
                HAVING count(*) > 1
            ) duplicates ON duplicates.artifact_id = s.artifact_id
               AND duplicates.trace_id = s.trace_id AND duplicates.span_id = s.span_id
            WHERE s.artifact_id = ?
            ORDER BY source_ordinal LIMIT ?
        """
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            rows = snapshot.execute(
                query, (artifact_id, artifact_id, artifact_id, artifact_id, bounded + 1)
            ).fetchall()
        items = tuple(
            LifecycleItem(
                kind="gap",
                run_id=row[1],
                artifact_id=row[2],
                trace_id=row[3],
                span_id=row[4],
                parent_span_id=row[5],
                name=row[6],
                source_ordinal=row[7],
                start_time_unix_nano=row[8],
                end_time_unix_nano=row[9],
                duration_ns=row[10],
                reason=row[11],
            )
            for row in rows[:bounded]
        )
        return self._result(
            "get_lifecycle_gaps",
            head.commit_id,
            artifact_id,
            {},
            items,
            len(rows),
            len(rows) > bounded,
            None,
        )

    def get_process_snapshot(
        self, *, run_id: str, phase: str | None = None, limit: int | None = None
    ) -> LifecycleQueryResult:
        bounded = self._limit(limit)
        head = self.workspace.corpus.read_head()
        where = "s.run_id = ?"
        values: list[object] = [run_id]
        if phase is not None:
            where += " AND s.phase = ?"
            values.append(phase)
        query = f"""SELECT e.snapshot_id, e.pid, e.create_time, e.parent_pid, e.name,
                           e.status, e.rss_bytes, e.observed_at, s.artifact_id, s.phase
                    FROM process_snapshot_entries e JOIN process_snapshots s
                      ON s.snapshot_id = e.snapshot_id WHERE {where}
                    ORDER BY e.pid LIMIT ?"""
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            rows = snapshot.execute(query, (*values, bounded + 1)).fetchall()
        items = tuple(
            LifecycleItem(
                kind="process",
                run_id=run_id,
                artifact_id=row[8],
                name=row[4],
                details={
                    "snapshot_id": row[0],
                    "pid": row[1],
                    "create_time": row[2],
                    "parent_pid": row[3],
                    "status": row[5],
                    "rss_bytes": row[6],
                    "observed_at": str(row[7]),
                    "phase": row[9],
                },
            )
            for row in rows[:bounded]
        )
        return self._result(
            "get_process_snapshot",
            head.commit_id,
            None,
            {"phase": phase},
            items,
            len(rows),
            len(rows) > bounded,
            None,
        )

    def _query_spans(
        self,
        *,
        operation: str,
        artifact_id: str,
        limit: int,
        cursor: str | None,
        parameters: dict[str, Any],
        where: str,
        values: tuple[object, ...],
        select: str,
        mapper: Any,
        trace_id: str | None,
        limitations: tuple[str, ...] = (),
    ) -> LifecycleQueryResult:
        head = self.workspace.corpus.read_head()
        digest = digest_model({"artifact_id": artifact_id, **parameters})
        after = self._cursor(cursor, operation, head.commit_id, digest)
        base_predicate = where
        base_values = list(values)
        if trace_id is not None:
            base_predicate += " AND trace_id = ?"
            base_values.append(trace_id)
        predicate = base_predicate
        query_values = list(base_values)
        if after is not None:
            predicate += " AND source_ordinal > ?"
            query_values.append(after)
        query = f"SELECT {select} FROM otel_spans WHERE {predicate} ORDER BY source_ordinal LIMIT ?"
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            total_row = snapshot.execute(
                f"SELECT count(*) FROM otel_spans WHERE {base_predicate}",
                tuple(base_values),
            ).fetchone()
            rows = snapshot.execute(query, (*query_values, limit + 1)).fetchall()
        items = tuple(mapper(row) for row in rows[:limit])
        return self._result(
            operation,
            head.commit_id,
            artifact_id,
            parameters,
            items,
            int(total_row[0]) if total_row is not None else 0,
            len(rows) > limit,
            items[-1].source_ordinal if len(rows) > limit and items else None,
            cursor_namespace=operation,
            digest=digest,
            limitations=limitations,
        )

    def _result(
        self,
        operation: str,
        commit: str,
        artifact: str | None,
        bounds: dict[str, Any],
        items: tuple[LifecycleItem, ...],
        total: int,
        truncated: bool,
        position: int | None,
        *,
        cursor_namespace: str | None = None,
        digest: str | None = None,
        limitations: tuple[str, ...] = (),
    ) -> LifecycleQueryResult:
        next_cursor = None
        if truncated and position is not None and cursor_namespace and digest:
            next_cursor = CursorCodec.encode(
                namespace=cursor_namespace,
                snapshot_id=commit,
                scope_digest=digest,
                position=(position,),
            )
        run_id = next((item.run_id for item in items if item.run_id), None)
        return LifecycleQueryResult(
            operation=operation,
            corpus_commit_id=commit,
            run_id=run_id,
            artifact_id=artifact,
            query_bounds=bounds,
            items=items,
            total=total,
            returned=len(items),
            truncated=truncated,
            next_cursor=next_cursor,
            limitations=limitations,
        )

    def _cursor(self, cursor: str | None, namespace: str, snapshot: str, digest: str) -> int | None:
        if cursor is None:
            return None
        values = CursorCodec.decode(
            cursor, namespace=namespace, snapshot_id=snapshot, scope_digest=digest
        )
        if len(values) != 1 or not isinstance(values[0], int):
            raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.")
        return values[0]

    def _limit(self, value: int | None) -> int:
        bounded = value or self.workspace.config.analysis.default_row_limit
        if bounded < 1 or bounded > min(1_000, self.workspace.config.analysis.max_row_limit):
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Result limit is outside the allowed range.",
            )
        return bounded
