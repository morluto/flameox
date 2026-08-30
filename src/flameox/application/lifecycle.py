from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, cast

from pydantic import Field

from flameox.action_graph import ActionId, NextAction, manual_action
from flameox.catalog import Catalog
from flameox.domain import CursorNamespace, DomainError, ErrorCode, digest_model
from flameox.evidence_status import (
    EvidenceAvailability,
    EvidenceStatus,
    available_availability,
    empty_availability,
    partial_availability,
    unavailable_availability,
)
from flameox.models import ContractModel
from flameox.pagination import CursorPageContract
from flameox.storage import RunStore, Workspace


class LifecycleItemKind(StrEnum):
    SPAN = "span"
    EVENT = "event"
    TRANSITION = "transition"
    GAP = "gap"
    PROCESS = "process"


class _LifecycleItem(ContractModel):
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


class SpanLifecycleItem(_LifecycleItem):
    kind: Literal[LifecycleItemKind.SPAN] = LifecycleItemKind.SPAN
    run_id: str
    artifact_id: str
    trace_id: str
    span_id: str
    name: str
    source_ordinal: int
    start_time_unix_nano: int
    end_time_unix_nano: int
    time_unix_nano: Literal[None] = None
    depth: Literal[None] = None
    reason: Literal[None] = None


class EventLifecycleItem(_LifecycleItem):
    kind: Literal[LifecycleItemKind.EVENT] = LifecycleItemKind.EVENT
    run_id: str
    artifact_id: str
    trace_id: str
    span_id: str
    parent_span_id: Literal[None] = None
    name: str
    start_time_unix_nano: Literal[None] = None
    end_time_unix_nano: Literal[None] = None
    time_unix_nano: int
    duration_ns: Literal[None] = None
    depth: Literal[None] = None
    reason: Literal[None] = None


class TransitionLifecycleItem(_LifecycleItem):
    kind: Literal[LifecycleItemKind.TRANSITION] = LifecycleItemKind.TRANSITION
    run_id: str
    artifact_id: str
    trace_id: str
    span_id: str
    name: str
    source_ordinal: int
    start_time_unix_nano: Literal[None] = None
    end_time_unix_nano: Literal[None] = None
    time_unix_nano: Literal[None] = None
    duration_ns: Literal[None] = None
    depth: int
    reason: Literal[None] = None


class GapLifecycleItem(_LifecycleItem):
    kind: Literal[LifecycleItemKind.GAP] = LifecycleItemKind.GAP
    run_id: str
    artifact_id: str
    trace_id: str
    span_id: str
    name: str
    source_ordinal: int
    start_time_unix_nano: int
    end_time_unix_nano: int
    time_unix_nano: Literal[None] = None
    depth: Literal[None] = None
    reason: str


class ProcessLifecycleItem(_LifecycleItem):
    kind: Literal[LifecycleItemKind.PROCESS] = LifecycleItemKind.PROCESS
    run_id: str
    trace_id: Literal[None] = None
    span_id: Literal[None] = None
    parent_span_id: Literal[None] = None
    source_ordinal: Literal[None] = None
    start_time_unix_nano: Literal[None] = None
    end_time_unix_nano: Literal[None] = None
    time_unix_nano: Literal[None] = None
    duration_ns: Literal[None] = None
    depth: Literal[None] = None
    reason: Literal[None] = None


type LifecycleItem = Annotated[
    SpanLifecycleItem
    | EventLifecycleItem
    | TransitionLifecycleItem
    | GapLifecycleItem
    | ProcessLifecycleItem,
    Field(discriminator="kind"),
]


class LifecycleQueryResult(CursorPageContract):
    page_items_field = "items"

    operation: str
    corpus_commit_id: str
    evidence_level: Literal["derived"] = "derived"
    run_id: str | None = None
    artifact_id: str | None = None
    query_bounds: dict[str, Any] = Field(default_factory=dict)
    items: tuple[LifecycleItem, ...]
    total: int
    next_cursor: str | None = None
    limitations: tuple[str, ...] = ()


class ProcessSnapshotQueryResult(LifecycleQueryResult):
    run_id: str
    evidence: EvidenceAvailability
    next_action: NextAction | None = None


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
            operation=CursorNamespace.GET_OPERATION_WINDOW,
            artifact_id=artifact_id,
            limit=bounded,
            cursor=cursor,
            parameters={"start_ns": start_ns, "end_ns": end_ns, "trace_id": trace_id},
            where=(
                "artifact_id = ? AND start_time_unix_nano > 0 AND end_time_unix_nano > 0 "
                "AND end_time_unix_nano >= start_time_unix_nano "
                "AND start_time_unix_nano < ? AND end_time_unix_nano > ?"
            ),
            values=(artifact_id, end_ns, start_ns),
            select="kind, run_id, artifact_id, trace_id, span_id, parent_span_id, name, "
            "source_ordinal, start_time_unix_nano, end_time_unix_nano, duration_ns",
            mapper=lambda row: SpanLifecycleItem(
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
        after = self._cursor(
            cursor,
            CursorNamespace.OPERATION_TRANSITIONS,
            head.commit_id,
            digest,
        )
        where = "s.artifact_id = ?"
        parameters: list[object] = [artifact_id]
        if trace_id is not None:
            where += " AND s.trace_id = ?"
            parameters.append(trace_id)
        after_sql = ""
        if after is not None:
            after_sql = " AND ancestry.source_ordinal > ?"
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
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            rows = snapshot.execute(query, tuple(parameters)).fetchall()
            orphan_query = """SELECT count(*) FROM otel_spans child
                   LEFT JOIN otel_spans parent ON parent.artifact_id = child.artifact_id
                     AND parent.trace_id = child.trace_id AND parent.span_id = child.parent_span_id
                   WHERE child.artifact_id = ? AND child.parent_span_id IS NOT NULL
                     AND parent.span_id IS NULL"""
            orphan_values: list[object] = [artifact_id]
            if trace_id is not None:
                orphan_query += " AND child.trace_id = ?"
                orphan_values.append(trace_id)
            orphan_rows = snapshot.execute(orphan_query, tuple(orphan_values)).fetchone()
        selected = rows[:bounded]
        items = tuple(
            TransitionLifecycleItem(
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
            cursor_namespace=CursorNamespace.OPERATION_TRANSITIONS,
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
        cursor: str | None = None,
    ) -> LifecycleQueryResult:
        bounded = self._limit(limit)
        if minimum_repetitions < 2 or minimum_repetitions > 100:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED, "minimum_repetitions must be 2..100."
            )
        head = self.workspace.corpus.read_head()
        bounds = {
            "minimum_repetitions": minimum_repetitions,
            "limit": bounded,
        }
        digest = digest_model({"artifact_id": artifact_id, **bounds})
        after = self._cursor(
            cursor,
            CursorNamespace.FIND_REPEATED_OPERATION_SEQUENCES,
            head.commit_id,
            digest,
        )
        cte = """
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
        """
        select_sql = """
            SELECT 'span', run_id, artifact_id, trace_id, span_id, parent_span_id, name,
                   source_ordinal, start_time_unix_nano, end_time_unix_nano, duration_ns,
                   repetition_count
            FROM repeated
        """
        query = cte + select_sql + " WHERE repetition_count >= ?"
        query_values: list[object] = [artifact_id, minimum_repetitions]
        if after is not None:
            query += " AND source_ordinal > ?"
            query_values.append(after)
        query += " ORDER BY source_ordinal LIMIT ?"
        query_values.append(bounded + 1)
        count_query = cte + " SELECT count(*) FROM repeated WHERE repetition_count >= ?"
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            total_row = snapshot.execute(count_query, (artifact_id, minimum_repetitions)).fetchone()
            rows = snapshot.execute(query, tuple(query_values)).fetchall()
        selected = rows[:bounded]
        items = tuple(
            SpanLifecycleItem(
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
            bounds,
            items,
            int(total_row[0]) if total_row else 0,
            len(rows) > bounded,
            items[-1].source_ordinal if len(rows) > bounded and items else None,
            cursor_namespace=CursorNamespace.FIND_REPEATED_OPERATION_SEQUENCES,
            digest=digest,
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
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            rows = snapshot.execute(
                query, (artifact_id, artifact_id, artifact_id, artifact_id, bounded + 1)
            ).fetchall()
        items = tuple(
            GapLifecycleItem(
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
        self,
        *,
        run_id: str,
        phase: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> ProcessSnapshotQueryResult:
        bounded = self._limit(limit)
        RunStore(self.workspace).read(run_id)
        head = self.workspace.corpus.read_head()
        where = "s.run_id = ?"
        values: list[object] = [run_id]
        if phase is not None:
            where += " AND s.phase = ?"
            values.append(phase)
        scope_digest = digest_model({"run_id": run_id, "phase": phase})
        position = self._cursor(
            cursor,
            CursorNamespace.GET_PROCESS_SNAPSHOT,
            head.commit_id,
            scope_digest,
        )
        offset = position or 0
        query = f"""SELECT e.snapshot_id, e.pid, e.create_time, e.parent_pid, e.name,
                           e.status, e.rss_bytes, e.observed_at, s.artifact_id, s.phase
                    FROM process_snapshot_entries e JOIN process_snapshots s
                      ON s.snapshot_id = e.snapshot_id WHERE {where}
                    ORDER BY s.phase, e.pid LIMIT ? OFFSET ?"""
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            summary_rows = snapshot.execute(
                f"""SELECT artifact_id, evidence_status, limitations, entry_count
                     FROM process_snapshots s WHERE {where} ORDER BY phase""",
                tuple(values),
            ).fetchall()
            rows = snapshot.execute(query, (*values, bounded, offset)).fetchall()
            entry_count_row = snapshot.execute(
                f"""SELECT count(*) FROM process_snapshot_entries e
                     JOIN process_snapshots s ON s.snapshot_id = e.snapshot_id
                     WHERE {where}""",
                tuple(values),
            ).fetchone()
        items = tuple(
            ProcessLifecycleItem(
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
        known_statuses = {
            EvidenceStatus.AVAILABLE.value,
            EvidenceStatus.EMPTY.value,
            EvidenceStatus.PARTIAL.value,
            EvidenceStatus.UNAVAILABLE.value,
        }
        raw_statuses = {row[1] for row in summary_rows}
        if not raw_statuses.issubset(known_statuses):
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Process snapshot evidence contains an invalid coverage status.",
                details={"statuses": sorted(str(status) for status in raw_statuses)},
                remediation=("Re-extract or recapture the process evidence.",),
            )
        statuses = {str(status) for status in raw_statuses}
        evidence_status: EvidenceStatus
        if not statuses or "unavailable" in statuses:
            evidence_status = EvidenceStatus.UNAVAILABLE
        elif "partial" in statuses:
            evidence_status = EvidenceStatus.PARTIAL
        elif statuses == {"empty"}:
            evidence_status = EvidenceStatus.EMPTY
        else:
            evidence_status = EvidenceStatus.AVAILABLE
        limitations = tuple(sorted({str(item) for row in summary_rows for item in (row[2] or [])}))
        if not summary_rows:
            limitations = ("no_process_snapshot_evidence",)
        next_action = (
            manual_action(
                "Capture a new run after granting process-enumeration visibility.",
                suggested_action=ActionId.PLAN_CAPTURE,
                missing_arguments=("workload_name", "adapter", "mode"),
            )
            if evidence_status in {EvidenceStatus.PARTIAL, EvidenceStatus.UNAVAILABLE}
            else None
        )
        evidence = (
            available_availability("process_observation_complete")
            if evidence_status is EvidenceStatus.AVAILABLE
            else (
                empty_availability("complete_process_observation_was_empty")
                if evidence_status is EvidenceStatus.EMPTY
                else (
                    partial_availability("process_observation_incomplete")
                    if evidence_status is EvidenceStatus.PARTIAL
                    else unavailable_availability("process_visibility_unavailable")
                )
            )
        )
        total = sum(int(row[3]) for row in summary_rows)
        observed_total = int(entry_count_row[0]) if entry_count_row is not None else 0
        if observed_total != total:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Process snapshot summary count does not match its preserved entries.",
                details={"summary_count": total, "entry_count": observed_total},
                remediation=("Re-extract or recapture the process evidence.",),
            )
        next_cursor = (
            self.workspace.cursors.issue(
                namespace=CursorNamespace.GET_PROCESS_SNAPSHOT,
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
                position=(offset + len(items),),
            )
            if offset + len(items) < total
            else None
        )
        return ProcessSnapshotQueryResult(
            operation="get_process_snapshot",
            corpus_commit_id=head.commit_id,
            run_id=run_id,
            artifact_id=(str(summary_rows[0][0]) if summary_rows else None),
            query_bounds={"phase": phase, "limit": bounded},
            items=items,
            total=total,
            next_cursor=next_cursor,
            limitations=limitations,
            evidence=evidence,
            next_action=next_action,
        )

    def _query_spans(
        self,
        *,
        operation: CursorNamespace,
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
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            total_row = snapshot.execute(
                f"SELECT count(*) FROM otel_spans WHERE {base_predicate}",
                tuple(base_values),
            ).fetchone()
            rows = snapshot.execute(query, (*query_values, limit + 1)).fetchall()
        items = tuple(mapper(row) for row in rows[:limit])
        return self._result(
            operation.value,
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
        cursor_namespace: CursorNamespace | None = None,
        digest: str | None = None,
        limitations: tuple[str, ...] = (),
    ) -> LifecycleQueryResult:
        next_cursor = None
        if truncated and position is not None and cursor_namespace and digest:
            next_cursor = self.workspace.cursors.issue(
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
            next_cursor=next_cursor,
            limitations=limitations,
        )

    def _cursor(
        self,
        cursor: str | None,
        namespace: CursorNamespace,
        snapshot: str,
        digest: str,
    ) -> int | None:
        if cursor is None:
            return None
        values = cast(
            tuple[int],
            self.workspace.cursors.resolve(
                cursor,
                namespace=namespace,
                snapshot_id=snapshot,
                scope_digest=digest,
            ),
        )
        return values[0]

    def _limit(self, value: int | None) -> int:
        bounded = value or self.workspace.config.analysis.default_row_limit
        if bounded < 1 or bounded > min(1_000, self.workspace.config.analysis.max_row_limit):
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                "Result limit is outside the allowed range.",
            )
        return bounded
