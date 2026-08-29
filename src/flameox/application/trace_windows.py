from __future__ import annotations

import json
from typing import cast

from pydantic import Field, JsonValue

from flameox.action_graph import ActionId, NextAction, tool_action
from flameox.catalog import Catalog
from flameox.domain import (
    CursorNamespace,
    DomainError,
    ErrorCode,
    digest_model,
)
from flameox.evidence_status import (
    EvidenceAvailability,
    available_availability,
    empty_availability,
)
from flameox.models import ContractModel
from flameox.pagination import CursorPageContract
from flameox.storage import Workspace


class TraceEvent(ContractModel):
    event_id: str
    name: str
    category: str | None
    start_ns: int
    end_ns: int | None = None
    duration_ns: int
    provider: str
    parent_id: str | None = None
    track_id: int | None = None
    phase: str | None = None
    process: JsonValue = None
    thread: JsonValue = None
    device: JsonValue = None
    context: JsonValue = None
    stream: JsonValue = None
    correlation_id: JsonValue = None
    event_kind: str | None = None
    event_type: int | None = None


class TraceWindowResult(CursorPageContract):
    page_items_field = "events"

    artifact_id: str
    start_ns: int
    end_ns: int
    events: tuple[TraceEvent, ...]
    total: int
    coverage: float
    provider: str
    limitations: tuple[str, ...]
    recovery: NextAction | None = None
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class TraceWindowService:
    """Route bounded windows to normalized evidence before requiring a native reader."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def get(
        self,
        artifact_id: str,
        *,
        start_ns: int,
        end_ns: int,
        limit: int = 100,
        cursor: str | None = None,
        run_id: str | None = None,
    ) -> TraceWindowResult:
        self._validate_bounds(start_ns=start_ns, end_ns=end_ns, limit=limit)
        scope_digest = digest_model(
            {"artifact_id": artifact_id, "start_ns": start_ns, "end_ns": end_ns}
        )
        if cursor is not None:
            if self.workspace.cursors.namespace(cursor) is CursorNamespace.TRACE_WINDOW:
                from flameox.adapters.perfetto import PerfettoExtractor

                return await PerfettoExtractor(self.workspace).trace_window(
                    artifact_id,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    limit=limit,
                    cursor=cursor,
                )
            position = self.workspace.cursors.resolve(
                cursor,
                namespace=CursorNamespace.NORMALIZED_TRACE_WINDOW,
                snapshot_id=artifact_id,
                scope_digest=scope_digest,
            )
            commit_id, cursor_run_id, after_start, after_id = cast(
                tuple[str, str, int, str], position
            )
            if run_id is not None and run_id != cursor_run_id:
                raise DomainError(ErrorCode.STALE_CURSOR, "Cursor belongs to another run.")
            return self._normalized(
                artifact_id,
                commit_id=commit_id,
                run_id=cursor_run_id,
                start_ns=start_ns,
                end_ns=end_ns,
                limit=limit,
                after_start=after_start,
                after_id=after_id,
                scope_digest=scope_digest,
            )
        commit_id = self.workspace.corpus.read_head().commit_id
        with Catalog(self.workspace).open_snapshot(commit_id) as snapshot:
            query = (
                "SELECT DISTINCT run_id FROM observations "
                "WHERE artifact_id = ? AND kind = 'trace.extraction'"
            )
            parameters: tuple[object, ...] = (artifact_id,)
            if run_id is not None:
                query += " AND run_id = ?"
                parameters = (artifact_id, run_id)
            extraction_runs = tuple(
                str(row[0]) for row in snapshot.execute(query, parameters).fetchall()
            )
        if len(extraction_runs) > 1:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Trace artifact has normalized evidence for multiple runs; run_id is required.",
                details={"run_ids": list(extraction_runs)},
            )
        if extraction_runs:
            return self._normalized(
                artifact_id,
                commit_id=commit_id,
                run_id=extraction_runs[0],
                start_ns=start_ns,
                end_ns=end_ns,
                limit=limit,
                after_start=None,
                after_id=None,
                scope_digest=scope_digest,
            )
        from flameox.adapters.perfetto import PerfettoExtractor

        return await PerfettoExtractor(self.workspace).trace_window(
            artifact_id,
            start_ns=start_ns,
            end_ns=end_ns,
            limit=limit,
            cursor=cursor,
        )

    def _normalized(
        self,
        artifact_id: str,
        *,
        commit_id: str,
        run_id: str,
        start_ns: int,
        end_ns: int,
        limit: int,
        after_start: int | None,
        after_id: str | None,
        scope_digest: str,
    ) -> TraceWindowResult:
        start_expression = "CAST(json_extract(value_json, '$.start_ns') AS BIGINT)"
        duration_expression = "CAST(json_extract(value_json, '$.duration_ns') AS BIGINT)"
        predicate = (
            "artifact_id = ? AND run_id = ? AND kind = 'trace.event' "
            f"AND {start_expression} < ? "
            f"AND ({start_expression} + {duration_expression} >= ? "
            "OR json_extract_string(value_json, '$.event_kind') = 'incomplete_range')"
        )
        parameters: list[object] = [artifact_id, run_id, end_ns, start_ns]
        if after_start is not None and after_id is not None:
            predicate += (
                f" AND ({start_expression} > ? OR ({start_expression} = ? AND observation_id > ?))"
            )
            parameters.extend((after_start, after_start, after_id))
        with Catalog(self.workspace).open_snapshot(commit_id) as snapshot:
            count_row = snapshot.execute(
                "SELECT COUNT(*) FROM observations WHERE "
                + (
                    "artifact_id = ? AND run_id = ? AND kind = 'trace.event' "
                    f"AND {start_expression} < ? "
                    f"AND ({start_expression} + {duration_expression} >= ? "
                    "OR json_extract_string(value_json, '$.event_kind') = 'incomplete_range')"
                ),
                (artifact_id, run_id, end_ns, start_ns),
            ).fetchone()
            if count_row is None:
                raise RuntimeError("Trace event count query returned no row.")
            total = int(count_row[0])
            rows = snapshot.execute(
                "SELECT observation_id, name, value_json FROM observations WHERE "
                + predicate
                + f" ORDER BY {start_expression}, observation_id LIMIT ?",
                (*parameters, limit + 1),
            ).fetchall()
            extraction_row = snapshot.execute(
                "SELECT run_id, value_json FROM observations "
                "WHERE artifact_id = ? AND run_id = ? AND kind = 'trace.extraction' "
                "ORDER BY observation_id LIMIT 1",
                (artifact_id, run_id),
            ).fetchone()
        selected = rows[:limit]
        events = tuple(self._event(row) for row in selected)
        has_more = len(rows) > limit
        next_cursor = (
            self.workspace.cursors.issue(
                namespace=CursorNamespace.NORMALIZED_TRACE_WINDOW,
                snapshot_id=artifact_id,
                scope_digest=scope_digest,
                position=(commit_id, run_id, events[-1].start_ns, events[-1].event_id),
            )
            if has_more and events
            else None
        )
        extraction_payload = (
            json.loads(str(extraction_row[1])) if extraction_row is not None else {}
        )
        if not isinstance(extraction_payload, dict):
            extraction_payload = {}
        truncated_tables = extraction_payload.get("truncated_tables", [])
        limitations = [
            "Events overlap the requested interval in the Nsight Systems export clock domain.",
            "Cross-provider clock alignment is unavailable.",
        ]
        recovery: NextAction | None = None
        if truncated_tables and extraction_row is not None:
            limitations.append(
                "Normalized evidence was truncated for: "
                + ", ".join(str(item) for item in truncated_tables)
                + "."
            )
            recovery = tool_action(ActionId.EXTRACT_NSIGHT_SYSTEMS, run_id=str(extraction_row[0]))
        return TraceWindowResult(
            artifact_id=artifact_id,
            start_ns=start_ns,
            end_ns=end_ns,
            events=events,
            total=total,
            coverage=(len(events) / total if total else 0.0),
            provider="nsight.systems",
            limitations=tuple(limitations),
            recovery=recovery,
            next_cursor=next_cursor,
            evidence=(
                available_availability() if events else empty_availability("no_events_in_window")
            ),
        )

    @staticmethod
    def _event(row: tuple[object, ...]) -> TraceEvent:
        payload = json.loads(str(row[2]))
        return TraceEvent(
            event_id=str(row[0]),
            name=str(row[1]),
            category=_optional_string(payload.get("category")),
            start_ns=int(payload["start_ns"]),
            end_ns=(int(payload["end_ns"]) if payload.get("end_ns") is not None else None),
            duration_ns=int(payload["duration_ns"]),
            provider="nsight.systems",
            phase=_optional_string(payload.get("phase")),
            process=payload.get("process"),
            thread=payload.get("thread"),
            device=payload.get("device"),
            context=payload.get("context"),
            stream=payload.get("stream"),
            correlation_id=payload.get("correlation_id"),
            event_kind=_optional_string(payload.get("event_kind")),
            event_type=(
                int(payload["event_type"]) if payload.get("event_type") is not None else None
            ),
        )

    def _validate_bounds(self, *, start_ns: int, end_ns: int, limit: int) -> None:
        if start_ns < 0 or end_ns <= start_ns:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Trace window requires 0 <= start_ns < end_ns.",
            )
        maximum = self.workspace.config.analysis.max_row_limit
        if limit < 1 or limit > maximum:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Limit must be between 1 and {maximum}.",
            )


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
