from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from flameox.catalog import Catalog
from flameox.domain import CursorCodec, DomainError, ErrorCode, digest_model
from flameox.models import ContractModel
from flameox.storage import Workspace


class RunFilter(ContractModel):
    source_state_id: str | None = None
    environment_id: str | None = None
    workload_definition_id: str | None = None
    orchestrator: str | None = None
    provider: str | None = None
    lease_id: str | None = None
    worker_id: str | None = None
    orchestration_run_id: str | None = None
    execution_status: tuple[
        Literal[
            "pending",
            "running",
            "succeeded",
            "failed",
            "timed_out",
            "cancelled",
            "not_applicable",
        ],
        ...,
    ] = Field(default=(), max_length=8)
    validation_status: tuple[
        Literal["not_requested", "passed", "failed", "inconclusive", "unsupported"],
        ...,
    ] = Field(default=(), max_length=5)
    created_after: datetime | None = None
    created_before: datetime | None = None


class RunSummary(ContractModel):
    run_id: str
    created_at: datetime
    run_type: str
    execution_status: str
    capture_status: str
    validation_status: str
    source_state_id: str | None
    environment_id: str
    workload_definition_id: str | None
    orchestrator: str | None
    provider: str | None
    lease_id: str | None
    worker_id: str | None
    orchestration_run_id: str | None
    artifact_kinds: tuple[str, ...]


class DiscoveryCoverage(ContractModel):
    filters_applied: tuple[str, ...]
    population_complete: bool = True
    unavailable_facets: tuple[str, ...] = ()


class RunListResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    runs: tuple[RunSummary, ...]
    total: int
    returned: int
    truncated: bool
    next_cursor: str | None
    coverage: DiscoveryCoverage


class RunDiscoveryService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def list(
        self,
        *,
        filter: RunFilter,
        limit: int,
        cursor: str | None = None,
    ) -> RunListResult:
        head = self.workspace.corpus.read_head()
        scope_digest = digest_model(filter)
        after_created: datetime | None = None
        after_run_id: str | None = None
        if cursor is not None:
            position = CursorCodec.decode(
                cursor,
                namespace="runs",
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
            )
            if (
                len(position) != 2
                or not isinstance(position[0], str)
                or not isinstance(position[1], str)
            ):
                raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.")
            try:
                after_created = datetime.fromisoformat(position[0])
            except ValueError as exc:
                raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.") from exc
            after_run_id = position[1]

        predicates = ["revision_order = 1"]
        parameters: list[object] = []
        applied: list[str] = []
        for field in (
            "source_state_id",
            "environment_id",
            "workload_definition_id",
            "orchestrator",
            "provider",
            "lease_id",
            "worker_id",
            "orchestration_run_id",
        ):
            value = getattr(filter, field)
            if value is not None:
                predicates.append(f"{field} = ?")
                parameters.append(value)
                applied.append(field)
        for field in ("execution_status", "validation_status"):
            values = getattr(filter, field)
            if values:
                placeholders = ", ".join("?" for _ in values)
                predicates.append(f"{field} IN ({placeholders})")
                parameters.extend(values)
                applied.append(field)
        if filter.created_after is not None:
            predicates.append("created_at >= ?")
            parameters.append(filter.created_after)
            applied.append("created_after")
        if filter.created_before is not None:
            predicates.append("created_at < ?")
            parameters.append(filter.created_before)
            applied.append("created_before")
        where = " AND ".join(predicates)
        latest = (
            "WITH latest AS (SELECT *, row_number() OVER (PARTITION BY run_id "
            "ORDER BY published_at DESC) AS revision_order FROM runs) "
        )
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            count_row = snapshot.execute(
                latest + "SELECT count(*) FROM latest WHERE " + where,
                tuple(parameters),
            ).fetchone()
            assert count_row is not None
            page_where = where
            page_parameters = list(parameters)
            if after_created is not None and after_run_id is not None:
                page_where += " AND (created_at < ? OR (created_at = ? AND run_id > ?))"
                page_parameters.extend((after_created, after_created, after_run_id))
            rows = snapshot.execute(
                latest + "SELECT run_id, created_at, run_type, execution_status, capture_status, "
                "validation_status, source_state_id, environment_id, workload_definition_id "
                ", orchestrator, provider, lease_id, worker_id, orchestration_run_id "
                ", coalesce((SELECT list_sort(list_distinct(list(kind))) "
                "FROM artifact_registrations WHERE run_id = latest.run_id), "
                "CAST([] AS VARCHAR[])) AS artifact_kinds "
                "FROM latest WHERE " + page_where + " ORDER BY created_at DESC, run_id LIMIT ?",
                (*page_parameters, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        runs = tuple(
            RunSummary(
                run_id=row[0],
                created_at=row[1],
                run_type=row[2],
                execution_status=row[3],
                capture_status=row[4],
                validation_status=row[5],
                source_state_id=row[6],
                environment_id=row[7],
                workload_definition_id=row[8],
                orchestrator=row[9],
                provider=row[10],
                lease_id=row[11],
                worker_id=row[12],
                orchestration_run_id=row[13],
                artifact_kinds=tuple(row[14]),
            )
            for row in rows[:limit]
        )
        next_cursor = (
            CursorCodec.encode(
                namespace="runs",
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
                position=(runs[-1].created_at.isoformat(), runs[-1].run_id),
            )
            if has_more and runs
            else None
        )
        return RunListResult(
            corpus_commit_id=head.commit_id,
            runs=runs,
            total=int(count_row[0]),
            returned=len(runs),
            truncated=has_more,
            next_cursor=next_cursor,
            coverage=DiscoveryCoverage(filters_applied=tuple(applied)),
        )
