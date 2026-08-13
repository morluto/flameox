from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

from pydantic import ConfigDict, Field, computed_field, model_validator

from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactKind,
    CaptureStatus,
    CursorNamespace,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ResourceAvailability,
    RunType,
    ValidationStatus,
    digest_model,
)
from flameox.models import ContractModel
from flameox.pagination import CursorPageContract
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
    execution_status: tuple[ExecutionStatus, ...] = Field(default=(), max_length=8)
    validation_status: tuple[ValidationStatus, ...] = Field(default=(), max_length=9)
    created_after: datetime | None = None
    created_before: datetime | None = None


class RunSummary(ContractModel):
    run_id: str
    created_at: datetime
    run_type: RunType
    execution_status: ExecutionStatus
    capture_status: CaptureStatus
    validation_status: ValidationStatus
    source_state_id: str | None
    environment_id: str
    workload_definition_id: str | None
    orchestrator: str | None
    provider: str | None
    lease_id: str | None
    worker_id: str | None
    orchestration_run_id: str | None
    artifact_kinds: tuple[ArtifactKind, ...]
    resource_availability: ResourceAvailability = ResourceAvailability.UNAVAILABLE


class DiscoveryCoverage(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    filters_applied: tuple[str, ...]
    unavailable_facets: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def parse_population_projection(cls, value: Any) -> Any:
        if not isinstance(value, Mapping) or "population_complete" not in value:
            return value
        parsed = dict(value)
        supplied = parsed.pop("population_complete")
        unavailable = parsed.get("unavailable_facets", ())
        if isinstance(unavailable, (list, tuple)) and supplied != (not unavailable):
            raise ValueError("population completeness must agree with unavailable facets")
        return parsed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def population_complete(self) -> bool:
        return not self.unavailable_facets


class RunListResult(CursorPageContract):
    page_items_field = "runs"

    schema_version: int = 1
    corpus_commit_id: str
    runs: tuple[RunSummary, ...]
    total: int
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
            position = cast(
                tuple[str, str],
                self.workspace.cursors.resolve(
                    cursor,
                    namespace=CursorNamespace.RUNS,
                    snapshot_id=head.commit_id,
                    scope_digest=scope_digest,
                ),
            )
            try:
                after_created = datetime.fromisoformat(position[0])
            except ValueError as exc:
                raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.") from exc
            after_run_id = position[1]

        predicates = ["1 = 1"]
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
                parameters.extend(value.value for value in values)
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
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            count_row = snapshot.execute(
                "SELECT count(*) FROM current_runs WHERE " + where,
                tuple(parameters),
            ).fetchone()
            assert count_row is not None
            page_where = where
            page_parameters = list(parameters)
            if after_created is not None and after_run_id is not None:
                page_where += " AND (created_at < ? OR (created_at = ? AND run_id > ?))"
                page_parameters.extend((after_created, after_created, after_run_id))
            rows = snapshot.execute(
                "SELECT run_id, created_at, run_type, execution_status, capture_status, "
                "validation_status, source_state_id, environment_id, workload_definition_id "
                ", orchestrator, provider, lease_id, worker_id, orchestration_run_id "
                ", coalesce((SELECT list_sort(list_distinct(list(kind))) "
                "FROM artifact_registrations WHERE run_id = current_runs.run_id), "
                "CAST([] AS VARCHAR[])) AS artifact_kinds, "
                "coalesce(resource_availability, 'unavailable') AS resource_availability "
                "FROM current_runs WHERE "
                + page_where
                + " ORDER BY created_at DESC, run_id LIMIT ?",
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
                resource_availability=row[15],
            )
            for row in rows[:limit]
        )
        next_cursor = (
            self.workspace.cursors.issue(
                namespace=CursorNamespace.RUNS,
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
            next_cursor=next_cursor,
            coverage=DiscoveryCoverage(filters_applied=tuple(applied)),
        )
