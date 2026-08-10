from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from flameox.catalog import Catalog
from flameox.domain import CursorCodec, DomainError, ErrorCode, EvidenceLevel, digest_model
from flameox.domain.scalars import NumericValue
from flameox.evidence import numeric_value_from_columns
from flameox.evidence_status import (
    EvidenceAvailability,
    available_availability,
    empty_availability,
    unavailable_availability,
)
from flameox.models import ContractModel
from flameox.storage import RunStore, Workspace


class MeasurementItem(ContractModel):
    measurement_id: str
    run_id: str
    artifact_id: str | None
    name: str
    value: NumericValue | None
    unit: str
    aggregation: str
    scope: str
    worker_id: str | None
    worker_run_index: int | None
    value_index: int | None
    loop_count: int | None
    is_warmup: bool
    block_id: str | None
    variant_id: str | None
    order_in_block: int | None
    phase: str | None
    dimensions: dict[str, str] = Field(default_factory=dict)
    evidence_level: EvidenceLevel


class MeasurementQueryResult(ContractModel):
    schema_version: Literal[2] = 2
    corpus_commit_id: str
    measurements: tuple[MeasurementItem, ...]
    total: int
    returned: int
    truncated: bool
    next_cursor: str | None
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class InferenceRequestItem(ContractModel):
    request_id: str
    run_id: str
    artifact_id: str
    source_request_id: str
    provider_request_id: str | None
    input_tokens: int
    output_tokens: int
    scheduled_ns: int | None
    observed_started_ns: int | None
    ttft_ns: int | None
    latency_ns: int | None
    tpot_ns: int | None
    mean_itl_ns: int | None
    success: bool | None
    cancelled: bool | None
    error_type: str | None
    error_code: str | None
    queue_ns: int | None
    prefill_ns: int | None
    decode_ns: int | None
    cache_hit: bool | None
    prefix_hash_count: int | None
    evidence_level: EvidenceLevel

    @model_validator(mode="after")
    def outcome_is_coherent(self) -> InferenceRequestItem:
        if self.success is True and (
            self.cancelled is not False
            or self.error_type is not None
            or self.error_code is not None
        ):
            raise ValueError("a successful request cannot be cancelled or carry an error")
        if self.cancelled is True and self.success is not False:
            raise ValueError("a cancelled request must be unsuccessful")
        if (
            self.error_type is not None or self.error_code is not None
        ) and self.success is not False:
            raise ValueError("request errors require a failed outcome")
        return self


class InferenceRequestQueryResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    requests: tuple[InferenceRequestItem, ...]
    total: int
    returned: int
    truncated: bool
    next_cursor: str | None
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class EvidenceQueryService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)

    def measurements(
        self,
        *,
        run_id: str | None = None,
        artifact_id: str | None = None,
        name_prefix: str | None = None,
        include_warmups: bool = False,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> MeasurementQueryResult:
        bounded = limit or self.workspace.config.analysis.default_row_limit
        if bounded < 1 or bounded > self.workspace.config.analysis.max_row_limit:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Limit must be between 1 and {self.workspace.config.analysis.max_row_limit}.",
            )
        head = self.workspace.corpus.read_head()
        scope_digest = digest_model(
            {
                "run_id": run_id,
                "artifact_id": artifact_id,
                "name_prefix": name_prefix,
                "include_warmups": include_warmups,
            }
        )
        after = (
            CursorCodec.decode(
                cursor,
                namespace="measurements",
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
            )[0]
            if cursor
            else None
        )
        if after is not None and not isinstance(after, str):
            raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.")
        predicates: list[str] = ["1 = 1"]
        parameters: list[object] = []
        if run_id is not None:
            predicates.append("run_id = ?")
            parameters.append(run_id)
        if artifact_id is not None:
            predicates.append("artifact_id = ?")
            parameters.append(artifact_id)
        published_where = " AND ".join(predicates)
        published_parameters = tuple(parameters)
        if not include_warmups:
            predicates.append("is_warmup = false")
        if name_prefix is not None:
            predicates.append("name LIKE ? ESCAPE '^'")
            escaped = name_prefix.replace("^", "^^").replace("%", "^%").replace("_", "^_")
            parameters.append(f"{escaped}%")
        where = " AND ".join(predicates)
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            count_row = snapshot.execute(
                "SELECT count(*) FROM measurements WHERE " + where,
                tuple(parameters),
            ).fetchone()
            assert count_row is not None
            published_count_row = snapshot.execute(
                "SELECT count(*) FROM measurements WHERE " + published_where,
                published_parameters,
            ).fetchone()
            assert published_count_row is not None
            page_where = where
            page_parameters = list(parameters)
            if after is not None:
                page_where += " AND measurement_id > ?"
                page_parameters.append(after)
            rows = snapshot.execute(
                "SELECT measurement_id, run_id, artifact_id, name, value_int, "
                "value_float, unit, aggregation, scope, worker_id, "
                "worker_run_index, value_index, loop_count, is_warmup, block_id, "
                "variant_id, order_in_block, phase, dimensions, evidence_level "
                "FROM measurements WHERE " + page_where + " ORDER BY measurement_id LIMIT ?",
                (*page_parameters, bounded + 1),
            ).fetchall()
        has_more = len(rows) > bounded
        selected = rows[:bounded]
        measurements = tuple(
            MeasurementItem(
                measurement_id=row[0],
                run_id=row[1],
                artifact_id=row[2],
                name=row[3],
                value=numeric_value_from_columns(
                    row[4],
                    row[5],
                    field_name="measurement value",
                ),
                unit=row[6],
                aggregation=row[7],
                scope=row[8],
                worker_id=row[9],
                worker_run_index=row[10],
                value_index=row[11],
                loop_count=row[12],
                is_warmup=row[13],
                block_id=row[14],
                variant_id=row[15],
                order_in_block=row[16],
                phase=row[17],
                dimensions=dict(row[18] or {}),
                evidence_level=row[19],
            )
            for row in selected
        )
        next_cursor = (
            CursorCodec.encode(
                namespace="measurements",
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
                position=(measurements[-1].measurement_id,),
            )
            if has_more and measurements
            else None
        )
        return MeasurementQueryResult(
            corpus_commit_id=head.commit_id,
            measurements=measurements,
            total=int(count_row[0]),
            returned=len(measurements),
            truncated=has_more,
            next_cursor=next_cursor,
            evidence=(
                unavailable_availability("measurements_not_published")
                if not measurements and int(published_count_row[0]) == 0
                else (
                    empty_availability("no_matching_measurements")
                    if not measurements
                    else available_availability("measurements_present")
                )
            ),
        )

    def inference_requests(
        self,
        *,
        run_id: str,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> InferenceRequestQueryResult:
        self.runs.read(run_id)
        bounded = limit or self.workspace.config.analysis.default_row_limit
        if bounded < 1 or bounded > self.workspace.config.analysis.max_row_limit:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Limit must be between 1 and {self.workspace.config.analysis.max_row_limit}.",
            )
        head = self.workspace.corpus.read_head()
        scope_digest = digest_model({"run_id": run_id})
        after = (
            CursorCodec.decode(
                cursor,
                namespace="inference_requests",
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
            )[0]
            if cursor
            else None
        )
        if after is not None and not isinstance(after, str):
            raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.")
        where = "run_id = ?"
        parameters: list[object] = [run_id]
        if after is not None:
            where += " AND request_id > ?"
            parameters.append(after)
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            count_row = snapshot.execute(
                "SELECT COUNT(*) FROM inference_requests WHERE run_id = ?", (run_id,)
            ).fetchone()
            assert count_row is not None
            rows = snapshot.execute(
                "SELECT request_id, run_id, artifact_id, source_request_id, "
                "provider_request_id, input_tokens, output_tokens, scheduled_ns, "
                "observed_started_ns, ttft_ns, latency_ns, tpot_ns, mean_itl_ns, success, "
                "cancelled, error_type, error_code, queue_ns, prefill_ns, decode_ns, cache_hit, "
                "prefix_hash_count, evidence_level FROM inference_requests WHERE "
                + where
                + " ORDER BY request_id LIMIT ?",
                (*parameters, bounded + 1),
            ).fetchall()
        has_more = len(rows) > bounded
        requests = tuple(
            InferenceRequestItem.model_validate(
                dict(
                    zip(
                        InferenceRequestItem.model_fields,
                        row,
                        strict=True,
                    )
                )
            )
            for row in rows[:bounded]
        )
        next_cursor = (
            CursorCodec.encode(
                namespace="inference_requests",
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
                position=(requests[-1].request_id,),
            )
            if has_more and requests
            else None
        )
        return InferenceRequestQueryResult(
            corpus_commit_id=head.commit_id,
            requests=requests,
            total=int(count_row[0]),
            returned=len(requests),
            truncated=has_more,
            next_cursor=next_cursor,
            evidence=(
                available_availability("inference_requests_present")
                if int(count_row[0]) > 0
                else unavailable_availability("inference_requests_not_published")
            ),
        )
