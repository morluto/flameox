from __future__ import annotations

import json
import math
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Literal, cast

import numpy as np
from pydantic import Field
from scipy.stats import bootstrap, spearmanr
from statsmodels.api import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson

from flameox.catalog import Catalog, Snapshot
from flameox.domain import DomainError, ErrorCode, digest_model
from flameox.evidence_scope import EvidenceScope, resolve_evidence_scope
from flameox.evidence_status import (
    EvidenceAvailability,
    available_availability,
    empty_availability,
)
from flameox.models import ContractModel
from flameox.storage import Workspace


class Hotspot(ContractModel):
    frame_id: str
    function: str | None
    file: str | None
    line: int | None
    metric: str
    self_value: int | None
    inclusive_value: int | None
    unit: str
    sample_count: int | None


class HotspotResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    hotspots: tuple[Hotspot, ...]
    total: int
    returned: int
    truncated: bool
    coverage: dict[str, int]
    limitations: tuple[str, ...]
    evidence_status: Literal["available", "empty", "unavailable", "partial", "unknown"] = (
        "available"
    )
    unavailable_reason: str | None = None
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class MeasurementSummary(ContractModel):
    name: str
    value_int: int | None
    value_float: float | None
    unit: str
    aggregation: str
    scope: str


class MemoryAnalysisResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    measurements: tuple[MeasurementSummary, ...]
    hotspots: tuple[Hotspot, ...]
    phase_growth: tuple[MemoryPhaseGrowth, ...] = ()
    limitations: tuple[str, ...]
    runtime_resources: tuple[RuntimeResourceObservation, ...] = ()
    runtime_resource_totals: RuntimeResourceTotals | None = None
    runtime_resources_truncated: bool = False
    truncated: bool = False
    writable_root_observations: tuple[WritableRootObservation, ...] = ()
    policy_termination: str | None = None
    unavailable_metrics: tuple[str, ...] = ()
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class RuntimeResourceObservation(ContractModel):
    run_id: str
    sampling_interval_ms: int
    minimum_free_bytes: int | None
    staging_growth_bytes: int | None
    peak_rss_bytes: int | None
    policy_termination: str | None
    unavailable_metrics: tuple[str, ...] = ()


class RuntimeResourceTotals(ContractModel):
    run_count: int = 0
    minimum_free_bytes: int | None = None
    total_staging_growth_bytes: int | None = None
    maximum_peak_rss_bytes: int | None = None


class WritableRootObservation(ContractModel):
    run_id: str
    writable_root_identity: str
    target_path: str
    growth_bytes: int | None
    available: bool
    unavailable_reason: str | None = None


class MemoryPhaseGrowth(ContractModel):
    phase: str
    metric: str
    value: float
    previous_value: float | None
    delta: float | None
    unit: str
    sample_count: int


class ExecutionObservation(ContractModel):
    observation_id: str
    kind: str
    name: str
    value_json: str
    file: str | None
    line_from: int | None
    line_to: int | None
    context: str | None
    evidence_level: str


class ExecutionAnalysisResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    observations: tuple[ExecutionObservation, ...]
    comparison_input_id: str | None = None
    added: tuple[ExecutionObservation, ...] = ()
    removed: tuple[ExecutionObservation, ...] = ()
    changed: tuple[ExecutionObservationChange, ...] = ()
    total: int
    returned: int
    truncated: bool
    limitations: tuple[str, ...]
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class ExecutionObservationChange(ContractModel):
    kind: str
    name: str
    file: str | None
    line_from: int | None
    line_to: int | None
    context: str | None
    baseline_value_json: str
    candidate_value_json: str


class OperatorSummary(ContractModel):
    frame_id: str
    operator: str
    category: str | None
    self_cpu_ns: int | None
    total_cpu_ns: int | None
    device_ns: int | None
    inclusive_ns: int
    event_count: int
    input_shapes: tuple[str, ...] = ()
    allocation_bytes: int | None = None
    synchronization: bool = False
    warmup: bool | None = None


class PyTorchAnalysisResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    operators: tuple[OperatorSummary, ...]
    total: int
    returned: int
    truncated: bool
    coverage: dict[str, bool]
    repeated_small_operations: tuple[OperatorSummary, ...] = ()
    synchronization_time_ns: int = 0
    compilation_time_ns: int = 0
    warmup_time_ns: int = 0
    allocation_bytes: int | None = None
    limitations: tuple[str, ...]
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class FailureCluster(ContractModel):
    collector: str | None
    execution_status: str
    capture_status: str
    validation_status: str
    exit_code: int | None
    workload_definition_id: str | None
    environment_id: str
    source_state_id: str | None
    run_count: int
    first_seen: str
    last_seen: str
    representative_artifact_ids: tuple[str, ...] = ()


class FailureChangePoint(ContractModel):
    observed_date: str
    run_count: int
    previous_run_count: int | None


class FailureAnalysisResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    cohort_id: str
    filters_applied: tuple[str, ...]
    eligible_runs: int = 0
    failed_runs: int = 0
    population_status: Literal["observed", "empty", "filtered_empty"] = "empty"
    empty_reason: Literal["no_runs", "no_matching_runs", "no_failures"] | None = "no_runs"
    failures: tuple[FailureCluster, ...]
    total_clusters: int
    returned: int
    truncated: bool
    change_points: tuple[FailureChangePoint, ...]
    coverage: dict[str, float]
    competing_hypotheses: tuple[str, ...]
    limitations: tuple[str, ...] = (
        "Clusters use lifecycle status and exit code; native crash signatures "
        "require a specialized extractor.",
    )
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class ScalingPoint(ContractModel):
    variant: str
    block_id: str | None
    input_value: float | None
    value: float
    dispersion: float
    confidence_low: float | None = None
    confidence_high: float | None = None
    confidence_level: float | None = None
    unit: str
    sample_count: int
    raw_sample_count: int = 0
    environment_count: int


class ScalingTrialSummary(ContractModel):
    trial_id: str
    variant: str
    block_id: str | None
    input_value: float | None
    median: float
    dispersion: float
    unit: str
    raw_sample_count: int
    environment_id: str


class ScalingCorrelatedHotspot(ContractModel):
    variant: str
    frame_id: str
    function: str | None
    file: str | None
    line: int | None
    metric: str
    unit: str
    spearman_rho: float
    p_value: float
    adjusted_p_value: float
    multiplicity_method: str
    # ``None`` while no hypotheses have been tested yet; the multiplicity
    # method string must not be combined with a zero count.
    tested_hypothesis_count: int | None = None
    independent_trial_count: int
    supported_min: float
    supported_max: float


class ScalingFit(ContractModel):
    model: str
    variant: str
    coefficients: tuple[float, ...]
    coefficient_standard_errors: tuple[float, ...]
    coefficient_confidence_intervals: tuple[tuple[float, float], ...]
    residual_rms: float
    r_squared: float | None
    aicc: float | None
    condition_number: float
    durbin_watson: float | None
    observation_count: int
    supported_min: float
    supported_max: float


class ScalingAnalysisResult(ContractModel):
    schema_version: int = 1
    corpus_commit_id: str
    experiment_id: str
    metric: str
    points: tuple[ScalingPoint, ...]
    trials: tuple[ScalingTrialSummary, ...]
    attempted_trials: int
    succeeded_trials: int
    failed_trials: int
    complete_blocks: int
    fits: tuple[ScalingFit, ...]
    correlated_hotspots: tuple[ScalingCorrelatedHotspot, ...]
    conclusion: str
    environment_stable: bool
    warnings: tuple[str, ...]
    limitations: tuple[str, ...] = (
        "Points are per-trial medians; statistical decisions belong to frozen run-set comparisons.",
    )
    evidence: EvidenceAvailability = Field(default_factory=available_availability)


class RecipeService:
    def __init__(
        self,
        workspace: Workspace,
        *,
        snapshot: Snapshot | None = None,
    ) -> None:
        self.workspace = workspace
        self.snapshot = snapshot

    def hotspots(
        self,
        input_id: str,
        *,
        limit: int | None = None,
        corpus_commit_id: str | None = None,
    ) -> HotspotResult:
        corpus_commit_id = self._pinned_commit_id(corpus_commit_id)
        bounded = self._limit(limit)
        with self._open_snapshot(corpus_commit_id) as snapshot:
            scope = resolve_evidence_scope(snapshot, input_id)
            where, parameters = scope.predicate(
                run_column="fm.run_id",
                artifact_column="fm.artifact_id",
            )
            profile_where, profile_parameters = self._profile_artifact_predicate(scope)
            profile_row = snapshot.execute(
                "SELECT count(*) FROM artifact_registrations ar WHERE " + profile_where,
                profile_parameters,
            ).fetchone()
            assert profile_row is not None
            if int(profile_row[0]) == 0:
                return HotspotResult(
                    corpus_commit_id=snapshot.commit.commit_id,
                    input_id=input_id,
                    hotspots=(),
                    total=0,
                    returned=0,
                    truncated=False,
                    coverage={"frame_measurements": 0, "completely_symbolized": 0},
                    limitations=(
                        "No registered profile artifact is available for this input; "
                        "profile parsing is extractor-owned.",
                    ),
                    evidence_status="unavailable",
                    unavailable_reason="no_profile_artifact",
                    evidence=EvidenceAvailability(
                        status="unavailable",
                        reason="no_profile_artifact",
                    ),
                )
            count_row = snapshot.execute(
                "SELECT count(*) FROM frame_measurements fm WHERE " + where,
                parameters,
            ).fetchone()
            assert count_row is not None
            total = int(count_row[0])
            rows = snapshot.execute(
                "SELECT fm.frame_id, f.function, f.file, f.line, fm.metric, "
                "fm.self_value, fm.inclusive_value, fm.unit, fm.sample_count "
                "FROM frame_measurements fm LEFT JOIN frames f "
                "ON f.frame_id = fm.frame_id WHERE "
                + where
                + " ORDER BY coalesce(fm.inclusive_value, fm.self_value, 0) DESC, "
                "fm.frame_id LIMIT ?",
                (*parameters, bounded),
            ).fetchall()
            symbolized_row = snapshot.execute(
                "SELECT count(*) FROM frame_measurements fm JOIN frames f "
                "ON f.frame_id = fm.frame_id WHERE " + where + " AND f.symbolization = 'complete'",
                parameters,
            ).fetchone()
            assert symbolized_row is not None
        hotspots = tuple(
            Hotspot(
                frame_id=row[0],
                function=row[1],
                file=row[2],
                line=row[3],
                metric=row[4],
                self_value=row[5],
                inclusive_value=row[6],
                unit=row[7],
                sample_count=row[8],
            )
            for row in rows
        )
        return HotspotResult(
            corpus_commit_id=snapshot.commit.commit_id,
            input_id=input_id,
            hotspots=hotspots,
            total=total,
            returned=len(hotspots),
            truncated=total > len(hotspots),
            coverage={
                "frame_measurements": total,
                "completely_symbolized": int(symbolized_row[0]),
            },
            limitations=(
                "Complete stacks remain in native artifacts; this result is a bounded "
                "frame aggregate.",
            ),
            evidence_status="available" if hotspots else "empty",
            evidence=(
                empty_availability("no_matching_hotspots")
                if not hotspots
                else available_availability()
            ),
        )

    @staticmethod
    def _profile_artifact_predicate(scope: EvidenceScope) -> tuple[str, tuple[object, ...]]:
        # The scope is deliberately converted here rather than exposing host paths or
        # asking callers to provide an artifact kind filter.
        run_ids = scope.run_ids
        artifact_ids = scope.artifact_ids
        predicates: list[str] = []
        parameters: list[object] = []
        if run_ids:
            placeholders = ", ".join("?" for _ in run_ids)
            predicates.append(
                f"(ar.run_id IN ({placeholders}) AND ar.kind IN "
                "('sample_profile', 'memory_profile', 'execution_trace'))"
            )
            parameters.extend(run_ids)
        if artifact_ids:
            placeholders = ", ".join("?" for _ in artifact_ids)
            predicates.append(
                f"(ar.artifact_id IN ({placeholders}) AND ar.kind IN "
                "('sample_profile', 'memory_profile', 'execution_trace'))"
            )
            parameters.extend(artifact_ids)
        return " OR ".join(predicates) or "FALSE", tuple(parameters)

    def memory(
        self,
        input_id: str,
        *,
        limit: int | None = None,
        corpus_commit_id: str | None = None,
    ) -> MemoryAnalysisResult:
        corpus_commit_id = self._pinned_commit_id(corpus_commit_id)
        bounded = self._limit(limit)
        with self._open_snapshot(corpus_commit_id) as snapshot:
            scope = resolve_evidence_scope(snapshot, input_id)
            where, parameters = scope.predicate(
                run_column="run_id",
                artifact_column="artifact_id",
            )
            rows = snapshot.execute(
                "SELECT name, value_int, value_float, unit, aggregation, scope "
                "FROM measurements WHERE "
                + where
                + " AND name LIKE 'memory.%' ORDER BY name LIMIT ?",
                (*parameters, bounded),
            ).fetchall()
            phase_rows = snapshot.execute(
                "SELECT phase, name, "
                "median(coalesce(CAST(value_int AS DOUBLE), value_float)), "
                "count(*), any_value(unit), "
                "min(coalesce(worker_run_index, 0) * 1000000 "
                "+ coalesce(value_index, 0)) AS phase_order "
                "FROM measurements WHERE "
                + where
                + " AND name LIKE 'memory.%' AND phase IS NOT NULL "
                "AND coalesce(CAST(value_int AS DOUBLE), value_float) IS NOT NULL "
                "GROUP BY phase, name ORDER BY phase_order, phase, name",
                parameters,
            ).fetchall()
            profile_where, profile_parameters = self._memory_profile_artifact_predicate(scope)
            profile_row = snapshot.execute(
                "SELECT count(*) FROM artifact_registrations ar WHERE " + profile_where,
                profile_parameters,
            ).fetchone()
            assert profile_row is not None
            has_memory_profile = int(profile_row[0]) > 0
            hotspot_result = RecipeService(
                self.workspace,
                snapshot=snapshot,
            ).hotspots(
                input_id,
                limit=bounded,
                corpus_commit_id=corpus_commit_id,
            )
            resource_rows, resource_total, resource_truncated = self._runtime_resources(
                snapshot,
                scope,
                bounded,
            )
            writable_rows = self._writable_root_observations(snapshot, scope, bounded)
            unavailable_metrics = tuple(
                sorted({metric for item in resource_rows for metric in item.unavailable_metrics})
            )
            policy_termination = next(
                (
                    item.policy_termination
                    for item in resource_rows
                    if item.policy_termination is not None
                ),
                None,
            )
        previous_by_metric: dict[str, float] = {}
        phase_growth: list[MemoryPhaseGrowth] = []
        for phase, metric, value, sample_count, unit, _ in phase_rows:
            numeric = float(value)
            previous = previous_by_metric.get(str(metric))
            phase_growth.append(
                MemoryPhaseGrowth(
                    phase=str(phase),
                    metric=str(metric),
                    value=numeric,
                    previous_value=previous,
                    delta=numeric - previous if previous is not None else None,
                    unit=str(unit),
                    sample_count=int(sample_count),
                )
            )
            previous_by_metric[str(metric)] = numeric
        limitations = [
            "High-water-mark, retained-end, and allocation volume are distinct "
            "concepts and are not substituted for one another."
        ]
        if not resource_rows:
            limitations.append(
                "Runtime resource summary was not published for this evidence generation."
            )
        memory_run_id = scope.run_ids[0] if len(scope.run_ids) == 1 else None
        evidence = (
            EvidenceAvailability(
                status="unavailable",
                reason=(
                    "memory_profile_not_extracted"
                    if has_memory_profile
                    else "no_memory_profile_artifact"
                ),
                next_tool="extract_memray" if has_memory_profile and memory_run_id else None,
                next_arguments=(
                    {"run_id": memory_run_id} if has_memory_profile and memory_run_id else None
                ),
            )
            if not rows and has_memory_profile
            else (
                EvidenceAvailability(
                    status="unavailable",
                    reason="no_memory_profile_artifact",
                )
                if not rows and hotspot_result.evidence_status == "unavailable"
                else empty_availability("no_memory_measurements")
                if not rows
                else available_availability()
            )
        )
        return MemoryAnalysisResult(
            corpus_commit_id=snapshot.commit.commit_id,
            input_id=input_id,
            measurements=tuple(
                MeasurementSummary(
                    name=row[0],
                    value_int=row[1],
                    value_float=row[2],
                    unit=row[3],
                    aggregation=row[4],
                    scope=row[5],
                )
                for row in rows
            ),
            hotspots=tuple(
                item for item in hotspot_result.hotspots if item.metric.startswith("memory.")
            ),
            phase_growth=tuple(phase_growth),
            limitations=tuple(limitations),
            runtime_resources=resource_rows,
            runtime_resource_totals=resource_total,
            runtime_resources_truncated=resource_truncated,
            truncated=resource_truncated,
            writable_root_observations=writable_rows,
            policy_termination=policy_termination,
            unavailable_metrics=unavailable_metrics,
            evidence=evidence,
        )

    @staticmethod
    def _memory_profile_artifact_predicate(scope: EvidenceScope) -> tuple[str, tuple[object, ...]]:
        predicates: list[str] = []
        parameters: list[object] = []
        if scope.run_ids:
            placeholders = ", ".join("?" for _ in scope.run_ids)
            predicates.append(f"(ar.run_id IN ({placeholders}) AND ar.kind = 'memory_profile')")
            parameters.extend(scope.run_ids)
        if scope.artifact_ids:
            placeholders = ", ".join("?" for _ in scope.artifact_ids)
            predicates.append(
                f"(ar.artifact_id IN ({placeholders}) AND ar.kind = 'memory_profile')"
            )
            parameters.extend(scope.artifact_ids)
        return " OR ".join(predicates) or "FALSE", tuple(parameters)

    def _runtime_resources(
        self,
        snapshot: Snapshot,
        scope: EvidenceScope,
        limit: int,
    ) -> tuple[tuple[RuntimeResourceObservation, ...], RuntimeResourceTotals, bool]:
        where, parameters = self._run_scope_predicate(scope, "rr.run_id")
        rows = snapshot.execute(
            "SELECT run_id, sampling_interval_ms, minimum_free_bytes, staging_growth_bytes, "
            "peak_rss_bytes, policy_termination, unavailable_metrics "
            "FROM runtime_resource_summaries rr WHERE "
            + where
            + " ORDER BY run_id, published_at DESC LIMIT ?",
            (*parameters, limit + 1),
        ).fetchall()
        total_row = snapshot.execute(
            "SELECT count(*), min(minimum_free_bytes), sum(staging_growth_bytes), "
            "max(peak_rss_bytes) FROM runtime_resource_summaries rr WHERE " + where,
            parameters,
        ).fetchone()
        assert total_row is not None
        observations = tuple(
            RuntimeResourceObservation(
                run_id=str(row[0]),
                sampling_interval_ms=int(row[1]),
                minimum_free_bytes=int(row[2]) if row[2] is not None else None,
                staging_growth_bytes=int(row[3]) if row[3] is not None else None,
                peak_rss_bytes=int(row[4]) if row[4] is not None else None,
                policy_termination=str(row[5]) if row[5] is not None else None,
                unavailable_metrics=tuple(str(item) for item in (row[6] or ())),
            )
            for row in rows[:limit]
        )
        return (
            observations,
            RuntimeResourceTotals(
                run_count=int(total_row[0]),
                minimum_free_bytes=(int(total_row[1]) if total_row[1] is not None else None),
                total_staging_growth_bytes=(
                    int(total_row[2]) if total_row[2] is not None else None
                ),
                maximum_peak_rss_bytes=(int(total_row[3]) if total_row[3] is not None else None),
            ),
            len(rows) > limit,
        )

    def _writable_root_observations(
        self,
        snapshot: Snapshot,
        scope: EvidenceScope,
        limit: int,
    ) -> tuple[WritableRootObservation, ...]:
        where, parameters = self._run_scope_predicate(scope, "rw.run_id")
        rows = snapshot.execute(
            "SELECT run_id, writable_root_identity, target_path, growth_bytes, available, "
            "unavailable_reason FROM runtime_writable_root_growth rw WHERE "
            + where
            + " ORDER BY run_id, target_path LIMIT ?",
            (*parameters, limit),
        ).fetchall()
        return tuple(
            WritableRootObservation(
                run_id=str(row[0]),
                writable_root_identity=str(row[1]),
                target_path=str(row[2]),
                growth_bytes=int(row[3]) if row[3] is not None else None,
                available=bool(row[4]),
                unavailable_reason=str(row[5]) if row[5] is not None else None,
            )
            for row in rows
        )

    @staticmethod
    def _run_scope_predicate(scope: EvidenceScope, column: str) -> tuple[str, tuple[object, ...]]:
        run_ids = scope.run_ids
        artifact_ids = scope.artifact_ids
        predicates: list[str] = []
        parameters: list[object] = []
        if run_ids:
            placeholders = ", ".join("?" for _ in run_ids)
            predicates.append(f"{column} IN ({placeholders})")
            parameters.extend(run_ids)
        if artifact_ids:
            placeholders = ", ".join("?" for _ in artifact_ids)
            predicates.append(
                f"{column} IN (SELECT run_id FROM artifact_registrations "
                f"WHERE artifact_id IN ({placeholders}))"
            )
            parameters.extend(artifact_ids)
        return " OR ".join(f"({item})" for item in predicates) or "FALSE", tuple(parameters)

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

    def pytorch(
        self,
        input_id: str,
        *,
        limit: int | None = None,
        corpus_commit_id: str | None = None,
    ) -> PyTorchAnalysisResult:
        corpus_commit_id = self._pinned_commit_id(corpus_commit_id)
        bounded = self._limit(limit)
        with self._open_snapshot(corpus_commit_id) as snapshot:
            scope = resolve_evidence_scope(snapshot, input_id)
            self._require_pytorch_source(snapshot, scope.run_ids, scope.artifact_ids)
            where, parameters = scope.predicate(
                run_column="fm.run_id",
                artifact_column="fm.artifact_id",
            )
            count_row = snapshot.execute(
                "SELECT count(DISTINCT fm.frame_id) FROM frame_measurements fm WHERE " + where,
                parameters,
            ).fetchone()
            assert count_row is not None
            total = int(count_row[0])
            if total == 0:
                if scope.run_ids:
                    run_ids = scope.run_ids
                elif scope.artifact_ids:
                    run_rows = snapshot.execute(
                        "SELECT DISTINCT run_id FROM artifact_registrations WHERE artifact_id IN ("
                        + ", ".join("?" for _ in scope.artifact_ids)
                        + ") ORDER BY run_id",
                        scope.artifact_ids,
                    ).fetchall()
                    run_ids = tuple(str(row[0]) for row in run_rows)
                else:
                    run_ids = ()
                details: dict[str, object] = {"next_tool": "extract_perfetto"}
                if len(run_ids) == 1:
                    details["run_id"] = run_ids[0]
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "PyTorch operator analysis requires Perfetto extraction for this imported "
                    "trace.",
                    details=details,
                    remediation=(
                        "Call extract_perfetto with the reported run_id, then retry "
                        "analyze_pytorch.",
                        "If Trace Processor is unavailable, call prepare_capabilities with "
                        "adapter='perfetto'.",
                    ),
                )
            rows = snapshot.execute(
                "SELECT fm.frame_id, coalesce(f.function, '<unnamed>'), f.module, "
                "sum(coalesce(fm.self_value, 0)), "
                "sum(coalesce(fm.inclusive_value, 0)), "
                "sum(coalesce(fm.sample_count, 0)) "
                "FROM frame_measurements fm JOIN frames f "
                "ON f.frame_id = fm.frame_id WHERE "
                + where
                + " GROUP BY fm.frame_id, f.function, f.module "
                "ORDER BY sum(coalesce(fm.inclusive_value, 0)) DESC, fm.frame_id "
                "LIMIT ?",
                (*parameters, bounded),
            ).fetchall()
            observation_where, observation_parameters = scope.predicate(
                run_column="run_id",
                artifact_column="artifact_id",
            )
            metadata_rows = snapshot.execute(
                "SELECT name, value_json, context FROM observations WHERE "
                + observation_where
                + " AND kind = 'pytorch.operator'",
                observation_parameters,
            ).fetchall()
        metadata_by_operator: dict[tuple[str, str], list[dict[str, object]]] = {}
        for name, value_json, context in metadata_rows:
            try:
                value = json.loads(str(value_json))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            frame_id = value.get("frame_id")
            if not isinstance(frame_id, str):
                continue
            if context is not None and value.get("phase") is None:
                value["phase"] = str(context)
            metadata_by_operator.setdefault((frame_id, str(name)), []).append(value)
        operators_list: list[OperatorSummary] = []
        device_time_present = False
        synchronization_present = False
        for row in rows:
            category = str(row[2]) if row[2] is not None else None
            category_lower = (category or "").lower()
            operator = str(row[1])
            operator_lower = operator.lower()
            frame_id = str(row[0])
            metadata = metadata_by_operator.get((frame_id, operator), [])
            is_device = any(
                token in category_lower
                for token in ("kernel", "gpu", "device", "xpu", "hip", "mps")
            )
            synchronization = any(
                token in operator_lower
                for token in (
                    "synchronize",
                    "cudadevicesynchronize",
                    "cudastreamsynchronize",
                    "event_synchronize",
                )
            )
            device_time_present = device_time_present or is_device
            synchronization_present = synchronization_present or synchronization
            inclusive = int(row[4])
            shapes = tuple(
                sorted(
                    {
                        str(value["input_shapes"])
                        for value in metadata
                        if value.get("input_shapes") not in {None, ""}
                    }
                )
            )
            allocations = [
                allocation
                for value in metadata
                if isinstance(
                    allocation := value.get("allocation_bytes"),
                    int,
                )
            ]
            phases = {
                str(value["phase"]).lower()
                for value in metadata
                if value.get("phase") not in {None, ""}
            }
            warmup_phases = {phase for phase in phases if "warm" in phase}
            warmup = (
                True
                if phases and warmup_phases == phases
                else False
                if phases and not warmup_phases
                else None
            )
            operators_list.append(
                OperatorSummary(
                    frame_id=frame_id,
                    operator=operator,
                    category=category,
                    self_cpu_ns=None if is_device else int(row[3]),
                    total_cpu_ns=None if is_device else inclusive,
                    device_ns=inclusive if is_device else None,
                    inclusive_ns=inclusive,
                    event_count=int(row[5]),
                    input_shapes=shapes,
                    allocation_bytes=sum(allocations) if allocations else None,
                    synchronization=synchronization,
                    warmup=warmup,
                )
            )
        operators = tuple(operators_list)
        synchronization_time_ns = sum(
            item.inclusive_ns for item in operators if item.synchronization
        )
        compilation_time_ns = sum(
            item.inclusive_ns
            for item in operators
            if any(
                token in item.operator.lower()
                for token in ("compile", "dynamo", "inductor", "graph_executor")
            )
        )
        warmup_time_ns = sum(
            duration
            for metadata in metadata_by_operator.values()
            for value in metadata
            if isinstance(duration := value.get("duration_ns"), int)
            and "warm" in str(value.get("phase", "")).lower()
        )
        allocation_bytes = sum(item.allocation_bytes or 0 for item in operators) or None
        # Use the 25th percentile of per-event inclusive time as the
        # "typical small" threshold. The median flags ~50% of all operators
        # (by definition) and is not a useful discriminator for "repeated
        # *small* operations"; the lower quartile targets operators whose
        # per-event cost is genuinely smaller than the bulk of the workload.
        per_event_times = [
            operator.inclusive_ns / max(operator.event_count, 1) for operator in operators
        ]
        typical_event_ns = max(
            1.0,
            float(np.percentile(per_event_times, 25)) if per_event_times else 1.0,
        )
        repeated_small = tuple(
            sorted(
                (
                    item
                    for item in operators
                    if item.event_count >= 3
                    and item.inclusive_ns / item.event_count <= typical_event_ns
                ),
                key=lambda item: (-item.event_count, item.inclusive_ns, item.frame_id),
            )[:bounded]
        )
        limitations = [
            "Operator categories and durations come from the exported torch.profiler trace.",
            "Nested operator durations can overlap; self time subtracts direct nested slices.",
        ]
        if not device_time_present:
            limitations.append("The trace contains no recognized accelerator kernel categories.")
        shapes_present = any(item.input_shapes for item in operators)
        allocations_present = any(item.allocation_bytes is not None for item in operators)
        warmup_present = any(item.warmup is not None for item in operators)
        if not shapes_present:
            limitations.append("Input shapes were not present in normalized trace evidence.")
        if not allocations_present:
            limitations.append(
                "Per-operator allocation bytes were not present in normalized trace evidence."
            )
        if not warmup_present:
            limitations.append("Warm-up separation requires profiler phase annotations.")
        return PyTorchAnalysisResult(
            corpus_commit_id=snapshot.commit.commit_id,
            input_id=input_id,
            operators=operators,
            total=total,
            returned=len(operators),
            truncated=total > len(operators),
            coverage={
                "self_cpu_time": True,
                "total_cpu_time": True,
                "device_time": device_time_present,
                "input_shapes": shapes_present,
                "memory_allocations": allocations_present,
                "synchronization": synchronization_present,
                "warmup_phases": warmup_present,
            },
            repeated_small_operations=repeated_small,
            synchronization_time_ns=synchronization_time_ns,
            compilation_time_ns=compilation_time_ns,
            warmup_time_ns=warmup_time_ns,
            allocation_bytes=allocation_bytes,
            limitations=tuple(limitations),
            evidence=(
                empty_availability("no_normalized_torch_operators")
                if total == 0
                else available_availability()
            ),
        )

    def failures(
        self,
        *,
        limit: int | None = None,
        corpus_commit_id: str | None = None,
        source_state_id: str | None = None,
        environment_id: str | None = None,
        workload_definition_id: str | None = None,
        execution_status: tuple[str, ...] = (),
        validation_status: tuple[str, ...] = (),
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> FailureAnalysisResult:
        corpus_commit_id = self._pinned_commit_id(corpus_commit_id)
        bounded = self._limit(limit)
        filters: list[str] = []
        parameters: list[object] = []
        applied: list[str] = []
        for field, value in (
            ("source_state_id", source_state_id),
            ("environment_id", environment_id),
            ("workload_definition_id", workload_definition_id),
        ):
            if value is not None:
                filters.append(f"{field} = ?")
                parameters.append(value)
                applied.append(field)
        for field, values in (
            ("execution_status", execution_status),
            ("validation_status", validation_status),
        ):
            if values:
                filters.append(f"{field} IN ({', '.join('?' for _ in values)})")
                parameters.extend(values)
                applied.append(field)
        if created_after is not None:
            filters.append("created_at >= ?")
            parameters.append(created_after)
            applied.append("created_after")
        if created_before is not None:
            filters.append("created_at < ?")
            parameters.append(created_before)
            applied.append("created_before")
        cohort_filter = "".join(f" AND {item}" for item in filters)
        query = f"""
            WITH latest AS (
                SELECT *,
                    row_number() OVER (
                        PARTITION BY run_id ORDER BY published_at DESC
                    ) AS revision_order
                FROM runs
            ),
            eligible AS (
                SELECT * FROM latest
                WHERE revision_order = 1
                  {cohort_filter}
            ),
            failed AS (
                SELECT * FROM eligible
                WHERE (
                    execution_status NOT IN ('succeeded', 'not_applicable')
                    OR capture_status IN ('failed', 'cancelled')
                    OR validation_status IN ('failed', 'error', 'inconclusive', 'unsupported')
                  )
            ),
            clusters AS (
                SELECT collector, execution_status, capture_status,
                    validation_status, exit_code, workload_definition_id,
                    environment_id, source_state_id, count(*) AS run_count,
                    min(created_at) AS first_seen, max(created_at) AS last_seen,
                    min(run_id) AS representative_run_id
                FROM failed
                GROUP BY collector, execution_status, capture_status,
                    validation_status, exit_code, workload_definition_id,
                    environment_id, source_state_id
            )
        """
        with self._open_snapshot(corpus_commit_id) as snapshot:
            population_row = snapshot.execute(
                query + " SELECT "
                "(SELECT count(*) FROM latest WHERE revision_order = 1), "
                "(SELECT count(*) FROM eligible), "
                "(SELECT count(*) FROM failed)",
                tuple(parameters),
            ).fetchone()
            assert population_row is not None
            count_row = snapshot.execute(
                query + " SELECT count(*) FROM clusters",
                tuple(parameters),
            ).fetchone()
            assert count_row is not None
            total = int(count_row[0])
            rows = snapshot.execute(
                query + " SELECT collector, execution_status, capture_status, "
                "validation_status, exit_code, workload_definition_id, "
                "environment_id, source_state_id, run_count, first_seen, last_seen, "
                "representative_run_id "
                "FROM clusters ORDER BY run_count DESC, last_seen DESC LIMIT ?",
                (*parameters, bounded),
            ).fetchall()
            change_rows = snapshot.execute(
                query + " , daily AS (SELECT CAST(created_at AS DATE) AS observed_date, "
                "count(*) AS run_count FROM failed GROUP BY observed_date), "
                "with_previous AS (SELECT observed_date, run_count, "
                "lag(run_count) OVER (ORDER BY observed_date) AS previous_run_count "
                "FROM daily) SELECT observed_date, run_count, previous_run_count "
                "FROM with_previous WHERE previous_run_count IS NULL "
                "OR run_count <> previous_run_count ORDER BY observed_date",
                tuple(parameters),
            ).fetchall()
            coverage_row = snapshot.execute(
                query + " SELECT count(*), "
                "count(*) FILTER (WHERE source_state_id IS NOT NULL), "
                "count(*) FILTER (WHERE EXISTS (SELECT 1 FROM artifact_registrations a "
                "WHERE a.run_id = failed.run_id)), "
                "count(*) FILTER (WHERE EXISTS (SELECT 1 FROM frame_measurements fm "
                "WHERE fm.run_id = failed.run_id)) FROM failed",
                tuple(parameters),
            ).fetchone()
            assert coverage_row is not None
            representative_artifacts: dict[str, tuple[str, ...]] = {}
            for row in rows:
                run_id = str(row[11])
                artifact_rows = snapshot.execute(
                    "SELECT DISTINCT artifact_id FROM artifact_registrations "
                    "WHERE run_id = ? ORDER BY artifact_id LIMIT 3",
                    (run_id,),
                ).fetchall()
                representative_artifacts[run_id] = tuple(
                    str(artifact[0]) for artifact in artifact_rows
                )
        failures = tuple(
            FailureCluster(
                collector=str(row[0]) if row[0] is not None else None,
                execution_status=str(row[1]),
                capture_status=str(row[2]),
                validation_status=str(row[3]),
                exit_code=int(row[4]) if row[4] is not None else None,
                workload_definition_id=str(row[5]) if row[5] is not None else None,
                environment_id=str(row[6]),
                source_state_id=str(row[7]) if row[7] is not None else None,
                run_count=int(row[8]),
                first_seen=row[9].isoformat(),
                last_seen=row[10].isoformat(),
                representative_artifact_ids=representative_artifacts[str(row[11])],
            )
            for row in rows
        )
        change_points = tuple(
            FailureChangePoint(
                observed_date=row[0].isoformat(),
                run_count=int(row[1]),
                previous_run_count=int(row[2]) if row[2] is not None else None,
            )
            for row in change_rows
        )
        failed_run_count = int(coverage_row[0])
        workspace_run_count = int(population_row[0])
        eligible_run_count = int(population_row[1])
        population_status: Literal["observed", "empty", "filtered_empty"]
        empty_reason: Literal["no_runs", "no_matching_runs", "no_failures"] | None
        if workspace_run_count == 0:
            population_status = "empty"
            empty_reason = "no_runs"
        elif eligible_run_count == 0:
            population_status = "filtered_empty"
            empty_reason = "no_matching_runs"
        else:
            population_status = "observed"
            empty_reason = "no_failures" if failed_run_count == 0 else None
        denominator = failed_run_count or 1
        hypotheses: list[str] = []
        if len({failure.collector for failure in failures}) > 1:
            hypotheses.append(
                "Collector-specific behavior is plausible because failures span "
                "distinct collector groups."
            )
        if len({failure.environment_id for failure in failures}) > 1:
            hypotheses.append(
                "Environment-specific behavior is plausible because failures span "
                "distinct environment identities."
            )
        if not hypotheses and failures:
            hypotheses.extend(
                (
                    "A workload-specific failure mode remains plausible.",
                    "A shared environment or dependency failure remains plausible.",
                )
            )
        return FailureAnalysisResult(
            corpus_commit_id=snapshot.commit.commit_id,
            cohort_id=digest_model(
                {
                    "corpus_commit_id": corpus_commit_id,
                    "source_state_id": source_state_id,
                    "environment_id": environment_id,
                    "workload_definition_id": workload_definition_id,
                    "execution_status": execution_status,
                    "validation_status": validation_status,
                    "created_after": created_after,
                    "created_before": created_before,
                }
            ),
            filters_applied=tuple(applied),
            eligible_runs=eligible_run_count,
            failed_runs=failed_run_count,
            population_status=population_status,
            empty_reason=empty_reason,
            failures=failures,
            total_clusters=total,
            returned=len(failures),
            truncated=total > len(failures),
            change_points=change_points,
            coverage={
                "source_identity": int(coverage_row[1]) / denominator,
                "artifact": int(coverage_row[2]) / denominator,
                "symbolized_frames": int(coverage_row[3]) / denominator,
            },
            competing_hypotheses=tuple(hypotheses),
            evidence=(
                empty_availability(empty_reason or "no_failures")
                if not failures
                else available_availability()
            ),
        )

    def scaling(
        self,
        experiment_id: str,
        *,
        corpus_commit_id: str | None = None,
    ) -> ScalingAnalysisResult:
        corpus_commit_id = self._pinned_commit_id(corpus_commit_id)
        with self._open_snapshot(corpus_commit_id) as snapshot:
            experiment_row = snapshot.execute(
                "SELECT primary_metric, confidence_level FROM experiments WHERE experiment_id = ? "
                "ORDER BY published_at DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
            if experiment_row is None:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Unknown experiment {experiment_id!r}.",
                )
            metric = str(experiment_row[0])
            confidence_level = float(experiment_row[1])
            trial_row = snapshot.execute(
                "SELECT count(*), "
                "count(*) FILTER (WHERE outcome = 'succeeded'), "
                "count(*) FILTER (WHERE outcome <> 'succeeded') "
                "FROM (SELECT DISTINCT trial_id, outcome FROM trials "
                "WHERE experiment_id = ?)",
                (experiment_id,),
            ).fetchone()
            assert trial_row is not None
            rows = snapshot.execute(
                "WITH latest_runs AS (SELECT *, row_number() OVER "
                "(PARTITION BY run_id ORDER BY published_at DESC) AS revision_order "
                "FROM runs) "
                "SELECT t.trial_id, v.name, t.block_id, t.parameter_value_int, "
                "t.parameter_value_float, r.environment_id, "
                "coalesce(CAST(m.value_int AS DOUBLE), m.value_float), m.unit "
                "FROM (SELECT DISTINCT trial_id, experiment_id, variant_id, "
                "run_id, block_id, outcome, parameter_value_int, "
                "parameter_value_float FROM trials) t "
                "JOIN (SELECT DISTINCT variant_id, name FROM variants) v "
                "ON v.variant_id = t.variant_id "
                "JOIN latest_runs r ON r.run_id = t.run_id AND r.revision_order = 1 "
                "JOIN measurements m ON m.run_id = t.run_id "
                "WHERE t.experiment_id = ? AND t.outcome = 'succeeded' "
                "AND m.name = ? AND m.is_warmup = false "
                "ORDER BY t.block_id, v.name, m.measurement_id",
                (experiment_id, metric),
            ).fetchall()
            hotspot_rows = snapshot.execute(
                "SELECT t.trial_id, v.name, t.parameter_value_int, "
                "t.parameter_value_float, fm.frame_id, f.function, f.file, f.line, "
                "fm.metric, fm.unit, "
                "coalesce(CAST(fm.inclusive_value AS DOUBLE), "
                "CAST(fm.self_value AS DOUBLE)) "
                "FROM (SELECT DISTINCT trial_id, experiment_id, variant_id, run_id, "
                "outcome, parameter_value_int, parameter_value_float FROM trials) t "
                "JOIN (SELECT DISTINCT variant_id, name FROM variants) v "
                "ON v.variant_id = t.variant_id "
                "JOIN frame_measurements fm ON fm.run_id = t.run_id "
                "LEFT JOIN frames f ON f.frame_id = fm.frame_id "
                "WHERE t.experiment_id = ? AND t.outcome = 'succeeded' "
                "AND coalesce(fm.inclusive_value, fm.self_value) IS NOT NULL",
                (experiment_id,),
            ).fetchall()
            complete_row = snapshot.execute(
                "WITH expected AS (SELECT count(DISTINCT variant_id) AS n "
                "FROM variants WHERE experiment_id = ?), "
                "blocks AS (SELECT block_id, count(DISTINCT variant_id) AS n "
                "FROM trials WHERE experiment_id = ? AND outcome = 'succeeded' "
                "GROUP BY block_id) "
                "SELECT count(*) FROM blocks, expected "
                "WHERE blocks.n = expected.n",
                (experiment_id, experiment_id),
            ).fetchone()
            assert complete_row is not None
        trial_groups: dict[
            tuple[str, str, str | None, float | None, str, str],
            list[float],
        ] = {}
        for row in rows:
            input_value = (
                float(row[3])
                if row[3] is not None
                else float(row[4])
                if row[4] is not None
                else None
            )
            key = (
                str(row[0]),
                str(row[1]),
                str(row[2]) if row[2] is not None else None,
                input_value,
                str(row[7]),
                str(row[5]),
            )
            if row[6] is not None:
                trial_groups.setdefault(key, []).append(float(row[6]))
        trials: list[ScalingTrialSummary] = []
        for (
            trial_id,
            variant,
            block_id,
            input_value,
            unit,
            environment_id,
        ), values in sorted(trial_groups.items()):
            median = float(np.median(values))
            trials.append(
                ScalingTrialSummary(
                    trial_id=trial_id,
                    variant=variant,
                    block_id=block_id,
                    input_value=input_value,
                    median=median,
                    dispersion=float(np.median(np.abs(np.asarray(values) - median))),
                    unit=unit,
                    raw_sample_count=len(values),
                    environment_id=environment_id,
                )
            )
        point_groups: dict[
            tuple[str, float | None, str],
            list[ScalingTrialSummary],
        ] = {}
        for trial in trials:
            point_groups.setdefault(
                (trial.variant, trial.input_value, trial.unit),
                [],
            ).append(trial)
        points_list: list[ScalingPoint] = []
        for (variant, input_value, unit), group in sorted(
            point_groups.items(),
            key=lambda item: (item[0][0], item[0][1] or -math.inf),
        ):
            trial_medians = np.asarray(
                [trial.median for trial in group],
                dtype=float,
            )
            median = float(np.median(trial_medians))
            dispersion = float(np.median(np.abs(trial_medians - median)))
            low, high = self._median_interval(
                trial_medians,
                confidence_level=confidence_level,
            )
            block_ids = {trial.block_id for trial in group}
            points_list.append(
                ScalingPoint(
                    variant=variant,
                    block_id=next(iter(block_ids)) if len(block_ids) == 1 else None,
                    input_value=input_value,
                    value=median,
                    dispersion=dispersion,
                    confidence_low=low,
                    confidence_high=high,
                    confidence_level=confidence_level if low is not None else None,
                    unit=unit,
                    sample_count=len(group),
                    raw_sample_count=sum(trial.raw_sample_count for trial in group),
                    environment_count=len({trial.environment_id for trial in group}),
                )
            )
        points = tuple(points_list)
        fits = self._scaling_fits(
            points,
            confidence_level=confidence_level,
        )
        correlated_hotspots = self._correlated_hotspots(hotspot_rows)
        conclusion = "inconclusive"
        conclusions: list[str] = []
        for variant in sorted({fit.variant for fit in fits}):
            comparable = [fit for fit in fits if fit.variant == variant and fit.aicc is not None]
            if len(comparable) < 2:
                continue
            ordered = sorted(
                comparable,
                key=lambda fit: fit.aicc if fit.aicc is not None else math.inf,
            )
            first_aicc = ordered[0].aicc
            second_aicc = ordered[1].aicc
            assert first_aicc is not None and second_aicc is not None
            if second_aicc - first_aicc >= 2:
                conclusions.append(f"{variant}:{ordered[0].model}")
        if conclusions:
            conclusion = "descriptive_best_fit:" + ",".join(conclusions)
        warnings = ["Fits describe only the measured input range and must not be extrapolated."]
        if any(point.input_value is None for point in points):
            warnings.append(
                "Some variants have no numeric input value and were excluded from fits."
            )
        environment_stable = all(point.environment_count == 1 for point in points)
        if not environment_stable:
            warnings.append("Environment identity varies within at least one scaling point.")
        return ScalingAnalysisResult(
            corpus_commit_id=snapshot.commit.commit_id,
            experiment_id=experiment_id,
            metric=metric,
            points=points,
            trials=tuple(trials),
            attempted_trials=int(trial_row[0]),
            succeeded_trials=int(trial_row[1]),
            failed_trials=int(trial_row[2]),
            complete_blocks=int(complete_row[0]),
            fits=fits,
            correlated_hotspots=correlated_hotspots,
            conclusion=conclusion,
            environment_stable=environment_stable,
            warnings=tuple(warnings),
            evidence=(
                empty_availability("no_scaling_trials") if not points else available_availability()
            ),
        )

    def _scaling_fits(
        self,
        points: tuple[ScalingPoint, ...],
        *,
        confidence_level: float,
    ) -> tuple[ScalingFit, ...]:
        fits: list[ScalingFit] = []
        for variant in sorted({point.variant for point in points}):
            numeric = [
                point
                for point in points
                if point.variant == variant
                and point.input_value is not None
                and point.input_value > 0
                and math.isfinite(point.input_value)
                and math.isfinite(point.value)
            ]
            if len(numeric) < 3 or len({point.input_value for point in numeric}) < 2:
                continue
            x = np.asarray([point.input_value for point in numeric], dtype=float)
            y = np.asarray([point.value for point in numeric], dtype=float)
            candidates = {
                "constant": np.column_stack((np.ones_like(x),)),
                "logarithmic": np.column_stack((np.ones_like(x), np.log(x))),
                "linear": np.column_stack((np.ones_like(x), x)),
                "n_log_n": np.column_stack((np.ones_like(x), x * np.log(x))),
                "quadratic": np.column_stack((np.ones_like(x), x, x * x)),
            }
            for name, design in candidates.items():
                observation_count, parameter_count = design.shape
                if observation_count <= parameter_count:
                    continue
                fitted = OLS(y, design).fit()
                residuals = np.asarray(fitted.resid, dtype=float)
                rss = float(np.dot(residuals, residuals))
                confidence = np.asarray(
                    fitted.conf_int(alpha=1 - confidence_level),
                    dtype=float,
                )
                aicc = (
                    float(fitted.aic)
                    + (2 * parameter_count * (parameter_count + 1))
                    / (observation_count - parameter_count - 1)
                    if observation_count > parameter_count + 1
                    else None
                )
                fits.append(
                    ScalingFit(
                        model=name,
                        variant=variant,
                        coefficients=tuple(float(value) for value in fitted.params),
                        coefficient_standard_errors=tuple(float(value) for value in fitted.bse),
                        coefficient_confidence_intervals=tuple(
                            (float(interval[0]), float(interval[1])) for interval in confidence
                        ),
                        residual_rms=math.sqrt(rss / observation_count),
                        r_squared=(
                            float(fitted.rsquared)
                            if math.isfinite(float(fitted.rsquared))
                            else None
                        ),
                        aicc=aicc,
                        condition_number=float(fitted.condition_number),
                        durbin_watson=(
                            float(durbin_watson(residuals)) if observation_count > 1 else None
                        ),
                        observation_count=observation_count,
                        supported_min=float(np.min(x)),
                        supported_max=float(np.max(x)),
                    )
                )
        return tuple(
            sorted(
                fits,
                key=lambda fit: (
                    fit.variant,
                    fit.aicc is None,
                    fit.aicc if fit.aicc is not None else math.inf,
                    fit.model,
                ),
            )
        )

    @staticmethod
    def _median_interval(
        values: np.ndarray,
        *,
        confidence_level: float,
    ) -> tuple[float | None, float | None]:
        if values.size < 2:
            return None, None
        median = float(np.median(values))
        if np.allclose(values, median):
            return median, median
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = bootstrap(
                    (values,),
                    np.median,
                    vectorized=False,
                    confidence_level=confidence_level,
                    n_resamples=1_999,
                    method="BCa",
                    rng=np.random.default_rng(0),
                )
        except ValueError:
            return None, None
        low = float(result.confidence_interval.low)
        high = float(result.confidence_interval.high)
        if not math.isfinite(low) or not math.isfinite(high):
            return None, None
        return low, high

    def _correlated_hotspots(
        self,
        rows: list[tuple[object, ...]],
    ) -> tuple[ScalingCorrelatedHotspot, ...]:
        per_trial: dict[
            tuple[
                str,
                str,
                str,
                str | None,
                str | None,
                int | None,
                str,
                str,
                float,
            ],
            float,
        ] = {}
        for row in rows:
            input_value = (
                float(cast(Any, row[2]))
                if row[2] is not None
                else float(cast(Any, row[3]))
                if row[3] is not None
                else None
            )
            if input_value is None or not math.isfinite(input_value):
                continue
            key = (
                str(row[0]),
                str(row[1]),
                str(row[4]),
                str(row[5]) if row[5] is not None else None,
                str(row[6]) if row[6] is not None else None,
                int(cast(Any, row[7])) if row[7] is not None else None,
                str(row[8]),
                str(row[9]),
                input_value,
            )
            per_trial[key] = per_trial.get(key, 0.0) + float(cast(Any, row[10]))
        groups: dict[
            tuple[str, str, str | None, str | None, int | None, str, str],
            list[tuple[float, float]],
        ] = {}
        for (
            _trial_id,
            variant,
            frame_id,
            function,
            file,
            line,
            metric,
            unit,
            input_value,
        ), value in per_trial.items():
            groups.setdefault(
                (variant, frame_id, function, file, line, metric, unit),
                [],
            ).append((input_value, value))
        results: list[ScalingCorrelatedHotspot] = []
        for (
            variant,
            frame_id,
            function,
            file,
            line,
            metric,
            unit,
        ), samples in groups.items():
            if len(samples) < 3 or len({sample[0] for sample in samples}) < 2:
                continue
            x = np.asarray([sample[0] for sample in samples], dtype=float)
            y = np.asarray([sample[1] for sample in samples], dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                correlation = spearmanr(x, y)
            rho = float(correlation.statistic)
            p_value = float(correlation.pvalue)
            if not math.isfinite(rho) or not math.isfinite(p_value):
                continue
            results.append(
                ScalingCorrelatedHotspot(
                    variant=variant,
                    frame_id=frame_id,
                    function=function,
                    file=file,
                    line=line,
                    metric=metric,
                    unit=unit,
                    spearman_rho=rho,
                    p_value=p_value,
                    adjusted_p_value=p_value,
                    multiplicity_method="benjamini-hochberg-fdr",
                    independent_trial_count=len(samples),
                    supported_min=float(np.min(x)),
                    supported_max=float(np.max(x)),
                )
            )
        if results:
            adjusted = multipletests(
                [item.p_value for item in results],
                method="fdr_bh",
            )[1]
            tested = len(results)
            results = [
                item.model_copy(
                    update={
                        "adjusted_p_value": float(adjusted[index]),
                        "tested_hypothesis_count": tested,
                    }
                )
                for index, item in enumerate(results)
            ]
        results.sort(
            key=lambda item: (
                -abs(item.spearman_rho),
                item.adjusted_p_value,
                item.variant,
                item.frame_id,
            )
        )
        return tuple(results[: self.workspace.config.analysis.default_row_limit])

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

    def _require_pytorch_source(
        self,
        snapshot: Snapshot,
        run_ids: tuple[str, ...],
        artifact_ids: tuple[str, ...],
    ) -> None:
        if run_ids:
            placeholders = ", ".join("?" for _ in run_ids)
            rows = snapshot.execute(
                "SELECT lower(coalesce(collector, '')) FROM ("
                "SELECT *, row_number() OVER (PARTITION BY run_id "
                "ORDER BY published_at DESC) AS revision_order FROM runs"
                f") WHERE revision_order = 1 AND run_id IN ({placeholders})",
                run_ids,
            ).fetchall()
            if len(rows) == len(set(run_ids)) and all("torch" in str(row[0]) for row in rows):
                return
            producer_rows = snapshot.execute(
                "SELECT DISTINCT run_id FROM artifact_registrations "
                f"WHERE run_id IN ({placeholders}) "
                "AND lower(coalesce(producer, '')) LIKE '%torch%'",
                run_ids,
            ).fetchall()
            if {str(row[0]) for row in producer_rows} == set(run_ids):
                return
        if artifact_ids:
            placeholders = ", ".join("?" for _ in artifact_ids)
            rows = snapshot.execute(
                "SELECT DISTINCT lower(coalesce(producer, '')) "
                "FROM artifact_registrations "
                f"WHERE artifact_id IN ({placeholders})",
                artifact_ids,
            ).fetchall()
            if any("torch" in str(row[0]) for row in rows):
                return
        raise DomainError(
            ErrorCode.COMPARISON_INVALID,
            "PyTorch operator analysis requires a torch.profiler-produced trace.",
            details={
                "next_tool": "import_artifact",
                "required_kind": "execution_trace",
                "required_producer": "torch.profiler",
            },
            remediation=(
                "Re-import the trace with kind='execution_trace'; Torch markers are detected "
                "automatically.",
                "If detection is ambiguous, set producer='torch.profiler' on import_artifact.",
            ),
        )

    def _limit(self, value: int | None) -> int:
        if value is None:
            return self.workspace.config.analysis.default_row_limit
        if value < 1 or value > self.workspace.config.analysis.max_row_limit:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Limit must be between 1 and {self.workspace.config.analysis.max_row_limit}.",
            )
        return value

    def _pinned_commit_id(self, value: str | None) -> str:
        if value is not None:
            if self.snapshot is not None and self.snapshot.commit.commit_id != value:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "Recipe snapshot does not match the requested corpus commit.",
                )
            return value
        if self.snapshot is not None:
            return self.snapshot.commit.commit_id
        return self.workspace.corpus.read_head().commit_id

    @contextmanager
    def _open_snapshot(self, corpus_commit_id: str) -> Iterator[Snapshot]:
        if self.snapshot is not None:
            if self.snapshot.commit.commit_id != corpus_commit_id:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "Recipe attempted to cross its pinned corpus snapshot.",
                )
            yield self.snapshot
            return
        with Catalog(self.workspace).open_snapshot(corpus_commit_id) as snapshot:
            yield snapshot
