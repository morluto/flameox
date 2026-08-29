from __future__ import annotations

from flameox.action_graph import ActionId, tool_action
from flameox.analysis.recipe_context import RecipeContext
from flameox.analysis.recipe_models import (
    Hotspot,
    HotspotResult,
    MeasurementSummary,
    MemoryAnalysisResult,
    MemoryPhaseGrowth,
    RuntimeResourceObservation,
    RuntimeResourceTotals,
    WritableRootObservation,
    parse_writable_root_observation,
)
from flameox.catalog import Snapshot
from flameox.domain import ProcessCancellationCause, digest_model
from flameox.evidence import numeric_value_from_columns
from flameox.evidence_scope import EvidenceScope, resolve_evidence_scope
from flameox.evidence_status import (
    available_availability,
    empty_availability,
    partial_availability,
    recoverable_empty_evidence,
    recoverable_unavailable_evidence,
    unavailable_availability,
)
from flameox.memory_query import MemoryFrameQuery, MemoryRanking


class HotspotRecipes(RecipeContext):
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
                    coverage={"frame_measurements": 0, "completely_symbolized": 0},
                    limitations=(
                        "No registered profile artifact is available for this input; "
                        "profile parsing is extractor-owned.",
                    ),
                    evidence=unavailable_availability("no_profile_artifact"),
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
                "ON f.frame_id = fm.frame_id AND f.artifact_id = fm.artifact_id WHERE "
                + where
                + " ORDER BY coalesce(fm.inclusive_value, fm.self_value, 0) DESC, "
                "fm.frame_id LIMIT ?",
                (*parameters, bounded),
            ).fetchall()
            symbolized_row = snapshot.execute(
                "SELECT count(*) FROM frame_measurements fm JOIN frames f "
                "ON f.frame_id = fm.frame_id AND f.artifact_id = fm.artifact_id WHERE "
                + where
                + " AND f.symbolization = 'complete'",
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
            coverage={
                "frame_measurements": total,
                "completely_symbolized": int(symbolized_row[0]),
            },
            limitations=(
                "Complete stacks remain in native artifacts; this result is a bounded "
                "frame aggregate.",
            ),
            evidence=(
                empty_availability("no_matching_hotspots")
                if not hotspots
                else available_availability()
            ),
        )

    @staticmethod
    def _profile_artifact_predicate(scope: EvidenceScope) -> tuple[str, tuple[object, ...]]:
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
        query: MemoryFrameQuery | None = None,
        corpus_commit_id: str | None = None,
    ) -> MemoryAnalysisResult:
        corpus_commit_id = self._pinned_commit_id(corpus_commit_id)
        bounded = self._limit(limit)
        query = query or MemoryFrameQuery()
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
                + " AND name LIKE 'memory.%' "
                "AND name NOT LIKE 'memory.frame_coverage.%' ORDER BY name LIMIT ?",
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
            (
                memory_hotspots,
                memory_hotspot_total,
                memory_view_available,
                memory_view_incomplete,
            ) = self._memory_hotspots(
                snapshot,
                scope,
                query,
                bounded,
            )
            resource_rows, resource_total = self._runtime_resources(
                snapshot,
                scope,
                bounded,
            )
            writable_rows = self._writable_root_observations(snapshot, scope, bounded)
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
        broadened_query = query.broadened()
        if memory_hotspots:
            hotspot_evidence = available_availability()
        elif memory_view_incomplete:
            recovery = (
                tool_action(ActionId.GET_NATIVE_VIEWER_PLAN, artifact_id=scope.artifact_ids[0])
                if scope.artifact_ids
                else tool_action(
                    ActionId.EXTRACT_MEMRAY,
                    run_id=scope.run_ids[0],
                    idempotency_key=digest_model(
                        {"action": ActionId.EXTRACT_MEMRAY, "run_id": scope.run_ids[0]}
                    ),
                )
            )
            hotspot_evidence = recoverable_unavailable_evidence(
                "memory_view_extraction_incomplete",
                next_action=recovery,
            )
            limitations.append(
                f"The selected {query.view.value} frame view was truncated during extraction."
            )
        elif not memory_view_available:
            hotspot_evidence = unavailable_availability("memory_view_not_extracted")
            limitations.append(
                f"The selected {query.view.value} frame view is unavailable in normalized evidence."
            )
        elif broadened_query != query:
            hotspot_evidence = recoverable_empty_evidence(
                "no_matching_memory_frames",
                next_action=tool_action(
                    ActionId.ANALYZE_MEMORY,
                    run_or_artifact=input_id,
                    limit=bounded,
                    query=broadened_query.model_dump(mode="json"),
                ),
            )
        else:
            hotspot_evidence = empty_availability("no_memory_frames_in_selected_view")
        has_runtime_evidence = bool(
            resource_rows
            or writable_rows
            or policy_termination is not None
            or (resource_total is not None and resource_total.run_count > 0)
        )
        if rows or memory_hotspots or phase_growth:
            evidence = available_availability()
        elif has_runtime_evidence:
            evidence = partial_availability(
                "memory_profile_not_extracted_runtime_evidence_present"
                if has_memory_profile
                else "no_memory_profile_artifact_runtime_evidence_present"
            )
        elif has_memory_profile and memory_run_id is not None:
            evidence = recoverable_unavailable_evidence(
                "memory_profile_not_extracted",
                next_action=tool_action(
                    ActionId.EXTRACT_MEMRAY,
                    run_id=memory_run_id,
                    idempotency_key=digest_model(
                        {"action": ActionId.EXTRACT_MEMRAY, "run_id": memory_run_id}
                    ),
                ),
            )
        else:
            evidence = unavailable_availability(
                "memory_profile_not_extracted"
                if has_memory_profile
                else "no_memory_profile_artifact"
            )
        return MemoryAnalysisResult(
            corpus_commit_id=snapshot.commit.commit_id,
            input_id=input_id,
            query=query,
            measurements=tuple(
                MeasurementSummary(
                    name=row[0],
                    value=numeric_value_from_columns(
                        row[1],
                        row[2],
                        field_name="memory measurement value",
                    ),
                    unit=row[3],
                    aggregation=row[4],
                    scope=row[5],
                )
                for row in rows
            ),
            hotspots=memory_hotspots,
            hotspot_total=memory_hotspot_total,
            hotspot_evidence=hotspot_evidence,
            phase_growth=tuple(phase_growth),
            limitations=tuple(limitations),
            runtime_resources=resource_rows,
            runtime_resource_totals=resource_total,
            writable_root_observations=writable_rows,
            evidence=evidence,
        )

    @staticmethod
    def _memory_hotspots(
        snapshot: Snapshot,
        scope: EvidenceScope,
        query: MemoryFrameQuery,
        limit: int,
    ) -> tuple[tuple[Hotspot, ...], int, bool, bool]:
        where, parameters = scope.predicate(
            run_column="fm.run_id",
            artifact_column="fm.artifact_id",
        )
        base_predicate = f"({where}) AND fm.metric = ?"
        base_values = (*parameters, query.view.metric)
        metric_count_row = snapshot.execute(
            "SELECT count(*) FROM frame_measurements fm WHERE " + base_predicate,
            base_values,
        ).fetchone()
        assert metric_count_row is not None
        measurement_name = {
            "high_watermark": "memory.peak",
            "retained_end": "memory.retained_end",
            "allocation_volume": "memory.allocated_bytes",
            "temporary": "memory.temporary",
        }[query.view.value]
        measurement_where, measurement_parameters = scope.predicate(
            run_column="m.run_id",
            artifact_column="m.artifact_id",
        )
        measurement_count_row = snapshot.execute(
            "SELECT count(*) FROM measurements m WHERE " + measurement_where + " AND m.name = ?",
            (*measurement_parameters, measurement_name),
        ).fetchone()
        assert measurement_count_row is not None
        coverage_name = f"memory.frame_coverage.{query.view.value}.complete"
        coverage_row = snapshot.execute(
            "SELECT count(*), min(coalesce(value_int, 0)) FROM measurements m WHERE "
            + measurement_where
            + " AND m.name = ?",
            (*measurement_parameters, coverage_name),
        ).fetchone()
        assert coverage_row is not None
        has_coverage = int(coverage_row[0]) > 0
        coverage_complete = has_coverage and int(coverage_row[1]) == 1
        predicates = [base_predicate]
        values: list[object] = [*parameters, query.view.metric]
        if query.project_only:
            predicates.append("f.source_state_id IS NOT NULL")
        if query.exclude_zero_self:
            predicates.append("coalesce(fm.self_value, 0) > 0")
        if query.include_file_prefixes:
            predicates.append(
                "("
                + " OR ".join(
                    "starts_with(coalesce(f.file, ''), ?)"
                    for _prefix in query.include_file_prefixes
                )
                + ")"
            )
            values.extend(query.include_file_prefixes)
        for prefix in query.exclude_file_prefixes:
            predicates.append("NOT starts_with(coalesce(f.file, ''), ?)")
            values.append(prefix)
        module_identity = (
            "coalesce(f.module, replace(regexp_replace(coalesce(f.file, ''), "
            "'\\.py$', ''), '/', '.'))"
        )
        if query.include_module_prefixes:
            predicates.append(
                "("
                + " OR ".join(
                    f"starts_with({module_identity}, ?)"
                    for _prefix in query.include_module_prefixes
                )
                + ")"
            )
            values.extend(query.include_module_prefixes)
        for prefix in query.exclude_module_prefixes:
            predicates.append(f"NOT starts_with({module_identity}, ?)")
            values.append(prefix)
        predicate = " AND ".join(f"({item})" for item in predicates)
        joined = (
            " FROM frame_measurements fm LEFT JOIN frames f "
            "ON f.frame_id = fm.frame_id AND f.artifact_id = fm.artifact_id WHERE " + predicate
        )
        count_row = snapshot.execute("SELECT count(*)" + joined, tuple(values)).fetchone()
        assert count_row is not None
        ranking = (
            "coalesce(fm.self_value, 0)"
            if query.ranking is MemoryRanking.SELF
            else "coalesce(fm.inclusive_value, fm.self_value, 0)"
        )
        rows = snapshot.execute(
            "SELECT fm.frame_id, f.function, f.file, f.line, fm.metric, "
            "fm.self_value, fm.inclusive_value, fm.unit, fm.sample_count"
            + joined
            + f" ORDER BY {ranking} DESC, fm.frame_id LIMIT ?",
            (*values, limit),
        ).fetchall()
        return (
            tuple(
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
            ),
            int(count_row[0]),
            int(metric_count_row[0]) > 0
            or coverage_complete
            or (not has_coverage and int(measurement_count_row[0]) > 0),
            has_coverage and not coverage_complete,
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
    ) -> tuple[tuple[RuntimeResourceObservation, ...], RuntimeResourceTotals]:
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
            RuntimeResourceObservation.model_validate(
                {
                    "run_id": str(row[0]),
                    "sampling_interval_ms": int(row[1]),
                    "minimum_free_bytes": int(row[2]) if row[2] is not None else None,
                    "staging_growth_bytes": int(row[3]) if row[3] is not None else None,
                    "peak_rss_bytes": int(row[4]) if row[4] is not None else None,
                    "policy_termination": (
                        ProcessCancellationCause(str(row[5])) if row[5] is not None else None
                    ),
                    "unavailable_metrics": tuple(str(item) for item in (row[6] or ())),
                }
            )
            for row in rows[:limit]
        )
        return observations, RuntimeResourceTotals(
            run_count=int(total_row[0]),
            minimum_free_bytes=(int(total_row[1]) if total_row[1] is not None else None),
            total_staging_growth_bytes=(int(total_row[2]) if total_row[2] is not None else None),
            maximum_peak_rss_bytes=(int(total_row[3]) if total_row[3] is not None else None),
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
            parse_writable_root_observation(
                {
                    "run_id": str(row[0]),
                    "writable_root_identity": str(row[1]),
                    "target_path": str(row[2]),
                    "growth_bytes": int(row[3]) if row[3] is not None else None,
                    "available": bool(row[4]),
                    "unavailable_reason": str(row[5]) if row[5] is not None else None,
                }
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
