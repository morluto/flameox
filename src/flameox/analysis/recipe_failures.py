from __future__ import annotations

from datetime import datetime

from flameox.analysis.recipe_context import RecipeContext
from flameox.analysis.recipe_models import (
    FailureAnalysisResult,
    FailureChangePoint,
    FailureCluster,
    FailurePopulationStatus,
)
from flameox.domain import CaptureStatus, ExecutionStatus, ValidationStatus, digest_model
from flameox.evidence_status import available_availability, empty_availability


class FailureRecipes(RecipeContext):
    def failures(
        self,
        *,
        limit: int | None = None,
        corpus_commit_id: str | None = None,
        source_state_id: str | None = None,
        environment_id: str | None = None,
        workload_definition_id: str | None = None,
        execution_status: tuple[ExecutionStatus, ...] = (),
        validation_status: tuple[ValidationStatus, ...] = (),
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
                parameters.extend(value.value for value in values)
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
                execution_status=ExecutionStatus(str(row[1])),
                capture_status=CaptureStatus(str(row[2])),
                validation_status=ValidationStatus(str(row[3])),
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
        population_status: FailurePopulationStatus
        empty_reason: str | None
        if workspace_run_count == 0:
            population_status = FailurePopulationStatus.EMPTY
            empty_reason = "no_runs"
        elif eligible_run_count == 0:
            population_status = FailurePopulationStatus.FILTERED_EMPTY
            empty_reason = "no_matching_runs"
        else:
            population_status = FailurePopulationStatus.OBSERVED
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
