from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict

from flamo.catalog import Catalog
from flamo.domain import DomainError, ErrorCode
from flamo.storage import RunStore, Workspace


class Hotspot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: str
    function: str | None
    file: str | None
    line: int | None
    metric: str
    self_value: int | None
    inclusive_value: int | None
    unit: str
    sample_count: int | None


class HotspotResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    hotspots: tuple[Hotspot, ...]
    total: int
    returned: int
    truncated: bool
    coverage: dict[str, int]
    limitations: tuple[str, ...]


class MeasurementSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    value_int: int | None
    value_float: float | None
    unit: str
    aggregation: str
    scope: str


class MemoryAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    measurements: tuple[MeasurementSummary, ...]
    hotspots: tuple[Hotspot, ...]
    limitations: tuple[str, ...]


class ExecutionObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str
    kind: str
    name: str
    value_json: str
    file: str | None
    line_from: int | None
    line_to: int | None
    context: str | None
    evidence_level: str


class ExecutionAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    observations: tuple[ExecutionObservation, ...]
    total: int
    returned: int
    truncated: bool
    limitations: tuple[str, ...]


class OperatorSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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


class PyTorchAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    corpus_commit_id: str
    input_id: str
    operators: tuple[OperatorSummary, ...]
    total: int
    returned: int
    truncated: bool
    coverage: dict[str, bool]
    limitations: tuple[str, ...]


class FailureCluster(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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


class FailureChangePoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observed_date: str
    run_count: int
    previous_run_count: int | None


class FailureAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    corpus_commit_id: str
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


class ScalingPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: str
    block_id: str | None
    input_value: float | None
    value: float
    dispersion: float
    unit: str
    sample_count: int
    environment_count: int


class ScalingFit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    coefficients: tuple[float, ...]
    residual_rms: float
    r_squared: float | None
    aicc: float | None
    observation_count: int
    supported_min: float
    supported_max: float


class ScalingAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    corpus_commit_id: str
    experiment_id: str
    metric: str
    points: tuple[ScalingPoint, ...]
    attempted_trials: int
    succeeded_trials: int
    failed_trials: int
    complete_blocks: int
    fits: tuple[ScalingFit, ...]
    conclusion: str
    environment_stable: bool
    warnings: tuple[str, ...]
    limitations: tuple[str, ...] = (
        "Points are per-trial medians; statistical decisions belong to frozen run-set comparisons.",
    )


class RecipeService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def hotspots(self, input_id: str, *, limit: int | None = None) -> HotspotResult:
        bounded = self._limit(limit)
        run_ids, artifact_ids = self._scope(input_id)
        where, parameters = self._scope_where(
            run_ids,
            artifact_ids,
            run_column="fm.run_id",
            artifact_column="fm.artifact_id",
        )
        with Catalog(self.workspace).open_snapshot() as snapshot:
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
        )

    def memory(self, input_id: str, *, limit: int | None = None) -> MemoryAnalysisResult:
        bounded = self._limit(limit)
        run_ids, artifact_ids = self._scope(input_id)
        where, parameters = self._scope_where(
            run_ids,
            artifact_ids,
            run_column="run_id",
            artifact_column="artifact_id",
        )
        with Catalog(self.workspace).open_snapshot() as snapshot:
            rows = snapshot.execute(
                "SELECT name, value_int, value_float, unit, aggregation, scope "
                "FROM measurements WHERE "
                + where
                + " AND name LIKE 'memory.%' ORDER BY name LIMIT ?",
                (*parameters, bounded),
            ).fetchall()
        hotspot_result = self.hotspots(input_id, limit=bounded)
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
            limitations=(
                "High-water-mark, retained-end, and allocation volume are distinct "
                "concepts and are not substituted for one another.",
            ),
        )

    def execution(
        self,
        input_id: str,
        *,
        limit: int | None = None,
    ) -> ExecutionAnalysisResult:
        bounded = self._limit(limit)
        run_ids, artifact_ids = self._scope(input_id)
        where, parameters = self._scope_where(
            run_ids,
            artifact_ids,
            run_column="run_id",
            artifact_column="artifact_id",
        )
        with Catalog(self.workspace).open_snapshot() as snapshot:
            count_row = snapshot.execute(
                "SELECT count(*) FROM observations WHERE " + where,
                parameters,
            ).fetchone()
            assert count_row is not None
            total = int(count_row[0])
            rows = snapshot.execute(
                "SELECT observation_id, kind, name, value_json, file, line_from, "
                "line_to, context, evidence_level FROM observations WHERE "
                + where
                + " ORDER BY file, line_from, line_to, observation_id LIMIT ?",
                (*parameters, bounded),
            ).fetchall()
        observations = tuple(
            ExecutionObservation(
                observation_id=row[0],
                kind=row[1],
                name=row[2],
                value_json=row[3],
                file=row[4],
                line_from=row[5],
                line_to=row[6],
                context=row[7],
                evidence_level=row[8],
            )
            for row in rows
        )
        return ExecutionAnalysisResult(
            corpus_commit_id=snapshot.commit.commit_id,
            input_id=input_id,
            observations=observations,
            total=total,
            returned=len(observations),
            truncated=total > len(observations),
            limitations=(
                "Coverage proves that a path executed, not why it executed or which "
                "values controlled it.",
            ),
        )

    def pytorch(
        self,
        input_id: str,
        *,
        limit: int | None = None,
    ) -> PyTorchAnalysisResult:
        bounded = self._limit(limit)
        run_ids, artifact_ids = self._scope(input_id)
        self._require_pytorch_source(run_ids, artifact_ids)
        where, parameters = self._scope_where(
            run_ids,
            artifact_ids,
            run_column="fm.run_id",
            artifact_column="fm.artifact_id",
        )
        with Catalog(self.workspace).open_snapshot() as snapshot:
            count_row = snapshot.execute(
                "SELECT count(DISTINCT fm.frame_id) FROM frame_measurements fm WHERE " + where,
                parameters,
            ).fetchone()
            assert count_row is not None
            total = int(count_row[0])
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
        operators_list: list[OperatorSummary] = []
        device_time_present = False
        synchronization_present = False
        for row in rows:
            category = str(row[2]) if row[2] is not None else None
            category_lower = (category or "").lower()
            operator = str(row[1])
            operator_lower = operator.lower()
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
            operators_list.append(
                OperatorSummary(
                    frame_id=str(row[0]),
                    operator=operator,
                    category=category,
                    self_cpu_ns=None if is_device else int(row[3]),
                    total_cpu_ns=None if is_device else inclusive,
                    device_ns=inclusive if is_device else None,
                    inclusive_ns=inclusive,
                    event_count=int(row[5]),
                    synchronization=synchronization,
                )
            )
        operators = tuple(operators_list)
        limitations = [
            "Operator categories and durations come from the exported torch.profiler trace.",
            "Nested operator durations can overlap; self time subtracts direct nested slices.",
        ]
        if not device_time_present:
            limitations.append("The trace contains no recognized accelerator kernel categories.")
        limitations.extend(
            (
                "Input shapes were not present in normalized trace evidence.",
                "Per-operator allocation bytes were not present in normalized trace evidence.",
                "Warm-up separation requires profiler phase annotations.",
            )
        )
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
                "input_shapes": False,
                "memory_allocations": False,
                "synchronization": synchronization_present,
                "warmup_phases": False,
            },
            limitations=tuple(limitations),
        )

    def failures(self, *, limit: int | None = None) -> FailureAnalysisResult:
        bounded = self._limit(limit)
        query = """
            WITH latest AS (
                SELECT *,
                    row_number() OVER (
                        PARTITION BY run_id ORDER BY published_at DESC
                    ) AS revision_order
                FROM runs
            ),
            failed AS (
                SELECT * FROM latest
                WHERE revision_order = 1
                  AND (
                    execution_status NOT IN ('succeeded', 'not_applicable')
                    OR capture_status IN ('failed', 'cancelled')
                    OR validation_status = 'failed'
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
        with Catalog(self.workspace).open_snapshot() as snapshot:
            count_row = snapshot.execute(query + " SELECT count(*) FROM clusters").fetchone()
            assert count_row is not None
            total = int(count_row[0])
            rows = snapshot.execute(
                query + " SELECT collector, execution_status, capture_status, "
                "validation_status, exit_code, workload_definition_id, "
                "environment_id, source_state_id, run_count, first_seen, last_seen, "
                "representative_run_id "
                "FROM clusters ORDER BY run_count DESC, last_seen DESC LIMIT ?",
                (bounded,),
            ).fetchall()
            change_rows = snapshot.execute(
                query
                + " , daily AS (SELECT CAST(created_at AS DATE) AS observed_date, "
                "count(*) AS run_count FROM failed GROUP BY observed_date), "
                "with_previous AS (SELECT observed_date, run_count, "
                "lag(run_count) OVER (ORDER BY observed_date) AS previous_run_count "
                "FROM daily) SELECT observed_date, run_count, previous_run_count "
                "FROM with_previous WHERE previous_run_count IS NULL "
                "OR run_count <> previous_run_count ORDER BY observed_date"
            ).fetchall()
            coverage_row = snapshot.execute(
                query
                + " SELECT count(*), "
                "count(*) FILTER (WHERE source_state_id IS NOT NULL), "
                "count(*) FILTER (WHERE EXISTS (SELECT 1 FROM artifact_registrations a "
                "WHERE a.run_id = failed.run_id)), "
                "count(*) FILTER (WHERE EXISTS (SELECT 1 FROM frame_measurements fm "
                "WHERE fm.run_id = failed.run_id)) FROM failed"
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
        )

    def scaling(self, experiment_id: str) -> ScalingAnalysisResult:
        with Catalog(self.workspace).open_snapshot() as snapshot:
            experiment_row = snapshot.execute(
                "SELECT primary_metric FROM experiments WHERE experiment_id = ? "
                "ORDER BY published_at DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
            if experiment_row is None:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Unknown experiment {experiment_id!r}.",
                )
            metric = str(experiment_row[0])
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
                "SELECT v.name, t.block_id, t.parameter_value_int, "
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
        grouped: dict[
            tuple[str, str | None, float | None, str],
            tuple[list[float], set[str]],
        ] = {}
        for row in rows:
            input_value = (
                float(row[2])
                if row[2] is not None
                else float(row[3])
                if row[3] is not None
                else None
            )
            key = (
                str(row[0]),
                str(row[1]) if row[1] is not None else None,
                input_value,
                str(row[6]),
            )
            values, environments = grouped.setdefault(key, ([], set()))
            if row[5] is not None:
                values.append(float(row[5]))
            environments.add(str(row[4]))
        points_list: list[ScalingPoint] = []
        for (variant, block_id, input_value, unit), (
            values,
            environments,
        ) in sorted(grouped.items(), key=lambda item: (item[0][1] or "", item[0][0])):
            if not values:
                continue
            median = float(np.median(values))
            dispersion = float(np.median(np.abs(np.asarray(values) - median)))
            points_list.append(
                ScalingPoint(
                    variant=variant,
                    block_id=block_id,
                    input_value=input_value,
                    value=median,
                    dispersion=dispersion,
                    unit=unit,
                    sample_count=len(values),
                    environment_count=len(environments),
                )
            )
        points = tuple(points_list)
        fits = self._scaling_fits(points)
        conclusion = "inconclusive"
        comparable = [fit for fit in fits if fit.aicc is not None]
        if len(comparable) >= 2:
            ordered = sorted(
                comparable,
                key=lambda fit: fit.aicc if fit.aicc is not None else math.inf,
            )
            first_aicc = ordered[0].aicc
            second_aicc = ordered[1].aicc
            assert first_aicc is not None and second_aicc is not None
            if second_aicc - first_aicc >= 2:
                conclusion = f"descriptive_best_fit:{ordered[0].model}"
        warnings = [
            "Fits describe only the measured input range and must not be extrapolated."
        ]
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
            attempted_trials=int(trial_row[0]),
            succeeded_trials=int(trial_row[1]),
            failed_trials=int(trial_row[2]),
            complete_blocks=int(complete_row[0]),
            fits=fits,
            conclusion=conclusion,
            environment_stable=environment_stable,
            warnings=tuple(warnings),
        )

    def _scaling_fits(self, points: tuple[ScalingPoint, ...]) -> tuple[ScalingFit, ...]:
        numeric = [
            point
            for point in points
            if point.input_value is not None
            and point.input_value > 0
            and math.isfinite(point.input_value)
            and math.isfinite(point.value)
        ]
        if len(numeric) < 3 or len({point.input_value for point in numeric}) < 2:
            return ()
        x = np.asarray([point.input_value for point in numeric], dtype=float)
        y = np.asarray([point.value for point in numeric], dtype=float)
        candidates = {
            "constant": np.column_stack((np.ones_like(x),)),
            "logarithmic": np.column_stack((np.ones_like(x), np.log(x))),
            "linear": np.column_stack((np.ones_like(x), x)),
            "n_log_n": np.column_stack((np.ones_like(x), x * np.log(x))),
            "quadratic": np.column_stack((np.ones_like(x), x, x * x)),
        }
        fits: list[ScalingFit] = []
        for name, design in candidates.items():
            observation_count, parameter_count = design.shape
            if observation_count <= parameter_count:
                continue
            coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            predicted = design @ coefficients
            residuals = y - predicted
            rss = float(np.dot(residuals, residuals))
            residual_rms = math.sqrt(rss / observation_count)
            total = float(np.dot(y - np.mean(y), y - np.mean(y)))
            r_squared = 1 - rss / total if total > 0 else None
            aic = observation_count * math.log(
                max(rss / observation_count, np.finfo(float).tiny)
            ) + 2 * parameter_count
            aicc = (
                aic
                + (2 * parameter_count * (parameter_count + 1))
                / (observation_count - parameter_count - 1)
                if observation_count > parameter_count + 1
                else None
            )
            fits.append(
                ScalingFit(
                    model=name,
                    coefficients=tuple(float(value) for value in coefficients),
                    residual_rms=residual_rms,
                    r_squared=r_squared,
                    aicc=aicc,
                    observation_count=observation_count,
                    supported_min=float(np.min(x)),
                    supported_max=float(np.max(x)),
                )
            )
        return tuple(
            sorted(
                fits,
                key=lambda fit: (
                    fit.aicc is None,
                    fit.aicc if fit.aicc is not None else math.inf,
                    fit.model,
                ),
            )
        )

    def _scope(self, input_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if input_id.startswith("sha256:"):
            return (), (input_id,)
        run = RunStore(self.workspace).read(input_id)
        return (run.run_id,), tuple(item.artifact_id for item in run.artifacts)

    def _require_pytorch_source(
        self,
        run_ids: tuple[str, ...],
        artifact_ids: tuple[str, ...],
    ) -> None:
        if run_ids and all(
            RunStore(self.workspace).read(run_id).collector == "torch"
            for run_id in run_ids
        ):
            return
        if artifact_ids:
            placeholders = ", ".join("?" for _ in artifact_ids)
            with Catalog(self.workspace).open_snapshot() as snapshot:
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
        )

    def _scope_where(
        self,
        run_ids: tuple[str, ...],
        artifact_ids: tuple[str, ...],
        *,
        run_column: str,
        artifact_column: str,
    ) -> tuple[str, tuple[object, ...]]:
        if run_ids:
            return f"{run_column} = ?", (run_ids[0],)
        if artifact_ids:
            return f"{artifact_column} = ?", (artifact_ids[0],)
        raise DomainError(ErrorCode.WORKSPACE_INVALID, "Analysis input has no scope.")

    def _limit(self, value: int | None) -> int:
        if value is None:
            return self.workspace.config.analysis.default_row_limit
        if value < 1 or value > self.workspace.config.analysis.max_row_limit:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Limit must be between 1 and {self.workspace.config.analysis.max_row_limit}.",
            )
        return value
