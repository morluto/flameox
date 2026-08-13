from __future__ import annotations

import math
import warnings
from typing import Any, Literal, cast

import numpy as np
from scipy.stats import bootstrap, spearmanr
from statsmodels.api import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson

from flameox.analysis.recipe_context import RecipeContext
from flameox.analysis.recipe_models import (
    ScalingAnalysisResult,
    ScalingCorrelatedHotspot,
    ScalingFit,
    ScalingPoint,
    ScalingTrialSummary,
)
from flameox.domain import ConfidenceInterval, DomainError, ErrorCode
from flameox.evidence_status import (
    EvidenceAvailability,
    available_availability,
    empty_availability,
    partial_availability,
    unavailable_availability,
)


class ScalingRecipes(RecipeContext):
    @staticmethod
    def _input_identity(
        integer_value: object,
        floating_value: object,
    ) -> tuple[float | None, Literal["integer", "floating"] | None]:
        if integer_value is not None:
            return float(cast(Any, integer_value)), "integer"
        if floating_value is not None:
            return float(cast(Any, floating_value)), "floating"
        return None, None

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
                "SELECT t.trial_id, v.name, t.block_id, t.parameter_value_int, "
                "t.parameter_value_float, r.environment_id, "
                "coalesce(CAST(m.value_int AS DOUBLE), m.value_float), m.unit "
                "FROM (SELECT DISTINCT trial_id, experiment_id, variant_id, "
                "run_id, block_id, outcome, parameter_value_int, "
                "parameter_value_float FROM trials) t "
                "JOIN (SELECT DISTINCT variant_id, name FROM variants) v "
                "ON v.variant_id = t.variant_id "
                "JOIN current_runs r ON r.run_id = t.run_id "
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
                "AND f.artifact_id = fm.artifact_id "
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
            tuple[
                str,
                str,
                str | None,
                float | None,
                Literal["integer", "floating"] | None,
                str,
                str,
            ],
            list[float],
        ] = {}
        for row in rows:
            input_value, input_kind = self._input_identity(row[3], row[4])
            key = (
                str(row[0]),
                str(row[1]),
                str(row[2]) if row[2] is not None else None,
                input_value,
                input_kind,
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
            input_kind,
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
                    input_kind=input_kind,
                    median=median,
                    dispersion=float(np.median(np.abs(np.asarray(values) - median))),
                    unit=unit,
                    raw_sample_count=len(values),
                    environment_id=environment_id,
                )
            )
        point_groups: dict[
            tuple[str, float | None, Literal["integer", "floating"] | None, str],
            list[ScalingTrialSummary],
        ] = {}
        for trial in trials:
            point_groups.setdefault(
                (trial.variant, trial.input_value, trial.input_kind, trial.unit),
                [],
            ).append(trial)
        points_list: list[ScalingPoint] = []
        for (variant, input_value, input_kind, unit), group in sorted(
            point_groups.items(),
            key=lambda item: (item[0][0], item[0][1] or -math.inf, item[0][2] or ""),
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
                    input_kind=input_kind,
                    value=median,
                    dispersion=dispersion,
                    confidence_interval=(
                        ConfidenceInterval(low=low, high=high, level=confidence_level)
                        if low is not None and high is not None
                        else None
                    ),
                    unit=unit,
                    sample_count=len(group),
                    raw_sample_count=sum(trial.raw_sample_count for trial in group),
                    environment_count=len({trial.environment_id for trial in group}),
                )
            )
        points = tuple(points_list)
        fits = self._scaling_fits(points, confidence_level=confidence_level)
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
        input_kinds_by_variant: dict[str, set[Literal["integer", "floating"]]] = {}
        for point in points:
            if (
                point.input_value is not None
                and point.input_value > 0
                and math.isfinite(point.input_value)
                and math.isfinite(point.value)
                and point.input_kind is not None
            ):
                input_kinds_by_variant.setdefault(point.variant, set()).add(point.input_kind)
        mixed_input_kind_variants = {
            variant
            for variant, input_kinds in input_kinds_by_variant.items()
            if input_kinds == {"integer", "floating"}
        }
        if mixed_input_kind_variants:
            warnings.append(
                "Fits were excluded for variants with mixed integer and floating scaling inputs: "
                + ", ".join(sorted(mixed_input_kind_variants))
                + "."
            )
        environment_stable = all(point.environment_count == 1 for point in points)
        if not environment_stable:
            warnings.append("Environment identity varies within at least one scaling point.")
        attempted_trials = int(trial_row[0])
        succeeded_trials = int(trial_row[1])
        failed_trials = int(trial_row[2])
        evidence, measurement_warning = self._scaling_evidence(
            points,
            attempted_trials=attempted_trials,
            succeeded_trials=succeeded_trials,
            measured_trials=len(trials),
        )
        if measurement_warning is not None:
            warnings.append(measurement_warning)
        return ScalingAnalysisResult(
            corpus_commit_id=snapshot.commit.commit_id,
            experiment_id=experiment_id,
            metric=metric,
            points=points,
            trials=tuple(trials),
            attempted_trials=attempted_trials,
            succeeded_trials=succeeded_trials,
            failed_trials=failed_trials,
            complete_blocks=int(complete_row[0]),
            fits=fits,
            correlated_hotspots=correlated_hotspots,
            conclusion=conclusion,
            warnings=tuple(warnings),
            evidence=evidence,
        )

    @staticmethod
    def _scaling_evidence(
        points: tuple[ScalingPoint, ...],
        *,
        attempted_trials: int,
        succeeded_trials: int,
        measured_trials: int,
    ) -> tuple[EvidenceAvailability, str | None]:
        if points and succeeded_trials > measured_trials:
            return (
                partial_availability("primary_metric_measurements_partial"),
                "Some succeeded trials published no measurements matching the experiment's "
                "primary metric.",
            )
        if points:
            return available_availability(), None
        if succeeded_trials:
            return (
                unavailable_availability("primary_metric_measurements_unavailable"),
                "Succeeded trials published no measurements matching the experiment's "
                "primary metric.",
            )
        if attempted_trials:
            return empty_availability("no_succeeded_trials"), None
        return empty_availability("no_scaling_trials"), None

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
            if {point.input_kind for point in numeric} == {"integer", "floating"}:
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
                Literal["integer", "floating"],
                float,
            ],
            float,
        ] = {}
        for row in rows:
            input_value, input_kind = self._input_identity(row[2], row[3])
            if input_value is None or not math.isfinite(input_value):
                continue
            assert input_kind is not None
            key = (
                str(row[0]),
                str(row[1]),
                str(row[4]),
                str(row[5]) if row[5] is not None else None,
                str(row[6]) if row[6] is not None else None,
                int(cast(Any, row[7])) if row[7] is not None else None,
                str(row[8]),
                str(row[9]),
                input_kind,
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
            _input_kind,
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
                item.validated_copy(
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
