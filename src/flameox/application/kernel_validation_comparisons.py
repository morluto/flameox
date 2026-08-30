from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, cast

from pydantic import Field, JsonValue, computed_field, field_validator, model_validator

from flameox.analysis import PairedDifferenceEstimate, paired_median_difference
from flameox.application.analysis_provenance import (
    AnalysisProvenanceInput,
    build_analysis_provenance,
    context_references,
)
from flameox.catalog import Catalog, Snapshot, SnapshotHandle
from flameox.domain import (
    AnalysisRecord,
    ComparisonDecision,
    ComparisonValidity,
    ConfidenceInterval,
    DomainError,
    ErrorCode,
    EvidenceReference,
    Experiment,
    ExperimentRole,
    MetricPolarity,
    MetricSource,
    RunSet,
    digest_model,
)
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import ControlRecordStore, Workspace

_LOWER_IS_BETTER = {"max_abs_error", "max_rel_error", "mse", "rmse"}
_HIGHER_IS_BETTER = {"psnr", "cosine_similarity"}
_MAX_COMPARISON_ROWS = 50_000

type _DimensionValue = str | int | float | bool


class KernelValidationInputIdentity(ContractModel):
    dtype: str
    shape: Annotated[tuple[int, ...], Field(max_length=16)]
    role: str | None = None

    @field_validator("shape")
    @classmethod
    def nonnegative_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(item < 0 for item in value):
            raise ValueError("input shape dimensions must be nonnegative")
        return value


class KernelValidationPsnrProfile(ContractModel):
    identity_quality: Literal["exact"] = "exact"
    data_range: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    log_base: Literal[10] = 10
    reduction: Literal["mean_squared_error"] = "mean_squared_error"
    zero_mse_convention: Literal["positive_infinity"] = "positive_infinity"


class KernelValidationMetricSelector(ContractModel):
    case_id: str
    dimensions: Annotated[dict[str, _DimensionValue], Field(max_length=32)] = Field(
        default_factory=dict
    )
    inputs: Annotated[dict[str, KernelValidationInputIdentity], Field(max_length=32)] = Field(
        default_factory=dict
    )
    seed: int | None = None
    output_name: str
    output_dtype: str
    output_shape: Annotated[tuple[int, ...], Field(max_length=16)]
    metric_name: Literal[
        "max_abs_error",
        "max_rel_error",
        "mse",
        "rmse",
        "psnr",
        "cosine_similarity",
    ]
    unit: str
    comparator: Literal["<=", ">="]
    threshold: Annotated[float, Field(allow_inf_nan=False)]
    psnr_profile: KernelValidationPsnrProfile | None = None
    device: str

    @field_validator("dimensions")
    @classmethod
    def finite_dimensions(cls, value: dict[str, _DimensionValue]) -> dict[str, _DimensionValue]:
        if any(isinstance(item, float) and not math.isfinite(item) for item in value.values()):
            raise ValueError("case dimensions must be finite")
        return value

    @field_validator("output_shape")
    @classmethod
    def nonnegative_output_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(item < 0 for item in value):
            raise ValueError("output shape dimensions must be nonnegative")
        return value

    @model_validator(mode="after")
    def coherent_metric_contract(self) -> KernelValidationMetricSelector:
        expected = "<=" if self.metric_name in _LOWER_IS_BETTER else ">="
        if self.comparator != expected:
            raise ValueError(f"{self.metric_name} requires comparator {expected}")
        if self.metric_name == "psnr" and self.psnr_profile is None:
            raise ValueError("confirmatory psnr selectors require an exact profile")
        if self.metric_name != "psnr" and self.psnr_profile is not None:
            raise ValueError("only psnr selectors may carry a PSNR profile")
        return self


class KernelValidationComparisonProtocol(ContractModel):
    reference_name: str
    reference_version: str | None = None
    reference_identity: str
    case_population: Annotated[
        tuple[KernelValidationMetricSelector, ...], Field(min_length=1, max_length=256)
    ]
    environment_policy: Literal["exact_across_treatments"] = "exact_across_treatments"
    missing_data_policy: Literal["invalidate"] = "invalidate"

    @computed_field(return_type=str)  # type: ignore[prop-decorator]
    @property
    def protocol_id(self) -> str:
        return digest_model(self.model_dump(mode="json", exclude={"protocol_id"}))

    @model_validator(mode="after")
    def primary_population_has_one_metric_contract(
        self,
    ) -> KernelValidationComparisonProtocol:
        identities = {
            digest_model(selector.model_dump(mode="json")) for selector in self.case_population
        }
        if len(identities) != len(self.case_population):
            raise ValueError("kernel-validation protocol selectors must be unique")
        metric_contracts = {
            (selector.metric_name, selector.unit) for selector in self.case_population
        }
        if len(metric_contracts) != 1:
            raise ValueError("one protocol must declare one primary metric and unit")
        return self


class KernelValidationCompareRequest(ContractModel):
    baseline_run_set_id: str
    candidate_run_set_id: str
    experiment_id: str | None = None
    protocol: KernelValidationComparisonProtocol | None = None

    @model_validator(mode="after")
    def protocol_is_bound_to_an_experiment(self) -> KernelValidationCompareRequest:
        if (self.experiment_id is None) != (self.protocol is None):
            raise ValueError(
                "confirmatory kernel-validation comparison requires both experiment and protocol"
            )
        return self


class KernelFiniteObservedValue(ContractModel):
    kind: Literal["finite"] = "finite"
    value: Annotated[float, Field(allow_inf_nan=False)]


class KernelPositiveInfinityObservedValue(ContractModel):
    kind: Literal["positive_infinity"] = "positive_infinity"
    reason: Literal["zero_mse_exact_agreement"]


class KernelUnavailableObservedValue(ContractModel):
    kind: Literal["unavailable"] = "unavailable"
    status: Literal["inconclusive", "unsupported"]
    limitation: str


type KernelObservedValue = Annotated[
    KernelFiniteObservedValue
    | KernelPositiveInfinityObservedValue
    | KernelUnavailableObservedValue,
    Field(discriminator="kind"),
]


class KernelMetricChangeKind(StrEnum):
    FINITE = "finite"
    POSITIVE_INFINITY = "positive_infinity"
    NEGATIVE_INFINITY = "negative_infinity"
    UNDEFINED = "undefined"


class KernelMetricDirection(StrEnum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    UNASSESSABLE = "unassessable"


class KernelValidationMetricPair(ContractModel):
    pair_id: str
    semantic_identity_id: str
    semantic_identity: dict[str, JsonValue]
    independent_unit_id: str
    baseline_run_id: str
    candidate_run_id: str
    baseline_artifact_id: str
    candidate_artifact_id: str
    case_id: str
    output_name: str
    metric_name: str
    unit: str
    comparator: str
    threshold: float | None
    polarity: MetricPolarity
    baseline_value: KernelObservedValue
    candidate_value: KernelObservedValue
    baseline_status: str
    candidate_status: str
    status_transition: str
    change_kind: KernelMetricChangeKind
    signed_change: float | None = None
    absolute_change: float | None = None
    relative_change_percent: float | None = None
    relative_change_unavailable_reason: str | None = None
    direction: KernelMetricDirection


class KernelValidationUnmatchedMetric(ContractModel):
    independent_unit_id: str
    run_id: str
    artifact_id: str
    semantic_identity_id: str
    case_id: str
    output_name: str
    metric_name: str
    status: str


class KernelValidationCompatibilityMismatch(ContractModel):
    independent_unit_id: str
    case_id: str
    output_name: str
    metric_name: str
    baseline_semantic_identity_id: str
    candidate_semantic_identity_id: str
    reasons: tuple[str, ...]


class KernelValidationMetricAggregate(ContractModel):
    semantic_identity_id: str
    metric_name: str
    unit: str
    polarity: MetricPolarity
    pair_count: int
    finite_pair_count: int
    positive_infinity_pair_count: int
    median_signed_change: float | None = None
    direction: KernelMetricDirection


class KernelValidationComparison(ContractModel):
    comparison_id: str
    baseline_run_set_id: str
    candidate_run_set_id: str
    experiment_id: str | None = None
    protocol_identity_id: str | None = None
    metric: str | None = None
    unit: str | None = None
    metric_source: Literal[MetricSource.KERNEL_VALIDATION] = MetricSource.KERNEL_VALIDATION
    polarity: MetricPolarity | None = None
    estimand: Literal["median_blockwise_median_difference"] | None = None
    practical_threshold: float | None = None
    estimate: float | None = None
    confidence_interval: ConfidenceInterval | None = None
    method: str
    independent_unit: Literal["randomized_block", "single_run"]
    complete_pair_n: int
    decision: ComparisonDecision
    validity: ComparisonValidity
    pairs: Annotated[tuple[KernelValidationMetricPair, ...], Field(max_length=_MAX_COMPARISON_ROWS)]
    aggregates: Annotated[
        tuple[KernelValidationMetricAggregate, ...], Field(max_length=_MAX_COMPARISON_ROWS)
    ]
    compatibility_mismatches: Annotated[
        tuple[KernelValidationCompatibilityMismatch, ...],
        Field(max_length=_MAX_COMPARISON_ROWS),
    ] = ()
    baseline_only: Annotated[
        tuple[KernelValidationUnmatchedMetric, ...], Field(max_length=_MAX_COMPARISON_ROWS)
    ] = ()
    candidate_only: Annotated[
        tuple[KernelValidationUnmatchedMetric, ...], Field(max_length=_MAX_COMPARISON_ROWS)
    ] = ()
    mismatches: tuple[str, ...] = ()
    input_artifact_ids: tuple[str, ...] = ()


class KernelValidationComparisonResult(ContractModel):
    comparison: KernelValidationComparison
    baseline_run_set: RunSet
    candidate_run_set: RunSet
    corpus_commit_id: str
    analysis: AnalysisRecord | None = None
    evidence: tuple[EvidenceReference, ...] = ()
    materialized_commit_id: str | None = None


@dataclass(frozen=True, slots=True)
class _Observation:
    independent_unit_id: str
    run_id: str
    artifact_id: str
    environment_id: str
    semantic_identity: dict[str, JsonValue]
    semantic_identity_id: str
    location_key: tuple[str, str, str, str]
    case_id: str
    output_name: str
    metric_name: str
    unit: str
    comparator: str | None
    threshold: float | None
    profile: dict[str, JsonValue] | None
    value: KernelObservedValue
    status: str
    coverage_complete: bool


@dataclass(frozen=True, slots=True)
class _ComparisonInputs:
    baseline: RunSet
    candidate: RunSet
    handle: SnapshotHandle


class KernelValidationComparisonService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.run_sets = ControlRecordStore(
            workspace,
            kind="run_sets",
            model=RunSet,
            id_field="run_set_id",
        )
        self.experiments = ControlRecordStore(
            workspace,
            kind="experiments",
            model=Experiment,
            id_field="experiment_id",
        )
        self.publisher = GenerationPublisher(workspace)

    def compare(
        self,
        request: KernelValidationCompareRequest,
    ) -> KernelValidationComparisonResult:
        inputs = self._inputs(request)
        with Catalog(self.workspace).open_snapshot(inputs.handle) as snapshot:
            return self._compare_at_snapshot(request, snapshot, inputs)

    def record(
        self,
        request: KernelValidationCompareRequest,
    ) -> KernelValidationComparisonResult:
        started = datetime.now(UTC)
        result = self.compare(request)
        completed = datetime.now(UTC)
        comparison = result.comparison
        result_digest = digest_model(comparison.model_dump(mode="json"))
        input_run_ids = tuple(
            dict.fromkeys(
                member.run_id
                for run_set in (result.baseline_run_set, result.candidate_run_set)
                for member in run_set.members
            )
        )
        provenance = build_analysis_provenance(
            AnalysisProvenanceInput(
                recipe="compare_kernel_validation",
                parameters=cast(dict[str, JsonValue], request.model_dump(mode="json")),
                corpus_commit_id=result.corpus_commit_id,
                input_run_ids=input_run_ids,
                input_artifact_ids=comparison.input_artifact_ids,
                result_digest=result_digest,
                coverage={
                    "paired_metrics": len(comparison.pairs),
                    "metric_aggregates": len(comparison.aggregates),
                    "compatibility_mismatches": len(comparison.compatibility_mismatches),
                    "baseline_only": len(comparison.baseline_only),
                    "candidate_only": len(comparison.candidate_only),
                },
                limitations=comparison.mismatches,
                started_at=started,
                completed_at=completed,
                references=context_references(
                    artifact_ids=comparison.input_artifact_ids,
                    run_set_ids=(
                        comparison.baseline_run_set_id,
                        comparison.candidate_run_set_id,
                    ),
                ),
            )
        )
        rows = provenance.rows()
        rows.update(self._comparison_rows(comparison))
        published = self.publisher.publish_rows_idempotent(
            rows,
            publisher="flameox.kernel_validation_comparisons",
            publisher_version="1",
            input_run_ids=input_run_ids,
            input_artifact_ids=comparison.input_artifact_ids,
            operation_identity={
                "comparison_id": comparison.comparison_id,
                "result_digest": result_digest,
                "corpus_commit_id": result.corpus_commit_id,
            },
            supersede_matching=False,
        )
        return result.validated_copy(
            update={
                "analysis": provenance.analysis,
                "evidence": provenance.evidence,
                "materialized_commit_id": published.commit.commit_id,
            }
        )

    def _inputs(self, request: KernelValidationCompareRequest) -> _ComparisonInputs:
        baseline = self.run_sets.read(request.baseline_run_set_id)
        candidate = self.run_sets.read(request.candidate_run_set_id)
        if baseline.corpus_commit_id != candidate.corpus_commit_id:
            raise DomainError(
                ErrorCode.COMPARISON_INVALID,
                "Kernel-validation run sets must be frozen at the same corpus snapshot.",
            )
        return _ComparisonInputs(
            baseline=baseline,
            candidate=candidate,
            handle=Catalog(self.workspace).pin(baseline.corpus_commit_id),
        )

    @staticmethod
    def _comparison_rows(
        comparison: KernelValidationComparison,
    ) -> dict[str, list[dict[str, object]]]:
        summary: dict[str, object] = {
            "comparison_id": comparison.comparison_id,
            "baseline_run_set_id": comparison.baseline_run_set_id,
            "candidate_run_set_id": comparison.candidate_run_set_id,
            "experiment_id": comparison.experiment_id,
            "protocol_identity_id": comparison.protocol_identity_id,
            "metric": comparison.metric,
            "unit": comparison.unit,
            "metric_source": comparison.metric_source.value,
            "polarity": comparison.polarity.value if comparison.polarity else None,
            "estimand": comparison.estimand,
            "practical_threshold": comparison.practical_threshold,
            "estimate": comparison.estimate,
            "confidence_low": (
                comparison.confidence_interval.low
                if comparison.confidence_interval is not None
                else None
            ),
            "confidence_high": (
                comparison.confidence_interval.high
                if comparison.confidence_interval is not None
                else None
            ),
            "confidence_level": (
                comparison.confidence_interval.level
                if comparison.confidence_interval is not None
                else None
            ),
            "method": comparison.method,
            "independent_unit": comparison.independent_unit,
            "complete_pair_n": comparison.complete_pair_n,
            "decision": comparison.decision.value,
            "validity": comparison.validity.value,
            "pair_count": len(comparison.pairs),
            "aggregate_count": len(comparison.aggregates),
            "compatibility_mismatch_count": len(comparison.compatibility_mismatches),
            "baseline_only_count": len(comparison.baseline_only),
            "candidate_only_count": len(comparison.candidate_only),
            "mismatches": list(comparison.mismatches),
            "input_artifact_ids": list(comparison.input_artifact_ids),
        }
        pair_rows: list[dict[str, object]] = [
            {
                "comparison_id": comparison.comparison_id,
                "pair_id": pair.pair_id,
                "semantic_identity_id": pair.semantic_identity_id,
                "semantic_identity_json": _json(pair.semantic_identity),
                "independent_unit_id": pair.independent_unit_id,
                "baseline_run_id": pair.baseline_run_id,
                "candidate_run_id": pair.candidate_run_id,
                "baseline_artifact_id": pair.baseline_artifact_id,
                "candidate_artifact_id": pair.candidate_artifact_id,
                "case_id": pair.case_id,
                "output_name": pair.output_name,
                "metric_name": pair.metric_name,
                "unit": pair.unit,
                "comparator": pair.comparator,
                "threshold": pair.threshold,
                "polarity": pair.polarity.value,
                "baseline_value_json": pair.baseline_value.model_dump_json(),
                "candidate_value_json": pair.candidate_value.model_dump_json(),
                "baseline_status": pair.baseline_status,
                "candidate_status": pair.candidate_status,
                "status_transition": pair.status_transition,
                "change_kind": pair.change_kind.value,
                "signed_change": pair.signed_change,
                "absolute_change": pair.absolute_change,
                "relative_change_percent": pair.relative_change_percent,
                "relative_change_unavailable_reason": (pair.relative_change_unavailable_reason),
                "direction": pair.direction.value,
            }
            for pair in comparison.pairs
        ]
        aggregate_rows: list[dict[str, object]] = [
            {
                "comparison_id": comparison.comparison_id,
                "semantic_identity_id": aggregate.semantic_identity_id,
                "metric_name": aggregate.metric_name,
                "unit": aggregate.unit,
                "polarity": aggregate.polarity.value,
                "pair_count": aggregate.pair_count,
                "finite_pair_count": aggregate.finite_pair_count,
                "positive_infinity_pair_count": (aggregate.positive_infinity_pair_count),
                "median_signed_change": aggregate.median_signed_change,
                "direction": aggregate.direction.value,
            }
            for aggregate in comparison.aggregates
        ]
        gap_rows: list[dict[str, object]] = []
        for kind, values in (
            ("compatibility_mismatch", comparison.compatibility_mismatches),
            ("baseline_only", comparison.baseline_only),
            ("candidate_only", comparison.candidate_only),
        ):
            for value in values:
                dumped = value.model_dump(mode="json")
                gap_rows.append(
                    {
                        "gap_id": digest_model(
                            {
                                "comparison_id": comparison.comparison_id,
                                "gap_kind": kind,
                                "detail": dumped,
                            }
                        ),
                        "comparison_id": comparison.comparison_id,
                        "gap_kind": kind,
                        "independent_unit_id": value.independent_unit_id,
                        "case_id": value.case_id,
                        "output_name": value.output_name,
                        "metric_name": value.metric_name,
                        "detail_json": _json(dumped),
                    }
                )
        return {
            "kernel_validation_comparisons": [summary],
            "kernel_validation_comparison_pairs": pair_rows,
            "kernel_validation_comparison_aggregates": aggregate_rows,
            "kernel_validation_comparison_gaps": gap_rows,
        }

    def _compare_at_snapshot(
        self,
        request: KernelValidationCompareRequest,
        snapshot: Snapshot,
        inputs: _ComparisonInputs,
    ) -> KernelValidationComparisonResult:
        baseline_units, baseline_issues = self._independent_units(snapshot, inputs.baseline)
        candidate_units, candidate_issues = self._independent_units(snapshot, inputs.candidate)
        issues = [*baseline_issues, *candidate_issues]
        if set(baseline_units.values()) != set(candidate_units.values()):
            issues.append("independent-unit coverage differs across treatments")
        randomized = any(
            member.trial_id is not None
            for run_set in (inputs.baseline, inputs.candidate)
            for member in run_set.members
            if member.included
        )
        baseline_observations, baseline_observation_issues, baseline_artifacts = self._observations(
            snapshot, baseline_units
        )
        candidate_observations, candidate_observation_issues, candidate_artifacts = (
            self._observations(snapshot, candidate_units)
        )
        issues.extend(f"baseline: {value}" for value in baseline_observation_issues)
        issues.extend(f"candidate: {value}" for value in candidate_observation_issues)
        (
            pairs,
            compatibility_mismatches,
            baseline_only,
            candidate_only,
            pairing_issues,
        ) = self._pair_observations(baseline_observations, candidate_observations)
        issues.extend(pairing_issues)
        if not pairs:
            issues.append("no exact kernel-validation metric identities could be paired")
        incomplete = tuple(
            item
            for item in (*baseline_observations, *candidate_observations)
            if not item.coverage_complete
        )
        if incomplete:
            issues.append("one or more kernel-validation artifacts declare incomplete coverage")
        if compatibility_mismatches:
            issues.append("one or more metric locations have incompatible semantic identities")
        if baseline_only or candidate_only:
            issues.append("kernel-validation metric coverage differs across treatments")
        aggregates = self._aggregates(pairs)
        experiment = (
            self.experiments.read(request.experiment_id)
            if request.experiment_id is not None
            else None
        )
        assessment = self._assess_protocol(
            snapshot,
            request,
            experiment,
            pairs,
            baseline_units,
            candidate_units,
            inputs.baseline,
            inputs.candidate,
        )
        issues.extend(assessment[0])
        estimate, metric, unit, polarity, practical_threshold = assessment[1]
        invalidating = tuple(dict.fromkeys(issues))
        if request.protocol is None:
            validity = ComparisonValidity.INVALID if not pairs else ComparisonValidity.EXPLORATORY
            decision = (
                ComparisonDecision.INCONCLUSIVE
                if validity is ComparisonValidity.INVALID
                else ComparisonDecision.DESCRIPTIVE_ONLY
            )
        elif invalidating:
            validity = ComparisonValidity.INVALID
            decision = ComparisonDecision.INCONCLUSIVE
        elif estimate.confidence_low is None or estimate.confidence_high is None:
            validity = ComparisonValidity.EXPLORATORY
            decision = ComparisonDecision.INCONCLUSIVE
            if estimate.limitation is not None:
                invalidating = (*invalidating, estimate.limitation)
        else:
            validity = ComparisonValidity.VALID
            assert polarity is not None
            assert practical_threshold is not None
            decision = self._decision(
                low=estimate.confidence_low,
                high=estimate.confidence_high,
                threshold=practical_threshold,
                polarity=polarity,
            )
        artifact_ids = tuple(dict.fromkeys((*baseline_artifacts, *candidate_artifacts)))
        comparison_id = digest_model(
            {
                "recipe": "compare_kernel_validation.v1",
                "request": request.model_dump(mode="json"),
                "baseline_membership": inputs.baseline.membership_digest,
                "candidate_membership": inputs.candidate.membership_digest,
                "corpus_commit_id": snapshot.handle.commit_id,
            }
        )
        comparison = KernelValidationComparison(
            comparison_id=comparison_id,
            baseline_run_set_id=inputs.baseline.run_set_id,
            candidate_run_set_id=inputs.candidate.run_set_id,
            experiment_id=request.experiment_id,
            protocol_identity_id=(
                request.protocol.protocol_id if request.protocol is not None else None
            ),
            metric=metric,
            unit=unit,
            polarity=polarity,
            estimand=(
                "median_blockwise_median_difference" if request.protocol is not None else None
            ),
            practical_threshold=practical_threshold,
            estimate=estimate.estimate,
            confidence_interval=(
                ConfidenceInterval(
                    low=estimate.confidence_low,
                    high=estimate.confidence_high,
                    level=experiment.confidence_level,
                )
                if (
                    estimate.confidence_low is not None
                    and estimate.confidence_high is not None
                    and experiment is not None
                )
                else None
            ),
            method=estimate.method,
            independent_unit="randomized_block" if randomized else "single_run",
            complete_pair_n=len({pair.independent_unit_id for pair in pairs}),
            decision=decision,
            validity=validity,
            pairs=pairs,
            aggregates=aggregates,
            compatibility_mismatches=compatibility_mismatches,
            baseline_only=baseline_only,
            candidate_only=candidate_only,
            mismatches=invalidating,
            input_artifact_ids=artifact_ids,
        )
        return KernelValidationComparisonResult(
            comparison=comparison,
            baseline_run_set=inputs.baseline,
            candidate_run_set=inputs.candidate,
            corpus_commit_id=snapshot.handle.commit_id,
        )

    @staticmethod
    def _independent_units(
        snapshot: Snapshot,
        run_set: RunSet,
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        included = tuple(member for member in run_set.members if member.included)
        issues: list[str] = []
        excluded = tuple(member for member in run_set.members if not member.included)
        if excluded:
            issues.append(f"run set {run_set.run_set_id} excludes {len(excluded)} declared members")
        if not included:
            return {}, (*issues, f"run set {run_set.run_set_id} has no included members")
        if len(included) == 1 and included[0].trial_id is None:
            return {included[0].run_id: "single"}, tuple(issues)
        experiment_id = run_set.selection.get("experiment_id")
        if not isinstance(experiment_id, str):
            return {}, (
                *issues,
                f"run set {run_set.run_set_id} lacks an exact experiment identity",
            )
        result: dict[str, str] = {}
        for member in included:
            if member.trial_id is None:
                issues.append(f"run {member.run_id} lacks a trial identity")
                continue
            rows = snapshot.execute(
                "SELECT DISTINCT block_id, outcome FROM trials WHERE experiment_id = ? "
                "AND trial_id = ? AND run_id = ?",
                (experiment_id, member.trial_id, member.run_id),
            ).fetchall()
            if len(rows) != 1 or rows[0][0] is None:
                issues.append(f"trial {member.trial_id} does not resolve to one randomized block")
                continue
            if str(rows[0][1]) != "succeeded":
                issues.append(f"trial {member.trial_id} did not succeed")
                continue
            block_id = str(rows[0][0])
            if block_id in result.values():
                issues.append(f"run set {run_set.run_set_id} repeats block {block_id}")
                continue
            result[member.run_id] = block_id
        return result, tuple(issues)

    def _observations(
        self,
        snapshot: Snapshot,
        units: Mapping[str, str],
    ) -> tuple[tuple[_Observation, ...], tuple[str, ...], tuple[str, ...]]:
        if not units:
            return (), (), ()
        run_ids = tuple(units)
        placeholders = ", ".join("?" for _ in run_ids)
        rows = snapshot.execute(
            "WITH latest_metrics AS (SELECT * FROM kernel_validation_metrics "
            "WHERE run_id IN ("
            + placeholders
            + ") QUALIFY row_number() OVER (PARTITION BY metric_id "
            "ORDER BY published_at DESC) = 1), latest_cases AS (SELECT * FROM "
            "kernel_validation_cases WHERE run_id IN ("
            + placeholders
            + ") QUALIFY row_number() OVER (PARTITION BY case_output_id "
            "ORDER BY published_at DESC) = 1), latest_runs AS (SELECT run_id, environment_id "
            "FROM current_runs WHERE run_id IN ("
            + placeholders
            + ")) SELECT m.run_id, m.artifact_id, "
            "m.metric_name, m.value, m.value_kind, m.positive_infinity_reason, "
            "m.comparator, m.threshold, m.unit, m.metric_profile_json, "
            "m.metric_identity_id, m.status, m.limitation, c.case_id, c.output_name, "
            "c.coverage_complete, c.reference_name, c.reference_version, "
            "c.reference_identity, c.device, c.seed, c.dimensions_json, c.inputs_json, "
            "c.dtype, c.shape, c.case_status, c.output_status, r.environment_id "
            "FROM latest_metrics m JOIN latest_cases c USING (case_output_id, run_id, "
            "artifact_id) JOIN latest_runs r USING (run_id) ORDER BY m.run_id, "
            "c.case_id, c.output_name, m.metric_name",
            (*run_ids, *run_ids, *run_ids),
        ).fetchall()
        observations = tuple(self._observation(row, units[str(row[0])]) for row in rows)
        case_rows = snapshot.execute(
            "SELECT run_id, artifact_id, case_id, output_name, coverage_complete "
            "FROM kernel_validation_cases WHERE run_id IN ("
            + placeholders
            + ") QUALIFY row_number() OVER (PARTITION BY case_output_id "
            "ORDER BY published_at DESC) = 1 ORDER BY run_id, case_id, output_name",
            run_ids,
        ).fetchall()
        issues: list[str] = []
        case_runs = {str(row[0]) for row in case_rows}
        for run_id in sorted(set(run_ids) - case_runs):
            issues.append(f"run {run_id} has no kernel-validation case evidence")
        observed_outputs = {
            (item.run_id, item.artifact_id, item.case_id, item.output_name) for item in observations
        }
        for run_id, artifact_id, case_id, output_name, coverage_complete in case_rows:
            identity = (str(run_id), str(artifact_id), str(case_id), str(output_name))
            if identity not in observed_outputs:
                issues.append(
                    f"run {run_id} output {case_id}/{output_name} has no metric observations"
                )
            if not bool(coverage_complete):
                issues.append(f"run {run_id} declares incomplete validation coverage")
        artifact_ids = tuple(dict.fromkeys(str(row[1]) for row in case_rows))
        return observations, tuple(issues), artifact_ids

    @staticmethod
    def _observation(row: tuple[object, ...], independent_unit_id: str) -> _Observation:
        try:
            dimensions = cast(dict[str, JsonValue], json.loads(str(row[21])))
            inputs = cast(dict[str, JsonValue], json.loads(str(row[22])))
            profile = (
                cast(dict[str, JsonValue], json.loads(str(row[9]))) if row[9] is not None else None
            )
        except (TypeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                "Kernel-validation identity JSON is malformed.",
            ) from exc
        status = str(row[11])
        value_kind = str(row[4]) if row[4] is not None else None
        if value_kind == "finite" and row[3] is not None:
            value: KernelObservedValue = KernelFiniteObservedValue(
                value=float(cast(str | int | float, row[3]))
            )
        elif value_kind == "positive_infinity" and row[5] is not None:
            value = KernelPositiveInfinityObservedValue(reason=cast(Any, str(row[5])))
        elif value_kind is None and status in {"inconclusive", "unsupported"} and row[12]:
            value = KernelUnavailableObservedValue(
                status=cast(Any, status),
                limitation=str(row[12]),
            )
        else:
            raise DomainError(
                ErrorCode.EVIDENCE_SCHEMA_MISMATCH,
                "Kernel-validation metric has an incoherent tagged value.",
            )
        semantic_identity: dict[str, JsonValue] = {
            "reference": {
                "name": str(row[16]),
                "version": str(row[17]) if row[17] is not None else None,
                "identity": str(row[18]),
            },
            "case": {
                "case_id": str(row[13]),
                "dimensions": dimensions,
                "inputs": inputs,
                "seed": int(cast(int, row[20])) if row[20] is not None else None,
            },
            "output": {
                "name": str(row[14]),
                "dtype": str(row[23]),
                "shape": [int(item) for item in cast(list[int], row[24])],
            },
            "metric": {
                "name": str(row[2]),
                "unit": str(row[8]),
                "comparator": str(row[6]) if row[6] is not None else None,
                "threshold": (
                    float(cast(str | int | float, row[7])) if row[7] is not None else None
                ),
                "profile": profile,
            },
            "device": str(row[19]),
            "environment_id": str(row[27]),
        }
        semantic_identity_id = digest_model(semantic_identity)
        return _Observation(
            independent_unit_id=independent_unit_id,
            run_id=str(row[0]),
            artifact_id=str(row[1]),
            environment_id=str(row[27]),
            semantic_identity=semantic_identity,
            semantic_identity_id=semantic_identity_id,
            location_key=(
                independent_unit_id,
                str(row[13]),
                str(row[14]),
                str(row[2]),
            ),
            case_id=str(row[13]),
            output_name=str(row[14]),
            metric_name=str(row[2]),
            unit=str(row[8]),
            comparator=str(row[6]) if row[6] is not None else None,
            threshold=(float(cast(str | int | float, row[7])) if row[7] is not None else None),
            profile=profile,
            value=value,
            status=status,
            coverage_complete=bool(row[15]),
        )

    def _pair_observations(
        self,
        baseline: tuple[_Observation, ...],
        candidate: tuple[_Observation, ...],
    ) -> tuple[
        tuple[KernelValidationMetricPair, ...],
        tuple[KernelValidationCompatibilityMismatch, ...],
        tuple[KernelValidationUnmatchedMetric, ...],
        tuple[KernelValidationUnmatchedMetric, ...],
        tuple[str, ...],
    ]:
        baseline_exact = self._unique_observations(baseline)
        candidate_exact = self._unique_observations(candidate)
        issues = [*baseline_exact[1], *candidate_exact[1]]
        baseline_by_key = baseline_exact[0]
        candidate_by_key = candidate_exact[0]
        exact_keys = set(baseline_by_key) & set(candidate_by_key)
        pairs = tuple(
            self._metric_pair(baseline_by_key[key], candidate_by_key[key])
            for key in sorted(exact_keys)
        )
        remaining_baseline = [
            value for key, value in baseline_by_key.items() if key not in exact_keys
        ]
        remaining_candidate = [
            value for key, value in candidate_by_key.items() if key not in exact_keys
        ]
        baseline_locations: dict[tuple[str, str, str, str], list[_Observation]] = defaultdict(list)
        candidate_locations: dict[tuple[str, str, str, str], list[_Observation]] = defaultdict(list)
        for item in remaining_baseline:
            baseline_locations[item.location_key].append(item)
        for item in remaining_candidate:
            candidate_locations[item.location_key].append(item)
        compatibility: list[KernelValidationCompatibilityMismatch] = []
        consumed_baseline: set[tuple[str, str]] = set()
        consumed_candidate: set[tuple[str, str]] = set()
        for location in sorted(set(baseline_locations) & set(candidate_locations)):
            left = baseline_locations[location]
            right = candidate_locations[location]
            if len(left) != 1 or len(right) != 1:
                issues.append(
                    "a kernel-validation metric location resolves to multiple semantic identities"
                )
                continue
            baseline_item, candidate_item = left[0], right[0]
            reasons = self._identity_mismatches(baseline_item, candidate_item)
            compatibility.append(
                KernelValidationCompatibilityMismatch(
                    independent_unit_id=location[0],
                    case_id=location[1],
                    output_name=location[2],
                    metric_name=location[3],
                    baseline_semantic_identity_id=baseline_item.semantic_identity_id,
                    candidate_semantic_identity_id=candidate_item.semantic_identity_id,
                    reasons=reasons,
                )
            )
            consumed_baseline.add((baseline_item.run_id, baseline_item.semantic_identity_id))
            consumed_candidate.add((candidate_item.run_id, candidate_item.semantic_identity_id))
        baseline_only = tuple(
            self._unmatched(item)
            for item in remaining_baseline
            if (item.run_id, item.semantic_identity_id) not in consumed_baseline
        )
        candidate_only = tuple(
            self._unmatched(item)
            for item in remaining_candidate
            if (item.run_id, item.semantic_identity_id) not in consumed_candidate
        )
        return (
            pairs,
            tuple(compatibility),
            baseline_only,
            candidate_only,
            tuple(dict.fromkeys(issues)),
        )

    @staticmethod
    def _unique_observations(
        values: tuple[_Observation, ...],
    ) -> tuple[dict[tuple[str, str], _Observation], tuple[str, ...]]:
        result: dict[tuple[str, str], _Observation] = {}
        duplicates: list[str] = []
        for value in values:
            key = (value.independent_unit_id, value.semantic_identity_id)
            if key in result:
                duplicates.append(
                    f"duplicate metric identity {value.semantic_identity_id} in independent "
                    f"unit {value.independent_unit_id}"
                )
            else:
                result[key] = value
        return result, tuple(duplicates)

    @staticmethod
    def _identity_mismatches(
        baseline: _Observation,
        candidate: _Observation,
    ) -> tuple[str, ...]:
        fields = (
            ("reference",),
            ("case", "dimensions"),
            ("case", "inputs"),
            ("case", "seed"),
            ("output", "dtype"),
            ("output", "shape"),
            ("metric", "unit"),
            ("metric", "comparator"),
            ("metric", "threshold"),
            ("metric", "profile"),
            ("device",),
            ("environment_id",),
        )
        reasons: list[str] = []
        for path in fields:
            left: object = baseline.semantic_identity
            right: object = candidate.semantic_identity
            for part in path:
                left = cast(Mapping[str, object], left)[part]
                right = cast(Mapping[str, object], right)[part]
            if left != right:
                reasons.append(".".join(path) + " differs")
        return tuple(reasons or ("semantic identity differs",))

    @staticmethod
    def _unmatched(value: _Observation) -> KernelValidationUnmatchedMetric:
        return KernelValidationUnmatchedMetric(
            independent_unit_id=value.independent_unit_id,
            run_id=value.run_id,
            artifact_id=value.artifact_id,
            semantic_identity_id=value.semantic_identity_id,
            case_id=value.case_id,
            output_name=value.output_name,
            metric_name=value.metric_name,
            status=value.status,
        )

    @classmethod
    def _metric_pair(
        cls,
        baseline: _Observation,
        candidate: _Observation,
    ) -> KernelValidationMetricPair:
        polarity = cls._polarity(baseline.metric_name)
        (
            change_kind,
            signed_change,
            absolute_change,
            relative_change,
            relative_unavailable,
            direction,
        ) = cls._change(baseline.value, candidate.value, polarity)
        return KernelValidationMetricPair(
            pair_id=digest_model(
                {
                    "independent_unit_id": baseline.independent_unit_id,
                    "semantic_identity_id": baseline.semantic_identity_id,
                    "baseline_run_id": baseline.run_id,
                    "candidate_run_id": candidate.run_id,
                }
            ),
            semantic_identity_id=baseline.semantic_identity_id,
            semantic_identity=baseline.semantic_identity,
            independent_unit_id=baseline.independent_unit_id,
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            baseline_artifact_id=baseline.artifact_id,
            candidate_artifact_id=candidate.artifact_id,
            case_id=baseline.case_id,
            output_name=baseline.output_name,
            metric_name=baseline.metric_name,
            unit=baseline.unit,
            comparator=baseline.comparator or "unavailable",
            threshold=baseline.threshold,
            polarity=polarity,
            baseline_value=baseline.value,
            candidate_value=candidate.value,
            baseline_status=baseline.status,
            candidate_status=candidate.status,
            status_transition=f"{baseline.status}_to_{candidate.status}",
            change_kind=change_kind,
            signed_change=signed_change,
            absolute_change=absolute_change,
            relative_change_percent=relative_change,
            relative_change_unavailable_reason=relative_unavailable,
            direction=direction,
        )

    @staticmethod
    def _polarity(metric_name: str) -> MetricPolarity:
        if metric_name in _LOWER_IS_BETTER:
            return MetricPolarity.LOWER_IS_BETTER
        if metric_name in _HIGHER_IS_BETTER:
            return MetricPolarity.HIGHER_IS_BETTER
        return MetricPolarity.NEUTRAL

    @staticmethod
    def _change(
        baseline: KernelObservedValue,
        candidate: KernelObservedValue,
        polarity: MetricPolarity,
    ) -> tuple[
        KernelMetricChangeKind,
        float | None,
        float | None,
        float | None,
        str | None,
        KernelMetricDirection,
    ]:
        if isinstance(baseline, KernelUnavailableObservedValue) or isinstance(
            candidate, KernelUnavailableObservedValue
        ):
            return (
                KernelMetricChangeKind.UNDEFINED,
                None,
                None,
                None,
                "one or both metric values are unavailable",
                KernelMetricDirection.UNASSESSABLE,
            )
        if isinstance(baseline, KernelPositiveInfinityObservedValue):
            if isinstance(candidate, KernelPositiveInfinityObservedValue):
                return (
                    KernelMetricChangeKind.UNDEFINED,
                    None,
                    None,
                    None,
                    "both values are exact positive infinity",
                    KernelMetricDirection.UNCHANGED,
                )
            return (
                KernelMetricChangeKind.NEGATIVE_INFINITY,
                None,
                None,
                None,
                "baseline is exact positive infinity",
                KernelMetricDirection.REGRESSED,
            )
        if isinstance(candidate, KernelPositiveInfinityObservedValue):
            return (
                KernelMetricChangeKind.POSITIVE_INFINITY,
                None,
                None,
                None,
                "candidate is exact positive infinity",
                KernelMetricDirection.IMPROVED,
            )
        signed = candidate.value - baseline.value
        absolute = abs(signed)
        relative = 100.0 * signed / baseline.value if baseline.value != 0 else None
        if signed == 0 or polarity is MetricPolarity.NEUTRAL:
            direction = KernelMetricDirection.UNCHANGED
        elif (polarity is MetricPolarity.LOWER_IS_BETTER and signed < 0) or (
            polarity is MetricPolarity.HIGHER_IS_BETTER and signed > 0
        ):
            direction = KernelMetricDirection.IMPROVED
        else:
            direction = KernelMetricDirection.REGRESSED
        return (
            KernelMetricChangeKind.FINITE,
            signed,
            absolute,
            relative,
            "baseline is zero" if relative is None else None,
            direction,
        )

    @staticmethod
    def _aggregates(
        pairs: tuple[KernelValidationMetricPair, ...],
    ) -> tuple[KernelValidationMetricAggregate, ...]:
        grouped: dict[str, list[KernelValidationMetricPair]] = defaultdict(list)
        for pair in pairs:
            grouped[pair.semantic_identity_id].append(pair)
        result: list[KernelValidationMetricAggregate] = []
        for identity_id, values in sorted(grouped.items()):
            finite_changes = [
                item.signed_change for item in values if item.signed_change is not None
            ]
            directions = {item.direction for item in values}
            if (
                directions
                <= {
                    KernelMetricDirection.IMPROVED,
                    KernelMetricDirection.UNCHANGED,
                }
                and KernelMetricDirection.IMPROVED in directions
            ):
                direction = KernelMetricDirection.IMPROVED
            elif (
                directions
                <= {
                    KernelMetricDirection.REGRESSED,
                    KernelMetricDirection.UNCHANGED,
                }
                and KernelMetricDirection.REGRESSED in directions
            ):
                direction = KernelMetricDirection.REGRESSED
            elif directions == {KernelMetricDirection.UNCHANGED}:
                direction = KernelMetricDirection.UNCHANGED
            else:
                direction = KernelMetricDirection.UNASSESSABLE
            first = values[0]
            result.append(
                KernelValidationMetricAggregate(
                    semantic_identity_id=identity_id,
                    metric_name=first.metric_name,
                    unit=first.unit,
                    polarity=first.polarity,
                    pair_count=len(values),
                    finite_pair_count=len(finite_changes),
                    positive_infinity_pair_count=sum(
                        isinstance(item.baseline_value, KernelPositiveInfinityObservedValue)
                        or isinstance(
                            item.candidate_value,
                            KernelPositiveInfinityObservedValue,
                        )
                        for item in values
                    ),
                    median_signed_change=(
                        statistics.median(finite_changes) if finite_changes else None
                    ),
                    direction=direction,
                )
            )
        return tuple(result)

    def _assess_protocol(
        self,
        snapshot: Snapshot,
        request: KernelValidationCompareRequest,
        experiment: Experiment | None,
        pairs: tuple[KernelValidationMetricPair, ...],
        baseline_units: Mapping[str, str],
        candidate_units: Mapping[str, str],
        baseline_run_set: RunSet,
        candidate_run_set: RunSet,
    ) -> tuple[
        tuple[str, ...],
        tuple[
            PairedDifferenceEstimate,
            str | None,
            str | None,
            MetricPolarity | None,
            float | None,
        ],
    ]:
        if request.protocol is None:
            return (
                (),
                (
                    PairedDifferenceEstimate(
                        estimate=None,
                        confidence_low=None,
                        confidence_high=None,
                        method="descriptive.exact_metric_pairs.v1",
                    ),
                    None,
                    None,
                    None,
                    None,
                ),
            )
        assert experiment is not None
        protocol = request.protocol
        first_selector = protocol.case_population[0]
        polarity = self._polarity(first_selector.metric_name)
        issues = self._protocol_experiment_issues(
            snapshot,
            experiment,
            protocol,
            polarity,
            baseline_units,
            candidate_units,
            baseline_run_set,
            candidate_run_set,
        )
        block_baseline, block_candidate, population_issues = self._protocol_block_samples(
            pairs,
            tuple(sorted(set(baseline_units.values()) | set(candidate_units.values()))),
            protocol,
        )
        issues.extend(population_issues)
        estimate = paired_median_difference(
            block_baseline,
            block_candidate,
            confidence_level=experiment.confidence_level,
            random_seed=experiment.random_seed,
        )
        if estimate.estimate is None and estimate.limitation is not None:
            issues.append(estimate.limitation)
        return (
            tuple(dict.fromkeys(issues)),
            (
                estimate,
                first_selector.metric_name,
                first_selector.unit,
                polarity,
                experiment.practical_threshold,
            ),
        )

    def _protocol_experiment_issues(
        self,
        snapshot: Snapshot,
        experiment: Experiment,
        protocol: KernelValidationComparisonProtocol,
        polarity: MetricPolarity,
        baseline_units: Mapping[str, str],
        candidate_units: Mapping[str, str],
        baseline_run_set: RunSet,
        candidate_run_set: RunSet,
    ) -> list[str]:
        first_selector = protocol.case_population[0]
        issues: list[str] = []
        if experiment.role is not ExperimentRole.CONFIRMATORY:
            issues.append("experiment is not registered as confirmatory")
        if experiment.validation_spec_id != protocol.protocol_id:
            issues.append("protocol does not match the experiment validation_spec_id")
        if experiment.metric_source is not MetricSource.KERNEL_VALIDATION:
            issues.append("experiment metric source is not kernel_validation")
        if experiment.primary_metric != first_selector.metric_name:
            issues.append("primary metric differs from the experiment protocol")
        if experiment.primary_metric_unit != first_selector.unit:
            issues.append("primary metric unit differs from the experiment protocol")
        if experiment.polarity is not polarity:
            issues.append("primary metric polarity differs from the experiment protocol")
        if experiment.estimand != "median_blockwise_median_difference":
            issues.append("experiment does not declare the kernel-validation estimand")
        if set(baseline_units.values()) == {"single"} or set(candidate_units.values()) == {
            "single"
        }:
            issues.append("confirmatory comparison requires randomized-block trial identities")
        if baseline_run_set.selection.get("experiment_id") != experiment.experiment_id or (
            candidate_run_set.selection.get("experiment_id") != experiment.experiment_id
        ):
            issues.append("run sets are not bound to the declared experiment")
        baseline_variant = baseline_run_set.selection.get("variant_id")
        candidate_variant = candidate_run_set.selection.get("variant_id")
        if not isinstance(baseline_variant, str) or not isinstance(candidate_variant, str):
            issues.append("run sets lack exact experiment variant identities")
        elif baseline_variant == candidate_variant:
            issues.append("baseline and candidate select the same experiment variant")
        fixed_blocks = experiment.stopping_rule.get("fixed_blocks")
        if not isinstance(fixed_blocks, int) or fixed_blocks < 1:
            issues.append("experiment has no fixed-block stopping rule")
        elif (
            len(set(baseline_units.values())) != fixed_blocks
            or len(set(candidate_units.values())) != fixed_blocks
        ):
            issues.append("run sets do not satisfy the experiment fixed-block stopping rule")
        issues.extend(self._experiment_population_issues(snapshot, experiment, baseline_run_set))
        issues.extend(self._experiment_population_issues(snapshot, experiment, candidate_run_set))
        return issues

    def _protocol_block_samples(
        self,
        pairs: tuple[KernelValidationMetricPair, ...],
        units: tuple[str, ...],
        protocol: KernelValidationComparisonProtocol,
    ) -> tuple[list[float], list[float], list[str]]:
        issues: list[str] = []
        block_baseline: list[float] = []
        block_candidate: list[float] = []
        for independent_unit_id in units:
            signed_changes: list[float] = []
            for selector in protocol.case_population:
                selected = [
                    pair
                    for pair in pairs
                    if pair.independent_unit_id == independent_unit_id
                    and self._selector_matches(pair.semantic_identity, protocol, selector)
                ]
                if len(selected) != 1:
                    issues.append(
                        f"block {independent_unit_id} does not contain exactly one "
                        f"preregistered {selector.case_id}/{selector.output_name}/"
                        f"{selector.metric_name} pair"
                    )
                    continue
                pair = selected[0]
                if pair.signed_change is None:
                    issues.append(f"block {independent_unit_id} preregistered metric is not finite")
                    continue
                signed_changes.append(pair.signed_change)
            if len(signed_changes) == len(protocol.case_population):
                block_baseline.append(0.0)
                block_candidate.append(float(statistics.median(signed_changes)))
        return block_baseline, block_candidate, issues

    @staticmethod
    def _selector_matches(
        semantic_identity: Mapping[str, JsonValue],
        protocol: KernelValidationComparisonProtocol,
        selector: KernelValidationMetricSelector,
    ) -> bool:
        expected: dict[str, JsonValue] = {
            "reference": {
                "name": protocol.reference_name,
                "version": protocol.reference_version,
                "identity": protocol.reference_identity,
            },
            "case": {
                "case_id": selector.case_id,
                "dimensions": cast(JsonValue, selector.dimensions),
                "inputs": {
                    name: value.model_dump(mode="json") for name, value in selector.inputs.items()
                },
                "seed": selector.seed,
            },
            "output": {
                "name": selector.output_name,
                "dtype": selector.output_dtype,
                "shape": list(selector.output_shape),
            },
            "metric": {
                "name": selector.metric_name,
                "unit": selector.unit,
                "comparator": selector.comparator,
                "threshold": selector.threshold,
                "profile": (
                    selector.psnr_profile.model_dump(mode="json")
                    if selector.psnr_profile is not None
                    else None
                ),
            },
            "device": selector.device,
        }
        return all(semantic_identity.get(key) == value for key, value in expected.items())

    @staticmethod
    def _experiment_population_issues(
        snapshot: Snapshot,
        experiment: Experiment,
        run_set: RunSet,
    ) -> tuple[str, ...]:
        issues: list[str] = []
        selected: set[tuple[str, str]] = set()
        variants: set[str] = set()
        for member in run_set.members:
            if member.trial_id is None:
                continue
            rows = snapshot.execute(
                "SELECT DISTINCT variant_id FROM trials WHERE experiment_id = ? "
                "AND trial_id = ? AND run_id = ?",
                (experiment.experiment_id, member.trial_id, member.run_id),
            ).fetchall()
            if len(rows) == 1:
                variants.add(str(rows[0][0]))
                selected.add((member.trial_id, member.run_id))
        if len(variants) != 1:
            return ("run set does not select exactly one experiment variant",)
        variant_id = next(iter(variants))
        rows = snapshot.execute(
            "SELECT DISTINCT trial_id, run_id FROM trials WHERE experiment_id = ? "
            "AND variant_id = ?",
            (experiment.experiment_id, variant_id),
        ).fetchall()
        declared = {(str(trial_id), str(run_id)) for trial_id, run_id in rows if run_id is not None}
        if len(declared) != len(rows) or selected != declared:
            issues.append("run set does not contain the full declared trial population")
        return tuple(issues)

    @staticmethod
    def _decision(
        *,
        low: float,
        high: float,
        threshold: float,
        polarity: MetricPolarity,
    ) -> ComparisonDecision:
        if polarity is MetricPolarity.LOWER_IS_BETTER:
            if high <= -threshold:
                return ComparisonDecision.MEANINGFUL_IMPROVEMENT
            if low >= threshold:
                return ComparisonDecision.MEANINGFUL_REGRESSION
        elif polarity is MetricPolarity.HIGHER_IS_BETTER:
            if low >= threshold:
                return ComparisonDecision.MEANINGFUL_IMPROVEMENT
            if high <= -threshold:
                return ComparisonDecision.MEANINGFUL_REGRESSION
        if low >= -threshold and high <= threshold:
            return ComparisonDecision.NO_MEANINGFUL_DIFFERENCE
        return ComparisonDecision.INCONCLUSIVE


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
