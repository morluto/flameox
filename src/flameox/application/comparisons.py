from __future__ import annotations

import json
import math
import statistics
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, cast

from pydantic import (
    Discriminator,
    Field,
    JsonValue,
    Tag,
    TypeAdapter,
    field_validator,
    model_validator,
)

from flameox.analysis import compare_paired_samples, compare_unpaired_samples
from flameox.analysis.inference_protocol import (
    InferenceProtocolIdentity,
    compare_inference_protocols,
)
from flameox.application.analysis_provenance import (
    AnalysisProvenanceInput,
    build_analysis_provenance,
    context_references,
)
from flameox.application.async_work import run_atomic_thread
from flameox.application.progress import ProgressReporter
from flameox.application.runtime_resources import (
    RuntimeResourceMetric,
    runtime_resource_metric_definition,
)
from flameox.catalog import Catalog, Snapshot, SnapshotHandle
from flameox.domain import (
    AnalysisRecord,
    Comparison,
    ComparisonDecision,
    ComparisonValidity,
    DomainError,
    ErrorCode,
    EvidenceReference,
    Experiment,
    IdentityQuality,
    MeasurementSeriesSelector,
    MetricPolarity,
    MetricSource,
    RunSet,
    RunSetMember,
    ValidationStatus,
    digest_model,
    validate_run_set_selection,
)
from flameox.domain.models import (
    MAX_RUN_SET_MEMBERS,
    ExcludedRunSetMember,
    IncludedRunSetMember,
)
from flameox.evidence import GenerationPublisher, numeric_value_to_columns
from flameox.models import ContractModel
from flameox.storage import (
    CompletedRetentionIntent,
    ControlRecordStore,
    RetentionIntentStore,
    StorageQuota,
    Workspace,
)


class ProfileChangeDirection(StrEnum):
    REGRESSED = "regressed"
    IMPROVED = "improved"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


class _FreezeRunSetMember(ContractModel):
    run_id: str
    trial_id: str | None = None


class IncludedFreezeRunSetMember(_FreezeRunSetMember):
    included: Literal[True] = True
    reason: Literal[None] = None


class ExcludedFreezeRunSetMember(_FreezeRunSetMember):
    included: Literal[False] = False
    reason: str


def _freeze_member_variant(value: Any) -> Literal["included", "excluded"]:
    if isinstance(value, ExcludedFreezeRunSetMember):
        return "excluded"
    if isinstance(value, Mapping) and value.get("included") is False:
        return "excluded"
    return "included"


type FreezeRunSetMember = Annotated[
    Annotated[IncludedFreezeRunSetMember, Tag("included")]
    | Annotated[ExcludedFreezeRunSetMember, Tag("excluded")],
    Discriminator(_freeze_member_variant),
]


class _FreezeRunSetRequest(ContractModel):
    selection: dict[str, JsonValue] = Field(default_factory=dict, max_length=16)
    corpus_commit_id: str | None = None

    @field_validator("selection")
    @classmethod
    def selection_is_bounded(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return validate_run_set_selection(value)


class FreezeRunIdsRequest(_FreezeRunSetRequest):
    run_ids: Annotated[tuple[str, ...], Field(min_length=1, max_length=MAX_RUN_SET_MEMBERS)]
    members: Annotated[tuple[FreezeRunSetMember, ...], Field(max_length=0)] = ()

    @model_validator(mode="after")
    def run_ids_are_unique(self) -> FreezeRunIdsRequest:
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("run-set run IDs must be unique")
        return self


class FreezeRunMembersRequest(_FreezeRunSetRequest):
    run_ids: Annotated[tuple[str, ...], Field(max_length=0)] = ()
    members: Annotated[
        tuple[FreezeRunSetMember, ...],
        Field(min_length=1, max_length=MAX_RUN_SET_MEMBERS),
    ]

    @model_validator(mode="after")
    def member_identities_are_unique(self) -> FreezeRunMembersRequest:
        identities = [(member.run_id, member.trial_id) for member in self.members]
        if len(set(identities)) != len(identities):
            raise ValueError("run-set member identities must be unique")
        run_ids = [member.run_id for member in self.members]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("a run can appear in a run set only once")
        return self


def _freeze_request_variant(value: Any) -> Literal["run_ids", "members"]:
    if isinstance(value, FreezeRunMembersRequest):
        return "members"
    if isinstance(value, Mapping) and value.get("members"):
        return "members"
    return "run_ids"


type FreezeRunSetRequest = Annotated[
    Annotated[FreezeRunIdsRequest, Tag("run_ids")]
    | Annotated[FreezeRunMembersRequest, Tag("members")],
    Discriminator(_freeze_request_variant),
]


class _CompareRunSetsRequest(ContractModel):
    baseline_run_set_id: str
    candidate_run_set_id: str
    experiment_id: str | None = None
    polarity: MetricPolarity
    practical_threshold: float = Field(ge=0, allow_inf_nan=False)
    confidence_level: float = Field(default=0.95, gt=0, lt=1, allow_inf_nan=False)
    random_seed: int = Field(default=0, ge=0)
    estimand: Literal["median_paired_log_ratio", "difference_in_median_logs"] | None = None

    @model_validator(mode="after")
    def experiment_comparisons_declare_the_estimand(self) -> _CompareRunSetsRequest:
        if self.experiment_id is not None and self.estimand is None:
            raise ValueError("experiment-linked comparisons require an explicit estimand")
        return self


class MeasurementCompareRunSetsRequest(_CompareRunSetsRequest):
    metric: str
    unit: str
    series: MeasurementSeriesSelector | None = None
    metric_source: Literal[MetricSource.MEASUREMENT] = MetricSource.MEASUREMENT


class RuntimeResourceCompareRunSetsRequest(_CompareRunSetsRequest):
    metric: RuntimeResourceMetric
    unit: Literal["bytes"]
    metric_source: Literal[MetricSource.RUNTIME_RESOURCE] = MetricSource.RUNTIME_RESOURCE


def _comparison_metric_source(
    value: Any,
) -> Literal[MetricSource.MEASUREMENT, MetricSource.RUNTIME_RESOURCE]:
    if isinstance(value, RuntimeResourceCompareRunSetsRequest):
        return MetricSource.RUNTIME_RESOURCE
    if isinstance(value, Mapping) and value.get("metric_source") == MetricSource.RUNTIME_RESOURCE:
        return MetricSource.RUNTIME_RESOURCE
    return MetricSource.MEASUREMENT


type CompareRunSetsRequest = Annotated[
    Annotated[MeasurementCompareRunSetsRequest, Tag("measurement")]
    | Annotated[RuntimeResourceCompareRunSetsRequest, Tag("runtime_resource")],
    Discriminator(_comparison_metric_source),
]

_COMPARE_RUN_SETS_REQUEST: TypeAdapter[CompareRunSetsRequest] = TypeAdapter(CompareRunSetsRequest)


def parse_compare_run_sets_request(value: Any) -> CompareRunSetsRequest:
    return _COMPARE_RUN_SETS_REQUEST.validate_python(value)


class ComparisonResult(ContractModel):
    schema_version: Literal[2] = 2
    comparison: Comparison
    baseline_run_set: RunSet
    candidate_run_set: RunSet
    corpus_commit_id: str
    profile_changes: tuple[ProfileChange, ...] = ()
    analysis: AnalysisRecord | None = None
    evidence: tuple[EvidenceReference, ...] = ()
    materialized_commit_id: str | None = None


class ProfileChange(ContractModel):
    frame_id: str
    function: str | None
    file: str | None
    line: int | None
    metric: str
    unit: str
    baseline_value: float
    candidate_value: float
    absolute_change: float
    relative_change: float | None
    direction: ProfileChangeDirection


@dataclass(frozen=True, slots=True)
class _SampleSet:
    values: dict[str, float]
    attempted: int
    eligible: int
    failed: int
    excluded: int
    evidence_digest: str
    nonpositive: int = 0
    nonfinite: int = 0
    missing: int = 0
    ambiguous: int = 0
    series_identities: frozenset[str] = frozenset()
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _TrialEvidence:
    block_id: str | None
    outcome: str
    experiment_id: str


@dataclass(frozen=True, slots=True)
class _MeasurementMemberSamples:
    values: dict[str, float]
    attempted: int
    failed: int = 0
    excluded: int = 0
    nonpositive: int = 0
    nonfinite: int = 0
    missing: int = 0
    ambiguous: int = 0
    series_identity: str | None = None
    evidence_digests: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()


def _evidence_digest(value: object) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return digest_model({"non_finite": "nan"})
        return digest_model({"non_finite": "positive" if value > 0 else "negative"})
    return digest_model(value)


@dataclass(frozen=True, slots=True)
class _ComparisonAssessment:
    invalidating: tuple[str, ...]
    exploratory: tuple[str, ...]
    paired: bool


@dataclass(frozen=True, slots=True)
class _ExperimentCohort:
    variant_id: str | None
    block_ids: frozenset[str]
    reasons: tuple[str, ...]


class RunSetService:
    def __init__(
        self,
        workspace: Workspace,
        *,
        snapshot_handle: SnapshotHandle | None = None,
    ) -> None:
        self.workspace = workspace
        self.snapshot_handle = snapshot_handle or Catalog(workspace).pin()
        self.store = ControlRecordStore(
            workspace,
            kind="run_sets",
            model=RunSet,
            id_field="run_set_id",
        )
        self.publisher = GenerationPublisher(workspace)
        self.retention = RetentionIntentStore(workspace)

    def freeze(self, request: FreezeRunSetRequest) -> RunSet:
        members: list[RunSetMember] = []
        selection = dict(request.selection)
        trial_experiment_ids: set[str] = set()
        trial_variant_ids: set[str] = set()
        requested: tuple[FreezeRunSetMember, ...]
        if isinstance(request, FreezeRunIdsRequest):
            requested = tuple(
                IncludedFreezeRunSetMember(run_id=run_id) for run_id in request.run_ids
            )
        else:
            requested = request.members
        catalog = Catalog(self.workspace)
        handle = (
            catalog.pin(request.corpus_commit_id)
            if request.corpus_commit_id is not None
            else self.snapshot_handle
        )
        with catalog.open_snapshot(handle) as snapshot:
            for order, item in enumerate(requested):
                exists = snapshot.execute(
                    "SELECT 1 FROM current_runs WHERE run_id = ?",
                    (item.run_id,),
                ).fetchone()
                if exists is None:
                    raise DomainError(
                        ErrorCode.RUN_NOT_FOUND,
                        f"Run {item.run_id!r} is absent from the pinned corpus snapshot.",
                        run_id=item.run_id,
                    )
                if item.trial_id is not None:
                    experiment_id = selection.get("experiment_id")
                    if isinstance(experiment_id, str):
                        trial_rows = snapshot.execute(
                            "SELECT DISTINCT experiment_id, variant_id, run_id FROM trials "
                            "WHERE trial_id = ? AND experiment_id = ?",
                            (item.trial_id, experiment_id),
                        ).fetchall()
                    else:
                        trial_rows = snapshot.execute(
                            "SELECT DISTINCT experiment_id, variant_id, run_id "
                            "FROM trials WHERE trial_id = ?",
                            (item.trial_id,),
                        ).fetchall()
                    if len(trial_rows) != 1 or str(trial_rows[0][2]) != item.run_id:
                        raise DomainError(
                            ErrorCode.COMPARISON_INVALID,
                            "Run-set trial identity is ambiguous or does not match its run.",
                            details={
                                "run_id": item.run_id,
                                "trial_id": item.trial_id,
                            },
                        )
                    trial_experiment_ids.add(str(trial_rows[0][0]))
                    trial_variant_ids.add(str(trial_rows[0][1]))
                if isinstance(item, ExcludedFreezeRunSetMember):
                    member: RunSetMember = ExcludedRunSetMember(
                        run_id=item.run_id,
                        trial_id=item.trial_id,
                        reason=item.reason,
                        order=order,
                    )
                else:
                    member = IncludedRunSetMember(
                        run_id=item.run_id,
                        trial_id=item.trial_id,
                        order=order,
                    )
                members.append(member)
            if len(trial_experiment_ids) > 1:
                raise DomainError(
                    ErrorCode.COMPARISON_INVALID,
                    "A run set cannot combine trials from different experiments.",
                )
            if trial_experiment_ids:
                selection.setdefault("experiment_id", next(iter(trial_experiment_ids)))
            if len(trial_variant_ids) == 1:
                selection.setdefault("variant_id", next(iter(trial_variant_ids)))
            corpus_commit_id = snapshot.handle.commit_id
            membership = [member.model_dump(mode="json") for member in members]
            membership_digest = digest_model(membership)
            run_set_id = digest_model(
                {
                    "corpus_commit_id": corpus_commit_id,
                    "selection": selection,
                    "members": membership,
                }
            )
            run_set = RunSet(
                run_set_id=run_set_id,
                corpus_commit_id=corpus_commit_id,
                selection=selection,
                members=tuple(members),
                membership_digest=membership_digest,
            )
            operation_identity = {
                "kind": "freeze_run_set",
                "run_set_id": run_set.run_set_id,
                "corpus_commit_id": run_set.corpus_commit_id,
                "membership_digest": run_set.membership_digest,
            }
            selection_json = json.dumps(
                run_set.selection,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            members_json = json.dumps(
                membership,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            estimated_bytes = (
                len(run_set.model_dump_json().encode("utf-8"))
                + len(selection_json.encode("utf-8"))
                + len(members_json.encode("utf-8"))
                + 64 * 1024
            )
            retention = self.retention.acquire(
                corpus_commit_id=corpus_commit_id,
                owner_kind="run_set",
                owner_id=run_set.run_set_id,
                operation_digest=digest_model(operation_identity),
            )
        if isinstance(retention, CompletedRetentionIntent):
            existing = self.store.read(run_set.run_set_id)
            if not self._same_run_set(existing, run_set):
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "A completed run-set publication has different immutable content.",
                    details={"run_set_id": run_set.run_set_id},
                )
            return existing
        StorageQuota(self.workspace).require_capacity(
            additional_bytes=estimated_bytes,
            staging=True,
        )
        try:
            self.store.create(run_set)
        except DomainError as error:
            if error.code is not ErrorCode.REVISION_CONFLICT:
                raise
            existing = self.store.read(run_set.run_set_id)
            if not self._same_run_set(existing, run_set):
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "The run-set identity is already bound to different immutable content.",
                    details={"run_set_id": run_set.run_set_id},
                ) from error
            run_set = existing
        published = self.publisher.publish_rows_idempotent(
            {
                "run_sets": [
                    {
                        "run_set_id": run_set.run_set_id,
                        "corpus_commit_id": run_set.corpus_commit_id,
                        "created_at": run_set.created_at,
                        "selection_json": selection_json,
                        "members_json": members_json,
                        "membership_digest": run_set.membership_digest,
                    }
                ]
            },
            publisher="flameox.run_sets",
            publisher_version="1",
            input_run_ids=tuple(member.run_id for member in members),
            operation_identity=operation_identity,
            supersede_matching=False,
        )
        self.retention.complete(
            retention,
            materialized_commit_id=published.commit.commit_id,
        )
        return run_set

    @staticmethod
    def _same_run_set(left: RunSet, right: RunSet) -> bool:
        return (
            left.run_set_id == right.run_set_id
            and left.corpus_commit_id == right.corpus_commit_id
            and left.selection == right.selection
            and left.members == right.members
            and left.membership_digest == right.membership_digest
        )


class ComparisonService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.run_sets = ControlRecordStore(
            workspace,
            kind="run_sets",
            model=RunSet,
            id_field="run_set_id",
        )
        self.publisher = GenerationPublisher(workspace)

    def compare(self, request: CompareRunSetsRequest) -> ComparisonResult:
        baseline_set, candidate_set, handle = self._comparison_inputs(request)
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(handle) as snapshot:
            return self._compare_at_snapshot(
                request,
                snapshot,
                baseline_set=baseline_set,
                candidate_set=candidate_set,
            )

    async def compare_async(
        self,
        request: CompareRunSetsRequest,
        *,
        progress: Callable[[float, float, str], Awaitable[None]] | None = None,
    ) -> ComparisonResult:
        baseline_set, candidate_set, handle = self._comparison_inputs(request)
        catalog = Catalog(self.workspace)
        reporter = ProgressReporter(progress)
        await reporter.report(0, 2, "Comparison snapshot pinned")
        result = await catalog.run_interruptible(
            lambda snapshot: self._compare_at_snapshot(
                request,
                snapshot,
                baseline_set=baseline_set,
                candidate_set=candidate_set,
            ),
            handle=handle,
            query_name="compare_run_sets",
        )
        await reporter.report(1, 2, "Comparison query complete")
        await reporter.report(2, 2, "Comparison result ready")
        return result

    def _compare_at_snapshot(
        self,
        request: CompareRunSetsRequest,
        snapshot: Snapshot,
        *,
        baseline_set: RunSet,
        candidate_set: RunSet,
    ) -> ComparisonResult:
        experiment: Experiment | None = None
        if request.experiment_id is not None:
            experiment = ControlRecordStore(
                self.workspace,
                kind="experiments",
                model=Experiment,
                id_field="experiment_id",
            ).read(request.experiment_id)
        corpus_commit_id = snapshot.commit.commit_id
        baseline = self._samples(snapshot, baseline_set, request)
        candidate = self._samples(snapshot, candidate_set, request)
        assessment = self._assess_comparison(
            snapshot,
            request,
            experiment,
            baseline_set,
            candidate_set,
            baseline,
            candidate,
        )
        profile_changes = self._profile_changes(
            snapshot,
            baseline_set,
            candidate_set,
            polarity=request.polarity,
        )
        comparison = self._calculate_comparison(
            request,
            experiment,
            baseline_set,
            candidate_set,
            baseline,
            candidate,
            assessment,
        )
        return ComparisonResult(
            comparison=comparison,
            baseline_run_set=baseline_set,
            candidate_run_set=candidate_set,
            corpus_commit_id=corpus_commit_id,
            profile_changes=profile_changes,
        )

    def _assess_comparison(
        self,
        snapshot: Snapshot,
        request: CompareRunSetsRequest,
        experiment: Experiment | None,
        baseline_set: RunSet,
        candidate_set: RunSet,
        baseline: _SampleSet,
        candidate: _SampleSet,
    ) -> _ComparisonAssessment:
        invalidating, exploratory = self._compatibility_mismatches(
            snapshot,
            baseline_set,
            candidate_set,
        )
        invalidating.extend(f"baseline: {reason}" for reason in baseline.issues)
        invalidating.extend(f"candidate: {reason}" for reason in candidate.issues)
        baseline_paired = self._run_set_is_paired(baseline_set)
        candidate_paired = self._run_set_is_paired(candidate_set)
        if baseline_paired != candidate_paired:
            invalidating.append("treatments disagree about the independent-unit design")
        paired = baseline_paired or candidate_paired
        expected_estimand = "median_paired_log_ratio" if paired else "difference_in_median_logs"
        if request.estimand is not None and request.estimand != expected_estimand:
            invalidating.append("declared estimand does not match the independent-unit design")
        if experiment is not None:
            invalidating.extend(
                self._experiment_protocol_mismatches(
                    snapshot,
                    request,
                    experiment,
                    baseline_set,
                    candidate_set,
                )
            )
        self._extend_metric_contract_mismatches(
            snapshot,
            request,
            baseline_set,
            candidate_set,
            baseline,
            candidate,
            invalidating,
        )
        self._extend_sample_coverage_mismatches(baseline, candidate, invalidating)
        if paired and set(baseline.values) != set(candidate.values):
            invalidating.append("paired block coverage differs across treatments")
        if baseline.eligible == 0:
            invalidating.append("baseline run set has no eligible measurements")
        if candidate.eligible == 0:
            invalidating.append("candidate run set has no eligible measurements")
        return _ComparisonAssessment(
            invalidating=tuple(dict.fromkeys(invalidating)),
            exploratory=tuple(dict.fromkeys(exploratory)),
            paired=paired,
        )

    def _calculate_comparison(
        self,
        request: CompareRunSetsRequest,
        experiment: Experiment | None,
        baseline_set: RunSet,
        candidate_set: RunSet,
        baseline: _SampleSet,
        candidate: _SampleSet,
        assessment: _ComparisonAssessment,
    ) -> Comparison:
        comparison_id = digest_model(
            {
                "recipe": "compare_run_sets.v1",
                "request": request.model_dump(
                    mode="json",
                    exclude={"baseline_run_set_id", "candidate_run_set_id"},
                ),
                "baseline_membership": baseline_set.membership_digest,
                "candidate_membership": candidate_set.membership_digest,
                "baseline_evidence": baseline.evidence_digest,
                "candidate_evidence": candidate.evidence_digest,
                "compatibility": {
                    "invalidating": assessment.invalidating,
                    "exploratory": assessment.exploratory,
                    "paired": assessment.paired,
                },
            }
        )
        measurement_series = (
            request.series if isinstance(request, MeasurementCompareRunSetsRequest) else None
        )
        if assessment.paired:
            comparison = compare_paired_samples(
                comparison_id=comparison_id,
                baseline_run_set_id=baseline_set.run_set_id,
                candidate_run_set_id=candidate_set.run_set_id,
                baseline_by_block=baseline.values,
                candidate_by_block=candidate.values,
                metric=request.metric,
                unit=request.unit,
                polarity=request.polarity,
                practical_threshold=request.practical_threshold,
                confidence_level=request.confidence_level,
                random_seed=request.random_seed,
                experiment_id=request.experiment_id,
                metric_source=request.metric_source,
                measurement_series=measurement_series,
            )
        else:
            comparison = compare_unpaired_samples(
                comparison_id=comparison_id,
                baseline_run_set_id=baseline_set.run_set_id,
                candidate_run_set_id=candidate_set.run_set_id,
                baseline_values=tuple(baseline.values.values()),
                candidate_values=tuple(candidate.values.values()),
                metric=request.metric,
                unit=request.unit,
                polarity=request.polarity,
                practical_threshold=request.practical_threshold,
                confidence_level=request.confidence_level,
                random_seed=request.random_seed,
                metric_source=request.metric_source,
                measurement_series=measurement_series,
            )
        series_ids = baseline.series_identities | candidate.series_identities
        series_id = next(iter(series_ids)) if len(series_ids) == 1 else None
        effective_estimand = (
            "median_paired_log_ratio" if assessment.paired else "difference_in_median_logs"
        )
        metric_contract_id = digest_model(
            {
                "schema_version": 1,
                "source": request.metric_source,
                "metric": request.metric,
                "unit": request.unit,
                "polarity": request.polarity,
                "estimand": effective_estimand,
                "value_domain": "strictly_positive",
                "zero_policy": "reject",
                "measurement_series_id": series_id,
            }
        )
        protocol_identity_id = (
            digest_model(
                {
                    "experiment": experiment.model_dump(mode="json"),
                    "metric_contract_id": metric_contract_id,
                    "baseline_membership": baseline_set.membership_digest,
                    "candidate_membership": candidate_set.membership_digest,
                }
            )
            if experiment is not None
            else None
        )
        comparison = comparison.validated_copy(
            update={
                "estimand": effective_estimand,
                "metric_contract_id": metric_contract_id,
                "measurement_series_id": series_id,
                "protocol_identity_id": protocol_identity_id,
                "baseline_attempted_n": baseline.attempted,
                "baseline_eligible_n": baseline.eligible,
                "baseline_failed_n": baseline.failed,
                "baseline_excluded_n": baseline.excluded,
                "baseline_missing_n": baseline.missing + baseline.ambiguous,
                "baseline_out_of_domain_n": baseline.nonfinite + baseline.nonpositive,
                "candidate_attempted_n": candidate.attempted,
                "candidate_eligible_n": candidate.eligible,
                "candidate_failed_n": candidate.failed,
                "candidate_excluded_n": candidate.excluded,
                "candidate_missing_n": candidate.missing + candidate.ambiguous,
                "candidate_out_of_domain_n": candidate.nonfinite + candidate.nonpositive,
            },
        )
        all_reasons = tuple(
            dict.fromkeys(
                (
                    *comparison.mismatches,
                    *assessment.invalidating,
                    *assessment.exploratory,
                )
            )
        )
        if assessment.invalidating:
            comparison = comparison.validated_copy(
                update={
                    "validity": ComparisonValidity.INVALID,
                    "decision": ComparisonDecision.INCONCLUSIVE,
                    "mismatches": all_reasons,
                },
            )
        elif assessment.exploratory:
            comparison = comparison.validated_copy(
                update={
                    "validity": ComparisonValidity.EXPLORATORY,
                    "decision": ComparisonDecision.INCONCLUSIVE,
                    "mismatches": all_reasons,
                },
            )
        return comparison

    @staticmethod
    def _run_set_is_paired(run_set: RunSet) -> bool:
        included = tuple(member for member in run_set.members if member.included)
        return bool(included) and all(member.trial_id is not None for member in included)

    def _extend_metric_contract_mismatches(
        self,
        snapshot: Snapshot,
        request: CompareRunSetsRequest,
        baseline_set: RunSet,
        candidate_set: RunSet,
        baseline: _SampleSet,
        candidate: _SampleSet,
        invalidating: list[str],
    ) -> None:
        if isinstance(request, RuntimeResourceCompareRunSetsRequest):
            invalidating.extend(
                self._runtime_resource_compatibility_mismatches(
                    snapshot,
                    baseline_set,
                    candidate_set,
                    request.metric,
                )
            )
            return
        if baseline.series_identities != candidate.series_identities:
            invalidating.append("measurement series identity differs across treatments")
        if len(baseline.series_identities | candidate.series_identities) != 1:
            invalidating.append("comparison requires one exact measurement series identity")

    @staticmethod
    def _extend_sample_coverage_mismatches(
        baseline: _SampleSet,
        candidate: _SampleSet,
        invalidating: list[str],
    ) -> None:
        for treatment, samples in (("baseline", baseline), ("candidate", candidate)):
            expected = samples.attempted - samples.failed - samples.excluded
            if samples.eligible != expected:
                invalidating.append(
                    f"{treatment} selected endpoint is incomplete: "
                    f"expected {expected} independent units, observed {samples.eligible}"
                )
            if samples.missing:
                invalidating.append(
                    f"{treatment} is missing {samples.missing} selected endpoint values"
                )
            if samples.ambiguous:
                invalidating.append(
                    f"{treatment} has {samples.ambiguous} ambiguous endpoint selections"
                )
            if samples.nonfinite:
                invalidating.append(
                    f"{treatment} contains {samples.nonfinite} non-finite endpoint values"
                )
            if samples.nonpositive:
                invalidating.append(
                    f"{treatment} contains {samples.nonpositive} values outside the "
                    "strictly-positive metric contract"
                )

    def _experiment_protocol_mismatches(
        self,
        snapshot: Snapshot,
        request: CompareRunSetsRequest,
        experiment: Experiment,
        baseline_set: RunSet,
        candidate_set: RunSet,
    ) -> list[str]:
        reasons = self._experiment_field_mismatches(request, experiment)
        baseline = self._experiment_cohort(
            snapshot,
            experiment.experiment_id,
            "baseline",
            baseline_set,
        )
        candidate = self._experiment_cohort(
            snapshot,
            experiment.experiment_id,
            "candidate",
            candidate_set,
        )
        reasons.extend(baseline.reasons)
        reasons.extend(candidate.reasons)
        if baseline.variant_id is not None and baseline.variant_id == candidate.variant_id:
            reasons.append("baseline and candidate select the same experiment variant")
        if baseline.block_ids != candidate.block_ids:
            reasons.append("experiment treatments do not represent the same declared blocks")
        fixed_blocks = experiment.stopping_rule.get("fixed_blocks")
        if isinstance(fixed_blocks, int) and fixed_blocks > 0:
            if len(baseline.block_ids) < fixed_blocks or len(candidate.block_ids) < fixed_blocks:
                reasons.append("experiment stopping rule has not reached its fixed block count")
        else:
            reasons.append("experiment has no evaluable fixed-block stopping rule")
        return list(dict.fromkeys(reasons))

    @staticmethod
    def _experiment_field_mismatches(
        request: CompareRunSetsRequest,
        experiment: Experiment,
    ) -> list[str]:
        reasons: list[str] = []
        observed = {
            "metric": request.metric,
            "polarity": request.polarity,
            "estimand": request.estimand,
            "practical_threshold": request.practical_threshold,
            "confidence_level": request.confidence_level,
            "random_seed": request.random_seed,
            "metric_source": request.metric_source,
            "unit": request.unit,
        }
        declared = {
            "metric": experiment.primary_metric,
            "polarity": experiment.polarity,
            "estimand": experiment.estimand,
            "practical_threshold": experiment.practical_threshold,
            "confidence_level": experiment.confidence_level,
            "random_seed": experiment.random_seed,
            "metric_source": experiment.metric_source,
            "unit": experiment.primary_metric_unit,
        }
        for field_name, observed_value in observed.items():
            if observed_value != declared[field_name]:
                reasons.append(f"comparison {field_name} differs from the experiment protocol")
        if experiment.value_domain is None or experiment.zero_policy is None:
            reasons.append("experiment metric-domain contract is incomplete")
        if isinstance(request, MeasurementCompareRunSetsRequest):
            if request.series != experiment.measurement_series:
                reasons.append("measurement series selector differs from the experiment protocol")
        elif experiment.measurement_series is not None:
            reasons.append("runtime-resource experiment cannot declare a measurement series")
        return reasons

    def _experiment_cohort(
        self,
        snapshot: Snapshot,
        experiment_id: str,
        treatment: str,
        run_set: RunSet,
    ) -> _ExperimentCohort:
        reasons: list[str] = []
        if run_set.selection.get("experiment_id") != experiment_id:
            reasons.append(f"{treatment} run set is not bound to the declared experiment")
        if any(member.trial_id is None for member in run_set.members):
            reasons.append(f"{treatment} experiment cohort contains a run without a trial identity")
            return _ExperimentCohort(None, frozenset(), tuple(reasons))
        member_rows: list[tuple[str, str, str, str | None]] = []
        for member in run_set.members:
            rows = snapshot.execute(
                "SELECT DISTINCT variant_id, block_id FROM trials "
                "WHERE experiment_id = ? AND trial_id = ? AND run_id = ?",
                (experiment_id, member.trial_id, member.run_id),
            ).fetchall()
            if len(rows) != 1:
                reasons.append(f"{treatment} member does not resolve to one exact experiment trial")
                continue
            member_rows.append(
                (
                    str(member.trial_id),
                    member.run_id,
                    str(rows[0][0]),
                    str(rows[0][1]) if rows[0][1] is not None else None,
                )
            )
        variant_ids = {row[2] for row in member_rows}
        if len(variant_ids) != 1:
            reasons.append(f"{treatment} run set does not select exactly one experiment variant")
            return _ExperimentCohort(None, frozenset(), tuple(reasons))
        variant_id = next(iter(variant_ids))
        if run_set.selection.get("variant_id") not in {None, variant_id}:
            reasons.append(f"{treatment} run-set variant identity is inconsistent")
        self._check_variant_label(snapshot, experiment_id, treatment, run_set, variant_id, reasons)
        population_rows = snapshot.execute(
            "SELECT DISTINCT trial_id, run_id, block_id FROM trials "
            "WHERE experiment_id = ? AND variant_id = ?",
            (experiment_id, variant_id),
        ).fetchall()
        declared_population = {
            (str(trial_id), str(run_id) if run_id is not None else None)
            for trial_id, run_id, _block_id in population_rows
        }
        selected_population = {(row[0], row[1]) for row in member_rows}
        if selected_population != declared_population:
            reasons.append(
                f"{treatment} run set does not represent the full declared trial population"
            )
        block_ids = frozenset(
            str(block_id)
            for _trial_id, _run_id, block_id in population_rows
            if block_id is not None
        )
        if len(block_ids) != len(population_rows):
            reasons.append(f"{treatment} experiment population has missing or duplicate blocks")
        return _ExperimentCohort(variant_id, block_ids, tuple(reasons))

    @staticmethod
    def _check_variant_label(
        snapshot: Snapshot,
        experiment_id: str,
        treatment: str,
        run_set: RunSet,
        variant_id: str,
        reasons: list[str],
    ) -> None:
        selected_name = run_set.selection.get("variant")
        if not isinstance(selected_name, str):
            return
        rows = snapshot.execute(
            "SELECT DISTINCT name FROM variants WHERE experiment_id = ? AND variant_id = ?",
            (experiment_id, variant_id),
        ).fetchall()
        if len(rows) != 1 or str(rows[0][0]) != selected_name:
            reasons.append(f"{treatment} run-set variant label is inconsistent")

    def record(self, request: CompareRunSetsRequest) -> ComparisonResult:
        started = datetime.now(UTC)
        result = self.compare(request)
        return self._record_result(request, result=result, started=started)

    async def record_async(
        self,
        request: CompareRunSetsRequest,
        *,
        progress: Callable[[float, float, str], Awaitable[None]] | None = None,
    ) -> ComparisonResult:
        started = datetime.now(UTC)
        reporter = ProgressReporter(progress)
        await reporter.report(0, 3, "Comparison snapshot pinned")
        baseline_set, candidate_set, handle = self._comparison_inputs(request)
        catalog = Catalog(self.workspace)
        result = await catalog.run_interruptible(
            lambda snapshot: self._compare_at_snapshot(
                request,
                snapshot,
                baseline_set=baseline_set,
                candidate_set=candidate_set,
            ),
            handle=handle,
            query_name="record_comparison",
        )
        await reporter.report(1, 3, "Comparison query complete")
        await reporter.report(2, 3, "Publishing comparison provenance")
        recorded = await run_atomic_thread(
            lambda: self._record_result(request, result=result, started=started)
        )
        await reporter.report(3, 3, "Comparison publication complete")
        return recorded

    def _comparison_inputs(
        self,
        request: CompareRunSetsRequest,
    ) -> tuple[RunSet, RunSet, SnapshotHandle]:
        baseline_set = self.run_sets.read(request.baseline_run_set_id)
        candidate_set = self.run_sets.read(request.candidate_run_set_id)
        if baseline_set.corpus_commit_id != candidate_set.corpus_commit_id:
            raise DomainError(
                ErrorCode.COMPARISON_INVALID,
                "Run sets from different corpus snapshots cannot be compared.",
                details={
                    "baseline_corpus_commit_id": baseline_set.corpus_commit_id,
                    "candidate_corpus_commit_id": candidate_set.corpus_commit_id,
                },
                remediation=(
                    "Freeze every treatment through one RunSetService request boundary, "
                    "or explicitly re-freeze them at the same corpus commit.",
                ),
            )
        return (
            baseline_set,
            candidate_set,
            Catalog(self.workspace).pin(baseline_set.corpus_commit_id),
        )

    def _record_result(
        self,
        request: CompareRunSetsRequest,
        *,
        result: ComparisonResult,
        started: datetime,
    ) -> ComparisonResult:
        completed = datetime.now(UTC)
        input_run_ids = tuple(
            member.run_id
            for run_set in (result.baseline_run_set, result.candidate_run_set)
            for member in run_set.members
        )
        result_digest = digest_model(
            {
                "comparison": result.comparison.model_dump(mode="json"),
                "profile_changes": [
                    item.model_dump(mode="json") for item in result.profile_changes
                ],
            }
        )
        parameters = request.model_dump(mode="json")
        provenance = build_analysis_provenance(
            AnalysisProvenanceInput(
                recipe="compare_run_sets",
                parameters=parameters,
                corpus_commit_id=result.corpus_commit_id,
                input_run_ids=input_run_ids,
                result_digest=result_digest,
                coverage={
                    "baseline_attempted": result.comparison.baseline_attempted_n,
                    "baseline_eligible": result.comparison.baseline_eligible_n,
                    "baseline_failed": result.comparison.baseline_failed_n,
                    "baseline_excluded": result.comparison.baseline_excluded_n,
                    "baseline_missing": result.comparison.baseline_missing_n,
                    "baseline_out_of_domain": (result.comparison.baseline_out_of_domain_n),
                    "candidate_attempted": result.comparison.candidate_attempted_n,
                    "candidate_eligible": result.comparison.candidate_eligible_n,
                    "candidate_failed": result.comparison.candidate_failed_n,
                    "candidate_excluded": result.comparison.candidate_excluded_n,
                    "candidate_missing": result.comparison.candidate_missing_n,
                    "candidate_out_of_domain": (result.comparison.candidate_out_of_domain_n),
                    "complete_pairs": result.comparison.complete_pair_n or 0,
                    "profile_changes": len(result.profile_changes),
                },
                limitations=(
                    ("Compatibility mismatches invalidate proof.",)
                    if result.comparison.validity is ComparisonValidity.INVALID
                    else (
                        "Comparison is exploratory; incomplete identity or "
                        "missing oracle prevents confirmatory proof.",
                    )
                    if result.comparison.validity is ComparisonValidity.EXPLORATORY
                    else ()
                ),
                started_at=started,
                completed_at=completed,
                references=context_references(
                    run_set_ids=(
                        result.baseline_run_set.run_set_id,
                        result.candidate_run_set.run_set_id,
                    )
                ),
            )
        )
        rows = provenance.rows()
        rows["comparisons"] = [self._comparison_row(result.comparison)]
        published = self.publisher.publish_rows(
            rows,
            publisher="flameox.comparisons",
            publisher_version="1",
            input_run_ids=input_run_ids,
        )
        return result.validated_copy(
            update={
                "analysis": provenance.analysis,
                "evidence": provenance.evidence,
                "materialized_commit_id": published.commit.commit_id,
            }
        )

    def _samples(
        self,
        snapshot: Snapshot,
        run_set: RunSet,
        request: CompareRunSetsRequest,
    ) -> _SampleSet:
        if isinstance(request, RuntimeResourceCompareRunSetsRequest):
            return self._runtime_resource_samples(snapshot, run_set, request.metric)
        return self._measurement_samples(snapshot, run_set, request)

    def _measurement_samples(
        self,
        snapshot: Snapshot,
        run_set: RunSet,
        request: MeasurementCompareRunSetsRequest,
    ) -> _SampleSet:
        values: dict[str, float] = {}
        attempted = 0
        failed = 0
        excluded = 0
        nonpositive = 0
        nonfinite = 0
        missing = 0
        ambiguous = 0
        series_identities: set[str] = set()
        issues: list[str] = []
        evidence_digests: list[str] = []
        included = tuple(member for member in run_set.members if member.included)
        if len(included) > 1 and any(member.trial_id is None for member in included):
            raise DomainError(
                ErrorCode.COMPARISON_INVALID,
                "Multi-run comparisons require explicit trial identities.",
            )
        for member in run_set.members:
            collected = self._measurement_member_samples(snapshot, run_set, member, request)
            overlap = set(values) & set(collected.values)
            if overlap:
                raise DomainError(
                    ErrorCode.COMPARISON_INVALID,
                    "Run set contains duplicate independent-unit identities.",
                    details={"independent_units": sorted(overlap)},
                )
            values.update(collected.values)
            attempted += collected.attempted
            failed += collected.failed
            excluded += collected.excluded
            nonpositive += collected.nonpositive
            nonfinite += collected.nonfinite
            missing += collected.missing
            ambiguous += collected.ambiguous
            if collected.series_identity is not None:
                series_identities.add(collected.series_identity)
            evidence_digests.extend(collected.evidence_digests)
            issues.extend(collected.issues)
        return _SampleSet(
            values=values,
            attempted=attempted,
            eligible=len(values),
            failed=failed,
            excluded=excluded,
            nonpositive=nonpositive,
            nonfinite=nonfinite,
            missing=missing,
            ambiguous=ambiguous,
            evidence_digest=digest_model(sorted(evidence_digests)),
            series_identities=frozenset(series_identities),
            issues=tuple(issues[:32]),
        )

    def _measurement_member_samples(
        self,
        snapshot: Snapshot,
        run_set: RunSet,
        member: RunSetMember,
        request: MeasurementCompareRunSetsRequest,
    ) -> _MeasurementMemberSamples:
        if not member.included:
            return _MeasurementMemberSamples(values={}, attempted=1, excluded=1)
        trial = self._trial_evidence(snapshot, run_set, member)
        if trial is not None and trial.outcome != "succeeded":
            return _MeasurementMemberSamples(values={}, attempted=1, failed=1)
        block_key = trial.block_id if trial is not None else None
        if trial is not None and block_key is None:
            raise DomainError(
                ErrorCode.COMPARISON_INVALID,
                "Paired experiment trials require block identities.",
                details={"trial_id": member.trial_id},
            )
        rows = self._selected_measurement_rows(snapshot, member.run_id, request)
        selected_evidence_digests = tuple(
            sorted(digest_model([_evidence_digest(value) for value in row]) for row in rows)
        )
        by_series: dict[str, list[tuple[object, ...]]] = {}
        for row in rows:
            signature = self._measurement_series_signature(row)
            series_id = digest_model(signature)
            by_series.setdefault(series_id, []).append(row)
        if not by_series:
            return _MeasurementMemberSamples(
                values={},
                attempted=1,
                missing=1,
                issues=(f"run {member.run_id} has no selected measurement series",),
            )
        if len(by_series) != 1:
            return _MeasurementMemberSamples(
                values={},
                attempted=1,
                ambiguous=1,
                evidence_digests=selected_evidence_digests,
                issues=(f"run {member.run_id} has {len(by_series)} matching measurement series",),
            )
        series_id, selected_rows = next(iter(by_series.items()))
        evidence_digests = selected_evidence_digests
        if self._measurement_series_signature(selected_rows[0])["numeric_kind"] == (
            "unsigned_integer"
        ):
            return _MeasurementMemberSamples(
                values={},
                attempted=1,
                missing=1,
                series_identity=series_id,
                evidence_digests=evidence_digests,
                issues=(
                    f"run {member.run_id} uses exact uint64 measurements; "
                    "confirmatory floating-point comparison is unsupported",
                ),
            )
        if block_key is not None:
            return self._paired_measurement_member(
                block_key, series_id, selected_rows, evidence_digests=evidence_digests
            )
        return self._unpaired_measurement_member(
            member.run_id, series_id, selected_rows, evidence_digests=evidence_digests
        )

    def _selected_measurement_rows(
        self,
        snapshot: Snapshot,
        run_id: str,
        request: MeasurementCompareRunSetsRequest,
    ) -> list[tuple[object, ...]]:
        rows = snapshot.execute(
            "SELECT value_int, value_float, value_uint, value_kind, block_id, worker_id, "
            "worker_run_index, value_index, aggregation, scope, loop_count, "
            "phase, dimensions, evidence_level, artifact_id, measurement_id "
            "FROM measurements WHERE run_id = ? AND name = ? AND unit = ? "
            "AND is_warmup = false ORDER BY measurement_id",
            (run_id, request.metric, request.unit),
        ).fetchall()
        if request.series is None:
            return rows
        return [
            row
            for row in rows
            if self._series_matches_selector(
                self._measurement_series_signature(row),
                request.series,
            )
        ]

    @classmethod
    def _paired_measurement_member(
        cls,
        block_key: str,
        series_id: str,
        rows: list[tuple[object, ...]],
        *,
        evidence_digests: tuple[str, ...],
    ) -> _MeasurementMemberSamples:
        valid, missing, nonfinite, nonpositive = cls._classify_measurement_rows(rows)
        values = {block_key: statistics.median(valid)} if len(valid) == len(rows) else {}
        return _MeasurementMemberSamples(
            values=values,
            attempted=1,
            nonpositive=nonpositive,
            nonfinite=nonfinite,
            missing=int(bool(missing)),
            series_identity=series_id,
            evidence_digests=evidence_digests,
        )

    @classmethod
    def _unpaired_measurement_member(
        cls,
        run_id: str,
        series_id: str,
        rows: list[tuple[object, ...]],
        *,
        evidence_digests: tuple[str, ...],
    ) -> _MeasurementMemberSamples:
        by_worker: dict[str, list[tuple[object, ...]]] = {}
        for row in rows:
            worker_id = str(row[5]) if row[5] is not None else "run"
            by_worker.setdefault(f"{run_id}:{worker_id}", []).append(row)
        values: dict[str, float] = {}
        missing = 0
        nonfinite = 0
        nonpositive = 0
        for worker_key, worker_rows in sorted(by_worker.items()):
            valid, worker_missing, worker_nonfinite, worker_nonpositive = (
                cls._classify_measurement_rows(worker_rows)
            )
            missing += int(bool(worker_missing))
            nonfinite += worker_nonfinite
            nonpositive += worker_nonpositive
            if len(valid) == len(worker_rows):
                values[worker_key] = statistics.median(valid)
        return _MeasurementMemberSamples(
            values=values,
            attempted=len(by_worker),
            missing=missing,
            nonfinite=nonfinite,
            nonpositive=nonpositive,
            series_identity=series_id,
            evidence_digests=evidence_digests,
        )

    @classmethod
    def _classify_measurement_rows(
        cls,
        rows: list[tuple[object, ...]],
    ) -> tuple[list[float], int, int, int]:
        values = [cls._measurement_value(row) for row in rows]
        return (
            [value for value in values if value is not None and math.isfinite(value) and value > 0],
            sum(value is None for value in values),
            sum(value is not None and not math.isfinite(value) for value in values),
            sum(value is not None and math.isfinite(value) and value <= 0 for value in values),
        )

    @staticmethod
    def _measurement_value(row: tuple[object, ...]) -> float | None:
        value = row[0] if row[0] is not None else row[1]
        return float(cast(str | int | float, value)) if value is not None else None

    @staticmethod
    def _trial_evidence(
        snapshot: Snapshot,
        run_set: RunSet,
        member: RunSetMember,
    ) -> _TrialEvidence | None:
        if member.trial_id is None:
            return None
        experiment_id = run_set.selection.get("experiment_id")
        if not isinstance(experiment_id, str):
            raise DomainError(
                ErrorCode.COMPARISON_INVALID,
                "Trial-bound run sets require an exact experiment identity.",
                details={"trial_id": member.trial_id, "run_id": member.run_id},
            )
        rows = snapshot.execute(
            "SELECT DISTINCT block_id, outcome, experiment_id FROM trials "
            "WHERE trial_id = ? AND experiment_id = ? AND run_id = ?",
            (member.trial_id, experiment_id, member.run_id),
        ).fetchall()
        if len(rows) != 1:
            raise DomainError(
                ErrorCode.COMPARISON_INVALID,
                "Run-set trial evidence is missing or ambiguous for its exact identity.",
                details={
                    "experiment_id": experiment_id,
                    "trial_id": member.trial_id,
                    "run_id": member.run_id,
                },
            )
        block_id, outcome, observed_experiment_id = rows[0]
        return _TrialEvidence(
            block_id=str(block_id) if block_id is not None else None,
            outcome=str(outcome),
            experiment_id=str(observed_experiment_id),
        )

    @staticmethod
    def _measurement_series_signature(row: tuple[object, ...]) -> dict[str, JsonValue]:
        raw_dimensions = row[12]
        raw_items = (
            cast(Mapping[object, object], raw_dimensions).items()
            if isinstance(raw_dimensions, Mapping)
            else cast(list[tuple[object, object]], raw_dimensions or [])
        )
        dimensions = {str(key): str(value) for key, value in raw_items}
        loop_count = (
            None
            if dimensions.get("loop_semantics") == "pyperf_normalized_per_loop"
            else int(cast(str | int, row[10]))
            if row[10] is not None
            else None
        )
        numeric_kind = (
            str(row[3])
            if row[3] is not None
            else "integer"
            if row[0] is not None
            else "floating"
            if row[1] is not None
            else "unsigned_integer"
            if row[2] is not None
            else "missing"
        )
        return {
            "aggregation": str(row[8]),
            "scope": str(row[9]),
            "loop_count": loop_count,
            "phase": str(row[11]) if row[11] is not None else None,
            "dimensions": cast(JsonValue, dimensions),
            "numeric_kind": numeric_kind,
            "evidence_level": str(row[13]),
            "artifact_bound": row[14] is not None,
        }

    @staticmethod
    def _series_matches_selector(
        signature: dict[str, JsonValue],
        selector: MeasurementSeriesSelector,
    ) -> bool:
        return (
            signature["scope"] == selector.scope
            and signature["aggregation"] == selector.aggregation
            and signature["phase"] == selector.phase
            and signature["loop_count"] == selector.loop_count
            and signature["dimensions"] == selector.dimensions
        )

    def _runtime_resource_samples(
        self,
        snapshot: Snapshot,
        run_set: RunSet,
        metric: str,
    ) -> _SampleSet:
        definition = runtime_resource_metric_definition(metric)
        column = definition.evidence_column
        values: dict[str, float] = {}
        attempted = len(run_set.members)
        failed = 0
        excluded = 0
        nonpositive = 0
        nonfinite = 0
        missing = 0
        evidence_rows: list[object] = []
        included = tuple(member for member in run_set.members if member.included)
        if len(included) > 1 and any(member.trial_id is None for member in included):
            raise DomainError(
                ErrorCode.COMPARISON_INVALID,
                "Multi-run paired comparisons require explicit trial identities.",
            )
        for member in run_set.members:
            if not member.included:
                excluded += 1
                continue
            block_key = str(member.order)
            trial = self._trial_evidence(snapshot, run_set, member)
            if trial is not None:
                if trial.outcome != "succeeded":
                    failed += 1
                    continue
                if trial.block_id is None:
                    raise DomainError(
                        ErrorCode.COMPARISON_INVALID,
                        "Paired experiment trials require block identities.",
                        details={"trial_id": member.trial_id},
                    )
                block_key = trial.block_id
            rows = snapshot.execute(
                f"SELECT {column}, unavailable_metrics, sampling_interval_ms, peak_rss_backend "
                "FROM runtime_resource_summaries "
                "WHERE run_id = ?",
                (member.run_id,),
            ).fetchall()
            if len(rows) > 1:
                raise DomainError(
                    ErrorCode.COMPARISON_INVALID,
                    "Run has ambiguous runtime-resource summary evidence.",
                    details={"run_id": member.run_id},
                )
            if not rows:
                missing += 1
                continue
            evidence_rows.append((member.run_id, rows[0]))
            value, unavailable_metrics, _, _ = rows[0]
            if value is None or definition.unavailable_key in set(unavailable_metrics or []):
                missing += 1
                continue
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                nonfinite += 1
                continue
            if numeric_value <= 0:
                nonpositive += 1
                continue
            if block_key in values:
                raise DomainError(
                    ErrorCode.COMPARISON_INVALID,
                    "Run set contains more than one included trial for a block.",
                    details={"block_id": block_key},
                )
            values[block_key] = numeric_value
        return _SampleSet(
            values=values,
            attempted=attempted,
            eligible=len(values),
            failed=failed,
            excluded=excluded,
            nonpositive=nonpositive,
            nonfinite=nonfinite,
            missing=missing,
            evidence_digest=digest_model(evidence_rows),
        )

    def _runtime_resource_compatibility_mismatches(
        self,
        snapshot: Snapshot,
        baseline: RunSet,
        candidate: RunSet,
        metric: str,
    ) -> list[str]:
        definition = runtime_resource_metric_definition(metric)
        reasons: list[str] = []
        configurations: dict[str, set[tuple[object, ...]]] = {"baseline": set(), "candidate": set()}
        for treatment, run_set in (("baseline", baseline), ("candidate", candidate)):
            for member in run_set.members:
                if not member.included:
                    continue
                trial = self._trial_evidence(snapshot, run_set, member)
                if trial is not None and trial.outcome != "succeeded":
                    continue
                column = definition.evidence_column
                rows = snapshot.execute(
                    f"SELECT {column}, unavailable_metrics, sampling_interval_ms, "
                    "peak_rss_backend "
                    "FROM runtime_resource_summaries WHERE run_id = ?",
                    (member.run_id,),
                ).fetchall()
                if len(rows) != 1:
                    continue
                value, unavailable_metrics, interval, backend = rows[0]
                if (
                    value is None
                    or value <= 0
                    or definition.unavailable_key in set(unavailable_metrics or [])
                ):
                    continue
                field_values = {
                    "sampling_interval_ms": interval,
                    "peak_rss_backend": backend,
                }
                configuration = tuple(
                    field_values[field] for field in definition.compatibility_fields
                )
                configurations[treatment].add(configuration)
        baseline_configs = configurations["baseline"]
        candidate_configs = configurations["candidate"]
        if (
            len(baseline_configs) > 1
            or len(candidate_configs) > 1
            or baseline_configs != candidate_configs
        ):
            reasons.append(
                "runtime-resource sampling interval or backend differs across treatments"
            )
        if "peak_rss_backend" in definition.compatibility_fields and (
            not baseline_configs
            or not candidate_configs
            or any(
                config[definition.compatibility_fields.index("peak_rss_backend")] is None
                for config in (*baseline_configs, *candidate_configs)
            )
        ):
            reasons.append("peak RSS backend identity is unavailable")
        if not definition.supports_confirmatory_paired_comparison:
            reasons.append("runtime-resource metric is not admitted for confirmatory comparison")
        return reasons

    def _compatibility_mismatches(
        self,
        snapshot: Snapshot,
        baseline: RunSet,
        candidate: RunSet,
    ) -> tuple[list[str], list[str]]:
        """Return ``(invalidating, exploratory)`` compatibility reasons.

        Invalidating reasons make the comparison INVALID. Exploratory reasons
        make it EXPLORATORY — missing evidence is a limitation, not a verdict.
        Inference replay runs use their persisted typed protocol for trace,
        provider, server, hardware, profiler, and semantic-oracle provenance.
        Incomplete protocol identity is exploratory; conflicting identity is
        invalidating.
        """
        invalidating: list[str] = []
        exploratory: list[str] = []
        run_ids = tuple(
            dict.fromkeys(
                member.run_id
                for run_set in (baseline, candidate)
                for member in run_set.members
                if member.included
            )
        )
        if not run_ids:
            return ["comparison has no included run-set members"], exploratory
        run_placeholders = ", ".join("?" for _ in run_ids)
        run_rows = snapshot.execute(
            "SELECT run_id, environment_id, source_state_id, "
            "workload_definition_id, validation_status, adapter, "
            "adapter_version, measurement_protocol_id, execution_identity_id, "
            "execution_identity_quality, execution_identity_json, "
            "inference_protocol_identity_json FROM current_runs "
            f"WHERE run_id IN ({run_placeholders})",
            run_ids,
        ).fetchall()
        by_id = {str(row[0]): row for row in run_rows}
        if set(by_id) != set(run_ids):
            invalidating.append("one or more run-set members are absent from the pinned corpus")
        baseline_runs = [
            by_id[member.run_id]
            for member in baseline.members
            if member.included and member.run_id in by_id
        ]
        candidate_runs = [
            by_id[member.run_id]
            for member in candidate.members
            if member.included and member.run_id in by_id
        ]
        all_runs = (*baseline_runs, *candidate_runs)
        inference_run_ids = {str(row[0]) for row in all_runs if row[11] is not None}
        all_inference = bool(run_ids) and set(run_ids).issubset(inference_run_ids)
        mixed_inference = bool(inference_run_ids) and not all_inference

        if mixed_inference:
            invalidating.append("cannot compare inference replay runs with non-inference runs")
        environments = {str(run[1]) for run in all_runs}
        invalidating.extend(self._environment_identity_mismatches(snapshot, environments))
        exec_mismatches = self._execution_identity_mismatches(all_runs)
        if all_inference:
            for mismatch in exec_mismatches:
                if "differs" in mismatch:
                    invalidating.append(mismatch)
                else:
                    exploratory.append(mismatch)
        else:
            invalidating.extend(exec_mismatches)
        self._source_state_checks(
            snapshot,
            baseline_runs,
            candidate_runs,
            all_inference,
            invalidating,
            exploratory,
        )
        definitions = {run[3] for run in all_runs}
        if len(definitions) > 1:
            invalidating.append("workload_definition_id differs across treatments")
        self._validation_checks(
            snapshot,
            all_runs,
            baseline,
            candidate,
            run_ids,
            run_placeholders,
            all_inference,
            invalidating,
            exploratory,
        )
        run_semantic_configurations = {
            (
                str(run[5]) if run[5] is not None else None,
                str(run[6]) if run[6] is not None else None,
                str(run[7]) if run[7] is not None else None,
            )
            for run in all_runs
        }
        if len(run_semantic_configurations) > 1:
            invalidating.append("adapter version or measurement protocol differs")
        artifact_configurations: dict[str, set[tuple[str, str | None, str | None, str]]] = {
            "baseline": set(),
            "candidate": set(),
        }
        artifact_rows = snapshot.execute(
            "SELECT run_id, kind, producer, producer_version, role "
            "FROM artifact_registrations "
            f"WHERE run_id IN ({run_placeholders})",
            run_ids,
        ).fetchall()
        baseline_ids = {member.run_id for member in baseline.members if member.included}
        candidate_ids = {member.run_id for member in candidate.members if member.included}
        for run_id, kind, producer, producer_version, role in artifact_rows:
            configuration = (
                str(kind),
                str(producer) if producer is not None else None,
                str(producer_version) if producer_version is not None else None,
                str(role),
            )
            if str(run_id) in baseline_ids:
                artifact_configurations["baseline"].add(configuration)
            if str(run_id) in candidate_ids:
                artifact_configurations["candidate"].add(configuration)
        if artifact_configurations["baseline"] != artifact_configurations["candidate"]:
            invalidating.append("profiler artifact configuration differs across treatments")
        if all_inference:
            invalidating.extend(self._inference_protocol_mismatches(baseline_runs, candidate_runs))
            exploratory.extend(self._inference_protocol_exploratory(baseline_runs, candidate_runs))
        return invalidating, exploratory

    @staticmethod
    def _source_state_checks(
        snapshot: Snapshot,
        baseline_runs: list[tuple[object, ...]],
        candidate_runs: list[tuple[object, ...]],
        all_inference: bool,
        invalidating: list[str],
        exploratory: list[str],
    ) -> None:
        if all_inference:
            # Inference source/server provenance is represented by the typed
            # protocol identity (trace/provider/managed-server digests). A
            # generic source checkout is neither required nor meaningful for
            # an existing local serving endpoint.
            return
        baseline_sources = {str(run[2]) if run[2] is not None else None for run in baseline_runs}
        candidate_sources = {str(run[2]) if run[2] is not None else None for run in candidate_runs}
        if len(baseline_sources) > 1 or len(candidate_sources) > 1:
            invalidating.append("source_state_id differs within a treatment")
        if None in baseline_sources or None in candidate_sources:
            msg = "one or more runs have no source_state_id"
            invalidating.append(msg)
        known_sources = {
            source for source in (*baseline_sources, *candidate_sources) if source is not None
        }
        if known_sources:
            source_placeholders = ", ".join("?" for _ in known_sources)
            qualities = snapshot.execute(
                "SELECT DISTINCT identity_quality FROM source_states "
                f"WHERE source_state_id IN ({source_placeholders})",
                tuple(sorted(known_sources)),
            ).fetchall()
            if not qualities or any(row[0] == "partial" for row in qualities):
                msg = "source identity is partial or unavailable"
                invalidating.append(msg)

    @staticmethod
    def _validation_checks(
        snapshot: Snapshot,
        all_runs: tuple[tuple[object, ...], ...],
        baseline: RunSet,
        candidate: RunSet,
        run_ids: tuple[str, ...],
        run_placeholders: str,
        all_inference: bool,
        invalidating: list[str],
        exploratory: list[str],
    ) -> None:
        if any(str(run[4]) != ValidationStatus.PASSED.value for run in all_runs):
            msg = "one or more runs lack passing validation"
            if all_inference:
                exploratory.append(msg)
            else:
                invalidating.append(msg)
        if all_inference:
            # Semantic validation identity and outcome are authoritative in
            # the inference protocol. Cross-treatment artifact registrations
            # are a contract for generic workloads and need not be duplicated.
            return
        validation_rows = snapshot.execute(
            "SELECT run_id, artifact_id FROM artifact_registrations "
            f"WHERE run_id IN ({run_placeholders}) "
            "AND role = 'validation_cross_treatment_equivalence'",
            run_ids,
        ).fetchall()
        validation_by_run: dict[str, set[str]] = {}
        for run_id, artifact_id in validation_rows:
            validation_by_run.setdefault(str(run_id), set()).add(str(artifact_id))
        baseline_validation = {
            artifact_id
            for member in baseline.members
            if member.included
            for artifact_id in validation_by_run.get(member.run_id, set())
        }
        candidate_validation = {
            artifact_id
            for member in candidate.members
            if member.included
            for artifact_id in validation_by_run.get(member.run_id, set())
        }
        if not baseline_validation and not candidate_validation:
            msg = "cross-treatment validation outputs are missing"
            if all_inference:
                exploratory.append(msg)
            else:
                invalidating.append(msg)
        elif baseline_validation != candidate_validation:
            invalidating.append("cross-treatment validation outputs differ")

    @staticmethod
    def _parse_protocol_identity(
        row: tuple[object, ...],
    ) -> tuple[InferenceProtocolIdentity | None, str | None]:
        """Parse a persisted protocol identity. Returns ``(identity, error)``.

        ``identity`` is the parsed model or ``None``. ``error`` is a non-``None``
        string when the persisted JSON is present but malformed, so the caller
        can report it as an invalidating compatibility reason instead of
        leaking a ``ValidationError``.
        """
        raw = row[11]
        if not isinstance(raw, str):
            return None, None
        try:
            return InferenceProtocolIdentity.model_validate_json(raw), None
        except (ValueError, TypeError) as exc:
            return None, f"malformed inference protocol identity JSON: {exc}"

    @staticmethod
    def _inference_protocol_mismatches(
        baseline_runs: list[tuple[object, ...]],
        candidate_runs: list[tuple[object, ...]],
    ) -> list[str]:
        reasons: list[str] = []
        # Check for malformed JSON within each treatment.
        for label, runs in (("baseline", baseline_runs), ("candidate", candidate_runs)):
            for row in runs:
                _identity, error = ComparisonService._parse_protocol_identity(row)
                if error is not None:
                    reasons.append(f"inference protocol malformed: {label} run {row[0]} — {error}")
        # Within-treatment: compare every included run against the treatment reference.
        for label, runs in (("baseline", baseline_runs), ("candidate", candidate_runs)):
            reasons.extend(ComparisonService._within_treatment_protocol_mismatches(label, runs))
        # Cross-treatment: compare treatment references.
        baseline_identity = ComparisonService._treatment_reference(baseline_runs)
        candidate_identity = ComparisonService._treatment_reference(candidate_runs)
        if baseline_identity is None or candidate_identity is None:
            return list(dict.fromkeys(reasons))
        result = compare_inference_protocols(baseline_identity, candidate_identity)
        reasons.extend(
            f"inference protocol mismatch: {m.field} "
            f"(baseline={m.baseline!r}, candidate={m.candidate!r})"
            for m in result.mismatches
        )
        return list(dict.fromkeys(reasons))

    @staticmethod
    def _inference_protocol_exploratory(
        baseline_runs: list[tuple[object, ...]],
        candidate_runs: list[tuple[object, ...]],
    ) -> list[str]:
        reasons: list[str] = []
        for label, runs in (("baseline", baseline_runs), ("candidate", candidate_runs)):
            reference = ComparisonService._treatment_reference(runs)
            if reference is None:
                continue
            for row in runs:
                identity, error = ComparisonService._parse_protocol_identity(row)
                if error is not None or identity is None or identity == reference:
                    continue
                comparison = compare_inference_protocols(reference, identity)
                reasons.extend(
                    f"inference protocol within-treatment exploratory: {label} "
                    f"run {row[0]} field {item.field} — {item.reason}"
                    for item in comparison.exploratory_reasons
                )
        baseline_identity = ComparisonService._treatment_reference(baseline_runs)
        candidate_identity = ComparisonService._treatment_reference(candidate_runs)
        if baseline_identity is None or candidate_identity is None:
            return reasons
        result = compare_inference_protocols(baseline_identity, candidate_identity)
        reasons.extend(
            f"inference protocol exploratory: {item.field} — {item.reason}"
            for item in result.exploratory_reasons
        )
        return list(dict.fromkeys(reasons))

    @staticmethod
    def _treatment_reference(
        runs: list[tuple[object, ...]],
    ) -> InferenceProtocolIdentity | None:
        """Return the first parseable protocol identity in a treatment."""
        for row in runs:
            identity, _error = ComparisonService._parse_protocol_identity(row)
            if identity is not None:
                return identity
        return None

    @staticmethod
    def _within_treatment_protocol_mismatches(
        label: str,
        runs: list[tuple[object, ...]],
    ) -> list[str]:
        """Compare every included run's protocol against the treatment reference.

        Reports exact dotted field mismatches so a repeated trial with a
        different model, schedule, or server configuration is caught.
        """
        reference = ComparisonService._treatment_reference(runs)
        if reference is None:
            return []
        reasons: list[str] = []
        for row in runs:
            identity, error = ComparisonService._parse_protocol_identity(row)
            if error is not None or identity is None:
                continue
            if identity == reference:
                continue
            result = compare_inference_protocols(reference, identity)
            for mismatch in result.mismatches:
                reasons.append(
                    f"inference protocol within-treatment mismatch: {label} "
                    f"run {row[0]} field {mismatch.field} "
                    f"(reference={mismatch.baseline!r}, run={mismatch.candidate!r})"
                )
        return reasons

    @staticmethod
    def _environment_identity_mismatches(
        snapshot: Snapshot,
        environments: set[str],
    ) -> tuple[str, ...]:
        mismatches: list[str] = []
        if len(environments) > 1:
            mismatches.append("environment_id differs across treatments")
        if environments:
            placeholders = ", ".join("?" for _ in environments)
            qualities = snapshot.execute(
                "SELECT DISTINCT identity_quality FROM environments "
                f"WHERE environment_id IN ({placeholders})",
                tuple(sorted(environments)),
            ).fetchall()
            if not qualities or any(row[0] != IdentityQuality.EXACT.value for row in qualities):
                mismatches.append("environment identity is partial or unavailable")
        return tuple(mismatches)

    @staticmethod
    def _execution_identity_mismatches(
        runs: tuple[tuple[object, ...], ...],
    ) -> tuple[str, ...]:
        applicable = [run for run in runs if run[9] not in {None, "not_applicable"}]
        identities = {str(run[8]) if run[8] is not None else None for run in applicable}
        qualities = {str(run[9]) if run[9] is not None else None for run in applicable}
        mismatches: list[str] = []
        if len(identities) > 1:
            mismatches.append("declared execution identity differs across treatments")
            details: dict[str, set[str]] = {}
            for run in applicable:
                if not isinstance(run[10], str):
                    continue
                try:
                    value = json.loads(run[10])
                except (TypeError, ValueError):
                    mismatches.append("declared execution identity JSON is malformed")
                    continue
                if not isinstance(value, dict) or not isinstance(value.get("inputs", []), list):
                    mismatches.append("declared execution identity JSON is malformed")
                    continue
                for item in value.get("inputs", []):
                    if not isinstance(item, dict):
                        mismatches.append("declared execution identity JSON is malformed")
                        continue
                    requested = item.get("requested")
                    if not isinstance(requested, str):
                        continue
                    resolved = (
                        item.get("loaded_path")
                        or item.get("resolved_path")
                        or item.get("configured_path")
                        or item.get("status")
                        or "unknown"
                    )
                    digest = item.get("content_digest")
                    label = f"{resolved} ({digest})" if digest else str(resolved)
                    details.setdefault(requested, set()).add(label)
            for requested, values in sorted(details.items()):
                if len(values) > 1:
                    mismatches.append(
                        f"execution identity input {requested!r} differs: "
                        + " vs ".join(sorted(values))
                    )
                    break
        if qualities - {"exact"}:
            mismatches.append("declared execution identity is partial or unavailable")
        return tuple(mismatches)

    def _profile_changes(
        self,
        snapshot: Snapshot,
        baseline: RunSet,
        candidate: RunSet,
        *,
        polarity: MetricPolarity,
    ) -> tuple[ProfileChange, ...]:
        baseline_values = self._frame_aggregates(snapshot, baseline)
        candidate_values = self._frame_aggregates(snapshot, candidate)
        keys = baseline_values.keys() | candidate_values.keys()
        changes: list[ProfileChange] = []
        for key in keys:
            baseline_entry = baseline_values.get(key)
            candidate_entry = candidate_values.get(key)
            metadata = (
                baseline_entry[1]
                if baseline_entry is not None
                else candidate_entry[1]
                if candidate_entry is not None
                else (None, None, None)
            )
            baseline_value = baseline_entry[0] if baseline_entry is not None else 0.0
            candidate_value = candidate_entry[0] if candidate_entry is not None else 0.0
            absolute = candidate_value - baseline_value
            relative = absolute / baseline_value if baseline_value != 0 else None
            if math.isclose(absolute, 0.0):
                direction = ProfileChangeDirection.UNCHANGED
            elif polarity == "neutral":
                direction = ProfileChangeDirection.CHANGED
            elif (absolute > 0) == (polarity == "lower_is_better"):
                direction = ProfileChangeDirection.REGRESSED
            else:
                direction = ProfileChangeDirection.IMPROVED
            changes.append(
                ProfileChange(
                    frame_id=key[0],
                    function=metadata[0],
                    file=metadata[1],
                    line=metadata[2],
                    metric=key[1],
                    unit=key[2],
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    absolute_change=absolute,
                    relative_change=relative,
                    direction=direction,
                )
            )
        changes.sort(
            key=lambda item: (
                -abs(item.relative_change or 0.0),
                -abs(item.absolute_change),
                item.frame_id,
            )
        )
        return tuple(changes[: self.workspace.config.analysis.default_row_limit])

    @staticmethod
    def _frame_aggregates(
        snapshot: Snapshot,
        run_set: RunSet,
    ) -> dict[
        tuple[str, str, str],
        tuple[float, tuple[str | None, str | None, int | None]],
    ]:
        run_ids = tuple(member.run_id for member in run_set.members if member.included)
        if not run_ids:
            return {}
        placeholders = ", ".join("?" for _ in run_ids)
        rows = snapshot.execute(
            "WITH per_run AS ("
            "SELECT fm.run_id, fm.frame_id, fm.metric, fm.unit, "
            "sum(coalesce(fm.inclusive_value, fm.self_value, 0)) AS value "
            "FROM frame_measurements fm "
            f"WHERE fm.run_id IN ({placeholders}) "
            "GROUP BY fm.run_id, fm.frame_id, fm.metric, fm.unit"
            ") SELECT p.frame_id, p.metric, p.unit, avg(p.value), "
            "any_value(f.function), any_value(f.file), any_value(f.line) "
            "FROM per_run p LEFT JOIN frames f ON f.frame_id = p.frame_id "
            "GROUP BY p.frame_id, p.metric, p.unit",
            run_ids,
        ).fetchall()
        return {
            (str(row[0]), str(row[1]), str(row[2])): (
                float(row[3]),
                (
                    str(row[4]) if row[4] is not None else None,
                    str(row[5]) if row[5] is not None else None,
                    int(row[6]) if row[6] is not None else None,
                ),
            )
            for row in rows
        }

    def _comparison_row(self, value: Comparison) -> dict[str, object]:
        row = value.model_dump(mode="python")
        baseline_value_int, baseline_value_float = numeric_value_to_columns(value.baseline_value)
        candidate_value_int, candidate_value_float = numeric_value_to_columns(value.candidate_value)
        absolute_change_int, absolute_change_float = numeric_value_to_columns(value.absolute_change)
        row.update(
            {
                "polarity": value.polarity,
                "metric_source": value.metric_source.value,
                "value_domain": value.value_domain.value,
                "zero_policy": value.zero_policy.value,
                "decision": value.decision.value,
                "validity": value.validity.value,
                "mismatches": list(value.mismatches),
                "multiplicity_json": (
                    json.dumps(
                        value.multiplicity,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    if value.multiplicity is not None
                    else None
                ),
                "baseline_value_int": baseline_value_int,
                "baseline_value_float": baseline_value_float,
                "candidate_value_int": candidate_value_int,
                "candidate_value_float": candidate_value_float,
                "absolute_change_int": absolute_change_int,
                "absolute_change_float": absolute_change_float,
            }
        )
        row.pop("multiplicity")
        row.pop("baseline_value")
        row.pop("candidate_value")
        row.pop("absolute_change")
        row.pop("schema_version")
        return row
