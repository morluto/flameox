from __future__ import annotations

import json
import math
import statistics
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from flameox.analysis import compare_paired_samples
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
from flameox.catalog import Catalog, Snapshot
from flameox.domain import (
    AnalysisRecord,
    Comparison,
    ComparisonValidity,
    DomainError,
    ErrorCode,
    EvidenceReference,
    Experiment,
    IdentityQuality,
    RunSet,
    RunSetMember,
    ValidationStatus,
    digest_model,
)
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import JsonRecordStore, RunStore, Workspace

_RUNTIME_RESOURCE_COLUMNS = {
    "runtime_resource.peak_rss_bytes": "peak_rss_bytes",
    "runtime_resource.minimum_free_bytes": "minimum_free_bytes",
    "runtime_resource.staging_growth_bytes": "staging_growth_bytes",
}


class FreezeRunSetMember(ContractModel):
    run_id: str
    trial_id: str | None = None
    included: bool = True
    reason: str | None = None

    @model_validator(mode="after")
    def excluded_members_have_reasons(self) -> FreezeRunSetMember:
        if not self.included and self.reason is None:
            raise ValueError("excluded run-set members require a reason")
        return self


class FreezeRunSetRequest(ContractModel):
    run_ids: tuple[str, ...] = ()
    members: tuple[FreezeRunSetMember, ...] = ()
    selection: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def choose_one_membership_form(self) -> FreezeRunSetRequest:
        if bool(self.run_ids) == bool(self.members):
            raise ValueError("provide exactly one of run_ids or members")
        return self


class CompareRunSetsRequest(ContractModel):
    baseline_run_set_id: str
    candidate_run_set_id: str
    experiment_id: str | None = None
    metric: str
    unit: str
    metric_source: Literal["measurement", "runtime_resource"] = "measurement"
    polarity: Literal["lower_is_better", "higher_is_better", "neutral"]
    practical_threshold: float = Field(ge=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    random_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_metric_source(self) -> CompareRunSetsRequest:
        if self.metric_source == "runtime_resource":
            if self.metric not in _RUNTIME_RESOURCE_COLUMNS:
                raise ValueError("runtime-resource comparisons require a catalog metric")
            if self.unit != "bytes":
                raise ValueError("runtime-resource comparisons require unit='bytes'")
        return self


class ComparisonResult(ContractModel):
    schema_version: int = 1
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
    direction: Literal["regressed", "improved", "changed", "unchanged"]


@dataclass(frozen=True, slots=True)
class _SampleSet:
    values: dict[str, float]
    attempted: int
    eligible: int
    failed: int
    excluded: int


class RunSetService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.store = JsonRecordStore(
            workspace,
            kind="run_sets",
            model=RunSet,
            id_field="run_set_id",
        )
        self.publisher = GenerationPublisher(workspace)

    def freeze(self, request: FreezeRunSetRequest) -> RunSet:
        members: list[RunSetMember] = []
        head = self.workspace.corpus.read_head()
        requested = request.members or tuple(
            FreezeRunSetMember(run_id=run_id) for run_id in request.run_ids
        )
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            for order, item in enumerate(requested):
                RunStore(self.workspace).read(item.run_id)
                if item.trial_id is not None:
                    experiment_id = request.selection.get("experiment_id")
                    if isinstance(experiment_id, str):
                        trial_rows = snapshot.execute(
                            "SELECT DISTINCT run_id FROM trials "
                            "WHERE trial_id = ? AND experiment_id = ?",
                            (item.trial_id, experiment_id),
                        ).fetchall()
                    else:
                        trial_rows = snapshot.execute(
                            "SELECT DISTINCT run_id FROM trials WHERE trial_id = ?",
                            (item.trial_id,),
                        ).fetchall()
                    if len(trial_rows) != 1 or str(trial_rows[0][0]) != item.run_id:
                        raise DomainError(
                            ErrorCode.COMPARISON_INVALID,
                            "Run-set trial identity does not match its run.",
                            details={
                                "run_id": item.run_id,
                                "trial_id": item.trial_id,
                            },
                        )
                members.append(
                    RunSetMember(
                        run_id=item.run_id,
                        trial_id=item.trial_id,
                        included=item.included,
                        reason=item.reason,
                        order=order,
                    )
                )
        membership = [member.model_dump(mode="json") for member in members]
        membership_digest = digest_model(membership)
        run_set_id = digest_model(
            {
                "corpus_commit_id": head.commit_id,
                "selection": request.selection,
                "members": membership,
            }
        )
        run_set = RunSet(
            run_set_id=run_set_id,
            corpus_commit_id=head.commit_id,
            selection=request.selection,
            members=tuple(members),
            membership_digest=membership_digest,
        )
        self.store.create(run_set)
        self.publisher.publish_rows(
            {
                "run_sets": [
                    {
                        "run_set_id": run_set.run_set_id,
                        "corpus_commit_id": run_set.corpus_commit_id,
                        "created_at": run_set.created_at,
                        "selection_json": json.dumps(
                            run_set.selection,
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        "members_json": json.dumps(
                            membership,
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        "membership_digest": run_set.membership_digest,
                    }
                ]
            },
            publisher="flameox.run_sets",
            publisher_version="1",
            input_run_ids=tuple(member.run_id for member in members),
        )
        return run_set


class ComparisonService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.run_sets = RunSetService(workspace).store
        self.publisher = GenerationPublisher(workspace)

    def compare(self, request: CompareRunSetsRequest) -> ComparisonResult:
        corpus_commit_id = self.workspace.corpus.read_head().commit_id
        with Catalog(self.workspace).open_snapshot(corpus_commit_id) as snapshot:
            return self._compare_at_snapshot(request, snapshot)

    async def compare_async(
        self,
        request: CompareRunSetsRequest,
        *,
        progress: Callable[[float, float, str], Awaitable[None]] | None = None,
    ) -> ComparisonResult:
        corpus_commit_id = self.workspace.corpus.read_head().commit_id
        if progress is not None:
            await progress(0, 2, "Comparison snapshot pinned")
        result = await Catalog(self.workspace).run_interruptible(
            lambda snapshot: self._compare_at_snapshot(request, snapshot),
            commit_id=corpus_commit_id,
            query_name="compare_run_sets",
        )
        if progress is not None:
            await progress(1, 2, "Comparison query complete")
            await progress(2, 2, "Comparison result ready")
        return result

    def _compare_at_snapshot(
        self,
        request: CompareRunSetsRequest,
        snapshot: Snapshot,
    ) -> ComparisonResult:
        if request.experiment_id is not None:
            JsonRecordStore(
                self.workspace,
                kind="experiments",
                model=Experiment,
                id_field="experiment_id",
            ).read(request.experiment_id)
        baseline_set = self.run_sets.read(request.baseline_run_set_id)
        candidate_set = self.run_sets.read(request.candidate_run_set_id)
        corpus_commit_id = snapshot.commit.commit_id
        invalidating, exploratory = self._compatibility_mismatches(
            snapshot,
            baseline_set,
            candidate_set,
        )
        baseline = self._samples(snapshot, baseline_set, request)
        candidate = self._samples(snapshot, candidate_set, request)
        if request.metric_source == "runtime_resource":
            invalidating.extend(
                self._runtime_resource_compatibility_mismatches(
                    snapshot, baseline_set, candidate_set, request.metric
                )
            )
            if baseline.eligible != baseline.attempted - baseline.failed - baseline.excluded:
                exploratory.append(
                    "baseline runtime-resource evidence is unavailable or incomplete"
                )
            if candidate.eligible != candidate.attempted - candidate.failed - candidate.excluded:
                exploratory.append(
                    "candidate runtime-resource evidence is unavailable or incomplete"
                )
        profile_changes = self._profile_changes(
            snapshot,
            baseline_set,
            candidate_set,
            polarity=request.polarity,
        )
        if baseline.eligible == 0:
            invalidating.append("baseline run set has no eligible measurements")
        if candidate.eligible == 0:
            invalidating.append("candidate run set has no eligible measurements")
        comparison_id = digest_model(
            {
                "recipe": "compare_run_sets.v1",
                "request": request.model_dump(mode="json"),
                "baseline_membership": baseline_set.membership_digest,
                "candidate_membership": candidate_set.membership_digest,
                "corpus_commit_id": corpus_commit_id,
            }
        )
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
        )
        comparison = comparison.model_copy(
            update={
                "baseline_attempted_n": baseline.attempted,
                "baseline_eligible_n": baseline.eligible,
                "baseline_failed_n": baseline.failed,
                "baseline_excluded_n": baseline.excluded,
                "candidate_attempted_n": candidate.attempted,
                "candidate_eligible_n": candidate.eligible,
                "candidate_failed_n": candidate.failed,
                "candidate_excluded_n": candidate.excluded,
            }
        )
        all_reasons = (*invalidating, *exploratory)
        if invalidating:
            comparison = comparison.model_copy(
                update={
                    "validity": ComparisonValidity.INVALID,
                    "mismatches": tuple(all_reasons),
                }
            )
        elif exploratory:
            comparison = comparison.model_copy(
                update={
                    "validity": ComparisonValidity.EXPLORATORY,
                    "mismatches": tuple(all_reasons),
                }
            )
        return ComparisonResult(
            comparison=comparison,
            baseline_run_set=baseline_set,
            candidate_run_set=candidate_set,
            corpus_commit_id=corpus_commit_id,
            profile_changes=profile_changes,
        )

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
        if progress is not None:
            await progress(0, 3, "Comparison snapshot pinned")
        result = await Catalog(self.workspace).run_interruptible(
            lambda snapshot: self._compare_at_snapshot(request, snapshot),
            query_name="record_comparison",
        )
        if progress is not None:
            await progress(1, 3, "Comparison query complete")
            await progress(2, 3, "Publishing comparison provenance")
        recorded = await run_atomic_thread(
            lambda: self._record_result(request, result=result, started=started)
        )
        if progress is not None:
            await progress(3, 3, "Comparison publication complete")
        return recorded

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
                    "candidate_attempted": result.comparison.candidate_attempted_n,
                    "candidate_eligible": result.comparison.candidate_eligible_n,
                    "candidate_failed": result.comparison.candidate_failed_n,
                    "candidate_excluded": result.comparison.candidate_excluded_n,
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
        return result.model_copy(
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
        if request.metric_source == "runtime_resource":
            return self._runtime_resource_samples(snapshot, run_set, request.metric)
        return self._measurement_samples(snapshot, run_set, request.metric, request.unit)

    def _measurement_samples(
        self,
        snapshot: Snapshot,
        run_set: RunSet,
        metric: str,
        unit: str,
    ) -> _SampleSet:
        values: dict[str, float] = {}
        attempted = len(run_set.members)
        failed = 0
        excluded = 0
        if len(run_set.members) > 1 and any(member.trial_id is None for member in run_set.members):
            raise DomainError(
                ErrorCode.COMPARISON_INVALID,
                "Multi-run paired comparisons require explicit trial identities.",
            )
        for member in run_set.members:
            if not member.included:
                excluded += 1
                continue
            block_key: str | None = None
            if member.trial_id is not None:
                trial_rows = snapshot.execute(
                    "SELECT DISTINCT block_id, outcome FROM trials WHERE trial_id = ?",
                    (member.trial_id,),
                ).fetchall()
                if len(trial_rows) != 1:
                    raise DomainError(
                        ErrorCode.COMPARISON_INVALID,
                        "Run-set trial evidence is missing or ambiguous.",
                        details={"trial_id": member.trial_id},
                    )
                block_id, outcome = trial_rows[0]
                if str(outcome) != "succeeded":
                    failed += 1
                    continue
                if block_id is None:
                    raise DomainError(
                        ErrorCode.COMPARISON_INVALID,
                        "Paired experiment trials require block identities.",
                        details={"trial_id": member.trial_id},
                    )
                block_key = str(block_id)
            rows = snapshot.execute(
                "SELECT value_int, value_float, block_id, worker_id, "
                "worker_run_index, value_index "
                "FROM measurements WHERE run_id = ? AND name = ? AND unit = ? "
                "AND is_warmup = false ORDER BY measurement_id",
                (member.run_id, metric, unit),
            ).fetchall()
            member_values: list[float] = []
            for index, row in enumerate(rows):
                value = row[0] if row[0] is not None else row[1]
                if value is None:
                    continue
                member_values.append(float(value))
                if block_key is not None:
                    continue
                unit_key = (
                    str(row[2])
                    if row[2] is not None
                    else ":".join(
                        (
                            str(row[3] or member.order),
                            str(row[4] if row[4] is not None else 0),
                            str(row[5] if row[5] is not None else index),
                        )
                    )
                )
                if unit_key in values:
                    raise DomainError(
                        ErrorCode.COMPARISON_INVALID,
                        "Run set contains duplicate measurement keys without block identities.",
                        details={"run_id": member.run_id, "key": unit_key},
                    )
                values[unit_key] = float(value)
            if block_key is not None and member_values:
                if block_key in values:
                    raise DomainError(
                        ErrorCode.COMPARISON_INVALID,
                        "Run set contains more than one included trial for a block.",
                        details={"block_id": block_key},
                    )
                values[block_key] = statistics.median(member_values)
        return _SampleSet(
            values=values,
            attempted=attempted,
            eligible=len(values),
            failed=failed,
            excluded=excluded,
        )

    def _runtime_resource_samples(
        self,
        snapshot: Snapshot,
        run_set: RunSet,
        metric: str,
    ) -> _SampleSet:
        column = _RUNTIME_RESOURCE_COLUMNS[metric]
        values: dict[str, float] = {}
        attempted = len(run_set.members)
        failed = 0
        excluded = 0
        if len(run_set.members) > 1 and any(member.trial_id is None for member in run_set.members):
            raise DomainError(
                ErrorCode.COMPARISON_INVALID,
                "Multi-run paired comparisons require explicit trial identities.",
            )
        for member in run_set.members:
            if not member.included:
                excluded += 1
                continue
            block_key = str(member.order)
            if member.trial_id is not None:
                trial_rows = snapshot.execute(
                    "SELECT DISTINCT block_id, outcome FROM trials WHERE trial_id = ?",
                    (member.trial_id,),
                ).fetchall()
                if len(trial_rows) != 1:
                    raise DomainError(
                        ErrorCode.COMPARISON_INVALID,
                        "Run-set trial evidence is missing or ambiguous.",
                        details={"trial_id": member.trial_id},
                    )
                block_id, outcome = trial_rows[0]
                if str(outcome) != "succeeded":
                    failed += 1
                    continue
                if block_id is None:
                    raise DomainError(
                        ErrorCode.COMPARISON_INVALID,
                        "Paired experiment trials require block identities.",
                        details={"trial_id": member.trial_id},
                    )
                block_key = str(block_id)
            rows = snapshot.execute(
                f"SELECT {column}, unavailable_metrics FROM runtime_resource_summaries "
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
                continue
            value, unavailable_metrics = rows[0]
            if value is None or metric.removeprefix("runtime_resource.") in set(
                unavailable_metrics or []
            ):
                continue
            if block_key in values:
                raise DomainError(
                    ErrorCode.COMPARISON_INVALID,
                    "Run set contains more than one included trial for a block.",
                    details={"block_id": block_key},
                )
            values[block_key] = float(value)
        return _SampleSet(
            values=values,
            attempted=attempted,
            eligible=len(values),
            failed=failed,
            excluded=excluded,
        )

    def _runtime_resource_compatibility_mismatches(
        self,
        snapshot: Snapshot,
        baseline: RunSet,
        candidate: RunSet,
        metric: str,
    ) -> list[str]:
        reasons: list[str] = []
        configurations: dict[str, set[tuple[object, ...]]] = {"baseline": set(), "candidate": set()}
        for treatment, run_set in (("baseline", baseline), ("candidate", candidate)):
            for member in run_set.members:
                if not member.included:
                    continue
                if member.trial_id is not None:
                    trial_rows = snapshot.execute(
                        "SELECT DISTINCT outcome FROM trials WHERE trial_id = ?",
                        (member.trial_id,),
                    ).fetchall()
                    if len(trial_rows) != 1 or str(trial_rows[0][0]) != "succeeded":
                        continue
                rows = snapshot.execute(
                    "SELECT sampling_interval_ms, peak_rss_backend "
                    "FROM runtime_resource_summaries WHERE run_id = ?",
                    (member.run_id,),
                ).fetchall()
                if len(rows) != 1:
                    continue
                interval, backend = rows[0]
                configuration = (
                    (interval, backend) if metric.endswith("peak_rss_bytes") else (interval,)
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
        if metric.endswith("peak_rss_bytes") and (
            not baseline_configs
            or not candidate_configs
            or any(config[1] is None for config in (*baseline_configs, *candidate_configs))
        ):
            reasons.append("peak RSS backend identity is unavailable")
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
            member.run_id for run_set in (baseline, candidate) for member in run_set.members
        )
        run_placeholders = ", ".join("?" for _ in run_ids)
        run_rows = snapshot.execute(
            "SELECT run_id, environment_id, source_state_id, "
            "workload_definition_id, validation_status, collector, "
            "collector_version, measurement_protocol_id, execution_identity_id, "
            "execution_identity_quality, execution_identity_json, "
            "inference_protocol_identity_json FROM ("
            "SELECT *, row_number() OVER (PARTITION BY run_id "
            "ORDER BY published_at DESC) AS revision_order FROM runs"
            f") WHERE revision_order = 1 AND run_id IN ({run_placeholders})",
            run_ids,
        ).fetchall()
        by_id = {str(row[0]): row for row in run_rows}
        if set(by_id) != set(run_ids):
            invalidating.append("one or more run-set members are absent from the pinned corpus")
        baseline_runs = [
            by_id[member.run_id] for member in baseline.members if member.run_id in by_id
        ]
        candidate_runs = [
            by_id[member.run_id] for member in candidate.members if member.run_id in by_id
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
        collector_configurations = {
            (
                str(run[5]) if run[5] is not None else None,
                str(run[6]) if run[6] is not None else None,
                str(run[7]) if run[7] is not None else None,
            )
            for run in all_runs
        }
        if len(collector_configurations) > 1:
            invalidating.append("collector version or measurement protocol differs")
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
        baseline_ids = {member.run_id for member in baseline.members}
        candidate_ids = {member.run_id for member in candidate.members}
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
            for artifact_id in validation_by_run.get(member.run_id, set())
        }
        candidate_validation = {
            artifact_id
            for member in candidate.members
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
        polarity: Literal["lower_is_better", "higher_is_better", "neutral"],
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
                direction: Literal["regressed", "improved", "changed", "unchanged"] = "unchanged"
            elif polarity == "neutral":
                direction = "changed"
            elif (absolute > 0) == (polarity == "lower_is_better"):
                direction = "regressed"
            else:
                direction = "improved"
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
        row.update(
            {
                "polarity": value.polarity,
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
            }
        )
        row.pop("multiplicity")
        row.pop("schema_version")
        return row
