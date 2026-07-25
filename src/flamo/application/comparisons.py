from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from flamo.analysis import compare_paired_samples
from flamo.application.analysis_rows import analysis_row
from flamo.catalog import Catalog
from flamo.domain import (
    AnalysisRecord,
    Comparison,
    ComparisonValidity,
    DomainError,
    ErrorCode,
    EvidenceReference,
    Experiment,
    RunSet,
    RunSetMember,
    ValidationStatus,
    digest_model,
    new_id,
)
from flamo.evidence import GenerationPublisher
from flamo.storage import JsonRecordStore, RunStore, Workspace


class FreezeRunSetMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    trial_id: str | None = None
    included: bool = True
    reason: str | None = None

    @model_validator(mode="after")
    def excluded_members_have_reasons(self) -> FreezeRunSetMember:
        if not self.included and self.reason is None:
            raise ValueError("excluded run-set members require a reason")
        return self


class FreezeRunSetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_ids: tuple[str, ...] = ()
    members: tuple[FreezeRunSetMember, ...] = ()
    selection: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def choose_one_membership_form(self) -> FreezeRunSetRequest:
        if bool(self.run_ids) == bool(self.members):
            raise ValueError("provide exactly one of run_ids or members")
        return self


class CompareRunSetsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_run_set_id: str
    candidate_run_set_id: str
    experiment_id: str | None = None
    metric: str
    unit: str
    polarity: Literal["lower_is_better", "higher_is_better", "neutral"]
    practical_threshold: float = Field(ge=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    random_seed: int = Field(default=0, ge=0)


class ComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    comparison: Comparison
    baseline_run_set: RunSet
    candidate_run_set: RunSet
    corpus_commit_id: str
    analysis: AnalysisRecord | None = None
    evidence: tuple[EvidenceReference, ...] = ()
    materialized_commit_id: str | None = None


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
            publisher="flamo.run_sets",
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
        if request.experiment_id is not None:
            JsonRecordStore(
                self.workspace,
                kind="experiments",
                model=Experiment,
                id_field="experiment_id",
            ).read(request.experiment_id)
        baseline_set = self.run_sets.read(request.baseline_run_set_id)
        candidate_set = self.run_sets.read(request.candidate_run_set_id)
        corpus_commit_id = self.workspace.corpus.read_head().commit_id
        mismatches = self._compatibility_mismatches(
            baseline_set,
            candidate_set,
            corpus_commit_id=corpus_commit_id,
        )
        baseline = self._samples(baseline_set, request.metric, request.unit)
        candidate = self._samples(candidate_set, request.metric, request.unit)
        if baseline.eligible == 0:
            mismatches.append("baseline run set has no eligible measurements")
        if candidate.eligible == 0:
            mismatches.append("candidate run set has no eligible measurements")
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
        if mismatches:
            comparison = comparison.model_copy(
                update={
                    "validity": ComparisonValidity.INVALID,
                    "mismatches": tuple(mismatches),
                }
            )
        return ComparisonResult(
            comparison=comparison,
            baseline_run_set=baseline_set,
            candidate_run_set=candidate_set,
            corpus_commit_id=corpus_commit_id,
        )

    def record(self, request: CompareRunSetsRequest) -> ComparisonResult:
        started = datetime.now(UTC)
        result = self.compare(request)
        completed = datetime.now(UTC)
        input_run_ids = tuple(
            member.run_id
            for run_set in (result.baseline_run_set, result.candidate_run_set)
            for member in run_set.members
        )
        result_digest = digest_model(result.comparison.model_dump(mode="json"))
        parameters = request.model_dump(mode="json")
        analysis = AnalysisRecord(
            analysis_id=new_id(),
            recipe="compare_run_sets",
            recipe_version="1",
            parameters=parameters,
            parameters_digest=digest_model(parameters),
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
            },
            limitations=(
                ("Compatibility mismatches invalidate proof.",)
                if result.comparison.mismatches
                else ()
            ),
            started_at=started,
            completed_at=completed,
        )
        references = (
            EvidenceReference(
                owner_type="analysis",
                owner_id=analysis.analysis_id,
                ref_type="run_set",
                ref_id=result.baseline_run_set.run_set_id,
                relation="context",
            ),
            EvidenceReference(
                owner_type="analysis",
                owner_id=analysis.analysis_id,
                ref_type="run_set",
                ref_id=result.candidate_run_set.run_set_id,
                relation="context",
            ),
        )
        published = self.publisher.publish_rows(
            {
                "comparisons": [self._comparison_row(result.comparison)],
                "analyses": [analysis_row(analysis)],
                "evidence_refs": [reference.model_dump(mode="python") for reference in references],
            },
            publisher="flamo.comparisons",
            publisher_version="1",
            input_run_ids=input_run_ids,
        )
        return result.model_copy(
            update={
                "analysis": analysis,
                "evidence": references,
                "materialized_commit_id": published.commit.commit_id,
            }
        )

    def _samples(
        self,
        run_set: RunSet,
        metric: str,
        unit: str,
    ) -> _SampleSet:
        values: dict[str, float] = {}
        attempted = len(run_set.members)
        failed = 0
        excluded = 0
        if len(run_set.members) > 1 and any(
            member.trial_id is None for member in run_set.members
        ):
            raise DomainError(
                ErrorCode.COMPARISON_INVALID,
                "Multi-run paired comparisons require explicit trial identities.",
            )
        with Catalog(self.workspace).open_snapshot(run_set.corpus_commit_id) as snapshot:
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

    def _compatibility_mismatches(
        self,
        baseline: RunSet,
        candidate: RunSet,
        *,
        corpus_commit_id: str,
    ) -> list[str]:
        mismatches: list[str] = []
        baseline_runs = [
            RunStore(self.workspace).read(member.run_id) for member in baseline.members
        ]
        candidate_runs = [
            RunStore(self.workspace).read(member.run_id) for member in candidate.members
        ]
        environments = {run.environment_id for run in (*baseline_runs, *candidate_runs)}
        if len(environments) > 1:
            mismatches.append("environment_id differs across treatments")
        baseline_sources = {run.source_state_id for run in baseline_runs}
        candidate_sources = {run.source_state_id for run in candidate_runs}
        if len(baseline_sources) > 1 or len(candidate_sources) > 1:
            mismatches.append("source_state_id differs within a treatment")
        if None in baseline_sources or None in candidate_sources:
            mismatches.append("one or more runs have no source_state_id")
        known_sources = {
            source for source in (*baseline_sources, *candidate_sources) if source is not None
        }
        if known_sources:
            placeholders = ", ".join("?" for _ in known_sources)
            with Catalog(self.workspace).open_snapshot(corpus_commit_id) as snapshot:
                qualities = snapshot.execute(
                    "SELECT DISTINCT identity_quality FROM source_states "
                    f"WHERE source_state_id IN ({placeholders})",
                    tuple(sorted(known_sources)),
                ).fetchall()
            if not qualities or any(row[0] == "partial" for row in qualities):
                mismatches.append("source identity is partial or unavailable")
        definitions = {run.workload_definition_id for run in (*baseline_runs, *candidate_runs)}
        if len(definitions) > 1:
            mismatches.append("workload_definition_id differs across treatments")
        if any(
            run.validation_status is not ValidationStatus.PASSED
            for run in (*baseline_runs, *candidate_runs)
        ):
            mismatches.append("one or more runs lack passing validation")
        baseline_validation = {
            artifact.artifact_id
            for run in baseline_runs
            for artifact in run.artifacts
            if artifact.role == "validation_cross_treatment_equivalence"
        }
        candidate_validation = {
            artifact.artifact_id
            for run in candidate_runs
            for artifact in run.artifacts
            if artifact.role == "validation_cross_treatment_equivalence"
        }
        if (
            not baseline_validation
            or not candidate_validation
            or baseline_validation != candidate_validation
        ):
            mismatches.append("cross-treatment validation outputs are missing or differ")
        return mismatches

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
