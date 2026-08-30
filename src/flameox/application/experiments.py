from __future__ import annotations

import asyncio
import json
import random
import secrets
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from functools import partial
from itertools import product
from typing import Literal, cast

from pydantic import Field, JsonValue, TypeAdapter
from scipy.stats import beta

from flameox.action_graph import ManualAction, manual_action
from flameox.adapters.benchmark_samples import BenchmarkSamplesExtractor
from flameox.adapters.pyperf import PyPerfExtractor
from flameox.adapters.pytest import PytestExtractor
from flameox.adapters.python_startup import PythonStartupExtractor
from flameox.application.async_work import run_atomic_thread
from flameox.application.capture import CaptureService
from flameox.application.comparisons import (
    ComparisonResult,
    ComparisonService,
    ExcludedFreezeRunSetMember,
    FreezeRunMembersRequest,
    FreezeRunSetMember,
    IncludedFreezeRunSetMember,
    RunSetService,
    parse_compare_run_sets_request,
)
from flameox.application.evidence_lookup import EvidenceLookupService, EvidenceSession
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.progress import ProgressReporter
from flameox.application.workloads import (
    ExperimentConfig,
    Scalar,
    WorkloadService,
    _OutcomeExperimentConfig,
    scalar_contains,
    scalar_equal,
    scalar_identity,
    scalar_identity_set,
    scalar_subset,
)
from flameox.catalog import Catalog, Snapshot
from flameox.domain import (
    ArtifactKind,
    ComparisonDecision,
    ComparisonValidity,
    ConfidenceInterval,
    CursorNamespace,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    Experiment,
    ExperimentOutcomeDisposition,
    ExperimentOutcomeGoal,
    ExperimentOutcomeMethod,
    ExperimentRole,
    Hypothesis,
    Investigation,
    MetricSource,
    OracleStrength,
    RunManifest,
    RunSet,
    Trial,
    TrialFailureClass,
    TrialOutcome,
    ValidationStatus,
    Variant,
    VariantIdentityQuality,
    canonical_json,
    digest_model,
    new_id,
)
from flameox.domain.models import parse_trial, utc_now
from flameox.domain.scalars import NumericValue, parse_numeric_value
from flameox.evidence import (
    GenerationPublisher,
    PublishedGeneration,
    numeric_value_from_columns,
    numeric_value_to_columns,
)
from flameox.models import ContractModel
from flameox.pagination import CursorPageContract
from flameox.storage import AuthorizedPlanStore, ControlRecordStore, RunStore, Workspace


def _extract_adapter_measurements(workspace: Workspace, adapter: str, run_id: str) -> None:
    if adapter == "pyperf":
        PyPerfExtractor(workspace).extract(run_id)
    elif adapter in {"benchmark-samples", "torch.benchmark"}:
        BenchmarkSamplesExtractor(workspace).extract(run_id)
    elif adapter == "python-startup":
        PythonStartupExtractor(workspace).extract(run_id)
    elif adapter == "pytest":
        PytestExtractor(workspace).extract(run_id)


def _has_extractable_artifact(run: RunManifest, adapter: str) -> bool:
    if adapter == "python-startup":
        return any(
            item.kind is ArtifactKind.BENCHMARK_SAMPLES and item.role == "startup_wall"
            for item in run.artifacts
        ) and any(
            item.kind is ArtifactKind.PYTHON_STARTUP and item.role == "import_trace"
            for item in run.artifacts
        )
    expected = {
        "benchmark-samples": ArtifactKind.BENCHMARK_SAMPLES,
        "torch.benchmark": ArtifactKind.BENCHMARK_SAMPLES,
        "pyperf": ArtifactKind.BENCHMARK_SAMPLES,
        "pytest": ArtifactKind.TEST_EXECUTION,
    }.get(adapter)
    return expected is not None and any(item.kind is expected for item in run.artifacts)


def _freeze_trial_member(trial: Trial) -> FreezeRunSetMember:
    if trial.run_id is None:
        raise DomainError(ErrorCode.WORKSPACE_INVALID, "A run-set trial has no run identity.")
    if trial.outcome is TrialOutcome.SUCCEEDED:
        return IncludedFreezeRunSetMember(run_id=trial.run_id, trial_id=trial.trial_id)
    if trial.exclusion_reason is None:
        raise DomainError(
            ErrorCode.WORKSPACE_INVALID,
            "An excluded run-set trial has no exclusion reason.",
            details={"trial_id": trial.trial_id, "outcome": trial.outcome.value},
        )
    return ExcludedFreezeRunSetMember(
        run_id=trial.run_id,
        trial_id=trial.trial_id,
        reason=trial.exclusion_reason,
    )


class ExperimentCell(ContractModel):
    trial_id: str
    combination_id: str
    treatment: str
    factors: dict[str, JsonValue]
    parameters: dict[str, JsonValue]


class ExperimentBlock(ContractModel):
    block_id: str
    order: tuple[str, ...]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    cells: tuple[ExperimentCell, ...] = ()


class ExperimentPlan(ContractModel):
    plan_token: str
    plan_id: str
    request_digest: str
    workspace_id: str
    experiment_name: str
    experiment: Experiment
    adapter: str
    metric_source: MetricSource = MetricSource.MEASUREMENT
    execution_policy: ExecutionPolicy
    variant_parameter: str
    variants: tuple[str, ...]
    baseline_variant: str | None = None
    factors: dict[str, tuple[JsonValue, ...]] = Field(default_factory=dict)
    parameter_overrides: dict[str, JsonValue]
    blocks: tuple[ExperimentBlock, ...]
    experiment_config_digest: str
    created_at: datetime
    expires_at: datetime


class ExperimentRunResult(ContractModel):
    experiment: Experiment
    variants: tuple[Variant, ...]
    trials: tuple[Trial, ...]
    run_sets: tuple[RunSet, ...]
    comparison: ComparisonResult | None
    outcome: OutcomeExperimentResult | None = None
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


class ExperimentTrialCollection(CursorPageContract):
    page_items_field = "trials"

    experiment_id: str
    trials: tuple[Trial, ...]
    next_cursor: str | None = None


class ExperimentResourceReference(ContractModel):
    """A typed MCP resource link without inlining the referenced evidence."""

    kind: Literal["trials", "run_set", "comparison", "analysis", "artifact"]
    resource_id: str
    uri: str


class ExperimentTrialCounts(ContractModel):
    observed: int = Field(ge=0)
    attempted: int = Field(ge=0)
    completed: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    excluded: int = Field(ge=0)
    unattempted: int = Field(ge=0)


class ExperimentBlockCoverage(ContractModel):
    observed_blocks: int = Field(ge=0)
    published_variant_count: int = Field(ge=0)
    complete_blocks: int | None = Field(default=None, ge=0)
    incomplete_blocks: int | None = Field(default=None, ge=0)


class ExperimentValidationEvidence(ContractModel):
    trial_id: str
    run_id: str
    role: Literal["validation_stdout", "validation_stderr"]
    artifact: ExperimentResourceReference


class ExperimentComparisonStatus(ContractModel):
    comparison: ExperimentResourceReference
    baseline_run_set: ExperimentResourceReference
    candidate_run_set: ExperimentResourceReference
    analysis: ExperimentResourceReference | None = None
    validity: ComparisonValidity
    decision: ComparisonDecision
    effect_size: float | None = None
    relative_change: float | None = None
    confidence_interval: ConfidenceInterval | None = None
    complete_pairs: int | None = Field(default=None, ge=0)
    mismatches: tuple[str, ...] = Field(default=(), max_length=16)
    mismatches_truncated: bool = False


class ExperimentStatus(ContractModel):
    """A bounded read-time projection over durable experiment evidence."""

    protocol: Experiment
    lifecycle: Literal["collecting_or_interrupted", "analyzing", "complete"]
    trial_counts: ExperimentTrialCounts
    block_coverage: ExperimentBlockCoverage
    trials: ExperimentResourceReference
    comparison: ExperimentComparisonStatus | None = None
    outcome: OutcomeExperimentResult | None = None
    validation_evidence: tuple[ExperimentValidationEvidence, ...] = Field(default=(), max_length=32)
    validation_evidence_truncated: bool = False
    corpus_commit_id: str
    recovery: ManualAction | None = None
    limitations: tuple[str, ...] = ()


MAX_TRIAL_PAGE_SIZE = 1_000
_STATUS_VALIDATION_EVIDENCE_LIMIT = 32
_TRIAL_SELECT = (
    "SELECT trial_id, experiment_id, variant_id, run_id, combination_id, "
    "factors_json, block_id, order_in_block, parameter_name, "
    "parameter_value_int, parameter_value_float, attempt, outcome, "
    "exclusion_reason, validation_status, failure_class, "
    "oracle_receipt_json, oracle_receipt_artifact_id FROM trials "
)


class OutcomeCount(ContractModel):
    treatment: str
    attempted: int
    eligible: int
    passed: int
    failed: int
    timed_out: int
    cancelled: int
    unsupported: int
    resource_policy: int
    oracle_failed: int
    infrastructure_failed: int
    oracle_inconclusive: int = 0
    oracle_unsupported: int = 0
    oracle_receipt_error: int = 0
    pass_rate: float | None = None
    failure_rate: float | None = None
    failure_rate_upper_bound: float | None = None


class OutcomeFirstFailure(ContractModel):
    trial_id: str
    factors: dict[str, JsonValue]


class OutcomeExperimentResult(ContractModel):
    experiment_id: str
    method: Literal[ExperimentOutcomeMethod.ABSENCE_OF_FAILURE_FIXED_ATTEMPTS_V1] = (
        ExperimentOutcomeMethod.ABSENCE_OF_FAILURE_FIXED_ATTEMPTS_V1
    )
    goal: ExperimentOutcomeGoal
    disposition: ExperimentOutcomeDisposition
    counts: tuple[OutcomeCount, ...]
    complete_pairs: int
    unmatched_cells: int
    first_failure: OutcomeFirstFailure | None = None
    limitations: tuple[str, ...] = ()


class ExperimentPlanRegistry:
    def __init__(self, *, workspace: Workspace | None = None, ttl_seconds: float = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._workspace: Workspace | None = None
        self._store: AuthorizedPlanStore[ExperimentPlan] | None = None
        if workspace is not None:
            self.bind(workspace)

    def bind(self, workspace: Workspace) -> None:
        if self._workspace is not None and self._workspace.paths.root != workspace.paths.root:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "An experiment plan registry cannot span multiple workspaces.",
            )
        self._workspace = workspace
        self._store = AuthorizedPlanStore(
            workspace,
            family="experiment",
            model=TypeAdapter(ExperimentPlan),
        )

    async def issue(self, plan: ExperimentPlan) -> None:
        self._require_store().issue(
            plan.plan_token,
            plan.request_digest,
            plan,
            expires_at=plan.expires_at,
        )

    async def consume(self, plan_token: str) -> ExperimentPlan:
        return self._require_store().consume(plan_token)

    def _require_store(self) -> AuthorizedPlanStore[ExperimentPlan]:
        if self._store is None:
            raise DomainError(
                ErrorCode.WORKSPACE_NOT_FOUND,
                "Experiment plan storage requires an initialized workspace.",
            )
        return self._store


class ExperimentService:
    def __init__(
        self,
        workspace: Workspace,
        *,
        captures: CaptureService | None = None,
        plans: ExperimentPlanRegistry | None = None,
    ) -> None:
        self.workspace = workspace
        self.workloads = WorkloadService(workspace)
        self.captures = captures or CaptureService(workspace)
        self.plans = plans or ExperimentPlanRegistry(workspace=workspace)
        self.plans.bind(workspace)
        self.publisher = GenerationPublisher(workspace)
        self.experiments = ControlRecordStore(
            workspace,
            kind="experiments",
            model=Experiment,
            id_field="experiment_id",
        )

    def list_trials(
        self,
        experiment_id: str,
        *,
        limit: int = MAX_TRIAL_PAGE_SIZE,
        cursor: str | None = None,
    ) -> ExperimentTrialCollection:
        if not 1 <= limit <= MAX_TRIAL_PAGE_SIZE:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Trial page size must be between 1 and {MAX_TRIAL_PAGE_SIZE}.",
            )
        self.experiments.read(experiment_id)
        head = self.workspace.corpus.read_head()
        scope_digest = digest_model({"experiment_id": experiment_id})
        offset = 0
        if cursor is not None:
            position = cast(
                tuple[int],
                self.workspace.cursors.resolve(
                    cursor,
                    namespace=CursorNamespace.EXPERIMENT_TRIALS,
                    snapshot_id=head.commit_id,
                    scope_digest=scope_digest,
                ),
            )
            offset = position[0]
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            rows = snapshot.execute(
                _TRIAL_SELECT + "WHERE experiment_id = ? QUALIFY row_number() OVER ("
                "PARTITION BY trial_id ORDER BY published_at DESC) = 1 "
                "ORDER BY block_id, order_in_block, trial_id LIMIT ? OFFSET ?",
                (experiment_id, limit + 1, offset),
            ).fetchall()
        truncated = len(rows) > limit
        trials = tuple(self._trial_from_row(row) for row in rows[:limit])
        next_cursor = (
            self.workspace.cursors.issue(
                namespace=CursorNamespace.EXPERIMENT_TRIALS,
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
                position=(offset + limit,),
            )
            if truncated
            else None
        )
        return ExperimentTrialCollection(
            experiment_id=experiment_id,
            trials=trials,
            next_cursor=next_cursor,
        )

    def get_trial(self, trial_id: str, *, experiment_id: str | None = None) -> Trial:
        head = self.workspace.corpus.read_head()
        where = "trial_id = ?"
        parameters: list[object] = [trial_id]
        if experiment_id is not None:
            where += " AND experiment_id = ?"
            parameters.append(experiment_id)
        catalog = Catalog(self.workspace)
        with catalog.open_snapshot(catalog.pin(head.commit_id)) as snapshot:
            rows = snapshot.execute(
                _TRIAL_SELECT + f"WHERE {where} QUALIFY row_number() OVER ("
                "PARTITION BY experiment_id ORDER BY published_at DESC) = 1 "
                "ORDER BY published_at DESC",
                tuple(parameters),
            ).fetchall()
        if experiment_id is None and len(rows) > 1:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Trial {trial_id!r} is ambiguous; provide its experiment ID.",
                details={"ambiguous_entity": "trial", "experiment_ids": [row[1] for row in rows]},
            )
        row = rows[0] if rows else None
        if row is None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Trial {trial_id!r} does not exist.",
                details={"missing_entity": "trial"},
            )
        trial = self._trial_from_row(row)
        return trial

    def status(self, experiment_id: str) -> ExperimentStatus:
        """Reconstruct bounded lifecycle evidence without recording a status object."""
        protocol = self.experiments.read(experiment_id)
        catalog = Catalog(self.workspace)
        handle = catalog.pin()
        with catalog.open_snapshot(handle) as snapshot:
            trial_counts = self._status_trial_counts(snapshot, experiment_id)
            variant_ids = self._status_variant_ids(snapshot, experiment_id)
            block_coverage = self._status_block_coverage(
                snapshot,
                experiment_id,
                published_variant_count=len(variant_ids),
            )
            outcome = self._status_outcome(snapshot, experiment_id)
            comparison = self._status_comparison(snapshot, experiment_id)
            validation_evidence, validation_evidence_truncated = self._status_validation_evidence(
                snapshot,
                experiment_id,
            )
            finalized_run_set_variants = self._status_run_set_variants(snapshot, experiment_id)

        lifecycle, recovery, limitations = self._status_lifecycle(
            variant_ids=variant_ids,
            finalized_run_set_variants=finalized_run_set_variants,
            comparison=comparison,
            outcome=outcome,
        )
        return ExperimentStatus(
            protocol=protocol,
            lifecycle=lifecycle,
            trial_counts=trial_counts,
            block_coverage=block_coverage,
            trials=self._resource("trials", experiment_id),
            comparison=comparison,
            outcome=outcome,
            validation_evidence=validation_evidence,
            validation_evidence_truncated=validation_evidence_truncated,
            corpus_commit_id=handle.commit_id,
            recovery=recovery,
            limitations=limitations,
        )

    @staticmethod
    def _latest_rows(table: str, identifier: str) -> str:
        return (
            f'(SELECT * FROM "{table}" QUALIFY row_number() OVER '
            f'(PARTITION BY "{identifier}" ORDER BY published_at DESC) = 1)'
        )

    def _status_trial_counts(
        self,
        snapshot: Snapshot,
        experiment_id: str,
    ) -> ExperimentTrialCounts:
        row = snapshot.execute(
            "SELECT count(*), "
            "count(*) FILTER (WHERE run_id IS NOT NULL), "
            "count(*) FILTER (WHERE outcome != 'unattempted'), "
            "count(*) FILTER (WHERE outcome = 'succeeded'), "
            "count(*) FILTER (WHERE outcome NOT IN ('succeeded', 'unattempted', 'unsupported')), "
            "count(*) FILTER (WHERE exclusion_reason IS NOT NULL), "
            "count(*) FILTER (WHERE outcome = 'unattempted') "
            "FROM " + self._latest_rows("trials", "trial_id") + " WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        assert row is not None
        return ExperimentTrialCounts(
            observed=int(row[0]),
            attempted=int(row[1]),
            completed=int(row[2]),
            succeeded=int(row[3]),
            failed=int(row[4]),
            excluded=int(row[5]),
            unattempted=int(row[6]),
        )

    def _status_variant_ids(self, snapshot: Snapshot, experiment_id: str) -> tuple[str, ...]:
        rows = snapshot.execute(
            "SELECT variant_id FROM "
            + self._latest_rows("variants", "variant_id")
            + " WHERE experiment_id = ? ORDER BY variant_id",
            (experiment_id,),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def _status_block_coverage(
        self,
        snapshot: Snapshot,
        experiment_id: str,
        *,
        published_variant_count: int,
    ) -> ExperimentBlockCoverage:
        row = snapshot.execute(
            "SELECT count(*) AS observed_blocks, "
            "count(*) FILTER (WHERE variant_count = ?) AS complete_blocks "
            "FROM (SELECT block_id, count(DISTINCT variant_id) AS variant_count FROM "
            + self._latest_rows("trials", "trial_id")
            + " WHERE experiment_id = ? AND block_id IS NOT NULL GROUP BY block_id)",
            (published_variant_count, experiment_id),
        ).fetchone()
        assert row is not None
        observed_blocks = int(row[0])
        if published_variant_count == 0:
            return ExperimentBlockCoverage(
                observed_blocks=observed_blocks,
                published_variant_count=0,
            )
        complete_blocks = int(row[1])
        return ExperimentBlockCoverage(
            observed_blocks=observed_blocks,
            published_variant_count=published_variant_count,
            complete_blocks=complete_blocks,
            incomplete_blocks=observed_blocks - complete_blocks,
        )

    def _status_outcome(
        self,
        snapshot: Snapshot,
        experiment_id: str,
    ) -> OutcomeExperimentResult | None:
        row = snapshot.execute(
            "SELECT experiment_id, method, goal, disposition, counts_json, complete_pairs, "
            "unmatched_cells, first_failure_trial_id, first_failure_factors_json, limitations FROM "
            + self._latest_rows("experiment_outcomes", "experiment_id")
            + " WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        first_failure = (
            {"trial_id": row[7], "factors": json.loads(str(row[8]))}
            if row[7] is not None and row[8] is not None
            else None
        )
        return OutcomeExperimentResult.model_validate(
            {
                "experiment_id": row[0],
                "method": row[1],
                "goal": row[2],
                "disposition": row[3],
                "counts": json.loads(str(row[4])),
                "complete_pairs": row[5],
                "unmatched_cells": row[6],
                "first_failure": first_failure,
                "limitations": row[9],
            }
        )

    def _status_comparison(
        self,
        snapshot: Snapshot,
        experiment_id: str,
    ) -> ExperimentComparisonStatus | None:
        row = snapshot.execute(
            "SELECT comparison_id, baseline_run_set_id, candidate_run_set_id, relative_change, "
            "effect_size, confidence_low, confidence_high, confidence_level, complete_pair_n, "
            "decision, validity, mismatches FROM "
            + self._latest_rows("comparisons", "comparison_id")
            + " WHERE experiment_id = ? ORDER BY published_at DESC, comparison_id DESC LIMIT 1",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        confidence_interval = (
            ConfidenceInterval(low=float(row[5]), high=float(row[6]), level=float(row[7]))
            if row[5] is not None and row[6] is not None and row[7] is not None
            else None
        )
        mismatches = tuple(str(item) for item in row[11])
        comparison_id = str(row[0])
        analysis_id = self._status_comparison_analysis(
            snapshot,
            experiment_id=experiment_id,
            baseline_run_set_id=str(row[1]),
            candidate_run_set_id=str(row[2]),
        )
        return ExperimentComparisonStatus(
            comparison=self._resource("comparison", comparison_id),
            baseline_run_set=self._resource("run_set", str(row[1])),
            candidate_run_set=self._resource("run_set", str(row[2])),
            analysis=(self._resource("analysis", analysis_id) if analysis_id is not None else None),
            validity=ComparisonValidity(str(row[10])),
            decision=ComparisonDecision(str(row[9])),
            effect_size=float(row[4]) if row[4] is not None else None,
            relative_change=float(row[3]) if row[3] is not None else None,
            confidence_interval=confidence_interval,
            complete_pairs=int(row[8]) if row[8] is not None else None,
            mismatches=mismatches[:16],
            mismatches_truncated=len(mismatches) > 16,
        )

    def _status_comparison_analysis(
        self,
        snapshot: Snapshot,
        *,
        experiment_id: str,
        baseline_run_set_id: str,
        candidate_run_set_id: str,
    ) -> str | None:
        row = snapshot.execute(
            "SELECT analysis_id FROM analyses WHERE recipe = 'compare_run_sets' "
            "AND json_extract_string(parameters_json, '$.experiment_id') = ? "
            "AND json_extract_string(parameters_json, '$.baseline_run_set_id') = ? "
            "AND json_extract_string(parameters_json, '$.candidate_run_set_id') = ? "
            "QUALIFY row_number() OVER (PARTITION BY analysis_id ORDER BY published_at DESC) = 1 "
            "ORDER BY completed_at DESC, analysis_id DESC LIMIT 1",
            (experiment_id, baseline_run_set_id, candidate_run_set_id),
        ).fetchone()
        return str(row[0]) if row is not None else None

    def _status_validation_evidence(
        self,
        snapshot: Snapshot,
        experiment_id: str,
    ) -> tuple[tuple[ExperimentValidationEvidence, ...], bool]:
        rows = snapshot.execute(
            "WITH latest_trials AS "
            + self._latest_rows("trials", "trial_id")
            + ", unstructured_trials AS ("
            "SELECT trial_id, run_id, block_id, order_in_block FROM latest_trials "
            "WHERE experiment_id = ? AND run_id IS NOT NULL "
            "AND oracle_receipt_artifact_id IS NULL"
            ") SELECT trial_id, run_id, artifact_id, role FROM unstructured_trials "
            "JOIN artifact_registrations USING (run_id) "
            "WHERE role IN ('validation_stdout', 'validation_stderr') "
            "QUALIFY row_number() OVER (PARTITION BY trial_id, artifact_id "
            "ORDER BY published_at DESC) = 1 "
            "ORDER BY block_id, order_in_block, role, artifact_id LIMIT ?",
            (experiment_id, _STATUS_VALIDATION_EVIDENCE_LIMIT + 1),
        ).fetchall()
        truncated = len(rows) > _STATUS_VALIDATION_EVIDENCE_LIMIT
        values = tuple(
            ExperimentValidationEvidence(
                trial_id=str(row[0]),
                run_id=str(row[1]),
                role=cast(Literal["validation_stdout", "validation_stderr"], str(row[3])),
                artifact=self._resource("artifact", str(row[2])),
            )
            for row in rows[:_STATUS_VALIDATION_EVIDENCE_LIMIT]
        )
        return values, truncated

    def _status_run_set_variants(self, snapshot: Snapshot, experiment_id: str) -> set[str]:
        rows = snapshot.execute(
            "SELECT DISTINCT json_extract_string(selection_json, '$.variant_id') FROM "
            + self._latest_rows("run_sets", "run_set_id")
            + " WHERE json_extract_string(selection_json, '$.experiment_id') = ? "
            "AND json_extract_string(selection_json, '$.variant_id') IS NOT NULL",
            (experiment_id,),
        ).fetchall()
        return {str(row[0]) for row in rows}

    @staticmethod
    def _status_lifecycle(
        *,
        variant_ids: tuple[str, ...],
        finalized_run_set_variants: set[str],
        comparison: ExperimentComparisonStatus | None,
        outcome: OutcomeExperimentResult | None,
    ) -> tuple[
        Literal["collecting_or_interrupted", "analyzing", "complete"],
        ManualAction | None,
        tuple[str, ...],
    ]:
        if not variant_ids:
            return (
                "collecting_or_interrupted",
                manual_action(
                    "Inspect the bounded trial resource. If no experiment call is still running, "
                    "plan a new experiment rather than reusing its consumed plan token."
                ),
                (
                    "No complete variant cohort is published, so durable evidence cannot "
                    "distinguish an active execution from an interrupted one.",
                ),
            )
        if outcome is not None:
            return "complete", None, ()
        run_sets_complete = set(variant_ids).issubset(finalized_run_set_variants)
        if len(variant_ids) == 2:
            if comparison is not None:
                return "complete", None, ()
            return (
                "analyzing",
                manual_action(
                    "Inspect the frozen run sets and record a comparison with the declared "
                    "metric contract if result publication was interrupted."
                ),
                (
                    "The full trial cohort is published, but no terminal paired comparison is "
                    "available.",
                ),
            )
        if run_sets_complete:
            return "complete", None, ()
        return (
            "analyzing",
            manual_action(
                "Inspect the bounded trial resource and frozen run sets; publish only the "
                "missing analysis from their immutable evidence."
            ),
            ("The full trial cohort is published, but its frozen run-set evidence is incomplete.",),
        )

    @staticmethod
    def _resource(
        kind: Literal["trials", "run_set", "comparison", "analysis", "artifact"],
        resource_id: str,
    ) -> ExperimentResourceReference:
        if kind == "trials":
            uri = f"flameox://experiments/{resource_id}/trials"
        elif kind == "run_set":
            uri = f"flameox://run-sets/{resource_id}"
        elif kind in {"comparison", "analysis"}:
            uri = f"flameox://evidence/{kind}/{resource_id}"
        else:
            uri = f"flameox://artifacts/{resource_id}"
        return ExperimentResourceReference(kind=kind, resource_id=resource_id, uri=uri)

    @staticmethod
    def _trial_from_row(row: tuple[object, ...]) -> Trial:
        receipt_json = str(row[16]) if row[16] is not None else None
        return parse_trial(
            {
                "trial_id": row[0],
                "experiment_id": row[1],
                "variant_id": row[2],
                "run_id": row[3],
                "combination_id": row[4],
                "factors": json.loads(str(row[5])),
                "block_id": row[6],
                "order_in_block": row[7],
                "parameter_name": row[8],
                "parameter_value": numeric_value_from_columns(
                    row[9],
                    row[10],
                    field_name="trial parameter value",
                ),
                "attempt": row[11],
                "outcome": row[12],
                "exclusion_reason": row[13],
                "validation_status": row[14],
                "failure_class": row[15],
                "oracle_receipt": json.loads(receipt_json) if receipt_json is not None else None,
                "oracle_receipt_artifact_id": row[17],
            }
        )

    async def plan(
        self,
        *,
        experiment_name: str,
        investigation_id: str,
        hypothesis_id: str | None = None,
        adapter: str,
        parameter_overrides: dict[str, Scalar] | None = None,
        execution_policy: ExecutionPolicy,
    ) -> ExperimentPlan:
        project = self.workloads.load()
        try:
            config = project.experiments[experiment_name]
            workload = project.workloads[config.workload]
        except KeyError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Unknown experiment {experiment_name!r}.",
            ) from exc
        investigations = ControlRecordStore(
            self.workspace,
            kind="investigations",
            model=Investigation,
            id_field="investigation_id",
        )
        investigations.read(investigation_id)
        if hypothesis_id is not None:
            hypotheses = ControlRecordStore(
                self.workspace,
                kind="hypotheses",
                model=Hypothesis,
                id_field="hypothesis_id",
                revision_field="revision",
            )
            hypothesis = hypotheses.read(hypothesis_id)
            if hypothesis.investigation_id != investigation_id:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "The hypothesis does not belong to the experiment investigation.",
                )
        supplied_overrides = parameter_overrides or {}
        variant_parameter, factors, combinations = self._materialize_combinations(
            config,
            workload.parameters,
        )
        variants = tuple(self._factor_label(value) for value in factors[variant_parameter])
        conflicting = sorted(set(supplied_overrides) & set(factors))
        if conflicting:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Experiment overrides cannot replace declared factors.",
                details={"parameters": conflicting},
            )
        trial_count = len(combinations) * config.blocks
        if trial_count > config.max_trials:
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Experiment plan exceeds its {config.max_trials}-trial budget.",
                details={
                    "trial_count": trial_count,
                    "blocks": config.blocks,
                    "combinations": len(combinations),
                    "limit": config.max_trials,
                },
            )
        for combination in combinations:
            self.workloads.resolve(
                config.workload,
                {**supplied_overrides, **combination},
            )
        definition = self.workloads.definition(config.workload)
        oracle = workload.oracle
        role = (
            ExperimentRole.CONFIRMATORY
            if oracle is not None and oracle.strength is OracleStrength.CROSS_TREATMENT_EQUIVALENCE
            else ExperimentRole.EXPLORATORY
        )
        design = {
            "name": experiment_name,
            "config": config.model_dump(mode="json"),
            "parameters": supplied_overrides,
            "metric_source": (
                MetricSource.RUNTIME_RESOURCE
                if config.primary_metric.startswith("runtime_resource.")
                else MetricSource.MEASUREMENT
            ),
        }
        metric_source = (
            MetricSource.RUNTIME_RESOURCE
            if config.primary_metric.startswith("runtime_resource.")
            else MetricSource.MEASUREMENT
        )
        if isinstance(config, _OutcomeExperimentConfig):
            primary_metric_unit = None
            measurement_series = None
            value_domain = None
            zero_policy = None
        else:
            primary_metric_unit = (
                "bytes"
                if metric_source is MetricSource.RUNTIME_RESOURCE
                else config.primary_metric_unit or "ns"
            )
            measurement_series = config.measurement_series
            value_domain = config.value_domain
            zero_policy = config.zero_policy
        experiment = Experiment(
            experiment_id=new_id(),
            investigation_id=investigation_id,
            hypothesis_id=hypothesis_id,
            recipe=(
                "absence_of_failure_counts" if config.analysis == "outcome" else "compare_run_sets"
            ),
            recipe_version="2" if config.analysis == "outcome" else "1",
            workload_definition_id=definition.workload_definition_id,
            experiment_design_id=digest_model(design),
            measurement_protocol_id=digest_model(
                {
                    "adapter": adapter,
                    "metric_source": metric_source,
                    "primary_metric": config.primary_metric,
                    "primary_metric_unit": primary_metric_unit,
                    "measurement_series": (
                        measurement_series.model_dump(mode="json")
                        if measurement_series is not None
                        else None
                    ),
                    "value_domain": value_domain,
                    "zero_policy": zero_policy,
                }
            ),
            validation_spec_id=definition.validation_spec_id,
            primary_metric=config.primary_metric,
            metric_source=metric_source,
            primary_metric_unit=primary_metric_unit,
            measurement_series=measurement_series,
            value_domain=value_domain,
            zero_policy=zero_policy,
            polarity=config.polarity,
            estimand=config.estimand,
            practical_threshold=config.practical_threshold,
            confidence_level=config.confidence_level,
            stopping_rule={
                "method": (
                    ExperimentOutcomeMethod.ABSENCE_OF_FAILURE_FIXED_ATTEMPTS_V1
                    if isinstance(config, _OutcomeExperimentConfig)
                    else ExperimentOutcomeMethod.FIXED_ATTEMPTS_V1
                ),
                "fixed_blocks": config.blocks,
                "minimum_attempts": config.minimum_attempts or config.blocks,
                "maximum_attempts": config.maximum_attempts or config.blocks,
            },
            random_seed=config.random_seed,
            role=role,
        )
        generator = random.Random(config.random_seed)
        blocks: list[ExperimentBlock] = []
        coordinates: dict[str, list[dict[str, Scalar]]] = {}
        for combination in combinations:
            coordinate = {
                name: value for name, value in combination.items() if name != variant_parameter
            }
            coordinate_id = digest_model(coordinate)
            coordinates.setdefault(coordinate_id, []).append(combination)
        config_digest = digest_model(config.model_dump(mode="json"))
        for point_index, (_coordinate_id, coordinate_combinations) in enumerate(
            coordinates.items(),
            start=1,
        ):
            for repetition in range(1, config.blocks + 1):
                ordered = list(coordinate_combinations)
                if config.design in {"randomized", "randomized_complete_blocks"}:
                    generator.shuffle(ordered)
                cells = tuple(
                    ExperimentCell(
                        trial_id=digest_model(
                            {
                                "experiment_config": config_digest,
                                "combination": combination,
                                "repetition": repetition,
                            }
                        ),
                        combination_id=digest_model(
                            {"experiment_config": config_digest, "factors": combination}
                        ),
                        treatment=self._factor_label(combination[variant_parameter]),
                        factors={
                            name: cast(JsonValue, value) for name, value in combination.items()
                        },
                        parameters={
                            name: cast(JsonValue, value) for name, value in combination.items()
                        },
                    )
                    for combination in ordered
                )
                blocks.append(
                    ExperimentBlock(
                        block_id=f"cell-{point_index:04d}-block-{repetition:04d}",
                        order=tuple(cell.treatment for cell in cells),
                        parameters={
                            name: cast(JsonValue, value)
                            for name, value in coordinate_combinations[0].items()
                            if name != variant_parameter
                        },
                        cells=cells,
                    )
                )
        created = utc_now()
        plan_id = secrets.token_hex(32)
        overrides = {name: cast(JsonValue, value) for name, value in supplied_overrides.items()}
        request = {
            "workspace_id": self.workspace.identity.workspace_id,
            "experiment": experiment.model_dump(mode="json"),
            "adapter": adapter,
            "variant_parameter": variant_parameter,
            "variants": variants,
            "factors": {
                name: [cast(JsonValue, value) for value in values]
                for name, values in factors.items()
            },
            "parameters": overrides,
            "blocks": [block.model_dump(mode="json") for block in blocks],
            "workload_definition_id": definition.workload_definition_id,
            "experiment_config_digest": digest_model(config.model_dump(mode="json")),
        }
        plan = ExperimentPlan(
            plan_token=secrets.token_hex(32),
            plan_id=plan_id,
            request_digest=digest_model(request),
            workspace_id=self.workspace.identity.workspace_id,
            experiment_name=experiment_name,
            experiment=experiment,
            adapter=adapter,
            metric_source=(
                MetricSource.RUNTIME_RESOURCE
                if config.primary_metric.startswith("runtime_resource.")
                else MetricSource.MEASUREMENT
            ),
            execution_policy=execution_policy,
            variant_parameter=variant_parameter,
            variants=variants,
            baseline_variant=(
                self._factor_label(config.baseline_value)
                if config.baseline_value is not None
                else None
            ),
            factors={
                name: tuple(cast(JsonValue, value) for value in values)
                for name, values in factors.items()
            },
            parameter_overrides=overrides,
            blocks=tuple(blocks),
            experiment_config_digest=digest_model(config.model_dump(mode="json")),
            created_at=created,
            expires_at=created + timedelta(seconds=self.plans.ttl_seconds),
        )
        await self.plans.issue(plan)
        return plan

    async def run(
        self,
        plan_token: str,
        *,
        progress: Callable[[float, float, str], Awaitable[None]] | None = None,
    ) -> ExperimentRunResult:
        plan = await self.plans.consume(plan_token)
        trial_count = sum(len(block.cells) for block in plan.blocks)
        total_phases = trial_count + 4
        completed = 0
        reporter = ProgressReporter(progress)

        async def report(message: str) -> None:
            await reporter.report(completed, total_phases, message)

        await report("Experiment plan consumed")
        config = self._validate_plan(plan)
        completed += 1
        await report("Experiment plan and workload definition validated")
        # Persist the predeclared protocol before the first treatment starts. If
        # execution is interrupted, the investigation still records what was
        # intended rather than leaving an unattributed run population.
        self.experiments.create(plan.experiment)
        await run_atomic_thread(
            lambda: self.publisher.publish_rows(
                {"experiments": [self._experiment_row(plan.experiment)]},
                publisher="flameox.experiments",
                publisher_version="1",
            )
        )
        completed += 1
        await report("Experiment protocol published")
        trials: list[Trial] = []
        trials_by_variant: dict[str, list[Trial]] = {name: [] for name in plan.variants}
        schedule = tuple(
            (block, order, cell) for block in plan.blocks for order, cell in enumerate(block.cells)
        )
        for schedule_index, (block, order, cell) in enumerate(schedule):
            variant_name = cell.treatment
            parameters = {
                **cast(dict[str, Scalar], plan.parameter_overrides),
                **cast(dict[str, Scalar], cell.parameters),
            }
            capture_plan = None
            run: RunManifest | None = None
            failure_class: TrialFailureClass
            try:
                capture_plan = await self.captures.plan(
                    workload_name=config.workload,
                    adapter=plan.adapter,
                    parameters=parameters,
                    execution_policy=plan.execution_policy,
                )
                captured = await self.captures.execute(capture_plan.plan_token)
                run = captured.run
                outcome, failure_class = self._classify_run(run)
                if _has_extractable_artifact(run, plan.adapter):
                    await run_atomic_thread(
                        partial(
                            _extract_adapter_measurements,
                            self.workspace,
                            plan.adapter,
                            run.run_id,
                        )
                    )
            except asyncio.CancelledError as cancellation:
                run = (
                    RunStore(self.workspace).read(capture_plan.run_id)
                    if capture_plan is not None
                    and RunStore(self.workspace).exists(capture_plan.run_id)
                    else None
                )
                trial = self._make_trial(
                    plan=plan,
                    cell=cell,
                    run=run,
                    block_id=block.block_id,
                    order=order,
                    outcome=TrialOutcome.CANCELLED,
                    failure_class=TrialFailureClass.CANCELLATION,
                )
                try:
                    await run_atomic_thread(partial(self._publish_trial, trial))
                    await self._publish_unattempted(
                        plan,
                        schedule[schedule_index + 1 :],
                    )
                finally:
                    raise cancellation
            except DomainError as error:
                if error.run_id is None:
                    if not isinstance(config, _OutcomeExperimentConfig):
                        failed = self._make_trial(
                            plan=plan,
                            cell=cell,
                            run=None,
                            block_id=block.block_id,
                            order=order,
                            outcome=TrialOutcome.INFRASTRUCTURE_FAILED,
                            failure_class=TrialFailureClass.INFRASTRUCTURE_FAILURE,
                        )
                        await run_atomic_thread(partial(self._publish_trial, failed))
                        await self._publish_unattempted(
                            plan,
                            schedule[schedule_index + 1 :],
                        )
                        raise
                    run = None
                    outcome = (
                        TrialOutcome.UNSUPPORTED
                        if error.code is ErrorCode.CAPABILITY_UNAVAILABLE
                        else TrialOutcome.INFRASTRUCTURE_FAILED
                    )
                    failure_class = (
                        TrialFailureClass.UNSUPPORTED_ENVIRONMENT
                        if outcome is TrialOutcome.UNSUPPORTED
                        else TrialFailureClass.INFRASTRUCTURE_FAILURE
                    )
                else:
                    run = RunStore(self.workspace).read(error.run_id)
                    outcome, failure_class = self._classify_run(run)
            trial = self._make_trial(
                plan=plan,
                cell=cell,
                run=run,
                block_id=block.block_id,
                order=order,
                outcome=outcome,
                failure_class=failure_class,
            )
            trials.append(trial)
            trials_by_variant[variant_name].append(trial)
            await run_atomic_thread(partial(self._publish_trial, trial))
            completed += 1
            await report(
                f"Trial {completed - 2}/{trial_count} published ({block.block_id}, {variant_name})"
            )
        variants: list[Variant] = []
        for name in plan.variants:
            variants.append(self._variant_for_treatment(plan, name, trials_by_variant[name]))
        await run_atomic_thread(
            lambda: self.publisher.publish_rows(
                {
                    "variants": [self._variant_row(value) for value in variants],
                },
                publisher="flameox.experiments",
                publisher_version="1",
                input_run_ids=tuple(trial.run_id for trial in trials if trial.run_id is not None),
            )
        )
        run_sets = await run_atomic_thread(partial(self._freeze_run_sets, plan, trials_by_variant))
        completed += 1
        await report("Variants and frozen run sets published")
        comparison: ComparisonResult | None = None
        outcome_result: OutcomeExperimentResult | None = None
        limitations: list[str] = []
        if isinstance(config, _OutcomeExperimentConfig):
            outcome_result = self._outcome_result(plan, config, trials)
            await run_atomic_thread(
                lambda: self.publisher.publish_rows(
                    {"experiment_outcomes": [self._outcome_row(outcome_result)]},
                    publisher="flameox.experiments",
                    publisher_version="1",
                    input_run_ids=tuple(
                        trial.run_id for trial in trials if trial.run_id is not None
                    ),
                )
            )
            limitations.extend(outcome_result.limitations)
        elif len(run_sets) != 2:
            limitations.append(
                "Automatic paired comparison currently requires exactly two variants."
            )
        else:
            comparison_run_sets: tuple[RunSet, RunSet] | None = run_sets
            if plan.baseline_variant is None:
                limitations.append(
                    "Baseline was determined by list position, not an explicit "
                    "baseline_value. Reordering the treatment list reverses the "
                    "comparison direction."
                )
            else:
                baseline_run_sets = tuple(
                    run_set
                    for run_set in run_sets
                    if run_set.selection["variant"] == plan.baseline_variant
                )
                candidate_run_sets = tuple(
                    run_set
                    for run_set in run_sets
                    if run_set.selection["variant"] != plan.baseline_variant
                )
                if len(baseline_run_sets) != 1 or len(candidate_run_sets) != 1:
                    limitations.append(
                        "Automatic paired comparison requires the declared baseline and exactly "
                        "one candidate treatment."
                    )
                    comparison_run_sets = None
                else:
                    comparison_run_sets = (baseline_run_sets[0], candidate_run_sets[0])
            if comparison_run_sets is not None:
                comparison = await run_atomic_thread(
                    lambda: ComparisonService(self.workspace).record(
                        parse_compare_run_sets_request(
                            {
                                "baseline_run_set_id": comparison_run_sets[0].run_set_id,
                                "candidate_run_set_id": comparison_run_sets[1].run_set_id,
                                "experiment_id": plan.experiment.experiment_id,
                                "metric": plan.experiment.primary_metric,
                                "unit": plan.experiment.primary_metric_unit,
                                "metric_source": plan.metric_source,
                                "polarity": plan.experiment.polarity,
                                "estimand": plan.experiment.estimand,
                                **(
                                    {
                                        "series": (
                                            plan.experiment.measurement_series.model_dump(
                                                mode="json"
                                            )
                                            if plan.experiment.measurement_series is not None
                                            else None
                                        )
                                    }
                                    if plan.metric_source is MetricSource.MEASUREMENT
                                    else {}
                                ),
                                "practical_threshold": plan.experiment.practical_threshold,
                                "confidence_level": plan.experiment.confidence_level,
                                "random_seed": plan.experiment.random_seed,
                            }
                        )
                    )
                )
        result_commit_id = self._result_snapshot(
            experiment=plan.experiment,
            variants=tuple(variants),
            trials=tuple(trials),
            run_sets=run_sets,
            comparison=comparison,
            outcome=outcome_result,
        )
        completed += 1
        await report("Experiment comparison and result complete")
        return ExperimentRunResult(
            experiment=plan.experiment,
            variants=tuple(variants),
            trials=tuple(trials),
            run_sets=run_sets,
            comparison=comparison,
            outcome=outcome_result,
            corpus_commit_id=result_commit_id,
            limitations=tuple(limitations),
        )

    def _result_snapshot(
        self,
        *,
        experiment: Experiment,
        variants: tuple[Variant, ...],
        trials: tuple[Trial, ...],
        run_sets: tuple[RunSet, ...],
        comparison: ComparisonResult | None,
        outcome: OutcomeExperimentResult | None,
    ) -> str:
        """Return one snapshot that contains every durable object in the result."""
        with EvidenceLookupService(self.workspace).session() as session:
            self._require_snapshot_ids(
                session,
                table="experiments",
                identifier="experiment_id",
                expected=(experiment.experiment_id,),
            )
            self._require_snapshot_ids(
                session,
                table="variants",
                identifier="variant_id",
                expected=tuple(value.variant_id for value in variants),
            )
            self._require_snapshot_ids(
                session,
                table="trials",
                identifier="trial_id",
                expected=tuple(value.trial_id for value in trials),
            )
            self._require_snapshot_ids(
                session,
                table="run_sets",
                identifier="run_set_id",
                expected=tuple(value.run_set_id for value in run_sets),
            )
            if comparison is not None:
                self._require_snapshot_ids(
                    session,
                    table="comparisons",
                    identifier="comparison_id",
                    expected=(comparison.comparison.comparison_id,),
                )
            if outcome is not None:
                self._require_snapshot_ids(
                    session,
                    table="experiment_outcomes",
                    identifier="experiment_id",
                    expected=(experiment.experiment_id,),
                )
            return session.commit_id

    @staticmethod
    def _require_snapshot_ids(
        session: EvidenceSession,
        *,
        table: str,
        identifier: str,
        expected: tuple[str, ...],
    ) -> None:
        if not expected:
            return
        placeholders = ", ".join("?" for _ in expected)
        rows = session.execute(
            f'SELECT DISTINCT "{identifier}" FROM "{table}" '
            f'WHERE "{identifier}" IN ({placeholders})',
            tuple(expected),
        ).fetchall()
        observed = {str(row[0]) for row in rows}
        missing = sorted(set(expected) - observed)
        if missing:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "The experiment result snapshot does not contain every returned output.",
                details={
                    "corpus_commit_id": session.commit_id,
                    "table": table,
                    "missing_ids": missing,
                },
            )

    def _freeze_run_sets(
        self,
        plan: ExperimentPlan,
        trials_by_variant: dict[str, list[Trial]],
    ) -> tuple[RunSet, ...]:
        service = RunSetService(self.workspace)
        return tuple(
            service.freeze(
                FreezeRunMembersRequest(
                    members=tuple(
                        _freeze_trial_member(trial)
                        for trial in trials_by_variant[name]
                        if trial.run_id is not None
                    ),
                    selection={
                        "experiment_id": plan.experiment.experiment_id,
                        "variant": name,
                        "variant_id": self._variant_id(
                            plan,
                            next(
                                cast(Scalar, value)
                                for value in plan.factors[plan.variant_parameter]
                                if self._factor_label(cast(Scalar, value)) == name
                            ),
                        ),
                        "treatment_factor": plan.variant_parameter,
                        "treatment_identity_id": self._treatment_identity_id(
                            plan.variant_parameter,
                            next(
                                cast(Scalar, value)
                                for value in plan.factors[plan.variant_parameter]
                                if self._factor_label(cast(Scalar, value)) == name
                            ),
                        ),
                        "experiment_design_id": plan.experiment.experiment_design_id,
                        "combination_population_digest": digest_model(
                            sorted({trial.combination_id for trial in trials_by_variant[name]})
                        ),
                        "combination_count": len(
                            {trial.combination_id for trial in trials_by_variant[name]}
                        ),
                    },
                )
            )
            for name in plan.variants
            if any(trial.run_id is not None for trial in trials_by_variant[name])
        )

    async def _publish_unattempted(
        self,
        plan: ExperimentPlan,
        schedule: tuple[tuple[ExperimentBlock, int, ExperimentCell], ...],
    ) -> None:
        for block, order, cell in schedule:
            trial = self._make_trial(
                plan=plan,
                cell=cell,
                run=None,
                block_id=block.block_id,
                order=order,
                outcome=TrialOutcome.UNATTEMPTED,
                failure_class=TrialFailureClass.UNATTEMPTED,
            )
            await run_atomic_thread(partial(self._publish_trial, trial))

    def _validate_plan(self, plan: ExperimentPlan) -> ExperimentConfig:
        if plan.workspace_id != self.workspace.identity.workspace_id:
            raise DomainError(ErrorCode.INVALID_CAPTURE_PLAN, "Workspace changed.")
        project = self.workloads.load()
        config = project.experiments[plan.experiment_name]
        if digest_model(config.model_dump(mode="json")) != plan.experiment_config_digest:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Experiment definition changed after planning.",
            )
        definition = self.workloads.definition(config.workload)
        if definition.workload_definition_id != plan.experiment.workload_definition_id:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Workload definition changed after experiment planning.",
            )
        return config

    def _materialize_combinations(
        self,
        config: ExperimentConfig,
        workload_parameters: dict[str, tuple[Scalar, ...]],
    ) -> tuple[str, dict[str, tuple[Scalar, ...]], tuple[dict[str, Scalar], ...]]:
        treatment_factor = config.treatment_factor
        factors = dict(config.factors)

        for name, values in factors.items():
            allowed = workload_parameters.get(name)
            if allowed is None:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Experiment factor {name!r} is not a workload parameter.",
                )
            if len(scalar_identity_set(list(values))) != len(values) or not scalar_subset(
                list(values), list(allowed)
            ):
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Experiment factor {name!r} contains duplicate or undeclared values.",
                )

        factor_names = tuple(factors)
        if config.combination_policy == "explicit":
            raw = [dict(combination) for combination in config.combinations]
        else:
            raw = [
                dict(zip(factor_names, values, strict=True))
                for values in product(*(factors[name] for name in factor_names))
            ]
        combinations: list[dict[str, Scalar]] = []
        identities: set[str] = set()
        for combination in raw:
            if set(combination) != set(factor_names):
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "Explicit combinations must contain every declared factor exactly once.",
                )
            if any(not scalar_contains(combination[name], factors[name]) for name in factor_names):
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "Explicit combination contains an undeclared factor value.",
                )
            identity = digest_model(combination)
            if identity in identities:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "Experiment combinations must be unique.",
                )
            identities.add(identity)
            combinations.append(combination)

        for rule in config.exclude:
            if not rule or not set(rule).issubset(factors):
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "Every exclusion must name at least one declared factor.",
                )
            if any(not scalar_contains(value, factors[name]) for name, value in rule.items()):
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "Exclusion contains an undeclared factor value.",
                )
        filtered = tuple(
            combination
            for combination in combinations
            if not any(
                all(scalar_equal(combination[name], value) for name, value in rule.items())
                for rule in config.exclude
            )
        )
        if not filtered:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Experiment combination rules exclude every cell.",
            )
        return treatment_factor, factors, filtered

    @staticmethod
    def _factor_label(value: Scalar) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _outcome_result(
        self,
        plan: ExperimentPlan,
        config: _OutcomeExperimentConfig,
        trials: list[Trial],
    ) -> OutcomeExperimentResult:
        counts: list[OutcomeCount] = []
        for treatment in plan.variants:
            treatment_value = next(
                value
                for value in plan.factors[plan.variant_parameter]
                if self._factor_label(cast(Scalar, value)) == treatment
            )
            selected = [
                trial
                for trial in trials
                if scalar_equal(
                    cast(Scalar, trial.factors.get(plan.variant_parameter)),
                    cast(Scalar, treatment_value),
                )
            ]
            attempted = sum(trial.outcome is not TrialOutcome.UNATTEMPTED for trial in selected)
            eligible = sum(
                trial.failure_class
                not in {
                    "unattempted",
                    "cancellation",
                    "unsupported_environment",
                    "oracle_inconclusive",
                    "oracle_unsupported",
                    "infrastructure_failure",
                }
                for trial in selected
            )
            passed = sum(trial.outcome is TrialOutcome.SUCCEEDED for trial in selected)
            failed = sum(
                trial.failure_class
                in {
                    "oracle_failure",
                    "oracle_receipt_error",
                    "process_failure",
                    "timeout",
                    "resource_policy",
                }
                for trial in selected
            )
            counts.append(
                OutcomeCount(
                    treatment=treatment,
                    attempted=attempted,
                    eligible=eligible,
                    passed=passed,
                    failed=failed,
                    timed_out=sum(trial.failure_class == "timeout" for trial in selected),
                    cancelled=sum(trial.failure_class == "cancellation" for trial in selected),
                    unsupported=sum(
                        trial.failure_class == "unsupported_environment" for trial in selected
                    ),
                    resource_policy=sum(
                        trial.failure_class == "resource_policy" for trial in selected
                    ),
                    oracle_failed=sum(
                        trial.failure_class == "oracle_failure" for trial in selected
                    ),
                    oracle_inconclusive=sum(
                        trial.failure_class == "oracle_inconclusive" for trial in selected
                    ),
                    oracle_unsupported=sum(
                        trial.failure_class == "oracle_unsupported" for trial in selected
                    ),
                    oracle_receipt_error=sum(
                        trial.failure_class == "oracle_receipt_error" for trial in selected
                    ),
                    infrastructure_failed=sum(
                        trial.failure_class == "infrastructure_failure" for trial in selected
                    ),
                    pass_rate=passed / eligible if eligible else None,
                    failure_rate=failed / eligible if eligible else None,
                    failure_rate_upper_bound=self._failure_rate_upper_bound(
                        failed,
                        eligible,
                        config.confidence_level,
                    ),
                )
            )
        by_block: dict[str, list[Trial]] = {}
        for trial in trials:
            if trial.block_id is not None and trial.outcome is not TrialOutcome.UNATTEMPTED:
                by_block.setdefault(trial.block_id, []).append(trial)
        complete_pairs = sum(
            len({trial.variant_id for trial in block}) == len(plan.variants)
            for block in by_block.values()
        )
        unmatched = sum(
            abs(len(plan.variants) - len({trial.variant_id for trial in block}))
            for block in by_block.values()
        )
        failures = [
            trial
            for trial in trials
            if trial.failure_class
            in {
                "oracle_failure",
                "oracle_receipt_error",
                "process_failure",
                "timeout",
                "resource_policy",
            }
        ]
        failed_treatments = {
            self._factor_label(cast(Scalar, trial.factors[plan.variant_parameter]))
            for trial in failures
        }
        minimum = config.minimum_attempts or config.blocks
        limitations: list[str] = [
            "Clean trials bound only the declared fixed attempts; the reported exact "
            "one-sided failure-rate bound does not prove absence of rare failures or race freedom."
        ]
        if unmatched:
            limitations.append("One or more pairing coordinates lack every treatment.")
        incomplete_receipts = any(
            trial.failure_class in {"oracle_inconclusive", "oracle_receipt_error", "unattempted"}
            for trial in trials
        )
        baseline_variant = plan.baseline_variant
        if baseline_variant is None and plan.variants:
            baseline_variant = plan.variants[0]
        if counts and all(
            item.unsupported + item.oracle_unsupported == item.attempted and item.attempted > 0
            for item in counts
        ):
            disposition = ExperimentOutcomeDisposition.UNSUPPORTED
        elif incomplete_receipts or unmatched or any(item.eligible < minimum for item in counts):
            disposition = ExperimentOutcomeDisposition.INSUFFICIENT_EVIDENCE
        elif not failures:
            disposition = ExperimentOutcomeDisposition.ALL_CLEAN
        elif (
            len(plan.variants) == 2
            and baseline_variant is not None
            and failed_treatments == {baseline_variant}
        ):
            disposition = ExperimentOutcomeDisposition.BASE_ONLY_FAILURE
        elif (
            len(plan.variants) == 2
            and baseline_variant is not None
            and failed_treatments == {v for v in plan.variants if v != baseline_variant}
        ):
            disposition = ExperimentOutcomeDisposition.CANDIDATE_ONLY_FAILURE
        else:
            disposition = ExperimentOutcomeDisposition.MIXED
        first_failure = failures[0] if failures else None
        return OutcomeExperimentResult(
            experiment_id=plan.experiment.experiment_id,
            goal=config.outcome_goal,
            disposition=disposition,
            counts=tuple(counts),
            complete_pairs=complete_pairs,
            unmatched_cells=unmatched,
            first_failure=(
                OutcomeFirstFailure(
                    trial_id=first_failure.trial_id,
                    factors=first_failure.factors,
                )
                if first_failure is not None
                else None
            ),
            limitations=tuple(limitations),
        )

    @staticmethod
    def _failure_rate_upper_bound(
        failures: int,
        attempts: int,
        confidence_level: float,
    ) -> float | None:
        if attempts == 0:
            return None
        if failures == attempts:
            return 1.0
        return float(beta.ppf(confidence_level, failures + 1, attempts - failures))

    @staticmethod
    def _outcome_row(value: OutcomeExperimentResult) -> dict[str, object]:
        return {
            "experiment_id": value.experiment_id,
            "method": value.method,
            "goal": value.goal,
            "disposition": value.disposition,
            "counts_json": json.dumps(
                [item.model_dump(mode="json") for item in value.counts],
                separators=(",", ":"),
                sort_keys=True,
            ),
            "complete_pairs": value.complete_pairs,
            "unmatched_cells": value.unmatched_cells,
            "first_failure_trial_id": (
                value.first_failure.trial_id if value.first_failure is not None else None
            ),
            "first_failure_factors_json": (
                json.dumps(
                    value.first_failure.factors,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                if value.first_failure is not None
                else None
            ),
            "limitations": list(value.limitations),
        }

    def _make_trial(
        self,
        *,
        plan: ExperimentPlan,
        cell: ExperimentCell,
        run: RunManifest | None,
        block_id: str,
        order: int,
        outcome: TrialOutcome,
        failure_class: TrialFailureClass,
    ) -> Trial:
        parameter_name, parameter_value = self._trial_parameter_value(plan, cell)
        return parse_trial(
            {
                "trial_id": cell.trial_id,
                "experiment_id": plan.experiment.experiment_id,
                "variant_id": self._variant_id(
                    plan,
                    cast(Scalar, cell.factors[plan.variant_parameter]),
                ),
                "run_id": run.run_id if run is not None else None,
                "combination_id": cell.combination_id,
                "factors": cell.factors,
                "block_id": block_id,
                "order_in_block": order,
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "attempt": 1,
                "outcome": outcome,
                "exclusion_reason": (
                    None
                    if outcome is TrialOutcome.SUCCEEDED
                    else f"capture outcome was {outcome.value}"
                ),
                "validation_status": (
                    run.validation_status if run is not None else ValidationStatus.NOT_REQUESTED
                ),
                "oracle_receipt": (
                    run.oracle_receipt.receipt
                    if run is not None and run.oracle_receipt is not None
                    else None
                ),
                "oracle_receipt_artifact_id": (
                    next(
                        (
                            artifact.artifact_id
                            for artifact in run.artifacts
                            if artifact.role == "validation_receipt"
                        ),
                        None,
                    )
                    if run is not None
                    else None
                ),
                "failure_class": failure_class,
            }
        )

    @staticmethod
    def _trial_parameter_value(
        plan: ExperimentPlan,
        cell: ExperimentCell,
    ) -> tuple[str | None, NumericValue | None]:
        """Parse the optional scalar parameter represented by this trial."""
        context_factors = tuple(name for name in cell.factors if name != plan.variant_parameter)
        parameter_name = context_factors[0] if len(context_factors) == 1 else None
        parameter_value = cell.factors[parameter_name] if parameter_name is not None else None
        return parameter_name, parse_numeric_value(parameter_value)

    @staticmethod
    def _classify_run(
        run: RunManifest,
    ) -> tuple[TrialOutcome, TrialFailureClass]:
        if (
            run.process is not None
            and run.process.resources is not None
            and run.process.resources.policy_termination is not None
        ):
            return TrialOutcome.RESOURCE_POLICY, TrialFailureClass.RESOURCE_POLICY
        if run.execution_status is ExecutionStatus.TIMED_OUT:
            return TrialOutcome.TIMED_OUT, TrialFailureClass.TIMEOUT
        if run.execution_status is ExecutionStatus.CANCELLED:
            return TrialOutcome.CANCELLED, TrialFailureClass.CANCELLATION
        if run.validation_status is ValidationStatus.INCONCLUSIVE:
            return TrialOutcome.INVALID, TrialFailureClass.ORACLE_INCONCLUSIVE
        if run.validation_status is ValidationStatus.UNSUPPORTED:
            return TrialOutcome.UNSUPPORTED, TrialFailureClass.ORACLE_UNSUPPORTED
        if run.validation_status is ValidationStatus.ERROR:
            if any(
                limitation.startswith("Oracle receipt validation failed:")
                for limitation in run.limitations
            ):
                return TrialOutcome.INVALID, TrialFailureClass.ORACLE_RECEIPT_ERROR
            return TrialOutcome.INFRASTRUCTURE_FAILED, TrialFailureClass.INFRASTRUCTURE_FAILURE
        if run.validation_status is ValidationStatus.FAILED:
            return TrialOutcome.ORACLE_FAILED, TrialFailureClass.ORACLE_FAILURE
        if run.execution_status is not ExecutionStatus.SUCCEEDED:
            return TrialOutcome.FAILED, TrialFailureClass.PROCESS_FAILURE
        return TrialOutcome.SUCCEEDED, TrialFailureClass.NONE

    def _publish_trial(self, trial: Trial) -> PublishedGeneration:
        return self.publisher.publish_rows(
            {"trials": [self._trial_row(trial)]},
            publisher="flameox.experiments",
            publisher_version="1",
            input_run_ids=((trial.run_id,) if trial.run_id is not None else ()),
        )

    def _experiment_row(self, value: Experiment) -> dict[str, object]:
        row = value.model_dump(mode="python")
        row.update(
            {
                "polarity": value.polarity,
                "metric_source": (
                    value.metric_source.value if value.metric_source is not None else None
                ),
                "measurement_series_json": (
                    canonical_json(value.measurement_series.model_dump(mode="json"))
                    if value.measurement_series is not None
                    else None
                ),
                "value_domain": (
                    value.value_domain.value if value.value_domain is not None else None
                ),
                "zero_policy": value.zero_policy.value if value.zero_policy is not None else None,
                "stopping_rule_json": json.dumps(
                    value.stopping_rule,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        )
        row.pop("stopping_rule")
        row.pop("measurement_series")
        return row

    def _variant_for_treatment(
        self,
        plan: ExperimentPlan,
        name: str,
        trials: list[Trial],
    ) -> Variant:
        cells = tuple(
            cell for block in plan.blocks for cell in block.cells if cell.treatment == name
        )
        if not cells:
            treatment_value = next(
                value
                for value in plan.factors[plan.variant_parameter]
                if self._factor_label(cast(Scalar, value)) == name
            )
            treatment_identity_id = self._treatment_identity_id(
                plan.variant_parameter,
                cast(Scalar, treatment_value),
            )
            return Variant(
                variant_id=self._variant_id(plan, cast(Scalar, treatment_value)),
                experiment_id=plan.experiment.experiment_id,
                name=name,
                treatment_factor=plan.variant_parameter,
                treatment_value=treatment_value,
                treatment_identity_id=treatment_identity_id,
                identity_quality=VariantIdentityQuality.INCOMPLETE,
                parameters={plan.variant_parameter: treatment_value},
                limitations=("No materialized experiment cell uses this treatment value.",),
            )
        factor_names = tuple(sorted(cells[0].factors))
        invariant_parameters: dict[str, JsonValue] = {}
        varying_factors: dict[str, tuple[JsonValue, ...]] = {}
        for factor_name in factor_names:
            values = self._distinct_factor_values(
                tuple(cast(Scalar, cell.factors[factor_name]) for cell in cells)
            )
            if len(values) == 1:
                invariant_parameters[factor_name] = cast(JsonValue, values[0])
            else:
                varying_factors[factor_name] = tuple(cast(JsonValue, value) for value in values)
        treatment_value = invariant_parameters.get(plan.variant_parameter)
        if not isinstance(treatment_value, str | int | float | bool):
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Treatment {name!r} does not have one exact typed treatment value.",
            )
        treatment_identity_id = self._treatment_identity_id(
            plan.variant_parameter,
            treatment_value,
        )
        runs = tuple(
            RunStore(self.workspace).read(trial.run_id)
            for trial in trials
            if trial.run_id is not None
        )
        source_state_ids = tuple(
            sorted({run.source_state_id for run in runs if run.source_state_id is not None})
        )
        workload_instance_ids = tuple(
            sorted(
                {run.workload_instance_id for run in runs if run.workload_instance_id is not None}
            )
        )
        environment_ids = tuple(sorted({run.environment_id for run in runs}))
        source_state_id = self._uniform_run_identity(
            tuple(run.source_state_id for run in runs), source_state_ids
        )
        workload_instance_id = self._uniform_run_identity(
            tuple(run.workload_instance_id for run in runs), workload_instance_ids
        )
        environment_id = self._uniform_run_identity(
            tuple(run.environment_id for run in runs), environment_ids
        )
        incomplete = (
            not runs
            or len(runs) != len(trials)
            or any(run.source_state_id is None for run in runs)
            or any(run.workload_instance_id is None for run in runs)
        )
        if incomplete:
            source_state_id = None
            workload_instance_id = None
            environment_id = None
        heterogeneous = bool(varying_factors) or any(
            len(values) > 1 for values in (source_state_ids, workload_instance_ids, environment_ids)
        )
        identity_quality = (
            VariantIdentityQuality.INCOMPLETE
            if incomplete
            else VariantIdentityQuality.HETEROGENEOUS
            if heterogeneous
            else VariantIdentityQuality.EXACT_UNIFORM
        )
        limitations: list[str] = []
        if varying_factors:
            limitations.append(
                "Treatment cohort varies across factors: " + ", ".join(sorted(varying_factors))
            )
        if len(runs) != len(trials):
            limitations.append("One or more treatment trials have no execution run identity.")
        if any(run.source_state_id is None for run in runs):
            limitations.append("One or more treatment runs lack exact source-state identity.")
        if any(run.workload_instance_id is None for run in runs):
            limitations.append("One or more treatment runs lack exact workload-instance identity.")
        return Variant(
            variant_id=self._variant_id(plan, treatment_value),
            experiment_id=plan.experiment.experiment_id,
            name=name,
            treatment_factor=plan.variant_parameter,
            treatment_value=treatment_value,
            treatment_identity_id=treatment_identity_id,
            identity_quality=identity_quality,
            source_state_id=source_state_id,
            workload_instance_id=workload_instance_id,
            environment_id=environment_id,
            source_state_ids=source_state_ids,
            workload_instance_ids=workload_instance_ids,
            environment_ids=environment_ids,
            combination_ids=tuple(dict.fromkeys(cell.combination_id for cell in cells)),
            parameters=invariant_parameters,
            varying_factors=varying_factors,
            environment_requirements={},
            limitations=tuple(limitations),
        )

    @staticmethod
    def _distinct_factor_values(values: tuple[Scalar, ...]) -> tuple[Scalar, ...]:
        result: list[Scalar] = []
        seen: set[tuple[str, object]] = set()
        for value in values:
            identity = scalar_identity(value)
            if identity not in seen:
                seen.add(identity)
                result.append(value)
        return tuple(result)

    @staticmethod
    def _treatment_identity_id(factor: str, value: Scalar) -> str:
        value_type, identity_value = scalar_identity(value)
        return digest_model(
            {
                "factor": factor,
                "value_type": value_type,
                "value": identity_value,
            }
        )

    @classmethod
    def _variant_id(cls, plan: ExperimentPlan, treatment_value: Scalar) -> str:
        return digest_model(
            {
                "experiment_id": plan.experiment.experiment_id,
                "treatment_identity_id": cls._treatment_identity_id(
                    plan.variant_parameter,
                    treatment_value,
                ),
            }
        )

    @staticmethod
    def _uniform_run_identity(
        observed: tuple[str | None, ...],
        identities: tuple[str, ...],
    ) -> str | None:
        if observed and all(value is not None for value in observed) and len(identities) == 1:
            return identities[0]
        return None

    def _variant_row(self, value: Variant) -> dict[str, object]:
        return {
            "variant_id": value.variant_id,
            "experiment_id": value.experiment_id,
            "name": value.name,
            "treatment_factor": value.treatment_factor,
            "treatment_value_json": (
                canonical_json(value.treatment_value)
                if value.treatment_factor is not None
                else None
            ),
            "treatment_identity_id": value.treatment_identity_id,
            "identity_quality": value.identity_quality.value,
            "source_state_id": value.source_state_id,
            "workload_instance_id": value.workload_instance_id,
            "environment_id": value.environment_id,
            "source_state_ids": list(value.source_state_ids),
            "workload_instance_ids": list(value.workload_instance_ids),
            "environment_ids": list(value.environment_ids),
            "combination_ids": list(value.combination_ids),
            "environment_requirements_json": canonical_json(value.environment_requirements),
            "parameters_json": canonical_json(value.parameters),
            "varying_factors_json": canonical_json(value.varying_factors),
            "limitations": list(value.limitations),
        }

    def _trial_row(self, value: Trial) -> dict[str, object]:
        row = value.model_dump(mode="python")
        parameter_value_int, parameter_value_float = numeric_value_to_columns(value.parameter_value)
        row.update(
            {
                "outcome": value.outcome.value,
                "validation_status": value.validation_status.value,
                "factors_json": canonical_json(value.factors),
                "oracle_receipt_json": (
                    canonical_json(value.oracle_receipt.model_dump(mode="json"))
                    if value.oracle_receipt is not None
                    else None
                ),
                "parameter_value_int": parameter_value_int,
                "parameter_value_float": parameter_value_float,
            }
        )
        row.pop("factors")
        row.pop("parameter_value")
        return row
