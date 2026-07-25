from __future__ import annotations

import asyncio
import json
import math
import random
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue

from flamo.adapters import PyPerfExtractor
from flamo.application.capture import CaptureService
from flamo.application.comparisons import (
    CompareRunSetsRequest,
    ComparisonResult,
    ComparisonService,
    FreezeRunSetMember,
    FreezeRunSetRequest,
    RunSetService,
)
from flamo.application.execution_policy import ExecutionPolicy
from flamo.application.workloads import Scalar, WorkloadService
from flamo.domain import (
    DomainError,
    ErrorCode,
    Experiment,
    Hypothesis,
    Investigation,
    OracleStrength,
    RunManifest,
    RunSet,
    Trial,
    TrialOutcome,
    Variant,
    digest_model,
    new_id,
)
from flamo.domain.models import utc_now
from flamo.evidence import GenerationPublisher, PublishedGeneration
from flamo.storage import JsonRecordStore, RunStore, Workspace


class ExperimentBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    block_id: str
    order: tuple[str, ...]


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    plan_id: str
    request_digest: str
    workspace_id: str
    experiment_name: str
    experiment: Experiment
    adapter: str
    execution_policy: ExecutionPolicy
    variant_parameter: str
    variants: tuple[str, ...]
    parameter_overrides: dict[str, JsonValue]
    blocks: tuple[ExperimentBlock, ...]
    workload_approval_digest: str
    experiment_config_digest: str
    created_at: datetime
    expires_at: datetime


class ExperimentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    experiment: Experiment
    variants: tuple[Variant, ...]
    trials: tuple[Trial, ...]
    run_sets: tuple[RunSet, ...]
    comparison: ComparisonResult | None
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


@dataclass(slots=True)
class _ExperimentEntry:
    plan: ExperimentPlan
    expires_monotonic: float
    consumed: bool = False


class ExperimentPlanRegistry:
    def __init__(self, *, capacity: int = 64, ttl_seconds: float = 300) -> None:
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._plans: dict[str, _ExperimentEntry] = {}
        self._lock = asyncio.Lock()

    async def issue(self, plan: ExperimentPlan) -> None:
        async with self._lock:
            self._evict()
            if len(self._plans) >= self.capacity:
                oldest = min(
                    self._plans,
                    key=lambda key: self._plans[key].expires_monotonic,
                )
                del self._plans[oldest]
            self._plans[plan.plan_id] = _ExperimentEntry(
                plan=plan,
                expires_monotonic=time.monotonic() + self.ttl_seconds,
            )

    async def consume(self, plan_id: str) -> ExperimentPlan:
        async with self._lock:
            self._evict()
            entry = self._plans.get(plan_id)
            if entry is None or entry.consumed:
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    "Experiment plan is missing, expired, or already consumed.",
                )
            entry.consumed = True
            return entry.plan

    def _evict(self) -> None:
        now = time.monotonic()
        for key in [key for key, entry in self._plans.items() if entry.expires_monotonic <= now]:
            del self._plans[key]


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
        self.plans = plans or ExperimentPlanRegistry()
        self.publisher = GenerationPublisher(workspace)
        self.experiments = JsonRecordStore(
            workspace,
            kind="experiments",
            model=Experiment,
            id_field="experiment_id",
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
        if config.design != "randomized_complete_blocks":
            raise DomainError(
                ErrorCode.COMPARISON_INVALID,
                "Initial experiment execution requires randomized complete blocks.",
            )
        investigations = JsonRecordStore(
            self.workspace,
            kind="investigations",
            model=Investigation,
            id_field="investigation_id",
        )
        investigations.read(investigation_id)
        if hypothesis_id is not None:
            hypotheses = JsonRecordStore(
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
        variant_parameters = [
            name
            for name, choices in workload.parameters.items()
            if set(config.variants).issubset(set(choices))
        ]
        if not variant_parameters:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Experiment variants must be declared choices of one workload parameter.",
            )
        if len(variant_parameters) > 1:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Experiment variants ambiguously match more than one workload parameter.",
                details={"parameters": variant_parameters},
            )
        variant_parameter = variant_parameters[0]
        supplied_overrides = parameter_overrides or {}
        if variant_parameter in supplied_overrides:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Experiment parameter overrides cannot replace the treatment parameter.",
                details={"parameter": variant_parameter},
            )
        for variant in config.variants:
            self.workloads.resolve(
                config.workload,
                {**supplied_overrides, variant_parameter: variant},
                require_approval=execution_policy.requires_workload_approval,
            )
        definition = self.workloads.definition(config.workload)
        if (
            execution_policy.requires_workload_approval
            and definition.approved_definition_digest is None
        ):
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Workload {config.workload!r} is not approved.",
            )
        oracle = workload.oracle
        role: Literal["exploratory", "confirmatory"] = (
            "confirmatory"
            if oracle is not None and oracle.strength is OracleStrength.CROSS_TREATMENT_EQUIVALENCE
            else "exploratory"
        )
        design = {
            "name": experiment_name,
            "config": config.model_dump(mode="json"),
            "parameters": supplied_overrides,
        }
        experiment = Experiment(
            experiment_id=new_id(),
            investigation_id=investigation_id,
            hypothesis_id=hypothesis_id,
            recipe="compare_run_sets",
            recipe_version="1",
            workload_definition_id=definition.workload_definition_id,
            experiment_design_id=digest_model(design),
            measurement_protocol_id=digest_model({"adapter": adapter}),
            validation_spec_id=definition.validation_spec_id,
            primary_metric=config.primary_metric,
            polarity=config.polarity,
            estimand=config.estimand,
            practical_threshold=config.practical_threshold,
            confidence_level=config.confidence_level,
            stopping_rule={"fixed_blocks": config.blocks},
            random_seed=config.random_seed,
            role=role,
        )
        generator = random.Random(config.random_seed)
        blocks: list[ExperimentBlock] = []
        for index in range(config.blocks):
            order = list(config.variants)
            generator.shuffle(order)
            blocks.append(
                ExperimentBlock(
                    block_id=f"block-{index + 1:04d}",
                    order=tuple(order),
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
            "variants": config.variants,
            "parameters": overrides,
            "blocks": [block.model_dump(mode="json") for block in blocks],
            "approval": (
                definition.approved_definition_digest or definition.workload_definition_id
            ),
            "experiment_config_digest": digest_model(config.model_dump(mode="json")),
        }
        plan = ExperimentPlan(
            plan_id=plan_id,
            request_digest=digest_model(request),
            workspace_id=self.workspace.identity.workspace_id,
            experiment_name=experiment_name,
            experiment=experiment,
            adapter=adapter,
            execution_policy=execution_policy,
            variant_parameter=variant_parameter,
            variants=config.variants,
            parameter_overrides=overrides,
            blocks=tuple(blocks),
            workload_approval_digest=(
                definition.approved_definition_digest or definition.workload_definition_id
            ),
            experiment_config_digest=digest_model(config.model_dump(mode="json")),
            created_at=created,
            expires_at=created + timedelta(seconds=self.plans.ttl_seconds),
        )
        await self.plans.issue(plan)
        return plan

    async def run(self, plan_id: str) -> ExperimentRunResult:
        plan = await self.plans.consume(plan_id)
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
        current_approval = (
            definition.approved_definition_digest or definition.workload_definition_id
        )
        if current_approval != plan.workload_approval_digest:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Workload approval changed after experiment planning.",
            )
        # Persist the predeclared protocol before the first treatment starts. If
        # execution is interrupted, the investigation still records what was
        # intended rather than leaving an unattributed run population.
        self.experiments.create(plan.experiment)
        published = self.publisher.publish_rows(
            {"experiments": [self._experiment_row(plan.experiment)]},
            publisher="flamo.experiments",
            publisher_version="1",
        )
        trials: list[Trial] = []
        trials_by_variant: dict[str, list[Trial]] = {name: [] for name in plan.variants}
        run_by_variant: dict[str, RunManifest] = {}
        for block in plan.blocks:
            for order, variant_name in enumerate(block.order):
                parameters = {
                    **cast(dict[str, Scalar], plan.parameter_overrides),
                    plan.variant_parameter: variant_name,
                }
                capture_plan = await self.captures.plan(
                    workload_name=config.workload,
                    adapter=plan.adapter,
                    parameters=parameters,
                    execution_policy=plan.execution_policy,
                )
                try:
                    captured = await self.captures.execute(capture_plan.plan_id)
                    run = captured.run
                    outcome = (
                        TrialOutcome.SUCCEEDED
                        if run.execution_status.value == "succeeded"
                        else TrialOutcome.FAILED
                    )
                    if outcome is TrialOutcome.SUCCEEDED and plan.adapter == "pyperf":
                        PyPerfExtractor(self.workspace).extract(run.run_id)
                except asyncio.CancelledError as cancellation:
                    run = RunStore(self.workspace).read(capture_plan.run_id)
                    trial = self._make_trial(
                        plan=plan,
                        variant_name=variant_name,
                        run=run,
                        block_id=block.block_id,
                        order=order,
                        outcome=TrialOutcome.CANCELLED,
                    )
                    try:
                        self._publish_trial(trial)
                    finally:
                        raise cancellation
                except DomainError as error:
                    if error.run_id is None:
                        raise
                    run = RunStore(self.workspace).read(error.run_id)
                    outcome = (
                        TrialOutcome.TIMED_OUT
                        if run.execution_status.value == "timed_out"
                        else TrialOutcome.FAILED
                    )
                run_by_variant.setdefault(variant_name, run)
                trial = self._make_trial(
                    plan=plan,
                    variant_name=variant_name,
                    run=run,
                    block_id=block.block_id,
                    order=order,
                    outcome=outcome,
                )
                trials.append(trial)
                trials_by_variant[variant_name].append(trial)
                published = self._publish_trial(trial)
        variants: list[Variant] = []
        for name in plan.variants:
            run = run_by_variant[name]
            source_state_id = run.source_state_id
            workload_instance_id = run.workload_instance_id
            if source_state_id is None or workload_instance_id is None:
                raise DomainError(
                    ErrorCode.COMPARISON_INVALID,
                    "Experiment run lacks source or workload identity.",
                )
            variants.append(
                Variant(
                    variant_id=digest_model(
                        {
                            "experiment_id": plan.experiment.experiment_id,
                            "name": name,
                        }
                    ),
                    experiment_id=plan.experiment.experiment_id,
                    name=name,
                    source_state_id=source_state_id,
                    workload_instance_id=workload_instance_id,
                    parameters={plan.variant_parameter: name},
                )
            )
        published = self.publisher.publish_rows(
            {
                "variants": [self._variant_row(value) for value in variants],
            },
            publisher="flamo.experiments",
            publisher_version="1",
            input_run_ids=tuple(trial.run_id for trial in trials),
        )
        run_sets = tuple(
            RunSetService(self.workspace).freeze(
                FreezeRunSetRequest(
                    members=tuple(
                        FreezeRunSetMember(
                            run_id=trial.run_id,
                            trial_id=trial.trial_id,
                            included=trial.outcome is TrialOutcome.SUCCEEDED,
                            reason=trial.exclusion_reason,
                        )
                        for trial in trials_by_variant[name]
                    ),
                    selection={
                        "experiment_id": plan.experiment.experiment_id,
                        "variant": name,
                    },
                )
            )
            for name in plan.variants
        )
        comparison: ComparisonResult | None = None
        limitations: list[str] = []
        if len(run_sets) != 2:
            limitations.append(
                "Automatic paired comparison currently requires exactly two variants."
            )
        elif plan.adapter != "pyperf":
            limitations.append(
                "Automatic experiment comparison currently requires pyperf measurements."
            )
        else:
            comparison = ComparisonService(self.workspace).record(
                CompareRunSetsRequest(
                    baseline_run_set_id=run_sets[0].run_set_id,
                    candidate_run_set_id=run_sets[1].run_set_id,
                    experiment_id=plan.experiment.experiment_id,
                    metric=plan.experiment.primary_metric,
                    unit="ns",
                    polarity=plan.experiment.polarity,
                    practical_threshold=plan.experiment.practical_threshold,
                    confidence_level=plan.experiment.confidence_level,
                    random_seed=plan.experiment.random_seed,
                )
            )
        result_commit_id = published.commit.commit_id
        if comparison is not None:
            result_commit_id = (
                comparison.materialized_commit_id or comparison.corpus_commit_id
            )
        return ExperimentRunResult(
            experiment=plan.experiment,
            variants=tuple(variants),
            trials=tuple(trials),
            run_sets=run_sets,
            comparison=comparison,
            corpus_commit_id=result_commit_id,
            limitations=tuple(limitations),
        )

    def _make_trial(
        self,
        *,
        plan: ExperimentPlan,
        variant_name: str,
        run: RunManifest,
        block_id: str,
        order: int,
        outcome: TrialOutcome,
    ) -> Trial:
        parameter_int: int | None = None
        parameter_float: float | None = None
        try:
            parameter_int = int(variant_name)
        except ValueError:
            try:
                parameter_float = float(variant_name)
                if not math.isfinite(parameter_float):
                    parameter_float = None
            except ValueError:
                pass
        return Trial(
            trial_id=new_id(),
            experiment_id=plan.experiment.experiment_id,
            variant_id=digest_model(
                {
                    "experiment_id": plan.experiment.experiment_id,
                    "name": variant_name,
                }
            ),
            run_id=run.run_id,
            block_id=block_id,
            order_in_block=order,
            parameter_name=plan.variant_parameter,
            parameter_value_int=parameter_int,
            parameter_value_float=parameter_float,
            attempt=1,
            outcome=outcome,
            exclusion_reason=(
                None
                if outcome is TrialOutcome.SUCCEEDED
                else f"capture outcome was {outcome.value}"
            ),
            validation_status=run.validation_status,
        )

    def _publish_trial(self, trial: Trial) -> PublishedGeneration:
        return self.publisher.publish_rows(
            {"trials": [self._trial_row(trial)]},
            publisher="flamo.experiments",
            publisher_version="1",
            input_run_ids=(trial.run_id,),
        )

    def _experiment_row(self, value: Experiment) -> dict[str, object]:
        row = value.model_dump(mode="python")
        row.update(
            {
                "polarity": value.polarity,
                "stopping_rule_json": json.dumps(
                    value.stopping_rule,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        )
        row.pop("stopping_rule")
        row.pop("schema_version")
        return row

    def _variant_row(self, value: Variant) -> dict[str, object]:
        return {
            "variant_id": value.variant_id,
            "experiment_id": value.experiment_id,
            "name": value.name,
            "source_state_id": value.source_state_id,
            "workload_instance_id": value.workload_instance_id,
            "environment_requirements_json": json.dumps(
                value.environment_requirements,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "parameters_json": json.dumps(
                value.parameters,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }

    def _trial_row(self, value: Trial) -> dict[str, object]:
        row = value.model_dump(mode="python")
        row.update(
            {
                "outcome": value.outcome.value,
                "validation_status": value.validation_status.value,
            }
        )
        row.pop("schema_version")
        return row
