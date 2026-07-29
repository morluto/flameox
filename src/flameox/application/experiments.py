from __future__ import annotations

import asyncio
import json
import math
import random
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from itertools import product
from typing import Literal, cast

from pydantic import Field, JsonValue

from flameox.adapters import PyPerfExtractor, PytestExtractor, PythonStartupExtractor
from flameox.application.async_work import run_atomic_thread
from flameox.application.capture import CaptureService
from flameox.application.comparisons import (
    CompareRunSetsRequest,
    ComparisonResult,
    ComparisonService,
    FreezeRunSetMember,
    FreezeRunSetRequest,
    RunSetService,
)
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.workloads import ExperimentConfig, Scalar, WorkloadService
from flameox.domain import (
    ArtifactKind,
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
    ValidationStatus,
    Variant,
    canonical_json,
    digest_model,
    new_id,
)
from flameox.domain.models import utc_now
from flameox.evidence import GenerationPublisher, PublishedGeneration
from flameox.models import ContractModel
from flameox.storage import JsonRecordStore, RunStore, Workspace


def _extract_adapter_measurements(workspace: Workspace, adapter: str, run_id: str) -> None:
    if adapter == "pyperf":
        PyPerfExtractor(workspace).extract(run_id)
    elif adapter == "python-startup":
        PythonStartupExtractor(workspace).extract(run_id)
    elif adapter == "pytest":
        PytestExtractor(workspace).extract(run_id)


def _has_extractable_artifact(run: RunManifest, adapter: str) -> bool:
    expected = {
        "pyperf": ArtifactKind.BENCHMARK_SAMPLES,
        "python-startup": ArtifactKind.PYTHON_STARTUP,
        "pytest": ArtifactKind.TEST_EXECUTION,
    }.get(adapter)
    return expected is not None and any(item.kind is expected for item in run.artifacts)


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
    factors: dict[str, tuple[JsonValue, ...]] = Field(default_factory=dict)
    parameter_overrides: dict[str, JsonValue]
    blocks: tuple[ExperimentBlock, ...]
    workload_approval_digest: str
    experiment_config_digest: str
    created_at: datetime
    expires_at: datetime


class ExperimentRunResult(ContractModel):
    schema_version: int = 1
    experiment: Experiment
    variants: tuple[Variant, ...]
    trials: tuple[Trial, ...]
    run_sets: tuple[RunSet, ...]
    comparison: ComparisonResult | None
    outcome: OutcomeExperimentResult | None = None
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


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
    pass_rate: float | None = None
    failure_rate: float | None = None


class OutcomeExperimentResult(ContractModel):
    schema_version: Literal[1] = 1
    experiment_id: str
    method: Literal["fixed_attempts_v1"] = "fixed_attempts_v1"
    goal: Literal["equivalence", "absence_of_failure", "bounded_rate"]
    disposition: Literal[
        "all_clean",
        "base_only_failure",
        "candidate_only_failure",
        "mixed",
        "unsupported",
        "insufficient_evidence",
    ]
    counts: tuple[OutcomeCount, ...]
    complete_pairs: int
    unmatched_cells: int
    first_failure_trial_id: str | None = None
    first_failure_factors: dict[str, JsonValue] | None = None
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
            recipe=("categorical_outcomes" if config.analysis == "outcome" else "compare_run_sets"),
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
            stopping_rule={
                "method": "fixed_attempts_v1",
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
            variants=variants,
            factors={
                name: tuple(cast(JsonValue, value) for value in values)
                for name, values in factors.items()
            },
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

    async def run(
        self,
        plan_id: str,
        *,
        progress: Callable[[float, float, str], Awaitable[None]] | None = None,
    ) -> ExperimentRunResult:
        plan = await self.plans.consume(plan_id)
        trial_count = sum(len(block.cells) for block in plan.blocks)
        total_phases = trial_count + 4
        completed = 0

        async def report(message: str) -> None:
            if progress is not None:
                await progress(completed, total_phases, message)

        await report("Experiment plan consumed")
        config = self._validate_plan(plan)
        completed += 1
        await report("Experiment plan and workload approval validated")
        # Persist the predeclared protocol before the first treatment starts. If
        # execution is interrupted, the investigation still records what was
        # intended rather than leaving an unattributed run population.
        self.experiments.create(plan.experiment)
        published = await run_atomic_thread(
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
        run_by_variant: dict[str, RunManifest] = {}
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
            failure_class: Literal[
                "none",
                "unattempted",
                "oracle_failure",
                "process_failure",
                "timeout",
                "cancellation",
                "unsupported_environment",
                "resource_policy",
                "infrastructure_failure",
            ]
            try:
                capture_plan = await self.captures.plan(
                    workload_name=config.workload,
                    adapter=plan.adapter,
                    parameters=parameters,
                    execution_policy=plan.execution_policy,
                )
                captured = await self.captures.execute(capture_plan.plan_id)
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
                    and (self.workspace.paths.runs / capture_plan.run_id).exists()
                    else None
                )
                trial = self._make_trial(
                    plan=plan,
                    cell=cell,
                    run=run,
                    block_id=block.block_id,
                    order=order,
                    outcome=TrialOutcome.CANCELLED,
                    failure_class="cancellation",
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
                    if config.analysis != "outcome":
                        failed = self._make_trial(
                            plan=plan,
                            cell=cell,
                            run=None,
                            block_id=block.block_id,
                            order=order,
                            outcome=TrialOutcome.INFRASTRUCTURE_FAILED,
                            failure_class="infrastructure_failure",
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
                        "unsupported_environment"
                        if outcome is TrialOutcome.UNSUPPORTED
                        else "infrastructure_failure"
                    )
                else:
                    run = RunStore(self.workspace).read(error.run_id)
                    outcome, failure_class = self._classify_run(run)
            if run is not None:
                run_by_variant.setdefault(variant_name, run)
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
            published = await run_atomic_thread(partial(self._publish_trial, trial))
            completed += 1
            await report(
                f"Trial {completed - 2}/{trial_count} published ({block.block_id}, {variant_name})"
            )
        variants: list[Variant] = []
        for name in plan.variants:
            run = run_by_variant.get(name)
            factor_values = next(
                (
                    cell.factors
                    for block in plan.blocks
                    for cell in block.cells
                    if cell.treatment == name
                ),
                {
                    plan.variant_parameter: next(
                        value
                        for value in plan.factors[plan.variant_parameter]
                        if self._factor_label(cast(Scalar, value)) == name
                    )
                },
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
                    source_state_id=run.source_state_id if run is not None else None,
                    workload_instance_id=(run.workload_instance_id if run is not None else None),
                    parameters=factor_values,
                    # A scaling variant spans several workload instances; its
                    # declared parameter family remains explicit here.
                    environment_requirements=(
                        {
                            "scaling_parameter": config.scaling_parameter,
                            "scaling_values": list(config.scaling_values),
                        }
                        if config.scaling_parameter is not None
                        else {}
                    ),
                )
            )
        published = await run_atomic_thread(
            lambda: self.publisher.publish_rows(
                {
                    "variants": [self._variant_row(value) for value in variants],
                },
                publisher="flameox.experiments",
                publisher_version="1",
                input_run_ids=tuple(trial.run_id for trial in trials if trial.run_id is not None),
            )
        )
        run_sets = await run_atomic_thread(
            lambda: tuple(
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
                            if trial.run_id is not None
                        ),
                        selection={
                            "experiment_id": plan.experiment.experiment_id,
                            "variant": name,
                        },
                    )
                )
                for name in plan.variants
                if any(trial.run_id is not None for trial in trials_by_variant[name])
            )
        )
        completed += 1
        await report("Variants and frozen run sets published")
        comparison: ComparisonResult | None = None
        outcome_result: OutcomeExperimentResult | None = None
        limitations: list[str] = []
        if config.analysis == "outcome":
            outcome_result = self._outcome_result(plan, config, trials)
            published = await run_atomic_thread(
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
        elif plan.adapter != "pyperf":
            limitations.append(
                "Automatic experiment comparison currently requires pyperf measurements."
            )
        else:
            comparison = await run_atomic_thread(
                lambda: ComparisonService(self.workspace).record(
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
            )
        result_commit_id = published.commit.commit_id
        if comparison is not None:
            result_commit_id = comparison.materialized_commit_id or comparison.corpus_commit_id
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
                failure_class="unattempted",
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
        current_approval = (
            definition.approved_definition_digest or definition.workload_definition_id
        )
        if current_approval != plan.workload_approval_digest:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Workload approval changed after experiment planning.",
            )
        return config

    def _materialize_combinations(
        self,
        config: ExperimentConfig,
        workload_parameters: dict[str, tuple[Scalar, ...]],
    ) -> tuple[str, dict[str, tuple[Scalar, ...]], tuple[dict[str, Scalar], ...]]:
        if config.factors:
            factors = dict(config.factors)
            assert config.treatment_factor is not None
            treatment_factor = config.treatment_factor
        else:
            matches = [
                name
                for name, choices in workload_parameters.items()
                if set(config.variants).issubset(set(choices))
            ]
            if not matches:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "Experiment variants must be declared choices of one workload parameter.",
                )
            if len(matches) > 1:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "Experiment variants ambiguously match more than one workload parameter.",
                    details={"parameters": matches},
                )
            treatment_factor = matches[0]
            factors = {treatment_factor: config.variants}
            if config.scaling_parameter is not None:
                if config.scaling_parameter == treatment_factor:
                    raise DomainError(
                        ErrorCode.WORKSPACE_INVALID,
                        "The scaling parameter must differ from the treatment parameter.",
                    )
                factors[config.scaling_parameter] = config.scaling_values

        for name, values in factors.items():
            allowed = workload_parameters.get(name)
            if allowed is None:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    f"Experiment factor {name!r} is not a workload parameter.",
                )
            if len(set(values)) != len(values) or not set(values).issubset(set(allowed)):
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
            if any(combination[name] not in factors[name] for name in factor_names):
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
            if any(value not in factors[name] for name, value in rule.items()):
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "Exclusion contains an undeclared factor value.",
                )
        filtered = tuple(
            combination
            for combination in combinations
            if not any(
                all(combination[name] == value for name, value in rule.items())
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
        config: ExperimentConfig,
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
                if trial.factors.get(plan.variant_parameter) == treatment_value
            ]
            attempted = sum(trial.outcome is not TrialOutcome.UNATTEMPTED for trial in selected)
            eligible = sum(
                trial.failure_class
                not in {
                    "unattempted",
                    "cancellation",
                    "unsupported_environment",
                    "infrastructure_failure",
                }
                for trial in selected
            )
            passed = sum(trial.outcome is TrialOutcome.SUCCEEDED for trial in selected)
            failed = sum(
                trial.failure_class
                in {"oracle_failure", "process_failure", "timeout", "resource_policy"}
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
                    infrastructure_failed=sum(
                        trial.failure_class == "infrastructure_failure" for trial in selected
                    ),
                    pass_rate=passed / eligible if eligible else None,
                    failure_rate=failed / eligible if eligible else None,
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
            "Clean trials bound only the declared fixed attempts; they do not prove "
            "absence of rare failures or race freedom."
        ]
        if unmatched:
            limitations.append("One or more pairing coordinates lack every treatment.")
        if counts and all(
            item.unsupported == item.attempted and item.attempted > 0 for item in counts
        ):
            disposition = "unsupported"
        elif any(item.eligible < minimum for item in counts):
            disposition = "insufficient_evidence"
        elif not failures:
            disposition = "all_clean"
        elif len(plan.variants) == 2 and failed_treatments == {plan.variants[0]}:
            disposition = "base_only_failure"
        elif len(plan.variants) == 2 and failed_treatments == {plan.variants[1]}:
            disposition = "candidate_only_failure"
        else:
            disposition = "mixed"
        first_failure = failures[0] if failures else None
        assert config.outcome_goal is not None
        return OutcomeExperimentResult(
            experiment_id=plan.experiment.experiment_id,
            goal=config.outcome_goal,
            disposition=cast(
                Literal[
                    "all_clean",
                    "base_only_failure",
                    "candidate_only_failure",
                    "mixed",
                    "unsupported",
                    "insufficient_evidence",
                ],
                disposition,
            ),
            counts=tuple(counts),
            complete_pairs=complete_pairs,
            unmatched_cells=unmatched,
            first_failure_trial_id=(first_failure.trial_id if first_failure is not None else None),
            first_failure_factors=(first_failure.factors if first_failure is not None else None),
            limitations=tuple(limitations),
        )

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
            "first_failure_trial_id": value.first_failure_trial_id,
            "first_failure_factors_json": (
                json.dumps(
                    value.first_failure_factors,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                if value.first_failure_factors is not None
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
        failure_class: Literal[
            "none",
            "unattempted",
            "oracle_failure",
            "process_failure",
            "timeout",
            "cancellation",
            "unsupported_environment",
            "resource_policy",
            "infrastructure_failure",
        ],
    ) -> Trial:
        parameter_name, parameter_int, parameter_float = self._legacy_parameter_projection(
            plan, cell
        )
        return Trial(
            trial_id=cell.trial_id,
            experiment_id=plan.experiment.experiment_id,
            variant_id=digest_model(
                {
                    "experiment_id": plan.experiment.experiment_id,
                    "name": cell.treatment,
                }
            ),
            run_id=run.run_id if run is not None else None,
            combination_id=cell.combination_id,
            factors=cell.factors,
            block_id=block_id,
            order_in_block=order,
            parameter_name=parameter_name,
            parameter_value_int=parameter_int,
            parameter_value_float=parameter_float,
            attempt=1,
            outcome=outcome,
            exclusion_reason=(
                None
                if outcome is TrialOutcome.SUCCEEDED
                else f"capture outcome was {outcome.value}"
            ),
            validation_status=(
                run.validation_status if run is not None else ValidationStatus.NOT_REQUESTED
            ),
            failure_class=failure_class,
        )

    @staticmethod
    def _legacy_parameter_projection(
        plan: ExperimentPlan,
        cell: ExperimentCell,
    ) -> tuple[str | None, int | None, float | None]:
        """Project one context factor for readers of the pre-factor trial schema."""
        context_factors = tuple(name for name in cell.factors if name != plan.variant_parameter)
        parameter_name = context_factors[0] if len(context_factors) == 1 else None
        parameter_value = cell.factors[parameter_name] if parameter_name is not None else None
        parameter_int: int | None = None
        parameter_float: float | None = None
        if isinstance(parameter_value, int) and not isinstance(parameter_value, bool):
            parameter_int = parameter_value
        elif isinstance(parameter_value, float):
            parameter_float = parameter_value if math.isfinite(parameter_value) else None
        elif isinstance(parameter_value, str):
            try:
                parameter_int = int(parameter_value)
            except ValueError:
                try:
                    parsed_float = float(parameter_value)
                except ValueError:
                    parsed_float = None
                parameter_float = (
                    parsed_float
                    if parsed_float is not None and math.isfinite(parsed_float)
                    else None
                )
        return parameter_name, parameter_int, parameter_float

    @staticmethod
    def _classify_run(
        run: RunManifest,
    ) -> tuple[
        TrialOutcome,
        Literal[
            "none",
            "oracle_failure",
            "process_failure",
            "timeout",
            "cancellation",
            "resource_policy",
        ],
    ]:
        if (
            run.process is not None
            and run.process.resources is not None
            and run.process.resources.policy_termination is not None
        ):
            return TrialOutcome.RESOURCE_POLICY, "resource_policy"
        if run.execution_status.value == "timed_out":
            return TrialOutcome.TIMED_OUT, "timeout"
        if run.execution_status.value == "cancelled":
            return TrialOutcome.CANCELLED, "cancellation"
        if run.validation_status.value in {"failed", "error"}:
            return TrialOutcome.ORACLE_FAILED, "oracle_failure"
        if run.execution_status.value != "succeeded":
            return TrialOutcome.FAILED, "process_failure"
        return TrialOutcome.SUCCEEDED, "none"

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
            "environment_requirements_json": canonical_json(value.environment_requirements),
            "parameters_json": canonical_json(value.parameters),
        }

    def _trial_row(self, value: Trial) -> dict[str, object]:
        row = value.model_dump(mode="python")
        row.update(
            {
                "outcome": value.outcome.value,
                "validation_status": value.validation_status.value,
                "factors_json": canonical_json(value.factors),
            }
        )
        row.pop("factors")
        row.pop("schema_version")
        return row
