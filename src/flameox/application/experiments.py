from __future__ import annotations

import asyncio
import json
import random
import secrets
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from functools import partial
from itertools import product
from typing import Any, Literal, cast

from pydantic import ConfigDict, Field, JsonValue, TypeAdapter, computed_field, model_validator

from flameox.adapters import PyPerfExtractor, PytestExtractor, PythonStartupExtractor
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
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.workloads import (
    ExperimentConfig,
    Scalar,
    WorkloadService,
    _FactorExperimentConfig,
    _OutcomeExperimentConfig,
    _ScaledLegacyExperimentConfig,
    scalar_contains,
    scalar_equal,
    scalar_identity_set,
    scalar_subset,
)
from flameox.catalog import Catalog
from flameox.domain import (
    ArtifactKind,
    CursorCodec,
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
from flameox.storage import AuthorizedPlanStore, JsonRecordStore, RunStore, Workspace


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
    schema_version: int = 1
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
    schema_version: Literal[2] = 2
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

    schema_version: Literal[2] = 2
    experiment_id: str
    trials: tuple[Trial, ...]
    next_cursor: str | None = None


MAX_TRIAL_PAGE_SIZE = 1_000
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


class OutcomeFirstFailure(ContractModel):
    trial_id: str
    factors: dict[str, JsonValue]


def _advertise_first_failure_projections(schema: dict[str, Any]) -> None:
    properties = schema.setdefault("properties", {})
    assert isinstance(properties, dict)
    properties.pop("first_failure", None)
    properties.update(
        {
            "first_failure_trial_id": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "First Failure Trial Id",
            },
            "first_failure_factors": {
                "anyOf": [
                    {
                        "additionalProperties": {"$ref": "#/$defs/JsonValue"},
                        "type": "object",
                    },
                    {"type": "null"},
                ],
                "default": None,
                "title": "First Failure Factors",
            },
        }
    )
    required = schema.setdefault("required", [])
    assert isinstance(required, list)
    for field_name in (
        "first_failure",
        "first_failure_trial_id",
        "first_failure_factors",
    ):
        if field_name in required:
            required.remove(field_name)


class OutcomeExperimentResult(ContractModel):
    model_config = ConfigDict(json_schema_extra=_advertise_first_failure_projections)

    schema_version: Literal[1] = 1
    experiment_id: str
    method: Literal[ExperimentOutcomeMethod.FIXED_ATTEMPTS_V1] = (
        ExperimentOutcomeMethod.FIXED_ATTEMPTS_V1
    )
    goal: ExperimentOutcomeGoal
    disposition: ExperimentOutcomeDisposition
    counts: tuple[OutcomeCount, ...]
    complete_pairs: int
    unmatched_cells: int
    first_failure: OutcomeFirstFailure | None = Field(default=None, exclude=True)
    limitations: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def parse_first_failure_projections(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        has_trial = "first_failure_trial_id" in value
        has_factors = "first_failure_factors" in value
        if not has_trial and not has_factors:
            return value
        if "first_failure" in value:
            raise ValueError("use either first_failure or flattened first-failure fields")
        if has_trial != has_factors:
            raise ValueError("first-failure trial and factors must appear together")
        parsed = dict(value)
        trial_id = parsed.pop("first_failure_trial_id")
        factors = parsed.pop("first_failure_factors")
        if (trial_id is None) != (factors is None):
            raise ValueError("first-failure trial and factors must appear together")
        parsed["first_failure"] = (
            None if trial_id is None else {"trial_id": trial_id, "factors": factors}
        )
        return parsed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def first_failure_trial_id(self) -> str | None:
        return self.first_failure.trial_id if self.first_failure is not None else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def first_failure_factors(self) -> dict[str, JsonValue] | None:
        return self.first_failure.factors if self.first_failure is not None else None


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
        self.experiments = JsonRecordStore(
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
            position = CursorCodec.decode(
                cursor,
                namespace="experiment-trials",
                snapshot_id=head.commit_id,
                scope_digest=scope_digest,
            )
            if len(position) != 1 or not isinstance(position[0], int):
                raise DomainError(ErrorCode.STALE_CURSOR, "Trial cursor position is invalid.")
            offset = position[0]
            if offset < 0:
                raise DomainError(ErrorCode.STALE_CURSOR, "Trial cursor position is invalid.")
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
            CursorCodec.encode(
                namespace="experiment-trials",
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
                "method": ExperimentOutcomeMethod.FIXED_ATTEMPTS_V1,
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
                if isinstance(config, _FactorExperimentConfig) and config.baseline_value is not None
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

        async def report(message: str) -> None:
            if progress is not None:
                await progress(completed, total_phases, message)

        await report("Experiment plan consumed")
        config = self._validate_plan(plan)
        completed += 1
        await report("Experiment plan and workload definition validated")
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
                    environment_requirements={},
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
                    FreezeRunMembersRequest(
                        members=tuple(
                            _freeze_trial_member(trial)
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
        if isinstance(config, _OutcomeExperimentConfig):
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
        elif plan.metric_source is MetricSource.MEASUREMENT and plan.adapter != "pyperf":
            limitations.append(
                "Automatic experiment comparison currently requires pyperf measurements."
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
                                "unit": (
                                    "bytes"
                                    if plan.metric_source is MetricSource.RUNTIME_RESOURCE
                                    else "ns"
                                ),
                                "metric_source": plan.metric_source,
                                "polarity": plan.experiment.polarity,
                                "practical_threshold": plan.experiment.practical_threshold,
                                "confidence_level": plan.experiment.confidence_level,
                                "random_seed": plan.experiment.random_seed,
                            }
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
        if isinstance(config, _FactorExperimentConfig):
            treatment_factor = config.treatment_factor
            factors = dict(config.factors)
        else:
            matches = [
                name
                for name, choices in workload_parameters.items()
                if scalar_subset(list(config.variants), list(choices))
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
            if isinstance(config, _ScaledLegacyExperimentConfig):
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
            "Clean trials bound only the declared fixed attempts; they do not prove "
            "absence of rare failures or race freedom."
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
        failure_class: TrialFailureClass,
    ) -> Trial:
        parameter_name, parameter_value = self._trial_parameter_value(plan, cell)
        return parse_trial(
            {
                "trial_id": cell.trial_id,
                "experiment_id": plan.experiment.experiment_id,
                "variant_id": digest_model(
                    {
                        "experiment_id": plan.experiment.experiment_id,
                        "name": cell.treatment,
                    }
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
        row.pop("schema_version")
        return row
