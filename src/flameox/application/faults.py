from __future__ import annotations

import asyncio
import json
import random
import secrets
import shutil
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from flameox.action_graph import ActionId, ManualAction, manual_action
from flameox.adapters.toxiproxy import ToxiproxyApiError, ToxiproxyClient, ToxiproxyToolManager
from flameox.application.async_work import run_atomic_thread
from flameox.application.capture import CaptureService
from flameox.application.comparisons import (
    ComparisonResult,
    ComparisonService,
    MeasurementCompareRunSetsRequest,
)
from flameox.application.environment import collect_environment
from flameox.application.evidence_rows import (
    artifact_registration_row,
    environment_row,
    process_observation_coverage,
    process_observation_rows,
    source_state_row,
)
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.experiments import (
    ExperimentBlock,
    ExperimentCell,
    ExperimentPlan,
    ExperimentService,
)
from flameox.application.progress import ProgressReporter
from flameox.application.run_rows import run_row
from flameox.application.source import collect_partial_source_state
from flameox.application.workloads import (
    BandwidthFault,
    FaultExperimentConfig,
    FaultScenario,
    LatencyFault,
    LimitDataFault,
    ProxyFault,
    ResetPeerFault,
    SlicerFault,
    SlowCloseFault,
    TimeoutFault,
    WorkloadService,
)
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    CapturePlan,
    CaptureStatus,
    ComparisonValidity,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    Experiment,
    ExperimentOutcomeMethod,
    ExperimentRole,
    Hypothesis,
    Investigation,
    MeasurementSeriesSelector,
    MetricPolarity,
    MetricSource,
    MetricValueDomain,
    MetricZeroPolicy,
    RunManifest,
    RunSemantics,
    RunSet,
    Sensitivity,
    Trial,
    TrialFailureClass,
    TrialOutcome,
    ValidationStatus,
    Variant,
    digest_model,
    new_id,
)
from flameox.domain.models import Digest, ExecutionRunManifest
from flameox.evidence import GenerationPublisher
from flameox.execution import (
    ManagedSidecarLease,
    ManagedSidecarOutcome,
    ManagedSidecarStartupCancelled,
    ManagedSidecarStartupError,
    SubprocessBroker,
)
from flameox.models import ContractModel
from flameox.storage import (
    ArtifactStore,
    AuthorizedPlanStore,
    ControlRecordStore,
    RunStore,
    Workspace,
)

_BASELINE = "baseline"
_FAULT_CLIENT_ELAPSED = "fault.client_elapsed"
_FAULT_CLIENT_ELAPSED_UNIT = "ns"
type FaultFailurePhase = Literal[
    "sidecar_readiness",
    "proxy_creation",
    "treatment_configuration",
    "workload_planning",
    "capture_execution",
    "measurement_receipt",
]


class FaultExperimentPlan(ContractModel):
    plan_token: str
    plan_id: Digest
    workspace_id: str
    experiment_name: str
    experiment_plan: ExperimentPlan
    fault_config: dict[str, JsonValue]
    endpoint_parameter: str
    upstream_host: str
    upstream_port: int
    endpoint_template: str
    scenarios: dict[str, dict[str, JsonValue]]
    parameter_overrides: dict[str, JsonValue] = Field(default_factory=dict)
    tool_version: str
    tool_asset: str
    tool_digest: str
    tool_executable_digest: str
    tool_manifest_revision: str
    containment: Literal["managed_process_group"] = "managed_process_group"
    workload_containment: ExecutionPolicy = ExecutionPolicy.TRUSTED_LOCAL
    limitations: tuple[str, ...] = ()


class FaultTrialDiagnostic(ContractModel):
    phase: FaultFailurePhase
    error_code: str
    message: str = Field(min_length=1, max_length=1_000)
    retryable: bool
    next_action: ManualAction | None = None


class _FaultElapsedReceipt(ContractModel):
    """The complete stdout contract for numeric fault-experiment observations."""

    elapsed_ns: Annotated[int, Field(strict=True, gt=0)]
    outcome: Literal["completed"]


class FaultExperimentResult(ContractModel):
    result_id: str
    plan_id: str
    experiment: Experiment
    trials: tuple[Trial, ...]
    variants: tuple[Variant, ...]
    run_sets: tuple[RunSet, ...] = ()
    comparisons: tuple[ComparisonResult, ...] = ()
    block_treatment_orders: tuple[tuple[str, ...], ...] = ()
    trial_artifacts: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    trial_diagnostics: dict[str, FaultTrialDiagnostic] = Field(default_factory=dict)
    measurement_recovery: ManualAction | None = None
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _format_endpoint(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _proxy_name(plan_id: Digest, trial_index: int) -> str:
    """Derive a Toxiproxy-safe name from an algorithm-qualified plan digest."""
    digest = plan_id.partition(":")[2]
    return f"flameox-{digest[:12]}-{trial_index:04d}"


@dataclass(frozen=True, slots=True)
class _FailedSidecarAttempt:
    phase: Literal["sidecar_readiness", "proxy_creation"]
    error: BaseException
    outcome: ManagedSidecarOutcome | None


@dataclass(frozen=True, slots=True)
class _SidecarEvidenceFile:
    path: Path
    kind: ArtifactKind
    role: str
    media_type: str
    process_outcome: ManagedSidecarOutcome | None = None


class _SidecarAttemptError(Exception):
    def __init__(
        self,
        attempts: tuple[_FailedSidecarAttempt, ...],
    ) -> None:
        super().__init__("managed sidecar attempts failed")
        self.attempts = attempts


class _SidecarAttemptCancelled(asyncio.CancelledError):
    def __init__(self, attempts: tuple[_FailedSidecarAttempt, ...]) -> None:
        super().__init__("managed sidecar attempt cancelled")
        self.attempts = attempts


def _scenario_attributes(scenario: FaultScenario) -> dict[str, int]:
    if isinstance(scenario, LatencyFault):
        return {"latency": scenario.latency_ms, "jitter": scenario.jitter_ms}
    if isinstance(scenario, TimeoutFault):
        return {"timeout": scenario.timeout_ms}
    if isinstance(scenario, ResetPeerFault):
        return {}
    if isinstance(scenario, BandwidthFault):
        return {"rate": scenario.bandwidth_limit}
    if isinstance(scenario, SlicerFault):
        return {
            "average_size": scenario.average_size,
            "size_variation": scenario.size_variation,
            "delay": scenario.delay_ms,
        }
    if isinstance(scenario, LimitDataFault):
        return {"bytes": scenario.bytes}
    if isinstance(scenario, SlowCloseFault):
        return {"delay": scenario.delay_ms}
    if isinstance(scenario, ProxyFault):
        return {}
    raise TypeError(f"unsupported fault scenario: {type(scenario)!r}")


class FaultExperimentService:
    """Run declared loopback transport experiments through a managed sidecar."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        captures: CaptureService | None = None,
        broker: SubprocessBroker | None = None,
        tool_manager: ToxiproxyToolManager | None = None,
    ) -> None:
        self.workspace = workspace
        self.captures = captures or CaptureService(workspace, broker=broker)
        self.broker = broker or self.captures.broker
        self.workloads = WorkloadService(workspace)
        self.tools = tool_manager or ToxiproxyToolManager(workspace.paths.root)
        self.publisher = GenerationPublisher(workspace)
        self.experiments = ControlRecordStore(
            workspace, kind="experiments", model=Experiment, id_field="experiment_id"
        )
        self.plans = AuthorizedPlanStore(
            workspace,
            family="fault_experiment",
            model=TypeAdapter(FaultExperimentPlan),
        )
        self.results = ControlRecordStore(
            workspace,
            kind="fault_experiment_results",
            model=FaultExperimentResult,
            id_field="result_id",
        )

    async def plan(
        self,
        *,
        experiment_name: str,
        investigation_id: str,
        hypothesis_id: str | None = None,
        parameter_overrides: dict[str, str | int | float | bool] | None = None,
        execution_policy: ExecutionPolicy,
    ) -> FaultExperimentPlan:
        config = self._config(experiment_name)
        self._validate_investigation(investigation_id, hypothesis_id)
        overrides = parameter_overrides or {}
        endpoint = config.endpoint_template.format(
            host=config.upstream_host,
            port=config.upstream_port,
        )
        self.workloads.resolve(
            config.workload,
            {**overrides, config.endpoint_parameter: endpoint},
            dynamic_parameters=(config.endpoint_parameter,),
        )
        definition = self.workloads.definition(config.workload)
        # Planning must not download or write a managed binary. Capability setup owns
        # that mutation and records its verification separately; planning merely binds
        # the already-prepared identity into the immutable plan.
        receipt = self.tools.staged_receipt()
        if receipt is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "Managed Toxiproxy is not prepared for fault-experiment planning.",
                details={"adapter": "toxiproxy"},
                remediation=(
                    "Call list_capabilities(adapter='toxiproxy'), then start_capability_setup "
                    "with adapters=['toxiproxy'] before planning again.",
                ),
                next_action=manual_action(
                    "Choose an idempotency key and start setup for the toxiproxy adapter.",
                    suggested_action=ActionId.START_CAPABILITY_SETUP,
                    missing_arguments=("idempotency_key",),
                ),
            )
        treatments = (_BASELINE, *tuple(config.scenarios))
        config_digest = digest_model(config.model_dump(mode="json"))
        measurement = config.measurement
        metric_source = MetricSource.MEASUREMENT if measurement is not None else None
        primary_metric = _FAULT_CLIENT_ELAPSED if measurement is not None else "categorical_outcome"
        primary_metric_unit = _FAULT_CLIENT_ELAPSED_UNIT if measurement is not None else None
        measurement_series = (
            MeasurementSeriesSelector(
                scope="workload",
                aggregation="single",
                phase="receipt",
                dimensions={"source": "stdout_json"},
            )
            if measurement is not None
            else None
        )
        experiment = Experiment(
            experiment_id=new_id(),
            investigation_id=investigation_id,
            hypothesis_id=hypothesis_id,
            recipe="fault_transport",
            recipe_version="1",
            workload_definition_id=definition.workload_definition_id,
            experiment_design_id=digest_model(
                {"name": experiment_name, "config": config.model_dump(mode="json")}
            ),
            measurement_protocol_id=digest_model(
                {
                    "adapter": "command",
                    "measurement": (
                        measurement.model_dump(mode="json") if measurement is not None else None
                    ),
                }
            ),
            validation_spec_id=definition.validation_spec_id,
            primary_metric=primary_metric,
            metric_source=metric_source,
            primary_metric_unit=primary_metric_unit,
            measurement_series=measurement_series,
            value_domain=(MetricValueDomain.STRICTLY_POSITIVE if measurement is not None else None),
            zero_policy=(MetricZeroPolicy.REJECT if measurement is not None else None),
            polarity=(
                MetricPolarity.LOWER_IS_BETTER
                if measurement is not None
                else MetricPolarity.NEUTRAL
            ),
            estimand=(
                "median_paired_log_ratio" if measurement is not None else "categorical_outcome"
            ),
            practical_threshold=(measurement.practical_threshold if measurement is not None else 0),
            confidence_level=(measurement.confidence_level if measurement is not None else 0.95),
            stopping_rule={
                "method": ExperimentOutcomeMethod.FIXED_ATTEMPTS_V1,
                "fixed_blocks": config.blocks * config.repetitions,
            },
            random_seed=config.random_seed,
            role=ExperimentRole.EXPLORATORY,
        )
        blocks: list[ExperimentBlock] = []
        for block_number in range(1, config.blocks * config.repetitions + 1):
            cell_treatments = list(treatments)
            generator = random.Random(f"{config.random_seed}:{block_number}")
            generator.shuffle(cell_treatments)
            cells = tuple(
                ExperimentCell(
                    trial_id=digest_model(
                        {"config": config_digest, "treatment": treatment, "block": block_number}
                    ),
                    combination_id=digest_model({"config": config_digest, "treatment": treatment}),
                    treatment=treatment,
                    factors={"scenario": treatment},
                    parameters={},
                )
                for treatment in cell_treatments
            )
            blocks.append(
                ExperimentBlock(
                    block_id=f"fault-block-{block_number:04d}",
                    order=tuple(cell_treatments),
                    cells=cells,
                )
            )
        created = experiment.created_at
        embedded = ExperimentPlan(
            plan_token=secrets.token_hex(32),
            plan_id=digest_model(
                {
                    "experiment": experiment.model_dump(mode="json"),
                    "blocks": [block.model_dump(mode="json") for block in blocks],
                }
            ),
            request_digest=config_digest,
            workspace_id=self.workspace.identity.workspace_id,
            experiment_name=experiment_name,
            experiment=experiment,
            adapter="command",
            metric_source=metric_source or MetricSource.MEASUREMENT,
            execution_policy=execution_policy,
            variant_parameter="scenario",
            variants=treatments,
            factors={"scenario": treatments},
            parameter_overrides=cast(dict[str, JsonValue], overrides),
            blocks=tuple(blocks),
            experiment_config_digest=config_digest,
            created_at=created,
            expires_at=created + timedelta(seconds=3600),
        )
        tool_identity = {
            "version": receipt.version,
            "asset": receipt.asset,
            "asset_sha256": receipt.sha256,
            "executable_sha256": receipt.executable_sha256,
            "manifest_revision": receipt.manifest_revision,
        }
        bound = {
            "workspace_id": self.workspace.identity.workspace_id,
            "experiment": embedded.model_dump(mode="json"),
            "fault_config": config.model_dump(mode="json"),
            "tool": tool_identity,
        }
        plan = FaultExperimentPlan(
            plan_token=secrets.token_hex(32),
            plan_id=digest_model(bound),
            workspace_id=self.workspace.identity.workspace_id,
            experiment_name=experiment_name,
            experiment_plan=embedded,
            fault_config=cast(dict[str, JsonValue], config.model_dump(mode="json")),
            endpoint_parameter=config.endpoint_parameter,
            upstream_host=config.upstream_host,
            upstream_port=config.upstream_port,
            endpoint_template=config.endpoint_template,
            scenarios={
                name: cast(dict[str, JsonValue], scenario.model_dump(mode="json"))
                for name, scenario in config.scenarios.items()
            },
            parameter_overrides=cast(dict[str, JsonValue], overrides),
            tool_version=receipt.version,
            tool_asset=receipt.asset,
            tool_digest=receipt.sha256,
            tool_executable_digest=receipt.executable_sha256,
            tool_manifest_revision=receipt.manifest_revision,
            workload_containment=execution_policy,
            limitations=(
                "The sidecar is a controlled transport perturbation; it does not establish "
                "causality "
                "without the declared workload metric and semantic oracle.",
            ),
        )
        self.plans.issue(
            plan.plan_token,
            plan.plan_id,
            plan,
            expires_at=plan.experiment_plan.expires_at,
        )
        return plan

    async def run(  # noqa: C901 - this is the bounded fault-trial state machine
        self,
        plan_token: str,
        *,
        progress: Callable[[float, float, str], Awaitable[None]] | None = None,
    ) -> FaultExperimentResult:
        plan = self.plans.consume(plan_token)
        config = self._config(plan.experiment_name)
        if (
            digest_model(config.model_dump(mode="json"))
            != plan.experiment_plan.experiment_config_digest
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Fault experiment definition changed after planning.",
            )
        definition = self.workloads.definition(config.workload)
        if (
            definition.workload_definition_id
            != plan.experiment_plan.experiment.workload_definition_id
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Fault workload definition changed after planning.",
            )
        receipt = await run_atomic_thread(self.tools.stage)
        if (
            receipt.version != plan.tool_version
            or receipt.asset != plan.tool_asset
            or receipt.sha256 != plan.tool_digest
            or receipt.executable_sha256 != plan.tool_executable_digest
            or receipt.manifest_revision != plan.tool_manifest_revision
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN, "Managed Toxiproxy identity changed after planning."
            )
        helper = ExperimentService(self.workspace, captures=self.captures)
        try:
            self.experiments.create(plan.experiment_plan.experiment)
        except DomainError as error:
            if error.code is not ErrorCode.REVISION_CONFLICT:
                raise
        self.publisher.publish_rows(
            {"experiments": [helper._experiment_row(plan.experiment_plan.experiment)]},
            publisher="flameox.faults",
            publisher_version="1",
        )
        schedule = tuple(
            (block, order, cell)
            for block in plan.experiment_plan.blocks
            for order, cell in enumerate(block.cells)
        )
        trials: list[Trial] = []
        artifact_ids: dict[str, list[str]] = {}
        trial_diagnostics: dict[str, FaultTrialDiagnostic] = {}
        total = len(schedule)
        reporter = ProgressReporter(progress)
        for index, (block, order, cell) in enumerate(schedule):
            await reporter.report(index, total, f"Running fault treatment {cell.treatment}")
            run: RunManifest | None = None
            capture_plan: CapturePlan | None = None
            failure_class = TrialFailureClass.INFRASTRUCTURE_FAILURE
            lease: ManagedSidecarLease | None = None
            sidecar_ids: tuple[str, ...] = ()
            cancellation: asyncio.CancelledError | None = None
            failure_phase: FaultFailurePhase | None = None
            failure_error: Exception | None = None
            failed_sidecar_attempts: tuple[_FailedSidecarAttempt, ...] = ()
            try:
                proxy_name = _proxy_name(plan.plan_id, index)
                (
                    active_lease,
                    admin_port,
                    listen_port,
                    failed_sidecar_attempts,
                ) = await self._start_sidecar(
                    receipt.executable,
                    receipt,
                    plan,
                    proxy_name,
                )
                lease = active_lease
                if cell.treatment != _BASELINE:
                    failure_phase = "treatment_configuration"
                    scenario = config.scenarios[cell.treatment]
                    if isinstance(scenario, ProxyFault):
                        await active_lease.update_proxy_async(
                            proxy_name,
                            enabled=scenario.enabled,
                        )
                    else:
                        await active_lease.add_toxic_async(
                            proxy=proxy_name,
                            name="treatment",
                            toxic_type=scenario.type,
                            stream=scenario.stream,
                            toxicity=scenario.toxicity,
                            attributes=_scenario_attributes(scenario),
                        )
                endpoint = plan.endpoint_template.format(host="127.0.0.1", port=listen_port)
                failure_phase = "workload_planning"
                capture_plan = await self.captures.plan(
                    workload_name=config.workload,
                    adapter="command",
                    parameters={
                        **cast(dict[str, str | int | float | bool], plan.parameter_overrides),
                        plan.endpoint_parameter: endpoint,
                    },
                    execution_policy=plan.experiment_plan.execution_policy,
                    dynamic_parameters=(plan.endpoint_parameter,),
                )
                failure_phase = "capture_execution"
                captured = await self.captures.execute(capture_plan.plan_token)
                run = captured.run
                outcome, failure_class = helper._classify_run(run)
                failure_phase = None
            except _SidecarAttemptCancelled as error:
                cancellation = error
                failed_sidecar_attempts = error.attempts
                terminal_attempt = error.attempts[-1]
                failure_phase = terminal_attempt.phase
                failure_error = DomainError(
                    ErrorCode.PROCESS_CANCELLED,
                    "Managed sidecar startup was cancelled.",
                    retryable=True,
                )
                outcome, failure_class = (
                    TrialOutcome.CANCELLED,
                    TrialFailureClass.CANCELLATION,
                )
            except asyncio.CancelledError as error:
                cancellation = error
                failure_error = DomainError(
                    ErrorCode.PROCESS_CANCELLED,
                    "Fault trial execution was cancelled.",
                    retryable=True,
                )
                if (
                    run is None
                    and capture_plan is not None
                    and RunStore(self.workspace).exists(capture_plan.run_id)
                ):
                    run = RunStore(self.workspace).read(capture_plan.run_id)
                outcome, failure_class = (
                    TrialOutcome.CANCELLED,
                    TrialFailureClass.CANCELLATION,
                )
            except _SidecarAttemptError as error:
                failed_sidecar_attempts = error.attempts
                terminal_attempt = error.attempts[-1]
                failure_phase = terminal_attempt.phase
                failure_error = cast(Exception, terminal_attempt.error)
                outcome, failure_class = (
                    TrialOutcome.INFRASTRUCTURE_FAILED,
                    TrialFailureClass.INFRASTRUCTURE_FAILURE,
                )
            except (ToxiproxyApiError, OSError, ValueError) as error:
                failure_error = error
                outcome, failure_class = (
                    TrialOutcome.INFRASTRUCTURE_FAILED,
                    TrialFailureClass.INFRASTRUCTURE_FAILURE,
                )
            except DomainError as error:
                failure_error = error
                if error.run_id is not None:
                    run = RunStore(self.workspace).read(error.run_id)
                outcome, failure_class = (
                    (TrialOutcome.UNSUPPORTED, TrialFailureClass.UNSUPPORTED_ENVIRONMENT)
                    if error.code is ErrorCode.CAPABILITY_UNAVAILABLE
                    else (
                        TrialOutcome.INFRASTRUCTURE_FAILED,
                        TrialFailureClass.INFRASTRUCTURE_FAILURE,
                    )
                )
            finally:
                if lease is not None:
                    await asyncio.shield(lease.close())
            if lease is not None and lease.outcome is not None:
                if run is None:
                    run = self._create_sidecar_run(plan, lease.outcome)
                sidecar_ids = await self._attach_sidecar_evidence(
                    plan,
                    cell,
                    run,
                    lease.outcome,
                    capture_plan=capture_plan,
                    admin_port=admin_port,
                    listen_port=listen_port,
                    proxy_name=proxy_name,
                    failure_phase=failure_phase,
                    failure_error=failure_error,
                )
                artifact_ids[cell.trial_id] = list(sidecar_ids)
            if failed_sidecar_attempts:
                if run is None:
                    terminal_outcome = failed_sidecar_attempts[-1].outcome
                    run = self._create_sidecar_run(plan, terminal_outcome)
                else:
                    run = RunStore(self.workspace).read(run.run_id)
                failed_ids = await self._attach_failed_sidecar_evidence(
                    plan,
                    run,
                    failed_sidecar_attempts,
                )
                artifact_ids.setdefault(cell.trial_id, []).extend(failed_ids)
            if failure_phase is not None and failure_error is not None:
                trial_diagnostics[cell.trial_id] = self._fault_diagnostic(
                    failure_phase,
                    failure_error,
                )
            trial = helper._make_trial(
                plan=plan.experiment_plan,
                cell=cell,
                run=run,
                block_id=block.block_id,
                order=order,
                outcome=outcome,
                failure_class=failure_class,
            )
            helper._publish_trial(trial)
            trials.append(trial)
            if cancellation is not None:
                for remaining_block, remaining_order, remaining_cell in schedule[index + 1 :]:
                    unattempted = helper._make_trial(
                        plan=plan.experiment_plan,
                        cell=remaining_cell,
                        run=None,
                        block_id=remaining_block.block_id,
                        order=remaining_order,
                        outcome=TrialOutcome.UNATTEMPTED,
                        failure_class=TrialFailureClass.UNATTEMPTED,
                    )
                    helper._publish_trial(unattempted)
                    trials.append(unattempted)
                self._publish_fault_variants(plan, trials)
                raise cancellation
        variants = self._publish_fault_variants(plan, trials)
        if plan.experiment_plan.experiment.primary_metric == _FAULT_CLIENT_ELAPSED:
            self._publish_measurement_receipts(trials, trial_diagnostics)
        run_sets, comparisons, measurement_recovery = self._finalize_measurement_comparisons(
            plan,
            trials,
        )
        result = FaultExperimentResult(
            result_id=new_id(),
            plan_id=plan.plan_id,
            experiment=plan.experiment_plan.experiment,
            trials=tuple(trials),
            variants=variants,
            run_sets=run_sets,
            comparisons=comparisons,
            block_treatment_orders=tuple(block.order for block in plan.experiment_plan.blocks),
            trial_artifacts={name: tuple(values) for name, values in artifact_ids.items()},
            trial_diagnostics=trial_diagnostics,
            measurement_recovery=measurement_recovery,
            corpus_commit_id=self.workspace.corpus.read_head().commit_id,
            limitations=(
                *plan.limitations,
                *(
                    (
                        "Only successful trials with a valid positive elapsed_ns stdout receipt "
                        "are eligible for numeric comparison.",
                    )
                    if plan.experiment_plan.experiment.primary_metric == _FAULT_CLIENT_ELAPSED
                    else (
                        "Categorical fault mode records trial outcomes only; it does not publish "
                        "numeric measurements or comparisons.",
                    )
                ),
            ),
        )
        self.results.create(result)
        return result

    def show(self, result_id: str) -> FaultExperimentResult:
        return self.results.read(result_id)

    def _publish_fault_variants(
        self, plan: FaultExperimentPlan, trials: list[Trial]
    ) -> tuple[Variant, ...]:
        helper = ExperimentService(self.workspace, captures=self.captures)
        variants: list[Variant] = []
        for name in plan.experiment_plan.variants:
            cell_ids = {
                cell.trial_id
                for block in plan.experiment_plan.blocks
                for cell in block.cells
                if cell.treatment == name
            }
            treatment_trials = [item for item in trials if item.trial_id in cell_ids]
            variants.append(
                helper._variant_for_treatment(
                    plan.experiment_plan,
                    name,
                    treatment_trials,
                )
            )
        self.publisher.publish_rows(
            {"variants": [helper._variant_row(value) for value in variants]},
            publisher="flameox.faults",
            publisher_version="1",
            input_run_ids=tuple(item.run_id for item in trials if item.run_id is not None),
        )
        return tuple(variants)

    def _publish_measurement_receipts(
        self,
        trials: list[Trial],
        diagnostics: dict[str, FaultTrialDiagnostic],
    ) -> None:
        """Publish one receipt-owned measurement per successful fault trial.

        The stdout artifact is already captured immutably by ``CaptureService``.  This method
        only interprets its exact bytes under the declared receipt contract; failures remain
        categorical trial outcomes and never acquire a synthetic numeric value.
        """

        successful = [
            trial
            for trial in trials
            if trial.outcome is TrialOutcome.SUCCEEDED and trial.run_id is not None
        ]
        if not successful:
            return
        artifacts = ArtifactStore(self.workspace)
        runs = RunStore(self.workspace)
        rows: list[dict[str, object]] = []
        artifact_ids: list[str] = []
        for trial in successful:
            assert trial.run_id is not None
            run = runs.read(trial.run_id)
            stdout = next((item for item in run.artifacts if item.role == "stdout"), None)
            if stdout is None:
                diagnostics[trial.trial_id] = self._receipt_diagnostic(
                    "stdout artifact is absent.",
                    error_code=ErrorCode.ARTIFACT_NOT_FOUND,
                )
                continue
            try:
                stored = artifacts.get(stdout.artifact_id)
                content, payload = artifacts.read_range(
                    stdout.artifact_id,
                    offset=0,
                    max_bytes=stored.content.byte_length,
                )
                if len(payload) != content.byte_length:
                    raise ValueError("stdout artifact was not read completely")
                receipt = _FaultElapsedReceipt.model_validate_json(payload)
            except (DomainError, ValidationError, ValueError) as error:
                diagnostics[trial.trial_id] = self._receipt_diagnostic(
                    f"stdout does not contain one valid elapsed-time receipt: {error}"
                )
                continue
            rows.append(
                {
                    "measurement_id": digest_model(
                        {
                            "run_id": trial.run_id,
                            "artifact_id": stdout.artifact_id,
                            "name": _FAULT_CLIENT_ELAPSED,
                        }
                    ),
                    "run_id": trial.run_id,
                    "artifact_id": stdout.artifact_id,
                    "name": _FAULT_CLIENT_ELAPSED,
                    "value_int": receipt.elapsed_ns,
                    "value_float": None,
                    "value_uint": None,
                    "value_kind": "integer",
                    "unit": _FAULT_CLIENT_ELAPSED_UNIT,
                    "aggregation": "single",
                    "scope": "workload",
                    "trial_id": trial.trial_id,
                    "worker_id": None,
                    "worker_run_index": None,
                    "value_index": None,
                    "loop_count": None,
                    "is_warmup": False,
                    "block_id": trial.block_id,
                    "variant_id": trial.variant_id,
                    "order_in_block": trial.order_in_block,
                    "phase": "receipt",
                    "dimensions": {"source": "stdout_json"},
                    "evidence_level": "observed",
                }
            )
            artifact_ids.append(stdout.artifact_id)
        if rows:
            self.publisher.publish_rows(
                {"measurements": rows},
                publisher="flameox.faults",
                publisher_version="1",
                input_run_ids=tuple(
                    trial.run_id for trial in successful if trial.run_id is not None
                ),
                input_artifact_ids=tuple(artifact_ids),
            )

    @staticmethod
    def _receipt_diagnostic(
        detail: str,
        *,
        error_code: ErrorCode = ErrorCode.ARTIFACT_PARSE_FAILED,
    ) -> FaultTrialDiagnostic:
        return FaultTrialDiagnostic(
            phase="measurement_receipt",
            error_code=error_code,
            message=detail[:1_000],
            retryable=False,
        )

    def _finalize_measurement_comparisons(
        self,
        plan: FaultExperimentPlan,
        trials: list[Trial],
    ) -> tuple[tuple[RunSet, ...], tuple[ComparisonResult, ...], ManualAction | None]:
        experiment = plan.experiment_plan.experiment
        if experiment.primary_metric != _FAULT_CLIENT_ELAPSED:
            return (), (), None
        helper = ExperimentService(self.workspace, captures=self.captures)
        trials_by_treatment = {
            name: [trial for trial in trials if trial.factors.get("scenario") == name]
            for name in plan.experiment_plan.variants
        }
        run_sets = helper._freeze_run_sets(plan.experiment_plan, trials_by_treatment)
        by_treatment = {
            str(run_set.selection["variant"]): run_set
            for run_set in run_sets
            if isinstance(run_set.selection.get("variant"), str)
        }
        baseline = by_treatment.get(_BASELINE)
        candidates = tuple(
            run_set for treatment, run_set in by_treatment.items() if treatment != _BASELINE
        )
        if baseline is None or not candidates:
            return run_sets, (), self._measurement_recovery()
        comparisons = tuple(
            ComparisonService(self.workspace).record(
                MeasurementCompareRunSetsRequest(
                    baseline_run_set_id=baseline.run_set_id,
                    candidate_run_set_id=candidate.run_set_id,
                    experiment_id=experiment.experiment_id,
                    metric=experiment.primary_metric,
                    unit=experiment.primary_metric_unit or "ns",
                    series=experiment.measurement_series,
                    polarity=experiment.polarity,
                    estimand="median_paired_log_ratio",
                    practical_threshold=experiment.practical_threshold,
                    confidence_level=experiment.confidence_level,
                    random_seed=experiment.random_seed,
                )
            )
            for candidate in candidates
        )
        has_invalid_comparison = any(
            comparison.comparison.validity is ComparisonValidity.INVALID
            for comparison in comparisons
        )
        recovery = self._measurement_recovery() if has_invalid_comparison else None
        return run_sets, comparisons, recovery

    @staticmethod
    def _measurement_recovery() -> ManualAction:
        return manual_action(
            "Inspect measurement diagnostics and stdout receipts, then plan a changed fault "
            "experiment with one completed positive elapsed_ns receipt per paired block.",
            suggested_action=ActionId.PLAN_FAULT_EXPERIMENT,
            missing_arguments=("experiment_name", "investigation_id"),
        )

    def _config(self, name: str) -> FaultExperimentConfig:
        try:
            return self.workloads.load().fault_experiments[name]
        except KeyError as error:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID, f"Unknown fault experiment {name!r}."
            ) from error

    async def _start_sidecar(
        self,
        executable: Path,
        receipt: object,
        plan: FaultExperimentPlan,
        proxy_name: str,
    ) -> tuple[ManagedSidecarLease, int, int, tuple[_FailedSidecarAttempt, ...]]:
        failed_attempts: list[_FailedSidecarAttempt] = []
        for sidecar_attempt in range(3):
            lease: ManagedSidecarLease | None = None
            phase: Literal["sidecar_readiness", "proxy_creation"] = "sidecar_readiness"
            try:
                admin_port = _free_loopback_port()
                client = ToxiproxyClient(f"http://127.0.0.1:{admin_port}")

                async def readiness(client: ToxiproxyClient = client) -> bool:
                    try:
                        version = await client.version_async()
                    except ToxiproxyApiError:
                        return False
                    return version == plan.tool_version

                lease = await self.broker.start_toxiproxy(
                    executable,
                    admin_host="127.0.0.1",
                    admin_port=admin_port,
                    readiness=readiness,
                    tool_receipt=receipt,
                )
                phase = "proxy_creation"
                listen_port = _free_loopback_port()
                await lease.create_proxy_async(
                    name=proxy_name,
                    listen=f"127.0.0.1:{listen_port}",
                    upstream=_format_endpoint(plan.upstream_host, plan.upstream_port),
                )
                return lease, admin_port, listen_port, tuple(failed_attempts)
            except ManagedSidecarStartupCancelled as error:
                failed_attempts.append(_FailedSidecarAttempt(phase, error, error.outcome))
                raise _SidecarAttemptCancelled(tuple(failed_attempts)) from None
            except asyncio.CancelledError as error:
                cancellation_outcome = (
                    await asyncio.shield(lease.close()) if lease is not None else None
                )
                failed_attempts.append(_FailedSidecarAttempt(phase, error, cancellation_outcome))
                raise _SidecarAttemptCancelled(tuple(failed_attempts)) from None
            except (DomainError, ToxiproxyApiError, OSError) as error:
                failed_outcome: ManagedSidecarOutcome | None = None
                if lease is not None:
                    failed_outcome = await asyncio.shield(lease.close())
                elif isinstance(error, ManagedSidecarStartupError):
                    failed_outcome = error.outcome
                failed_attempts.append(_FailedSidecarAttempt(phase, error, failed_outcome))
                if sidecar_attempt == 2:
                    raise _SidecarAttemptError(tuple(failed_attempts)) from error
        raise AssertionError("sidecar retry loop did not return")

    def _validate_investigation(self, investigation_id: str, hypothesis_id: str | None) -> None:
        ControlRecordStore(
            self.workspace, kind="investigations", model=Investigation, id_field="investigation_id"
        ).read(investigation_id)
        if hypothesis_id is not None:
            hypothesis = ControlRecordStore(
                self.workspace,
                kind="hypotheses",
                model=Hypothesis,
                id_field="hypothesis_id",
                revision_field="revision",
            ).read(hypothesis_id)
            if hypothesis.investigation_id != investigation_id:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "The hypothesis does not belong to the investigation.",
                )

    async def _attach_sidecar_evidence(
        self,
        plan: FaultExperimentPlan,
        cell: ExperimentCell,
        run: RunManifest,
        outcome: ManagedSidecarOutcome,
        *,
        capture_plan: CapturePlan | None,
        admin_port: int,
        listen_port: int,
        proxy_name: str,
        failure_phase: FaultFailurePhase | None = None,
        failure_error: Exception | None = None,
    ) -> tuple[str, ...]:
        root = self.workspace.paths.staging / "fault-sidecars" / run.run_id
        root.mkdir(parents=True, exist_ok=True)
        config_path = root / "configuration.json"
        snapshot_path = root / "process-snapshot.json"
        stdout_path = root / "stdout.log"
        stderr_path = root / "stderr.log"
        payload: dict[str, object] = {
            "plan_id": plan.plan_id,
            "experiment_id": plan.experiment_plan.experiment.experiment_id,
            "trial_id": cell.trial_id,
            "treatment": cell.treatment,
            "fault_config": plan.fault_config,
            "scenario": plan.scenarios.get(cell.treatment),
            "failure": (
                self._fault_diagnostic(failure_phase, failure_error).model_dump(
                    mode="json", exclude={"next_action"}
                )
                if failure_phase is not None and failure_error is not None
                else None
            ),
            "tool": {
                "version": plan.tool_version,
                "asset": plan.tool_asset,
                "sha256": plan.tool_digest,
                "executable_sha256": plan.tool_executable_digest,
                "manifest_revision": plan.tool_manifest_revision,
            },
            "observed": {
                "admin_host": "127.0.0.1",
                "admin_port": admin_port,
                "proxy_name": proxy_name,
                "proxy_listen": f"127.0.0.1:{listen_port}",
                "proxy_upstream": _format_endpoint(plan.upstream_host, plan.upstream_port),
                "containment": {
                    "sidecar": outcome.containment,
                    "workload_policy": plan.workload_containment,
                    "workload_effective": (
                        capture_plan.containment
                        if capture_plan is not None
                        else plan.workload_containment
                    ),
                    "workload_network_contained": (
                        capture_plan.network_contained if capture_plan is not None else None
                    ),
                },
                "started_at": outcome.started_at.isoformat(),
                "finished_at": outcome.finished_at.isoformat(),
                "process": outcome.process.model_dump(mode="json"),
                "oracle": {
                    "validation_status": run.validation_status.value,
                    "receipt": (
                        run.oracle_receipt.model_dump(mode="json")
                        if run.oracle_receipt is not None
                        else None
                    ),
                },
            },
        }
        config_path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        evidence_status, snapshot_limitations = process_observation_coverage(
            outcome.process_observations
        )
        snapshot_path.write_text(
            json.dumps(
                {
                    "evidence_status": evidence_status.value,
                    "limitations": snapshot_limitations,
                    "observations": [
                        item.model_dump(mode="json") for item in outcome.process_observations
                    ],
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        stdout_path.write_bytes(outcome.stdout)
        stderr_path.write_bytes(outcome.stderr)
        files = (
            _SidecarEvidenceFile(
                config_path,
                ArtifactKind.EXPERIMENT_CONFIGURATION,
                "fault_configuration",
                "application/json",
            ),
            _SidecarEvidenceFile(
                snapshot_path,
                ArtifactKind.PROCESS_TREE_SNAPSHOT,
                "fault_process_observation",
                "application/json",
                outcome,
            ),
            _SidecarEvidenceFile(
                stdout_path, ArtifactKind.PROCESS_OUTPUT, "toxiproxy_stdout", "text/plain"
            ),
            _SidecarEvidenceFile(
                stderr_path, ArtifactKind.PROCESS_OUTPUT, "toxiproxy_stderr", "text/plain"
            ),
        )
        try:
            return await self._publish_sidecar_evidence(plan, run, files)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    async def _attach_failed_sidecar_evidence(
        self,
        plan: FaultExperimentPlan,
        run: RunManifest,
        attempts: tuple[_FailedSidecarAttempt, ...],
    ) -> tuple[str, ...]:
        root = self.workspace.paths.staging / "fault-sidecars" / run.run_id
        root.mkdir(parents=True, exist_ok=True)
        files: list[_SidecarEvidenceFile] = []
        for index, attempt in enumerate(attempts, start=1):
            prefix = f"attempt-{index:02d}"
            diagnostic_path = root / f"{prefix}-diagnostic.json"
            diagnostic_path.write_text(
                self._fault_diagnostic(attempt.phase, attempt.error).model_dump_json(
                    exclude={"next_action"}
                )
            )
            files.append(
                _SidecarEvidenceFile(
                    diagnostic_path,
                    ArtifactKind.COLLECTOR_METADATA,
                    f"fault_startup_{prefix}_diagnostic",
                    "application/json",
                )
            )
            outcome = attempt.outcome
            if outcome is None:
                continue
            snapshot_path = root / f"{prefix}-process-snapshot.json"
            stdout_path = root / f"{prefix}-stdout.log"
            stderr_path = root / f"{prefix}-stderr.log"
            evidence_status, limitations = process_observation_coverage(
                outcome.process_observations
            )
            snapshot_path.write_text(
                json.dumps(
                    {
                        "evidence_status": evidence_status.value,
                        "limitations": limitations,
                        "observations": [
                            item.model_dump(mode="json") for item in outcome.process_observations
                        ],
                        "process": outcome.process.model_dump(mode="json"),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            stdout_path.write_bytes(outcome.stdout)
            stderr_path.write_bytes(outcome.stderr)
            files.extend(
                (
                    _SidecarEvidenceFile(
                        snapshot_path,
                        ArtifactKind.PROCESS_TREE_SNAPSHOT,
                        f"fault_startup_{prefix}_process_observation",
                        "application/json",
                        outcome,
                    ),
                    _SidecarEvidenceFile(
                        stdout_path,
                        ArtifactKind.PROCESS_OUTPUT,
                        f"fault_startup_{prefix}_stdout",
                        "text/plain",
                    ),
                    _SidecarEvidenceFile(
                        stderr_path,
                        ArtifactKind.PROCESS_OUTPUT,
                        f"fault_startup_{prefix}_stderr",
                        "text/plain",
                    ),
                )
            )
        try:
            return await self._publish_sidecar_evidence(plan, run, tuple(files))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    async def _publish_sidecar_evidence(
        self,
        plan: FaultExperimentPlan,
        run: RunManifest,
        files: tuple[_SidecarEvidenceFile, ...],
    ) -> tuple[str, ...]:
        registrations: list[ArtifactRegistration] = []
        byte_lengths: list[int] = []
        snapshot_rows: list[dict[str, object]] = []
        snapshot_entries: list[dict[str, object]] = []
        for evidence_file in files:
            registration, byte_length = await self.captures._register_path_async(
                run.run_id,
                evidence_file.path,
                kind=evidence_file.kind,
                role=evidence_file.role,
                media_type=evidence_file.media_type,
                producer="flameox.toxiproxy",
                producer_version=plan.tool_version,
                sensitivity=Sensitivity.INTERNAL,
            )
            registrations.append(registration)
            byte_lengths.append(byte_length)
            if evidence_file.process_outcome is not None:
                attempt_outcome = evidence_file.process_outcome
                evidence_status, limitations = process_observation_coverage(
                    attempt_outcome.process_observations
                )
                rows, entries = process_observation_rows(
                    run.run_id,
                    attempt_outcome.process_observations,
                    artifact_id=registration.artifact_id,
                    evidence_status=evidence_status,
                    limitations=limitations,
                )
                snapshot_rows.extend(rows)
                snapshot_entries.extend(entries)
        updated = run.validated_copy(
            update={"revision": run.revision + 1, "artifacts": run.artifacts + tuple(registrations)}
        )
        RunStore(self.workspace).append(updated, expected_revision=run.revision)
        self.publisher.publish_rows(
            {
                "runs": [run_row(updated)],
                "artifact_registrations": [
                    artifact_registration_row(registration, byte_length=byte_length)
                    for registration, byte_length in zip(registrations, byte_lengths, strict=True)
                ],
                "process_snapshots": snapshot_rows,
                "process_snapshot_entries": snapshot_entries,
            },
            publisher="flameox.toxiproxy",
            publisher_version=plan.tool_version,
            input_run_ids=(run.run_id,),
        )
        return tuple(registration.artifact_id for registration in registrations)

    @staticmethod
    def _fault_diagnostic(
        phase: FaultFailurePhase,
        error: BaseException,
    ) -> FaultTrialDiagnostic:
        domain_error = error if isinstance(error, DomainError) else None
        retryable = (
            domain_error.retryable
            if domain_error is not None
            else isinstance(error, (asyncio.CancelledError, OSError, ToxiproxyApiError))
        )
        message = {
            "sidecar_readiness": "The managed sidecar did not become ready.",
            "proxy_creation": "The managed proxy could not be created.",
            "treatment_configuration": "The fault treatment could not be configured.",
            "workload_planning": "The workload capture plan could not be created.",
            "capture_execution": "The workload capture could not be completed.",
        }[phase]
        return FaultTrialDiagnostic(
            phase=phase,
            error_code=(
                domain_error.code.value
                if domain_error is not None
                else (
                    ErrorCode.PROCESS_CANCELLED.value
                    if isinstance(error, asyncio.CancelledError)
                    else type(error).__name__
                )
            ),
            message=message,
            retryable=retryable,
            next_action=(
                manual_action(
                    "Call plan_fault_experiment with the same declared experiment after "
                    "addressing the reported failure.",
                    suggested_action=ActionId.PLAN_FAULT_EXPERIMENT,
                    missing_arguments=(
                        "experiment_name",
                        "investigation_id",
                        "parameters",
                    ),
                )
                if retryable
                else None
            ),
        )

    def _create_sidecar_run(
        self, plan: FaultExperimentPlan, outcome: ManagedSidecarOutcome | None
    ) -> RunManifest:
        environment = collect_environment()
        source_state = collect_partial_source_state(self.workspace)
        now = datetime.now(UTC)
        run = ExecutionRunManifest(
            run_id=new_id(),
            started_at=outcome.started_at if outcome is not None else now,
            finished_at=outcome.finished_at if outcome is not None else now,
            execution_status=ExecutionStatus.FAILED,
            capture_status=CaptureStatus.FAILED,
            validation_status=ValidationStatus.NOT_REQUESTED,
            workload_definition_id=plan.experiment_plan.experiment.workload_definition_id,
            measurement_protocol_id=plan.experiment_plan.experiment.measurement_protocol_id,
            environment_id=environment.environment_id,
            source_state_id=source_state.source_state_id,
            semantics=RunSemantics.unavailable(
                origin="internal",
                adapter="flameox.toxiproxy",
                adapter_version=plan.tool_version,
                fields=("effective_options",),
            ),
            process=outcome.process if outcome is not None else None,
            limitations=("The workload did not start; this run contains sidecar-only evidence.",),
        )
        RunStore(self.workspace).create(run)
        self.publisher.publish_rows(
            {
                "runs": [run_row(run)],
                "environments": [environment_row(environment)],
                "source_states": [source_state_row(source_state)],
            },
            publisher="flameox.faults",
            publisher_version="1",
            input_run_ids=(run.run_id,),
        )
        return run
