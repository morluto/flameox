from __future__ import annotations

import asyncio
import json
import random
import secrets
import shutil
import socket
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, JsonValue, TypeAdapter, computed_field

from flameox.action_graph import ActionId, manual_action
from flameox.adapters.toxiproxy import ToxiproxyApiError, ToxiproxyClient, ToxiproxyToolManager
from flameox.application.async_work import run_atomic_thread
from flameox.application.capture import CaptureService
from flameox.application.environment import collect_environment
from flameox.application.evidence_rows import (
    artifact_registration_row,
    environment_row,
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
    DomainError,
    ErrorCode,
    ExecutionStatus,
    Experiment,
    ExperimentOutcomeMethod,
    ExperimentRole,
    Hypothesis,
    Investigation,
    MetricSource,
    RunManifest,
    RunSemantics,
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
from flameox.execution import ManagedSidecarLease, ManagedSidecarOutcome, SubprocessBroker
from flameox.models import ContractModel
from flameox.storage import AuthorizedPlanStore, ControlRecordStore, RunStore, Workspace

_BASELINE = "baseline"


class FaultExperimentPlan(ContractModel):
    schema_version: Literal[2] = 2
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def request_digest(self) -> Digest:
        return self.plan_id


class FaultExperimentResult(ContractModel):
    schema_version: Literal[1, 2] = 2
    result_id: str
    plan_id: str
    experiment: Experiment
    trials: tuple[Trial, ...]
    treatment_order: tuple[str, ...] | None = None
    block_treatment_orders: tuple[tuple[str, ...], ...] = ()
    trial_artifacts: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    corpus_commit_id: str
    limitations: tuple[str, ...] = ()


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _format_endpoint(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


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
            output_only_fields={"request_digest"},
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
            measurement_protocol_id=digest_model({"adapter": "command"}),
            validation_spec_id=definition.validation_spec_id,
            primary_metric=config.primary_metric,
            metric_source=(
                MetricSource.RUNTIME_RESOURCE
                if config.primary_metric.startswith("runtime_resource.")
                else MetricSource.MEASUREMENT
            ),
            primary_metric_unit=(
                "bytes" if config.primary_metric.startswith("runtime_resource.") else None
            ),
            polarity=config.polarity,
            estimand=config.estimand,
            practical_threshold=config.practical_threshold,
            confidence_level=config.confidence_level,
            stopping_rule={
                "method": ExperimentOutcomeMethod.FIXED_ATTEMPTS_V1,
                "blocks": config.blocks,
                "repetitions": config.repetitions,
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
            metric_source=(
                MetricSource.RUNTIME_RESOURCE
                if config.primary_metric.startswith("runtime_resource.")
                else MetricSource.MEASUREMENT
            ),
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
            plan.request_digest,
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
            try:
                proxy_name = f"flameox-{plan.plan_id[:12]}-{index:04d}"
                active_lease, admin_port, listen_port = await self._start_sidecar(
                    receipt.executable,
                    receipt,
                    plan,
                    proxy_name,
                )
                lease = active_lease
                if cell.treatment != _BASELINE:
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
                captured = await self.captures.execute(capture_plan.plan_token)
                run = captured.run
                outcome, failure_class = helper._classify_run(run)
            except asyncio.CancelledError as error:
                cancellation = error
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
            except (ToxiproxyApiError, OSError, ValueError):
                outcome, failure_class = (
                    TrialOutcome.INFRASTRUCTURE_FAILED,
                    TrialFailureClass.INFRASTRUCTURE_FAILURE,
                )
            except DomainError as error:
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
                )
                artifact_ids[cell.trial_id] = list(sidecar_ids)
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
        self._publish_fault_variants(plan, trials)
        result = FaultExperimentResult(
            result_id=new_id(),
            plan_id=plan.plan_id,
            experiment=plan.experiment_plan.experiment,
            trials=tuple(trials),
            block_treatment_orders=tuple(block.order for block in plan.experiment_plan.blocks),
            trial_artifacts={name: tuple(values) for name, values in artifact_ids.items()},
            corpus_commit_id=self.workspace.corpus.read_head().commit_id,
            limitations=(
                *plan.limitations,
                "Fault results are descriptive unless normal measurement and oracle evidence "
                "support a comparison.",
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
    ) -> tuple[ManagedSidecarLease, int, int]:
        for sidecar_attempt in range(3):
            admin_port = _free_loopback_port()
            client = ToxiproxyClient(f"http://127.0.0.1:{admin_port}")

            async def readiness(client: ToxiproxyClient = client) -> bool:
                try:
                    version = await client.version_async()
                except ToxiproxyApiError:
                    return False
                return version == plan.tool_version

            lease: ManagedSidecarLease | None = None
            try:
                lease = await self.broker.start_toxiproxy(
                    executable,
                    admin_host="127.0.0.1",
                    admin_port=admin_port,
                    readiness=readiness,
                    tool_receipt=receipt,
                )
                listen_port = _free_loopback_port()
                await lease.create_proxy_async(
                    name=proxy_name,
                    listen=f"127.0.0.1:{listen_port}",
                    upstream=_format_endpoint(plan.upstream_host, plan.upstream_port),
                )
                return lease, admin_port, listen_port
            except (DomainError, ToxiproxyApiError, OSError):
                if lease is not None:
                    await asyncio.shield(lease.close())
                if sidecar_attempt == 2:
                    raise
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
        snapshot_path.write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in outcome.process_observations],
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        stdout_path.write_bytes(outcome.stdout)
        stderr_path.write_bytes(outcome.stderr)
        registrations: list[ArtifactRegistration] = []
        byte_lengths: list[int] = []
        ids: list[str] = []
        for path, kind, role in (
            (config_path, ArtifactKind.EXPERIMENT_CONFIGURATION, "fault_configuration"),
            (snapshot_path, ArtifactKind.PROCESS_TREE_SNAPSHOT, "fault_process_observation"),
            (stdout_path, ArtifactKind.PROCESS_OUTPUT, "toxiproxy_stdout"),
            (stderr_path, ArtifactKind.PROCESS_OUTPUT, "toxiproxy_stderr"),
        ):
            registration, byte_length = await self.captures._register_path_async(
                run.run_id,
                path,
                kind=kind,
                role=role,
                media_type="application/json" if path.suffix == ".json" else "text/plain",
                producer="flameox.toxiproxy",
                producer_version=plan.tool_version,
                sensitivity=Sensitivity.INTERNAL,
            )
            registrations.append(registration)
            byte_lengths.append(byte_length)
            ids.append(registration.artifact_id)
        updated = run.validated_copy(
            update={"revision": run.revision + 1, "artifacts": run.artifacts + tuple(registrations)}
        )
        RunStore(self.workspace).append(updated, expected_revision=run.revision)
        process_rows, entry_rows = process_observation_rows(
            run.run_id,
            outcome.process_observations,
            artifact_id=ids[1],
        )
        self.publisher.publish_rows(
            {
                "runs": [run_row(updated)],
                "artifact_registrations": [
                    artifact_registration_row(registration, byte_length=byte_length)
                    for registration, byte_length in zip(registrations, byte_lengths, strict=True)
                ],
                "process_snapshots": process_rows,
                "process_snapshot_entries": entry_rows,
            },
            publisher="flameox.toxiproxy",
            publisher_version=plan.tool_version,
            input_run_ids=(run.run_id,),
        )
        shutil.rmtree(root, ignore_errors=True)
        return tuple(ids)

    def _create_sidecar_run(
        self, plan: FaultExperimentPlan, outcome: ManagedSidecarOutcome
    ) -> RunManifest:
        environment = collect_environment()
        source_state = collect_partial_source_state(self.workspace)
        run = ExecutionRunManifest(
            run_id=new_id(),
            started_at=outcome.started_at,
            finished_at=outcome.finished_at,
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
            process=outcome.process,
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
