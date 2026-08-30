"""Plan and run inference benchmark scenarios against declared local servers.

The service loads a declared scenario and server from ``flameox.toml``, either
leases a managed vLLM workload or passively probes an existing loopback server,
constructs the typed AIPerf or vLLM bench command, and executes it through the
canonical ``SubprocessBroker`` under one absolute deadline. The broker owns
containment, quotas, cancellation, and process snapshots; this module never
adds an inference request client or unrestricted command surface.

Benchmark measurements are exploratory by default: a single scenario run does not
establish equivalence or causality. The plan and result preserve the exact
rendered argv, provider/tool identity, resolved output paths, and an explicit
exploratory reason so a later confirmatory experiment can reference the same
inputs.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, assert_never
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, TypeAdapter, computed_field

from flameox.action_graph import ActionId, ToolAction, manual_action, tool_action
from flameox.analysis.inference_protocol import (
    HardwareIdentity,
    InferenceProtocolIdentity,
    ModelIdentity,
    OracleIdentity,
    OracleResult,
    ProfilerState,
    ScheduleIdentity,
    ServerConfigIdentity,
    TraceIdentity,
)
from flameox.application.environment import AcceleratorIdentityService, collect_environment
from flameox.application.imports import ImportDescriptorRequest, ImportService
from flameox.application.inference_providers import (
    AIPerfProfileRequest,
    ExistingServerProbe,
    InferenceEndpointType,
    InferenceScenarioProvider,
    InferenceServerMode,
    InferenceServerProvider,
    InferenceTool,
    QualifiedInferenceTool,
    SglangBenchServingRequest,
    VllmBenchServeRequest,
    discover_inference_tool,
    discover_sglang,
    probe_existing_vllm_server,
    probe_existing_vllm_server_async,
)
from flameox.application.oracle_receipts import parse_oracle_receipt
from flameox.application.projections import ProjectionCoordinator
from flameox.application.provider_runtime import ProviderRuntimeManager
from flameox.application.source import collect_partial_source_state
from flameox.application.workloads import (
    InferenceScenarioConfig,
    InferenceServerConfig,
    ProjectConfig,
    WorkloadService,
    _AIPerfInferenceScenarioConfig,
    _ManagedInferenceServerConfig,
    _SglangBenchInferenceScenarioConfig,
    _SglangInferenceServerConfig,
    _VllmBenchInferenceScenarioConfig,
)
from flameox.command_binding import ExecutableResolver
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    CapabilityExtra,
    CaptureStatus,
    CommandSpec,
    DomainError,
    EnvironmentRecord,
    ErrorCode,
    ExecutionStatus,
    OracleStatus,
    OracleStrength,
    ProcessCancellationCause,
    ProcessTermination,
    RunManifest,
    RunSemantics,
    Sensitivity,
    SourceState,
    ValidationStatus,
    digest_model,
    new_id,
    process_exit_code,
)
from flameox.domain.executables import ResolvedExecutable
from flameox.domain.models import ExecutionRunManifest, UnreportedProcessTermination, utc_now
from flameox.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    ManagedSidecarOutcome,
    ProcessContainment,
    SubprocessBroker,
)
from flameox.filesystem_authority import (
    BoundDirectory,
    BoundDirectoryReference,
    TrustedRoot,
)
from flameox.http_transport import BoundedHttpClient
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, AuthorizedPlanStore, RunStore, Workspace

_PROVIDER_TOOL: dict[
    InferenceScenarioProvider,
    Literal[InferenceTool.AIPERF, InferenceTool.VLLM],
] = {
    InferenceScenarioProvider.AIPERF: InferenceTool.AIPERF,
    InferenceScenarioProvider.VLLM_BENCH: InferenceTool.VLLM,
}


def _configure_inference_server_action(
    name: str,
    server: InferenceServerConfig,
    *,
    configuration_id: str,
) -> ToolAction:
    payload = server.model_dump(mode="json")
    return tool_action(
        ActionId.CONFIGURE_INFERENCE_SERVER,
        name=name,
        operation="replace",
        mode=payload["mode"],
        model=payload["model"],
        provider=payload["provider"],
        benchmark_python=payload.get("benchmark_python"),
        workload=payload.get("workload"),
        base_url=payload["base_url"],
        model_revision=payload.get("model_revision"),
        tokenizer=payload.get("tokenizer"),
        tokenizer_revision=payload.get("tokenizer_revision"),
        quantization=payload.get("quantization"),
        expected_configuration_id=configuration_id,
    )


_MAX_RESULT_LIMITATIONS = 16


class _InferenceScenarioPlan(ContractModel):
    """A validated benchmark intent with an exclusively allocated output authority."""

    plan_id: str
    plan_token: str = ""
    scenario_name: str
    server_name: str
    server_mode: InferenceServerMode
    base_url: str
    model: Annotated[str, Field(min_length=1, max_length=500)]
    model_revision: str | None = None
    tokenizer: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    tokenizer_revision: str | None = None
    quantization: str | None = None
    endpoint_type: InferenceEndpointType = InferenceEndpointType.CHAT
    executable_binding: ResolvedExecutable
    provider_version: str | None = None
    argv: Annotated[tuple[str, ...], Field(min_length=1, max_length=1_024)]
    provider_environment_id: str | None = None
    server_executable_digest: str | None = None
    health_ready: bool | None = None
    probed_model_ids: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    output_root: BoundDirectoryReference
    output_relative_path: Annotated[
        str,
        Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"),
    ]
    output_path: str
    num_prompts: Annotated[int, Field(gt=0, le=10_000_000)] = 1
    concurrency: Annotated[int, Field(gt=0, le=100_000)] | None = None
    request_rate: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    warmup_request_count: Annotated[int, Field(ge=0, le=1_000_000)] = 0
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0
    semantic_oracle_workload: str | None = None
    timeout_seconds: Annotated[float, Field(gt=0, le=86_400)]
    deadline_at: datetime
    exploratory_reason: Annotated[str, Field(min_length=1, max_length=500)]
    configuration_id: str
    created_at: datetime = Field(default_factory=utc_now)


class AIPerfScenarioPlan(_InferenceScenarioPlan):
    provider: Literal[InferenceScenarioProvider.AIPERF]
    server_provider: Literal[InferenceServerProvider.VLLM] = InferenceServerProvider.VLLM
    streaming: bool = True
    trace_artifact_id: str | None = None
    burstiness: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    speedup_ratio: Annotated[float, Field(gt=0, le=100)] = 1.0
    random_input_len: Literal[None] = None
    random_output_len: Literal[None] = None
    random_range_ratio: Literal[None] = None


class VllmBenchScenarioPlan(_InferenceScenarioPlan):
    provider: Literal[InferenceScenarioProvider.VLLM_BENCH]
    server_provider: Literal[InferenceServerProvider.VLLM] = InferenceServerProvider.VLLM
    streaming: Literal[True] = True
    trace_artifact_id: Literal[None] = None
    burstiness: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    speedup_ratio: Annotated[float, Field(ge=1, le=1)] = 1.0
    random_input_len: Literal[None] = None
    random_output_len: Literal[None] = None
    random_range_ratio: Literal[None] = None


class SglangBenchScenarioPlan(_InferenceScenarioPlan):
    provider: Literal[InferenceScenarioProvider.SGLANG_BENCH]
    server_provider: Literal[InferenceServerProvider.SGLANG] = InferenceServerProvider.SGLANG
    benchmark_capabilities: Annotated[tuple[str, ...], Field(min_length=1, max_length=32)]
    streaming: Literal[True] = True
    trace_artifact_id: Literal[None] = None
    burstiness: Literal[None] = None
    speedup_ratio: Annotated[float, Field(ge=1, le=1)] = 1.0
    random_input_len: Annotated[int, Field(gt=0, le=1_000_000)]
    random_output_len: Annotated[int, Field(gt=0, le=1_000_000)]
    random_range_ratio: Annotated[float, Field(gt=0, le=1)] = 1.0


type InferenceScenarioPlan = Annotated[
    AIPerfScenarioPlan | VllmBenchScenarioPlan | SglangBenchScenarioPlan,
    Field(discriminator="provider"),
]

_INFERENCE_SCENARIO_PLAN_ADAPTER: TypeAdapter[InferenceScenarioPlan] = TypeAdapter(
    InferenceScenarioPlan
)


def parse_inference_scenario_plan(value: object) -> InferenceScenarioPlan:
    """Parse a scenario plan into the provider case that can execute it."""

    return _INFERENCE_SCENARIO_PLAN_ADAPTER.validate_python(value)


class InferenceScenarioResult(ContractModel):
    """The bounded outcome of one executed inference benchmark plan."""

    model_config = ConfigDict(json_schema_mode_override="serialization")

    run_id: str
    scenario_name: str
    server_name: str
    provider: InferenceScenarioProvider
    termination: ProcessTermination = Field(default_factory=UnreportedProcessTermination)
    output_path: str | None = None
    output_path_retained: bool = False
    wall_time_ns: Annotated[int, Field(ge=0)] | None = None
    cancellation_cause: ProcessCancellationCause | None = None
    stdout_bytes: Annotated[int, Field(ge=0)] = 0
    stderr_bytes: Annotated[int, Field(ge=0)] = 0
    containment: ProcessContainment = ProcessContainment.BROKER
    peak_rss_backend: str | None = None
    health_ready: bool
    probed_model_ids: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    exploratory_reason: Annotated[str, Field(min_length=1, max_length=500)]
    limitations: Annotated[tuple[str, ...], Field(max_length=16)] = ()
    server_cleanup_complete: bool | None = None
    artifact_ids: tuple[str, ...] = ()
    artifact_run_ids: tuple[str, ...] = ()
    oracle_status: OracleStatus | None = None
    completed_at: datetime = Field(default_factory=utc_now)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def timed_out(self) -> bool:
        return self.cancellation_cause is ProcessCancellationCause.TIMEOUT


class _OracleObservation(ContractModel):
    identity: OracleIdentity
    result: OracleResult | None = None
    validation_status: ValidationStatus = ValidationStatus.UNSUPPORTED
    limitations: tuple[str, ...] = ()


class InferenceScenarioService:
    """Plan and run inference benchmark scenarios against declared local servers."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        broker: SubprocessBroker | None = None,
        probe_timeout_seconds: float = 2.0,
    ) -> None:
        self.workspace = workspace
        self.workloads = WorkloadService(workspace)
        self.artifacts = ArtifactStore(workspace)
        self.runs = RunStore(workspace)
        self.projections = ProjectionCoordinator(workspace)
        self.broker = broker or SubprocessBroker()
        self.probe_timeout_seconds = probe_timeout_seconds
        self.plans = AuthorizedPlanStore(
            workspace,
            family="inference_scenario",
            model=_INFERENCE_SCENARIO_PLAN_ADAPTER,
        )

    def plan(
        self,
        scenario_name: str,
        *,
        timeout_seconds: float | None = None,
        expected_plan_id: str | None = None,
    ) -> InferenceScenarioPlan:
        """Build and authorize a reviewed benchmark scenario."""
        plan = self._build_plan(
            scenario_name,
            timeout_seconds=timeout_seconds,
            expected_plan_id=expected_plan_id,
        )
        authorized = plan.validated_copy(update={"plan_token": secrets.token_hex(32)})
        self.plans.issue(
            authorized.plan_token,
            authorized.plan_id,
            authorized,
            expires_at=authorized.deadline_at,
        )
        return authorized

    def _build_plan(
        self,
        scenario_name: str,
        *,
        timeout_seconds: float | None = None,
        expected_plan_id: str | None = None,
    ) -> InferenceScenarioPlan:
        """Build a validated scenario plan without executing the benchmark tool.

        The plan passively probes an ``existing_local`` server so the agent can
        decide whether to run. Managed servers are probed only after their
        declared workload has started during execution.
        """
        project = self.workloads.load()
        scenario, server = self._resolve(scenario_name, project)
        deadline = self._deadline(timeout_seconds, scenario, server)
        deadline_at = utc_now() + timedelta(seconds=deadline)
        configuration_id = digest_model(project.model_dump(mode="json"))
        if scenario.provider is InferenceScenarioProvider.SGLANG_BENCH:
            if server.benchmark_python is None:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "SGLang benchmark scenarios require a declared benchmark_python launcher.",
                )
            try:
                discovery = discover_sglang(Path(server.benchmark_python), broker=self.broker)
            except DomainError as error:
                raise DomainError(
                    error.code,
                    error.message,
                    retryable=error.retryable,
                    details=error.details,
                    remediation=error.remediation,
                    run_id=error.run_id,
                    next_action=_configure_inference_server_action(
                        scenario.server,
                        server,
                        configuration_id=configuration_id,
                    ),
                ) from error
        else:
            discovery = self._discover_tool(_PROVIDER_TOOL[scenario.provider])
        probe: ExistingServerProbe | None = None
        if server.mode is InferenceServerMode.EXISTING_LOCAL:
            probe = probe_existing_vllm_server(
                server.base_url, timeout_seconds=self.probe_timeout_seconds
            )
            if probe.model_ids and server.model not in probe.model_ids:
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    f"Configured model {server.model!r} is not advertised by the local server.",
                    remediation=(
                        "Update the inference server model identity or start the intended server.",
                    ),
                    details={
                        "server": scenario.server,
                        "configured_model": server.model,
                        "probed_model_ids": probe.model_ids,
                    },
                    next_action=_configure_inference_server_action(
                        scenario.server,
                        server,
                        configuration_id=configuration_id,
                    ),
                )
        server_executable_digest = self._server_tool_identity(server)
        plan_id = digest_model(
            {
                "scenario": scenario.model_dump(mode="json"),
                "server": server.model_dump(mode="json"),
                "provider_executable_binding": discovery.executable_binding.model_dump(mode="json"),
                "provider_version": discovery.version,
                **(
                    {"benchmark_capabilities": discovery.benchmark_capabilities}
                    if scenario.provider is InferenceScenarioProvider.SGLANG_BENCH
                    else {}
                ),
                "provider_environment_id": discovery.provider_environment_id,
                "server_executable_digest": server_executable_digest,
                "timeout_seconds": deadline,
                "configuration_id": configuration_id,
            }
        )
        if expected_plan_id is not None and expected_plan_id != plan_id:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "The inference plan no longer matches the reviewed plan identity.",
                remediation=("Plan the scenario again and review the replacement plan.",),
                details={"expected_plan_id": expected_plan_id, "actual_plan_id": plan_id},
            )
        output_relative_path = (
            "result.jsonl"
            if scenario.provider is InferenceScenarioProvider.SGLANG_BENCH
            else "result.json"
        )
        with TrustedRoot(self.workspace.paths.staging) as trusted_root:
            output = trusted_root.allocate_directory(f"inference-scenario/{secrets.token_hex(16)}")
            try:
                output_path = output.absolute_display_path(output_relative_path)
                request = self._build_request(
                    scenario,
                    server,
                    discovery.executable_binding.canonical_target,
                    output_path=output_path,
                )
                plan = parse_inference_scenario_plan(
                    dict(
                        plan_id=plan_id,
                        plan_token="",
                        scenario_name=scenario_name,
                        server_name=scenario.server,
                        provider=scenario.provider,
                        server_provider=server.provider,
                        server_mode=server.mode,
                        base_url=server.base_url,
                        model=server.model,
                        model_revision=server.model_revision,
                        tokenizer=server.tokenizer,
                        tokenizer_revision=server.tokenizer_revision,
                        quantization=server.quantization,
                        endpoint_type=scenario.endpoint_type,
                        streaming=scenario.streaming,
                        executable_binding=discovery.executable_binding,
                        provider_version=discovery.version,
                        **(
                            {"benchmark_capabilities": discovery.benchmark_capabilities}
                            if scenario.provider is InferenceScenarioProvider.SGLANG_BENCH
                            else {}
                        ),
                        argv=request.argv(),
                        provider_environment_id=discovery.provider_environment_id,
                        server_executable_digest=server_executable_digest,
                        health_ready=probe.health_ready if probe is not None else None,
                        probed_model_ids=probe.model_ids if probe is not None else (),
                        output_root=output.reference,
                        output_relative_path=output_relative_path,
                        output_path=str(output_path),
                        trace_artifact_id=scenario.trace_artifact_id,
                        num_prompts=scenario.num_prompts,
                        concurrency=scenario.concurrency,
                        request_rate=scenario.request_rate,
                        burstiness=scenario.burstiness,
                        warmup_request_count=scenario.warmup_request_count,
                        seed=scenario.seed,
                        speedup_ratio=scenario.speedup_ratio,
                        semantic_oracle_workload=scenario.semantic_oracle_workload,
                        random_input_len=scenario.random_input_len,
                        random_output_len=scenario.random_output_len,
                        random_range_ratio=(
                            scenario.random_range_ratio or 1.0
                            if scenario.provider is InferenceScenarioProvider.SGLANG_BENCH
                            else scenario.random_range_ratio
                        ),
                        timeout_seconds=deadline,
                        deadline_at=deadline_at,
                        exploratory_reason=(
                            "Single benchmark run is exploratory; equivalence or causality "
                            "requires "
                            "a predeclared confirmatory experiment with a semantic oracle."
                        ),
                        configuration_id=configuration_id,
                    )
                )
                return plan
            except BaseException:
                output.close()
                trusted_root.remove_directory(output.reference)
                raise
            finally:
                output.close()

    async def run(
        self,
        plan_token: str,
        *,
        expected_plan_id: str | None = None,
    ) -> InferenceScenarioResult:
        """Consume and execute one server-owned benchmark intent."""
        plan = self.plans.consume(plan_token, expected_digest=expected_plan_id)
        self._validate_plan(plan)
        with (
            TrustedRoot(self.workspace.paths.staging) as trusted_root,
            trusted_root.open_directory(plan.output_root) as output,
        ):
            return await self._run_bound(plan, trusted_root, output)

    async def _run_bound(
        self,
        plan: InferenceScenarioPlan,
        trusted_root: TrustedRoot,
        output: BoundDirectory,
    ) -> InferenceScenarioResult:
        output_path = output.absolute_display_path(plan.output_relative_path)
        environment = await self._managed_environment(plan)
        execution_argv = self._runtime_argv(plan, output)
        run, environment, source_state = self._start_run(
            plan,
            environment=environment,
            execution_argv=execution_argv,
        )
        try:
            outcome, probe, output_path, server_outcome, oracle = await self._execute(plan, output)
        except asyncio.CancelledError:
            artifact_ids, artifact_run_ids, _preserved, _limitations = (
                self._preserve_outputs_safely(plan, output)
            )
            self._finish_cancelled_run(
                run, environment, source_state, artifact_ids, artifact_run_ids
            )
            raise
        except DomainError as error:
            artifact_ids, artifact_run_ids, _preserved, limitations = self._preserve_outputs_safely(
                plan, output
            )
            self._finish_failed_run(
                run,
                environment,
                source_state,
                error,
                artifact_ids,
                artifact_run_ids,
                limitations,
            )
            error.run_id = run.run_id
            if artifact_ids:
                error.details = {
                    **error.details,
                    "partial_artifact_ids": artifact_ids,
                    "partial_artifact_run_ids": artifact_run_ids,
                }
            raise
        except Exception as cause:
            artifact_ids, artifact_run_ids, _preserved, limitations = self._preserve_outputs_safely(
                plan, output
            )
            internal_error = DomainError(
                ErrorCode.INTERNAL_ERROR,
                "Unexpected inference benchmark failure.",
                run_id=run.run_id,
            )
            self._finish_failed_run(
                run,
                environment,
                source_state,
                internal_error,
                artifact_ids,
                artifact_run_ids,
                limitations,
            )
            raise internal_error from cause
        artifact_ids, artifact_run_ids, preserved, preservation_limitations = (
            self._preserve_outputs_safely(plan, output)
        )
        try:
            extraction_limitations = (
                *preservation_limitations,
                *self._extract_outputs(plan, run.run_id, preserved),
            )
        except Exception as cause:
            internal_error = DomainError(
                ErrorCode.INTERNAL_ERROR,
                "Unexpected inference extraction failure.",
                run_id=run.run_id,
            )
            self._finish_failed_run(
                run,
                environment,
                source_state,
                internal_error,
                artifact_ids,
                artifact_run_ids,
                preservation_limitations,
            )
            raise internal_error from cause
        if server_outcome is not None and server_outcome.process.cleanup_complete is not True:
            extraction_limitations = (
                *extraction_limitations,
                "Managed server process cleanup was incomplete.",
            )
        finished = self._finish_run(
            run,
            environment,
            source_state,
            outcome,
            artifact_ids,
            artifact_run_ids,
            (*extraction_limitations, *oracle.limitations),
            oracle,
        )
        cleanup_limitation = self._cleanup_staging(
            trusted_root,
            plan.output_root,
            preservation_complete=not preservation_limitations,
        )
        if cleanup_limitation is not None:
            extraction_limitations = (*extraction_limitations, cleanup_limitation)
            finished = self._append_limitation(
                finished,
                environment,
                source_state,
                cleanup_limitation,
            )
        return self._result(
            plan,
            finished.run_id,
            outcome,
            probe,
            output_path,
            server_outcome,
            artifact_ids,
            artifact_run_ids,
            (*extraction_limitations, *oracle.limitations),
            oracle,
        )

    async def _execute(
        self,
        plan: InferenceScenarioPlan,
        output: BoundDirectory,
    ) -> tuple[
        ExecutionOutcome,
        ExistingServerProbe,
        Path,
        ManagedSidecarOutcome | None,
        _OracleObservation,
    ]:
        if plan.server_mode == "existing_local":
            output_path, server = self._prepare_existing_target(plan, output)
            async with BoundedHttpClient() as http_client:
                probe = await probe_existing_vllm_server_async(
                    server.base_url,
                    timeout_seconds=self.probe_timeout_seconds,
                    http_client=http_client,
                )
            outcome = await self.broker.run(self._request(plan, output))
            oracle = await self._run_oracle(plan, output, outcome)
            return outcome, probe, output_path, None, oracle
        project = self.workloads.load()
        _scenario, server = self._resolve(plan.scenario_name, project)
        if server.workload is None:
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Managed server workload is absent.")
        instance = self.workloads.resolve(server.workload)
        output_path = output.absolute_display_path(plan.output_relative_path)
        parsed = urlsplit(server.base_url)
        assert parsed.hostname is not None
        port = parsed.port or 80
        remaining = (plan.deadline_at - utc_now()).total_seconds()
        if remaining <= 0:
            raise DomainError(ErrorCode.PROCESS_TIMEOUT, "Inference startup deadline expired.")
        absolute_deadline = time.monotonic() + remaining

        command = instance.command
        server_request = ExecutionRequest(
            argv=command.argv,
            executable_binding=instance.executable_binding,
            cwd=Path(command.cwd),
            environment_allowlist=self.workspace.config.execution.child_environment_allowlist,
            environment_overrides=command.env_overrides,
            allowed_working_roots=(self.workspace.project_root,),
            timeout_seconds=remaining,
            max_output_bytes=16 * 1024 * 1024,
        )
        async with BoundedHttpClient() as http_client:

            async def readiness() -> bool:
                try:
                    await probe_existing_vllm_server_async(
                        server.base_url,
                        timeout_seconds=min(self.probe_timeout_seconds, 0.5),
                        http_client=http_client,
                    )
                except DomainError:
                    return False
                return True

            lease = await self.broker.start_inference_server(
                server_request,
                host=parsed.hostname,
                port=port,
                readiness=readiness,
                absolute_deadline=absolute_deadline,
            )
            try:
                probe = await probe_existing_vllm_server_async(
                    server.base_url,
                    timeout_seconds=min(self.probe_timeout_seconds, 0.5),
                    http_client=http_client,
                )
                outcome = await self.broker.run(self._request(plan, output))
                oracle = await self._run_oracle(plan, output, outcome)
            finally:
                server_outcome = await asyncio.shield(lease.close())
        self._write_server_output(output, server_outcome)
        return outcome, probe, output_path, server_outcome, oracle

    @staticmethod
    def _write_server_output(output: BoundDirectory, outcome: ManagedSidecarOutcome) -> None:
        """Stage bounded broker-captured server logs for immutable preservation."""
        if outcome.stdout:
            output.write_bytes("server.stdout", outcome.stdout)
        if outcome.stderr:
            output.write_bytes("server.stderr", outcome.stderr)

    def _prepare_existing_target(
        self,
        plan: InferenceScenarioPlan,
        output: BoundDirectory,
    ) -> tuple[Path, InferenceServerConfig]:
        project = self.workloads.load()
        _scenario, server = self._resolve(plan.scenario_name, project)
        if server.mode is not InferenceServerMode.EXISTING_LOCAL:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Scenario {plan.scenario_name!r} does not target an existing local server.",
                details={"server": plan.server_name, "mode": server.mode},
            )
        output_path = output.absolute_display_path(plan.output_relative_path)
        return output_path, server

    def _request(
        self,
        plan: InferenceScenarioPlan,
        output: BoundDirectory,
    ) -> ExecutionRequest:
        remaining = (plan.deadline_at - utc_now()).total_seconds()
        if remaining <= 0:
            raise DomainError(
                ErrorCode.PROCESS_TIMEOUT,
                "The inference scenario deadline expired before benchmark execution.",
            )
        return ExecutionRequest(
            argv=self._runtime_argv(plan, output),
            executable_binding=plan.executable_binding,
            cwd=self.workspace.project_root,
            environment_allowlist=self.workspace.config.execution.child_environment_allowlist,
            allowed_working_roots=(self.workspace.project_root,),
            timeout_seconds=min(plan.timeout_seconds, remaining),
            max_output_bytes=16 * 1024 * 1024,
            inherited_directory_fds=output.inherited_descriptors(),
        )

    @staticmethod
    def _runtime_argv(
        plan: InferenceScenarioPlan,
        output: BoundDirectory,
    ) -> tuple[str, ...]:
        """Replace display-only output paths with the inherited descriptor path."""

        display_output = plan.output_path
        display_root = str(Path(display_output).parent)
        process_output = str(output.child_process_path(plan.output_relative_path))
        process_root = str(output.child_process_root)
        return tuple(
            process_output
            if argument == display_output
            else process_root
            if argument == display_root
            else argument
            for argument in plan.argv
        )

    def _oracle_request(
        self,
        plan: InferenceScenarioPlan,
        output: BoundDirectory,
    ) -> tuple[OracleIdentity, ExecutionRequest] | None:
        if plan.semantic_oracle_workload is None:
            return None
        oracle = self.workloads.resolve_oracle(plan.semantic_oracle_workload)
        if oracle is None or oracle.receipt_schema is None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "The inference semantic oracle must emit a validated receipt.",
            )
        remaining = (plan.deadline_at - utc_now()).total_seconds()
        if remaining <= 0:
            raise DomainError(
                ErrorCode.PROCESS_TIMEOUT,
                "The inference scenario deadline expired before semantic validation.",
            )
        process_root = output.child_process_root
        receipt_path = output.child_process_path("oracle-receipt.json")
        identity = OracleIdentity(
            kind=(
                "cross_treatment_equivalence"
                if oracle.strength is OracleStrength.CROSS_TREATMENT_EQUIVALENCE
                else "contract_check"
            ),
            estimand=oracle.strength.value,
            command_digest=digest_model(oracle.command.model_dump(mode="json")),
        )
        request = ExecutionRequest(
            argv=oracle.command.argv,
            executable_binding=oracle.executable_binding,
            cwd=Path(oracle.command.cwd),
            environment_allowlist=self.workspace.config.execution.child_environment_allowlist,
            environment_overrides={
                **oracle.command.env_overrides,
                "FLAMEOX_ORACLE_RECEIPT": str(receipt_path),
                "FLAMEOX_INFERENCE_RESULT_DIR": str(process_root),
                "FLAMEOX_INFERENCE_BASE_URL": plan.base_url,
            },
            allowed_working_roots=(self.workspace.project_root,),
            timeout_seconds=min(oracle.command.timeout_seconds, remaining),
            max_output_bytes=16 * 1024 * 1024,
            inherited_directory_fds=output.inherited_descriptors(),
        )
        return identity, request

    async def _run_oracle(
        self,
        plan: InferenceScenarioPlan,
        output: BoundDirectory,
        benchmark: ExecutionOutcome,
    ) -> _OracleObservation:
        prepared = self._oracle_request(plan, output)
        if prepared is None:
            return _OracleObservation(
                identity=OracleIdentity(kind="none"),
                limitations=("No semantic oracle was declared for this inference scenario.",),
            )
        identity, request = prepared
        if process_exit_code(benchmark.process.termination) != 0:
            return _OracleObservation(
                identity=identity,
                limitations=("Semantic validation was skipped because the benchmark failed.",),
            )
        validation = await self.broker.run(request)
        return self._oracle_observation(identity, validation, output)

    @staticmethod
    def _oracle_observation(
        identity: OracleIdentity,
        validation: ExecutionOutcome,
        output: BoundDirectory,
    ) -> _OracleObservation:
        output.write_bytes("oracle.stdout", validation.stdout)
        if validation.stderr:
            output.write_bytes("oracle.stderr", validation.stderr)
        if process_exit_code(validation.process.termination) != 0:
            return _OracleObservation(
                identity=identity,
                validation_status=ValidationStatus.FAILED,
                limitations=("The declared semantic oracle exited unsuccessfully.",),
            )
        try:
            receipt = parse_oracle_receipt(
                output.read_bytes("oracle-receipt.json", max_bytes=1024 * 1024)
            )
        except DomainError as error:
            message = error.message
            return _OracleObservation(
                identity=identity,
                validation_status=ValidationStatus.ERROR,
                limitations=(f"Semantic oracle receipt validation failed: {message}",),
            )
        result = OracleResult(
            status=receipt.status,
            reason=receipt.reason,
            absolute_error=receipt.absolute_error,
            relative_error=receipt.relative_error,
        )
        tolerance = receipt.tolerance
        observed_identity = OracleIdentity.model_validate(
            {
                **identity.model_dump(mode="python"),
                "tolerance_absolute": tolerance.absolute if tolerance is not None else None,
                "tolerance_relative": tolerance.relative if tolerance is not None else None,
            }
        )
        status = {
            OracleStatus.PASS: ValidationStatus.PASSED,
            OracleStatus.FAIL: ValidationStatus.FAILED,
            OracleStatus.INCONCLUSIVE: ValidationStatus.INCONCLUSIVE,
            OracleStatus.UNSUPPORTED: ValidationStatus.UNSUPPORTED,
        }[receipt.status]
        limitations = tuple(receipt.limitations)
        if receipt.status is not OracleStatus.PASS:
            limitations = (*limitations, f"Semantic oracle reported {receipt.status}.")
        return _OracleObservation(
            identity=observed_identity,
            result=result,
            validation_status=status,
            limitations=limitations,
        )

    def _resolve(
        self, scenario_name: str, project: ProjectConfig
    ) -> tuple[InferenceScenarioConfig, InferenceServerConfig]:
        try:
            scenario = project.inference_scenarios[scenario_name]
        except KeyError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Inference scenario {scenario_name!r} is not declared.",
                remediation=("List or configure a typed inference scenario, then retry.",),
                details={"scenario": scenario_name},
                next_action=tool_action(ActionId.LIST_INFERENCE_CONFIGURATIONS),
            ) from exc
        try:
            server = project.inference_servers[scenario.server]
        except KeyError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Inference scenario {scenario_name!r} references unknown server "
                f"{scenario.server!r}.",
                remediation=("Configure the referenced inference server, then retry.",),
                details={
                    "scenario": scenario_name,
                    "server": scenario.server,
                },
                next_action=manual_action(
                    "Supply a complete declaration for the referenced inference server.",
                    suggested_action=ActionId.CONFIGURE_INFERENCE_SERVER,
                    missing_arguments=("operation", "mode", "model"),
                ),
            ) from exc
        return scenario, server

    def _server_tool_identity(self, server: InferenceServerConfig) -> str | None:
        if not isinstance(server, _ManagedInferenceServerConfig):
            return None
        workload = self.workloads.resolve(server.workload)
        binding = workload.executable_binding
        if binding is None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "A managed inference workload is missing its executable binding.",
            )
        return binding.identity.sha256

    def _discover_tool(
        self,
        tool: Literal[InferenceTool.AIPERF, InferenceTool.VLLM],
    ) -> QualifiedInferenceTool:
        runtime = (
            ProviderRuntimeManager(
                self.workspace.paths.records / "provider-runtimes",
                broker=self.broker,
            ).find(
                extra=CapabilityExtra.INFERENCE,
                requirement="aiperf>=0.12,<0.13",
            )
            if tool is InferenceTool.AIPERF
            else None
        )
        return discover_inference_tool(tool, provider_runtime=runtime)

    def _validate_plan(self, plan: InferenceScenarioPlan) -> None:
        output_parts = plan.output_root.parts()
        if (
            len(output_parts) != 2
            or output_parts[0] != "inference-scenario"
            or len(output_parts[1]) != 32
            or any(character not in "0123456789abcdef" for character in output_parts[1])
        ):
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Inference output authority is not an allocated scenario directory.",
            )
        expected_output_path = self.workspace.paths.staging.joinpath(
            *output_parts,
            plan.output_relative_path,
        ).absolute()
        if Path(plan.output_path).absolute() != expected_output_path:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Inference output display path does not match its bound authority.",
            )
        project = self.workloads.load()
        configuration_id = digest_model(project.model_dump(mode="json"))
        if configuration_id != plan.configuration_id:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Inference configuration changed after this plan was created.",
                remediation=("Plan the inference scenario again, then retry execution.",),
            )
        try:
            ExecutableResolver().revalidate(plan.executable_binding)
        except DomainError as error:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Inference provider executable changed after planning.",
                remediation=("Plan the inference scenario again, then retry execution.",),
            ) from error
        if plan.provider_environment_id is not None:
            runtime = ProviderRuntimeManager(
                self.workspace.paths.records / "provider-runtimes",
                broker=self.broker,
            ).get(plan.provider_environment_id)
            if (
                runtime is None
                or runtime.executable is None
                or runtime.executable.resolve() != plan.executable_binding.canonical_target
            ):
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "The qualified inference provider environment changed after planning.",
                    remediation=("Plan the inference scenario again, then retry execution.",),
                )
        _scenario, server = self._resolve(plan.scenario_name, project)
        server_digest = self._server_tool_identity(server)
        if server_digest != plan.server_executable_digest:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Managed server executable identity changed after planning.",
                remediation=("Plan the inference scenario again, then retry execution.",),
            )

    def _deadline(
        self,
        timeout_seconds: float | None,
        scenario: InferenceScenarioConfig,
        server: InferenceServerConfig,
    ) -> float:
        if timeout_seconds is not None:
            if timeout_seconds <= 0 or timeout_seconds > 86_400:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "timeout_seconds must be positive and at most 86400.",
                    details={"timeout_seconds": timeout_seconds},
                )
            return timeout_seconds
        # One absolute deadline for the whole scenario. AIPerf trace replay can be
        # long, so default to a generous bounded value rather than the workload
        # default.
        return 1800.0 if scenario.provider is InferenceScenarioProvider.AIPERF else 600.0

    def _build_request(
        self,
        scenario: InferenceScenarioConfig,
        server: InferenceServerConfig,
        executable: Path,
        *,
        output_path: Path,
    ) -> AIPerfProfileRequest | VllmBenchServeRequest | SglangBenchServingRequest:
        # AIPerf's fixed_schedule requires a Mooncake trace; without a declared
        # trace_artifact_id the benchmark uses the tool's default schedule.
        fixed_schedule = scenario.trace_artifact_id is not None
        trace_path = (
            self.artifacts.get(scenario.trace_artifact_id).payload_path
            if scenario.trace_artifact_id is not None
            else None
        )
        if isinstance(scenario, _AIPerfInferenceScenarioConfig):
            return AIPerfProfileRequest(
                executable=executable,
                base_url=server.base_url,
                model=server.model,
                tokenizer=server.tokenizer,
                endpoint_type=scenario.endpoint_type,
                streaming=scenario.streaming,
                trace_path=trace_path,
                output_dir=output_path.parent,
                fixed_schedule=fixed_schedule,
                concurrency=scenario.concurrency,
                request_rate=scenario.request_rate,
                burstiness=scenario.burstiness,
                warmup_request_count=scenario.warmup_request_count,
                seed=scenario.seed,
                request_count=scenario.num_prompts if trace_path is None else None,
                speedup_ratio=scenario.speedup_ratio,
            )
        if isinstance(scenario, _SglangBenchInferenceScenarioConfig):
            if not isinstance(server, _SglangInferenceServerConfig):
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "sglang_bench scenarios require an sglang inference server",
                )
            return SglangBenchServingRequest(
                executable=executable,
                base_url=server.base_url,
                model=server.model,
                tokenizer=server.tokenizer,
                endpoint_type=scenario.endpoint_type,
                streaming=scenario.streaming,
                num_prompts=scenario.num_prompts,
                random_input_len=scenario.random_input_len,
                random_output_len=scenario.random_output_len,
                random_range_ratio=scenario.random_range_ratio or 1.0,
                request_rate=scenario.request_rate,
                max_concurrency=scenario.concurrency,
                warmup_request_count=scenario.warmup_request_count,
                seed=scenario.seed,
                result_path=output_path,
            )
        if isinstance(scenario, _VllmBenchInferenceScenarioConfig):
            return VllmBenchServeRequest(
                executable=executable,
                base_url=server.base_url,
                model=server.model,
                endpoint_type=scenario.endpoint_type,
                streaming=scenario.streaming,
                num_prompts=scenario.num_prompts,
                request_rate=scenario.request_rate,
                burstiness=scenario.burstiness,
                max_concurrency=scenario.concurrency,
                warmup_request_count=scenario.warmup_request_count,
                seed=scenario.seed,
                result_path=output_path,
            )
        assert_never(scenario)

    def _result(
        self,
        plan: InferenceScenarioPlan,
        run_id: str,
        outcome: ExecutionOutcome,
        probe: ExistingServerProbe,
        output_path: Path,
        server_outcome: ManagedSidecarOutcome | None,
        artifact_ids: tuple[str, ...],
        artifact_run_ids: tuple[str, ...],
        extraction_limitations: tuple[str, ...],
        oracle: _OracleObservation,
    ) -> InferenceScenarioResult:
        process = outcome.process
        limitations: list[str] = []
        if not probe.health_ready:
            limitations.append("Server health probe was not ready before the benchmark.")
        if not probe.model_ids:
            limitations.append("Server exposed no model ids before the benchmark.")
        if process.timed_out:
            limitations.append("The benchmark was terminated by the absolute deadline.")
        limitations.extend(extraction_limitations)
        limitations.append(plan.exploratory_reason)
        return InferenceScenarioResult(
            run_id=run_id,
            scenario_name=plan.scenario_name,
            server_name=plan.server_name,
            provider=plan.provider,
            output_path=str(output_path),
            output_path_retained=output_path.parent.exists(),
            termination=process.termination,
            wall_time_ns=process.wall_time_ns,
            cancellation_cause=process.cancellation_cause,
            stdout_bytes=len(outcome.stdout),
            stderr_bytes=len(outcome.stderr),
            containment=outcome.containment,
            peak_rss_backend=outcome.peak_rss_backend,
            health_ready=probe.health_ready,
            probed_model_ids=probe.model_ids,
            exploratory_reason=plan.exploratory_reason,
            limitations=self._bounded_result_limitations(limitations),
            server_cleanup_complete=(
                server_outcome.process.cleanup_complete if server_outcome is not None else None
            ),
            artifact_ids=artifact_ids,
            artifact_run_ids=artifact_run_ids,
            oracle_status=oracle.result.status if oracle.result is not None else None,
        )

    @staticmethod
    def _bounded_result_limitations(limitations: list[str]) -> tuple[str, ...]:
        unique = tuple(dict.fromkeys(limitations))
        if len(unique) <= _MAX_RESULT_LIMITATIONS:
            return unique
        omitted = len(unique) - (_MAX_RESULT_LIMITATIONS - 1)
        return (
            *unique[: _MAX_RESULT_LIMITATIONS - 1],
            f"{omitted} additional limitations are recorded on the canonical run.",
        )

    async def _managed_environment(self, plan: InferenceScenarioPlan) -> EnvironmentRecord:
        if plan.server_mode != "managed":
            return collect_environment()
        remaining = (plan.deadline_at - utc_now()).total_seconds()
        if remaining <= 0:
            return collect_environment()
        try:
            accelerator = await asyncio.wait_for(
                AcceleratorIdentityService(
                    self.workspace,
                ).observe(("cuda.driver", "cuda.runtime", "cuda.devices", "cuda.peer_topology")),
                timeout=remaining,
            )
        except TimeoutError:
            return collect_environment()
        return collect_environment(accelerator)

    def _start_run(
        self,
        plan: InferenceScenarioPlan,
        *,
        environment: EnvironmentRecord | None = None,
        execution_argv: tuple[str, ...] | None = None,
    ) -> tuple[RunManifest, EnvironmentRecord, SourceState]:
        """Create the canonical execution run that owns normalized benchmark evidence."""
        environment = environment or collect_environment()
        executable = plan.executable_binding.canonical_target
        source_state = collect_partial_source_state(self.workspace, executable=executable)
        protocol = self._protocol_identity(plan, environment=environment)
        protocol_json = json.dumps(
            protocol.model_dump(mode="json"),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        protocol_id = digest_model(protocol.model_dump(mode="json"))
        run = ExecutionRunManifest(
            run_id=new_id(),
            started_at=utc_now(),
            execution_status=ExecutionStatus.RUNNING,
            capture_status=CaptureStatus.RUNNING,
            validation_status=ValidationStatus.UNSUPPORTED,
            workload_definition_id=protocol_id,
            workload_instance_id=plan.plan_id,
            measurement_protocol_id=protocol_id,
            environment_id=environment.environment_id,
            source_state_id=source_state.source_state_id,
            semantics=RunSemantics(
                origin="internal",
                adapter=plan.provider,
                adapter_version=plan.provider_version,
                configuration={"protocol_id": protocol_id},
            ),
            command=CommandSpec(
                argv=execution_argv or plan.argv,
                cwd=str(self.workspace.project_root),
                timeout_seconds=plan.timeout_seconds,
            ),
            inference_protocol_identity_id=protocol_id,
            inference_protocol_identity_json=protocol_json,
            limitations=(plan.exploratory_reason,),
        )
        projected = self.projections.create_run(
            run,
            environment=environment,
            source_state=source_state,
        )
        return projected.run, environment, source_state

    def _protocol_identity(
        self,
        plan: InferenceScenarioPlan,
        *,
        environment: EnvironmentRecord,
    ) -> InferenceProtocolIdentity:
        trace_digest = None
        if plan.trace_artifact_id is not None:
            trace_digest = self.artifacts.get(plan.trace_artifact_id).content.artifact_id
        else:
            trace_digest = digest_model(
                {
                    "kind": "synthetic",
                    "provider": plan.provider,
                    "provider_executable_digest": plan.executable_binding.identity.sha256,
                    "request_count": plan.num_prompts,
                    "endpoint_type": plan.endpoint_type,
                    "streaming": plan.streaming,
                    "model": plan.model,
                    "tokenizer": plan.tokenizer,
                    "random_input_len": plan.random_input_len,
                    "random_output_len": plan.random_output_len,
                    "random_range_ratio": plan.random_range_ratio,
                }
            )
        managed_command_digest = None
        if plan.server_mode == "managed":
            project = self.workloads.load()
            _scenario, server = self._resolve(plan.scenario_name, project)
            if not isinstance(server, _ManagedInferenceServerConfig):
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "Inference server lifecycle changed after planning.",
                )
            instance = self.workloads.resolve(server.workload)
            managed_command_digest = digest_model(instance.command.model_dump(mode="json"))
        oracle_identity = OracleIdentity(kind="none")
        if plan.semantic_oracle_workload is not None:
            resolved_oracle = self.workloads.resolve_oracle(plan.semantic_oracle_workload)
            if resolved_oracle is not None:
                oracle_identity = OracleIdentity(
                    kind=(
                        "cross_treatment_equivalence"
                        if resolved_oracle.strength is OracleStrength.CROSS_TREATMENT_EQUIVALENCE
                        else "contract_check"
                    ),
                    estimand=resolved_oracle.strength.value,
                    command_digest=digest_model(resolved_oracle.command.model_dump(mode="json")),
                )
        return InferenceProtocolIdentity(
            provider=plan.provider,
            provider_version=plan.provider_version,
            provider_executable_digest=plan.executable_binding.identity.sha256,
            provider_environment_id=plan.provider_environment_id,
            trace=TraceIdentity(
                format=(
                    "mooncake"
                    if plan.trace_artifact_id is not None
                    else "aiperf"
                    if plan.provider is InferenceScenarioProvider.AIPERF
                    else "sglang.benchmark.serving"
                    if plan.provider is InferenceScenarioProvider.SGLANG_BENCH
                    else "vllm"
                ),
                producer=(
                    "mooncake"
                    if plan.trace_artifact_id is not None
                    else "aiperf"
                    if plan.provider is InferenceScenarioProvider.AIPERF
                    else "sglang.benchmark.serving"
                    if plan.provider is InferenceScenarioProvider.SGLANG_BENCH
                    else "vllm"
                ),
                producer_version=(
                    None if plan.trace_artifact_id is not None else plan.provider_version
                ),
                artifact_digest=trace_digest,
                request_count=plan.num_prompts if plan.trace_artifact_id is None else None,
            ),
            schedule=ScheduleIdentity(
                preserve_timing=(plan.trace_artifact_id is not None and plan.speedup_ratio == 1.0),
                time_scale=plan.speedup_ratio,
                max_concurrency=plan.concurrency,
                request_rate=plan.request_rate,
                burstiness=plan.burstiness,
                warmup_request_count=plan.warmup_request_count,
                seed=plan.seed,
            ),
            model=ModelIdentity(
                model_id=plan.model,
                model_revision=plan.model_revision,
                tokenizer_id=plan.tokenizer or plan.model,
                tokenizer_revision=plan.tokenizer_revision or plan.model_revision,
                quantization=plan.quantization,
            ),
            server=ServerConfigIdentity(
                backend=plan.server_provider.value,
                cache_backend=(
                    "custom"
                    if plan.server_provider is InferenceServerProvider.SGLANG
                    else "vllm_paged"
                ),
                endpoint=(
                    "/v1/chat/completions"
                    if plan.endpoint_type is InferenceEndpointType.CHAT
                    else "/v1/completions"
                ),
                managed_server_command_digest=managed_command_digest,
                server_executable_digest=plan.server_executable_digest,
                server_version=None,
            ),
            hardware=self._hardware_identity(environment),
            profiler=ProfilerState(),
            oracle=oracle_identity,
        )

    @staticmethod
    def _hardware_identity(environment: EnvironmentRecord) -> HardwareIdentity:
        value = environment.fields.get("accelerator")
        if not isinstance(value, dict) or value.get("status") != "available":
            return HardwareIdentity()
        devices = value.get("devices")
        typed_devices = devices if isinstance(devices, list) else []
        models = {
            str(device["model"])
            for device in typed_devices
            if isinstance(device, dict) and device.get("model")
        }
        return HardwareIdentity(
            accelerator_kind="cuda",
            accelerator_count=len(typed_devices),
            accelerator_model=next(iter(models)) if len(models) == 1 else None,
            driver_version=(str(value["driver_version"]) if value.get("driver_version") else None),
            runtime_version=(
                str(value["runtime_version"]) if value.get("runtime_version") else None
            ),
            topology_digest=digest_model(
                {
                    "devices": typed_devices,
                    "links": value.get("links", []),
                }
            ),
        )

    def _finish_run(
        self,
        run: RunManifest,
        environment: EnvironmentRecord,
        source_state: SourceState,
        outcome: ExecutionOutcome,
        artifact_ids: tuple[str, ...],
        artifact_run_ids: tuple[str, ...],
        extraction_limitations: tuple[str, ...],
        oracle: _OracleObservation,
    ) -> RunManifest:
        process = outcome.process
        status = (
            ExecutionStatus.TIMED_OUT
            if process.timed_out
            else ExecutionStatus.SUCCEEDED
            if process_exit_code(process.termination) == 0
            else ExecutionStatus.FAILED
        )
        limitations = tuple(dict.fromkeys((*run.limitations, *extraction_limitations)))
        initial_protocol = InferenceProtocolIdentity.model_validate_json(
            run.inference_protocol_identity_json or "{}"
        )
        protocol = InferenceProtocolIdentity.model_validate(
            {
                **initial_protocol.model_dump(mode="python"),
                "oracle": oracle.identity,
                "oracle_result": oracle.result,
            }
        )
        protocol_json = json.dumps(
            protocol.model_dump(mode="json"),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        protocol_id = digest_model(protocol.model_dump(mode="json"))
        registrations = self._canonical_registrations(run.run_id, artifact_run_ids)
        finished = run.validated_copy(
            update={
                "revision": 1,
                "finished_at": utc_now(),
                "execution_status": status,
                "capture_status": (
                    CaptureStatus.REGISTERED if artifact_ids else CaptureStatus.FAILED
                ),
                "process": process,
                "artifacts": registrations,
                "limitations": limitations,
                "validation_status": oracle.validation_status,
                "inference_protocol_identity_id": protocol_id,
                "inference_protocol_identity_json": protocol_json,
            }
        )
        return self.projections.append_run(
            finished,
            expected_revision=0,
            environment=environment,
            source_state=source_state,
        ).run

    def _finish_cancelled_run(
        self,
        run: RunManifest,
        environment: EnvironmentRecord,
        source_state: SourceState,
        artifact_ids: tuple[str, ...],
        artifact_run_ids: tuple[str, ...],
    ) -> RunManifest:
        registrations = self._canonical_registrations(run.run_id, artifact_run_ids)
        finished = run.validated_copy(
            update={
                "revision": 1,
                "finished_at": utc_now(),
                "execution_status": ExecutionStatus.CANCELLED,
                "capture_status": (
                    CaptureStatus.REGISTERED if artifact_ids else CaptureStatus.CANCELLED
                ),
                "artifacts": registrations,
                "limitations": tuple(
                    dict.fromkeys((*run.limitations, "Inference benchmark was cancelled."))
                ),
            }
        )
        return self.projections.append_run(
            finished,
            expected_revision=0,
            environment=environment,
            source_state=source_state,
        ).run

    def _finish_failed_run(
        self,
        run: RunManifest,
        environment: EnvironmentRecord,
        source_state: SourceState,
        error: DomainError,
        artifact_ids: tuple[str, ...],
        artifact_run_ids: tuple[str, ...],
        extra_limitations: tuple[str, ...] = (),
    ) -> RunManifest:
        registrations = self._canonical_registrations(run.run_id, artifact_run_ids)
        finished = run.validated_copy(
            update={
                "revision": 1,
                "finished_at": utc_now(),
                "execution_status": (
                    ExecutionStatus.TIMED_OUT
                    if error.code is ErrorCode.PROCESS_TIMEOUT
                    else ExecutionStatus.FAILED
                ),
                "capture_status": (
                    CaptureStatus.REGISTERED if artifact_ids else CaptureStatus.FAILED
                ),
                "artifacts": registrations,
                "limitations": tuple(
                    dict.fromkeys((*run.limitations, error.message, *extra_limitations))
                ),
            }
        )
        return self.projections.append_run(
            finished,
            expected_revision=0,
            environment=environment,
            source_state=source_state,
        ).run

    def _canonical_registrations(
        self, run_id: str, artifact_run_ids: tuple[str, ...]
    ) -> tuple[ArtifactRegistration, ...]:
        registrations: list[ArtifactRegistration] = []
        for artifact_run_id in artifact_run_ids:
            source = self.runs.read(artifact_run_id)
            registrations.extend(
                ArtifactRegistration.model_validate(
                    {
                        **registration.model_dump(mode="python"),
                        "registration_id": new_id(),
                        "run_id": run_id,
                    }
                )
                for registration in source.artifacts
            )
        return tuple(registrations)

    def _append_limitation(
        self,
        run: RunManifest,
        environment: EnvironmentRecord,
        source_state: SourceState,
        limitation: str,
    ) -> RunManifest:
        updated = run.validated_copy(
            update={
                "revision": run.revision + 1,
                "limitations": tuple(dict.fromkeys((*run.limitations, limitation))),
            }
        )
        return self.projections.append_run(
            updated,
            expected_revision=run.revision,
            environment=environment,
            source_state=source_state,
        ).run

    def _extract_outputs(
        self,
        plan: InferenceScenarioPlan,
        evidence_run_id: str,
        preserved: tuple[tuple[str, str, str], ...],
    ) -> tuple[str, ...]:
        """Normalize supported provider output without making native preservation contingent."""
        from flameox.adapters.inference import InferenceArtifactExtractor

        aiperf_runtime = None
        if plan.provider is InferenceScenarioProvider.AIPERF:
            if plan.provider_environment_id is None:
                return ("AIPerf provider environment identity is absent from the plan.",)
            aiperf_runtime = ProviderRuntimeManager(
                self.workspace.paths.records / "provider-runtimes"
            ).get(plan.provider_environment_id)
            if aiperf_runtime is None:
                return (
                    "The exact AIPerf provider environment bound during planning is unavailable; "
                    "no substitute parser was used.",
                )
        extractor = InferenceArtifactExtractor(
            self.workspace,
            aiperf_runtime=aiperf_runtime,
        )
        limitations: list[str] = []
        try:
            if plan.provider in {
                InferenceScenarioProvider.VLLM_BENCH,
                InferenceScenarioProvider.SGLANG_BENCH,
            }:
                result_artifact = next(
                    (item for item in preserved if item[0] == plan.output_relative_path),
                    None,
                )
                if result_artifact is not None:
                    result = (
                        extractor.extract_sglang_result
                        if plan.provider is InferenceScenarioProvider.SGLANG_BENCH
                        else extractor.extract_vllm_result
                    )(result_artifact[1], evidence_run_id=evidence_run_id)
                    return result.limitations
                return ("Inference benchmark result was not emitted.",)
            export = next((item for item in preserved if item[0] == "profile_export.jsonl"), None)
            inputs = next((item for item in preserved if item[0] == "inputs.json"), None)
            if export is None:
                return ("AIPerf profile_export.jsonl was not emitted; request evidence is absent.",)
            result = extractor.extract_aiperf_result(
                export[1],
                evidence_run_id=evidence_run_id,
                inputs_run_id=inputs[1] if inputs is not None else None,
            )
            limitations.extend(result.limitations)
        except DomainError as error:
            limitations.append(f"Provider output extraction was incomplete: {error.message}")
        return tuple(limitations)

    def _preserve_outputs(
        self,
        plan: InferenceScenarioPlan,
        output: BoundDirectory,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[tuple[str, str, str], ...],
        tuple[str, ...],
    ]:
        admitted_basenames = {
            plan.output_relative_path,
            "oracle-receipt.json",
            "oracle.stdout",
            "oracle.stderr",
            "server.stdout",
            "server.stderr",
        }
        if plan.provider is InferenceScenarioProvider.AIPERF:
            admitted_basenames.update({"inputs.json", "profile_export.jsonl"})
        candidates = output.admitted_files(
            frozenset(admitted_basenames),
            max_depth=2,
            max_entries=4_096,
            max_files=32,
        )
        artifacts: list[str] = []
        runs: list[str] = []
        preserved: list[tuple[str, str, str]] = []
        limitations: list[str] = []
        importer = ImportService(self.workspace)
        for candidate in candidates:
            display_name = Path(candidate.relative_path).name
            is_inputs = (
                plan.provider is InferenceScenarioProvider.AIPERF and display_name == "inputs.json"
            )
            is_oracle = display_name.startswith("oracle")
            is_server_output = display_name.startswith("server.")
            try:
                with output.open_file(candidate) as descriptor:
                    imported = importer.import_descriptor(
                        ImportDescriptorRequest(
                            descriptor=descriptor,
                            display_name=display_name,
                            kind=(
                                ArtifactKind.INFERENCE_REQUEST_TRACE
                                if is_inputs
                                else ArtifactKind.VALIDATION_OUTPUT
                                if is_oracle and display_name != "oracle.stderr"
                                else ArtifactKind.PROCESS_OUTPUT
                                if is_oracle or is_server_output
                                else ArtifactKind.INFERENCE_RESULT
                            ),
                            media_type=self._provider_media_type(display_name),
                            sensitivity=(
                                Sensitivity.SENSITIVE
                                if plan.provider
                                in {
                                    InferenceScenarioProvider.AIPERF,
                                    InferenceScenarioProvider.SGLANG_BENCH,
                                }
                                or is_oracle
                                or is_server_output
                                else Sensitivity.INTERNAL
                            ),
                            role=(
                                "inference_oracle_output"
                                if is_oracle
                                else "inference_server_output"
                                if is_server_output
                                else "inference_provider_output"
                            ),
                            producer=(
                                (plan.semantic_oracle_workload or "inference_oracle")
                                if is_oracle
                                else plan.provider
                            ),
                            producer_version=(
                                "flameox.oracle-receipt.v1" if is_oracle else plan.provider_version
                            ),
                        )
                    )
            except DomainError as error:
                limitations.append(
                    f"Provider artifact {display_name!r} could not be preserved: {error.message}"
                )
                continue
            artifacts.append(imported.artifact_id)
            runs.append(imported.run.run_id)
            preserved.append((display_name, imported.run.run_id, imported.artifact_id))
        return tuple(artifacts), tuple(runs), tuple(preserved), tuple(limitations)

    def _preserve_outputs_safely(
        self,
        plan: InferenceScenarioPlan,
        output: BoundDirectory,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[tuple[str, str, str], ...],
        tuple[str, ...],
    ]:
        try:
            return self._preserve_outputs(plan, output)
        except DomainError as error:
            return (), (), (), (f"Provider artifact preservation failed: {error.message}",)

    @staticmethod
    def _provider_media_type(display_name: str) -> str:
        return {
            ".json": "application/json",
            ".jsonl": "application/x-ndjson",
            ".csv": "text/csv",
            ".log": "text/plain",
            ".txt": "text/plain",
        }.get(Path(display_name).suffix.lower(), "application/octet-stream")

    @staticmethod
    def _cleanup_staging(
        trusted_root: TrustedRoot,
        reference: BoundDirectoryReference,
        *,
        preservation_complete: bool,
    ) -> str | None:
        if not preservation_complete:
            return (
                "Provider staging was retained because native artifact preservation was incomplete."
            )
        try:
            trusted_root.remove_directory(reference)
        except DomainError:
            return "Provider staging cleanup failed; immutable artifacts remain authoritative."
        return None
