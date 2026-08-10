"""Plan and run inference replay scenarios against declared local servers.

The service loads a declared scenario and server from ``flameox.toml``, either
leases a managed vLLM workload or passively probes an existing loopback server,
constructs the typed AIPerf or vLLM bench command, and executes it through the
canonical ``SubprocessBroker`` under one absolute deadline. The broker owns
containment, quotas, cancellation, and process snapshots; this module never
adds an inference request client or unrestricted command surface.

Replay measurements are exploratory by default: a single replay run does not
establish equivalence or causality. The plan and result preserve the exact
rendered argv, provider/tool identity, resolved output paths, and an explicit
exploratory reason so a later confirmatory experiment can reference the same
inputs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, assert_never
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, TypeAdapter, computed_field, model_validator

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
from flameox.application.evidence_rows import (
    artifact_registration_row,
    environment_row,
    source_state_row,
)
from flameox.application.imports import ImportArtifactRequest, ImportService
from flameox.application.inference_providers import (
    AIPerfProfileRequest,
    ExistingServerProbe,
    SglangBenchServingRequest,
    VllmBenchServeRequest,
    discover_inference_tool,
    discover_sglang,
    probe_existing_vllm_server,
)
from flameox.application.oracle_receipts import parse_oracle_receipt
from flameox.application.run_rows import run_row
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
from flameox.atomic import atomic_write_bytes
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    CaptureStatus,
    CommandSpec,
    DomainError,
    EnvironmentRecord,
    ErrorCode,
    ExecutionStatus,
    OracleStatus,
    OracleStrength,
    ProcessCancellationCause,
    RunManifest,
    Sensitivity,
    SourceState,
    ValidationStatus,
    digest_model,
    new_id,
)
from flameox.domain.models import ExecutionRunManifest, utc_now
from flameox.evidence import GenerationPublisher
from flameox.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    ManagedSidecarOutcome,
    ProcessContainment,
    SubprocessBroker,
)
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, RunStore, Workspace

_PROVIDER_TOOL: dict[str, Literal["aiperf", "vllm"]] = {
    "aiperf": "aiperf",
    "vllm_bench": "vllm",
    "sglang_bench": "vllm",
}
_MAX_RESULT_LIMITATIONS = 16


class _InferenceReplayPlan(ContractModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    """A validated, side-effect-free plan for one inference replay run."""

    schema_version: Literal[1] = 1
    plan_id: str
    scenario_name: str
    server_name: str
    server_mode: Literal["managed", "existing_local"]
    base_url: str
    model: Annotated[str, Field(min_length=1, max_length=500)]
    model_revision: str | None = None
    tokenizer: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    tokenizer_revision: str | None = None
    quantization: str | None = None
    endpoint_type: Literal["chat", "completions"] = "chat"
    tool_executable: str | None = None
    tool_version: str | None = None
    tool_executable_digest: str | None = None
    server_executable_digest: str | None = None
    server_version: str | None = None
    tool_compatibility_reason: str | None = None
    tool_remediation: Annotated[tuple[str, ...], Field(max_length=8)] = ()
    health_ready: bool | None = None
    probed_model_ids: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    argv: Annotated[tuple[str, ...], Field(max_length=1_024)] = ()
    output_path: str | None = None
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

    @model_validator(mode="before")
    @classmethod
    def parse_tool_readiness_projections(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        available = value.get("tool_available")
        compatible = value.get("tool_compatible")
        if available is not None and compatible is not None and available != compatible:
            raise ValueError("tool availability and compatibility must agree")
        parsed = dict(value)
        parsed.pop("tool_available", None)
        parsed.pop("tool_compatible", None)
        planned = bool(parsed.get("argv", ()))
        if available is not None and available != planned:
            raise ValueError("tool availability must match executable argv")
        if compatible is not None and compatible != planned:
            raise ValueError("tool compatibility must match executable argv")
        return parsed

    @model_validator(mode="after")
    def tool_readiness_is_coherent(self) -> _InferenceReplayPlan:
        if self.argv and self.tool_executable is None:
            raise ValueError("an available tool requires an executable")
        if self.argv and (self.tool_compatibility_reason is not None or self.tool_remediation):
            raise ValueError("an available tool cannot carry incompatibility recovery")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tool_available(self) -> bool:
        return bool(self.argv)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tool_compatible(self) -> bool:
        return self.tool_available


class AIPerfReplayPlan(_InferenceReplayPlan):
    provider: Literal["aiperf"]
    server_provider: Literal["vllm"] = "vllm"
    streaming: bool = True
    trace_artifact_id: str | None = None
    burstiness: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    speedup_ratio: Annotated[float, Field(gt=0, le=100)] = 1.0
    random_input_len: Literal[None] = None
    random_output_len: Literal[None] = None
    random_range_ratio: Literal[None] = None


class VllmBenchReplayPlan(_InferenceReplayPlan):
    provider: Literal["vllm_bench"]
    server_provider: Literal["vllm"] = "vllm"
    streaming: Literal[True] = True
    trace_artifact_id: Literal[None] = None
    burstiness: Annotated[float, Field(gt=0, le=1_000_000)] | None = None
    speedup_ratio: Annotated[float, Field(ge=1, le=1)] = 1.0
    random_input_len: Literal[None] = None
    random_output_len: Literal[None] = None
    random_range_ratio: Literal[None] = None


class SglangBenchReplayPlan(_InferenceReplayPlan):
    provider: Literal["sglang_bench"]
    server_provider: Literal["sglang"] = "sglang"
    streaming: Literal[True] = True
    trace_artifact_id: Literal[None] = None
    burstiness: Literal[None] = None
    speedup_ratio: Annotated[float, Field(ge=1, le=1)] = 1.0
    random_input_len: Annotated[int, Field(gt=0, le=1_000_000)]
    random_output_len: Annotated[int, Field(gt=0, le=1_000_000)]
    random_range_ratio: Annotated[float, Field(gt=0, le=1)] = 1.0


type InferenceReplayPlan = Annotated[
    AIPerfReplayPlan | VllmBenchReplayPlan | SglangBenchReplayPlan,
    Field(discriminator="provider"),
]

_INFERENCE_REPLAY_PLAN_ADAPTER: TypeAdapter[InferenceReplayPlan] = TypeAdapter(InferenceReplayPlan)


def parse_inference_replay_plan(value: object) -> InferenceReplayPlan:
    """Parse a replay plan into the provider case that can execute it."""

    return _INFERENCE_REPLAY_PLAN_ADAPTER.validate_python(value)


class InferenceReplayResult(ContractModel):
    """The bounded outcome of one executed inference replay plan."""

    model_config = ConfigDict(json_schema_mode_override="serialization")

    schema_version: Literal[1] = 1
    result_id: str
    run_id: str
    plan_id: str
    scenario_name: str
    server_name: str
    provider: Literal["aiperf", "vllm_bench", "sglang_bench"]
    argv: Annotated[tuple[str, ...], Field(max_length=1_024)]
    output_path: str | None = None
    output_path_retained: bool = False
    exit_code: int | None = None
    terminating_signal: int | None = None
    wall_time_ns: Annotated[int, Field(ge=0)] | None = None
    cancellation_cause: ProcessCancellationCause | None = None
    stdout_bytes: Annotated[int, Field(ge=0)] = 0
    stderr_bytes: Annotated[int, Field(ge=0)] = 0
    containment: ProcessContainment = "broker"
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

    @model_validator(mode="before")
    @classmethod
    def parse_legacy_timed_out(cls, value: object) -> object:
        if not isinstance(value, dict) or "timed_out" not in value:
            return value
        timed_out = value["timed_out"]
        cause = value.get("cancellation_cause")
        if cause is not None and (cause == ProcessCancellationCause.TIMEOUT) != timed_out:
            raise ValueError("timed_out must match a timeout cancellation cause")
        parsed = dict(value)
        del parsed["timed_out"]
        if timed_out:
            parsed["cancellation_cause"] = ProcessCancellationCause.TIMEOUT
        return parsed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def timed_out(self) -> bool:
        return self.cancellation_cause is ProcessCancellationCause.TIMEOUT

    @model_validator(mode="after")
    def termination_is_coherent(self) -> InferenceReplayResult:
        if self.exit_code is not None and self.terminating_signal is not None:
            raise ValueError("replay cannot have both an exit code and a terminating signal")
        return self


class _OracleObservation(ContractModel):
    identity: OracleIdentity
    result: OracleResult | None = None
    validation_status: ValidationStatus = ValidationStatus.UNSUPPORTED
    limitations: tuple[str, ...] = ()


class InferenceReplayService:
    """Plan and run inference replay scenarios against declared local servers."""

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
        self.publisher = GenerationPublisher(workspace)
        self.broker = broker or SubprocessBroker()
        self.probe_timeout_seconds = probe_timeout_seconds

    def plan(
        self,
        scenario_name: str,
        *,
        timeout_seconds: float | None = None,
        expected_plan_id: str | None = None,
    ) -> InferenceReplayPlan:
        """Build a validated replay plan without executing the replay tool.

        The plan passively probes an ``existing_local`` server so the agent can
        decide whether to run. Managed servers are probed only after their
        declared workload has started during execution.
        """
        project = self.workloads.load()
        scenario, server = self._resolve(scenario_name, project)
        deadline = self._deadline(timeout_seconds, scenario, server)
        deadline_at = utc_now() + timedelta(seconds=deadline)
        tool = _PROVIDER_TOOL[scenario.provider]
        discovery = (
            discover_sglang(Path(server.benchmark_python), broker=self.broker)
            if scenario.provider == "sglang_bench" and server.benchmark_python is not None
            else discover_inference_tool(tool)
        )
        probe: ExistingServerProbe | None = None
        if server.mode == "existing_local" and discovery.available:
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
                        "next_tool": "configure_inference_server",
                    },
                )
        server_executable_digest, server_version = self._server_tool_identity(server)
        output_path = self._output_path(scenario_name, new_id())
        if scenario.provider == "sglang_bench":
            output_path = output_path.with_suffix(".jsonl")
        if discovery.available:
            assert discovery.executable is not None
            request = self._build_request(
                scenario,
                server,
                discovery.executable,
                output_path=output_path,
            )
        else:
            request = None
        configuration_id = digest_model(project.model_dump(mode="json"))
        plan_id = digest_model(
            {
                "scenario": scenario.model_dump(mode="json"),
                "server": server.model_dump(mode="json"),
                "provider_executable": str(discovery.executable),
                "provider_version": discovery.version,
                "provider_executable_digest": discovery.executable_digest,
                "server_executable_digest": server_executable_digest,
                "server_version": server_version,
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
        return parse_inference_replay_plan(
            dict(
                plan_id=plan_id,
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
                tool_executable=str(discovery.executable) if discovery.executable else None,
                tool_version=discovery.version,
                tool_executable_digest=discovery.executable_digest,
                server_executable_digest=server_executable_digest,
                server_version=server_version,
                tool_compatibility_reason=discovery.compatibility_reason,
                tool_remediation=discovery.remediation,
                health_ready=probe.health_ready if probe is not None else None,
                probed_model_ids=probe.model_ids if probe is not None else (),
                argv=request.argv() if request is not None else (),
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
                    if scenario.provider == "sglang_bench"
                    else scenario.random_range_ratio
                ),
                timeout_seconds=deadline,
                deadline_at=deadline_at,
                exploratory_reason=(
                    "Single replay run is exploratory; equivalence or causality requires "
                    "a predeclared confirmatory experiment with a semantic oracle."
                ),
                configuration_id=configuration_id,
            )
        )

    async def run(self, plan: InferenceReplayPlan) -> InferenceReplayResult:
        """Execute a validated plan through the canonical subprocess broker."""
        self._validate_plan(plan)
        output_path = Path(plan.output_path or self._output_path(plan.scenario_name))
        environment = await self._managed_environment(plan)
        run, environment, source_state = self._start_run(plan, environment=environment)
        try:
            outcome, probe, output_path, server_outcome, oracle = await self._execute(plan)
        except asyncio.CancelledError:
            artifact_ids, artifact_run_ids, _preserved, _limitations = (
                self._preserve_outputs_safely(plan, output_path)
            )
            self._finish_cancelled_run(
                run, environment, source_state, artifact_ids, artifact_run_ids
            )
            raise
        except DomainError as error:
            artifact_ids, artifact_run_ids, _preserved, limitations = self._preserve_outputs_safely(
                plan, output_path
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
                plan, output_path
            )
            internal_error = DomainError(
                ErrorCode.INTERNAL_ERROR,
                "Unexpected inference replay failure.",
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
            self._preserve_outputs_safely(plan, output_path)
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
            output_path, preservation_complete=not preservation_limitations
        )
        if cleanup_limitation is not None:
            extraction_limitations = (*extraction_limitations, cleanup_limitation)
            finished = self._append_limitation(
                finished,
                environment,
                source_state,
                artifact_ids,
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

    def run_sync(self, plan: InferenceReplayPlan) -> InferenceReplayResult:
        """Synchronous wrapper around :meth:`run` for non-async callers."""
        self._validate_plan(plan)
        output_path = Path(plan.output_path or self._output_path(plan.scenario_name))
        run, environment, source_state = self._start_run(plan)
        try:
            outcome, probe, output_path, oracle = self._execute_sync(plan)
        except (KeyboardInterrupt, SystemExit):
            artifact_ids, artifact_run_ids, _preserved, _limitations = (
                self._preserve_outputs_safely(plan, output_path)
            )
            self._finish_cancelled_run(
                run, environment, source_state, artifact_ids, artifact_run_ids
            )
            raise
        except DomainError as error:
            artifact_ids, artifact_run_ids, _preserved, limitations = self._preserve_outputs_safely(
                plan, output_path
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
                plan, output_path
            )
            internal_error = DomainError(
                ErrorCode.INTERNAL_ERROR,
                "Unexpected inference replay failure.",
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
            self._preserve_outputs_safely(plan, output_path)
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
            output_path, preservation_complete=not preservation_limitations
        )
        if cleanup_limitation is not None:
            extraction_limitations = (*extraction_limitations, cleanup_limitation)
            finished = self._append_limitation(
                finished,
                environment,
                source_state,
                artifact_ids,
                cleanup_limitation,
            )
        return self._result(
            plan,
            finished.run_id,
            outcome,
            probe,
            output_path,
            None,
            artifact_ids,
            artifact_run_ids,
            (*extraction_limitations, *oracle.limitations),
            oracle,
        )

    async def _execute(
        self, plan: InferenceReplayPlan
    ) -> tuple[
        ExecutionOutcome,
        ExistingServerProbe,
        Path,
        ManagedSidecarOutcome | None,
        _OracleObservation,
    ]:
        if plan.server_mode == "existing_local":
            output_path, probe = self._prepare(plan)
            outcome = await self.broker.run(self._request(plan, output_path))
            oracle = await self._run_oracle(plan, output_path, outcome)
            return outcome, probe, output_path, None, oracle
        project = self.workloads.load()
        _scenario, server = self._resolve(plan.scenario_name, project)
        if server.workload is None:
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Managed server workload is absent.")
        instance = self.workloads.resolve(server.workload)
        output_path = Path(plan.output_path or self._output_path(plan.scenario_name))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        parsed = urlsplit(server.base_url)
        assert parsed.hostname is not None
        port = parsed.port or 80
        remaining = (plan.deadline_at - utc_now()).total_seconds()
        if remaining <= 0:
            raise DomainError(ErrorCode.PROCESS_TIMEOUT, "Inference startup deadline expired.")
        absolute_deadline = time.monotonic() + remaining

        async def readiness() -> bool:
            try:
                await asyncio.to_thread(
                    probe_existing_vllm_server,
                    server.base_url,
                    timeout_seconds=min(self.probe_timeout_seconds, 0.5),
                )
            except DomainError:
                return False
            return True

        command = instance.command
        server_request = ExecutionRequest(
            argv=command.argv,
            cwd=Path(command.cwd),
            environment_allowlist=self.workspace.config.execution.child_environment_allowlist,
            environment_overrides=command.env_overrides,
            allowed_working_roots=(self.workspace.project_root,),
            timeout_seconds=remaining,
            max_output_bytes=16 * 1024 * 1024,
        )
        lease = await self.broker.start_inference_server(
            server_request,
            host=parsed.hostname,
            port=port,
            readiness=readiness,
            absolute_deadline=absolute_deadline,
        )
        try:
            probe = await asyncio.to_thread(
                probe_existing_vllm_server,
                server.base_url,
                timeout_seconds=min(self.probe_timeout_seconds, 0.5),
            )
            outcome = await self.broker.run(self._request(plan, output_path))
            oracle = await self._run_oracle(plan, output_path, outcome)
        finally:
            server_outcome = await asyncio.shield(lease.close())
        self._write_server_output(output_path.parent, server_outcome)
        return outcome, probe, output_path, server_outcome, oracle

    @staticmethod
    def _write_server_output(root: Path, outcome: ManagedSidecarOutcome) -> None:
        """Stage bounded broker-captured server logs for immutable preservation."""
        if outcome.stdout:
            atomic_write_bytes(root / "server.stdout", outcome.stdout)
        if outcome.stderr:
            atomic_write_bytes(root / "server.stderr", outcome.stderr)

    def _execute_sync(
        self, plan: InferenceReplayPlan
    ) -> tuple[ExecutionOutcome, ExistingServerProbe, Path, _OracleObservation]:
        output_path, probe = self._prepare(plan)
        request = self._request(plan, output_path)
        outcome = self.broker.run_sync(request)
        oracle = self._run_oracle_sync(plan, output_path, outcome)
        return outcome, probe, output_path, oracle

    def _prepare(self, plan: InferenceReplayPlan) -> tuple[Path, ExistingServerProbe]:
        if not plan.tool_available:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Inference replay tool for scenario {plan.scenario_name!r} is unavailable.",
                remediation=plan.tool_remediation
                or ("Install the project's inference extra, then retry planning.",),
                details={
                    "scenario": plan.scenario_name,
                    "provider": plan.provider,
                    "next_tool": "manual",
                },
            )
        project = self.workloads.load()
        _scenario, server = self._resolve(plan.scenario_name, project)
        if server.mode != "existing_local":
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Synchronous replay is limited to existing_local servers; use async run for "
                f"managed server {plan.server_name!r}.",
                details={"server": plan.server_name, "mode": server.mode},
            )
        probe = probe_existing_vllm_server(
            server.base_url, timeout_seconds=self.probe_timeout_seconds
        )
        output_path = (
            Path(plan.output_path) if plan.output_path else self._output_path(plan.scenario_name)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path, probe

    def _request(self, plan: InferenceReplayPlan, output_path: Path) -> ExecutionRequest:
        remaining = (plan.deadline_at - utc_now()).total_seconds()
        if remaining <= 0:
            raise DomainError(
                ErrorCode.PROCESS_TIMEOUT,
                "The inference replay deadline expired before benchmark execution.",
            )
        return ExecutionRequest(
            argv=plan.argv,
            cwd=self.workspace.project_root,
            environment_allowlist=self.workspace.config.execution.child_environment_allowlist,
            allowed_working_roots=(self.workspace.project_root, output_path.parent),
            timeout_seconds=min(plan.timeout_seconds, remaining),
            max_output_bytes=16 * 1024 * 1024,
        )

    def _oracle_request(
        self, plan: InferenceReplayPlan, output_path: Path
    ) -> tuple[OracleIdentity, ExecutionRequest, Path] | None:
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
                "The inference replay deadline expired before semantic validation.",
            )
        output_root = output_path.parent
        output_root.mkdir(parents=True, exist_ok=True)
        receipt_path = output_root / "oracle-receipt.json"
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
            cwd=Path(oracle.command.cwd),
            environment_allowlist=self.workspace.config.execution.child_environment_allowlist,
            environment_overrides={
                **oracle.command.env_overrides,
                "FLAMEOX_ORACLE_RECEIPT": str(receipt_path),
                "FLAMEOX_INFERENCE_RESULT_DIR": str(output_root),
                "FLAMEOX_INFERENCE_BASE_URL": plan.base_url,
            },
            allowed_working_roots=(self.workspace.project_root, output_root),
            timeout_seconds=min(oracle.command.timeout_seconds, remaining),
            max_output_bytes=16 * 1024 * 1024,
        )
        return identity, request, receipt_path

    async def _run_oracle(
        self, plan: InferenceReplayPlan, output_path: Path, benchmark: ExecutionOutcome
    ) -> _OracleObservation:
        prepared = self._oracle_request(plan, output_path)
        if prepared is None:
            return _OracleObservation(
                identity=OracleIdentity(kind="none"),
                limitations=("No semantic oracle was declared for this inference scenario.",),
            )
        identity, request, receipt_path = prepared
        if benchmark.process.exit_code != 0:
            return _OracleObservation(
                identity=identity,
                limitations=("Semantic validation was skipped because the benchmark failed.",),
            )
        validation = await self.broker.run(request)
        return self._oracle_observation(identity, validation, receipt_path)

    def _run_oracle_sync(
        self, plan: InferenceReplayPlan, output_path: Path, benchmark: ExecutionOutcome
    ) -> _OracleObservation:
        prepared = self._oracle_request(plan, output_path)
        if prepared is None:
            return _OracleObservation(
                identity=OracleIdentity(kind="none"),
                limitations=("No semantic oracle was declared for this inference scenario.",),
            )
        identity, request, receipt_path = prepared
        if benchmark.process.exit_code != 0:
            return _OracleObservation(
                identity=identity,
                limitations=("Semantic validation was skipped because the benchmark failed.",),
            )
        validation = self.broker.run_sync(request)
        return self._oracle_observation(identity, validation, receipt_path)

    @staticmethod
    def _oracle_observation(
        identity: OracleIdentity,
        validation: ExecutionOutcome,
        receipt_path: Path,
    ) -> _OracleObservation:
        output_root = receipt_path.parent
        atomic_write_bytes(output_root / "oracle.stdout", validation.stdout)
        if validation.stderr:
            atomic_write_bytes(output_root / "oracle.stderr", validation.stderr)
        if validation.process.exit_code != 0:
            return _OracleObservation(
                identity=identity,
                validation_status=ValidationStatus.FAILED,
                limitations=("The declared semantic oracle exited unsuccessfully.",),
            )
        try:
            receipt = parse_oracle_receipt(receipt_path.read_bytes())
        except (OSError, DomainError) as error:
            message = error.message if isinstance(error, DomainError) else "receipt was not emitted"
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
                details={"scenario": scenario_name, "next_tool": "list_inference_configurations"},
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
                    "next_tool": "configure_inference_server",
                },
            ) from exc
        return scenario, server

    def _server_tool_identity(self, server: InferenceServerConfig) -> tuple[str | None, str | None]:
        if not isinstance(server, _ManagedInferenceServerConfig):
            return None, None
        command = self.workloads.resolve(server.workload).command
        executable = self._resolve_server_executable(command)
        executable_digest = self._executable_digest(executable)
        discovery = discover_inference_tool("vllm") if server.provider == "vllm" else None
        version = (
            discovery.version
            if discovery is not None
            and discovery.executable is not None
            and executable.resolve() == discovery.executable.resolve()
            else None
        )
        return executable_digest, version

    @staticmethod
    def _resolve_server_executable(command: CommandSpec) -> Path:
        candidate = Path(command.argv[0])
        if candidate.is_absolute() or candidate.parent != Path("."):
            return (
                candidate if candidate.is_absolute() else (Path(command.cwd) / candidate).resolve()
            )
        environment = {**os.environ, **command.env_overrides}
        located = shutil.which(command.argv[0], path=environment.get("PATH"))
        return Path(located).resolve() if located is not None else candidate

    @staticmethod
    def _executable_digest(path: Path) -> str | None:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return None
        return f"sha256:{digest.hexdigest()}"

    def _validate_plan(self, plan: InferenceReplayPlan) -> None:
        project = self.workloads.load()
        configuration_id = digest_model(project.model_dump(mode="json"))
        if configuration_id != plan.configuration_id:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Inference configuration changed after this plan was created.",
                remediation=("Plan the inference scenario again, then retry execution.",),
            )
        provider = (
            discover_sglang(Path(plan.tool_executable), broker=self.broker)
            if plan.provider == "sglang_bench" and plan.tool_executable is not None
            else discover_inference_tool(_PROVIDER_TOOL[plan.provider])
        )
        if not plan.tool_available:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Inference replay tool for scenario {plan.scenario_name!r} is unavailable.",
                remediation=plan.tool_remediation
                or ("Install the project's inference extra, then retry planning.",),
                details={
                    "scenario": plan.scenario_name,
                    "provider": plan.provider,
                    "next_tool": "manual",
                },
            )
        if (
            provider.executable_digest != plan.tool_executable_digest
            or provider.version != plan.tool_version
        ):
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Inference provider executable changed after planning.",
                remediation=("Plan the inference scenario again, then retry execution.",),
            )
        _scenario, server = self._resolve(plan.scenario_name, project)
        server_digest, server_version = self._server_tool_identity(server)
        if server_digest != plan.server_executable_digest or server_version != plan.server_version:
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
        # One absolute deadline for the whole replay. AIPerf trace replay can be
        # long, so default to a generous bounded value rather than the workload
        # default.
        return 1800.0 if scenario.provider == "aiperf" else 600.0

    def _output_path(self, scenario_name: str, run_token: str | None = None) -> Path:
        root = self.workspace.paths.staging / f"inference-replay-{scenario_name}"
        return (root / run_token if run_token is not None else root) / "result.json"

    def _build_request(
        self,
        scenario: InferenceScenarioConfig,
        server: InferenceServerConfig,
        executable: Path,
        *,
        output_path: Path,
    ) -> AIPerfProfileRequest | VllmBenchServeRequest | SglangBenchServingRequest:
        # AIPerf's fixed_schedule requires a Mooncake trace; without a declared
        # trace_artifact_id the replay uses the tool's default schedule.
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
                benchmark_python=Path(server.benchmark_python),
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
        plan: InferenceReplayPlan,
        run_id: str,
        outcome: ExecutionOutcome,
        probe: ExistingServerProbe,
        output_path: Path,
        server_outcome: ManagedSidecarOutcome | None,
        artifact_ids: tuple[str, ...],
        artifact_run_ids: tuple[str, ...],
        extraction_limitations: tuple[str, ...],
        oracle: _OracleObservation,
    ) -> InferenceReplayResult:
        process = outcome.process
        limitations: list[str] = []
        if not probe.health_ready:
            limitations.append("Server health probe was not ready before replay.")
        if not probe.model_ids:
            limitations.append("Server exposed no model ids before replay.")
        if process.timed_out:
            limitations.append("Replay was terminated by the absolute deadline.")
        limitations.extend(extraction_limitations)
        limitations.append(plan.exploratory_reason)
        return InferenceReplayResult(
            result_id=digest_model(
                {
                    "plan_id": plan.plan_id,
                    "exit_code": process.exit_code,
                    "wall_time_ns": process.wall_time_ns,
                    "output_path": str(output_path),
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            ),
            run_id=run_id,
            plan_id=plan.plan_id,
            scenario_name=plan.scenario_name,
            server_name=plan.server_name,
            provider=plan.provider,
            argv=plan.argv,
            output_path=str(output_path),
            output_path_retained=output_path.parent.exists(),
            exit_code=process.exit_code,
            terminating_signal=process.terminating_signal,
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

    async def _managed_environment(self, plan: InferenceReplayPlan) -> EnvironmentRecord:
        if plan.server_mode != "managed":
            return collect_environment()
        remaining = (plan.deadline_at - utc_now()).total_seconds()
        if remaining <= 0:
            return collect_environment()
        try:
            accelerator = await asyncio.wait_for(
                AcceleratorIdentityService(
                    self.workspace.project_root,
                    broker=self.broker,
                ).observe(("cuda.driver", "cuda.runtime", "cuda.devices", "cuda.peer_topology")),
                timeout=remaining,
            )
        except TimeoutError:
            return collect_environment()
        return collect_environment(accelerator)

    def _start_run(
        self,
        plan: InferenceReplayPlan,
        *,
        environment: EnvironmentRecord | None = None,
    ) -> tuple[RunManifest, EnvironmentRecord, SourceState]:
        """Create the canonical execution run that owns normalized replay evidence."""
        environment = environment or collect_environment()
        candidate = Path(plan.tool_executable) if plan.tool_executable is not None else None
        executable = candidate if candidate is not None and candidate.is_file() else None
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
            collector=plan.provider,
            collector_version=plan.tool_version,
            command=CommandSpec(
                argv=plan.argv,
                cwd=str(self.workspace.project_root),
                timeout_seconds=plan.timeout_seconds,
            ),
            inference_protocol_identity_id=protocol_id,
            inference_protocol_identity_json=protocol_json,
            limitations=(plan.exploratory_reason,),
        )
        self.runs.create(run)
        self._publish_run(run, environment, source_state, ())
        return run, environment, source_state

    def _protocol_identity(
        self,
        plan: InferenceReplayPlan,
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
                    "provider_executable_digest": plan.tool_executable_digest,
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
            provider_version=plan.tool_version,
            provider_executable_digest=plan.tool_executable_digest,
            trace=TraceIdentity(
                format=(
                    "mooncake"
                    if plan.trace_artifact_id is not None
                    else "aiperf"
                    if plan.provider == "aiperf"
                    else "sglang.bench_serving"
                    if plan.provider == "sglang_bench"
                    else "vllm"
                ),
                producer=(
                    "mooncake"
                    if plan.trace_artifact_id is not None
                    else "aiperf"
                    if plan.provider == "aiperf"
                    else "sglang.bench_serving"
                    if plan.provider == "sglang_bench"
                    else "vllm"
                ),
                producer_version=(
                    None if plan.trace_artifact_id is not None else plan.tool_version
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
                quantization=plan.quantization or "none",
            ),
            server=ServerConfigIdentity(
                backend=plan.server_provider,
                cache_backend="custom" if plan.server_provider == "sglang" else "vllm_paged",
                endpoint=(
                    "/v1/chat/completions" if plan.endpoint_type == "chat" else "/v1/completions"
                ),
                managed_server_command_digest=managed_command_digest,
                server_executable_digest=plan.server_executable_digest,
                server_version=plan.server_version,
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
            if process.exit_code == 0
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
        finished = self.runs.append(finished, expected_revision=0)
        self._publish_run(finished, environment, source_state, artifact_ids)
        return finished

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
                    dict.fromkeys((*run.limitations, "Inference replay was cancelled."))
                ),
            }
        )
        finished = self.runs.append(finished, expected_revision=0)
        self._publish_run(finished, environment, source_state, artifact_ids)
        return finished

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
        finished = self.runs.append(finished, expected_revision=0)
        self._publish_run(finished, environment, source_state, artifact_ids)
        return finished

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
        artifact_ids: tuple[str, ...],
        limitation: str,
    ) -> RunManifest:
        updated = run.validated_copy(
            update={
                "revision": run.revision + 1,
                "limitations": tuple(dict.fromkeys((*run.limitations, limitation))),
            }
        )
        updated = self.runs.append(updated, expected_revision=run.revision)
        self._publish_run(updated, environment, source_state, artifact_ids)
        return updated

    def _publish_run(
        self,
        run: RunManifest,
        environment: EnvironmentRecord,
        source_state: SourceState,
        artifact_ids: tuple[str, ...],
    ) -> None:
        input_artifact_ids = list(artifact_ids)
        if run.inference_protocol_identity_json is not None:
            protocol = InferenceProtocolIdentity.model_validate_json(
                run.inference_protocol_identity_json
            )
            if protocol.trace.artifact_digest is not None:
                input_artifact_ids.append(protocol.trace.artifact_digest)
        self.publisher.publish_rows(
            {
                "runs": [run_row(run)],
                "artifact_registrations": [
                    artifact_registration_row(
                        registration,
                        byte_length=self.artifacts.get(
                            registration.artifact_id
                        ).content.byte_length,
                    )
                    for registration in run.artifacts
                ],
                "environments": [environment_row(environment)],
                "source_states": [source_state_row(source_state)],
            },
            publisher="flameox.inference",
            publisher_version="1",
            input_run_ids=(run.run_id,),
            input_artifact_ids=tuple(dict.fromkeys(input_artifact_ids)),
        )

    def _extract_outputs(
        self,
        plan: InferenceReplayPlan,
        evidence_run_id: str,
        preserved: tuple[tuple[str, str, str], ...],
    ) -> tuple[str, ...]:
        """Normalize supported provider output without making native preservation contingent."""
        from flameox.adapters.inference import InferenceArtifactExtractor

        extractor = InferenceArtifactExtractor(self.workspace)
        limitations: list[str] = []
        try:
            if plan.provider in {"vllm_bench", "sglang_bench"}:
                if preserved:
                    result = (
                        extractor.extract_sglang_result
                        if plan.provider == "sglang_bench"
                        else extractor.extract_vllm_result
                    )(preserved[0][1], evidence_run_id=evidence_run_id)
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
        self, plan: InferenceReplayPlan, output_path: Path
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[tuple[str, str, str], ...],
        tuple[str, ...],
    ]:
        discovery_limitations: tuple[str, ...] = ()
        if plan.provider == "aiperf" and output_path.parent.is_dir():
            candidates, discovery_limitations = self._bounded_output_candidates(output_path.parent)
        else:
            candidates = tuple(
                path
                for path in (
                    output_path,
                    output_path.parent / "oracle-receipt.json",
                    output_path.parent / "oracle.stdout",
                    output_path.parent / "oracle.stderr",
                    output_path.parent / "server.stdout",
                    output_path.parent / "server.stderr",
                )
                if path.is_file()
            )
        artifacts: list[str] = []
        runs: list[str] = []
        preserved: list[tuple[str, str, str]] = []
        limitations = list(discovery_limitations)
        importer = ImportService(self.workspace)
        for path in candidates:
            is_inputs = plan.provider == "aiperf" and path.name == "inputs.json"
            is_oracle = path.name.startswith("oracle")
            is_server_output = path.name.startswith("server.")
            try:
                imported = importer.import_artifact(
                    ImportArtifactRequest(
                        path=path,
                        kind=(
                            ArtifactKind.INFERENCE_REQUEST_TRACE
                            if is_inputs
                            else ArtifactKind.VALIDATION_OUTPUT
                            if is_oracle and path.name != "oracle.stderr"
                            else ArtifactKind.PROCESS_OUTPUT
                            if is_oracle or is_server_output
                            else ArtifactKind.INFERENCE_RESULT
                        ),
                        media_type=self._provider_media_type(path),
                        sensitivity=(
                            Sensitivity.SENSITIVE
                            if plan.provider in {"aiperf", "sglang_bench"}
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
                            plan.semantic_oracle_workload or "inference_oracle"
                            if is_oracle
                            else plan.provider
                        ),
                        producer_version=(
                            "flameox.oracle-receipt.v1" if is_oracle else plan.tool_version
                        ),
                        allow_external_path=True,
                    )
                )
            except DomainError as error:
                limitations.append(
                    f"Provider artifact {path.name!r} could not be preserved: {error.message}"
                )
                continue
            artifacts.append(imported.artifact_id)
            runs.append(imported.run.run_id)
            preserved.append((path.name, imported.run.run_id, imported.artifact_id))
        return tuple(artifacts), tuple(runs), tuple(preserved), tuple(limitations)

    def _preserve_outputs_safely(
        self, plan: InferenceReplayPlan, output_path: Path
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[tuple[str, str, str], ...],
        tuple[str, ...],
    ]:
        try:
            return self._preserve_outputs(plan, output_path)
        except DomainError as error:
            return (), (), (), (f"Provider artifact preservation failed: {error.message}",)

    @staticmethod
    def _provider_media_type(path: Path) -> str:
        return {
            ".json": "application/json",
            ".jsonl": "application/x-ndjson",
            ".csv": "text/csv",
            ".log": "text/plain",
            ".txt": "text/plain",
        }.get(path.suffix.lower(), "application/octet-stream")

    def _cleanup_staging(self, output_path: Path, *, preservation_complete: bool) -> str | None:
        if not preservation_complete:
            return (
                "Provider staging was retained because native artifact preservation was incomplete."
            )
        root = output_path.parent.absolute()
        staging = self.workspace.paths.staging.resolve()
        if root.is_symlink():
            return "Provider staging cleanup was refused because the operation path is a symlink."
        try:
            resolved_root = root.resolve(strict=True)
        except FileNotFoundError:
            return None
        if resolved_root == staging or not resolved_root.is_relative_to(staging):
            return "Provider staging cleanup was refused because the path was not operation-owned."
        try:
            shutil.rmtree(root)
        except FileNotFoundError:
            return None
        except OSError:
            return "Provider staging cleanup failed; immutable artifacts remain authoritative."
        return None

    @staticmethod
    def _bounded_output_candidates(root: Path) -> tuple[tuple[Path, ...], tuple[str, ...]]:
        candidates: list[Path] = []
        limitations: list[str] = []
        directories = [root]
        observed_entries = 0
        while directories:
            directory = directories.pop()
            try:
                entries = os.scandir(directory)
                with entries:
                    for entry in entries:
                        observed_entries += 1
                        if observed_entries > 4_096:
                            limitations.append(
                                "Provider output discovery stopped at 4096 filesystem entries."
                            )
                            return tuple(candidates), tuple(limitations)
                        if entry.is_symlink():
                            limitations.append("Provider output discovery skipped a symbolic link.")
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            directories.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            candidates.append(Path(entry.path))
                            if len(candidates) >= 128:
                                limitations.append(
                                    "Provider output discovery stopped at 128 files."
                                )
                                return tuple(candidates), tuple(limitations)
            except OSError:
                limitations.append("Provider output discovery could not inspect one directory.")
        return tuple(candidates), tuple(limitations)
