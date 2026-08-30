from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import Discriminator, Field, JsonValue, Tag, TypeAdapter

from flameox.action_graph import ActionId, tool_action
from flameox.analysis.inference_protocol import (
    AttachedProfilerState,
    InferenceProtocolIdentity,
    ProfilerKind,
)
from flameox.application.evidence_query import EvidenceQueryService
from flameox.application.imports import ImportDescriptorRequest, ImportService
from flameox.application.inference import InferenceScenarioPlan, InferenceScenarioService
from flameox.application.inference_providers import (
    InferenceServerMode,
    InferenceServerProvider,
    discover_sglang,
    probe_existing_vllm_server_async,
)
from flameox.application.projections import ProjectionCoordinator
from flameox.application.source import collect_partial_source_state
from flameox.application.workloads import WorkloadService, _ManagedInferenceServerConfig
from flameox.command_binding import ExecutableResolver
from flameox.domain import (
    ArtifactKind,
    ArtifactRegistration,
    CaptureStatus,
    CommandSpec,
    DomainError,
    EnvironmentRecord,
    ErrorCode,
    ExecutionStatus,
    ProcessResult,
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
from flameox.domain.models import ExecutionRunManifest, utc_now
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.filesystem_authority import (
    BoundDirectory,
    BoundDirectoryReference,
    TrustedRoot,
)
from flameox.http_transport import (
    BoundedHttpClient,
    BoundedHttpError,
    HttpMethod,
    LoopbackHttpRequest,
    validate_loopback_base_url,
)
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, AuthorizedPlanStore, RunStore, Workspace

type SupportedInferenceProfiler = Literal[
    ProfilerKind.TORCH_PROFILER,
    ProfilerKind.NSIGHT_SYSTEMS,
]
_SUPPORTED_INFERENCE_PROFILER: TypeAdapter[SupportedInferenceProfiler] = TypeAdapter(
    SupportedInferenceProfiler
)


class InferenceProfileCoverage(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class SglangProfileOptions(ContractModel):
    """The SGLang Torch-profiler request Flameox supports."""

    start_step: Literal[5] = 5
    num_steps: Literal[2] = 2
    activities: tuple[Literal["CPU"], Literal["GPU"]] = ("CPU", "GPU")
    profile_by_stage: Literal[True] = True
    record_shapes: Literal[True] = True
    with_stack: Literal[True] = True


class _InferenceProfilingPlan(ContractModel):
    plan_id: str
    plan_token: str = ""
    scenario_name: str | None = None
    measurement_run_id: str | None = None
    timeout_seconds: float | None = None
    scenario_plan: InferenceScenarioPlan | None = None
    server_name: str
    base_url: str
    server_argv: tuple[str, ...]
    server_executable_binding: ResolvedExecutable
    launch_executable_binding: ResolvedExecutable
    server_cwd: Path
    environment_names: tuple[str, ...]
    environment_digest: str
    output_root: BoundDirectoryReference
    output_relative_path: str
    output_path: Path
    configuration_id: str
    diagnostic_only: Literal[True] = True
    limitations: tuple[str, ...]


class VllmTorchProfilingPlan(_InferenceProfilingPlan):
    server_provider: Literal[InferenceServerProvider.VLLM] = InferenceServerProvider.VLLM
    profiler: Literal[ProfilerKind.TORCH_PROFILER] = ProfilerKind.TORCH_PROFILER
    nsys_executable: Literal[None] = None
    nsys_executable_binding: Literal[None] = None
    sglang_profile_id: Literal[None] = None
    sglang_profile_options: Literal[None] = None


class SglangTorchProfilingPlan(_InferenceProfilingPlan):
    server_provider: Literal[InferenceServerProvider.SGLANG] = InferenceServerProvider.SGLANG
    profiler: Literal[ProfilerKind.TORCH_PROFILER] = ProfilerKind.TORCH_PROFILER
    nsys_executable: Literal[None] = None
    nsys_executable_binding: Literal[None] = None
    sglang_profile_id: Annotated[str, Field(min_length=1, max_length=100)]
    sglang_profile_options: SglangProfileOptions


class NsightSystemsProfilingPlan(_InferenceProfilingPlan):
    server_provider: InferenceServerProvider
    profiler: Literal[ProfilerKind.NSIGHT_SYSTEMS] = ProfilerKind.NSIGHT_SYSTEMS
    nsys_executable: Path
    nsys_executable_binding: ResolvedExecutable
    symbol_resolution: Literal["disabled"] = "disabled"
    sglang_profile_id: Literal[None] = None
    sglang_profile_options: Literal[None] = None


def _profiling_plan_variant(value: Any) -> Literal["vllm_torch", "sglang_torch", "nsight"]:
    if isinstance(value, NsightSystemsProfilingPlan):
        return "nsight"
    if isinstance(value, SglangTorchProfilingPlan):
        return "sglang_torch"
    if isinstance(value, Mapping):
        if value.get("profiler") == "nsight_systems":
            return "nsight"
        if value.get("server_provider") == "sglang":
            return "sglang_torch"
    return "vllm_torch"


type InferenceProfilingPlan = Annotated[
    Annotated[VllmTorchProfilingPlan, Tag("vllm_torch")]
    | Annotated[SglangTorchProfilingPlan, Tag("sglang_torch")]
    | Annotated[NsightSystemsProfilingPlan, Tag("nsight")],
    Discriminator(_profiling_plan_variant),
]

_INFERENCE_PROFILING_PLAN: TypeAdapter[InferenceProfilingPlan] = TypeAdapter(InferenceProfilingPlan)


def parse_inference_profiling_plan(value: Any) -> InferenceProfilingPlan:
    return _INFERENCE_PROFILING_PLAN.validate_python(value)


class InferenceProfilingResult(ContractModel):
    run_id: str
    plan_id: str
    measurement_protocol_id: str
    measurement_run_id: str
    scenario_name: str
    profiler: SupportedInferenceProfiler
    benchmark_exit_code: int | None
    server_cleanup_complete: bool
    artifact_ids: tuple[str, ...]
    artifact_run_ids: tuple[str, ...]
    extracted_run_ids: tuple[str, ...] = ()
    workload_duration_ns: int | None
    finalization_duration_ns: int
    export_duration_ns: int | None
    symbol_resolution_status: Literal["disabled"] | None
    symbol_resolution_duration_ns: int | None
    coverage: InferenceProfileCoverage
    limitations: tuple[str, ...]


class InferenceProfilingService:
    """Build bounded diagnostic capture plans for Flameox-managed vLLM servers."""

    def __init__(self, workspace: Workspace, *, broker: SubprocessBroker | None = None) -> None:
        self.workspace = workspace
        self.workloads = WorkloadService(workspace)
        self.broker = broker or SubprocessBroker()
        self.artifacts = ArtifactStore(workspace)
        self.runs = RunStore(workspace)
        self.projections = ProjectionCoordinator(workspace)
        self.plans = AuthorizedPlanStore(
            workspace,
            family="inference_profiling",
            model=_INFERENCE_PROFILING_PLAN,
        )

    def plan(  # noqa: C901 - one typed planner owns the mutually exclusive profiler cases
        self,
        server_name: str,
        *,
        profiler: SupportedInferenceProfiler | str,
        nsys_executable: Path | None = None,
        expected_plan_id: str | None = None,
        scenario_name: str | None = None,
        measurement_run_id: str | None = None,
        timeout_seconds: Annotated[float, Field(gt=0, le=86_400)] | None = None,
    ) -> InferenceProfilingPlan:
        operation_values = (scenario_name, measurement_run_id, timeout_seconds)
        if any(value is not None for value in operation_values) and not all(
            value is not None for value in operation_values
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "A profiling execution plan requires scenario, measurement run, and timeout.",
            )
        selected_profiler = _SUPPORTED_INFERENCE_PROFILER.validate_python(profiler)
        project = self.workloads.load()
        try:
            server = project.inference_servers[server_name]
        except KeyError as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Inference server {server_name!r} is not declared.",
                remediation=("List or configure a managed inference server, then retry.",),
                details={"server": server_name},
                next_action=tool_action(ActionId.LIST_INFERENCE_CONFIGURATIONS),
            ) from exc
        if server.mode is not InferenceServerMode.MANAGED or server.workload is None:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Inference profiling requires a Flameox-managed server workload.",
            )
        if (
            server.provider is InferenceServerProvider.SGLANG
            and selected_profiler is not ProfilerKind.TORCH_PROFILER
        ):
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "SGLang profiling supports only its stage-separated Torch profiler.",
                remediation=("Use profiler='torch_profiler' for the declared SGLang server.",),
            )
        workload = self.workloads.resolve(server.workload)
        scenario_service = InferenceScenarioService(self.workspace, broker=self.broker)
        if (
            server.provider is InferenceServerProvider.SGLANG
            and scenario_name is None
            and server.benchmark_python is not None
        ):
            discover_sglang(Path(server.benchmark_python), broker=self.broker)
        native_argv = workload.command.argv
        environment = dict(workload.command.env_overrides)
        limitations: tuple[str, ...]
        nsys_binding = None
        if (
            selected_profiler is ProfilerKind.TORCH_PROFILER
            and server.provider is InferenceServerProvider.VLLM
        ):
            output_relative_path = "torch"
            limitations = (
                "Diagnostic profile; measurements must come from a separate unprofiled run.",
            )
        elif selected_profiler is ProfilerKind.TORCH_PROFILER:
            output_relative_path = "torch"
            limitations = (
                "Diagnostic profile; measurements must come from a separate unprofiled run.",
                "SGLang captures separate prefill/decode traces; kernel dominance supports a "
                "hypothesis, not a bandwidth conclusion.",
            )
        else:
            if nsys_executable is None:
                nsys_binding = ExecutableResolver().resolve_host_tool("nsys")
            else:
                nsys_binding = ExecutableResolver().resolve_host_tool(str(nsys_executable))
            nsys_executable = nsys_binding.canonical_target if nsys_binding is not None else None
            if (
                nsys_executable is None
                or not nsys_executable.is_file()
                or not os.access(nsys_executable, os.X_OK)
            ):
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "Nsight Systems profiling requires an installed nsys executable.",
                )
            output_relative_path = "capture.nsys-rep"
            limitations = (
                "Diagnostic profile; measurements must come from a separate unprofiled run.",
                "The native .nsys-rep must be exported with official nsys export "
                "before extraction.",
            )
        sglang_profile_options = (
            SglangProfileOptions()
            if (
                server.provider is InferenceServerProvider.SGLANG
                and selected_profiler is ProfilerKind.TORCH_PROFILER
            )
            else None
        )
        environment_names = set(environment)
        if (
            selected_profiler is ProfilerKind.TORCH_PROFILER
            and server.provider is InferenceServerProvider.VLLM
        ):
            environment_names.add("VLLM_TORCH_PROFILER_DIR")
        environment_digest = digest_model(workload.command.env_overrides)
        scenario_plan = (
            scenario_service._build_plan(scenario_name, timeout_seconds=timeout_seconds)
            if scenario_name is not None and timeout_seconds is not None
            else None
        )
        if scenario_plan is not None and (
            scenario_plan.server_name != server_name or scenario_plan.server_mode != "managed"
        ):
            self._discard_planned_outputs(scenario_plan.output_root)
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Profiling scenario must target the planned managed server.",
            )
        if measurement_run_id is not None:
            try:
                self.runs.read(measurement_run_id)
            except BaseException:
                assert scenario_plan is not None
                self._discard_planned_outputs(scenario_plan.output_root)
                raise
        try:
            with (
                TrustedRoot(self.workspace.paths.staging) as trusted_root,
                trusted_root.allocate_directory(
                    f"inference-profile/{secrets.token_hex(16)}"
                ) as output,
            ):
                output_reference = output.reference
                output_path = output.absolute_display_path(output_relative_path)
                if selected_profiler is ProfilerKind.TORCH_PROFILER:
                    output.ensure_directory(output_relative_path)
                    argv = native_argv
                else:
                    assert nsys_executable is not None
                    argv = (
                        str(nsys_executable.resolve()),
                        "profile",
                        "--trace-fork-before-exec=true",
                        "--cuda-graph-trace=node",
                        "--capture-range=cudaProfilerApi",
                        "--capture-range-end=repeat",
                        "--resolve-symbols=false",
                        "--output",
                        str(output_path.with_suffix("")),
                        *native_argv,
                        "--profiler-config.profiler",
                        "cuda",
                    )
        except BaseException:
            if scenario_plan is not None:
                self._discard_planned_outputs(scenario_plan.output_root)
            raise
        identity = {
            "server": server.model_dump(mode="json"),
            "profiler": selected_profiler,
            "native_argv": native_argv,
            "server_executable_binding": workload.executable_binding.model_dump(mode="json"),
            "launch_executable_binding": (nsys_binding or workload.executable_binding).model_dump(
                mode="json"
            ),
            "cwd": workload.command.cwd,
            "environment_digest": environment_digest,
            "output_relative_path": output_relative_path,
            "nsys_executable": str(nsys_executable.resolve()) if nsys_executable else None,
            "nsys_executable_digest": (
                nsys_binding.identity.sha256 if nsys_binding is not None else None
            ),
            "configuration_id": digest_model(project.model_dump(mode="json")),
            "server_provider": server.provider,
            "sglang_profile_options": (
                sglang_profile_options.model_dump(mode="json")
                if sglang_profile_options is not None
                else None
            ),
            "scenario_name": scenario_name,
            "measurement_run_id": measurement_run_id,
            "timeout_seconds": timeout_seconds,
            "scenario_plan": (
                scenario_plan.model_dump(mode="json", exclude={"plan_token"})
                if scenario_plan is not None
                else None
            ),
        }
        plan_id = digest_model(identity)
        if expected_plan_id is not None and expected_plan_id != plan_id:
            references = (output_reference,) + (
                (scenario_plan.output_root,) if scenario_plan is not None else ()
            )
            self._discard_planned_outputs(*references)
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "The inference profiling plan no longer matches the reviewed plan identity.",
                remediation=("Plan the inference profile again and review the replacement.",),
                details={"expected_plan_id": expected_plan_id, "actual_plan_id": plan_id},
            )
        plan = parse_inference_profiling_plan(
            {
                "plan_id": plan_id,
                "plan_token": secrets.token_hex(32) if scenario_plan is not None else "",
                "scenario_name": scenario_name,
                "measurement_run_id": measurement_run_id,
                "timeout_seconds": timeout_seconds,
                "scenario_plan": scenario_plan,
                "server_name": server_name,
                "server_provider": server.provider,
                "profiler": selected_profiler,
                "base_url": server.base_url,
                "server_argv": argv,
                "server_executable_binding": workload.executable_binding,
                "launch_executable_binding": nsys_binding or workload.executable_binding,
                "server_cwd": Path(workload.command.cwd),
                "environment_names": tuple(sorted(environment_names)),
                "environment_digest": environment_digest,
                "output_root": output_reference,
                "output_relative_path": output_relative_path,
                "output_path": output_path,
                "nsys_executable": nsys_executable,
                "nsys_executable_binding": nsys_binding,
                "configuration_id": digest_model(project.model_dump(mode="json")),
                "limitations": limitations,
                # This is a server-side operation token, not a source of plan churn.
                "sglang_profile_id": (
                    f"flameox-{plan_id[7:31]}" if sglang_profile_options else None
                ),
                "sglang_profile_options": sglang_profile_options,
            }
        )
        if scenario_plan is not None:
            self.plans.issue(
                plan.plan_token,
                plan.plan_id,
                plan,
                expires_at=scenario_plan.deadline_at,
            )
        return plan

    def _discard_planned_outputs(self, *references: BoundDirectoryReference) -> None:
        with TrustedRoot(self.workspace.paths.staging) as trusted_root:
            for reference in references:
                try:
                    trusted_root.remove_directory(reference)
                except DomainError:
                    continue

    @staticmethod
    def _profile_runtime_argv(
        plan: InferenceProfilingPlan,
        output: BoundDirectory,
    ) -> tuple[str, ...]:
        display_output = str(plan.output_path)
        display_stem = str(plan.output_path.with_suffix(""))
        process_output = str(output.child_process_path(plan.output_relative_path))
        process_stem = str(
            output.child_process_path(str(Path(plan.output_relative_path).with_suffix("")))
        )
        return tuple(
            process_output
            if argument == display_output
            else process_stem
            if argument == display_stem
            else argument
            for argument in plan.server_argv
        )

    def _validate_output_authority(self, plan: InferenceProfilingPlan) -> None:
        parts = plan.output_root.parts()
        expected_relative = (
            "torch" if plan.profiler is ProfilerKind.TORCH_PROFILER else "capture.nsys-rep"
        )
        expected_path = self.workspace.paths.staging.joinpath(
            *parts,
            plan.output_relative_path,
        ).absolute()
        if (
            len(parts) != 2
            or parts[0] != "inference-profile"
            or len(parts[1]) != 32
            or any(character not in "0123456789abcdef" for character in parts[1])
            or plan.output_relative_path != expected_relative
            or plan.output_path.absolute() != expected_path
        ):
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Inference profiler output authority does not match its reviewed plan.",
            )

    async def capture(
        self,
        plan_token: str,
        *,
        expected_plan_id: str | None = None,
    ) -> InferenceProfilingResult:
        """Consume and execute one complete server-owned profiling intent."""

        plan = self.plans.consume(plan_token, expected_digest=expected_plan_id)
        if (
            plan.scenario_name is None
            or plan.measurement_run_id is None
            or plan.timeout_seconds is None
            or plan.scenario_plan is None
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The profiling capability does not contain a complete execution intent.",
            )
        self._validate_output_authority(plan)
        with (
            TrustedRoot(self.workspace.paths.staging) as trusted_root,
            trusted_root.open_directory(plan.output_root) as profile_output,
            trusted_root.open_directory(plan.scenario_plan.output_root) as scenario_output,
        ):
            return await self._capture_bound(
                plan,
                trusted_root,
                profile_output,
                scenario_output,
            )

    async def _capture_bound(  # noqa: C901
        self,
        plan: InferenceProfilingPlan,
        trusted_root: TrustedRoot,
        profile_output: BoundDirectory,
        scenario_output: BoundDirectory,
    ) -> InferenceProfilingResult:
        scenario_name = plan.scenario_name
        measurement_run_id = plan.measurement_run_id
        timeout_seconds = plan.timeout_seconds
        assert scenario_name is not None
        assert measurement_run_id is not None
        assert timeout_seconds is not None
        scenario_service = InferenceScenarioService(self.workspace, broker=self.broker)
        project = self.workloads.load()
        if digest_model(project.model_dump(mode="json")) != plan.configuration_id:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Inference configuration changed after this profiling plan was created.",
                remediation=("Plan the inference profile again, then retry capture.",),
            )
        server = project.inference_servers.get(plan.server_name)
        if server is None:
            raise DomainError(ErrorCode.REVISION_CONFLICT, "Planned inference server was removed.")
        if not isinstance(server, _ManagedInferenceServerConfig):
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Planned inference server is no longer managed.",
            )
        server_environment = dict(self.workloads.resolve(server.workload).command.env_overrides)
        if digest_model(server_environment) != plan.environment_digest:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Managed server environment identity changed after profiling was planned.",
                remediation=("Plan the inference profile again, then retry capture.",),
            )
        if (
            plan.profiler is ProfilerKind.TORCH_PROFILER
            and plan.server_provider is InferenceServerProvider.VLLM
        ):
            server_environment["VLLM_TORCH_PROFILER_DIR"] = str(
                profile_output.child_process_path(plan.output_relative_path)
            )
        try:
            ExecutableResolver().revalidate(plan.server_executable_binding)
        except DomainError as error:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Managed server executable identity changed after profiling was planned.",
                remediation=("Plan the inference profile again, then retry capture.",),
            ) from error
        scenario_plan = plan.scenario_plan
        assert scenario_plan is not None
        scenario_service._validate_plan(scenario_plan)
        if scenario_plan.server_name != plan.server_name or scenario_plan.server_mode != "managed":
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Profiling scenario must target the planned managed server.",
            )
        environment = await scenario_service._managed_environment(scenario_plan)
        measurement_protocol = scenario_service._protocol_identity(
            scenario_plan, environment=environment
        )
        measurement_protocol_id = digest_model(measurement_protocol.model_dump(mode="json"))
        self._validate_measurement_run(
            measurement_run_id,
            measurement_protocol_id,
            scenario_name=scenario_name,
        )
        diagnostic_protocol = InferenceProtocolIdentity.model_validate(
            {
                **measurement_protocol.model_dump(mode="python"),
                "profiler": AttachedProfilerState(profiler=plan.profiler),
            }
        )
        deadline_at = scenario_plan.deadline_at
        parsed = urlsplit(plan.base_url)
        assert parsed.hostname is not None
        port = parsed.port or 80
        http_client = BoundedHttpClient()

        async def readiness() -> bool:
            try:
                await probe_existing_vllm_server_async(
                    plan.base_url,
                    timeout_seconds=0.5,
                    http_client=http_client,
                )
            except DomainError:
                return False
            return True

        remaining = (deadline_at - utc_now()).total_seconds()
        if remaining <= 0:
            raise DomainError(ErrorCode.PROCESS_TIMEOUT, "Profiling startup deadline expired.")
        server_request = ExecutionRequest(
            argv=self._profile_runtime_argv(plan, profile_output),
            executable_binding=plan.launch_executable_binding,
            cwd=plan.server_cwd,
            environment_allowlist=self.workspace.config.execution.child_environment_allowlist,
            environment_overrides=server_environment,
            allowed_working_roots=(self.workspace.project_root,),
            timeout_seconds=remaining,
            max_output_bytes=16 * 1024 * 1024,
            inherited_directory_fds=profile_output.inherited_descriptors(),
        )
        run, source_state = self._start_run(
            plan,
            timeout_seconds=timeout_seconds,
            server_argv=server_request.argv,
            environment=environment,
            source_protocol_id=measurement_protocol_id,
            diagnostic_protocol=diagnostic_protocol,
            measurement_run_id=measurement_run_id,
        )
        try:
            try:
                lease = await self.broker.start_inference_server(
                    server_request,
                    host=parsed.hostname,
                    port=port,
                    readiness=readiness,
                    absolute_deadline=time.monotonic() + remaining,
                )
            except BaseException:
                await http_client.aclose()
                raise
            limitations = list(run.limitations)
            benchmark_exit_code: int | None = None
            benchmark_process = None
            finalization_duration_ns = 0
            export_duration_ns: int | None = None
            control = InferenceProfilerControlClient(
                plan.base_url,
                provider=plan.server_provider,
                http_client=http_client,
            )
            try:
                try:
                    control_deadline = time.monotonic() + max(
                        0.01,
                        (deadline_at - utc_now()).total_seconds(),
                    )
                    if plan.server_provider is InferenceServerProvider.SGLANG:
                        await control.start_async(
                            output_dir=profile_output.child_process_path(plan.output_relative_path),
                            profile_id=plan.sglang_profile_id,
                            options=plan.sglang_profile_options,
                            deadline_monotonic=control_deadline,
                        )
                    else:
                        await control.start_async(deadline_monotonic=control_deadline)
                    remaining = (deadline_at - utc_now()).total_seconds()
                    if remaining <= 0:
                        raise DomainError(ErrorCode.PROCESS_TIMEOUT, "Profiling deadline expired.")
                    outcome = await self.broker.run(
                        scenario_service._request(scenario_plan, scenario_output)
                    )
                    benchmark_process = outcome.process
                    benchmark_exit_code = process_exit_code(outcome.process.termination)
                    if benchmark_exit_code != 0:
                        limitations.append(
                            f"Profile workload exited with status {benchmark_exit_code}."
                        )
                except DomainError as error:
                    limitations.append(f"Profile window failed: {error.message}")
            finally:
                stop_remaining = (deadline_at - utc_now()).total_seconds()
                if stop_remaining <= 0:
                    limitations.append("No deadline remained for profiler flushing.")
                else:
                    try:
                        await control.stop_async(
                            deadline_monotonic=time.monotonic() + min(5.0, stop_remaining)
                        )
                    except DomainError as error:
                        limitations.append(error.message)
                try:
                    finalization_started = time.monotonic_ns()
                    server_outcome = await asyncio.shield(lease.close())
                    finalization_duration_ns = time.monotonic_ns() - finalization_started
                finally:
                    await http_client.aclose()

            if server_outcome.stdout:
                profile_output.write_bytes("server.stdout", server_outcome.stdout)
            if server_outcome.stderr:
                profile_output.write_bytes("server.stderr", server_outcome.stderr)

            if server_outcome.process.cleanup_complete is not True:
                limitations.append("Managed server process cleanup was incomplete.")

            if plan.profiler is ProfilerKind.NSIGHT_SYSTEMS:
                export_duration_ns = await self._export_nsight(
                    plan, profile_output, deadline_at, limitations
                )
            artifacts, runs, preservation_limitations = self._preserve(plan, profile_output)
            limitations.extend(preservation_limitations)
            extracted_runs = await self._extract_preserved(plan, runs, deadline_at, limitations)
            scenario_cleanup_limitation = scenario_service._cleanup_staging(
                trusted_root,
                scenario_plan.output_root,
                preservation_complete=True,
            )
            if scenario_cleanup_limitation is not None:
                limitations.append(scenario_cleanup_limitation)
            if artifacts and not extracted_runs:
                limitations.append(
                    "No recognized profiler trace was extracted from preserved artifacts."
                )
            operational_limitations = limitations[len(run.limitations) :]
            coverage = (
                InferenceProfileCoverage.COMPLETE
                if extracted_runs and not operational_limitations
                else (
                    InferenceProfileCoverage.PARTIAL
                    if artifacts
                    else InferenceProfileCoverage.UNAVAILABLE
                )
            )
            finished = self._finish_run(
                run,
                environment,
                source_state,
                artifact_ids=artifacts,
                artifact_run_ids=runs,
                process=benchmark_process,
                coverage=coverage,
                limitations=tuple(limitations),
            )
            cleanup_limitation = self._cleanup_staging(
                trusted_root,
                plan.output_root,
                preservation_complete=not preservation_limitations,
            )
            if cleanup_limitation is not None:
                limitations.append(cleanup_limitation)
                finished = self._append_limitation(
                    finished,
                    environment,
                    source_state,
                    cleanup_limitation,
                )
            return InferenceProfilingResult(
                run_id=finished.run_id,
                plan_id=plan.plan_id,
                measurement_protocol_id=measurement_protocol_id,
                measurement_run_id=measurement_run_id,
                scenario_name=scenario_name,
                profiler=plan.profiler,
                benchmark_exit_code=benchmark_exit_code,
                server_cleanup_complete=server_outcome.process.cleanup_complete is True,
                artifact_ids=artifacts,
                artifact_run_ids=runs,
                extracted_run_ids=extracted_runs,
                workload_duration_ns=(
                    benchmark_process.wall_time_ns if benchmark_process is not None else None
                ),
                finalization_duration_ns=finalization_duration_ns,
                export_duration_ns=export_duration_ns,
                symbol_resolution_status=(
                    "disabled"
                    if finished.semantics.configuration.get("symbol_resolution") == "disabled"
                    else None
                ),
                symbol_resolution_duration_ns=None,
                coverage=coverage,
                limitations=tuple(limitations),
            )
        except asyncio.CancelledError:
            artifacts, artifact_runs, _preservation_limitations = self._preserve(
                plan, profile_output
            )
            self._finish_cancelled_run(run, environment, source_state, artifacts, artifact_runs)
            raise
        except DomainError as error:
            artifacts, artifact_runs, _preservation_limitations = self._preserve(
                plan, profile_output
            )
            self._finish_failed_run(run, environment, source_state, error, artifacts, artifact_runs)
            error.run_id = run.run_id
            if artifacts:
                error.details = {
                    **error.details,
                    "partial_artifact_ids": artifacts,
                    "partial_artifact_run_ids": artifact_runs,
                }
            raise
        except Exception as cause:
            artifacts, artifact_runs, _preservation_limitations = self._preserve(
                plan, profile_output
            )
            internal_error = DomainError(
                ErrorCode.INTERNAL_ERROR,
                "Unexpected inference profiling failure.",
                run_id=run.run_id,
            )
            self._finish_failed_run(
                run, environment, source_state, internal_error, artifacts, artifact_runs
            )
            raise internal_error from cause

    def _start_run(
        self,
        plan: InferenceProfilingPlan,
        *,
        timeout_seconds: float,
        server_argv: tuple[str, ...],
        environment: EnvironmentRecord,
        source_protocol_id: str,
        diagnostic_protocol: InferenceProtocolIdentity,
        measurement_run_id: str,
    ) -> tuple[RunManifest, SourceState]:
        """Create the diagnostic run before server startup can mutate external state."""

        executable = Path(server_argv[0])
        if not executable.is_absolute():
            executable = plan.server_cwd / executable
        source_state = collect_partial_source_state(
            self.workspace,
            executable=executable if executable.is_file() else None,
        )
        diagnostic_json = json.dumps(
            diagnostic_protocol.model_dump(mode="json"),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        diagnostic_id = digest_model(diagnostic_protocol.model_dump(mode="json"))
        run = ExecutionRunManifest(
            run_id=new_id(),
            started_at=utc_now(),
            execution_status=ExecutionStatus.RUNNING,
            capture_status=CaptureStatus.RUNNING,
            validation_status=ValidationStatus.UNSUPPORTED,
            workload_definition_id=source_protocol_id,
            workload_instance_id=plan.plan_id,
            measurement_protocol_id=source_protocol_id,
            source_measurement_run_id=measurement_run_id,
            environment_id=environment.environment_id,
            source_state_id=source_state.source_state_id,
            semantics=RunSemantics(
                origin="internal",
                adapter=plan.profiler,
                configuration={
                    "diagnostic_protocol_id": diagnostic_id,
                    **(
                        {"symbol_resolution": plan.symbol_resolution}
                        if isinstance(plan, NsightSystemsProfilingPlan)
                        else {}
                    ),
                },
            ),
            command=CommandSpec(
                argv=server_argv,
                cwd=str(plan.server_cwd),
                env_overrides={name: "<redacted>" for name in plan.environment_names},
                timeout_seconds=timeout_seconds,
            ),
            inference_protocol_identity_id=diagnostic_id,
            inference_protocol_identity_json=diagnostic_json,
            limitations=(
                *plan.limitations,
                f"Diagnostic profile linked to measurement run {measurement_run_id}.",
            ),
        )
        projected = self.projections.create_run(
            run,
            environment=environment,
            source_state=source_state,
        )
        return projected.run, source_state

    def _validate_measurement_run(
        self,
        measurement_run_id: str,
        measurement_protocol_id: str,
        *,
        scenario_name: str,
    ) -> None:
        measurement_run = self.runs.read(measurement_run_id)
        if measurement_run.execution_status is not ExecutionStatus.SUCCEEDED:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The linked measurement run must have completed successfully.",
                run_id=measurement_run_id,
                next_action=tool_action(
                    ActionId.PLAN_INFERENCE_SCENARIO,
                    scenario_name=scenario_name,
                ),
            )
        if (
            measurement_run.capture_status is not CaptureStatus.REGISTERED
            or not measurement_run.artifacts
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The linked measurement run has no preserved provider evidence.",
                run_id=measurement_run_id,
                remediation=("Run the unprofiled inference scenario successfully, then retry.",),
                next_action=tool_action(
                    ActionId.PLAN_INFERENCE_SCENARIO,
                    scenario_name=scenario_name,
                ),
            )
        if (
            EvidenceQueryService(self.workspace)
            .measurements(run_id=measurement_run_id, limit=1)
            .total
            == 0
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The linked measurement run has no published measurements.",
                run_id=measurement_run_id,
                remediation=("Run the unprofiled inference scenario successfully, then retry.",),
                next_action=tool_action(
                    ActionId.PLAN_INFERENCE_SCENARIO,
                    scenario_name=scenario_name,
                ),
            )
        if measurement_run.measurement_protocol_id != measurement_protocol_id:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The linked measurement run does not match the profiling workload protocol.",
                run_id=measurement_run_id,
                details={
                    "expected_measurement_protocol_id": measurement_protocol_id,
                    "actual_measurement_protocol_id": measurement_run.measurement_protocol_id,
                },
                next_action=tool_action(
                    ActionId.PLAN_INFERENCE_SCENARIO,
                    scenario_name=scenario_name,
                ),
            )
        try:
            identity = InferenceProtocolIdentity.model_validate_json(
                measurement_run.inference_protocol_identity_json or ""
            )
        except ValueError as exc:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The linked run has no valid inference protocol identity.",
                run_id=measurement_run_id,
                next_action=tool_action(
                    ActionId.PLAN_INFERENCE_SCENARIO,
                    scenario_name=scenario_name,
                ),
            ) from exc
        if identity.profiler.attached:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The linked measurement run must be unprofiled.",
                run_id=measurement_run_id,
                next_action=tool_action(
                    ActionId.PLAN_INFERENCE_SCENARIO,
                    scenario_name=scenario_name,
                ),
            )

    def _finish_run(
        self,
        run: RunManifest,
        environment: EnvironmentRecord,
        source_state: SourceState,
        *,
        artifact_ids: tuple[str, ...],
        artifact_run_ids: tuple[str, ...],
        process: ProcessResult | None,
        coverage: InferenceProfileCoverage,
        limitations: tuple[str, ...],
    ) -> RunManifest:
        status = (
            ExecutionStatus.TIMED_OUT
            if process is not None and process.timed_out
            else ExecutionStatus.SUCCEEDED
            if (
                process is not None
                and process_exit_code(process.termination) == 0
                and coverage is InferenceProfileCoverage.COMPLETE
            )
            else ExecutionStatus.FAILED
        )
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
                "limitations": tuple(dict.fromkeys(limitations)),
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
                    dict.fromkeys((*run.limitations, "Inference profiling was cancelled."))
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
                "limitations": tuple(dict.fromkeys((*run.limitations, error.message))),
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

    async def _export_nsight(
        self,
        plan: InferenceProfilingPlan,
        output: BoundDirectory,
        deadline_at: datetime,
        limitations: list[str],
    ) -> int | None:
        if plan.nsys_executable is None:
            limitations.append("Nsight export executable identity is unavailable.")
            return None
        remaining = (deadline_at - utc_now()).total_seconds()
        if remaining <= 0:
            limitations.append("No deadline remained for Nsight SQLite export.")
            return None
        try:
            with output.open_file(plan.output_relative_path):
                pass
        except DomainError:
            limitations.append("Nsight did not emit its native report.")
            return None
        sqlite_relative_path = str(Path(plan.output_relative_path).with_suffix(".sqlite"))
        sqlite_path = output.child_process_path(sqlite_relative_path)
        report_path = output.child_process_path(plan.output_relative_path)
        started_ns = time.monotonic_ns()
        try:
            outcome = await self.broker.run(
                ExecutionRequest(
                    argv=(
                        str(plan.nsys_executable),
                        "export",
                        "--type",
                        "sqlite",
                        "--output",
                        str(sqlite_path),
                        str(report_path),
                    ),
                    executable_binding=plan.nsys_executable_binding,
                    cwd=self.workspace.project_root,
                    environment_allowlist=(
                        self.workspace.config.execution.child_environment_allowlist
                    ),
                    allowed_working_roots=(self.workspace.project_root,),
                    timeout_seconds=remaining,
                    max_output_bytes=4 * 1024 * 1024,
                    inherited_directory_fds=output.inherited_descriptors(),
                )
            )
            if process_exit_code(outcome.process.termination) != 0:
                limitations.append(
                    "Nsight SQLite export exited with status "
                    f"{process_exit_code(outcome.process.termination)}."
                )
        except DomainError as error:
            limitations.append(f"Nsight SQLite export failed: {error.message}")
        return time.monotonic_ns() - started_ns

    def _preserve(
        self,
        plan: InferenceProfilingPlan,
        output: BoundDirectory | None = None,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        if output is None:
            with (
                TrustedRoot(self.workspace.paths.staging) as trusted_root,
                trusted_root.open_directory(plan.output_root) as reopened,
            ):
                return self._preserve(plan, reopened)
        candidates = output.admitted_files(
            frozenset(
                {
                    plan.output_relative_path,
                    str(Path(plan.output_relative_path).with_suffix(".sqlite")),
                    "server.stdout",
                    "server.stderr",
                }
            ),
            suffixes=(".pt.trace.json", ".pt.trace.json.gz", ".pftrace"),
            max_depth=3,
            max_entries=4_096,
            max_files=64,
        )
        artifacts: list[str] = []
        runs: list[str] = []
        limitations: list[str] = []
        importer = ImportService(self.workspace)
        for candidate in candidates:
            display_name = Path(candidate.relative_path).name
            trace_file = display_name.endswith((".json", ".json.gz", ".pftrace", ".sqlite"))
            server_output = display_name.startswith("server.")
            try:
                with output.open_file(candidate) as descriptor:
                    imported = importer.import_descriptor(
                        ImportDescriptorRequest(
                            descriptor=descriptor,
                            display_name=display_name,
                            kind=(
                                ArtifactKind.EXECUTION_TRACE
                                if trace_file
                                else ArtifactKind.PROCESS_OUTPUT
                                if server_output
                                else ArtifactKind.COLLECTOR_METADATA
                            ),
                            sensitivity=(
                                Sensitivity.SENSITIVE if server_output else Sensitivity.INTERNAL
                            ),
                            role=(
                                "inference_server_output" if server_output else "inference_profile"
                            ),
                            producer=(
                                "nsys"
                                if plan.profiler is ProfilerKind.NSIGHT_SYSTEMS
                                else "sglang.torch_profiler"
                                if plan.server_provider is InferenceServerProvider.SGLANG
                                else "torch_profiler"
                            ),
                        )
                    )
            except DomainError as error:
                limitations.append(
                    f"Profiler artifact {display_name!r} could not be preserved: {error.message}"
                )
                continue
            artifacts.append(imported.artifact_id)
            runs.append(imported.run.run_id)
        return tuple(artifacts), tuple(runs), tuple(limitations)

    async def _extract_preserved(
        self,
        plan: InferenceProfilingPlan,
        run_ids: tuple[str, ...],
        deadline_at: datetime,
        limitations: list[str],
    ) -> tuple[str, ...]:
        """Feed supported native traces through existing bounded extractors."""
        extracted: list[str] = []
        run_store = RunStore(self.workspace)
        for run_id in run_ids:
            run = run_store.read(run_id)
            traces = [item for item in run.artifacts if item.kind is ArtifactKind.EXECUTION_TRACE]
            if len(traces) != 1:
                continue
            name = traces[0].display_name
            remaining = (deadline_at - utc_now()).total_seconds()
            if remaining <= 0:
                limitations.append("No deadline remained for profile evidence extraction.")
                break
            try:
                if plan.profiler is ProfilerKind.NSIGHT_SYSTEMS and name.endswith(".sqlite"):
                    from flameox.adapters.nsight_systems import NsightSystemsExtractor

                    await asyncio.wait_for(
                        NsightSystemsExtractor(self.workspace, broker=self.broker).extract(run_id),
                        timeout=remaining,
                    )
                elif plan.profiler is ProfilerKind.TORCH_PROFILER and name.endswith(
                    (".json", ".json.gz", ".pftrace")
                ):
                    from flameox.adapters.perfetto import PerfettoExtractor

                    await asyncio.wait_for(
                        PerfettoExtractor(self.workspace, broker=self.broker).extract(run_id),
                        timeout=remaining,
                    )
                else:
                    continue
            except TimeoutError:
                limitations.append(f"Profile extraction deadline expired for {name}.")
            except DomainError as error:
                limitations.append(f"Profile extraction failed for {name}: {error.message}")
            else:
                extracted.append(run_id)
        return tuple(extracted)

    @staticmethod
    def _cleanup_staging(
        trusted_root: TrustedRoot,
        reference: BoundDirectoryReference,
        *,
        preservation_complete: bool,
    ) -> str | None:
        if not preservation_complete:
            return (
                "Profiler staging was retained because native artifact preservation was incomplete."
            )
        try:
            trusted_root.remove_directory(reference)
        except DomainError:
            return "Profiler staging cleanup failed; immutable artifacts remain authoritative."
        return None


class InferenceProfilerControlClient:
    """Provider protocol over Flameox's single bounded loopback transport."""

    def __init__(
        self,
        base_url: str,
        *,
        provider: InferenceServerProvider = InferenceServerProvider.VLLM,
        timeout_seconds: float = 5.0,
        http_client: BoundedHttpClient | None = None,
    ) -> None:
        self.base_url = validate_loopback_base_url(base_url)
        self.provider = provider
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client

    def start(
        self,
        *,
        output_dir: Path | None = None,
        profile_id: str | None = None,
        options: SglangProfileOptions | None = None,
        deadline_monotonic: float | None = None,
    ) -> None:
        self._post(
            "/start_profile",
            self._start_payload(output_dir, profile_id, options),
            deadline_monotonic=deadline_monotonic,
        )

    async def start_async(
        self,
        *,
        output_dir: Path | None = None,
        profile_id: str | None = None,
        options: SglangProfileOptions | None = None,
        deadline_monotonic: float | None = None,
    ) -> None:
        await self._post_async(
            "/start_profile",
            self._start_payload(output_dir, profile_id, options),
            deadline_monotonic=deadline_monotonic,
        )

    def stop(self, *, deadline_monotonic: float | None = None) -> None:
        self._post("/stop_profile", None, deadline_monotonic=deadline_monotonic)

    async def stop_async(self, *, deadline_monotonic: float | None = None) -> None:
        await self._post_async(
            "/stop_profile",
            None,
            deadline_monotonic=deadline_monotonic,
        )

    def _post(
        self,
        path: Literal["/start_profile", "/stop_profile"],
        payload: dict[str, JsonValue] | None,
        *,
        deadline_monotonic: float | None,
    ) -> None:
        client = self._http_client or BoundedHttpClient()
        try:
            client.request_loopback(
                self._request(path, payload, deadline_monotonic=deadline_monotonic)
            )
        except BoundedHttpError as exc:
            raise DomainError(
                ErrorCode.PROCESS_FAILED,
                f"{self.provider} profiler control failed at {path}.",
            ) from exc
        finally:
            if self._http_client is None:
                client.close()

    async def _post_async(
        self,
        path: Literal["/start_profile", "/stop_profile"],
        payload: dict[str, JsonValue] | None,
        *,
        deadline_monotonic: float | None,
    ) -> None:
        client = self._http_client or BoundedHttpClient()
        try:
            await client.request_loopback_async(
                self._request(path, payload, deadline_monotonic=deadline_monotonic)
            )
        except BoundedHttpError as exc:
            raise DomainError(
                ErrorCode.PROCESS_FAILED,
                f"{self.provider} profiler control failed at {path}.",
            ) from exc
        finally:
            if self._http_client is None:
                await client.aclose()

    def _request(
        self,
        path: Literal["/start_profile", "/stop_profile"],
        payload: dict[str, JsonValue] | None,
        *,
        deadline_monotonic: float | None,
    ) -> LoopbackHttpRequest:
        return LoopbackHttpRequest(
            base_url=self.base_url,
            method=HttpMethod.POST,
            path=path,
            deadline_monotonic=(
                deadline_monotonic
                if deadline_monotonic is not None
                else time.monotonic() + self.timeout_seconds
            ),
            max_response_bytes=64 * 1024,
            json_body=payload,
        )

    def _start_payload(
        self,
        output_dir: Path | None,
        profile_id: str | None,
        options: SglangProfileOptions | None,
    ) -> dict[str, JsonValue] | None:
        if self.provider is not InferenceServerProvider.SGLANG:
            return None
        if output_dir is None or profile_id is None or options is None:
            raise ValueError("SGLang profiling requires generated profile options")
        return {
            "output_dir": str(output_dir),
            "profile_id": profile_id,
            "start_step": options.start_step,
            "num_steps": options.num_steps,
            "activities": list(options.activities),
            "profile_by_stage": options.profile_by_stage,
            "record_shapes": options.record_shapes,
            "with_stack": options.with_stack,
        }
