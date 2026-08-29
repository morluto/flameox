from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import secrets
import shutil
import stat
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, cast

from packaging.requirements import Requirement
from pydantic import JsonValue, TypeAdapter

from flameox.action_graph import ActionId, NextAction, manual_action, tool_action
from flameox.adapters.builtins import (
    BUILTIN_ADAPTERS,
    AdapterDependencyKind,
    build_capture_invocation,
    builtin_adapter,
    node_version_is_supported,
    replace_compute_sanitizer_suppression,
)
from flameox.adapters.compute_sanitizer import (
    compute_sanitizer_compatibility_limitations,
    inspect_compute_sanitizer_report,
)
from flameox.adapters.kernel_build import KernelBuildManifestV2
from flameox.adapters.options import (
    bind_adapter_options,
    compute_sanitizer_options,
    read_compute_sanitizer_suppression,
)
from flameox.adapters.options import (
    run_semantics as build_run_semantics,
)
from flameox.adapters.registry import AdapterDescriptor, AdapterRegistry
from flameox.adapters.torch_profiler import SdkTorchProfilerOptions, torch_profiler_options
from flameox.application.async_work import run_atomic_thread
from flameox.application.capabilities import CapabilityService
from flameox.application.capture_admission import CaptureAdmission, CaptureAdmissionService
from flameox.application.environment import AcceleratorIdentityService, collect_environment
from flameox.application.evidence_rows import (
    process_observation_rows,
    runtime_resource_summary_row,
    runtime_writable_root_rows,
)
from flameox.application.execution_identity import ExecutionIdentityService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.kernel_builds import (
    KernelBuildCaptureCollector,
    kernel_build_pipeline_request,
    managed_kernel_build_context,
)
from flameox.application.nvbench_imports import (
    collect_nvbench_sidecar_specs,
    load_nvbench_document_with_integrity,
    load_nvbench_sidecar_specs,
    resolve_nvbench_sidecar_path,
    validate_nvbench_sidecar_file,
)
from flameox.application.oracle_receipts import parse_oracle_receipt
from flameox.application.pipelines import ArtifactPipelineService
from flameox.application.preflight import PreflightService
from flameox.application.proc import read_boot_id, read_proc_stat_start_identity
from flameox.application.progress import ProgressReporter
from flameox.application.projections import ProjectionCoordinator
from flameox.application.python_environment import PythonEnvironmentProbe
from flameox.application.quarantine import QuarantineService
from flameox.application.source import collect_source_state
from flameox.application.staging_ownership import StagingOwnership, StagingOwnershipService
from flameox.application.workloads import Scalar, WorkloadService
from flameox.atomic import atomic_write_bytes
from flameox.command_binding import ExecutableResolver
from flameox.config import ContainmentPolicy, NetworkPolicy
from flameox.domain import (
    AcceleratorIdentityFacet,
    AcceleratorIdentityStatus,
    AdapterExecutionPlan,
    AdapterExtractionResult,
    AdapterPlanRequest,
    AdapterProbeContext,
    AdapterProbeResult,
    AdapterProbeStatus,
    AdapterValidationResult,
    ArtifactKind,
    ArtifactRegistration,
    CapabilityPermissionStatus,
    CapabilityProvisioning,
    CapabilityReport,
    CapabilityStatus,
    CaptureContainment,
    CaptureLease,
    CapturePlan,
    CaptureStatus,
    CommandSpec,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ExternalExecutionContext,
    IdentityQuality,
    LimitationDetail,
    LimitationSource,
    OracleReceiptRecord,
    OracleStatus,
    OracleStrength,
    PreflightDisposition,
    PreflightMode,
    PreflightReport,
    ProbeKind,
    ProcessCancellationCause,
    ProcessResult,
    RequirementKind,
    RequirementStatus,
    RunManifest,
    Sensitivity,
    ValidationStatus,
    WritableRootBinding,
    digest_model,
    new_id,
)
from flameox.domain.executables import (
    ExecutableResolutionRequest,
    ExecutableTrustPolicy,
    ResolvedExecutable,
)
from flameox.domain.models import ExecutionRunManifest, parse_capture_plan, utc_now
from flameox.evidence import GenerationPublisher
from flameox.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    ProcessContainment,
    ProcessExecutionError,
    ProcessObservation,
    ResourcePolicy,
    SubprocessBroker,
)
from flameox.models import ContractModel
from flameox.observability import OperationLogger, elapsed_ms
from flameox.startup_profile import PYTHON_STARTUP_PROFILE
from flameox.storage import ArtifactStore, AuthorizedPlanStore, RunStore, StorageQuota, Workspace


def _limitation(source: LimitationSource, code: str, message: str) -> LimitationDetail:
    return LimitationDetail(source=source, code=code, message=message)


def _merge_limitation_details(
    *groups: tuple[LimitationDetail, ...],
) -> tuple[LimitationDetail, ...]:
    result: list[LimitationDetail] = []
    seen: set[tuple[str, str, str]] = set()
    for group in groups:
        for detail in group:
            key = (detail.source, detail.code, detail.message)
            if key not in seen:
                seen.add(key)
                result.append(detail)
    return tuple(result)


def _limitation_projection(
    details: tuple[LimitationDetail, ...],
    existing: tuple[str, ...] = (),
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for message in (*existing, *(item.message for item in details)):
        if message not in seen:
            seen.add(message)
            result.append(message)
    return tuple(result)


def _preflight_limitation_details(preflight: PreflightReport) -> tuple[LimitationDetail, ...]:
    details: list[LimitationDetail] = []
    for item in preflight.requirements:
        details.extend(
            _limitation(
                LimitationSource.PREFLIGHT,
                f"requirement.{item.status}",
                f"{item.requirement}: {message}",
            )
            for message in item.limitations
            or (
                (f"Requirement status is {item.status}.",)
                if item.status is not RequirementStatus.AVAILABLE
                else ()
            )
        )
    details.extend(
        _limitation(LimitationSource.PREFLIGHT, "preflight.limitation", message)
        for message in preflight.limitations
    )
    return _merge_limitation_details(tuple(details))


class CaptureResult(ContractModel):
    schema_version: int = 1
    plan: CapturePlan
    run: RunManifest
    corpus_commit_id: str


@dataclass(frozen=True, slots=True)
class _AdapterBinding:
    argv: tuple[str, ...]
    artifact_kinds: tuple[ArtifactKind, ...]
    expected_overhead: str
    limitations: tuple[str, ...]
    environment: dict[str, str]
    limitation_details: tuple[LimitationDetail, ...]
    permissions: tuple[str, ...]
    version: str | None
    execution_plan: AdapterExecutionPlan | None = None
    package_identity: str | None = None


@dataclass(slots=True)
class _CaptureExecution:
    service: CaptureService
    plan: CapturePlan
    output_root: Path
    logger: OperationLogger
    operation_id: str
    started_monotonic: float
    progress: ProgressReporter
    staging_ownership: StagingOwnership
    run: RunManifest | None = None
    cleanup_complete: bool | None = None

    async def report(self, completed: int, message: str) -> None:
        self.logger.emit(
            operation_id=self.operation_id,
            operation="capture.execute",
            phase=message,
            run_id=self.plan.run_id,
            adapter=self.plan.adapter,
            elapsed_ms=elapsed_ms(self.started_monotonic),
        )
        await self.progress.report(completed, 8, message)

    async def record_lease(self, process_id: int) -> None:
        if self.run is None:
            raise DomainError(ErrorCode.INTERNAL_ERROR, "capture run is not initialized")
        lease = self.service._lease(process_id)
        if lease is None:
            return
        current = self.run
        leased = current.validated_copy(
            update={
                "revision": current.revision + 1,
                "lease": lease,
            }
        )
        lease_write = asyncio.create_task(
            run_atomic_thread(
                lambda: self.service.runs.append(
                    leased,
                    expected_revision=current.revision,
                )
            )
        )
        try:
            self.run = await asyncio.shield(lease_write)
        except asyncio.CancelledError:
            try:
                self.run = await asyncio.shield(lease_write)
            except Exception as lease_error:
                logging.getLogger("flameox.capture").warning(
                    "Lease write after cancellation raised: %s",
                    lease_error,
                )
            raise

    async def record_cleanup(self, complete: bool) -> None:
        self.cleanup_complete = complete

    async def terminate_cancelled(
        self,
        *,
        message: str,
        phase: str,
        cleanup_complete: bool | None = None,
    ) -> None:
        """Persist cancellation without allowing finalization failures to mask it."""

        try:
            await asyncio.shield(
                self.terminate(
                    execution=ExecutionStatus.CANCELLED,
                    message=message,
                    phase=phase,
                    error_code="cancelled",
                    cleanup_complete=cleanup_complete,
                )
            )
        except Exception as terminate_error:
            logging.getLogger("flameox.capture").warning(
                "Cleanup after cancellation raised: %s",
                terminate_error,
            )

    async def terminate(
        self,
        *,
        execution: ExecutionStatus,
        message: str,
        phase: str,
        error_code: str,
        cleanup_complete: bool | None = None,
        process: ProcessResult | None = None,
        process_observations: tuple[ProcessObservation, ...] = (),
        limitation_details: tuple[LimitationDetail, ...] = (),
        artifacts: tuple[ArtifactRegistration, ...] = (),
    ) -> RunManifest:
        if self.run is None:
            raise DomainError(ErrorCode.INTERNAL_ERROR, "capture run is not initialized")
        self.logger.emit(
            operation_id=self.operation_id,
            operation="capture.execute",
            phase=phase,
            run_id=self.plan.run_id,
            adapter=self.plan.adapter,
            elapsed_ms=elapsed_ms(self.started_monotonic),
            error_code=error_code,
        )
        current = self.run
        snapshot_rows: tuple[list[dict[str, object]], list[dict[str, object]]] = ([], [])
        try:
            snapshot_path = self.output_root / "process-snapshot.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": self.plan.run_id,
                        "observations": [
                            item.model_dump(mode="json") for item in process_observations
                        ],
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            snapshot_artifact = self.service.artifacts.import_path(
                snapshot_path,
                allowed_roots=(self.output_root,),
                max_bytes=self.service.workspace.config.capture.max_artifact_bytes,
            )
            snapshot_registration = ArtifactRegistration(
                registration_id=new_id(),
                run_id=self.plan.run_id,
                artifact_id=snapshot_artifact.content.artifact_id,
                display_name="process-snapshot.json",
                media_type="application/json",
                kind=ArtifactKind.PROCESS_TREE_SNAPSHOT,
                role="process_observation",
                producer="flameox.execution",
                producer_version="1",
                sensitivity=Sensitivity.INTERNAL,
            )
            snapshot_rows = process_observation_rows(
                self.plan.run_id,
                process_observations,
                artifact_id=snapshot_artifact.content.artifact_id,
            )
            artifacts = (*artifacts, snapshot_registration)
        except (DomainError, OSError) as snapshot_error:
            limitation_details = (
                *limitation_details,
                _limitation(
                    LimitationSource.COLLECTOR,
                    "snapshot_artifact_unavailable",
                    f"Process snapshot artifact could not be retained: {snapshot_error}",
                ),
            )

        def finish_error(run: RunManifest) -> RunManifest:
            return self.service._finish_error(
                run,
                execution=execution,
                message=message,
                cleanup_complete=(
                    self.cleanup_complete if cleanup_complete is None else cleanup_complete
                ),
                process=process,
                process_snapshot_rows=snapshot_rows[0],
                process_snapshot_entries=snapshot_rows[1],
                limitation_details=limitation_details,
                artifacts=artifacts,
            )

        try:
            for attempt in range(4):
                try:
                    terminal = await run_atomic_thread(partial(finish_error, current))
                    break
                except DomainError as error:
                    if error.code is not ErrorCode.REVISION_CONFLICT or attempt == 3:
                        raise
                    current = await run_atomic_thread(
                        lambda: self.service.runs.read(self.plan.run_id)
                    )
                    if current.finished_at is not None:
                        terminal = current
                        break
                    self.run = current
        finally:
            self.cleanup_staging()
        self.run = terminal
        return terminal

    def cleanup_staging(self) -> None:
        shutil.rmtree(self.output_root, ignore_errors=True)
        self.staging_ownership.release()
        self.staging_ownership.forget_if_removed(self.output_root)

    async def terminate_startup_failure(
        self,
        *,
        message: str,
        error_code: str,
    ) -> RunManifest | None:
        try:
            return await self.terminate(
                execution=ExecutionStatus.FAILED,
                message=message,
                phase="startup identity collection failed",
                error_code=error_code,
            )
        except Exception as terminate_error:
            self.cleanup_staging()
            logging.getLogger("flameox.capture").warning(
                "Finalizing a startup identity failure raised: %s",
                terminate_error,
            )
            return None


class CapturePlanRegistry:
    """Durable, server-owned authorization tokens for capture execution."""

    def __init__(
        self,
        *,
        workspace: Workspace | None = None,
        ttl_seconds: float = 300,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self._workspace: Workspace | None = None
        self._store: AuthorizedPlanStore[CapturePlan] | None = None
        self._admissions: CaptureAdmissionService | None = None
        if workspace is not None:
            self.bind(workspace)

    def bind(self, workspace: Workspace) -> None:
        if self._workspace is not None and self._workspace.paths.root != workspace.paths.root:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "A capture plan registry cannot span multiple workspaces.",
            )
        self._workspace = workspace
        self._store = AuthorizedPlanStore(
            workspace,
            family="capture",
            model=TypeAdapter(CapturePlan),
        )
        self._admissions = CaptureAdmissionService(
            workspace,
            limit=workspace.config.capture.max_parallel_captures,
        )

    async def acquire_capture_slot(self, run_id: str) -> CaptureAdmission:
        if self._admissions is None:
            raise DomainError(
                ErrorCode.WORKSPACE_NOT_FOUND,
                "Capture admission requires an initialized workspace.",
            )
        return await self._admissions.acquire(run_id)

    async def issue(self, plan: CapturePlan) -> None:
        self._require_store().issue(
            plan.plan_token,
            plan.request_digest,
            plan,
            expires_at=plan.expires_at,
        )

    async def consume(self, plan_token: str) -> CapturePlan:
        return self._require_store().consume(plan_token)

    async def inspect(self, plan_token: str) -> CapturePlan:
        return self._require_store().inspect(plan_token)

    def _require_store(self) -> AuthorizedPlanStore[CapturePlan]:
        if self._store is None:
            raise DomainError(
                ErrorCode.WORKSPACE_NOT_FOUND,
                "Capture plan storage requires an initialized workspace.",
            )
        return self._store


class CaptureService:
    def __init__(
        self,
        workspace: Workspace,
        *,
        plans: CapturePlanRegistry | None = None,
        broker: SubprocessBroker | None = None,
        capabilities: CapabilityService | None = None,
    ) -> None:
        self.workspace = workspace
        self.workloads = WorkloadService(workspace)
        self.capabilities = capabilities or CapabilityService(workspace, broker=broker)
        self.plans = plans or CapturePlanRegistry(
            workspace=workspace,
        )
        self.plans.bind(workspace)
        self.broker = broker or self.capabilities.broker
        self.runs = RunStore(workspace)
        self.artifacts = ArtifactStore(workspace)
        self.publisher = GenerationPublisher(workspace)
        self.projections = ProjectionCoordinator(workspace)
        self.executables = ExecutableResolver()

    async def plan(
        self,
        *,
        workload_name: str,
        adapter: str,
        parameters: dict[str, Scalar] | None = None,
        adapter_options: dict[str, JsonValue] | None = None,
        execution_policy: ExecutionPolicy,
        dynamic_parameters: tuple[str, ...] = (),
        preflight_mode: PreflightMode = PreflightMode.AUTO,
        external_context: ExternalExecutionContext | None = None,
    ) -> CapturePlan:
        instance = self.workloads.resolve(
            workload_name,
            parameters,
            dynamic_parameters=dynamic_parameters,
        )
        definition = self.workloads.definition(workload_name)
        inspection_mode = (
            ProbeKind.ACTIVE if preflight_mode is PreflightMode.AUTO else ProbeKind(preflight_mode)
        )
        preflight = await PreflightService(
            self.workspace,
            capabilities=self.capabilities,
        ).inspect(workload_name, mode=inspection_mode)
        if preflight.disposition is PreflightDisposition.BLOCKED:
            missing_distributions = tuple(
                item.requirement
                for item in preflight.requirements
                if item.kind is RequirementKind.PYTHON_DISTRIBUTION
                and item.status is RequirementStatus.ABSENT
            )
            next_action: NextAction = tool_action(
                ActionId.GET_DECLARED_WORKFLOW,
                kind="workload",
                name=workload_name,
            )
            if missing_distributions:
                next_action = manual_action(
                    "Install the missing distributions in the workload's declared Python "
                    "environment or select another workload, then plan capture again.",
                    suggested_action=ActionId.GET_DECLARED_WORKFLOW,
                )
            elif requirement_action := next(
                (
                    item.next_action
                    for item in preflight.requirements
                    if item.next_action is not None
                ),
                None,
            ):
                next_action = requirement_action
            elif any(item.kind is RequirementKind.CAPABILITY for item in preflight.requirements):
                next_action = tool_action(ActionId.INSPECT_CAPABILITIES)
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Required preflight checks failed for workload {workload_name!r}.",
                details={
                    "preflight": preflight.model_dump(mode="json"),
                    "workload_name": workload_name,
                    "missing_python_distributions": list(missing_distributions),
                },
                remediation=tuple(
                    remediation
                    for item in preflight.requirements
                    for remediation in item.remediation
                ),
                next_action=next_action,
            )
        adapter_capability = await self._adapter_capability(
            adapter,
            mode=preflight_mode,
            workload_name=workload_name,
        )
        if adapter == "compute-sanitizer" and execution_policy is not ExecutionPolicy.TRUSTED_LOCAL:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Managed Compute Sanitizer capture cannot access NVIDIA devices in the current "
                "containment backend.",
                details={"adapter": adapter, "requested_execution_policy": execution_policy.value},
                remediation=(
                    "Use trusted-local capture on a trusted NVIDIA host until bounded GPU device "
                    "bindings are supported.",
                ),
            )
        planned_execution_identity = ExecutionIdentityService(
            self.workspace,
            broker=self.broker,
        ).plan(workload_name, cwd=Path(instance.command.cwd))
        plan_id = secrets.token_hex(32)
        run_id = new_id()
        output_root = self.workspace.paths.staging / "captures" / plan_id
        writable_roots = tuple(
            WritableRootBinding(
                target_path=str(target),
                storage_path=str(output_root / "writable" / str(index)),
                target_identity=identity,
            )
            for index, (target, identity) in enumerate(
                self.workloads.writable_targets(workload_name)
            )
        )
        collector_environment = {
            "FLAMEOX_OBSERVATIONS_PATH": str(output_root / "observations.jsonl")
        }
        bound_adapter_options = bind_adapter_options(
            adapter,
            adapter_options,
            project_root=self.workspace.project_root,
        )
        adapter_binding = await self._adapter_command(
            adapter,
            instance.command,
            output_root,
            capability=adapter_capability,
            options=bound_adapter_options,
            workload_executable=str(instance.executable_binding.invocation_path),
        )
        collector_environment.update(adapter_binding.environment)
        collector_argv = adapter_binding.argv
        kinds = adapter_binding.artifact_kinds
        overhead = adapter_binding.expected_overhead
        warnings = adapter_binding.limitations
        limitation_details = _merge_limitation_details(
            _preflight_limitation_details(preflight),
            adapter_binding.limitation_details,
        )
        adapter_version = adapter_binding.version
        semantics = build_run_semantics(adapter, adapter_version, bound_adapter_options)
        use_containment = execution_policy is not ExecutionPolicy.TRUSTED_LOCAL
        containment, network_contained, systemd_scope_unit, collector_argv = await self._contain(
            collector_argv,
            cwd=Path(instance.command.cwd),
            writable=output_root,
            writable_roots=writable_roots,
            unit_name=f"flameox-capture-{plan_id[:24]}.scope",
            required=(
                execution_policy.requires_containment(self.workspace.config.execution.containment)
            ),
            use_containment=use_containment,
        )
        collector_executable_binding = self.executables.resolve(
            ExecutableResolutionRequest(
                token=collector_argv[0],
                cwd=Path(instance.command.cwd),
                environment={**os.environ, **collector_environment},
                policy=ExecutableTrustPolicy.TRUSTED_HOST_TOOL,
            )
        )
        collector_argv = (
            str(collector_executable_binding.invocation_path),
            *collector_argv[1:],
        )
        oracle = self.workloads.resolve_oracle(
            workload_name,
            parameters,
            dynamic_parameters=dynamic_parameters,
        )
        oracle_argv: tuple[str, ...] | None = None
        oracle_executable_binding: ResolvedExecutable | None = None
        oracle_launch_executable_binding: ResolvedExecutable | None = None
        oracle_containment: CaptureContainment | None = None
        oracle_network_contained: bool | None = None
        oracle_systemd_scope_unit: str | None = None
        if oracle is not None:
            oracle_executable_binding = oracle.executable_binding
            (
                oracle_containment,
                oracle_network_contained,
                oracle_systemd_scope_unit,
                oracle_argv,
            ) = await self._contain(
                oracle.command.argv,
                cwd=Path(oracle.command.cwd),
                writable=output_root,
                writable_roots=writable_roots,
                unit_name=f"flameox-validation-{plan_id[:21]}.scope",
                required=execution_policy.requires_containment(
                    self.workspace.config.execution.containment
                ),
                use_containment=use_containment,
            )
            if containment in {"active", "degraded"} and oracle_containment not in {
                "active",
                "degraded",
            }:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "The validation oracle cannot preserve the capture containment.",
                )
            if network_contained and not oracle_network_contained:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "The validation oracle cannot preserve capture network isolation.",
                )
            oracle_launch_executable_binding = self.executables.resolve(
                ExecutableResolutionRequest(
                    token=oracle_argv[0],
                    cwd=Path(oracle.command.cwd),
                    environment=dict(os.environ),
                    policy=ExecutableTrustPolicy.TRUSTED_HOST_TOOL,
                )
            )
            oracle_argv = (
                str(oracle_launch_executable_binding.invocation_path),
                *oracle_argv[1:],
            )
        if execution_policy is ExecutionPolicy.TRUSTED_LOCAL:
            warnings = (
                *warnings,
                "Trusted-local execution selected; the workload runs directly without enforced "
                "descendant containment.",
            )
            limitation_details = (
                *limitation_details,
                _limitation(
                    LimitationSource.CONTAINMENT,
                    "trusted_local_execution",
                    "The workload runs directly; descendant cleanup and resource isolation are "
                    "not enforced by a containment backend.",
                ),
            )
        identities: dict[str, JsonValue] = {
            "collector_executable": cast(
                JsonValue,
                collector_executable_binding.model_dump(mode="json"),
            ),
            "workload_executable": cast(
                JsonValue,
                instance.executable_binding.model_dump(mode="json"),
            ),
        }
        if adapter_binding.package_identity is not None:
            identities["adapter_package_identity"] = adapter_binding.package_identity
        created_at = utc_now()
        request: dict[str, Any] = {
            "workspace_id": self.workspace.identity.workspace_id,
            "run_id": run_id,
            "workload_name": workload_name,
            "workload_definition_id": definition.workload_definition_id,
            "instance": instance.model_dump(mode="json"),
            "dynamic_parameters": dynamic_parameters,
            "semantics": semantics.model_dump(mode="json"),
            "adapter_execution_plan": (
                adapter_binding.execution_plan.model_dump(mode="json")
                if adapter_binding.execution_plan is not None
                else None
            ),
            "execution_policy": execution_policy.value,
            "collector_argv": collector_argv,
            "collector_executable_binding": collector_executable_binding.model_dump(mode="json"),
            "collector_environment": collector_environment,
            "oracle_argv": oracle_argv,
            "oracle_executable_binding": (
                oracle_executable_binding.model_dump(mode="json")
                if oracle_executable_binding is not None
                else None
            ),
            "oracle_launch_executable_binding": (
                oracle_launch_executable_binding.model_dump(mode="json")
                if oracle_launch_executable_binding is not None
                else None
            ),
            "oracle_containment": oracle_containment,
            "oracle_network_contained": oracle_network_contained,
            "oracle_systemd_scope_unit": oracle_systemd_scope_unit,
            "bound_identities": identities,
            "preflight": preflight.model_dump(mode="json"),
            "adapter_capability": (
                adapter_capability.model_dump(mode="json")
                if adapter_capability is not None
                else None
            ),
            "warnings": warnings,
            "limitation_details": [item.model_dump(mode="json") for item in limitation_details],
            "writable_roots": [item.model_dump(mode="json") for item in writable_roots],
            "external_context": (
                external_context.model_dump(mode="json") if external_context is not None else None
            ),
            "planned_execution_identity": planned_execution_identity.model_dump(mode="json"),
            "policy": self.workspace.config.model_dump(mode="json"),
            "containment": containment,
            "systemd_scope_unit": systemd_scope_unit,
        }
        plan = parse_capture_plan(
            {
                "plan_token": secrets.token_hex(32),
                "plan_id": plan_id,
                "run_id": run_id,
                "request_digest": digest_model(request),
                "workspace_id": self.workspace.identity.workspace_id,
                "workload_name": workload_name,
                "workload_definition_id": definition.workload_definition_id,
                "workload_instance": instance,
                "semantics": semantics,
                "dynamic_parameters": dynamic_parameters,
                "adapter_execution_plan": (
                    adapter_binding.execution_plan.model_dump(mode="json")
                    if adapter_binding.execution_plan is not None
                    else None
                ),
                "execution_policy": execution_policy.value,
                "collector_argv": collector_argv,
                "collector_executable_binding": collector_executable_binding,
                "collector_environment": collector_environment,
                "oracle_argv": oracle_argv,
                "oracle_executable_binding": oracle_executable_binding,
                "oracle_launch_executable_binding": oracle_launch_executable_binding,
                "oracle_containment": oracle_containment,
                "oracle_network_contained": oracle_network_contained,
                "oracle_systemd_scope_unit": oracle_systemd_scope_unit,
                "expected_artifact_kinds": kinds,
                "expected_overhead": overhead,
                "containment": containment,
                "network_contained": network_contained,
                "systemd_scope_unit": systemd_scope_unit,
                "permissions": adapter_binding.permissions,
                "preflight": preflight,
                "writable_roots": writable_roots,
                "external_context": external_context,
                "planned_execution_identity": planned_execution_identity,
                "adapter_capability": adapter_capability,
                "bound_identities": identities,
                "limits": {
                    "timeout_seconds": instance.command.timeout_seconds,
                    "max_output_bytes": self.workspace.config.execution.max_output_bytes,
                    "max_artifact_bytes": self.workspace.config.capture.max_artifact_bytes,
                    "max_cpu_percent": self.workspace.config.execution.max_cpu_percent,
                    "max_memory_bytes": self.workspace.config.execution.max_memory_bytes,
                    "max_processes": self.workspace.config.execution.max_processes,
                    "minimum_free_bytes": self.workspace.config.storage.min_free_bytes,
                    "resource_sampling_interval_ms": (
                        self.workspace.config.execution.resource_sampling_interval_ms
                    ),
                    "max_resource_observed_files": (
                        self.workspace.config.execution.max_resource_observed_files
                    ),
                },
                "warnings": warnings,
                "limitation_details": limitation_details,
                "created_at": created_at,
                "expires_at": created_at + timedelta(seconds=self.plans.ttl_seconds),
            }
        )
        await self.plans.issue(plan)
        return plan

    async def execute(
        self,
        plan_token: str,
        *,
        progress: Callable[[float, float, str], Awaitable[None]] | None = None,
    ) -> CaptureResult:
        logger = OperationLogger(self.workspace.paths.root)
        operation_id = logger.new_id()
        started = time.monotonic()

        plan = await self.plans.consume(plan_token)
        await self._recheck(plan)
        StorageQuota(self.workspace).require_capacity(staging=True)
        output_root = self.workspace.paths.staging / "captures" / plan.plan_id
        output_root.mkdir(parents=True, exist_ok=False)
        for binding in plan.writable_roots:
            Path(binding.storage_path).mkdir(parents=True, exist_ok=False)
        if plan.adapter in {"triton.compiler", "cute.compiler"}:
            self._create_kernel_build_dump_dir(plan, output_root)
        try:
            staging_ownership = StagingOwnershipService(self.workspace).acquire(
                output_root,
                owner_kind="capture",
                owner_id=plan.run_id,
            )
        except BaseException:
            shutil.rmtree(output_root, ignore_errors=True)
            raise
        capture = _CaptureExecution(
            service=self,
            plan=plan,
            output_root=output_root,
            logger=logger,
            operation_id=operation_id,
            started_monotonic=started,
            progress=ProgressReporter(progress),
            staging_ownership=staging_ownership,
        )
        startup_lease: CaptureLease | None = None
        startup_lease_error: DomainError | None = None
        try:
            startup_lease = self._lease(os.getpid())
        except DomainError as error:
            startup_lease_error = error
        identity_requirements = (
            WorkloadService(self.workspace)
            .load()
            .workloads[plan.workload_name]
            .identity.environment.required
        )
        planned_accelerator = (
            AcceleratorIdentityFacet(
                provider=(
                    "metal"
                    if any(
                        item.startswith("metal.") or item == "macos.build"
                        for item in identity_requirements
                    )
                    else "cuda"
                ),
                status=AcceleratorIdentityStatus.UNKNOWN,
                identity_quality=IdentityQuality.PARTIAL,
                missing_fields=identity_requirements,
                limitations=("Declared accelerator identity has not been observed yet.",),
            )
            if identity_requirements
            else None
        )
        environment = collect_environment(planned_accelerator)
        run_id = plan.run_id
        initial = ExecutionRunManifest(
            run_id=plan.run_id,
            execution_status=ExecutionStatus.PLANNED,
            capture_status=CaptureStatus.PENDING,
            validation_status=ValidationStatus.NOT_REQUESTED,
            workload_definition_id=plan.workload_definition_id,
            workload_instance_id=plan.workload_instance.workload_instance_id,
            measurement_protocol_id=digest_model(
                {
                    "run_semantic_id": plan.semantics.semantic_id,
                    "collector_executable_identity": plan.bound_identities.get(
                        "collector_executable"
                    ),
                    "expected_artifact_kinds": plan.expected_artifact_kinds,
                    "expected_overhead": plan.expected_overhead,
                    "permissions": plan.permissions,
                    "warnings": plan.warnings,
                }
            ),
            environment_id=environment.environment_id,
            source_state_id=None,
            semantics=plan.semantics,
            command=plan.workload_instance.command,
            preflight=plan.preflight,
            writable_roots=plan.writable_roots,
            external_context=plan.external_context,
            execution_identity=plan.planned_execution_identity,
            lease=startup_lease,
            limitations=_limitation_projection(plan.limitation_details),
            limitation_details=plan.limitation_details,
        )
        self.runs.create(initial)
        capture.run = initial
        if startup_lease_error is not None:
            terminal = await capture.terminate_startup_failure(
                message=startup_lease_error.message,
                error_code=startup_lease_error.code.value,
            )
            if terminal is not None:
                startup_lease_error.run_id = terminal.run_id
            raise startup_lease_error
        try:
            accelerator = await AcceleratorIdentityService(self.workspace).observe(
                identity_requirements
            )
            environment = collect_environment(accelerator)
            await capture.report(1, "Capture plan validated")
            await capture.report(2, "Run lifecycle initialized")
            source_state = await collect_source_state(
                self.workspace,
                workload_executable=plan.workload_instance.command.argv[0],
                broker=self.broker,
            )
            execution_identity = await ExecutionIdentityService(
                self.workspace,
                broker=self.broker,
            ).observe(
                plan.workload_name,
                parameters=cast(
                    dict[str, str | int | float | bool],
                    plan.workload_instance.parameters,
                ),
                dynamic_parameters=plan.dynamic_parameters,
            )
            await capture.report(3, "Source and environment identity collected")
        except asyncio.CancelledError:
            await capture.terminate_cancelled(
                message="Capture cancelled while collecting source identity.",
                phase="capture cancelled during source identity",
                cleanup_complete=True,
            )
            raise
        except DomainError as error:
            terminal = await capture.terminate_startup_failure(
                message=error.message,
                error_code=error.code.value,
            )
            if terminal is not None:
                error.run_id = terminal.run_id
            raise
        except Exception:
            await capture.terminate_startup_failure(
                message="Capture startup identity collection failed unexpectedly.",
                error_code=ErrorCode.INTERNAL_ERROR.value,
            )
            raise
        prepared = initial.validated_copy(
            update={
                "revision": 1,
                "environment_id": environment.environment_id,
                "source_state_id": source_state.source_state_id,
                "execution_identity": execution_identity,
            }
        )
        prepared = self.runs.append(prepared, expected_revision=0)
        capture.run = prepared
        admission: CaptureAdmission | None = None
        collector_limitation_details: list[LimitationDetail] = []

        try:
            admission = await self.plans.acquire_capture_slot(run_id)
            running = prepared.validated_copy(
                update={
                    "revision": 2,
                    "started_at": utc_now(),
                    "execution_status": ExecutionStatus.RUNNING,
                    "capture_status": CaptureStatus.RUNNING,
                }
            )
            running = self.runs.append(running, expected_revision=1)
            capture.run = running
            await capture.report(4, "Capture slot acquired")
            collector_argv = plan.collector_argv
            if plan.adapter == "compute-sanitizer":
                selected = compute_sanitizer_options(cast(dict[str, object], plan.adapter_options))
                suppression = read_compute_sanitizer_suppression(
                    selected,
                    project_root=self.workspace.project_root,
                )
                if suppression is not None:
                    staged_suppression = output_root / "inputs" / "compute-sanitizer.supp"
                    atomic_write_bytes(staged_suppression, suppression, mode=0o400)
                    collector_argv = replace_compute_sanitizer_suppression(
                        collector_argv,
                        staged_suppression,
                        workload_argv=plan.workload_instance.command.argv,
                    )
            outcome = await admission.run(
                self.broker.run(
                    ExecutionRequest(
                        argv=collector_argv,
                        executable_binding=plan.collector_executable_binding,
                        cwd=Path(plan.workload_instance.command.cwd),
                        environment_allowlist=(
                            self.workspace.config.execution.child_environment_allowlist
                        ),
                        environment_overrides=(
                            {
                                **plan.workload_instance.command.env_overrides,
                                **plan.collector_environment,
                            }
                        ),
                        allowed_working_roots=self._allowed_roots(),
                        timeout_seconds=plan.workload_instance.command.timeout_seconds,
                        max_output_bytes=self.workspace.config.execution.max_output_bytes,
                        systemd_scope_unit=plan.systemd_scope_unit,
                        resource_policy=ResourcePolicy(
                            filesystem_path=self.workspace.paths.root,
                            staging_root=output_root,
                            writable_roots=(
                                *(Path(item.storage_path) for item in plan.writable_roots),
                            ),
                            minimum_free_bytes=cast(
                                int,
                                plan.limits["minimum_free_bytes"],
                            ),
                            sampling_interval_ms=cast(
                                int,
                                plan.limits["resource_sampling_interval_ms"],
                            ),
                            max_observed_files=cast(
                                int,
                                plan.limits["max_resource_observed_files"],
                            ),
                        ),
                    ),
                    on_started=capture.record_lease,
                    on_cleanup=capture.record_cleanup,
                )
            )
            capture.cleanup_complete = outcome.process.cleanup_complete
            StorageQuota(self.workspace).require_capacity(staging=True)
            await capture.report(5, "Collector execution complete")
        except asyncio.CancelledError:
            await capture.terminate_cancelled(
                message="Capture cancelled by caller after bounded cleanup.",
                phase="capture cancelled during collector execution",
                cleanup_complete=(
                    capture.cleanup_complete if capture.cleanup_complete is not None else True
                ),
            )
            raise
        except DomainError as error:
            status = (
                ExecutionStatus.TIMED_OUT
                if error.code is ErrorCode.PROCESS_TIMEOUT
                else ExecutionStatus.FAILED
            )
            partial_process = error.process if isinstance(error, ProcessExecutionError) else None
            process_observations = tuple(
                ProcessObservation.model_validate(item)
                for item in error.details.get("process_observations", ())
            )
            if partial_process is None:
                partial_process = ProcessResult(
                    cancellation_cause=(
                        ProcessCancellationCause.TIMEOUT
                        if status is ExecutionStatus.TIMED_OUT
                        else ProcessCancellationCause.PROCESS_ERROR
                    ),
                    cleanup_complete=True,
                )
            native_paths = self._native_output_paths(plan, output_root)
            valid_native_paths = self._valid_native_output_paths(plan, output_root)
            if (
                status is ExecutionStatus.TIMED_OUT
                and partial_process is not None
                and valid_native_paths
            ):
                outcome = ExecutionOutcome(
                    process=partial_process,
                    stdout=(partial_process.stdout or "").encode(),
                    stderr=(partial_process.stderr or "").encode(),
                    resolved_executable=Path(plan.collector_argv[0]).resolve(),
                    executable_binding=plan.collector_executable_binding,
                    containment=(
                        ProcessContainment.SYSTEMD_SCOPE
                        if plan.systemd_scope_unit is not None
                        else (
                            ProcessContainment.PROCESS_GROUP
                            if os.name == "posix"
                            else ProcessContainment.PROCESS
                        )
                    ),
                    process_observations=process_observations,
                )
                collector_limitation_details.append(
                    _limitation(
                        LimitationSource.COLLECTOR,
                        "timeout_partial_artifact",
                        "Collector timed out; the non-empty native output is partial evidence.",
                    )
                )
                await capture.report(5, "Collector timed out - preserving partial evidence")
            else:
                quarantined = await run_atomic_thread(
                    lambda: self._quarantine_native_output(
                        plan,
                        output_root,
                        reason=(
                            "Collector timed out without a non-empty regular output."
                            if status is ExecutionStatus.TIMED_OUT
                            else "Collector failed; native output is not a completed profile."
                        ),
                    )
                )
                if quarantined is not None:
                    collector_limitation_details.append(quarantined)
                early_artifacts: list[ArtifactRegistration] = []
                torch_diagnostics = output_root / "torch-profiler-diagnostics.json"
                if plan.adapter == "torch.profiler" and torch_diagnostics.is_file():
                    try:
                        # Bound diagnostic read to prevent memory exhaustion.
                        max_diagnostics_bytes = 1 * 1024 * 1024  # 1 MiB
                        if torch_diagnostics.stat().st_size > max_diagnostics_bytes:
                            collector_limitation_details.append(
                                _limitation(
                                    LimitationSource.COLLECTOR,
                                    "diagnostics_oversized",
                                    "Torch profiler diagnostics exceed 1 MiB; skipping parse.",
                                )
                            )
                            diagnostic_phase = None
                            diagnostic_status = None
                        else:
                            diagnostic_payload = json.loads(
                                torch_diagnostics.read_text(encoding="utf-8")
                            )
                            diagnostic_phase = diagnostic_payload.get("phase")
                            diagnostic_status = diagnostic_payload.get("status")
                        if diagnostic_status == "failed" and isinstance(diagnostic_phase, str):
                            collector_limitation_details.append(
                                _limitation(
                                    LimitationSource.COLLECTOR,
                                    "failure_phase",
                                    f"Torch profiler collector failed during {diagnostic_phase}.",
                                )
                            )
                        diagnostic_registration = await self._register_path_async(
                            run_id,
                            torch_diagnostics,
                            kind=ArtifactKind.COLLECTOR_METADATA,
                            role="torch_profiler_diagnostics",
                            media_type="application/json",
                            producer=plan.adapter,
                            producer_version=plan.adapter_version,
                        )
                        early_artifacts.append(diagnostic_registration[0])
                    except (
                        DomainError,
                        OSError,
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                    ) as diagnostic_error:
                        collector_limitation_details.append(
                            _limitation(
                                LimitationSource.COLLECTOR,
                                "diagnostics_registration_failed",
                                "Torch profiler diagnostics could not be registered: "
                                f"{diagnostic_error}",
                            )
                        )
                terminal = await capture.terminate(
                    execution=status,
                    message=error.message,
                    phase="collector execution failed",
                    error_code=error.code.value,
                    process=partial_process,
                    process_observations=process_observations,
                    limitation_details=tuple(
                        [
                            _limitation(
                                LimitationSource.COLLECTOR,
                                error.code.value.lower(),
                                error.message,
                            ),
                            *collector_limitation_details,
                        ]
                    ),
                    artifacts=tuple(early_artifacts),
                )
                error.run_id = terminal.run_id
                raise
        finally:
            if admission is not None:
                await run_atomic_thread(admission.release)
        running = capture.run
        if running is None:
            raise DomainError(
                ErrorCode.INTERNAL_ERROR,
                "capture run disappeared after collector execution",
            )

        sanitizer_finding_exit = (
            plan.adapter == "compute-sanitizer"
            and outcome.process.exit_code == plan.adapter_options.get("finding_exit_code")
        )
        sanitizer_finding = False
        collector_succeeded = outcome.process.exit_code == 0 or sanitizer_finding_exit
        native_paths = self._native_output_paths(plan, output_root)
        valid_native_paths = self._valid_native_output_paths(plan, output_root)
        unexpected_native_paths = self._unexpected_native_output_paths(plan, output_root)
        native_complete = bool(native_paths) and (
            len(valid_native_paths) == len(native_paths)
            and not unexpected_native_paths
            and self._native_output_manifest_is_valid(plan, output_root)
        )
        native_definition = builtin_adapter(plan.adapter)
        preserve_nonzero_artifact = bool(
            native_definition is not None and native_definition.preserve_artifact_on_nonzero
        )
        if not collector_succeeded and not outcome.process.timed_out:
            collector_limitation_details.append(
                _limitation(
                    LimitationSource.COLLECTOR,
                    "nonzero_exit",
                    f"Collector exited with status {outcome.process.exit_code}.",
                )
            )
        if (
            native_paths
            and not collector_succeeded
            and not outcome.process.timed_out
            and not preserve_nonzero_artifact
        ):
            quarantined = await run_atomic_thread(
                lambda: self._quarantine_native_output(
                    plan,
                    output_root,
                    reason=(
                        "Collector exited unsuccessfully; native output is not a completed profile."
                    ),
                )
            )
            if quarantined is not None:
                collector_limitation_details.append(quarantined)
            valid_native_paths = ()
            native_complete = False
        elif (
            native_paths
            and not native_complete
            and not outcome.process.timed_out
            and not preserve_nonzero_artifact
        ):
            reason = (
                "Collector completed without every expected native output."
                if not any(path.exists() for path in native_paths)
                else "Collector emitted an incomplete, extra, empty, or non-regular output set."
            )
            quarantined = await run_atomic_thread(
                lambda: self._quarantine_native_output(plan, output_root, reason=reason)
            )
            if quarantined is not None:
                collector_limitation_details.append(quarantined)
            collector_limitation_details.append(
                _limitation(LimitationSource.ARTIFACT, "expected_output_invalid", reason)
            )
            valid_native_paths = ()
        elif (
            native_paths
            and not native_complete
            and not outcome.process.timed_out
            and preserve_nonzero_artifact
        ):
            # Provider configured to preserve partial artifacts:
            # keep nonempty invalid/partial native output as partial evidence
            # rather than quarantining.  A nonzero limitation is already recorded.
            reason = (
                "Collector completed without every expected native output."
                if not any(path.exists() for path in native_paths)
                else "Collector emitted an incomplete, extra, empty, or non-regular output set."
            )
            collector_limitation_details.append(
                _limitation(LimitationSource.ARTIFACT, "expected_output_invalid", reason)
            )
            collector_succeeded = False
        if native_paths and outcome.process.timed_out and valid_native_paths:
            collector_limitation_details.append(
                _limitation(
                    LimitationSource.COLLECTOR,
                    "timeout_partial_artifact",
                    "Collector timed out; the non-empty native output is partial evidence.",
                )
            )

        registrations: list[tuple[ArtifactRegistration, int]] = []
        kernel_build_manifest: KernelBuildManifestV2 | None = None
        process_snapshot_rows: list[dict[str, object]] = []
        process_snapshot_entries: list[dict[str, object]] = []
        adapter_validation_rows: list[dict[str, object]] = []
        adapter_extraction_rows: list[dict[str, object]] = []
        validation_status = ValidationStatus.NOT_REQUESTED
        validation_limitations: list[str] = []
        oracle_receipt_record: OracleReceiptRecord | None = None
        try:
            snapshot_path = output_root / "process-snapshot.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "observations": [
                            item.model_dump(mode="json") for item in outcome.process_observations
                        ],
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            snapshot_artifact = self.artifacts.import_path(
                snapshot_path,
                allowed_roots=(output_root,),
                max_bytes=self.workspace.config.capture.max_artifact_bytes,
            )
            snapshot_registration = ArtifactRegistration(
                registration_id=new_id(),
                run_id=run_id,
                artifact_id=snapshot_artifact.content.artifact_id,
                display_name="process-snapshot.json",
                media_type="application/json",
                kind=ArtifactKind.PROCESS_TREE_SNAPSHOT,
                role="process_observation",
                producer="flameox.execution",
                producer_version="1",
                sensitivity=Sensitivity.INTERNAL,
            )
            process_snapshot_rows, process_snapshot_entries = process_observation_rows(
                run_id,
                outcome.process_observations,
                artifact_id=snapshot_artifact.content.artifact_id,
            )
            oracle = self.workloads.resolve_oracle(
                plan.workload_name,
                cast(dict[str, Scalar], plan.workload_instance.parameters),
                dynamic_parameters=plan.dynamic_parameters,
            )
            if oracle is not None and outcome.process.exit_code == 0:
                try:
                    if plan.oracle_argv is None or plan.oracle_launch_executable_binding is None:
                        raise DomainError(
                            ErrorCode.INVALID_CAPTURE_PLAN,
                            "The capture plan is missing validation-oracle execution authority.",
                        )
                    validation = await self.broker.run(
                        ExecutionRequest(
                            argv=plan.oracle_argv,
                            executable_binding=plan.oracle_launch_executable_binding,
                            cwd=Path(oracle.command.cwd),
                            environment_allowlist=(
                                self.workspace.config.execution.child_environment_allowlist
                            ),
                            environment_overrides=(
                                {"FLAMEOX_ORACLE_RECEIPT": str(output_root / "oracle-receipt.json")}
                                if oracle.receipt_schema is not None
                                else {}
                            ),
                            allowed_working_roots=self._allowed_roots(),
                            timeout_seconds=oracle.command.timeout_seconds,
                            max_output_bytes=(self.workspace.config.execution.max_output_bytes),
                            systemd_scope_unit=plan.oracle_systemd_scope_unit,
                            resource_policy=ResourcePolicy(
                                filesystem_path=self.workspace.paths.root,
                                staging_root=output_root,
                                writable_roots=(
                                    *(Path(item.storage_path) for item in plan.writable_roots),
                                ),
                                minimum_free_bytes=cast(
                                    int,
                                    plan.limits["minimum_free_bytes"],
                                ),
                                sampling_interval_ms=cast(
                                    int,
                                    plan.limits["resource_sampling_interval_ms"],
                                ),
                                max_observed_files=cast(
                                    int,
                                    plan.limits["max_resource_observed_files"],
                                ),
                            ),
                        ),
                        on_cleanup=capture.record_cleanup,
                    )
                    validation_status = ValidationStatus.FAILED
                    validation_output = output_root / "validation.stdout"
                    atomic_write_bytes(validation_output, validation.stdout)
                    role = (
                        "validation_cross_treatment_equivalence"
                        if oracle.strength is OracleStrength.CROSS_TREATMENT_EQUIVALENCE
                        else f"validation_{oracle.strength.value}"
                    )
                    stdout_registration = await self._register_path_async(
                        run_id,
                        validation_output,
                        kind=ArtifactKind.VALIDATION_OUTPUT,
                        role=role,
                        media_type="application/octet-stream",
                    )
                    registrations.append(stdout_registration)
                    stderr_artifact_id: str | None = None
                    if validation.stderr:
                        validation_stderr = output_root / "validation.stderr"
                        atomic_write_bytes(validation_stderr, validation.stderr)
                        stderr_registration = await self._register_path_async(
                            run_id,
                            validation_stderr,
                            kind=ArtifactKind.PROCESS_OUTPUT,
                            role="validation_stderr",
                            media_type="application/octet-stream",
                        )
                        registrations.append(stderr_registration)
                        stderr_artifact_id = stderr_registration[0].artifact_id
                    if oracle.receipt_schema is None:
                        validation_status = (
                            ValidationStatus.PASSED
                            if validation.process.exit_code == 0
                            else ValidationStatus.FAILED
                        )
                    else:
                        receipt_path = output_root / "oracle-receipt.json"
                        receipt_registration: tuple[ArtifactRegistration, int] | None = None
                        try:
                            receipt_registration = await self._register_path_async(
                                run_id,
                                receipt_path,
                                kind=ArtifactKind.VALIDATION_OUTPUT,
                                role="validation_receipt",
                                media_type="application/json",
                                producer=plan.workload_name,
                                producer_version=oracle.receipt_schema,
                            )
                            registrations.append(receipt_registration)
                            authoritative_receipt = self.artifacts.get(
                                receipt_registration[0].artifact_id
                            )
                            receipt = parse_oracle_receipt(
                                authoritative_receipt.payload_path.read_bytes()
                            )
                            oracle_receipt_record = OracleReceiptRecord(
                                receipt=receipt,
                                receipt_artifact_id=receipt_registration[0].artifact_id,
                                validation_stdout_artifact_id=stdout_registration[0].artifact_id,
                                validation_stderr_artifact_id=stderr_artifact_id,
                            )
                            if validation.process.exit_code == 0:
                                validation_status = {
                                    OracleStatus.PASS: ValidationStatus.PASSED,
                                    OracleStatus.FAIL: ValidationStatus.FAILED,
                                    OracleStatus.INCONCLUSIVE: ValidationStatus.INCONCLUSIVE,
                                    OracleStatus.UNSUPPORTED: ValidationStatus.UNSUPPORTED,
                                }[receipt.status]
                            elif receipt.status is OracleStatus.PASS:
                                validation_limitations.append(
                                    "The oracle process failed despite claiming a passing receipt."
                                )
                        except DomainError as error:
                            validation_status = (
                                ValidationStatus.FAILED
                                if validation.process.exit_code != 0
                                else ValidationStatus.ERROR
                            )
                            validation_limitations.append(
                                f"Oracle receipt validation failed: {error.message}"
                            )
                    if validation.process.exit_code != 0:
                        validation_limitations.append(
                            "The declared validation oracle exited unsuccessfully."
                        )
                    elif validation_status is ValidationStatus.FAILED:
                        validation_limitations.append(
                            "The declared validation oracle reported a semantic failure."
                        )
                except DomainError as error:
                    validation_status = ValidationStatus.ERROR
                    validation_limitations.append(error.message)
            await capture.report(6, "Validation complete")
            torch_diagnostics = output_root / "torch-profiler-diagnostics.json"
            if plan.adapter == "torch.profiler" and torch_diagnostics.is_file():
                try:
                    # Bound diagnostic read to prevent memory exhaustion from
                    # a malicious or corrupted multi-gigabyte diagnostics file.
                    max_diagnostics_bytes = 1 * 1024 * 1024  # 1 MiB
                    if torch_diagnostics.stat().st_size > max_diagnostics_bytes:
                        collector_limitation_details.append(
                            _limitation(
                                LimitationSource.COLLECTOR,
                                "diagnostics_oversized",
                                "Torch profiler diagnostics exceed 1 MiB; skipping parse.",
                            )
                        )
                        diagnostic_phase = None
                        diagnostic_status = None
                    else:
                        diagnostic_payload = json.loads(
                            torch_diagnostics.read_text(encoding="utf-8")
                        )
                        diagnostic_phase = diagnostic_payload.get("phase")
                        diagnostic_status = diagnostic_payload.get("status")
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    diagnostic_phase = None
                    diagnostic_status = None
                if diagnostic_status == "failed" and isinstance(diagnostic_phase, str):
                    collector_limitation_details.append(
                        _limitation(
                            LimitationSource.COLLECTOR,
                            "failure_phase",
                            f"Torch profiler collector failed during {diagnostic_phase}.",
                        )
                    )
                registrations.append(
                    await self._register_path_async(
                        run_id,
                        torch_diagnostics,
                        kind=ArtifactKind.COLLECTOR_METADATA,
                        role="torch_profiler_diagnostics",
                        media_type="application/json",
                        producer=plan.adapter,
                        producer_version=plan.adapter_version,
                    )
                )
            for name, payload, kind, role, media_type in (
                (
                    "stdout.bin",
                    outcome.stdout,
                    ArtifactKind.PROCESS_OUTPUT,
                    "stdout",
                    "application/octet-stream",
                ),
                (
                    "stderr.bin",
                    outcome.stderr,
                    ArtifactKind.PROCESS_OUTPUT,
                    "stderr",
                    "application/octet-stream",
                ),
            ):
                if not payload and not (
                    plan.adapter == "torch.profiler" and not collector_succeeded
                ):
                    continue
                path = output_root / name
                atomic_write_bytes(path, payload)
                registrations.append(
                    await self._register_path_async(
                        run_id,
                        path,
                        kind=kind,
                        role=role,
                        media_type=media_type,
                    )
                )
            if plan.adapter in {"triton.compiler", "cute.compiler"}:
                (
                    kernel_build_registrations,
                    kernel_build_manifest,
                ) = await self._collect_kernel_build(
                    plan,
                    output_root,
                    run_id,
                    outcome.process.exit_code if outcome.process.exit_code is not None else 1,
                )
                registrations.extend(kernel_build_registrations)
            elif plan.adapter_execution_plan is not None and collector_succeeded:
                (
                    plugin_registrations,
                    plugin_validations,
                    plugin_extractions,
                    plugin_limitations,
                ) = await self._process_adapter_artifacts(plan, output_root)
                registrations.extend(plugin_registrations)
                adapter_validation_rows.extend(plugin_validations)
                adapter_extraction_rows.extend(plugin_extractions)
                validation_limitations.extend(plugin_limitations)
            elif plan.adapter_execution_plan is not None:
                for declaration in AdapterExecutionPlan.model_validate(
                    plan.adapter_execution_plan
                ).artifacts:
                    candidate = output_root / declaration.relative_path
                    if candidate.exists():
                        try:
                            quarantine_id = await run_atomic_thread(
                                partial(
                                    self._quarantine_adapter_output,
                                    candidate,
                                    expected_format=declaration.media_type,
                                    run_id=plan.run_id,
                                    adapter=plan.adapter,
                                )
                            )
                        except DomainError as error:
                            collector_limitation_details.append(
                                _limitation(
                                    LimitationSource.ARTIFACT,
                                    "adapter_output_quarantine_failed",
                                    error.message,
                                )
                            )
                        else:
                            collector_limitation_details.append(
                                _limitation(
                                    LimitationSource.ARTIFACT,
                                    "adapter_output_quarantined",
                                    f"Declared adapter output was quarantined ({quarantine_id}).",
                                )
                            )
            else:
                for native in valid_native_paths:
                    cycle_index = native_paths.index(native)
                    kind = plan.expected_artifact_kinds[0]
                    producer = plan.adapter
                    producer_version = plan.adapter_version
                    media_type = mimetypes.guess_type(native.name)[0] or "application/octet-stream"
                    if plan.adapter == "python-startup":
                        is_wall = native.name == PYTHON_STARTUP_PROFILE.wall_output_name
                        kind = (
                            ArtifactKind.BENCHMARK_SAMPLES
                            if is_wall
                            else ArtifactKind.PYTHON_STARTUP
                        )
                        role = "startup_wall" if is_wall else "import_trace"
                        producer = "pyperf" if is_wall else "cpython"
                        producer_version = plan.adapter_version if is_wall else None
                        media_type = "application/json" if is_wall else "text/plain"
                        if outcome.process.timed_out:
                            role = f"partial_{role}"
                    elif outcome.process.timed_out:
                        role = (
                            f"partial_cycle_{cycle_index:04d}"
                            if len(native_paths) > 1
                            else "partial"
                        )
                    elif len(native_paths) > 1:
                        role = f"cycle_{cycle_index:04d}"
                    else:
                        role = "primary"
                    registration = await self._register_path_async(
                        run_id,
                        native,
                        kind=kind,
                        role=role,
                        media_type=media_type,
                        producer=producer,
                        producer_version=producer_version,
                    )
                    registrations.append(registration)
                    if plan.adapter == "compute-sanitizer":
                        immutable = self.artifacts.get(registration[0].artifact_id)
                        try:
                            inspection = await run_atomic_thread(
                                partial(
                                    inspect_compute_sanitizer_report,
                                    self.workspace,
                                    str(immutable.payload_path),
                                    max_records=max(
                                        1,
                                        min(
                                            10_000,
                                            self.workspace.config.storage.max_rows_per_generation
                                            - 1,
                                        ),
                                    ),
                                )
                            )
                        except DomainError as error:
                            collector_succeeded = False
                            validation_status = ValidationStatus.ERROR
                            validation_limitations.append(
                                f"Compute Sanitizer report validation failed: {error.message}"
                            )
                        else:
                            compatibility_limitations = compute_sanitizer_compatibility_limitations(
                                plan.adapter_version
                            )
                            inspection = inspection.validated_copy(
                                update={
                                    "limitations": tuple(
                                        dict.fromkeys(
                                            (
                                                *compatibility_limitations,
                                                *inspection.limitations,
                                            )
                                        )
                                    )
                                }
                            )
                            has_findings = bool(inspection.records)
                            supported_exit = outcome.process.exit_code in {
                                0,
                                plan.adapter_options.get("finding_exit_code"),
                            }
                            sanitizer_finding = has_findings and supported_exit
                            if not supported_exit or (sanitizer_finding_exit and not has_findings):
                                collector_succeeded = False
                                validation_status = ValidationStatus.ERROR
                                validation_limitations.append(
                                    "Compute Sanitizer exit status contradicts the parsed report."
                                )
                            else:
                                validation_status = (
                                    ValidationStatus.FAILED
                                    if has_findings
                                    else (
                                        ValidationStatus.INCONCLUSIVE
                                        if inspection.limitations
                                        else ValidationStatus.PASSED
                                    )
                                )
                            validation_limitations.extend(inspection.limitations)
                manifest = output_root / "torch-profiler-cycles.json"
                if (
                    plan.adapter == "torch.profiler"
                    and torch_profiler_options(cast(dict[str, object], plan.adapter_options)).mode
                    == "sdk"
                    and native_complete
                ):
                    registrations.append(
                        await self._register_path_async(
                            run_id,
                            manifest,
                            kind=ArtifactKind.COLLECTOR_METADATA,
                            role="torch_profiler_cycle_manifest",
                            media_type="application/json",
                            producer=plan.adapter,
                            producer_version=plan.adapter_version,
                        )
                    )
                if plan.adapter == "nvbench" and valid_native_paths:
                    primary_registration = next(
                        registered
                        for registered in registrations
                        if registered[0].kind is ArtifactKind.BENCHMARK_SAMPLES
                        and registered[0].role == "primary"
                    )
                    sidecar_regs = await self._register_nvbench_sidecars(
                        plan,
                        output_root,
                        run_id,
                        primary_registration=primary_registration,
                        partial=not native_complete or not collector_succeeded,
                    )
                    registrations.extend(sidecar_regs)
            observations = output_root / "observations.jsonl"
            if observations.is_file():
                registrations.append(
                    await self._register_path_async(
                        run_id,
                        observations,
                        kind=ArtifactKind.SEMANTIC_OBSERVATIONS,
                        role="semantic_observations",
                        media_type="application/x-ndjson",
                    )
                )
            registrations.append((snapshot_registration, snapshot_artifact.content.byte_length))
            if oracle_receipt_record is not None:
                by_role = {
                    registration.role: registration.artifact_id for registration, _ in registrations
                }
                missing_roles = tuple(
                    role
                    for role in oracle_receipt_record.receipt.diagnostic_roles
                    if role not in by_role
                )
                oracle_receipt_record = oracle_receipt_record.validated_copy(
                    update={
                        "diagnostic_artifact_ids": tuple(
                            by_role[role]
                            for role in oracle_receipt_record.receipt.diagnostic_roles
                            if role in by_role
                        ),
                        "parsing_limitations": (
                            (
                                "Some diagnostic roles were not registered on this run: "
                                + ", ".join(missing_roles),
                            )
                            if missing_roles
                            else ()
                        ),
                    }
                )
            await capture.report(7, "Artifacts registered")
        except asyncio.CancelledError:
            await capture.terminate_cancelled(
                message="Capture cancelled during validation or artifact registration.",
                phase="capture cancelled during validation or registration",
            )
            raise
        except DomainError as error:
            quarantined = await run_atomic_thread(
                lambda: self._quarantine_native_output(
                    plan,
                    output_root,
                    reason=(
                        "Artifact validation or registration failed; native output was not "
                        "published."
                    ),
                )
            )
            details = [
                _limitation(LimitationSource.ARTIFACT, error.code.value.lower(), error.message),
            ]
            if quarantined is not None:
                details.append(quarantined)
            terminal = await capture.terminate(
                execution=ExecutionStatus.FAILED,
                message=error.message,
                phase="validation or artifact registration failed",
                error_code=error.code.value,
                limitation_details=tuple(details),
            )
            error.run_id = terminal.run_id
            raise
        succeeded = collector_succeeded and (not sanitizer_finding or native_complete)
        timed_out = outcome.process.timed_out
        detail_groups = [
            tuple(collector_limitation_details),
            tuple(
                _limitation(LimitationSource.VALIDATION, "validation.limitation", message)
                for message in validation_limitations
            ),
            tuple(
                _limitation(
                    LimitationSource.RESOURCE,
                    "resource_metric_unavailable",
                    f"Runtime resource metric {metric!r} was unavailable.",
                )
                for metric in (
                    outcome.process.resources.unavailable_metrics
                    if outcome.process.resources is not None
                    else ("resource_summary",)
                )
            ),
            (
                (
                    _limitation(
                        LimitationSource.RESOURCE,
                        "storage_reserve_exceeded",
                        "Runtime storage reserve terminated the collector.",
                    ),
                )
                if outcome.process.resources is not None
                and outcome.process.resources.policy_termination is not None
                else ()
            ),
        ]
        terminal_limitation_details = _merge_limitation_details(
            running.limitation_details,
            *detail_groups,
        )
        terminal = running.validated_copy(
            update={
                "revision": running.revision + 1,
                "finished_at": utc_now(),
                "execution_status": (
                    ExecutionStatus.TIMED_OUT
                    if timed_out
                    else (ExecutionStatus.SUCCEEDED if succeeded else ExecutionStatus.FAILED)
                ),
                "capture_status": (
                    CaptureStatus.REGISTERED
                    if (
                        (succeeded and (not native_paths or native_complete))
                        or (preserve_nonzero_artifact and bool(valid_native_paths))
                        or (timed_out and bool(valid_native_paths))
                        or (
                            plan.adapter in {"triton.compiler", "cute.compiler"}
                            and any(
                                reg.kind is ArtifactKind.KERNEL_BUILD for reg, _ in registrations
                            )
                        )
                    )
                    else CaptureStatus.FAILED
                ),
                "validation_status": validation_status,
                "process": outcome.process,
                "artifacts": tuple(registration for registration, _ in registrations),
                "oracle_receipt": oracle_receipt_record,
                "limitations": _limitation_projection(
                    terminal_limitation_details,
                    tuple(
                        list(running.limitations)
                        + (
                            []
                            if succeeded
                            else (
                                ["Collector timed out; registered artifacts may be partial."]
                                if timed_out
                                else [f"Collector exited with status {outcome.process.exit_code}."]
                            )
                        )
                        + validation_limitations
                    ),
                ),
                "limitation_details": terminal_limitation_details,
            }
        )
        terminal_run = cast(RunManifest, terminal)
        try:
            projected = await run_atomic_thread(
                lambda: self.projections.append_run(
                    terminal_run,
                    expected_revision=running.revision,
                    environment=environment,
                    source_state=source_state,
                )
            )
        except BaseException:
            capture.cleanup_staging()
            raise
        terminal = projected.run
        capture.run = terminal
        if kernel_build_manifest is not None:
            registration_ids_by_path = {
                registration.display_name: registration.registration_id
                for registration in terminal.artifacts
                if registration.kind is ArtifactKind.KERNEL_BUILD
                and registration.role != "kernel_build_manifest"
            }
            try:
                await run_atomic_thread(
                    lambda: ArtifactPipelineService(self.workspace).register_managed(
                        kernel_build_pipeline_request(
                            kernel_build_manifest,
                            run_id=terminal.run_id,
                            registration_ids_by_path=registration_ids_by_path,
                        ),
                        workload_instance=plan.workload_instance,
                    )
                )
            except DomainError as error:
                pipeline_failed = terminal.validated_copy(
                    update={
                        "revision": terminal.revision + 1,
                        "execution_status": ExecutionStatus.FAILED,
                        "limitations": (*terminal.limitations, error.message),
                    }
                )
                pipeline_failed = self.projections.append_run(
                    pipeline_failed,
                    expected_revision=terminal.revision,
                    environment=environment,
                    source_state=source_state,
                ).run
                capture.run = pipeline_failed
                error.run_id = pipeline_failed.run_id
                raise
        measurement_rows: list[dict[str, object]] = []
        if terminal.process is not None and terminal.process.wall_time_ns is not None:
            measurement_rows.append(
                {
                    "measurement_id": digest_model(
                        {
                            "run_id": run_id,
                            "name": "process.wall_time",
                            "unit": "ns",
                        }
                    ),
                    "run_id": run_id,
                    "artifact_id": None,
                    "name": "process.wall_time",
                    "value_int": terminal.process.wall_time_ns,
                    "value_float": None,
                    "unit": "ns",
                    "aggregation": "single",
                    "scope": "process",
                    "trial_id": None,
                    "worker_id": None,
                    "worker_run_index": None,
                    "value_index": None,
                    "loop_count": None,
                    "is_warmup": False,
                    "block_id": None,
                    "variant_id": None,
                    "order_in_block": None,
                    "phase": None,
                    "dimensions": {},
                    "evidence_level": "observed",
                }
            )
        publication_rows = {
            "adapter_validations": adapter_validation_rows,
            "adapter_extractions": adapter_extraction_rows,
            "measurements": measurement_rows,
            "runtime_resource_summaries": [
                runtime_resource_summary_row(
                    terminal,
                    sampling_interval_ms=cast(
                        int,
                        plan.limits["resource_sampling_interval_ms"],
                    ),
                )
            ],
            "runtime_writable_root_growth": runtime_writable_root_rows(
                terminal,
                project_root=self.workspace.project_root,
            ),
            "process_snapshots": process_snapshot_rows,
            "process_snapshot_entries": process_snapshot_entries,
        }
        try:
            published = await run_atomic_thread(
                lambda: self.publisher.publish_rows(
                    publication_rows,
                    publisher="flameox.capture.observations",
                    publisher_version="1",
                    input_run_ids=(run_id,),
                    input_artifact_ids=tuple(
                        registration.artifact_id for registration, _ in registrations
                    ),
                )
            )
        except asyncio.CancelledError:
            await capture.terminate_cancelled(
                message="Capture cancelled during atomic evidence publication.",
                phase="capture cancelled during evidence publication",
                cleanup_complete=True,
            )
            raise
        except Exception as error:
            message = (
                error.message
                if isinstance(error, DomainError)
                else f"Evidence publication failed: {type(error).__name__}."
            )
            try:
                failed = await capture.terminate(
                    execution=ExecutionStatus.FAILED,
                    message=message,
                    phase="evidence publication failed",
                    error_code=(
                        error.code.value if isinstance(error, DomainError) else "publication_failed"
                    ),
                    cleanup_complete=True,
                )
                if isinstance(error, DomainError):
                    error.run_id = failed.run_id
            except Exception as terminalization_error:
                error.add_note(
                    "Capture terminalization also failed: "
                    f"{type(terminalization_error).__name__}: {terminalization_error}"
                )
            raise
        capture.cleanup_staging()
        await capture.report(8, "Evidence publication complete")
        return CaptureResult(
            plan=plan,
            run=terminal,
            corpus_commit_id=published.commit.commit_id,
        )

    async def _adapter_capability(
        self,
        adapter: str,
        *,
        mode: PreflightMode,
        workload_name: str,
    ) -> CapabilityReport:
        choices = self._capture_adapter_choices()
        approved_third_party = self._is_approved_third_party(adapter)
        if adapter not in choices and not approved_third_party:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Unknown or non-capture adapter {adapter!r}.",
                details={
                    "allowed_adapters": list(choices),
                },
                remediation=(
                    "Choose one of the bounded adapter options returned by get_declared_workflow.",
                ),
                next_action=tool_action(
                    ActionId.GET_DECLARED_WORKFLOW,
                    kind="workload",
                    name=workload_name,
                ),
            )
        report = self.capabilities.get(adapter)
        definition = builtin_adapter(adapter)
        if (
            definition is not None
            and definition.dependency_kind is AdapterDependencyKind.PACKAGE
            and definition.dependency is not None
        ):
            requirement = Requirement(definition.managed_requirement or definition.dependency)
            environment = await PythonEnvironmentProbe(
                self.workspace,
                broker=self.broker,
            ).inspect(workload_name, (requirement,))
            observed = environment.versions[requirement.name]
            available = observed is not None and requirement.specifier.contains(
                observed,
                prereleases=True,
            )
            report = CapabilityReport(
                adapter=adapter,
                status=(CapabilityStatus.AVAILABLE if available else CapabilityStatus.UNAVAILABLE),
                provisioning=CapabilityProvisioning.WORKLOAD_ENVIRONMENT,
                version=observed,
                supported_modes=definition.supported_modes if available else (),
                supported_formats=definition.supported_formats if available else (),
                platform=report.platform,
                architecture=report.architecture,
                features=definition.features,
                limitations=(
                    "Package availability was inspected through the exact workload interpreter; "
                    "module importability was not exercised.",
                ),
                remediation=(
                    ()
                    if available
                    else (
                        f"Install {requirement!s} in the declared workload interpreter "
                        f"{environment.interpreter!s}, then plan again.",
                    )
                ),
            )
        permission_sensitive = report.permission_status in {
            CapabilityPermissionStatus.UNKNOWN_UNTIL_ACTIVE_PROBE,
            CapabilityPermissionStatus.NOT_EXERCISED,
        }
        if permission_sensitive:
            if mode is PreflightMode.PASSIVE:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    f"Adapter {adapter!r} requires an active permission probe before planning.",
                    details={"required_preflight_mode": "active"},
                    remediation=(
                        "Re-plan with preflight_mode='auto' so FlameOx can perform the bounded "
                        "active permission probe during planning.",
                    ),
                    next_action=manual_action(
                        "Supply the declared parameters and re-plan with preflight_mode='auto'.",
                        suggested_action=ActionId.PLAN_CAPTURE,
                        missing_arguments=("parameters",),
                    ),
                )
            report = await self.capabilities.probe(adapter, refresh=True)
        if (
            report.status is not CapabilityStatus.AVAILABLE
            and adapter != "command"
            and not approved_third_party
        ):
            setup_pending = report.setup is not None
            fallback_adapters = tuple(
                choice
                for choice in choices
                if choice != adapter
                and self.capabilities.get(choice).status is CapabilityStatus.AVAILABLE
            )
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Adapter {adapter!r} is unavailable for capture planning.",
                details={
                    "adapter": adapter,
                    "capability_status": report.status.value,
                    "setup_adapters": [adapter] if setup_pending else [],
                    "fallback_adapters": list(fallback_adapters),
                },
                remediation=report.remediation
                or (
                    (
                        f"Call start_capability_setup with adapters=['{adapter}'], then "
                        "call list_capabilities again.",
                    )
                    if setup_pending
                    else (
                        "Choose an available adapter option from get_declared_workflow; "
                        "the fallback changes the evidence collected.",
                    )
                ),
                next_action=(
                    report.setup.next_action
                    if report.setup is not None
                    else tool_action(ActionId.INSPECT_CAPABILITIES, adapter=adapter)
                ),
            )
        return report

    def _capture_adapter_choices(self) -> tuple[str, ...]:
        builtins = tuple(
            sorted(adapter.name for adapter in BUILTIN_ADAPTERS.values() if adapter.artifact_kinds)
        )
        approved = tuple(
            sorted(
                item.adapter
                for item in AdapterRegistry(self.workspace).discover().adapters
                if item.approved
            )
        )
        return tuple(dict.fromkeys((*builtins, *approved)))[:64]

    def _is_approved_third_party(self, adapter: str) -> bool:
        if builtin_adapter(adapter) is not None:
            return False
        try:
            AdapterRegistry(self.workspace).approved_descriptor(adapter)
        except DomainError:
            return False
        return True

    def _native_output_paths(self, plan: CapturePlan, output_root: Path) -> tuple[Path, ...]:
        definition = builtin_adapter(plan.adapter)
        if definition is None or plan.adapter == "command" or definition.output_filename is None:
            return ()
        if plan.adapter in {"triton.compiler", "cute.compiler"}:
            return ()
        if plan.adapter == "torch.profiler":
            options = torch_profiler_options(cast(dict[str, object], plan.adapter_options))
            return tuple(output_root / filename for filename in options.output_filenames)
        if plan.adapter == "python-startup":
            return (
                output_root / PYTHON_STARTUP_PROFILE.wall_output_name,
                output_root / PYTHON_STARTUP_PROFILE.import_trace_output_name,
            )
        return (output_root / definition.output_filename,)

    async def _register_nvbench_sidecars(
        self,
        plan: CapturePlan,
        output_root: Path,
        run_id: str,
        *,
        primary_registration: tuple[ArtifactRegistration, int],
        partial: bool = False,
    ) -> list[tuple[ArtifactRegistration, int]]:
        """Register provider-declared NVBench jsonbin sidecars.

        Parses the primary JSON via the shared selector
        (``load_nvbench_sidecar_specs``) to discover which sidecar files
        the document references.  Only those files are registered with
        ``role="nvbench_sidecar"`` and ``display_name`` matching the
        declared relative path.  No sibling globbing is performed.

        Every declared filename is validated as a normalized relative
        POSIX path that resolves under ``output_root``.  Each sidecar
        file must be a regular non-linked file with exact declared byte
        length before registration.

        When ``partial`` is False (successful capture), any parse error,
        unsupported hint, missing/mismatched sidecar, or path escape
        raises ``DomainError`` so that
        ``_native_output_manifest_is_valid`` returns ``False`` and the
        run fails with a bounded limitation.

        When ``partial`` is True (nonzero exit), parse failures and
        missing/mismatched sidecars are silently skipped — the missing
        sidecar evidence is a bounded limitation/proof gap, not a
        trigger for globbing.
        """
        definition = builtin_adapter(plan.adapter)
        if definition is None or definition.output_filename is None:
            return []
        json_path = output_root / definition.output_filename
        max_bytes = self.workspace.config.capture.max_artifact_bytes
        try:
            document, document_bytes, document_sha256 = load_nvbench_document_with_integrity(
                json_path,
                max_bytes=max_bytes,
            )
            self._require_registered_integrity(
                primary_registration,
                expected_byte_length=document_bytes,
                expected_sha256=document_sha256,
            )
            specs = collect_nvbench_sidecar_specs(document)
        except DomainError:
            if not partial:
                raise
            return []
        registrations: list[tuple[ArtifactRegistration, int]] = []
        for spec in specs:
            try:
                sidecar_path = resolve_nvbench_sidecar_path(spec.filename, output_root)
                validate_nvbench_sidecar_file(sidecar_path, spec.byte_length)
            except DomainError:
                if not partial:
                    raise
                continue
            registered = await self._register_path_async(
                run_id,
                sidecar_path,
                kind=plan.expected_artifact_kinds[0],
                role="nvbench_sidecar",
                media_type="application/octet-stream",
                producer=plan.adapter,
                producer_version=plan.adapter_version,
                display_name=spec.filename,
            )
            self._require_registered_integrity(
                registered,
                expected_byte_length=spec.byte_length,
            )
            registrations.append(registered)
        return registrations

    @staticmethod
    def _create_kernel_build_dump_dir(plan: CapturePlan, output_root: Path) -> None:
        env_key = "TRITON_DUMP_DIR" if plan.adapter == "triton.compiler" else "CUTE_DSL_DUMP_DIR"
        dump_path = plan.collector_environment.get(env_key)
        if dump_path is not None:
            Path(dump_path).mkdir(parents=True, exist_ok=True)

    async def _collect_kernel_build(
        self,
        plan: CapturePlan,
        output_root: Path,
        run_id: str,
        exit_code: int,
    ) -> tuple[list[tuple[ArtifactRegistration, int]], KernelBuildManifestV2]:
        env_key = "TRITON_DUMP_DIR" if plan.adapter == "triton.compiler" else "CUTE_DSL_DUMP_DIR"
        dump_path = plan.collector_environment.get(env_key)
        dump_dir = Path(dump_path) if dump_path is not None else output_root / "dumps"
        source_environment = {
            key: value
            for key, value in plan.collector_environment.items()
            if key not in {"FLAMEOX_OBSERVATIONS_PATH"}
        }
        cute_keep: tuple[str, ...] | None = None
        if plan.adapter == "cute.compiler":
            keep_value = plan.adapter_options.get("keep_allowlist")
            if isinstance(keep_value, list):
                cute_keep = tuple(str(item) for item in keep_value)
        reproducer_path: Path | None = None
        reproducer_env = plan.collector_environment.get("TRITON_REPRODUCER_PATH")
        if reproducer_env is not None:
            reproducer_path = Path(reproducer_env)
        collector = KernelBuildCaptureCollector(self.workspace)
        build_context = managed_kernel_build_context(
            workload_instance=plan.workload_instance,
            adapter=plan.adapter,
            producer_version=plan.adapter_version,
            adapter_options=plan.adapter_options,
        )
        manifest, manifest_path, native_paths = await run_atomic_thread(
            lambda: collector.collect(
                adapter=plan.adapter,
                dump_dir=dump_dir,
                output_root=output_root,
                workload_label=plan.workload_name,
                build_context=build_context,
                exit_code=exit_code,
                producer_version=plan.adapter_version,
                source_environment=source_environment,
                cute_keep_allowlist=cute_keep,
                reproducer_path=reproducer_path,
            )
        )
        registrations: list[tuple[ArtifactRegistration, int]] = []
        declarations = {
            stage.artifact.path: stage for stage in manifest.stages if stage.artifact is not None
        }
        for native in native_paths:
            relative = native.relative_to(output_root).as_posix()
            stage = declarations[relative]
            declaration = stage.artifact
            assert declaration is not None
            registered = await self._register_path_async(
                run_id,
                native,
                kind=ArtifactKind.KERNEL_BUILD,
                role=declaration.role,
                media_type=declaration.media_type,
                producer=plan.adapter,
                producer_version=plan.adapter_version,
                display_name=relative,
            )
            self._require_registered_integrity(
                registered,
                expected_byte_length=declaration.byte_length,
                expected_sha256=declaration.sha256,
            )
            registrations.append(registered)
        manifest_bytes = manifest_path.stat().st_size
        with manifest_path.open("rb") as stream:
            manifest_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        registered_manifest = await self._register_path_async(
            run_id,
            manifest_path,
            kind=ArtifactKind.KERNEL_BUILD,
            role="kernel_build_manifest",
            media_type="application/json",
            producer=plan.adapter,
            producer_version=plan.adapter_version,
        )
        self._require_registered_integrity(
            registered_manifest,
            expected_byte_length=manifest_bytes,
            expected_sha256=manifest_sha256,
        )
        registrations.append(registered_manifest)
        return registrations, manifest

    def _require_registered_integrity(
        self,
        registered: tuple[ArtifactRegistration, int],
        *,
        expected_byte_length: int,
        expected_sha256: str | None = None,
    ) -> None:
        registration, byte_length = registered
        stored = self.artifacts.get(registration.artifact_id)
        if byte_length != expected_byte_length or (
            expected_sha256 is not None
            and stored.content.integrity.sha256 != expected_sha256.removeprefix("sha256:").lower()
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                f"Registered artifact {registration.display_name!r} changed after inventory.",
            )

    def _valid_native_output_paths(
        self,
        plan: CapturePlan,
        output_root: Path,
    ) -> tuple[Path, ...]:
        return tuple(
            path
            for path in self._native_output_paths(plan, output_root)
            if self._native_output_is_valid(path)
        )

    @staticmethod
    def _native_output_is_valid(path: Path) -> bool:
        try:
            metadata = path.stat()
            return not path.is_symlink() and stat.S_ISREG(metadata.st_mode) and metadata.st_size > 0
        except OSError:
            return False

    def _unexpected_native_output_paths(
        self,
        plan: CapturePlan,
        output_root: Path,
    ) -> tuple[Path, ...]:
        if plan.adapter != "torch.profiler":
            return ()
        options = torch_profiler_options(cast(dict[str, object], plan.adapter_options))
        if not isinstance(options, SdkTorchProfilerOptions):
            return ()
        expected = set(self._native_output_paths(plan, output_root))
        return tuple(
            path
            for path in sorted(output_root.glob("torch-trace-cycle-*.json"))
            if path not in expected
        )

    def _nvbench_manifest_is_valid(
        self,
        plan: CapturePlan,
        output_root: Path,
    ) -> bool:
        """Validate the NVBench JSON and all declared sidecars.

        Returns ``True`` only if the JSON parses successfully, every
        declared sidecar filename is a safe relative path under
        ``output_root``, and each sidecar file exists as a regular
        non-linked file with exact declared byte length.  Any parse
        error, unsupported hint, missing/mismatched sidecar, or path
        escape returns ``False``.
        """
        definition = builtin_adapter(plan.adapter)
        if definition is None or definition.output_filename is None:
            return False
        json_path = output_root / definition.output_filename
        max_bytes = self.workspace.config.capture.max_artifact_bytes
        try:
            specs = load_nvbench_sidecar_specs(json_path, max_bytes=max_bytes)
        except DomainError:
            return False
        for spec in specs:
            try:
                sidecar_path = resolve_nvbench_sidecar_path(spec.filename, output_root)
                validate_nvbench_sidecar_file(sidecar_path, spec.byte_length)
            except DomainError:
                return False
        return True

    def _native_output_manifest_is_valid(
        self,
        plan: CapturePlan,
        output_root: Path,
    ) -> bool:
        if plan.adapter == "nvbench":
            return self._nvbench_manifest_is_valid(plan, output_root)
        if plan.adapter == "compute-sanitizer":
            path = output_root / "compute-sanitizer.xml"
            try:
                if (
                    path.is_symlink()
                    or path.stat().st_size > self.workspace.config.capture.max_artifact_bytes
                ):
                    return False
                with path.open("rb") as stream:
                    prefix = stream.read(1_024)
                    stream.seek(max(0, path.stat().st_size - 1_024))
                    suffix = stream.read(1_024)
            except OSError:
                return False
            return b"<ComputeSanitizerOutput" in prefix and (
                b"</ComputeSanitizerOutput>" in suffix or b"<ComputeSanitizerOutput/>" in suffix
            )
        if plan.adapter != "torch.profiler":
            return True
        options = torch_profiler_options(cast(dict[str, object], plan.adapter_options))
        if not isinstance(options, SdkTorchProfilerOptions):
            return True
        path = output_root / "torch-profiler-cycles.json"
        try:
            if path.is_symlink() or path.stat().st_size > 65_536:
                return False
            payload = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        return payload == {
            "schema_version": "flameox.torch-profiler-cycles.v1",
            "expected_cycles": options.expected_cycles,
            "files": list(options.output_filenames),
        }

    def _quarantine_native_output(
        self,
        plan: CapturePlan,
        output_root: Path,
        *,
        reason: str,
    ) -> LimitationDetail | None:
        paths = tuple(
            path
            for path in (
                *self._native_output_paths(plan, output_root),
                *self._unexpected_native_output_paths(plan, output_root),
                *(
                    (output_root / "torch-profiler-cycles.json",)
                    if plan.adapter == "torch.profiler"
                    else ()
                ),
            )
            if path.exists()
        )
        if not paths:
            return None
        definition = builtin_adapter(plan.adapter)
        expected_format = (
            definition.supported_formats[0]
            if definition is not None and definition.supported_formats
            else None
        )
        quarantine_ids: list[str] = []
        try:
            for path in paths:
                manifest = QuarantineService(self.workspace).quarantine(
                    path,
                    operation="capture.native_output",
                    reason=reason,
                    expected_format=expected_format,
                    actual_format=(
                        "symlink"
                        if path.is_symlink()
                        else ("regular_file" if path.is_file() else "non_regular_file")
                    ),
                    adapter=plan.adapter,
                    originating_run_id=plan.run_id,
                )
                quarantine_ids.append(manifest.quarantine_id)
        except DomainError as error:
            return _limitation(
                LimitationSource.ARTIFACT,
                "native_output_quarantine_failed",
                f"Invalid {plan.adapter} output could not be quarantined: {error.message}",
            )
        return _limitation(
            LimitationSource.ARTIFACT,
            "native_output_quarantined",
            f"Invalid {plan.adapter} output was quarantined "
            f"({', '.join(quarantine_ids)}): {reason}",
        )

    def _quarantine_adapter_output(
        self,
        path: Path,
        *,
        expected_format: str,
        run_id: str,
        adapter: str,
    ) -> str:
        manifest = QuarantineService(self.workspace).quarantine(
            path,
            operation="capture.adapter_output",
            reason="Collector failed; declared adapter output was not published.",
            expected_format=expected_format,
            adapter=adapter,
            originating_run_id=run_id,
        )
        return manifest.quarantine_id

    async def _adapter_command(
        self,
        adapter: str,
        workload: CommandSpec,
        output_root: Path,
        *,
        capability: CapabilityReport | None = None,
        options: dict[str, JsonValue] | None = None,
        workload_executable: str | None = None,
    ) -> _AdapterBinding:
        adapter_definition = builtin_adapter(adapter)
        if adapter_definition is not None:
            capability = capability or self.capabilities.get(adapter)
            if adapter != "command" and capability.status is not CapabilityStatus.AVAILABLE:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    f"Adapter {adapter!r} is unavailable.",
                    remediation=capability.remediation
                    or ("Call list_capabilities and choose an available capture adapter.",),
                    next_action=tool_action(
                        ActionId.INSPECT_CAPABILITIES,
                        adapter=adapter,
                    ),
                )
            invocation = build_capture_invocation(
                adapter,
                workload.argv,
                output_root,
                executable=capability.executable,
                timeout_seconds=workload.timeout_seconds,
                options=cast(dict[str, object] | None, options),
                project_root=self.workspace.project_root,
                workload_executable=workload_executable,
            )
            if invocation.environment:
                conflicts = {
                    key: {
                        "workload_value": workload.env_overrides[key],
                        "adapter_value": invocation.environment[key],
                    }
                    for key in invocation.environment
                    if key in workload.env_overrides
                    and workload.env_overrides[key] != invocation.environment[key]
                }
                if conflicts:
                    raise DomainError(
                        ErrorCode.INVALID_CAPTURE_PLAN,
                        f"Workload env_overrides conflict with {adapter} adapter environment.",
                        details={"conflicts": conflicts},
                        remediation=(
                            "Remove the conflicting env_override from the workload or set it "
                            "to the identical value required by the adapter.",
                        ),
                    )
            version = capability.version
            if adapter in {"node-cpu-prof", "node-heap-prof"}:
                if workload_executable is None:
                    raise DomainError(
                        ErrorCode.INVALID_CAPTURE_PLAN,
                        "The V8 capture plan is missing the declared Node executable.",
                    )
                version = await self.capabilities.probe_executable_version(
                    workload_executable,
                    cwd=Path(workload.cwd),
                )
                if not node_version_is_supported(version):
                    raise DomainError(
                        ErrorCode.ADAPTER_INCOMPATIBLE,
                        "The declared Node.js executable is too old for stable V8 profiling.",
                        remediation=("Use Node.js 20.16+ or 22.4+ for V8 profiling.",),
                    )
            return _AdapterBinding(
                argv=invocation.argv,
                artifact_kinds=invocation.artifact_kinds,
                expected_overhead=invocation.expected_overhead,
                limitations=invocation.limitations,
                environment=invocation.environment,
                limitation_details=tuple(
                    _limitation(LimitationSource.ADAPTER, "capture.limitation", message)
                    for message in invocation.limitations
                ),
                permissions=adapter_definition.permissions,
                version=version,
            )

        registry = AdapterRegistry(self.workspace)
        descriptor, contract = registry.load_contract(adapter)
        try:
            probe = AdapterProbeResult.model_validate(
                await contract.probe(
                    AdapterProbeContext(project_root=str(self.workspace.project_root))
                )
            )
            if probe.status is AdapterProbeStatus.UNAVAILABLE:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    f"Approved adapter {adapter!r} is unavailable.",
                    remediation=probe.remediation,
                )
            execution_plan = AdapterExecutionPlan.model_validate(
                await contract.plan(
                    AdapterPlanRequest(
                        project_root=str(self.workspace.project_root),
                        output_root=str(output_root),
                        workload=workload,
                    )
                )
            )
        except DomainError:
            raise
        except Exception as error:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                f"Approved adapter {adapter!r} failed during bounded planning.",
                details={"exception_type": type(error).__name__},
            ) from error
        if execution_plan.adapter != adapter:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The approved adapter returned a plan for another adapter name.",
            )
        return _AdapterBinding(
            argv=(*execution_plan.argv_prefix, "--", *workload.argv),
            artifact_kinds=tuple(item.kind for item in execution_plan.artifacts),
            expected_overhead=execution_plan.expected_overhead,
            limitations=(*probe.limitations, *execution_plan.limitations),
            environment={},
            limitation_details=tuple(
                _limitation(LimitationSource.ADAPTER, "probe.limitation", message)
                for message in (*probe.limitations, *execution_plan.limitations)
            ),
            permissions=execution_plan.permissions,
            version=descriptor.version,
            execution_plan=execution_plan,
            package_identity=descriptor.package_identity,
        )

    async def _contain(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        writable: Path,
        writable_roots: tuple[WritableRootBinding, ...],
        unit_name: str,
        required: bool,
        use_containment: bool = True,
    ) -> tuple[
        CaptureContainment,
        bool,
        str | None,
        tuple[str, ...],
    ]:
        if not use_containment:
            return "uncontained", False, None, argv
        if self.workspace.config.execution.containment is ContainmentPolicy.DISABLED:
            if required:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "MCP capture requires containment but containment is disabled.",
                )
            return "uncontained", False, None, argv
        bwrap_binding = (
            ExecutableResolver().resolve_host_tool("bwrap") if os.name == "posix" else None
        )
        if bwrap_binding is None:
            if required:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "Managed MCP capture requires Linux bubblewrap containment.",
                    remediation=("Install bubblewrap or change the trusted local policy.",),
                )
            return "unavailable", False, None, argv
        bwrap_argv = self._bubblewrap_argv(
            str(bwrap_binding.invocation_path),
            argv,
            cwd=cwd,
            writable=writable,
            writable_roots=writable_roots,
        )
        systemd_binding = ExecutableResolver().resolve_host_tool("systemd-run")
        if systemd_binding is None or not await self._systemd_user_scope_available(systemd_binding):
            if required:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "Managed MCP capture requires a systemd user scope for descendant containment.",
                    remediation=(
                        "Run under a systemd user manager or use the trusted local policy.",
                    ),
                )
            return (
                "degraded",
                self.workspace.config.execution.network is NetworkPolicy.DENY_WHEN_CONTAINED,
                None,
                bwrap_argv,
            )
        wrapped = (
            str(systemd_binding.invocation_path),
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            "--expand-environment=no",
            f"--unit={unit_name}",
            "--property=KillMode=control-group",
            f"--property=CPUQuota={self.workspace.config.execution.max_cpu_percent}%",
            f"--property=MemoryMax={self.workspace.config.execution.max_memory_bytes}",
            f"--property=TasksMax={self.workspace.config.execution.max_processes}",
            "--",
            *bwrap_argv,
        )
        return (
            "active",
            self.workspace.config.execution.network is NetworkPolicy.DENY_WHEN_CONTAINED,
            unit_name,
            wrapped,
        )

    async def _systemd_user_scope_available(self, systemd_run: ResolvedExecutable) -> bool:
        true_executable = ExecutableResolver().resolve_host_tool("true")
        if true_executable is None:
            return False
        unit_name = f"flameox-probe-{secrets.token_hex(8)}.scope"
        try:
            outcome = await self.broker.run(
                ExecutionRequest(
                    argv=(
                        str(systemd_run.invocation_path),
                        "--user",
                        "--scope",
                        "--quiet",
                        "--collect",
                        "--expand-environment=no",
                        f"--unit={unit_name}",
                        "--",
                        str(true_executable.invocation_path),
                    ),
                    cwd=self.workspace.project_root,
                    environment_allowlist=(
                        self.workspace.config.execution.child_environment_allowlist
                    ),
                    allowed_working_roots=self._allowed_roots(),
                    timeout_seconds=5,
                    executable_binding=systemd_run,
                    max_output_bytes=65_536,
                    systemd_scope_unit=unit_name,
                )
            )
        except DomainError:
            return False
        return outcome.process.exit_code == 0

    def _bubblewrap_argv(
        self,
        bwrap: str,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        writable: Path,
        writable_roots: tuple[WritableRootBinding, ...],
    ) -> tuple[str, ...]:
        project_root = self.workspace.project_root.resolve()
        resolved_cwd = cwd.resolve()
        diagnostics = self.workspace.paths.root.resolve()
        wrapped: list[str] = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-pid",
            "--unsetenv",
            "DBUS_SESSION_BUS_ADDRESS",
            "--unsetenv",
            "XDG_RUNTIME_DIR",
        ]
        if self.workspace.config.execution.network is NetworkPolicy.DENY_WHEN_CONTAINED:
            wrapped.append("--unshare-net")
        wrapped.extend(("--tmpfs", "/tmp"))
        for path in ("/usr", "/usr/local", "/bin", "/lib", "/lib64", "/sbin", "/sys"):
            if Path(path).exists():
                wrapped.extend(("--ro-bind", path, path))
        for path in (
            "/etc/ld.so.cache",
            "/etc/ld.so.conf",
            "/etc/passwd",
            "/etc/group",
            "/etc/nsswitch.conf",
            "/etc/localtime",
        ):
            if Path(path).exists():
                wrapped.extend(("--ro-bind", path, path))
        wrapped.extend(("--ro-bind", str(project_root), str(project_root)))
        diagnostics = diagnostics.resolve()
        launcher_paths = {
            Path(argument).resolve().parent
            for argument in argv
            if os.path.isabs(argument)
            and Path(argument).name.endswith("_launcher.py")
            and Path(argument).is_file()
            and diagnostics not in Path(argument).resolve().parents
        }
        for launcher_path in sorted(launcher_paths):
            try:
                launcher_path.relative_to(project_root)
            except ValueError:
                wrapped.extend(("--ro-bind", str(launcher_path), str(launcher_path)))
        for binding in writable_roots:
            wrapped.extend(("--bind", binding.storage_path, binding.target_path))
        executable = Path(argv[0])
        if executable.is_absolute() and executable.exists():
            executable_candidates = [executable, executable.resolve()]
            if executable.is_symlink():
                link_target = Path(os.readlink(executable))
                executable_candidates.append(
                    link_target if link_target.is_absolute() else executable.parent / link_target
                )
            executable_roots = {
                (
                    candidate.parent.parent
                    if candidate.parent.name in {"bin", "sbin"}
                    else candidate.parent
                ).absolute()
                for candidate in executable_candidates
            }
            for executable_root in sorted(executable_roots):
                try:
                    executable_root.resolve().relative_to(project_root)
                except ValueError:
                    wrapped.extend(("--ro-bind", str(executable_root), str(executable_root)))
        try:
            resolved_cwd.relative_to(project_root)
        except ValueError:
            wrapped.extend(("--ro-bind", str(resolved_cwd), str(resolved_cwd)))
        wrapped.extend(
            (
                "--tmpfs",
                str(diagnostics),
                "--dir",
                str(writable),
                "--bind",
                str(writable),
                str(writable),
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--chdir",
                str(resolved_cwd),
                "--",
                *argv,
            )
        )
        return tuple(wrapped)

    async def _recheck(self, plan: CapturePlan) -> None:
        request = {
            "workspace_id": plan.workspace_id,
            "run_id": plan.run_id,
            "workload_name": plan.workload_name,
            "workload_definition_id": plan.workload_definition_id,
            "instance": plan.workload_instance.model_dump(mode="json"),
            "dynamic_parameters": plan.dynamic_parameters,
            "semantics": plan.semantics.model_dump(mode="json"),
            "adapter_execution_plan": plan.adapter_execution_plan,
            "execution_policy": plan.execution_policy,
            "collector_argv": plan.collector_argv,
            "collector_executable_binding": plan.collector_executable_binding.model_dump(
                mode="json"
            ),
            "collector_environment": plan.collector_environment,
            "oracle_argv": plan.oracle_argv,
            "oracle_executable_binding": (
                plan.oracle_executable_binding.model_dump(mode="json")
                if plan.oracle_executable_binding is not None
                else None
            ),
            "oracle_launch_executable_binding": (
                plan.oracle_launch_executable_binding.model_dump(mode="json")
                if plan.oracle_launch_executable_binding is not None
                else None
            ),
            "oracle_containment": plan.oracle_containment,
            "oracle_network_contained": plan.oracle_network_contained,
            "oracle_systemd_scope_unit": plan.oracle_systemd_scope_unit,
            "bound_identities": plan.bound_identities,
            "preflight": plan.preflight.model_dump(mode="json"),
            "adapter_capability": (
                plan.adapter_capability.model_dump(mode="json")
                if plan.adapter_capability is not None
                else None
            ),
            "warnings": plan.warnings,
            "limitation_details": [
                item.model_dump(mode="json") for item in plan.limitation_details
            ],
            "writable_roots": [item.model_dump(mode="json") for item in plan.writable_roots],
            "external_context": (
                plan.external_context.model_dump(mode="json")
                if plan.external_context is not None
                else None
            ),
            "planned_execution_identity": (plan.planned_execution_identity.model_dump(mode="json")),
            "policy": self.workspace.config.model_dump(mode="json"),
            "containment": plan.containment,
            "systemd_scope_unit": plan.systemd_scope_unit,
        }
        if digest_model(request) != plan.request_digest:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Capture plan contents changed after authorization.",
            )
        if plan.adapter_capability is not None and plan.adapter_capability.probe_kind == "active":
            current_capability = await self.capabilities.probe(plan.adapter, refresh=True)
            planned_capability = plan.adapter_capability.model_dump(
                mode="json",
                exclude={"probed_at"},
            )
            current_capability_data = current_capability.model_dump(
                mode="json",
                exclude={"probed_at"},
            )
            if current_capability_data != planned_capability:
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    "Adapter capability changed after active planning.",
                    details={
                        "adapter": plan.adapter,
                        "planned_capability": planned_capability,
                        "current_capability": current_capability_data,
                    },
                )
        if plan.adapter_execution_plan is not None:
            descriptor = AdapterRegistry(self.workspace).approved_descriptor(plan.adapter)
            if (
                descriptor.version != plan.adapter_version
                or descriptor.package_identity
                != plan.bound_identities.get("adapter_package_identity")
            ):
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    "The approved adapter package identity changed after planning.",
                )
            AdapterRegistry(self.workspace).load_contract(plan.adapter)
        current_execution_identity = ExecutionIdentityService(
            self.workspace,
            broker=self.broker,
        ).plan(plan.workload_name, cwd=Path(plan.workload_instance.command.cwd))
        if current_execution_identity.identity_id != plan.planned_execution_identity.identity_id:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "A declared module or native identity input changed after planning.",
            )
        if self.workspace.identity.workspace_id != plan.workspace_id:
            raise DomainError(ErrorCode.INVALID_CAPTURE_PLAN, "Workspace identity changed.")
        definition = self.workloads.definition(plan.workload_name)
        if definition.workload_definition_id != plan.workload_definition_id:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Workload definition changed after planning.",
            )
        self.executables.revalidate(plan.collector_executable_binding)
        self.executables.revalidate(plan.workload_instance.executable_binding)
        if plan.oracle_executable_binding is not None:
            self.executables.revalidate(plan.oracle_executable_binding)
        if plan.oracle_launch_executable_binding is not None:
            self.executables.revalidate(plan.oracle_launch_executable_binding)
        current_preflight = await PreflightService(
            self.workspace,
            capabilities=self.capabilities,
        ).inspect(plan.workload_name, mode=plan.preflight.mode)
        if (
            current_preflight.preflight_id != plan.preflight.preflight_id
            or current_preflight.disposition is PreflightDisposition.BLOCKED
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Workload preflight changed after planning.",
                details={
                    "planned_preflight_id": plan.preflight.preflight_id,
                    "current_preflight_id": current_preflight.preflight_id,
                },
            )
        current_writable = self.workloads.writable_targets(plan.workload_name)
        planned_writable = tuple(
            (Path(item.target_path), item.target_identity) for item in plan.writable_roots
        )
        if current_writable != planned_writable:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "A writable build-output root changed after planning.",
            )

    def _finish_error(
        self,
        running: RunManifest,
        *,
        execution: ExecutionStatus,
        message: str,
        cleanup_complete: bool | None = True,
        process: ProcessResult | None = None,
        process_snapshot_rows: list[dict[str, object]] | None = None,
        process_snapshot_entries: list[dict[str, object]] | None = None,
        limitation_details: tuple[LimitationDetail, ...] = (),
        artifacts: tuple[ArtifactRegistration, ...] = (),
    ) -> RunManifest:
        process = process or ProcessResult(
            cancellation_cause=(
                ProcessCancellationCause.CALLER_CANCELLED
                if execution is ExecutionStatus.CANCELLED
                else (
                    ProcessCancellationCause.TIMEOUT
                    if execution is ExecutionStatus.TIMED_OUT
                    else ProcessCancellationCause.PROCESS_ERROR
                )
            ),
            cleanup_complete=cleanup_complete,
        )
        process_details: list[LimitationDetail] = []
        if process.resources is None:
            process_details.append(
                _limitation(
                    LimitationSource.RESOURCE,
                    "resource_summary_unavailable",
                    "Runtime resource sampling did not produce a summary.",
                )
            )
        else:
            process_details.extend(
                _limitation(
                    LimitationSource.RESOURCE,
                    "resource_metric_unavailable",
                    f"Runtime resource metric {metric!r} was unavailable.",
                )
                for metric in process.resources.unavailable_metrics
            )
            if process.resources.policy_termination is not None:
                process_details.append(
                    _limitation(
                        LimitationSource.RESOURCE,
                        "storage_reserve_exceeded",
                        "Runtime storage reserve terminated the collector.",
                    )
                )
        details = _merge_limitation_details(
            running.limitation_details,
            limitation_details,
            tuple(process_details),
        )
        terminal = running.validated_copy(
            update={
                "revision": running.revision + 1,
                "finished_at": utc_now(),
                "execution_status": execution,
                "capture_status": (
                    CaptureStatus.CANCELLED
                    if execution is ExecutionStatus.CANCELLED
                    else CaptureStatus.FAILED
                ),
                "process": process,
                "artifacts": (*running.artifacts, *artifacts),
                "limitations": _limitation_projection(
                    details,
                    (*running.limitations, message),
                ),
                "limitation_details": details,
            }
        )
        terminal = self.projections.append_run(
            terminal,
            expected_revision=running.revision,
            environment=None,
            source_state=None,
        ).run
        self.publisher.publish_rows(
            {
                "runtime_resource_summaries": [
                    runtime_resource_summary_row(
                        terminal,
                        sampling_interval_ms=self.workspace.config.execution.resource_sampling_interval_ms,
                    )
                ],
                "runtime_writable_root_growth": runtime_writable_root_rows(
                    terminal,
                    project_root=self.workspace.project_root,
                ),
                "process_snapshots": process_snapshot_rows or [],
                "process_snapshot_entries": process_snapshot_entries or [],
            },
            publisher="flameox.capture.observations",
            publisher_version="1",
            input_run_ids=(terminal.run_id,),
        )
        return terminal

    async def _process_adapter_artifacts(
        self,
        plan: CapturePlan,
        output_root: Path,
    ) -> tuple[
        list[tuple[ArtifactRegistration, int]],
        list[dict[str, object]],
        list[dict[str, object]],
        list[str],
    ]:
        execution_plan = AdapterExecutionPlan.model_validate(plan.adapter_execution_plan)
        registry = AdapterRegistry(self.workspace)
        descriptor, contract = registry.load_contract(plan.adapter)
        self._require_bound_adapter_identity(plan, descriptor)
        registrations: list[tuple[ArtifactRegistration, int]] = []
        validations: list[dict[str, object]] = []
        extractions: list[dict[str, object]] = []
        limitations: list[str] = []
        for declaration in execution_plan.artifacts:
            path = output_root / declaration.relative_path
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(output_root.resolve())
            except (FileNotFoundError, ValueError) as error:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    f"Adapter {plan.adapter!r} did not produce declared artifact "
                    f"{declaration.relative_path!r}.",
                    run_id=plan.run_id,
                ) from error
            if not resolved.is_file():
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    "A declared adapter artifact is not a regular file.",
                    run_id=plan.run_id,
                )
            stored = await run_atomic_thread(
                partial(
                    self.artifacts.import_path,
                    resolved,
                    allowed_roots=(self.workspace.paths.staging,),
                    max_bytes=self.workspace.config.capture.max_artifact_bytes,
                )
            )
            self._require_bound_adapter_identity(
                plan,
                registry.approved_descriptor(plan.adapter),
            )
            try:
                validation = AdapterValidationResult.model_validate(
                    await contract.validate(str(stored.payload_path), declaration)
                )
            except Exception as error:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    f"Adapter {plan.adapter!r} artifact validation failed.",
                    details={"exception_type": type(error).__name__},
                    run_id=plan.run_id,
                ) from error
            if validation.validator_version != execution_plan.validator_version:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    "Adapter validation version differs from the bound plan.",
                    run_id=plan.run_id,
                )
            validation_identity = {
                "run_id": plan.run_id,
                "input_artifact_id": stored.content.artifact_id,
                "adapter": plan.adapter,
                "adapter_package_identity": descriptor.package_identity,
                "validator_version": validation.validator_version,
                "valid": validation.valid,
                "limitations": list(validation.limitations),
            }
            validation_row = {
                "validation_id": digest_model(validation_identity),
                **validation_identity,
                "input_byte_length": stored.content.byte_length,
            }
            if not validation.valid:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    f"Adapter {plan.adapter!r} rejected its declared artifact.",
                    details={"limitations": list(validation.limitations)},
                    run_id=plan.run_id,
                )
            limitations.extend(validation.limitations)
            validations.append(validation_row)
            registration = ArtifactRegistration(
                registration_id=new_id(),
                run_id=plan.run_id,
                artifact_id=stored.content.artifact_id,
                display_name=resolved.name,
                media_type=declaration.media_type,
                kind=declaration.kind,
                role=declaration.role,
                producer=plan.adapter,
                producer_version=plan.adapter_version,
                sensitivity=declaration.sensitivity,
            )
            registrations.append((registration, stored.content.byte_length))
            self._require_bound_adapter_identity(
                plan,
                registry.approved_descriptor(plan.adapter),
            )
            try:
                extraction = AdapterExtractionResult.model_validate(
                    await contract.extract(
                        str(stored.payload_path),
                        declaration,
                    )
                )
            except Exception as error:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    f"Adapter {plan.adapter!r} extraction failed.",
                    details={"exception_type": type(error).__name__},
                    run_id=plan.run_id,
                ) from error
            if extraction.extractor_version != execution_plan.extractor_version:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    "Adapter extraction version differs from the bound plan.",
                    run_id=plan.run_id,
                )
            limitations.extend(extraction.limitations)
            identity = {
                "run_id": plan.run_id,
                "input_artifact_id": registration.artifact_id,
                "adapter": plan.adapter,
                "adapter_package_identity": descriptor.package_identity,
                "extractor_version": extraction.extractor_version,
                "summary": extraction.summary,
            }
            extractions.append(
                {
                    "extraction_id": digest_model(identity),
                    "run_id": plan.run_id,
                    "input_artifact_id": registration.artifact_id,
                    "adapter": plan.adapter,
                    "adapter_package_identity": descriptor.package_identity,
                    "extractor_version": extraction.extractor_version,
                    "summary_json": json.dumps(
                        extraction.summary,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "limitations": list(extraction.limitations),
                }
            )
        return registrations, validations, extractions, limitations

    @staticmethod
    def _require_bound_adapter_identity(
        plan: CapturePlan,
        descriptor: AdapterDescriptor,
    ) -> None:
        if (
            descriptor.version != plan.adapter_version
            or descriptor.package_identity != plan.bound_identities.get("adapter_package_identity")
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "The approved adapter package identity changed before artifact processing.",
            )

    def _register_path(
        self,
        run_id: str,
        path: Path,
        *,
        kind: ArtifactKind,
        role: str,
        media_type: str,
        producer: str = "flameox.capture",
        producer_version: str | None = None,
        sensitivity: Sensitivity = Sensitivity.INTERNAL,
        display_name: str | None = None,
    ) -> tuple[ArtifactRegistration, int]:
        stored = self.artifacts.import_path(
            path,
            allowed_roots=(self.workspace.paths.staging,),
            max_bytes=self.workspace.config.capture.max_artifact_bytes,
        )
        registration = ArtifactRegistration(
            registration_id=new_id(),
            run_id=run_id,
            artifact_id=stored.content.artifact_id,
            display_name=display_name or path.name,
            media_type=media_type,
            kind=kind,
            role=role,
            producer=producer,
            producer_version=producer_version,
            sensitivity=sensitivity,
        )
        return registration, stored.content.byte_length

    async def _register_path_async(
        self,
        run_id: str,
        path: Path,
        *,
        kind: ArtifactKind,
        role: str,
        media_type: str,
        producer: str = "flameox.capture",
        producer_version: str | None = None,
        sensitivity: Sensitivity = Sensitivity.INTERNAL,
        display_name: str | None = None,
    ) -> tuple[ArtifactRegistration, int]:
        return await run_atomic_thread(
            lambda: self._register_path(
                run_id,
                path,
                kind=kind,
                role=role,
                media_type=media_type,
                producer=producer,
                producer_version=producer_version,
                sensitivity=sensitivity,
                display_name=display_name,
            )
        )

    def _allowed_roots(self) -> tuple[Path, ...]:
        roots = tuple(
            (self.workspace.paths.root / value).resolve()
            for value in self.workspace.config.execution.allowed_working_roots
        )
        return (*roots, self.workspace.project_root)

    def _lease(self, process_id: int) -> CaptureLease | None:
        observed = utc_now()
        try:
            boot_id = read_boot_id()
            process_start_identity = read_proc_stat_start_identity(process_id)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise DomainError(
                ErrorCode.PROCESS_FAILED,
                "Could not establish the child process lease identity.",
            ) from exc
        return CaptureLease(
            process_start_identity=process_start_identity,
            process_id=process_id,
            boot_id=boot_id,
            heartbeat_monotonic_ns=time.monotonic_ns(),
            observed_at=observed,
            expires_at=observed + timedelta(seconds=60),
        )
