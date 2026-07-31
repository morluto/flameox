from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import secrets
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import JsonValue

from flameox.adapters.builtins import (
    build_capture_invocation,
    builtin_adapter,
)
from flameox.adapters.registry import AdapterRegistry
from flameox.application.async_work import run_atomic_thread
from flameox.application.capabilities import CapabilityService
from flameox.application.environment import AcceleratorIdentityService, collect_environment
from flameox.application.evidence_rows import (
    artifact_registration_row,
    environment_row,
    source_state_row,
)
from flameox.application.execution_identity import ExecutionIdentityService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.oracle_receipts import parse_oracle_receipt
from flameox.application.preflight import PreflightService
from flameox.application.run_rows import run_row
from flameox.application.source import collect_source_state
from flameox.application.workloads import Scalar, WorkloadService
from flameox.domain import (
    AcceleratorIdentityFacet,
    AdapterExecutionPlan,
    AdapterExtractionResult,
    AdapterPlanRequest,
    AdapterProbeContext,
    AdapterProbeResult,
    AdapterValidationResult,
    ArtifactKind,
    ArtifactRegistration,
    CapabilityStatus,
    CaptureLease,
    CapturePlan,
    CaptureStatus,
    CommandSpec,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ExternalExecutionContext,
    IdentityQuality,
    OracleReceiptRecord,
    OracleStrength,
    ProcessResult,
    RunManifest,
    RunType,
    Sensitivity,
    ValidationStatus,
    WritableRootBinding,
    digest_model,
    new_id,
)
from flameox.domain.models import utc_now
from flameox.evidence import GenerationPublisher
from flameox.execution import ExecutionOutcome, ExecutionRequest, ResourcePolicy, SubprocessBroker
from flameox.models import ContractModel
from flameox.observability import OperationLogger, elapsed_ms
from flameox.storage import ArtifactStore, RunStore, StorageQuota, Workspace
from flameox.storage.atomic import atomic_write_bytes

_MAX_PYTEST_SIDECAR_BYTES = 16 * 1024 * 1024


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
    permissions: tuple[str, ...]
    version: str | None
    execution_plan: AdapterExecutionPlan | None = None
    package_identity: str | None = None


@dataclass(slots=True)
class _PlanEntry:
    plan: CapturePlan
    expires_monotonic: float
    consumed: bool = False


@dataclass(slots=True)
class _CaptureExecution:
    service: CaptureService
    plan: CapturePlan
    output_root: Path
    logger: OperationLogger
    operation_id: str
    started_monotonic: float
    progress: Callable[[float, float, str], Awaitable[None]] | None
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
        if self.progress is not None:
            await self.progress(completed, 8, message)

    async def record_lease(self, process_id: int) -> None:
        if self.run is None:
            raise DomainError(ErrorCode.INTERNAL_ERROR, "capture run is not initialized")
        lease = self.service._lease(process_id)
        if lease is None:
            return
        current = self.run
        leased = current.model_copy(
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
            self.run = await asyncio.shield(lease_write)
            raise

    async def record_cleanup(self, complete: bool) -> None:
        self.cleanup_complete = complete

    async def terminate(
        self,
        *,
        execution: ExecutionStatus,
        message: str,
        phase: str,
        error_code: str,
        cleanup_complete: bool | None = None,
        process: ProcessResult | None = None,
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

        def finish_error(run: RunManifest) -> RunManifest:
            return self.service._finish_error(
                run,
                execution=execution,
                message=message,
                cleanup_complete=(
                    self.cleanup_complete if cleanup_complete is None else cleanup_complete
                ),
                process=process,
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


class CapturePlanRegistry:
    """Bounded in-memory authorization tokens for one server process."""

    def __init__(
        self,
        *,
        capacity: int = 256,
        ttl_seconds: float = 300,
        max_parallel_captures: int = 2,
    ) -> None:
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._plans: dict[str, _PlanEntry] = {}
        self._lock = asyncio.Lock()
        self._capture_slots = asyncio.Semaphore(max_parallel_captures)

    async def acquire_capture_slot(self) -> None:
        await self._capture_slots.acquire()

    def release_capture_slot(self) -> None:
        self._capture_slots.release()

    async def issue(self, plan: CapturePlan) -> None:
        async with self._lock:
            self._evict_expired()
            if len(self._plans) >= self.capacity:
                oldest = min(
                    self._plans,
                    key=lambda key: self._plans[key].expires_monotonic,
                )
                del self._plans[oldest]
            self._plans[plan.plan_id] = _PlanEntry(
                plan=plan,
                expires_monotonic=time.monotonic() + self.ttl_seconds,
            )

    async def consume(self, plan_id: str) -> CapturePlan:
        async with self._lock:
            self._evict_expired()
            entry = self._plans.get(plan_id)
            if entry is None:
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    "Capture plan is missing, expired, or belongs to another server process.",
                )
            if entry.consumed:
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    "Capture plan has already been consumed.",
                )
            entry.consumed = True
            return entry.plan

    async def inspect(self, plan_id: str) -> CapturePlan:
        async with self._lock:
            self._evict_expired()
            entry = self._plans.get(plan_id)
            if entry is None:
                raise DomainError(
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    "Capture plan is missing, expired, or belongs to another server process.",
                )
            return entry.plan

    def _evict_expired(self) -> None:
        now = time.monotonic()
        for key in [key for key, entry in self._plans.items() if entry.expires_monotonic <= now]:
            del self._plans[key]


class CaptureService:
    def __init__(
        self,
        workspace: Workspace,
        *,
        plans: CapturePlanRegistry | None = None,
        broker: SubprocessBroker | None = None,
    ) -> None:
        self.workspace = workspace
        self.workloads = WorkloadService(workspace)
        self.capabilities = CapabilityService(workspace)
        self.plans = plans or CapturePlanRegistry(
            max_parallel_captures=workspace.config.capture.max_parallel_captures
        )
        self.broker = broker or SubprocessBroker()
        self.runs = RunStore(workspace)
        self.artifacts = ArtifactStore(workspace)
        self.publisher = GenerationPublisher(workspace)

    async def plan(
        self,
        *,
        workload_name: str,
        adapter: str,
        parameters: dict[str, Scalar] | None = None,
        execution_policy: ExecutionPolicy,
        preflight_mode: Literal["passive", "active"] = "passive",
        external_context: ExternalExecutionContext | None = None,
    ) -> CapturePlan:
        instance = self.workloads.resolve(
            workload_name,
            parameters,
            require_approval=execution_policy.requires_workload_approval,
        )
        definition = self.workloads.definition(workload_name)
        approval = definition.approved_definition_digest
        if execution_policy.requires_workload_approval and approval is None:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Workload {workload_name!r} is not approved.",
            )
        preflight = await PreflightService(
            self.workspace,
            capabilities=self.capabilities,
        ).inspect(workload_name, mode=preflight_mode)
        if preflight.disposition == "blocked":
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Required preflight checks failed for workload {workload_name!r}.",
                details={
                    "preflight": preflight.model_dump(mode="json"),
                    "next_tool": "get_declared_workflow",
                },
                remediation=tuple(
                    remediation
                    for item in preflight.requirements
                    for remediation in item.remediation
                ),
            )
        planned_execution_identity = ExecutionIdentityService(
            self.workspace,
            broker=self.broker,
        ).plan(workload_name)
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
        adapter_binding = await self._adapter_command(
            adapter,
            instance.command,
            output_root,
        )
        collector_argv = adapter_binding.argv
        kinds = adapter_binding.artifact_kinds
        overhead = adapter_binding.expected_overhead
        warnings = adapter_binding.limitations
        adapter_version = adapter_binding.version
        containment, network_contained, systemd_scope_unit, collector_argv = await self._contain(
            collector_argv,
            cwd=Path(instance.command.cwd),
            writable=output_root,
            writable_roots=writable_roots,
            unit_name=f"flameox-capture-{plan_id[:24]}.scope",
            required=(
                execution_policy.requires_containment(self.workspace.config.execution.containment)
            ),
        )
        identities: dict[str, JsonValue] = {
            "collector_executable": cast(
                JsonValue,
                self._executable_identity(collector_argv[0]),
            ),
            "workload_executable": cast(
                JsonValue,
                self._executable_identity(instance.command.argv[0]),
            ),
        }
        if adapter_binding.package_identity is not None:
            identities["adapter_package_identity"] = adapter_binding.package_identity
        created_at = utc_now()
        request: dict[str, Any] = {
            "workspace_id": self.workspace.identity.workspace_id,
            "run_id": run_id,
            "workload_name": workload_name,
            "definition_id": definition.workload_definition_id,
            "approval": approval or definition.workload_definition_id,
            "instance": instance.model_dump(mode="json"),
            "adapter": adapter,
            "adapter_version": adapter_version,
            "adapter_execution_plan": (
                adapter_binding.execution_plan.model_dump(mode="json")
                if adapter_binding.execution_plan is not None
                else None
            ),
            "execution_policy": execution_policy.value,
            "collector_argv": collector_argv,
            "collector_environment": collector_environment,
            "bound_identities": identities,
            "preflight": preflight.model_dump(mode="json"),
            "writable_roots": [item.model_dump(mode="json") for item in writable_roots],
            "external_context": (
                external_context.model_dump(mode="json") if external_context is not None else None
            ),
            "planned_execution_identity": planned_execution_identity.model_dump(mode="json"),
            "policy": self.workspace.config.model_dump(mode="json"),
            "containment": containment,
            "systemd_scope_unit": systemd_scope_unit,
        }
        plan = CapturePlan(
            plan_id=plan_id,
            run_id=run_id,
            request_digest=digest_model(request),
            workspace_id=self.workspace.identity.workspace_id,
            workload_name=workload_name,
            workload_definition_id=definition.workload_definition_id,
            approval_digest=approval or definition.workload_definition_id,
            workload_instance=instance,
            adapter=adapter,
            adapter_version=adapter_version,
            adapter_execution_plan=(
                adapter_binding.execution_plan.model_dump(mode="json")
                if adapter_binding.execution_plan is not None
                else None
            ),
            execution_policy=execution_policy.value,
            collector_argv=collector_argv,
            collector_environment=collector_environment,
            expected_artifact_kinds=kinds,
            expected_overhead=overhead,
            containment=containment,
            network_contained=network_contained,
            systemd_scope_unit=systemd_scope_unit,
            permissions=adapter_binding.permissions,
            preflight=preflight,
            writable_roots=writable_roots,
            external_context=external_context,
            planned_execution_identity=planned_execution_identity,
            bound_identities=identities,
            limits={
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
            warnings=warnings,
            created_at=created_at,
            expires_at=created_at + timedelta(seconds=self.plans.ttl_seconds),
        )
        await self.plans.issue(plan)
        return plan

    async def execute(
        self,
        plan_id: str,
        *,
        progress: Callable[[float, float, str], Awaitable[None]] | None = None,
    ) -> CaptureResult:
        logger = OperationLogger(self.workspace.paths.root)
        operation_id = logger.new_id()
        started = time.monotonic()

        plan = await self.plans.consume(plan_id)
        await self._recheck(plan)
        StorageQuota(self.workspace).require_capacity(staging=True)
        output_root = self.workspace.paths.staging / "captures" / plan.plan_id
        output_root.mkdir(parents=True, exist_ok=False)
        for binding in plan.writable_roots:
            Path(binding.storage_path).mkdir(parents=True, exist_ok=False)
        capture = _CaptureExecution(
            service=self,
            plan=plan,
            output_root=output_root,
            logger=logger,
            operation_id=operation_id,
            started_monotonic=started,
            progress=progress,
        )
        identity_requirements = (
            WorkloadService(self.workspace)
            .load()
            .workloads[plan.workload_name]
            .identity.environment.required
        )
        planned_accelerator = (
            AcceleratorIdentityFacet(
                provider="cuda",
                status="unknown",
                identity_quality=IdentityQuality.PARTIAL,
                missing_fields=identity_requirements,
                limitations=("Declared accelerator identity has not been observed yet.",),
            )
            if identity_requirements
            else None
        )
        environment = collect_environment(planned_accelerator)
        run_id = plan.run_id
        initial = RunManifest(
            run_id=plan.run_id,
            run_type=RunType.EXECUTION,
            execution_status=ExecutionStatus.PLANNED,
            capture_status=CaptureStatus.PENDING,
            validation_status=ValidationStatus.NOT_REQUESTED,
            workload_definition_id=plan.workload_definition_id,
            workload_instance_id=plan.workload_instance.workload_instance_id,
            measurement_protocol_id=digest_model(
                {
                    "adapter": plan.adapter,
                    "adapter_version": plan.adapter_version,
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
            collector=plan.adapter,
            collector_version=plan.adapter_version,
            command=plan.workload_instance.command,
            preflight=plan.preflight,
            writable_roots=plan.writable_roots,
            external_context=plan.external_context,
            execution_identity=plan.planned_execution_identity,
            limitations=plan.preflight.limitations,
        )
        self.runs.create(initial)
        capture.run = initial
        try:
            accelerator = await AcceleratorIdentityService(self.workspace.project_root).observe(
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
            )
            await capture.report(3, "Source and environment identity collected")
        except asyncio.CancelledError as cancellation:
            try:
                await asyncio.shield(
                    capture.terminate(
                        execution=ExecutionStatus.CANCELLED,
                        message="Capture cancelled while collecting source identity.",
                        phase="capture cancelled during source identity",
                        error_code="cancelled",
                    )
                )
            finally:
                raise cancellation
        prepared = initial.model_copy(
            update={
                "revision": 1,
                "environment_id": environment.environment_id,
                "source_state_id": source_state.source_state_id,
                "execution_identity": execution_identity,
            }
        )
        self.runs.append(prepared, expected_revision=0)
        running = initial.model_copy(
            update={
                "revision": 2,
                "started_at": utc_now(),
                "execution_status": ExecutionStatus.RUNNING,
                "capture_status": CaptureStatus.RUNNING,
                "environment_id": environment.environment_id,
                "source_state_id": source_state.source_state_id,
                "execution_identity": execution_identity,
            }
        )
        self.runs.append(running, expected_revision=1)
        capture.run = running
        acquired_slot = False

        try:
            await self.plans.acquire_capture_slot()
            acquired_slot = True
            await capture.report(4, "Capture slot acquired")
            outcome = await self.broker.run(
                ExecutionRequest(
                    argv=plan.collector_argv,
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
            StorageQuota(self.workspace).require_capacity(staging=True)
            await capture.report(5, "Collector execution complete")
        except asyncio.CancelledError as cancellation:
            try:
                await asyncio.shield(
                    capture.terminate(
                        execution=ExecutionStatus.CANCELLED,
                        message="Capture cancelled by caller after bounded cleanup.",
                        phase="capture cancelled during collector execution",
                        error_code="cancelled",
                    )
                )
            finally:
                raise cancellation
        except DomainError as error:
            status = (
                ExecutionStatus.TIMED_OUT
                if error.code is ErrorCode.PROCESS_TIMEOUT
                else ExecutionStatus.FAILED
            )
            partial_process = (
                ProcessResult.model_validate(error.details["process"])
                if "process" in error.details
                else None
            )
            native = output_root / self._native_filename(plan.adapter)
            if (
                status is ExecutionStatus.TIMED_OUT
                and partial_process is not None
                and native.is_file()
            ):
                outcome = ExecutionOutcome(
                    process=partial_process,
                    stdout=b"",
                    stderr=b"",
                    resolved_executable=Path(plan.collector_argv[0]).resolve(),
                    containment=plan.containment,
                )
                await capture.report(5, "Collector timed out; preserving partial evidence")
            else:
                terminal = await capture.terminate(
                    execution=status,
                    message=error.message,
                    phase="collector execution failed",
                    error_code=error.code.value,
                    process=partial_process,
                )
                error.run_id = terminal.run_id
                raise
        finally:
            if acquired_slot:
                self.plans.release_capture_slot()
        running = capture.run
        if running is None:
            raise DomainError(
                ErrorCode.INTERNAL_ERROR,
                "capture run disappeared after collector execution",
            )

        registrations: list[tuple[ArtifactRegistration, int]] = []
        adapter_extraction_rows: list[dict[str, object]] = []
        validation_status = ValidationStatus.NOT_REQUESTED
        validation_limitations: list[str] = []
        oracle_receipt_record: OracleReceiptRecord | None = None
        try:
            oracle = self.workloads.resolve_oracle(
                plan.workload_name,
                cast(dict[str, Scalar], plan.workload_instance.parameters),
            )
            if oracle is not None and outcome.process.exit_code == 0:
                try:
                    (
                        oracle_containment,
                        oracle_network_contained,
                        oracle_scope_unit,
                        oracle_argv,
                    ) = await self._contain(
                        oracle.command.argv,
                        cwd=Path(oracle.command.cwd),
                        writable=output_root,
                        writable_roots=plan.writable_roots,
                        unit_name=f"flameox-validation-{plan.plan_id[:21]}.scope",
                        required=ExecutionPolicy(plan.execution_policy).requires_containment(
                            self.workspace.config.execution.containment
                        ),
                    )
                    if plan.containment in {"active", "degraded"} and oracle_containment not in {
                        "active",
                        "degraded",
                    }:
                        raise DomainError(
                            ErrorCode.EXECUTION_REFUSED,
                            "The validation oracle cannot preserve the capture containment.",
                        )
                    if plan.network_contained and not oracle_network_contained:
                        raise DomainError(
                            ErrorCode.EXECUTION_REFUSED,
                            "The validation oracle cannot preserve capture network isolation.",
                        )
                    validation = await self.broker.run(
                        ExecutionRequest(
                            argv=oracle_argv,
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
                            systemd_scope_unit=oracle_scope_unit,
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
                                    "pass": ValidationStatus.PASSED,
                                    "fail": ValidationStatus.FAILED,
                                    "inconclusive": ValidationStatus.INCONCLUSIVE,
                                    "unsupported": ValidationStatus.UNSUPPORTED,
                                }[receipt.status]
                            elif receipt.status == "pass":
                                validation_limitations.append(
                                    "The oracle process failed despite claiming a passing receipt."
                                )
                        except DomainError as error:
                            validation_status = (
                                ValidationStatus.FAILED
                                if validation.process.exit_code != 0
                                else ValidationStatus.ERROR
                            )
                            validation_limitations.append(error.message)
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
                if not payload:
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
            if plan.adapter_execution_plan is not None:
                (
                    plugin_registrations,
                    plugin_extractions,
                    plugin_limitations,
                ) = await self._process_adapter_artifacts(plan, output_root)
                registrations.extend(plugin_registrations)
                adapter_extraction_rows.extend(plugin_extractions)
                validation_limitations.extend(plugin_limitations)
            else:
                native = output_root / self._native_filename(plan.adapter)
                if native.is_file():
                    registrations.append(
                        await self._register_path_async(
                            run_id,
                            native,
                            kind=plan.expected_artifact_kinds[0],
                            role="primary",
                            media_type=(
                                mimetypes.guess_type(native.name)[0] or "application/octet-stream"
                            ),
                            producer=plan.adapter,
                            producer_version=plan.adapter_version,
                        )
                    )
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
            if oracle_receipt_record is not None:
                by_role = {
                    registration.role: registration.artifact_id
                    for registration, _ in registrations
                }
                missing_roles = tuple(
                    role
                    for role in oracle_receipt_record.receipt.diagnostic_roles
                    if role not in by_role
                )
                oracle_receipt_record = oracle_receipt_record.model_copy(
                    update={
                        "diagnostic_artifact_ids": tuple(
                            by_role[role]
                            for role in oracle_receipt_record.receipt.diagnostic_roles
                            if role in by_role
                        ),
                        "parsing_limitations": (
                            ("Some diagnostic roles were not registered on this run: "
                             + ", ".join(missing_roles),)
                            if missing_roles
                            else ()
                        ),
                    }
                )
            await capture.report(7, "Artifacts registered")
        except asyncio.CancelledError as cancellation:
            try:
                await asyncio.shield(
                    capture.terminate(
                        execution=ExecutionStatus.CANCELLED,
                        message=("Capture cancelled during validation or artifact registration."),
                        phase="capture cancelled during validation or registration",
                        error_code="cancelled",
                    )
                )
            finally:
                raise cancellation
        except DomainError as error:
            terminal = await capture.terminate(
                execution=ExecutionStatus.FAILED,
                message=error.message,
                phase="validation or artifact registration failed",
                error_code=error.code.value,
            )
            error.run_id = terminal.run_id
            raise
        succeeded = outcome.process.exit_code == 0
        timed_out = outcome.process.timed_out
        terminal = running.model_copy(
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
                    if succeeded or (timed_out and registrations)
                    else CaptureStatus.FAILED
                ),
                "validation_status": validation_status,
                "process": outcome.process,
                "artifacts": tuple(registration for registration, _ in registrations),
                "oracle_receipt": oracle_receipt_record,
                "limitations": tuple(
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
            }
        )
        self.runs.append(terminal, expected_revision=running.revision)
        capture.run = terminal
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
            "runs": [run_row(terminal)],
            "artifact_registrations": [
                artifact_registration_row(registration, byte_length=byte_length)
                for registration, byte_length in registrations
            ],
            "adapter_extractions": adapter_extraction_rows,
            "environments": [environment_row(environment)],
            "source_states": [source_state_row(source_state)],
            "measurements": measurement_rows,
        }
        try:
            published = await run_atomic_thread(
                lambda: self.publisher.publish_rows(
                    publication_rows,
                    publisher="flameox.capture",
                    publisher_version="1",
                    input_run_ids=(run_id,),
                    input_artifact_ids=tuple(
                        registration.artifact_id for registration, _ in registrations
                    ),
                )
            )
        except asyncio.CancelledError as cancellation:
            await asyncio.shield(
                capture.terminate(
                    execution=ExecutionStatus.CANCELLED,
                    message="Capture cancelled during atomic evidence publication.",
                    phase="capture cancelled during evidence publication",
                    error_code="cancelled",
                    cleanup_complete=True,
                )
            )
            raise cancellation
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

    async def _adapter_command(
        self,
        adapter: str,
        workload: CommandSpec,
        output_root: Path,
    ) -> _AdapterBinding:
        adapter_definition = builtin_adapter(adapter)
        if adapter_definition is not None:
            capability = self.capabilities.get(adapter)
            if adapter != "command" and capability.status is not CapabilityStatus.AVAILABLE:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    f"Adapter {adapter!r} is unavailable.",
                    remediation=capability.remediation,
                )
            invocation = build_capture_invocation(
                adapter,
                workload.argv,
                output_root,
                executable=capability.executable,
            )
            return _AdapterBinding(
                argv=invocation.argv,
                artifact_kinds=invocation.artifact_kinds,
                expected_overhead=invocation.expected_overhead,
                limitations=invocation.limitations,
                permissions=adapter_definition.permissions,
                version=capability.version,
            )

        descriptor, contract = AdapterRegistry(self.workspace).load_contract(adapter)
        try:
            probe = AdapterProbeResult.model_validate(
                await contract.probe(
                    AdapterProbeContext(project_root=str(self.workspace.project_root))
                )
            )
            if probe.status == "unavailable":
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
    ) -> tuple[
        Literal["active", "degraded", "uncontained", "unavailable"],
        bool,
        str | None,
        tuple[str, ...],
    ]:
        if self.workspace.config.execution.containment == "disabled":
            if required:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "MCP capture requires containment but containment is disabled.",
                )
            return "uncontained", False, None, argv
        bwrap = shutil.which("bwrap") if os.name == "posix" else None
        if bwrap is None:
            if required:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "MCP capture requires Linux bubblewrap containment.",
                    remediation=("Install bubblewrap or change the trusted local policy.",),
                )
            return "unavailable", False, None, argv
        bwrap_argv = self._bubblewrap_argv(
            str(Path(bwrap).resolve()),
            argv,
            cwd=cwd,
            writable=writable,
            writable_roots=writable_roots,
        )
        systemd_run = shutil.which("systemd-run")
        if systemd_run is None or not await self._systemd_user_scope_available(systemd_run):
            if required:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "MCP capture requires a systemd user scope for descendant containment.",
                    remediation=(
                        "Run under a systemd user manager or use the trusted local policy.",
                    ),
                )
            return (
                "degraded",
                self.workspace.config.execution.network == "deny_when_contained",
                None,
                bwrap_argv,
            )
        wrapped = (
            str(Path(systemd_run).resolve()),
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
            self.workspace.config.execution.network == "deny_when_contained",
            unit_name,
            wrapped,
        )

    async def _systemd_user_scope_available(self, systemd_run: str) -> bool:
        true_executable = shutil.which("true")
        if true_executable is None:
            return False
        unit_name = f"flameox-probe-{secrets.token_hex(8)}.scope"
        try:
            outcome = await self.broker.run(
                ExecutionRequest(
                    argv=(
                        str(Path(systemd_run).resolve()),
                        "--user",
                        "--scope",
                        "--quiet",
                        "--collect",
                        "--expand-environment=no",
                        f"--unit={unit_name}",
                        "--",
                        str(Path(true_executable).resolve()),
                    ),
                    cwd=self.workspace.project_root,
                    environment_allowlist=(
                        self.workspace.config.execution.child_environment_allowlist
                    ),
                    allowed_working_roots=self._allowed_roots(),
                    timeout_seconds=5,
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
        if self.workspace.config.execution.network == "deny_when_contained":
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
            "definition_id": plan.workload_definition_id,
            "approval": plan.approval_digest,
            "instance": plan.workload_instance.model_dump(mode="json"),
            "adapter": plan.adapter,
            "adapter_version": plan.adapter_version,
            "adapter_execution_plan": plan.adapter_execution_plan,
            "execution_policy": plan.execution_policy,
            "collector_argv": plan.collector_argv,
            "collector_environment": plan.collector_environment,
            "bound_identities": plan.bound_identities,
            "preflight": plan.preflight.model_dump(mode="json"),
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
        ).plan(plan.workload_name)
        if current_execution_identity.identity_id != plan.planned_execution_identity.identity_id:
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "A declared module or native identity input changed after planning.",
            )
        if self.workspace.identity.workspace_id != plan.workspace_id:
            raise DomainError(ErrorCode.INVALID_CAPTURE_PLAN, "Workspace identity changed.")
        definition = self.workloads.definition(plan.workload_name)
        current_approval = (
            definition.approved_definition_digest or definition.workload_definition_id
        )
        if (
            definition.workload_definition_id != plan.workload_definition_id
            or current_approval != plan.approval_digest
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Workload definition or approval changed after planning.",
            )
        expected_collector = plan.bound_identities.get("collector_executable")
        expected_workload = plan.bound_identities.get("workload_executable")
        if (
            self._executable_identity(plan.collector_argv[0]) != expected_collector
            or self._executable_identity(plan.workload_instance.command.argv[0])
            != expected_workload
        ):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "A bound executable changed after planning.",
            )
        current_preflight = await PreflightService(
            self.workspace,
            capabilities=self.capabilities,
        ).inspect(plan.workload_name, mode=plan.preflight.mode)
        if (
            current_preflight.preflight_id != plan.preflight.preflight_id
            or current_preflight.disposition == "blocked"
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
    ) -> RunManifest:
        process = process or ProcessResult(
            timed_out=execution is ExecutionStatus.TIMED_OUT,
            cancellation_cause=(
                "caller_cancelled"
                if execution is ExecutionStatus.CANCELLED
                else ("timeout" if execution is ExecutionStatus.TIMED_OUT else "process_error")
            ),
            cleanup_complete=cleanup_complete,
        )
        terminal = running.model_copy(
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
                "limitations": (*running.limitations, message),
            }
        )
        terminal = self.runs.append(terminal, expected_revision=running.revision)
        self.publisher.publish_rows(
            {"runs": [run_row(terminal)]},
            publisher="flameox.capture",
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
        list[str],
    ]:
        execution_plan = AdapterExecutionPlan.model_validate(plan.adapter_execution_plan)
        descriptor, contract = AdapterRegistry(self.workspace).load_contract(plan.adapter)
        registrations: list[tuple[ArtifactRegistration, int]] = []
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
            try:
                validation = AdapterValidationResult.model_validate(
                    await contract.validate(str(resolved), declaration)
                )
            except Exception as error:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    f"Adapter {plan.adapter!r} artifact validation failed.",
                    details={"exception_type": type(error).__name__},
                    run_id=plan.run_id,
                ) from error
            if not validation.valid:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    f"Adapter {plan.adapter!r} rejected its declared artifact.",
                    details={"limitations": list(validation.limitations)},
                    run_id=plan.run_id,
                )
            limitations.extend(validation.limitations)
            registration, byte_length = await self._register_path_async(
                plan.run_id,
                resolved,
                kind=declaration.kind,
                role=declaration.role,
                media_type=declaration.media_type,
                producer=plan.adapter,
                producer_version=plan.adapter_version,
                sensitivity=declaration.sensitivity,
            )
            registrations.append((registration, byte_length))
            immutable = self.artifacts.get(registration.artifact_id)
            try:
                extraction = AdapterExtractionResult.model_validate(
                    await contract.extract(
                        str(immutable.payload_path),
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
        if plan.adapter == "pytest":
            sidecars, sidecar_limitations = await self._preserve_pytest_sidecars(
                plan,
                output_root,
                execution_plan,
            )
            registrations.extend(sidecars)
            limitations.extend(sidecar_limitations)
        return registrations, extractions, limitations

    async def _preserve_pytest_sidecars(
        self,
        plan: CapturePlan,
        output_root: Path,
        execution_plan: AdapterExecutionPlan,
    ) -> tuple[list[tuple[ArtifactRegistration, int]], list[str]]:
        registrations: list[tuple[ArtifactRegistration, int]] = []
        limitations: list[str] = []
        for declaration in execution_plan.artifacts:
            if declaration.kind is not ArtifactKind.TEST_EXECUTION:
                continue
            primary = output_root / declaration.relative_path
            for sidecar in sorted(primary.parent.glob(f"{primary.name}.*.worker")):
                if sidecar.is_symlink() or not sidecar.is_file():
                    limitations.append(
                        "An unrecovered pytest worker sidecar was not a regular file."
                    )
                    continue
                if sidecar.stat().st_size > _MAX_PYTEST_SIDECAR_BYTES:
                    limitations.append(
                        "An unrecovered pytest worker sidecar exceeded the preservation limit."
                    )
                    continue
                registrations.append(
                    await self._register_path_async(
                        plan.run_id,
                        sidecar,
                        kind=ArtifactKind.PROCESS_OUTPUT,
                        role="pytest_worker_sidecar_unrecovered",
                        media_type="application/x-ndjson",
                        producer=plan.adapter,
                        producer_version=plan.adapter_version,
                        sensitivity=declaration.sensitivity,
                    )
                )
        if registrations:
            limitations.append(
                "Unrecovered pytest worker sidecars were preserved as native artifacts."
            )
        return registrations, limitations

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
            display_name=path.name,
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
            )
        )

    def _allowed_roots(self) -> tuple[Path, ...]:
        roots = tuple(
            (self.workspace.paths.root / value).resolve()
            for value in self.workspace.config.execution.allowed_working_roots
        )
        return (*roots, self.workspace.project_root)

    def _native_filename(self, adapter: str) -> str:
        definition = builtin_adapter(adapter)
        return (
            definition.output_filename
            if definition is not None and definition.output_filename is not None
            else "capture.bin"
        )

    def _executable_identity(self, executable: str) -> dict[str, Any]:
        resolved = shutil.which(executable) if os.sep not in executable else executable
        if resolved is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Executable {executable!r} is unavailable.",
            )
        invoked_path = Path(resolved).absolute()
        path = invoked_path.resolve()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
        return {
            "invoked_path": str(invoked_path),
            "path": str(path),
            "sha256": digest.hexdigest(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "symlink_mtime_ns": invoked_path.lstat().st_mtime_ns,
        }

    def _lease(self, process_id: int) -> CaptureLease | None:
        observed = utc_now()
        boot_id_path = Path("/proc/sys/kernel/random/boot_id")
        stat_path = Path("/proc") / str(process_id) / "stat"
        try:
            boot_id = boot_id_path.read_text().strip()

            # Read /proc/[pid]/stat with a bounded low-level read instead of
            # read_text(), which streams the whole file and can block
            # indefinitely on a stuck /proc mount. The stat line is small
            # (well under 8 KiB), so a single bounded os.read is sufficient.
            # The process name (field 2, "comm") is parenthesized and may
            # contain spaces or parentheses (e.g. "Web Content"), so split
            # on the last ")" and index the remainder rather than the whole
            # line: starttime is field 22, i.e. index 19 in the slice that
            # starts at field 3.
            stat_fd = os.open(str(stat_path), os.O_RDONLY)
            try:
                stat_raw = os.read(stat_fd, 8192)
            finally:
                os.close(stat_fd)
            stat_text = stat_raw.decode("utf-8", errors="replace")
            comm_end = stat_text.rindex(")")
            stat_fields = stat_text[comm_end + 1 :].split()

            process_start_identity = stat_fields[19]
        except FileNotFoundError:
            return None
        except (OSError, IndexError, ValueError) as exc:
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
