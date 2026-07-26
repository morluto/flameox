from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
import secrets
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import JsonValue

from flamo.adapters.builtins import (
    build_capture_invocation,
    builtin_adapter,
)
from flamo.application.async_work import run_atomic_thread
from flamo.application.capabilities import CapabilityService
from flamo.application.environment import collect_environment
from flamo.application.evidence_rows import (
    artifact_registration_row,
    environment_row,
    source_state_row,
)
from flamo.application.execution_policy import ExecutionPolicy
from flamo.application.run_rows import run_row
from flamo.application.source import collect_source_state
from flamo.application.workloads import Scalar, WorkloadService
from flamo.domain import (
    ArtifactKind,
    ArtifactRegistration,
    CapabilityStatus,
    CaptureLease,
    CapturePlan,
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    OracleStrength,
    ProcessResult,
    RunManifest,
    RunType,
    Sensitivity,
    ValidationStatus,
    digest_model,
    new_id,
)
from flamo.domain.models import utc_now
from flamo.evidence import GenerationPublisher
from flamo.execution import ExecutionRequest, SubprocessBroker
from flamo.models import ContractModel
from flamo.observability import OperationLogger, elapsed_ms
from flamo.storage import ArtifactStore, RunStore, StorageQuota, Workspace
from flamo.storage.atomic import atomic_write_bytes


class CaptureResult(ContractModel):
    schema_version: int = 1
    plan: CapturePlan
    run: RunManifest
    corpus_commit_id: str


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
            raise RuntimeError("capture run is not initialized")
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
        await run_atomic_thread(
            lambda: self.service.runs.append(
                leased,
                expected_revision=current.revision,
            )
        )
        self.run = leased

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
    ) -> RunManifest:
        if self.run is None:
            raise RuntimeError("capture run is not initialized")
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
        try:
            terminal = await run_atomic_thread(
                lambda: self.service._finish_error(
                    current,
                    execution=execution,
                    message=message,
                    cleanup_complete=(
                        self.cleanup_complete if cleanup_complete is None else cleanup_complete
                    ),
                )
            )
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
        plan_id = secrets.token_hex(32)
        run_id = new_id()
        output_root = self.workspace.paths.staging / "captures" / plan_id
        collector_environment = {"FLAMO_OBSERVATIONS_PATH": str(output_root / "observations.jsonl")}
        collector_argv, kinds, overhead, warnings = self._adapter_command(
            adapter,
            instance.command.argv,
            output_root,
        )
        adapter_version = self.capabilities.get(adapter).version
        adapter_definition = builtin_adapter(adapter)
        containment, network_contained, systemd_scope_unit, collector_argv = await self._contain(
            collector_argv,
            cwd=Path(instance.command.cwd),
            writable=output_root,
            unit_name=f"flamo-capture-{plan_id[:24]}.scope",
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
            "execution_policy": execution_policy.value,
            "collector_argv": collector_argv,
            "collector_environment": collector_environment,
            "bound_identities": identities,
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
            execution_policy=execution_policy.value,
            collector_argv=collector_argv,
            collector_environment=collector_environment,
            expected_artifact_kinds=kinds,
            expected_overhead=overhead,
            containment=containment,
            network_contained=network_contained,
            systemd_scope_unit=systemd_scope_unit,
            permissions=(adapter_definition.permissions if adapter_definition is not None else ()),
            bound_identities=identities,
            limits={
                "timeout_seconds": instance.command.timeout_seconds,
                "max_output_bytes": self.workspace.config.execution.max_output_bytes,
                "max_artifact_bytes": self.workspace.config.capture.max_artifact_bytes,
                "max_cpu_percent": self.workspace.config.execution.max_cpu_percent,
                "max_memory_bytes": self.workspace.config.execution.max_memory_bytes,
                "max_processes": self.workspace.config.execution.max_processes,
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
        self._recheck(plan)
        StorageQuota(self.workspace).require_capacity(staging=True)
        output_root = self.workspace.paths.staging / "captures" / plan.plan_id
        output_root.mkdir(parents=True, exist_ok=False)
        capture = _CaptureExecution(
            service=self,
            plan=plan,
            output_root=output_root,
            logger=logger,
            operation_id=operation_id,
            started_monotonic=started,
            progress=progress,
        )
        environment = collect_environment()
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
        )
        self.runs.create(initial)
        capture.run = initial
        try:
            await capture.report(1, "Capture plan validated")
            await capture.report(2, "Run lifecycle initialized")
            source_state = await collect_source_state(
                self.workspace,
                workload_executable=plan.workload_instance.command.argv[0],
                broker=self.broker,
            )
            await capture.report(3, "Source and environment identity collected")
        except asyncio.CancelledError as cancellation:
            try:
                await capture.terminate(
                    execution=ExecutionStatus.CANCELLED,
                    message="Capture cancelled while collecting source identity.",
                    phase="capture cancelled during source identity",
                    error_code="cancelled",
                )
            finally:
                raise cancellation
        prepared = initial.model_copy(
            update={
                "revision": 1,
                "source_state_id": source_state.source_state_id,
            }
        )
        self.runs.append(prepared, expected_revision=0)
        running = initial.model_copy(
            update={
                "revision": 2,
                "started_at": utc_now(),
                "execution_status": ExecutionStatus.RUNNING,
                "capture_status": CaptureStatus.RUNNING,
                "source_state_id": source_state.source_state_id,
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
                ),
                on_started=capture.record_lease,
                on_cleanup=capture.record_cleanup,
            )
            StorageQuota(self.workspace).require_capacity(staging=True)
            await capture.report(5, "Collector execution complete")
        except asyncio.CancelledError as cancellation:
            try:
                await capture.terminate(
                    execution=ExecutionStatus.CANCELLED,
                    message="Capture cancelled by caller after bounded cleanup.",
                    phase="capture cancelled during collector execution",
                    error_code="cancelled",
                )
            finally:
                raise cancellation
        except DomainError as error:
            status = (
                ExecutionStatus.TIMED_OUT
                if error.code is ErrorCode.PROCESS_TIMEOUT
                else ExecutionStatus.FAILED
            )
            terminal = await capture.terminate(
                execution=status,
                message=error.message,
                phase="collector execution failed",
                error_code=error.code.value,
            )
            error.run_id = terminal.run_id
            raise
        finally:
            if acquired_slot:
                self.plans.release_capture_slot()
        running = capture.run
        if running is None:
            raise RuntimeError("capture run disappeared after collector execution")

        registrations: list[tuple[ArtifactRegistration, int]] = []
        validation_status = ValidationStatus.NOT_REQUESTED
        validation_limitations: list[str] = []
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
                        unit_name=f"flamo-validation-{plan.plan_id[:21]}.scope",
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
                            allowed_working_roots=self._allowed_roots(),
                            timeout_seconds=oracle.command.timeout_seconds,
                            max_output_bytes=(self.workspace.config.execution.max_output_bytes),
                            systemd_scope_unit=oracle_scope_unit,
                        ),
                        on_cleanup=capture.record_cleanup,
                    )
                    validation_status = (
                        ValidationStatus.PASSED
                        if validation.process.exit_code == 0
                        else ValidationStatus.FAILED
                    )
                    validation_output = output_root / "validation.stdout"
                    atomic_write_bytes(validation_output, validation.stdout)
                    role = (
                        "validation_cross_treatment_equivalence"
                        if oracle.strength is OracleStrength.CROSS_TREATMENT_EQUIVALENCE
                        else f"validation_{oracle.strength.value}"
                    )
                    registrations.append(
                        await self._register_path_async(
                            run_id,
                            validation_output,
                            kind=ArtifactKind.VALIDATION_OUTPUT,
                            role=role,
                            media_type="application/octet-stream",
                        )
                    )
                    if validation.stderr:
                        validation_stderr = output_root / "validation.stderr"
                        atomic_write_bytes(validation_stderr, validation.stderr)
                        registrations.append(
                            await self._register_path_async(
                                run_id,
                                validation_stderr,
                                kind=ArtifactKind.PROCESS_OUTPUT,
                                role="validation_stderr",
                                media_type="application/octet-stream",
                            )
                        )
                    if validation_status is not ValidationStatus.PASSED:
                        validation_limitations.append(
                            "The declared validation oracle exited unsuccessfully."
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
            await capture.report(7, "Artifacts registered")
        except asyncio.CancelledError as cancellation:
            try:
                await capture.terminate(
                    execution=ExecutionStatus.CANCELLED,
                    message=("Capture cancelled during validation or artifact registration."),
                    phase="capture cancelled during validation or registration",
                    error_code="cancelled",
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
        terminal = running.model_copy(
            update={
                "revision": running.revision + 1,
                "finished_at": utc_now(),
                "execution_status": (
                    ExecutionStatus.SUCCEEDED if succeeded else ExecutionStatus.FAILED
                ),
                "capture_status": (CaptureStatus.REGISTERED if succeeded else CaptureStatus.FAILED),
                "validation_status": validation_status,
                "process": outcome.process,
                "artifacts": tuple(registration for registration, _ in registrations),
                "limitations": tuple(
                    (
                        []
                        if succeeded
                        else [f"Collector exited with status {outcome.process.exit_code}."]
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
            "environments": [environment_row(environment)],
            "source_states": [source_state_row(source_state)],
            "measurements": measurement_rows,
        }
        try:
            published = await run_atomic_thread(
                lambda: self.publisher.publish_rows(
                    publication_rows,
                    publisher="flamo.capture",
                    publisher_version="1",
                    input_run_ids=(run_id,),
                    input_artifact_ids=tuple(
                        registration.artifact_id for registration, _ in registrations
                    ),
                )
            )
        except asyncio.CancelledError as cancellation:
            await capture.terminate(
                execution=ExecutionStatus.CANCELLED,
                message="Capture cancelled during atomic evidence publication.",
                phase="capture cancelled during evidence publication",
                error_code="cancelled",
                cleanup_complete=True,
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

    def _adapter_command(
        self,
        adapter: str,
        workload_argv: tuple[str, ...],
        output_root: Path,
    ) -> tuple[tuple[str, ...], tuple[ArtifactKind, ...], str, tuple[str, ...]]:
        capability = self.capabilities.get(adapter)
        if adapter != "command" and capability.status is not CapabilityStatus.AVAILABLE:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Adapter {adapter!r} is unavailable.",
                remediation=capability.remediation,
            )
        invocation = build_capture_invocation(
            adapter,
            workload_argv,
            output_root,
            executable=capability.executable,
        )
        return (
            invocation.argv,
            invocation.artifact_kinds,
            invocation.expected_overhead,
            invocation.limitations,
        )

    async def _contain(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        writable: Path,
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
        unit_name = f"flamo-probe-{secrets.token_hex(8)}.scope"
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

    def _recheck(self, plan: CapturePlan) -> None:
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

    def _finish_error(
        self,
        running: RunManifest,
        *,
        execution: ExecutionStatus,
        message: str,
        cleanup_complete: bool | None = True,
    ) -> RunManifest:
        process = ProcessResult(
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
                "limitations": (message,),
            }
        )
        terminal = self.runs.append(terminal, expected_revision=running.revision)
        self.publisher.publish_rows(
            {"runs": [run_row(terminal)]},
            publisher="flamo.capture",
            publisher_version="1",
            input_run_ids=(terminal.run_id,),
        )
        return terminal

    def _register_path(
        self,
        run_id: str,
        path: Path,
        *,
        kind: ArtifactKind,
        role: str,
        media_type: str,
        producer: str = "flamo.capture",
        producer_version: str | None = None,
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
            sensitivity=Sensitivity.INTERNAL,
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
        producer: str = "flamo.capture",
        producer_version: str | None = None,
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
            stat_fields = stat_path.read_text().split()
            process_start_identity = stat_fields[21]
        except FileNotFoundError:
            return None
        except (OSError, IndexError) as exc:
            raise DomainError(
                ErrorCode.PROCESS_FAILED,
                "Could not establish the child process lease identity.",
            ) from exc
        return CaptureLease(
            process_id=process_id,
            process_start_identity=process_start_identity,
            boot_id=boot_id,
            heartbeat_monotonic_ns=time.monotonic_ns(),
            observed_at=observed,
            expires_at=observed + timedelta(seconds=60),
        )
