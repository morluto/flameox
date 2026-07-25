from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import secrets
import shutil
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue

from flamo.application.capabilities import CapabilityService
from flamo.application.environment import collect_environment
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
from flamo.storage import ArtifactStore, RunStore, Workspace
from flamo.storage.atomic import atomic_write_bytes


class CaptureResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    plan: CapturePlan
    run: RunManifest
    corpus_commit_id: str


@dataclass(slots=True)
class _PlanEntry:
    plan: CapturePlan
    expires_monotonic: float
    consumed: bool = False


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
        containment, network_contained, collector_argv = self._contain(
            collector_argv,
            cwd=Path(instance.command.cwd),
            writable=output_root,
            required=(
                execution_policy.requires_containment(
                    self.workspace.config.execution.containment
                )
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
            "execution_policy": execution_policy.value,
            "collector_argv": collector_argv,
            "collector_environment": collector_environment,
            "bound_identities": identities,
            "policy": self.workspace.config.model_dump(mode="json"),
            "containment": containment,
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
            execution_policy=execution_policy.value,
            collector_argv=collector_argv,
            collector_environment=collector_environment,
            expected_artifact_kinds=kinds,
            expected_overhead=overhead,
            containment=containment,
            network_contained=network_contained,
            permissions=("ptrace",) if adapter == "py-spy" else (),
            bound_identities=identities,
            limits={
                "timeout_seconds": instance.command.timeout_seconds,
                "max_output_bytes": self.workspace.config.execution.max_output_bytes,
                "max_artifact_bytes": self.workspace.config.capture.max_artifact_bytes,
            },
            warnings=warnings,
            created_at=created_at,
            expires_at=created_at + timedelta(seconds=self.plans.ttl_seconds),
        )
        await self.plans.issue(plan)
        return plan

    async def execute(self, plan_id: str) -> CaptureResult:
        plan = await self.plans.consume(plan_id)
        self._recheck(plan)
        output_root = self.workspace.paths.staging / "captures" / plan.plan_id
        output_root.mkdir(parents=True, exist_ok=False)
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
                    "collector_argv": plan.collector_argv,
                }
            ),
            environment_id=environment.environment_id,
            source_state_id=None,
            collector=plan.adapter,
            collector_version=plan.adapter_version,
            command=plan.workload_instance.command,
        )
        self.runs.create(initial)
        try:
            source_state = await collect_source_state(
                self.workspace,
                workload_executable=plan.workload_instance.command.argv[0],
                broker=self.broker,
            )
        except asyncio.CancelledError as cancellation:
            try:
                self._finish_error(
                    initial,
                    execution=ExecutionStatus.CANCELLED,
                    message="Capture cancelled while collecting source identity.",
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
        acquired_slot = False

        async def record_lease(process_id: int) -> None:
            nonlocal running
            lease = self._lease(process_id)
            leased = running.model_copy(
                update={
                    "revision": running.revision + 1,
                    "lease": lease,
                }
            )
            self.runs.append(leased, expected_revision=running.revision)
            running = leased

        try:
            await self.plans.acquire_capture_slot()
            acquired_slot = True
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
                ),
                on_started=record_lease,
            )
        except asyncio.CancelledError as cancellation:
            try:
                self._finish_error(
                    running,
                    execution=ExecutionStatus.CANCELLED,
                    message="Capture cancelled by caller after contained cleanup.",
                )
            finally:
                raise cancellation
        except DomainError as error:
            status = (
                ExecutionStatus.TIMED_OUT
                if error.code is ErrorCode.PROCESS_TIMEOUT
                else ExecutionStatus.FAILED
            )
            terminal = self._finish_error(running, execution=status, message=error.message)
            error.run_id = terminal.run_id
            raise
        finally:
            if acquired_slot:
                self.plans.release_capture_slot()

        registrations: list[tuple[ArtifactRegistration, int]] = []
        validation_status = ValidationStatus.NOT_REQUESTED
        validation_limitations: list[str] = []
        oracle = self.workloads.resolve_oracle(
            plan.workload_name,
            cast(dict[str, Scalar], plan.workload_instance.parameters),
        )
        if oracle is not None and outcome.process.exit_code == 0:
            try:
                validation = await self.broker.run(
                    ExecutionRequest(
                        argv=oracle.command.argv,
                        cwd=Path(oracle.command.cwd),
                        environment_allowlist=(
                            self.workspace.config.execution.child_environment_allowlist
                        ),
                        allowed_working_roots=self._allowed_roots(),
                        timeout_seconds=oracle.command.timeout_seconds,
                        max_output_bytes=(self.workspace.config.execution.max_output_bytes),
                    )
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
                    self._register_path(
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
                        self._register_path(
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
            if payload:
                path = output_root / name
                atomic_write_bytes(path, payload)
                registrations.append(
                    self._register_path(
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
                self._register_path(
                    run_id,
                    native,
                    kind=plan.expected_artifact_kinds[0],
                    role="primary",
                    media_type=mimetypes.guess_type(native.name)[0] or "application/octet-stream",
                )
            )
        observations = output_root / "observations.jsonl"
        if observations.is_file():
            registrations.append(
                self._register_path(
                    run_id,
                    observations,
                    kind=ArtifactKind.SEMANTIC_OBSERVATIONS,
                    role="semantic_observations",
                    media_type="application/x-ndjson",
                )
            )
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
        published = self.publisher.publish_rows(
            {
                "runs": [run_row(terminal)],
                "artifact_registrations": [
                    {
                        **registration.model_dump(mode="python"),
                        "kind": registration.kind.value,
                        "sensitivity": registration.sensitivity.value,
                        "byte_length": byte_length,
                    }
                    for registration, byte_length in registrations
                ],
                "environments": [
                    {
                        "environment_id": environment.environment_id,
                        "observed_at": environment.observed_at,
                        "identity_quality": environment.identity_quality.value,
                        "fields_json": json.dumps(
                            environment.fields,
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        "missing_fields": list(environment.missing_fields),
                    }
                ],
                "source_states": [
                    {
                        "source_state_id": source_state.source_state_id,
                        "identity_quality": source_state.identity_quality.value,
                        "repository_root": source_state.repository_root,
                        "head_commit": source_state.head_commit,
                        "diff_digest": source_state.diff_digest,
                        "executable_digest": source_state.executable_digest,
                        "build_id": source_state.build_id,
                        "fields_json": json.dumps(
                            source_state.fields,
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        "missing_fields": list(source_state.missing_fields),
                    }
                ],
                "measurements": measurement_rows,
            },
            publisher="flamo.capture",
            publisher_version="1",
            input_run_ids=(run_id,),
            input_artifact_ids=tuple(registration.artifact_id for registration, _ in registrations),
        )
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
        if adapter == "command":
            return (
                workload_argv,
                (ArtifactKind.PROCESS_OUTPUT,),
                "No profiler overhead; process output only.",
                ("No sampled stack or operator evidence is collected.",),
            )
        capability = self.capabilities.get(adapter)
        if capability.status is not CapabilityStatus.AVAILABLE:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                f"Adapter {adapter!r} is unavailable.",
                remediation=capability.remediation,
            )
        if adapter == "py-spy":
            assert capability.executable is not None
            return (
                (
                    capability.executable,
                    "record",
                    "--format",
                    "chrometrace",
                    "--output",
                    str(output_root / self._native_filename(adapter)),
                    "--",
                    *workload_argv,
                ),
                (ArtifactKind.SAMPLE_PROFILE,),
                "Sampling overhead; exact rate and native-frame scope are collector defaults.",
                ("GIL, native-frame, idle-thread, and subprocess modes are disabled.",),
            )
        if adapter == "perf":
            assert capability.executable is not None
            return (
                (
                    capability.executable,
                    "record",
                    "-o",
                    str(output_root / self._native_filename(adapter)),
                    "--",
                    *workload_argv,
                ),
                (ArtifactKind.SAMPLE_PROFILE,),
                "Kernel sampling overhead; symbol coverage depends on build identities.",
                (),
            )
        if adapter in {"coverage", "memray"}:
            python, target = self._python_target(workload_argv)
            output = str(output_root / self._native_filename(adapter))
            if adapter == "coverage":
                prefix = (
                    python,
                    "-m",
                    "coverage",
                    "run",
                    "--branch",
                    f"--data-file={output}",
                )
                return (
                    (*prefix, *target),
                    (ArtifactKind.EXECUTION_COVERAGE,),
                    "Tracing overhead; branch collection is enabled.",
                    ("Coverage records execution, not values or control-flow causes.",),
                )
            prefix = (
                python,
                "-m",
                "memray",
                "run",
                "--output",
                output,
            )
            return (
                (*prefix, *target),
                (ArtifactKind.MEMORY_PROFILE,),
                "Allocation tracing overhead; native traces are disabled by default.",
                (
                    "The capture records the main process unless follow-fork is "
                    "explicitly added in a future plan mode.",
                ),
            )
        if adapter == "torch.profiler":
            python, target = self._python_target(workload_argv)
            output = str(output_root / self._native_filename(adapter))
            if target[0] == "-m":
                if len(target) < 2:
                    raise DomainError(
                        ErrorCode.INVALID_CAPTURE_PLAN,
                        "The Python module name is missing.",
                    )
                launcher_target = ("--module", target[1], *target[2:])
            else:
                launcher_target = ("--script", target[0], *target[1:])
            return (
                (
                    python,
                    "-m",
                    "flamo.collectors.torch_launcher",
                    "--output",
                    output,
                    *launcher_target,
                ),
                (ArtifactKind.EXECUTION_TRACE,),
                "Operator tracing with shapes, memory, and Python stacks has substantial overhead.",
                (
                    "Whole-entrypoint mode cannot distinguish application-specific "
                    "warm-up and steady-state phases.",
                    "FLOP estimates and module hierarchy are not enabled.",
                ),
            )
        if adapter == "pyperf":
            python, _ = self._python_target(workload_argv)
            output = str(output_root / self._native_filename(adapter))
            return (
                (
                    python,
                    "-m",
                    "pyperf",
                    "command",
                    "--output",
                    output,
                    "--processes",
                    "3",
                    "--values",
                    "3",
                    "--warmups",
                    "1",
                    "--name",
                    "workload",
                    *workload_argv,
                ),
                (ArtifactKind.BENCHMARK_SAMPLES,),
                "Repeated calibrated process execution; three workers, three values, "
                "and one warm-up per worker.",
                (
                    "Experiment-level treatment randomization is separate from "
                    "pyperf's worker hierarchy.",
                ),
            )
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            f"Adapter {adapter!r} does not yet support capture planning.",
        )

    def _contain(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        writable: Path,
        required: bool,
    ) -> tuple[
        Literal["active", "uncontained", "unavailable"],
        bool,
        tuple[str, ...],
    ]:
        if self.workspace.config.execution.containment == "disabled":
            if required:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "MCP capture requires containment but containment is disabled.",
                )
            return "uncontained", False, argv
        bwrap = shutil.which("bwrap") if os.name == "posix" else None
        if bwrap is None:
            if required:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "MCP capture requires Linux bubblewrap containment.",
                    remediation=("Install bubblewrap or change the trusted local policy.",),
                )
            return "unavailable", False, argv
        wrapped = (
            str(Path(bwrap).resolve()),
            "--die-with-parent",
            "--new-session",
            "--unshare-pid",
            "--unshare-net",
            "--ro-bind",
            "/",
            "/",
            "--bind",
            str(writable),
            str(writable),
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--chdir",
            str(cwd),
            "--",
            *argv,
        )
        return "active", True, wrapped

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
    ) -> RunManifest:
        process = ProcessResult(
            timed_out=execution is ExecutionStatus.TIMED_OUT,
            cancellation_cause=(
                "caller_cancelled"
                if execution is ExecutionStatus.CANCELLED
                else ("timeout" if execution is ExecutionStatus.TIMED_OUT else "process_error")
            ),
            cleanup_complete=True,
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
            producer="flamo.capture",
            sensitivity=Sensitivity.INTERNAL,
        )
        return registration, stored.content.byte_length

    def _allowed_roots(self) -> tuple[Path, ...]:
        roots = tuple(
            (self.workspace.paths.root / value).resolve()
            for value in self.workspace.config.execution.allowed_working_roots
        )
        return (*roots, self.workspace.project_root)

    def _native_filename(self, adapter: str) -> str:
        return {
            "coverage": ".coverage",
            "memray": "memory.bin",
            "py-spy": "profile.json",
            "perf": "perf.data",
            "pyperf": "benchmark.json",
            "torch.profiler": "torch-trace.json",
        }.get(adapter, "capture.bin")

    def _python_target(
        self,
        workload_argv: tuple[str, ...],
    ) -> tuple[str, tuple[str, ...]]:
        executable = Path(workload_argv[0]).name
        if not executable.startswith("python"):
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "This adapter can launch only a declared Python script or module.",
            )
        arguments = workload_argv[1:]
        if not arguments or arguments[0] == "-c":
            raise DomainError(
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Inline Python commands cannot be wrapped by this adapter.",
                remediation=("Declare a script or `python -m module` workload.",),
            )
        return workload_argv[0], arguments

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

    def _lease(self, process_id: int) -> CaptureLease:
        observed = utc_now()
        boot_id_path = Path("/proc/sys/kernel/random/boot_id")
        stat_path = Path("/proc") / str(process_id) / "stat"
        try:
            boot_id = boot_id_path.read_text().strip()
            stat_fields = stat_path.read_text().split()
            process_start_identity = stat_fields[21]
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
