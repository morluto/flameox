from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, computed_field, model_validator

from flameox.action_graph import ActionId
from flameox.adapters.artifact_workers import IsolatedWorkerHarness
from flameox.application.operations import OperationAdapter, OperationRunner, OperationState
from flameox.application.provider_runtime import ProviderRuntime, ProviderRuntimeManager
from flameox.application.reduction_contracts import (
    PredicateClassification,
    PredicateObservation,
    ReductionAttemptReceipt,
    ReductionDisposition,
    ReductionFormat,
    ReductionMinimality,
    collapse_predicate_observations,
)
from flameox.application.staging_ownership import StagingOwnershipService
from flameox.application.task_supervisor import TaskSupervisor
from flameox.application.workloads import WorkloadService
from flameox.atomic import atomic_write_bytes
from flameox.command_binding import ExecutableResolver
from flameox.domain import CapabilityExtra, DomainError, ErrorCode, digest_model
from flameox.domain.executables import ResolvedExecutable
from flameox.domain.models import CommandSpec, Digest, utc_now
from flameox.evidence import GenerationPublisher
from flameox.execution import ExecutionRequest, ResourcePolicy, SubprocessBroker
from flameox.filesystem import BoundedFileSystem
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, ControlRecordStore, StorageQuota, Workspace
from flameox.workers.reduction_contract import (
    SHRINKRAY_PROFILE,
    SHRINKRAY_REQUIREMENT,
    SHRINKRAY_VERSION,
    SHRINKRAY_WORKER,
    ReductionPredicateConfig,
    ShrinkRayWorkerRequest,
    ShrinkRayWorkerResult,
)


class ReductionLimits(ContractModel):
    max_attempts: Annotated[int, Field(ge=1, le=100_000)] = 1_000
    wall_time_seconds: Annotated[float, Field(gt=0, le=86_400)] = 900
    predicate_timeout_seconds: Annotated[float, Field(gt=0, le=3_600)] = 30
    predicate_repetitions: Annotated[int, Field(ge=1, le=20)] = 1
    parallelism: Literal[1] = 1
    max_staging_bytes: Annotated[int, Field(gt=0)] | None = None
    max_retained_candidate_bytes: Annotated[int, Field(gt=0)] | None = None
    max_staging_files: Annotated[int, Field(ge=8, le=100_000)] = 2_048


class ReductionExecutionLimits(ReductionLimits):
    max_staging_bytes: Annotated[int, Field(gt=0)]
    max_retained_candidate_bytes: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def budgets_are_coherent(self) -> ReductionExecutionLimits:
        if self.max_retained_candidate_bytes > self.max_staging_bytes:
            raise ValueError("retained-candidate bytes cannot exceed staging bytes")
        if self.max_attempts + 16 > self.max_staging_files:
            raise ValueError("attempt receipts do not fit the staging-file budget")
        return self


class PlanReductionRequest(ContractModel):
    original_artifact_id: str
    predicate_workload: str
    input_format: ReductionFormat = ReductionFormat.BINARY
    predicate_parameters: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        max_length=128,
    )
    limits: ReductionLimits = Field(default_factory=ReductionLimits)


class ReductionPlan(ContractModel):
    schema_version: Literal[5] = 5
    plan_id: Digest
    workspace_id: str
    original_artifact_id: str
    engine: Literal["shrinkray"] = "shrinkray"
    input_format: ReductionFormat
    predicate_workload: str
    predicate_definition_id: str
    predicate_instance_id: str
    predicate_command: CommandSpec
    predicate_executable_binding: ResolvedExecutable
    predicate_parameters: dict[str, str | int | float | bool]
    predicate_executable_digest: str
    provider_environment_id: str
    provider_python_digest: str
    shrinkray_version: Literal["26.7.8.0"] = SHRINKRAY_VERSION
    shrinkray_profile: Literal["flameox.shrinkray.offline-v1"] = SHRINKRAY_PROFILE
    shrinkray_executable_binding: ResolvedExecutable
    shrinkray_executable_digest: str
    predicate_bridge_binding: ResolvedExecutable
    predicate_bridge_digest: str
    limits: ReductionExecutionLimits
    created_at: datetime = Field(default_factory=utc_now)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def request_digest(self) -> Digest:
        return self.plan_id


class ReductionAttemptSummary(ContractModel):
    attempted: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    unresolved: int = Field(ge=0)
    contradictory: int = Field(ge=0)
    timed_out: int = Field(ge=0)


class ReductionRepeatabilityStatus(StrEnum):
    NOT_ASSESSED = "not_assessed"
    CONSISTENT = "consistent"
    VIOLATED = "violated"


class ReductionResult(ContractModel):
    schema_version: Literal[4] = 4
    reduction_id: str
    plan_id: str
    disposition: ReductionDisposition
    original_artifact_id: str
    final_artifact_id: str | None = None
    best_known_artifact_id: str | None = None
    predicate_definition_id: str
    predicate_instance_id: str
    attempts: ReductionAttemptSummary
    attempt_receipts_artifact_id: str | None = None
    shrinkray_history_artifact_id: str | None = None
    reducer_stdout_artifact_id: str | None = None
    reducer_stderr_artifact_id: str | None = None
    final_predicate_stdout_artifact_id: str | None = None
    final_predicate_stderr_artifact_id: str | None = None
    cleanup_complete: bool
    limitations: tuple[str, ...] = ()
    engine: Literal["shrinkray"] = "shrinkray"
    input_format: ReductionFormat
    provider_environment_id: str
    provider_python_digest: str
    shrinkray_version: Literal["26.7.8.0"] = SHRINKRAY_VERSION
    shrinkray_profile: Literal["flameox.shrinkray.offline-v1"] = SHRINKRAY_PROFILE
    shrinkray_executable_digest: str
    predicate_bridge_digest: str
    original_size_bytes: int = Field(ge=0)
    final_size_bytes: int | None = Field(default=None, ge=0)
    minimality: Literal[ReductionMinimality.NOT_CLAIMED] = ReductionMinimality.NOT_CLAIMED
    final_revalidation_status: PredicateClassification
    predicate_repetitions: Annotated[int, Field(ge=1, le=20)]
    predicate_collapse_rule: Literal["unanimous_v1"] = "unanimous_v1"
    repeatability_status: ReductionRepeatabilityStatus
    budget_exhausted: bool = False
    staging_byte_limit: Annotated[int, Field(gt=0)]
    retained_candidate_byte_limit: Annotated[int, Field(gt=0)]
    finished_at: datetime = Field(default_factory=utc_now)


class _ConsumedReduction(ContractModel):
    worker: ShrinkRayWorkerResult
    attempts: tuple[ReductionAttemptReceipt, ...]
    best_known_artifact_id: str
    final_artifact_id: str | None
    final_classification: PredicateClassification
    final_observations: tuple[PredicateObservation, ...]
    attempt_receipts_artifact_id: str
    history_artifact_id: str | None
    reducer_stdout_artifact_id: str
    reducer_stderr_artifact_id: str
    predicate_stdout_artifact_id: str
    predicate_stderr_artifact_id: str


class ReductionService:
    """Coordinate one exact ShrinkRay provider around a declared predicate."""

    _OPERATION = OperationAdapter(
        kind="reduction.execute",
        start_action=ActionId.EXECUTE_REDUCTION,
        status_action=ActionId.GET_REDUCTION,
        status_identifier="reduction_id",
        retry_with_idempotency_key=False,
        recover_unmanaged=True,
    )

    def __init__(
        self,
        workspace: Workspace,
        *,
        supervisor: TaskSupervisor | None = None,
        provider_runtime: ProviderRuntime | None = None,
    ) -> None:
        self.workspace = workspace
        self.artifacts = ArtifactStore(workspace)
        self.workloads = WorkloadService(workspace)
        self.broker = SubprocessBroker()
        self.provider_runtimes = ProviderRuntimeManager(
            workspace.paths.records / "provider-runtimes"
        )
        self._provided_runtime = provider_runtime
        self.plans: ControlRecordStore[ReductionPlan] = ControlRecordStore(
            workspace,
            kind="reduction_plans",
            model=ReductionPlan,
            id_field="plan_id",
            output_only_fields={"request_digest"},
        )
        self.results = ControlRecordStore(
            workspace,
            kind="reduction_results",
            model=ReductionResult,
            id_field="reduction_id",
        )
        self.publisher = GenerationPublisher(workspace)
        self.runner = OperationRunner(workspace, self._OPERATION, supervisor=supervisor)

    def plan(self, request: PlanReductionRequest) -> ReductionPlan:
        original = self.artifacts.get(request.original_artifact_id)
        limits = self._bind_limits(request.limits, original_size=original.content.byte_length)
        predicate = self.workloads.resolve(
            request.predicate_workload,
            request.predicate_parameters,
        )
        runtime, shrinkray, bridge = self._runtime_bindings()
        bound = {
            "schema_version": 5,
            "workspace_id": self.workspace.identity.workspace_id,
            **request.model_dump(mode="json", exclude={"limits"}),
            "limits": limits.model_dump(mode="json"),
            "predicate_definition_id": predicate.workload_definition_id,
            "predicate_instance_id": predicate.workload_instance_id,
            "predicate_command": predicate.command.model_dump(mode="json"),
            "predicate_executable_binding": predicate.executable_binding.model_dump(mode="json"),
            "provider_environment_id": runtime.receipt.environment_id,
            "provider_python_digest": runtime.receipt.python_sha256,
            "shrinkray_version": SHRINKRAY_VERSION,
            "shrinkray_profile": SHRINKRAY_PROFILE,
            "shrinkray_executable_binding": shrinkray.model_dump(mode="json"),
            "predicate_bridge_binding": bridge.model_dump(mode="json"),
        }
        plan = ReductionPlan(
            plan_id=digest_model(bound),
            workspace_id=self.workspace.identity.workspace_id,
            original_artifact_id=request.original_artifact_id,
            input_format=request.input_format,
            predicate_workload=request.predicate_workload,
            predicate_definition_id=predicate.workload_definition_id,
            predicate_instance_id=predicate.workload_instance_id,
            predicate_command=predicate.command,
            predicate_executable_binding=predicate.executable_binding,
            predicate_parameters=request.predicate_parameters,
            predicate_executable_digest=predicate.executable_binding.identity.sha256,
            provider_environment_id=runtime.receipt.environment_id,
            provider_python_digest=runtime.receipt.python_sha256,
            shrinkray_executable_binding=shrinkray,
            shrinkray_executable_digest=shrinkray.identity.sha256,
            predicate_bridge_binding=bridge,
            predicate_bridge_digest=bridge.identity.sha256,
            limits=limits,
        )
        try:
            return self.plans.create(plan)
        except DomainError as error:
            if error.code is ErrorCode.REVISION_CONFLICT:
                return self.plans.read(plan.plan_id)
            raise

    async def execute(self, plan_id: str) -> ReductionResult:
        plan = self.plans.read(plan_id)
        runtime = self._revalidate(plan)
        reduction_id = digest_model({"plan_id": plan.plan_id, "contract": "reduction-v4"})
        try:
            return self.results.read(reduction_id)
        except DomainError as error:
            if error.code is not ErrorCode.WORKSPACE_INVALID:
                raise

        async def run(operation_id: str, progress: object) -> dict[str, object]:
            del progress
            result = await self._execute_shrinkray(
                plan,
                runtime,
                reduction_id=reduction_id,
                operation_id=operation_id,
            )
            return {"result": result.model_dump(mode="json")}

        prior_operation = self.runner.store.find_subject(
            operation=self._OPERATION.kind,
            subject_id=reduction_id,
        )
        operation = await self.runner.start(
            {"plan_id": plan.plan_id, "reduction_id": reduction_id},
            reduction_id,
            run,
            subject_id=reduction_id,
        )
        if (
            prior_operation is not None
            and prior_operation.state in {OperationState.FAILED, OperationState.CANCELLED}
            and operation.operation_id == prior_operation.operation_id
        ):
            operation = await self.runner.retry_terminal(operation.operation_id, run)
        operation = await self.runner.wait(
            operation.operation_id,
            timeout_seconds=plan.limits.wall_time_seconds
            + plan.limits.predicate_timeout_seconds * plan.limits.predicate_repetitions
            + 20,
        )
        receipt = operation.terminal_receipt
        if operation.state is OperationState.TERMINAL and isinstance(receipt, dict):
            try:
                return ReductionResult.model_validate(receipt.get("result"))
            except ValueError as error:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "The durable reduction receipt is invalid.",
                    details={"operation_id": operation.operation_id},
                ) from error
        if operation.state in {OperationState.STARTING, OperationState.RUNNING}:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Another owner is still executing this reduction plan.",
                retryable=True,
                details={"operation_id": operation.operation_id},
            )
        if operation.state is OperationState.UNMANAGED_AFTER_RESTART:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "The prior reduction owner is not yet stale enough for recovery.",
                retryable=True,
                details={"operation_id": operation.operation_id},
            )
        raise DomainError(
            ErrorCode.INTERNAL_ERROR,
            operation.failure_message or "Reduction execution did not produce a terminal result.",
            details={"operation_id": operation.operation_id},
        )

    def get(self, reduction_id: str) -> ReductionResult:
        return self.results.read(reduction_id)

    async def _execute_shrinkray(
        self,
        plan: ReductionPlan,
        runtime: ProviderRuntime,
        *,
        reduction_id: str,
        operation_id: str,
    ) -> ReductionResult:
        root = self.workspace.paths.staging / (
            f"reduction-{reduction_id.removeprefix('sha256:')[:16]}-"
            f"{operation_id.removeprefix('op-')[:12]}-{self.runner.owner_id[:12]}"
        )
        ownership = StagingOwnershipService(self.workspace).acquire(
            root,
            owner_kind="reduction",
            owner_id=operation_id,
        )
        original = self.artifacts.get(plan.original_artifact_id)
        request = ShrinkRayWorkerRequest(
            artifact_path=str(original.payload_path),
            shrinkray_executable=str(plan.shrinkray_executable_binding.invocation_path),
            shrinkray_executable_binding=plan.shrinkray_executable_binding,
            predicate_bridge_executable=str(plan.predicate_bridge_binding.invocation_path),
            predicate_bridge_binding=plan.predicate_bridge_binding,
            predicate_config=ReductionPredicateConfig(
                operation_root=str(root),
                receipt_root=str(root),
                counter_path=str(root / "counter"),
                deadline_monotonic=time.monotonic() + plan.limits.wall_time_seconds,
                predicate_command=plan.predicate_command,
                predicate_executable_binding=plan.predicate_executable_binding,
                predicate_repetitions=plan.limits.predicate_repetitions,
                predicate_timeout_seconds=plan.limits.predicate_timeout_seconds,
                max_attempts=plan.limits.max_attempts,
                max_candidate_bytes=plan.limits.max_retained_candidate_bytes,
                max_output_bytes=self.workspace.config.execution.max_output_bytes,
                project_root=str(self.workspace.project_root),
                workspace_root=str(self.workspace.paths.root),
                staging_root=str(self.workspace.paths.staging),
                minimum_free_bytes=self.workspace.config.storage.min_free_bytes,
                maximum_rss_bytes=self.workspace.config.execution.max_memory_bytes,
                sampling_interval_ms=(
                    self.workspace.config.execution.resource_sampling_interval_ms
                ),
                max_observed_files=(self.workspace.config.execution.max_resource_observed_files),
            ),
            input_format=plan.input_format,
            wall_time_seconds=plan.limits.wall_time_seconds,
            max_staging_bytes=plan.limits.max_staging_bytes,
            max_staging_files=plan.limits.max_staging_files,
        )
        StorageQuota(self.workspace).require_capacity(additional_bytes=4096, staging=True)
        try:
            consumed = await IsolatedWorkerHarness(
                self.workspace,
                broker=self.broker,
                python=runtime.python,
            ).run_typed_session(
                SHRINKRAY_WORKER,
                request,
                timeout_seconds=plan.limits.wall_time_seconds + 10,
                job_root=root,
                consume=lambda worker, worker_root: self._consume_worker_result(
                    plan,
                    worker,
                    worker_root,
                    original_artifact_id=original.content.artifact_id,
                    original_size=original.content.byte_length,
                ),
            )
            return self._publish_result(
                plan,
                reduction_id,
                consumed,
                cleanup_complete=not root.exists(),
            )
        except asyncio.CancelledError:
            raise
        finally:
            shutil.rmtree(root, ignore_errors=True)
            ownership.release()
            ownership.forget_if_removed(root)

    def _consume_worker_result(
        self,
        plan: ReductionPlan,
        worker: ShrinkRayWorkerResult,
        root: Path,
        *,
        original_artifact_id: str,
        original_size: int,
    ) -> _ConsumedReduction:
        harness = IsolatedWorkerHarness(self.workspace, broker=self.broker)
        if (
            worker.original_sha256 != original_artifact_id
            or worker.original_size_bytes != original_size
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "ShrinkRay worker described another original artifact.",
            )
        final_path = harness.validate_output_file(root, worker.final_candidate)
        attempts_path = harness.validate_output_file(root, worker.attempt_receipts)
        attempts = self._read_attempts(
            attempts_path,
            maximum=plan.limits.max_attempts,
            max_bytes=plan.limits.max_staging_bytes,
        )
        classification, observations, predicate_stdout, predicate_stderr = self._evaluate_candidate(
            plan, final_path
        )
        best = self.artifacts.import_path(
            final_path,
            allowed_roots=(root,),
            max_bytes=plan.limits.max_retained_candidate_bytes,
            expected_artifact_id=worker.final_candidate.sha256,
            expected_byte_length=worker.final_candidate.byte_length,
        ).content.artifact_id
        final_id = (
            best
            if classification is PredicateClassification.INTERESTING and worker.tool_completed
            else None
        )
        attempts_id = self.artifacts.import_path(
            attempts_path,
            allowed_roots=(root,),
            max_bytes=plan.limits.max_staging_bytes,
            expected_artifact_id=worker.attempt_receipts.sha256,
            expected_byte_length=worker.attempt_receipts.byte_length,
        ).content.artifact_id
        history_id = None
        if worker.history is not None:
            history_path = harness.validate_output_file(root, worker.history)
            history_id = self.artifacts.import_path(
                history_path,
                allowed_roots=(root,),
                max_bytes=plan.limits.max_staging_bytes,
                expected_artifact_id=worker.history.sha256,
                expected_byte_length=worker.history.byte_length,
            ).content.artifact_id
        reducer_stdout = self._import_declared(root, worker.stdout, harness)
        reducer_stderr = self._import_declared(root, worker.stderr, harness)
        predicate_stdout_path = root / "final-predicate.stdout"
        predicate_stderr_path = root / "final-predicate.stderr"
        atomic_write_bytes(predicate_stdout_path, predicate_stdout)
        atomic_write_bytes(predicate_stderr_path, predicate_stderr)
        predicate_stdout_id = self.artifacts.import_path(
            predicate_stdout_path,
            allowed_roots=(root,),
            max_bytes=self.workspace.config.execution.max_output_bytes,
        ).content.artifact_id
        predicate_stderr_id = self.artifacts.import_path(
            predicate_stderr_path,
            allowed_roots=(root,),
            max_bytes=self.workspace.config.execution.max_output_bytes,
        ).content.artifact_id
        return _ConsumedReduction(
            worker=worker,
            attempts=attempts,
            best_known_artifact_id=best,
            final_artifact_id=final_id,
            final_classification=classification,
            final_observations=observations,
            attempt_receipts_artifact_id=attempts_id,
            history_artifact_id=history_id,
            reducer_stdout_artifact_id=reducer_stdout,
            reducer_stderr_artifact_id=reducer_stderr,
            predicate_stdout_artifact_id=predicate_stdout_id,
            predicate_stderr_artifact_id=predicate_stderr_id,
        )

    def _evaluate_candidate(
        self,
        plan: ReductionPlan,
        candidate: Path,
    ) -> tuple[PredicateClassification, tuple[PredicateObservation, ...], bytes, bytes]:
        observations: list[PredicateObservation] = []
        stdout = b""
        stderr = b""
        for repetition in range(plan.limits.predicate_repetitions):
            started = time.monotonic()
            try:
                outcome = self.broker.run_sync(
                    ExecutionRequest(
                        argv=plan.predicate_command.argv,
                        executable_binding=plan.predicate_executable_binding,
                        cwd=Path(plan.predicate_command.cwd),
                        environment_allowlist=("PATH",),
                        environment_overrides={
                            **plan.predicate_command.env_overrides,
                            "FLAMEOX_REDUCTION_CANDIDATE": str(candidate),
                        },
                        allowed_working_roots=(self.workspace.project_root,),
                        timeout_seconds=plan.limits.predicate_timeout_seconds,
                        max_output_bytes=self.workspace.config.execution.max_output_bytes,
                        resource_policy=ResourcePolicy(
                            filesystem_path=self.workspace.paths.root,
                            staging_root=self.workspace.paths.staging,
                            writable_roots=(candidate.parent,),
                            minimum_free_bytes=self.workspace.config.storage.min_free_bytes,
                            maximum_rss_bytes=self.workspace.config.execution.max_memory_bytes,
                            sampling_interval_ms=(
                                self.workspace.config.execution.resource_sampling_interval_ms
                            ),
                            max_observed_files=(
                                self.workspace.config.execution.max_resource_observed_files
                            ),
                        ),
                    )
                )
                stdout, stderr = outcome.stdout, outcome.stderr
                observations.append(
                    PredicateObservation(
                        repetition=repetition,
                        classification=(
                            PredicateClassification.INTERESTING
                            if outcome.process.exit_code == 0
                            else PredicateClassification.NOT_INTERESTING
                        ),
                        exit_code=outcome.process.exit_code,
                        duration_ms=(time.monotonic() - started) * 1_000,
                    )
                )
            except DomainError as error:
                observations.append(
                    PredicateObservation(
                        repetition=repetition,
                        classification=PredicateClassification.UNRESOLVED,
                        failure_category=error.code.value,
                        duration_ms=(time.monotonic() - started) * 1_000,
                    )
                )
        frozen = tuple(observations)
        return (
            collapse_predicate_observations(tuple(item.classification for item in frozen)),
            frozen,
            stdout,
            stderr,
        )

    def _publish_result(
        self,
        plan: ReductionPlan,
        reduction_id: str,
        consumed: _ConsumedReduction,
        *,
        cleanup_complete: bool,
    ) -> ReductionResult:
        worker = consumed.worker
        classification = consumed.final_classification
        if classification is PredicateClassification.INTERESTING and worker.tool_completed:
            disposition = (
                ReductionDisposition.UNCHANGED
                if consumed.best_known_artifact_id == plan.original_artifact_id
                else ReductionDisposition.SUCCEEDED
            )
        elif worker.disposition is ReductionDisposition.ORIGINAL_NOT_INTERESTING:
            disposition = ReductionDisposition.ORIGINAL_NOT_INTERESTING
        else:
            disposition = ReductionDisposition.INCONCLUSIVE
        contradictory = worker.contradictory + (
            1 if len({item.classification for item in consumed.final_observations}) > 1 else 0
        )
        attempts = ReductionAttemptSummary(
            attempted=worker.attempted,
            passed=worker.passed,
            failed=worker.failed,
            unresolved=worker.unresolved,
            contradictory=contradictory,
            timed_out=worker.timed_out,
        )
        limitations = list(worker.limitations)
        if classification is not PredicateClassification.INTERESTING:
            limitations.append("Independent final predicate revalidation was not conclusive.")
        result = ReductionResult(
            reduction_id=reduction_id,
            plan_id=plan.plan_id,
            disposition=disposition,
            original_artifact_id=plan.original_artifact_id,
            final_artifact_id=consumed.final_artifact_id,
            best_known_artifact_id=consumed.best_known_artifact_id,
            predicate_definition_id=plan.predicate_definition_id,
            predicate_instance_id=plan.predicate_instance_id,
            attempts=attempts,
            attempt_receipts_artifact_id=consumed.attempt_receipts_artifact_id,
            shrinkray_history_artifact_id=consumed.history_artifact_id,
            reducer_stdout_artifact_id=consumed.reducer_stdout_artifact_id,
            reducer_stderr_artifact_id=consumed.reducer_stderr_artifact_id,
            final_predicate_stdout_artifact_id=consumed.predicate_stdout_artifact_id,
            final_predicate_stderr_artifact_id=consumed.predicate_stderr_artifact_id,
            cleanup_complete=cleanup_complete,
            limitations=tuple(dict.fromkeys(limitations)),
            input_format=plan.input_format,
            provider_environment_id=plan.provider_environment_id,
            provider_python_digest=plan.provider_python_digest,
            shrinkray_executable_digest=plan.shrinkray_executable_digest,
            predicate_bridge_digest=plan.predicate_bridge_digest,
            original_size_bytes=worker.original_size_bytes,
            final_size_bytes=worker.final_size_bytes,
            final_revalidation_status=classification,
            predicate_repetitions=plan.limits.predicate_repetitions,
            repeatability_status=(
                ReductionRepeatabilityStatus.NOT_ASSESSED
                if plan.limits.predicate_repetitions < 2
                else (
                    ReductionRepeatabilityStatus.VIOLATED
                    if contradictory
                    else ReductionRepeatabilityStatus.CONSISTENT
                )
            ),
            budget_exhausted=worker.budget_exhausted,
            staging_byte_limit=plan.limits.max_staging_bytes,
            retained_candidate_byte_limit=plan.limits.max_retained_candidate_bytes,
        )
        try:
            created = self.results.create(result)
        except DomainError as error:
            if error.code is ErrorCode.REVISION_CONFLICT:
                return self.results.read(reduction_id)
            raise
        self._publish_rows(created, consumed.attempts)
        return created

    def _publish_rows(
        self,
        result: ReductionResult,
        attempts: tuple[ReductionAttemptReceipt, ...],
    ) -> None:
        rows: dict[str, list[dict[str, object]]] = {
            "reduction_results": [
                {
                    "reduction_id": result.reduction_id,
                    "plan_id": result.plan_id,
                    "disposition": result.disposition,
                    "original_artifact_id": result.original_artifact_id,
                    "final_artifact_id": result.final_artifact_id,
                    "predicate_definition_id": result.predicate_definition_id,
                    "predicate_instance_id": result.predicate_instance_id,
                    "attempts_json": result.attempts.model_dump_json(),
                    "attempt_receipts_artifact_id": result.attempt_receipts_artifact_id,
                    "shrinkray_history_artifact_id": result.shrinkray_history_artifact_id,
                    "reducer_stdout_artifact_id": result.reducer_stdout_artifact_id,
                    "reducer_stderr_artifact_id": result.reducer_stderr_artifact_id,
                    "final_predicate_stdout_artifact_id": (
                        result.final_predicate_stdout_artifact_id
                    ),
                    "final_predicate_stderr_artifact_id": (
                        result.final_predicate_stderr_artifact_id
                    ),
                    "cleanup_complete": result.cleanup_complete,
                    "limitations": list(result.limitations),
                    "finished_at": result.finished_at,
                    "engine": result.engine,
                    "input_format": result.input_format,
                    "provider_environment_id": result.provider_environment_id,
                    "provider_python_digest": result.provider_python_digest,
                    "shrinkray_version": result.shrinkray_version,
                    "shrinkray_profile": result.shrinkray_profile,
                    "shrinkray_executable_digest": result.shrinkray_executable_digest,
                    "predicate_bridge_digest": result.predicate_bridge_digest,
                    "original_size_bytes": result.original_size_bytes,
                    "final_size_bytes": result.final_size_bytes,
                    "minimality": result.minimality,
                    "best_known_artifact_id": result.best_known_artifact_id,
                    "final_revalidation_status": result.final_revalidation_status,
                    "predicate_repetitions": result.predicate_repetitions,
                    "predicate_collapse_rule": result.predicate_collapse_rule,
                    "repeatability_status": result.repeatability_status,
                    "budget_exhausted": result.budget_exhausted,
                    "staging_byte_limit": result.staging_byte_limit,
                    "retained_candidate_byte_limit": result.retained_candidate_byte_limit,
                }
            ],
            "reduction_attempts": [
                {
                    "reduction_id": result.reduction_id,
                    "attempt_id": attempt.attempt_id,
                    "candidate_sha256": attempt.candidate_sha256,
                    "candidate_size_bytes": attempt.candidate_size_bytes,
                    "observations_json": json.dumps(
                        [item.model_dump(mode="json") for item in attempt.observations],
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "classification": attempt.classification,
                    "recorded_at": attempt.recorded_at,
                }
                for attempt in attempts
            ],
        }
        self.publisher.publish_rows(
            rows,
            publisher="flameox.reductions",
            publisher_version="3",
            input_artifact_ids=tuple(
                artifact_id
                for artifact_id in (
                    result.original_artifact_id,
                    result.final_artifact_id,
                    result.best_known_artifact_id,
                    result.attempt_receipts_artifact_id,
                    result.shrinkray_history_artifact_id,
                    result.reducer_stdout_artifact_id,
                    result.reducer_stderr_artifact_id,
                    result.final_predicate_stdout_artifact_id,
                    result.final_predicate_stderr_artifact_id,
                )
                if artifact_id is not None
            ),
        )

    def _runtime_bindings(
        self,
    ) -> tuple[ProviderRuntime, ResolvedExecutable, ResolvedExecutable]:
        runtime = self._provided_runtime or self.provider_runtimes.find(
            extra=CapabilityExtra.REDUCTION,
            requirement=SHRINKRAY_REQUIREMENT,
        )
        if runtime is None or runtime.executable is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "A verified ShrinkRay 26.7.8.0 provider environment is required.",
                remediation=(
                    "Call start_capability_setup with adapter='shrinkray', then plan the "
                    "reduction again.",
                ),
            )
        bridge_path = runtime.executable.parent / "flameox-reduction-predicate"
        if os.name == "nt":
            bridge_path = bridge_path.with_suffix(".exe")
        resolver = ExecutableResolver()
        shrinkray = resolver.require_host_tool(str(runtime.executable), cwd=runtime.root)
        bridge = resolver.require_host_tool(str(bridge_path), cwd=runtime.root)
        if runtime.receipt.distributions.get("shrinkray") != SHRINKRAY_VERSION:
            raise DomainError(
                ErrorCode.ADAPTER_INCOMPATIBLE,
                "The managed reduction provider is not the qualified ShrinkRay release.",
            )
        return runtime, shrinkray, bridge

    def _revalidate(self, plan: ReductionPlan) -> ProviderRuntime:
        if plan.workspace_id != self.workspace.identity.workspace_id:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Reduction plan is for another workspace.",
            )
        self.artifacts.get(plan.original_artifact_id)
        predicate = self.workloads.resolve(plan.predicate_workload, plan.predicate_parameters)
        runtime = (
            self._provided_runtime
            if self._provided_runtime is not None
            and self._provided_runtime.receipt.environment_id == plan.provider_environment_id
            else self.provider_runtimes.get(plan.provider_environment_id)
        )
        if runtime is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "The exact ShrinkRay provider bound during planning is unavailable.",
            )
        current_runtime, shrinkray, bridge = self._runtime_bindings_for(runtime)
        if (
            predicate.workload_definition_id != plan.predicate_definition_id
            or predicate.workload_instance_id != plan.predicate_instance_id
            or predicate.command != plan.predicate_command
            or predicate.executable_binding != plan.predicate_executable_binding
            or predicate.executable_binding.identity.sha256 != plan.predicate_executable_digest
            or current_runtime.receipt.python_sha256 != plan.provider_python_digest
            or shrinkray != plan.shrinkray_executable_binding
            or shrinkray.identity.sha256 != plan.shrinkray_executable_digest
            or bridge != plan.predicate_bridge_binding
            or bridge.identity.sha256 != plan.predicate_bridge_digest
        ):
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Reduction authority changed after planning.",
            )
        return current_runtime

    def _runtime_bindings_for(
        self,
        runtime: ProviderRuntime,
    ) -> tuple[ProviderRuntime, ResolvedExecutable, ResolvedExecutable]:
        previous = self._provided_runtime
        try:
            self._provided_runtime = runtime
            return self._runtime_bindings()
        finally:
            self._provided_runtime = previous

    def _bind_limits(
        self,
        requested: ReductionLimits,
        *,
        original_size: int,
    ) -> ReductionExecutionLimits:
        storage_limit = self.workspace.config.storage.max_staging_bytes
        staging_limit = requested.max_staging_bytes or storage_limit
        retained_limit = requested.max_retained_candidate_bytes or min(
            staging_limit,
            self.workspace.config.capture.max_artifact_bytes,
        )
        if staging_limit > storage_limit or retained_limit > staging_limit:
            raise DomainError(
                ErrorCode.STORAGE_QUOTA_EXCEEDED,
                "Reduction byte budgets exceed workspace staging policy.",
            )
        if original_size > retained_limit:
            raise DomainError(
                ErrorCode.STORAGE_QUOTA_EXCEEDED,
                "The original artifact exceeds the reduction candidate budget.",
            )
        observed_file_limit = self.workspace.config.execution.max_resource_observed_files
        if requested.max_staging_files > observed_file_limit:
            raise DomainError(
                ErrorCode.STORAGE_QUOTA_EXCEEDED,
                "Reduction file budget exceeds workspace observation policy.",
            )
        try:
            return ReductionExecutionLimits(
                **requested.model_dump(
                    mode="python",
                    exclude={"max_staging_bytes", "max_retained_candidate_bytes"},
                ),
                max_staging_bytes=staging_limit,
                max_retained_candidate_bytes=retained_limit,
            )
        except ValueError as error:
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "Reduction attempt receipts do not fit the declared staging-file budget.",
            ) from error

    def _read_attempts(
        self,
        path: Path,
        *,
        maximum: int,
        max_bytes: int,
    ) -> tuple[ReductionAttemptReceipt, ...]:
        raw = BoundedFileSystem((path.parent,)).read_bytes(
            path,
            max_bytes=max_bytes,
            require_single_link=True,
        )
        attempts: list[ReductionAttemptReceipt] = []
        for line in raw.splitlines():
            if len(attempts) >= maximum or len(line) > 64 * 1024:
                raise DomainError(
                    ErrorCode.QUERY_BUDGET_EXCEEDED,
                    "Predicate attempt evidence exceeds its declared bounds.",
                )
            attempts.append(ReductionAttemptReceipt.model_validate_json(line))
        for index, attempt in enumerate(attempts):
            if attempt.attempt_id != f"attempt-{index:08d}":
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "Predicate attempt evidence is not contiguous.",
                )
        return tuple(attempts)

    def _import_declared(
        self,
        root: Path,
        output: object,
        harness: IsolatedWorkerHarness,
    ) -> str:
        from flameox.workers.protocol import WorkerOutputFile

        if not isinstance(output, WorkerOutputFile):
            raise DomainError(ErrorCode.ARTIFACT_PARSE_FAILED, "Worker output is undeclared.")
        path = harness.validate_output_file(root, output)
        return self.artifacts.import_path(
            path,
            allowed_roots=(root,),
            max_bytes=self.workspace.config.execution.max_output_bytes,
            expected_artifact_id=output.sha256,
            expected_byte_length=output.byte_length,
        ).content.artifact_id
