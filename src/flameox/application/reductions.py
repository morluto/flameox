from __future__ import annotations

import asyncio
import hashlib
import shutil
import threading
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from flameox.application.native_reducer import (
    NativeDdminReducer,
    NativePredicateClassification,
    NativeReductionLimits,
    NativeReductionResult,
)
from flameox.application.workloads import WorkloadService
from flameox.domain import DomainError, ErrorCode, digest_model
from flameox.domain.models import CommandSpec, utc_now
from flameox.evidence import GenerationPublisher
from flameox.execution import ExecutionOutcome, ExecutionRequest, ResourcePolicy, SubprocessBroker
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, JsonRecordStore, Workspace

ReductionDisposition = Literal["succeeded", "unchanged", "inconclusive", "original_not_interesting"]
ReductionPartitioner = Literal[
    "text_lines",
    "binary_chunks",
    "json_top_level",
    "jsonl_records",
    "otlp_spans",
    "chrome_trace_events",
]


class ReductionLimits(ContractModel):
    max_attempts: Annotated[int, Field(ge=1, le=100_000)] = 1_000
    wall_time_seconds: Annotated[float, Field(gt=0, le=86_400)] = 900
    predicate_timeout_seconds: Annotated[float, Field(gt=0, le=3_600)] = 30
    predicate_repetitions: Annotated[int, Field(ge=1, le=20)] = 1


class PlanReductionRequest(ContractModel):
    original_artifact_id: str
    engine: Literal["native_ddmin"] = "native_ddmin"
    partitioner: ReductionPartitioner
    chunk_size: int | None = Field(default=None, ge=1, le=16 * 1024 * 1024)
    predicate_workload: str
    predicate_parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    limits: ReductionLimits = Field(default_factory=ReductionLimits)
    expected_determinism: Literal["deterministic", "repeated"] = "deterministic"

    @model_validator(mode="after")
    def validate_chunk_size(self) -> PlanReductionRequest:
        if self.partitioner == "binary_chunks" and self.chunk_size is None:
            raise ValueError("binary_chunks reductions require chunk_size")
        if self.partitioner != "binary_chunks" and self.chunk_size is not None:
            raise ValueError("chunk_size is only valid for binary_chunks reductions")
        return self


class ReductionPlan(ContractModel):
    schema_version: Literal[2] = 2
    plan_id: str
    request_digest: str
    workspace_id: str
    original_artifact_id: str
    engine: Literal["native_ddmin"] = "native_ddmin"
    partitioner: ReductionPartitioner
    chunk_size: int | None = None
    predicate_workload: str
    predicate_definition_id: str
    predicate_instance_id: str
    predicate_command: CommandSpec
    predicate_parameters: dict[str, str | int | float | bool]
    predicate_executable_digest: str
    limits: ReductionLimits
    expected_determinism: Literal["deterministic", "repeated"]
    created_at: datetime = Field(default_factory=utc_now)


class ReductionAttemptSummary(ContractModel):
    attempted: int
    passed: int
    failed: int
    contradictory: int
    timed_out: int


class ReductionResult(ContractModel):
    schema_version: Literal[2] = 2
    reduction_id: str
    plan_id: str
    disposition: ReductionDisposition
    original_artifact_id: str
    final_artifact_id: str | None = None
    predicate_definition_id: str
    predicate_instance_id: str
    attempts: ReductionAttemptSummary
    predicate_stdout_artifact_id: str | None = None
    predicate_stderr_artifact_id: str | None = None
    cleanup_complete: bool
    limitations: tuple[str, ...] = ()
    engine: Literal["native_ddmin"] = "native_ddmin"
    partitioner: ReductionPartitioner
    original_unit_count: int | None = None
    final_unit_count: int | None = None
    minimality: Literal["one_minimal", "not_claimed", "partitioner_incompatible"] | None = None
    best_known_artifact_id: str | None = None
    final_revalidation_status: str | None = None
    budget_exhausted: bool = False
    finished_at: datetime = Field(default_factory=utc_now)


class ReductionService:
    """Coordinate a declared reducer around a bounded predicate."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.artifacts = ArtifactStore(workspace)
        self.workloads = WorkloadService(workspace)
        self.broker = SubprocessBroker()
        self.plans = JsonRecordStore(
            workspace,
            kind="reduction_plans",
            model=ReductionPlan,
            id_field="plan_id",
        )
        self.results = JsonRecordStore(
            workspace,
            kind="reduction_results",
            model=ReductionResult,
            id_field="reduction_id",
        )
        self.publisher = GenerationPublisher(workspace)

    def plan(self, request: PlanReductionRequest) -> ReductionPlan:
        self.artifacts.get(request.original_artifact_id)
        predicate = self.workloads.resolve(
            request.predicate_workload,
            request.predicate_parameters,
        )
        bound = {
            "workspace_id": self.workspace.identity.workspace_id,
            **request.model_dump(mode="json"),
            "predicate_definition_id": predicate.workload_definition_id,
            "predicate_instance_id": predicate.workload_instance_id,
            "predicate_command": predicate.command.model_dump(mode="json"),
            "predicate_executable_digest": _executable_digest(predicate.command),
        }
        request_digest = digest_model(bound)
        plan = ReductionPlan(
            schema_version=2,
            plan_id=request_digest,
            request_digest=request_digest,
            workspace_id=self.workspace.identity.workspace_id,
            original_artifact_id=request.original_artifact_id,
            engine="native_ddmin",
            partitioner=request.partitioner,
            chunk_size=request.chunk_size,
            predicate_workload=request.predicate_workload,
            predicate_definition_id=predicate.workload_definition_id,
            predicate_instance_id=predicate.workload_instance_id,
            predicate_command=predicate.command,
            predicate_parameters=request.predicate_parameters,
            predicate_executable_digest=_executable_digest(predicate.command),
            limits=request.limits,
            expected_determinism=request.expected_determinism,
        )
        return self._create_plan(plan)

    def _create_plan(self, plan: ReductionPlan) -> ReductionPlan:
        try:
            return self.plans.create(plan)
        except DomainError as error:
            if error.code is ErrorCode.REVISION_CONFLICT:
                return self.plans.read(plan.plan_id)
            raise

    async def execute(self, plan_id: str) -> ReductionResult:
        plan = self.plans.read(plan_id)
        self._revalidate(plan)
        return await self._execute_native(plan)

    def get(self, reduction_id: str) -> ReductionResult:
        return self.results.read(reduction_id)

    async def _wait_for_result(
        self,
        reduction_id: str,
        root: Path,
        *,
        timeout_seconds: float,
    ) -> ReductionResult:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            try:
                return self.results.read(reduction_id)
            except DomainError as error:
                if error.code is not ErrorCode.WORKSPACE_INVALID:
                    raise
            await asyncio.sleep(0.1)
        shutil.rmtree(root, ignore_errors=True)
        raise DomainError(
            ErrorCode.REVISION_CONFLICT,
            "Another execution of this reduction plan did not reach a terminal result.",
            retryable=True,
        )

    async def _execute_native(  # noqa: C901 - coordinates bounded predicate lifecycle and publication
        self, plan: ReductionPlan
    ) -> ReductionResult:
        reduction_id = digest_model({"plan_id": plan.plan_id, "contract": "reduction-v2"})
        try:
            return self.results.read(reduction_id)
        except DomainError as error:
            if error.code is not ErrorCode.WORKSPACE_INVALID:
                raise
        root = self.workspace.paths.staging / f"reduction-{reduction_id}"
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            return await self._wait_for_result(
                reduction_id,
                root,
                timeout_seconds=(
                    plan.limits.wall_time_seconds + plan.limits.predicate_timeout_seconds + 10
                ),
            )
        candidate = root / "candidate"
        original = self.artifacts.get(plan.original_artifact_id).payload_path.read_bytes()
        latest_output: tuple[bytes, bytes] | None = None
        cancellation = threading.Event()
        failure_state: list[str | None] = [None]
        wall_deadline = time.monotonic() + plan.limits.wall_time_seconds

        def predicate(payload: bytes) -> NativePredicateClassification:
            nonlocal latest_output
            if cancellation.is_set():
                failure_state[0] = "cancelled"
                return "unresolved"
            candidate.write_bytes(payload)
            remaining = wall_deadline - time.monotonic()
            if remaining <= 0:
                failure_state[0] = "reduction_wall_time"
                return "unresolved"
            timeout = min(plan.limits.predicate_timeout_seconds, remaining)

            async def run_predicate() -> ExecutionOutcome | None:
                task = asyncio.create_task(
                    self.broker.run(
                        self._execution_request(
                            plan.predicate_command,
                            timeout=timeout,
                            writable_root=root,
                            overrides={"FLAMEOX_REDUCTION_CANDIDATE": str(candidate)},
                        )
                    )
                )
                while not task.done():
                    if cancellation.is_set():
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                        failure_state[0] = "cancelled"
                        return None
                    await asyncio.sleep(0.01)
                return await task

            try:
                outcome = asyncio.run(run_predicate())
            except DomainError as error:
                failure_state[0] = error.code.value
                return "unresolved"
            if outcome is None:
                return "unresolved"
            failure_state[0] = None
            latest_output = (outcome.stdout, outcome.stderr)
            return "interesting" if outcome.process.exit_code == 0 else "not_interesting"

        try:
            reducer = NativeDdminReducer(
                plan.partitioner,
                chunk_size=plan.chunk_size,
                limits=NativeReductionLimits(
                    max_attempts=plan.limits.max_attempts,
                    wall_time_seconds=plan.limits.wall_time_seconds,
                    repetitions=plan.limits.predicate_repetitions,
                ),
            )
            worker = asyncio.create_task(
                asyncio.to_thread(
                    reducer.reduce,
                    original,
                    predicate,
                    failure_detail=lambda: failure_state[0],
                )
            )
            cancelled = False
            try:
                native = await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
                cancellation.set()
                native = await asyncio.shield(worker)
                native = native.model_copy(
                    update={
                        "limitations": (*native.limitations, "reduction_cancelled"),
                    }
                )
            accepted_ids: list[str] = []
            for index, payload in enumerate(native.accepted_best_payloads):
                path = root / f"best-{index:08d}"
                path.write_bytes(payload)
                stored = self.artifacts.import_path(
                    path,
                    allowed_roots=(root,),
                    max_bytes=self.workspace.config.capture.max_artifact_bytes,
                )
                if stored.content.artifact_id not in accepted_ids:
                    accepted_ids.append(stored.content.artifact_id)
            final_path = root / "final-candidate"
            final_path.write_bytes(
                native.final_payload if native.final_payload is not None else original
            )
            final_stored = self.artifacts.import_path(
                final_path,
                allowed_roots=(root,),
                max_bytes=self.workspace.config.capture.max_artifact_bytes,
            )
            stdout_id = stderr_id = None
            if latest_output is not None:
                stdout_id = self._preserve_output(root, "predicate.stdout", latest_output[0])
                stderr_id = self._preserve_output(root, "predicate.stderr", latest_output[1])
            summary = ReductionAttemptSummary(
                attempted=len(native.attempts),
                passed=sum(item.classification == "interesting" for item in native.attempts),
                failed=sum(item.classification == "not_interesting" for item in native.attempts),
                contradictory=sum(
                    item.classification == "unresolved"
                    and len(item.predicate_outcomes) > 1
                    and len(set(item.predicate_outcomes)) > 1
                    for item in native.attempts
                ),
                timed_out=sum(
                    item.classification == "unresolved"
                    and item.failure
                    in {
                        ErrorCode.PROCESS_TIMEOUT.value,
                        "reduction_wall_time",
                    }
                    for item in native.attempts
                ),
            )
            result = self._publish_native_result(
                plan,
                reduction_id,
                native,
                summary,
                final_artifact_id=final_stored.content.artifact_id,
                best_known_artifact_id=(
                    accepted_ids[-1] if accepted_ids else plan.original_artifact_id
                ),
                stdout_id=stdout_id,
                stderr_id=stderr_id,
                cleanup_complete=self._cleanup(root),
            )
            if cancelled:
                raise asyncio.CancelledError
            return result
        except asyncio.CancelledError:
            raise
        except DomainError as error:
            return self._publish_native_result(
                plan,
                reduction_id,
                NativeReductionResult(
                    disposition="inconclusive",
                    original_digest=f"sha256:{hashlib.sha256(original).hexdigest()}",
                    final_digest=f"sha256:{hashlib.sha256(original).hexdigest()}",
                    original_unit_count=0,
                    final_unit_count=0,
                    minimality="not_claimed",
                    final_revalidation="unresolved",
                    limitations=(error.message,),
                ),
                ReductionAttemptSummary(
                    attempted=0,
                    passed=0,
                    failed=0,
                    contradictory=0,
                    timed_out=0,
                ),
                cleanup_complete=self._cleanup(root),
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def _publish_native_result(
        self,
        plan: ReductionPlan,
        reduction_id: str,
        native: NativeReductionResult,
        attempts: ReductionAttemptSummary,
        *,
        final_artifact_id: str | None = None,
        best_known_artifact_id: str | None = None,
        stdout_id: str | None = None,
        stderr_id: str | None = None,
        cleanup_complete: bool,
    ) -> ReductionResult:
        result = ReductionResult(
            schema_version=2,
            reduction_id=reduction_id,
            plan_id=plan.plan_id,
            disposition=native.disposition,
            original_artifact_id=plan.original_artifact_id,
            final_artifact_id=final_artifact_id,
            predicate_definition_id=plan.predicate_definition_id,
            predicate_instance_id=plan.predicate_instance_id,
            attempts=attempts,
            predicate_stdout_artifact_id=stdout_id,
            predicate_stderr_artifact_id=stderr_id,
            cleanup_complete=cleanup_complete,
            limitations=native.limitations,
            engine="native_ddmin",
            partitioner=plan.partitioner,
            original_unit_count=native.original_unit_count,
            final_unit_count=native.final_unit_count,
            minimality=native.minimality,
            best_known_artifact_id=best_known_artifact_id,
            final_revalidation_status=native.final_revalidation,
            budget_exhausted=native.budget_exhausted,
        )
        try:
            created = self.results.create(result)
        except DomainError as error:
            if error.code is ErrorCode.REVISION_CONFLICT:
                return self.results.read(reduction_id)
            raise
        rows = {
            "reduction_results": [
                {
                    "reduction_id": created.reduction_id,
                    "plan_id": created.plan_id,
                    "disposition": created.disposition,
                    "original_artifact_id": created.original_artifact_id,
                    "final_artifact_id": created.final_artifact_id,
                    "reducer_definition_id": None,
                    "reducer_instance_id": None,
                    "predicate_definition_id": created.predicate_definition_id,
                    "predicate_instance_id": created.predicate_instance_id,
                    "attempts_json": created.attempts.model_dump_json(),
                    "reducer_stdout_artifact_id": None,
                    "reducer_stderr_artifact_id": None,
                    "predicate_stdout_artifact_id": created.predicate_stdout_artifact_id,
                    "predicate_stderr_artifact_id": created.predicate_stderr_artifact_id,
                    "cleanup_complete": created.cleanup_complete,
                    "limitations": list(created.limitations),
                    "finished_at": created.finished_at,
                    "engine": created.engine,
                    "partitioner": created.partitioner,
                    "original_unit_count": created.original_unit_count,
                    "final_unit_count": created.final_unit_count,
                    "minimality": created.minimality,
                    "best_known_artifact_id": created.best_known_artifact_id,
                    "final_revalidation_status": created.final_revalidation_status,
                    "budget_exhausted": created.budget_exhausted,
                }
            ],
            "reduction_attempts": [
                {
                    "reduction_id": created.reduction_id,
                    **attempt.model_dump(mode="json"),
                }
                for attempt in native.attempts
            ],
        }
        self.publisher.publish_rows(
            rows,
            publisher="flameox.reductions",
            publisher_version="2",
            input_artifact_ids=tuple(
                artifact_id
                for artifact_id in (
                    created.original_artifact_id,
                    created.final_artifact_id,
                    created.best_known_artifact_id,
                    created.predicate_stdout_artifact_id,
                    created.predicate_stderr_artifact_id,
                )
                if artifact_id is not None
            ),
        )
        return created

    def _revalidate(self, plan: ReductionPlan) -> None:
        if plan.workspace_id != self.workspace.identity.workspace_id:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Reduction plan is for another workspace.",
            )
        self.artifacts.get(plan.original_artifact_id)
        predicate = self.workloads.resolve(
            plan.predicate_workload,
            plan.predicate_parameters,
        )
        if (
            predicate.workload_definition_id != plan.predicate_definition_id
            or predicate.workload_instance_id != plan.predicate_instance_id
            or predicate.command != plan.predicate_command
            or _executable_digest(predicate.command) != plan.predicate_executable_digest
        ):
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Predicate identity changed after reduction planning.",
            )

    def _execution_request(
        self,
        command: CommandSpec,
        *,
        timeout: float,
        writable_root: Path,
        overrides: dict[str, str],
    ) -> ExecutionRequest:
        return ExecutionRequest(
            argv=command.argv,
            cwd=Path(command.cwd),
            environment_allowlist=("PATH",),
            environment_overrides={**command.env_overrides, **overrides},
            allowed_working_roots=(self.workspace.project_root,),
            timeout_seconds=timeout,
            max_output_bytes=self.workspace.config.execution.max_output_bytes,
            resource_policy=ResourcePolicy(
                filesystem_path=self.workspace.paths.root,
                staging_root=self.workspace.paths.staging,
                writable_roots=(writable_root,),
                minimum_free_bytes=self.workspace.config.storage.min_free_bytes,
                sampling_interval_ms=(
                    self.workspace.config.execution.resource_sampling_interval_ms
                ),
                max_observed_files=self.workspace.config.execution.max_resource_observed_files,
            ),
        )

    def _preserve_output(self, root: Path, name: str, content: bytes) -> str | None:
        if not content:
            return None
        path = root / name
        path.write_bytes(content)
        return self.artifacts.import_path(
            path,
            allowed_roots=(root,),
            max_bytes=self.workspace.config.execution.max_output_bytes,
        ).content.artifact_id

    @staticmethod
    def _cleanup(root: Path) -> bool:
        shutil.rmtree(root, ignore_errors=True)
        return not root.exists()


def _executable_digest(command: CommandSpec) -> str:
    try:
        with Path(command.argv[0]).open("rb") as stream:
            return f"sha256:{hashlib.file_digest(stream, 'sha256').hexdigest()}"
    except OSError as error:
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            f"Cannot bind executable identity for {command.argv[0]!r}.",
        ) from error
