from __future__ import annotations

import asyncio
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from flameox.application.artifact_workers import ArtifactWorker
from flameox.application.native_reducer import NativeReductionResult
from flameox.application.workloads import WorkloadService
from flameox.domain import DomainError, ErrorCode, digest_model
from flameox.domain.models import CommandSpec, utc_now
from flameox.evidence import GenerationPublisher
from flameox.execution import SubprocessBroker
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
        plan: ReductionPlan,
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
            if not root.exists():
                return await self._execute_native(plan)
            await asyncio.sleep(0.1)
        shutil.rmtree(root, ignore_errors=True)
        raise DomainError(
            ErrorCode.REVISION_CONFLICT,
            "Another execution of this reduction plan did not reach a terminal result.",
            retryable=True,
        )

    async def _execute_native(self, plan: ReductionPlan) -> ReductionResult:
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
                plan,
                reduction_id,
                root,
                timeout_seconds=(
                    plan.limits.wall_time_seconds + plan.limits.predicate_timeout_seconds + 10
                ),
            )
        try:
            artifact_path = self.artifacts.get(plan.original_artifact_id).payload_path

            def consume(
                response: dict[str, object], worker_root: Path
            ) -> tuple[NativeReductionResult, list[str], str, str | None, str | None, Path]:
                raw_result = response.get("result")
                accepted_paths = response.get("accepted_paths")
                if not isinstance(raw_result, dict) or not isinstance(accepted_paths, list):
                    raise DomainError(
                        ErrorCode.ARTIFACT_PARSE_FAILED,
                        "Reduction worker returned an invalid result contract.",
                    )
                native = NativeReductionResult.model_validate(raw_result)
                accepted_ids: list[str] = []
                for raw_path in accepted_paths:
                    path = self._worker_file(worker_root, raw_path)
                    stored = self.artifacts.import_path(
                        path,
                        allowed_roots=(worker_root,),
                        max_bytes=self.workspace.config.capture.max_artifact_bytes,
                    )
                    if stored.content.artifact_id not in accepted_ids:
                        accepted_ids.append(stored.content.artifact_id)
                final_path = self._worker_file(worker_root, response.get("final_path"))
                final_id = self.artifacts.import_path(
                    final_path,
                    allowed_roots=(worker_root,),
                    max_bytes=self.workspace.config.capture.max_artifact_bytes,
                ).content.artifact_id
                stdout_id = self._import_worker_output(worker_root, response.get("stdout_path"))
                stderr_id = self._import_worker_output(worker_root, response.get("stderr_path"))
                return native, accepted_ids, final_id, stdout_id, stderr_id, worker_root

            worker_result = await ArtifactWorker(self.workspace, broker=self.broker).run(
                "flameox.workers.reduction",
                {
                    "artifact_path": str(artifact_path),
                    "partitioner": plan.partitioner,
                    "chunk_size": plan.chunk_size,
                    "predicate_command": plan.predicate_command.model_dump(mode="json"),
                    "limits": {
                        "max_attempts": plan.limits.max_attempts,
                        "wall_time_seconds": plan.limits.wall_time_seconds,
                        "repetitions": plan.limits.predicate_repetitions,
                    },
                    "predicate_timeout_seconds": plan.limits.predicate_timeout_seconds,
                    "project_root": str(self.workspace.project_root),
                    "workspace_root": str(self.workspace.paths.root),
                    "staging_root": str(self.workspace.paths.staging),
                    "max_output_bytes": self.workspace.config.execution.max_output_bytes,
                    "minimum_free_bytes": self.workspace.config.storage.min_free_bytes,
                    "maximum_rss_bytes": self.workspace.config.execution.max_memory_bytes,
                    "sampling_interval_ms": (
                        self.workspace.config.execution.resource_sampling_interval_ms
                    ),
                    "max_observed_files": (
                        self.workspace.config.execution.max_resource_observed_files
                    ),
                },
                name="reduction",
                timeout_seconds=(
                    plan.limits.wall_time_seconds + plan.limits.predicate_timeout_seconds + 10
                ),
                consume=consume,
            )
            native, accepted_ids, final_id, stdout_id, stderr_id, worker_root = worker_result
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
                final_artifact_id=final_id,
                best_known_artifact_id=(
                    accepted_ids[-1] if accepted_ids else plan.original_artifact_id
                ),
                stdout_id=stdout_id,
                stderr_id=stderr_id,
                cleanup_complete=self._cleanup(root) and not worker_root.exists(),
            )
            return result
        except asyncio.CancelledError:
            raise
        except DomainError as error:
            return self._publish_native_result(
                plan,
                reduction_id,
                NativeReductionResult(
                    disposition="inconclusive",
                    original_digest=plan.original_artifact_id,
                    final_digest=plan.original_artifact_id,
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

    def _import_worker_output(self, root: Path, raw_path: object) -> str | None:
        if raw_path is None:
            return None
        path = self._worker_file(root, raw_path)
        return self.artifacts.import_path(
            path,
            allowed_roots=(root,),
            max_bytes=self.workspace.config.execution.max_output_bytes,
        ).content.artifact_id

    @staticmethod
    def _worker_file(root: Path, raw_path: object) -> Path:
        if not isinstance(raw_path, str) or Path(raw_path).name != raw_path:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Reduction worker returned an invalid staged-file reference.",
            )
        path = root / raw_path
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Reduction worker staged-file reference escapes its job directory.",
            ) from error
        if not resolved.is_file():
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                "Reduction worker staged-file reference is not a regular file.",
            )
        return resolved

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
