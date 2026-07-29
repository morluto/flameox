from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import stat
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from flameox.application.workloads import Scalar, WorkloadService
from flameox.domain import DomainError, ErrorCode, digest_model
from flameox.domain.models import CommandSpec, utc_now
from flameox.evidence import GenerationPublisher
from flameox.execution import ExecutionRequest, ResourcePolicy, SubprocessBroker
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, JsonRecordStore, Workspace

ReductionDisposition = Literal[
    "succeeded",
    "unchanged",
    "inconclusive",
    "failed",
    "cancelled",
    "timed_out",
]


class ReductionLimits(ContractModel):
    max_attempts: Annotated[int, Field(ge=1, le=100_000)] = 1_000
    wall_time_seconds: Annotated[float, Field(gt=0, le=86_400)] = 900
    predicate_timeout_seconds: Annotated[float, Field(gt=0, le=3_600)] = 30
    predicate_repetitions: Annotated[int, Field(ge=1, le=20)] = 1


class PlanReductionRequest(ContractModel):
    original_artifact_id: str
    reducer_workload: str
    predicate_workload: str
    reducer_parameters: dict[str, Scalar] = Field(default_factory=dict)
    predicate_parameters: dict[str, Scalar] = Field(default_factory=dict)
    limits: ReductionLimits = Field(default_factory=ReductionLimits)
    expected_determinism: Literal["deterministic", "repeated"] = "deterministic"


class ReductionPlan(ContractModel):
    schema_version: Literal[1] = 1
    plan_id: str
    request_digest: str
    workspace_id: str
    original_artifact_id: str
    reducer_workload: str
    reducer_definition_id: str
    reducer_instance_id: str
    reducer_command: CommandSpec
    reducer_parameters: dict[str, Scalar]
    reducer_executable_digest: str
    predicate_workload: str
    predicate_definition_id: str
    predicate_instance_id: str
    predicate_command: CommandSpec
    predicate_parameters: dict[str, Scalar]
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
    schema_version: Literal[1] = 1
    reduction_id: str
    plan_id: str
    disposition: ReductionDisposition
    original_artifact_id: str
    final_artifact_id: str | None = None
    reducer_definition_id: str
    reducer_instance_id: str
    predicate_definition_id: str
    predicate_instance_id: str
    attempts: ReductionAttemptSummary
    reducer_stdout_artifact_id: str | None = None
    reducer_stderr_artifact_id: str | None = None
    cleanup_complete: bool
    limitations: tuple[str, ...] = ()
    finished_at: datetime = Field(default_factory=utc_now)


class ReductionService:
    """Coordinate an approved reducer around an approved bounded predicate."""

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
        reducer = self.workloads.resolve(
            request.reducer_workload,
            request.reducer_parameters,
            require_approval=True,
        )
        predicate = self.workloads.resolve(
            request.predicate_workload,
            request.predicate_parameters,
            require_approval=True,
        )
        bound = {
            "workspace_id": self.workspace.identity.workspace_id,
            **request.model_dump(mode="json"),
            "reducer_definition_id": reducer.workload_definition_id,
            "reducer_instance_id": reducer.workload_instance_id,
            "reducer_command": reducer.command.model_dump(mode="json"),
            "reducer_executable_digest": _executable_digest(reducer.command),
            "predicate_definition_id": predicate.workload_definition_id,
            "predicate_instance_id": predicate.workload_instance_id,
            "predicate_command": predicate.command.model_dump(mode="json"),
            "predicate_executable_digest": _executable_digest(predicate.command),
        }
        request_digest = digest_model(bound)
        plan = ReductionPlan(
            plan_id=request_digest,
            request_digest=request_digest,
            workspace_id=self.workspace.identity.workspace_id,
            original_artifact_id=request.original_artifact_id,
            reducer_workload=request.reducer_workload,
            reducer_definition_id=reducer.workload_definition_id,
            reducer_instance_id=reducer.workload_instance_id,
            reducer_command=reducer.command,
            reducer_parameters=request.reducer_parameters,
            reducer_executable_digest=_executable_digest(reducer.command),
            predicate_workload=request.predicate_workload,
            predicate_definition_id=predicate.workload_definition_id,
            predicate_instance_id=predicate.workload_instance_id,
            predicate_command=predicate.command,
            predicate_parameters=request.predicate_parameters,
            predicate_executable_digest=_executable_digest(predicate.command),
            limits=request.limits,
            expected_determinism=request.expected_determinism,
        )
        try:
            return self.plans.create(plan)
        except DomainError as error:
            if error.code is ErrorCode.REVISION_CONFLICT:
                return self.plans.read(plan.plan_id)
            raise

    async def execute(self, plan_id: str) -> ReductionResult:
        plan = self.plans.read(plan_id)
        self._revalidate(plan)
        reduction_id = digest_model({"plan_id": plan_id, "contract": "reduction-v1"})
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
                    plan.limits.wall_time_seconds
                    + plan.limits.predicate_timeout_seconds
                    + 10
                ),
            )
        candidate = root / "candidate"
        wrapper = root / "predicate"
        socket_root = Path(tempfile.mkdtemp(prefix="flameox-reduce-"))
        socket_path = socket_root / "predicate.sock"
        original = self.artifacts.get(plan.original_artifact_id).payload_path
        self._write_predicate_wrapper(socket_path, wrapper)
        attempts = ReductionAttemptSummary(
            attempted=0,
            passed=0,
            failed=0,
            contradictory=0,
            timed_out=0,
        )
        try:
            outcomes: list[str] = []
            attempt_lock = asyncio.Lock()
            handlers: set[asyncio.Task[None]] = set()

            def connected(
                reader: asyncio.StreamReader,
                writer: asyncio.StreamWriter,
            ) -> None:
                task = asyncio.create_task(
                    self._handle_predicate_request(
                        plan,
                        root,
                        reader,
                        writer,
                        outcomes,
                        attempt_lock,
                    )
                )
                handlers.add(task)

                def finished(completed: asyncio.Task[None]) -> None:
                    handlers.discard(completed)
                    if not completed.cancelled():
                        completed.exception()

                task.add_done_callback(finished)

            server = await asyncio.start_unix_server(connected, path=socket_path)
            try:
                reducer = await self.broker.run(
                    self._execution_request(
                        plan.reducer_command,
                        timeout=plan.limits.wall_time_seconds,
                        writable_root=root,
                        overrides={
                            "FLAMEOX_REDUCTION_ORIGINAL": str(original),
                            "FLAMEOX_REDUCTION_CANDIDATE": str(candidate),
                            "FLAMEOX_PREDICATE_WRAPPER": str(wrapper),
                        },
                    )
                )
            finally:
                server.close()
                await server.wait_closed()
                if handlers:
                    await asyncio.gather(*tuple(handlers), return_exceptions=True)
            stdout_id = self._preserve_output(root, "reducer.stdout", reducer.stdout)
            stderr_id = self._preserve_output(root, "reducer.stderr", reducer.stderr)
            attempts = self._summarize_attempts(outcomes)
            if attempts.contradictory or attempts.timed_out:
                return self._publish_result(
                    plan,
                    reduction_id,
                    "inconclusive",
                    attempts,
                    stdout_id=stdout_id,
                    stderr_id=stderr_id,
                    cleanup_complete=self._cleanup(root),
                    limitations=(
                        "The repeated predicate was contradictory or timed out.",
                    ),
                )
            if reducer.process.exit_code != 0:
                return self._publish_result(
                    plan,
                    reduction_id,
                    "failed",
                    attempts,
                    stdout_id=stdout_id,
                    stderr_id=stderr_id,
                    cleanup_complete=self._cleanup(root),
                    limitations=("The approved reducer exited unsuccessfully.",),
                )
            try:
                if candidate.is_symlink():
                    raise ValueError("candidate is a symbolic link")
                resolved_candidate = candidate.resolve(strict=True)
                resolved_candidate.relative_to(root.resolve())
            except (OSError, ValueError) as error:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    "Reducer candidate must be a regular file at the bound candidate path.",
                ) from error
            if not resolved_candidate.is_file():
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    "Reducer candidate must be a regular file at the bound candidate path.",
                )
            revalidation = await self._revalidate_candidate(plan, resolved_candidate, root)
            attempts = self._merge_attempts(attempts, revalidation)
            if revalidation.contradictory or revalidation.timed_out:
                return self._publish_result(
                    plan,
                    reduction_id,
                    "inconclusive",
                    attempts,
                    stdout_id=stdout_id,
                    stderr_id=stderr_id,
                    cleanup_complete=self._cleanup(root),
                    limitations=(
                        "Final predicate revalidation was contradictory or timed out.",
                    ),
                )
            if revalidation.failed or not revalidation.passed:
                return self._publish_result(
                    plan,
                    reduction_id,
                    "failed",
                    attempts,
                    stdout_id=stdout_id,
                    stderr_id=stderr_id,
                    cleanup_complete=self._cleanup(root),
                    limitations=("The final candidate did not preserve the predicate.",),
                )
            stored = self.artifacts.import_path(
                resolved_candidate,
                allowed_roots=(root,),
                max_bytes=self.workspace.config.capture.max_artifact_bytes,
            )
            disposition: Literal["succeeded", "unchanged"] = (
                "unchanged"
                if stored.content.artifact_id == plan.original_artifact_id
                else "succeeded"
            )
            return self._publish_result(
                plan,
                reduction_id,
                disposition,
                attempts,
                final_artifact_id=stored.content.artifact_id,
                stdout_id=stdout_id,
                stderr_id=stderr_id,
                cleanup_complete=self._cleanup(root),
                limitations=(
                    "Candidate size is observational metadata, not a minimality or quality claim.",
                ),
            )
        except asyncio.CancelledError:
            self._publish_result(
                plan,
                reduction_id,
                "cancelled",
                attempts,
                cleanup_complete=self._cleanup(root),
                limitations=("Reduction was cancelled; no unchecked checkpoint was preserved.",),
            )
            raise
        except DomainError as execution_error:
            failure_disposition: Literal["failed", "timed_out"] = (
                "timed_out"
                if execution_error.code is ErrorCode.PROCESS_TIMEOUT
                else "failed"
            )
            return self._publish_result(
                plan,
                reduction_id,
                failure_disposition,
                attempts,
                cleanup_complete=self._cleanup(root),
                limitations=(execution_error.message,),
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(socket_root, ignore_errors=True)

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

    def _revalidate(self, plan: ReductionPlan) -> None:
        if plan.workspace_id != self.workspace.identity.workspace_id:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Reduction plan is for another workspace.",
            )
        self.artifacts.get(plan.original_artifact_id)
        reducer = self.workloads.resolve(
            plan.reducer_workload,
            plan.reducer_parameters,
            require_approval=True,
        )
        predicate = self.workloads.resolve(
            plan.predicate_workload,
            plan.predicate_parameters,
            require_approval=True,
        )
        if (
            reducer.workload_definition_id != plan.reducer_definition_id
            or predicate.workload_definition_id != plan.predicate_definition_id
            or reducer.workload_instance_id != plan.reducer_instance_id
            or predicate.workload_instance_id != plan.predicate_instance_id
            or reducer.command != plan.reducer_command
            or predicate.command != plan.predicate_command
            or _executable_digest(reducer.command) != plan.reducer_executable_digest
            or _executable_digest(predicate.command) != plan.predicate_executable_digest
        ):
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Reducer or predicate identity changed after reduction planning.",
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

    async def _revalidate_candidate(
        self,
        plan: ReductionPlan,
        candidate: Path,
        root: Path,
    ) -> ReductionAttemptSummary:
        outcomes: list[Literal["passed", "failed", "timed_out"]] = []
        for _ in range(plan.limits.predicate_repetitions):
            try:
                result = await self.broker.run(
                    self._execution_request(
                        plan.predicate_command,
                        timeout=plan.limits.predicate_timeout_seconds,
                        writable_root=root,
                        overrides={"FLAMEOX_REDUCTION_CANDIDATE": str(candidate)},
                    )
                )
                outcomes.append("passed" if result.process.exit_code == 0 else "failed")
            except DomainError as error:
                if error.code is not ErrorCode.PROCESS_TIMEOUT:
                    raise
                outcomes.append("timed_out")
        contradictory = int("passed" in outcomes and "failed" in outcomes)
        return ReductionAttemptSummary(
            attempted=len(outcomes),
            passed=outcomes.count("passed"),
            failed=outcomes.count("failed"),
            contradictory=contradictory,
            timed_out=outcomes.count("timed_out"),
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

    def _publish_result(
        self,
        plan: ReductionPlan,
        reduction_id: str,
        disposition: ReductionDisposition,
        attempts: ReductionAttemptSummary,
        *,
        final_artifact_id: str | None = None,
        stdout_id: str | None = None,
        stderr_id: str | None = None,
        cleanup_complete: bool,
        limitations: tuple[str, ...] = (),
    ) -> ReductionResult:
        result = ReductionResult(
            reduction_id=reduction_id,
            plan_id=plan.plan_id,
            disposition=disposition,
            original_artifact_id=plan.original_artifact_id,
            final_artifact_id=final_artifact_id,
            reducer_definition_id=plan.reducer_definition_id,
            reducer_instance_id=plan.reducer_instance_id,
            predicate_definition_id=plan.predicate_definition_id,
            predicate_instance_id=plan.predicate_instance_id,
            attempts=attempts,
            reducer_stdout_artifact_id=stdout_id,
            reducer_stderr_artifact_id=stderr_id,
            cleanup_complete=cleanup_complete,
            limitations=limitations,
        )
        try:
            created = self.results.create(result)
        except DomainError as error:
            if error.code is ErrorCode.REVISION_CONFLICT:
                return self.results.read(reduction_id)
            raise
        self.publisher.publish_rows(
            {
                "reduction_results": [
                    {
                        "reduction_id": created.reduction_id,
                        "plan_id": created.plan_id,
                        "disposition": created.disposition,
                        "original_artifact_id": created.original_artifact_id,
                        "final_artifact_id": created.final_artifact_id,
                        "reducer_definition_id": created.reducer_definition_id,
                        "reducer_instance_id": created.reducer_instance_id,
                        "predicate_definition_id": created.predicate_definition_id,
                        "predicate_instance_id": created.predicate_instance_id,
                        "attempts_json": created.attempts.model_dump_json(),
                        "reducer_stdout_artifact_id": (
                            created.reducer_stdout_artifact_id
                        ),
                        "reducer_stderr_artifact_id": (
                            created.reducer_stderr_artifact_id
                        ),
                        "cleanup_complete": created.cleanup_complete,
                        "limitations": list(created.limitations),
                        "finished_at": created.finished_at,
                    }
                ]
            },
            publisher="flameox.reductions",
            publisher_version="1",
            input_artifact_ids=tuple(
                artifact_id
                for artifact_id in (
                    created.original_artifact_id,
                    created.final_artifact_id,
                    created.reducer_stdout_artifact_id,
                    created.reducer_stderr_artifact_id,
                )
                if artifact_id is not None
            ),
        )
        return created

    @staticmethod
    def _merge_attempts(
        left: ReductionAttemptSummary,
        right: ReductionAttemptSummary,
    ) -> ReductionAttemptSummary:
        return ReductionAttemptSummary(
            attempted=left.attempted + right.attempted,
            passed=left.passed + right.passed,
            failed=left.failed + right.failed,
            contradictory=left.contradictory + right.contradictory,
            timed_out=left.timed_out + right.timed_out,
        )

    @staticmethod
    def _cleanup(root: Path) -> bool:
        shutil.rmtree(root, ignore_errors=True)
        return not root.exists()

    @staticmethod
    def _summarize_attempts(outcomes: list[str]) -> ReductionAttemptSummary:
        return ReductionAttemptSummary(
            attempted=len(outcomes),
            passed=outcomes.count("passed"),
            failed=outcomes.count("failed"),
            contradictory=outcomes.count("contradictory"),
            timed_out=outcomes.count("timed_out"),
        )

    async def _handle_predicate_request(
        self,
        plan: ReductionPlan,
        root: Path,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        outcomes: list[str],
        lock: asyncio.Lock,
    ) -> None:
        response = 125
        reservation: int | None = None
        try:
            line = await reader.readline()
            if len(line) > 4_096:
                raise ValueError("predicate request is too long")
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("candidate"), str):
                raise ValueError("invalid predicate request")
            candidate = Path(value["candidate"])
            if candidate.is_symlink():
                raise ValueError("candidate is a symbolic link")
            candidate = candidate.resolve(strict=True)
            candidate.relative_to(root.resolve())
            if not candidate.is_file():
                raise ValueError("candidate is not a regular file")
            async with lock:
                if len(outcomes) >= plan.limits.max_attempts:
                    raise ValueError("attempt limit reached")
                reservation = len(outcomes)
                outcomes.append("pending")
            attempt = await self._revalidate_candidate(plan, candidate, root)
            if attempt.contradictory:
                outcome = "contradictory"
            elif attempt.timed_out:
                outcome = "timed_out"
            elif attempt.passed and not attempt.failed:
                outcome = "passed"
            else:
                outcome = "failed"
            outcomes[reservation] = outcome
            response = 0 if outcome == "passed" else 1
        except (DomainError, OSError, ValueError, json.JSONDecodeError):
            if reservation is not None:
                outcomes[reservation] = "failed"
        finally:
            writer.write(f"{response}\n".encode())
            try:
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

    @staticmethod
    def _write_predicate_wrapper(socket_path: Path, wrapper: Path) -> None:
        wrapper.write_text(
            _PREDICATE_WRAPPER.replace("__SOCKET_PATH__", repr(str(socket_path)))
        )
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)


_PREDICATE_WRAPPER = f"""#!{sys.executable}
import json, socket, sys
if len(sys.argv) != 2:
    raise SystemExit(126)
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    connection.connect(__SOCKET_PATH__)
    connection.sendall((json.dumps({{"candidate": sys.argv[1]}}) + "\\n").encode())
    response = connection.makefile().readline()
raise SystemExit(int(response.strip()))
"""


def _executable_digest(command: CommandSpec) -> str:
    try:
        with Path(command.argv[0]).open("rb") as stream:
            return f"sha256:{hashlib.file_digest(stream, 'sha256').hexdigest()}"
    except OSError as error:
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            f"Cannot bind executable identity for {command.argv[0]!r}.",
        ) from error
