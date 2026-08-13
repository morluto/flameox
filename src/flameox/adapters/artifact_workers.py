from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar, cast

from pydantic import JsonValue, ValidationError

from flameox.atomic import atomic_write_json
from flameox.command_binding import ExecutableResolver
from flameox.domain import DomainError, ErrorCode
from flameox.domain.executables import (
    ExecutableResolutionRequest,
    ExecutableTrustPolicy,
)
from flameox.execution import ExecutionRequest, ResourcePolicy, SubprocessBroker
from flameox.filesystem import BoundedFileSystem
from flameox.storage import Workspace
from flameox.workers.protocol import (
    TYPED_WORKER_RESPONSE,
    WorkerDefinition,
    WorkerFailed,
    WorkerOutputFile,
    WorkerRequestEnvelope,
    WorkerSucceeded,
    worker_failure_error_code,
)

T = TypeVar("T")
RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class IsolatedWorkerHarness:
    """Own the one bounded request/response protocol for isolated native readers."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        broker: SubprocessBroker | None = None,
        python: Path | None = None,
    ) -> None:
        self.workspace = workspace
        self.broker = broker or SubprocessBroker()
        python = python or Path(sys.executable)
        self.executable_binding = ExecutableResolver().resolve(
            ExecutableResolutionRequest(
                token=str(python),
                cwd=workspace.project_root,
                environment={},
                policy=ExecutableTrustPolicy.TRUSTED_HOST_TOOL,
            )
        )

    @contextmanager
    def run_typed_sync_session(
        self,
        definition: WorkerDefinition[RequestT, ResponseT],
        request: RequestT,
    ) -> Iterator[tuple[ResponseT, Path]]:
        """Run one closed typed protocol and retain staged outputs during consumption."""
        job_root = self._job_root(definition.name)
        job_root.mkdir(parents=True, exist_ok=False)
        request_path = job_root / "request.json"
        response_path = job_root / "response.json"
        request_id = secrets.token_hex(16)
        payload = cast(
            JsonValue,
            definition.request.dump_python(
                definition.request.validate_python(request),
                mode="json",
            ),
        )
        envelope = WorkerRequestEnvelope(
            request_id=request_id,
            operation=definition.operation,
            implementation=definition.implementation,
            payload=payload,
        )
        atomic_write_json(request_path, envelope.model_dump(mode="json"))
        try:
            outcome = self.broker.run_sync(
                self._execution_request(
                    definition.module,
                    request_path,
                    response_path,
                    job_root,
                    timeout_seconds=definition.timeout_seconds,
                )
            )
            response = self._load_typed_response(
                outcome.process.exit_code,
                outcome.stderr,
                response_path,
                definition=definition,
                request_id=request_id,
            )
            yield response, job_root
        finally:
            shutil.rmtree(job_root, ignore_errors=True)

    def run_typed_sync(
        self,
        definition: WorkerDefinition[RequestT, ResponseT],
        request: RequestT,
    ) -> ResponseT:
        with self.run_typed_sync_session(definition, request) as (response, _job_root):
            return response

    async def run_typed(
        self,
        definition: WorkerDefinition[RequestT, ResponseT],
        request: RequestT,
    ) -> ResponseT:
        """Run a typed worker asynchronously when no staged output outlives the call."""
        job_root = self._job_root(definition.name)
        job_root.mkdir(parents=True, exist_ok=False)
        request_path = job_root / "request.json"
        response_path = job_root / "response.json"
        request_id = secrets.token_hex(16)
        payload = cast(
            JsonValue,
            definition.request.dump_python(
                definition.request.validate_python(request),
                mode="json",
            ),
        )
        atomic_write_json(
            request_path,
            WorkerRequestEnvelope(
                request_id=request_id,
                operation=definition.operation,
                implementation=definition.implementation,
                payload=payload,
            ).model_dump(mode="json"),
        )
        try:
            outcome = await self.broker.run(
                self._execution_request(
                    definition.module,
                    request_path,
                    response_path,
                    job_root,
                    timeout_seconds=definition.timeout_seconds,
                )
            )
            return self._load_typed_response(
                outcome.process.exit_code,
                outcome.stderr,
                response_path,
                definition=definition,
                request_id=request_id,
            )
        finally:
            shutil.rmtree(job_root, ignore_errors=True)

    async def run_typed_session(
        self,
        definition: WorkerDefinition[RequestT, ResponseT],
        request: RequestT,
        *,
        consume: Callable[[ResponseT, Path], T],
        timeout_seconds: float | None = None,
        job_root: Path | None = None,
    ) -> T:
        """Keep typed worker outputs alive while one host-side consumer validates them."""
        job_root = job_root or self._job_root(definition.name)
        job_root.mkdir(parents=True, exist_ok=False)
        request_path = job_root / "request.json"
        response_path = job_root / "response.json"
        request_id = secrets.token_hex(16)
        payload = cast(
            JsonValue,
            definition.request.dump_python(
                definition.request.validate_python(request),
                mode="json",
            ),
        )
        atomic_write_json(
            request_path,
            WorkerRequestEnvelope(
                request_id=request_id,
                operation=definition.operation,
                implementation=definition.implementation,
                payload=payload,
            ).model_dump(mode="json"),
        )
        try:
            outcome = await self.broker.run(
                self._execution_request(
                    definition.module,
                    request_path,
                    response_path,
                    job_root,
                    timeout_seconds=timeout_seconds or definition.timeout_seconds,
                )
            )
            response = self._load_typed_response(
                outcome.process.exit_code,
                outcome.stderr,
                response_path,
                definition=definition,
                request_id=request_id,
            )
            return consume(response, job_root)
        finally:
            shutil.rmtree(job_root, ignore_errors=True)

    def validate_output_file(self, job_root: Path, output: WorkerOutputFile) -> Path:
        """Open and identify one declared worker output through the trusted-root boundary."""
        path = job_root / output.relative_path
        filesystem = BoundedFileSystem((job_root,))
        with filesystem.open_regular(
            path,
            max_bytes=output.byte_length,
            require_single_link=True,
        ) as descriptor:
            metadata = os.fstat(descriptor)
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
        if metadata.st_size != output.byte_length:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Worker output size does not match its declaration.",
            )
        actual = "sha256:" + digest.hexdigest()
        if actual != output.sha256:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Worker output digest does not match its declaration.",
            )
        return path

    def _job_root(self, name: str) -> Path:
        return self.workspace.paths.staging / "artifact-workers" / f"{name}-{secrets.token_hex(16)}"

    def _execution_request(
        self,
        module: str,
        request_path: Path,
        response_path: Path,
        job_root: Path,
        *,
        timeout_seconds: float,
    ) -> ExecutionRequest:
        return ExecutionRequest(
            argv=(
                str(self.executable_binding.invocation_path),
                "-m",
                module,
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            ),
            executable_binding=self.executable_binding,
            cwd=self.workspace.project_root,
            environment_allowlist=tuple(
                self.workspace.config.execution.child_environment_allowlist
            ),
            allowed_working_roots=(self.workspace.project_root,),
            timeout_seconds=timeout_seconds,
            max_output_bytes=1_048_576,
            resource_policy=ResourcePolicy(
                filesystem_path=self.workspace.paths.root,
                staging_root=self.workspace.paths.staging,
                writable_roots=(job_root,),
                minimum_free_bytes=self.workspace.config.storage.min_free_bytes,
                maximum_rss_bytes=self.workspace.config.execution.max_memory_bytes,
                sampling_interval_ms=self.workspace.config.execution.resource_sampling_interval_ms,
                max_observed_files=self.workspace.config.execution.max_resource_observed_files,
            ),
        )

    def _load_typed_response(
        self,
        exit_code: int | None,
        stderr: bytes,
        response_path: Path,
        *,
        definition: WorkerDefinition[RequestT, ResponseT],
        request_id: str,
    ) -> ResponseT:
        if exit_code not in {0, 1} or not response_path.is_file():
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"{definition.name} worker transport failed before a trustworthy response.",
                details={
                    "exit_code": exit_code,
                    "stderr": stderr.decode(errors="replace")[-2_000:],
                },
            )
        try:
            raw = BoundedFileSystem((response_path.parent,)).read_bytes(
                response_path,
                max_bytes=self.workspace.config.execution.max_output_bytes,
                require_single_link=True,
            )
            envelope = TYPED_WORKER_RESPONSE.validate_json(raw)
        except (OSError, ValidationError, ValueError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"{definition.name} worker response violates the transport contract.",
            ) from exc
        if (
            envelope.request_id != request_id
            or envelope.operation is not definition.operation
            or envelope.implementation != definition.implementation
        ):
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                f"{definition.name} worker response is bound to another request.",
            )
        if isinstance(envelope, WorkerSucceeded):
            if exit_code != 0:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    f"{definition.name} worker success envelope has a failure exit status.",
                )
            try:
                return definition.response.validate_python(envelope.payload)
            except ValidationError as exc:
                raise DomainError(
                    ErrorCode.ARTIFACT_PARSE_FAILED,
                    f"{definition.name} worker success payload violates its exact schema.",
                ) from exc
        if not isinstance(envelope, WorkerFailed) or exit_code != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_PARSE_FAILED,
                f"{definition.name} worker failure envelope has an invalid exit status.",
            )
        raise DomainError(
            worker_failure_error_code(envelope.failure.kind),
            envelope.failure.message,
        )
