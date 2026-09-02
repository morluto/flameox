from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
import shutil
import sys
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from pydantic import JsonValue, ValidationError

from flameox.atomic import atomic_write_json
from flameox.canonical import sha256_id
from flameox.command_binding import ExecutableResolver
from flameox.executable_models import (
    ExecutableResolutionRequest,
    ExecutableTrustPolicy,
)
from flameox.execution import ExecutionRequest, ResourcePolicy, SubprocessBroker
from flameox.filesystem import BoundedFileSystem
from flameox.process_models import process_exit_code
from flameox.runtime_errors import DomainError, ErrorCode
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


@dataclass(frozen=True, slots=True)
class WorkerRuntimeConfig:
    working_directory: Path
    staging_root: Path
    filesystem_path: Path
    child_environment_allowlist: tuple[str, ...] = ("PATH",)
    minimum_free_bytes: int = 64 * 1024 * 1024
    maximum_rss_bytes: int | None = 1024**3
    resource_sampling_interval_ms: int = 250
    max_resource_observed_files: int = 10_000
    max_response_bytes: int = 4 * 1024 * 1024


class IsolatedWorkerHarness:
    """Own the one bounded request/response protocol for isolated native readers."""

    def __init__(
        self,
        runtime: WorkerRuntimeConfig,
        *,
        broker: SubprocessBroker | None = None,
        python: Path | None = None,
    ) -> None:
        self.runtime = runtime
        self.broker = broker or SubprocessBroker()
        python = python or Path(sys.executable)
        self.executable_binding = ExecutableResolver().resolve(
            ExecutableResolutionRequest(
                token=str(python),
                cwd=runtime.working_directory,
                environment={},
                policy=ExecutableTrustPolicy.TRUSTED_HOST_TOOL,
            )
        )

    @contextmanager
    def run_typed_sync_session(
        self,
        definition: WorkerDefinition[RequestT, ResponseT],
        request: RequestT,
        *,
        timeout_seconds: float | None = None,
        maximum_rss_bytes: int | None = None,
        maximum_writable_growth_bytes: int | None = None,
    ) -> Iterator[tuple[ResponseT, Path]]:
        """Run one closed typed protocol and retain staged outputs during consumption."""
        job_root = self._job_root(definition.name)
        job_root.mkdir(parents=True, exist_ok=False)
        request_path, response_path, request_id = self._prepare_request(
            definition, request, job_root
        )
        try:
            outcome = self.broker.run_sync(
                self._execution_request(
                    definition.module,
                    request_path,
                    response_path,
                    job_root,
                    timeout_seconds=(
                        definition.timeout_seconds if timeout_seconds is None else timeout_seconds
                    ),
                    maximum_rss_bytes=maximum_rss_bytes,
                    maximum_writable_growth_bytes=maximum_writable_growth_bytes,
                )
            )
            response = self._load_typed_response(
                process_exit_code(outcome.process.termination),
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
        *,
        timeout_seconds: float | None = None,
        maximum_rss_bytes: int | None = None,
        maximum_writable_growth_bytes: int | None = None,
    ) -> ResponseT:
        with self.run_typed_sync_session(
            definition,
            request,
            timeout_seconds=timeout_seconds,
            maximum_rss_bytes=maximum_rss_bytes,
            maximum_writable_growth_bytes=maximum_writable_growth_bytes,
        ) as (response, _job_root):
            return response

    async def run_typed(
        self,
        definition: WorkerDefinition[RequestT, ResponseT],
        request: RequestT,
    ) -> ResponseT:
        """Run a typed worker asynchronously when no staged output outlives the call."""
        job_root = self._job_root(definition.name)
        job_root.mkdir(parents=True, exist_ok=False)
        request_path, response_path, request_id = self._prepare_request(
            definition, request, job_root
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
                process_exit_code(outcome.process.termination),
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
        maximum_rss_bytes: int | None = None,
        maximum_writable_growth_bytes: int | None = None,
        heartbeat: Callable[[Path], Awaitable[None]] | None = None,
        job_root: Path | None = None,
    ) -> T:
        """Keep typed worker outputs alive while one host-side consumer validates them."""
        job_root = job_root or self._job_root(definition.name)
        job_root.mkdir(parents=True, exist_ok=False)
        request_path, response_path, request_id = self._prepare_request(
            definition, request, job_root
        )
        try:
            task = asyncio.create_task(
                self.broker.run(
                    self._execution_request(
                        definition.module,
                        request_path,
                        response_path,
                        job_root,
                        timeout_seconds=(
                            definition.timeout_seconds
                            if timeout_seconds is None
                            else timeout_seconds
                        ),
                        maximum_rss_bytes=maximum_rss_bytes,
                        maximum_writable_growth_bytes=maximum_writable_growth_bytes,
                    )
                )
            )
            try:
                while not task.done():
                    done, _pending = await asyncio.wait({task}, timeout=0.25)
                    if task in done:
                        break
                    if heartbeat is not None:
                        await heartbeat(job_root)
                outcome = await task
            except asyncio.CancelledError:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                raise
            if heartbeat is not None:
                await heartbeat(job_root)
            response = self._load_typed_response(
                process_exit_code(outcome.process.termination),
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
                ErrorCode.MISSING_OR_CHANGED_INPUT,
                "Worker output size does not match its declaration.",
            )
        actual = sha256_id(digest.hexdigest())
        if actual != output.sha256:
            raise DomainError(
                ErrorCode.MISSING_OR_CHANGED_INPUT,
                "Worker output digest does not match its declaration.",
            )
        return path

    def read_staged_bytes(
        self,
        job_root: Path,
        relative_path: str,
        *,
        max_bytes: int,
    ) -> bytes | None:
        """Read one optional bounded worker side-channel without trusting its path type."""
        path = job_root / relative_path
        if not path.exists():
            return None
        filesystem = BoundedFileSystem((job_root,))
        with filesystem.open_regular(
            path,
            max_bytes=max_bytes,
            require_single_link=True,
        ) as descriptor:
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, min(4_096, max_bytes)):
                chunks.append(chunk)
        return b"".join(chunks)

    def _job_root(self, name: str) -> Path:
        return self.runtime.staging_root / "artifact-workers" / f"{name}-{secrets.token_hex(16)}"

    def _prepare_request(
        self,
        definition: WorkerDefinition[RequestT, ResponseT],
        request: RequestT,
        job_root: Path,
    ) -> tuple[Path, Path, str]:
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
        return request_path, response_path, request_id

    def _execution_request(
        self,
        module: str,
        request_path: Path,
        response_path: Path,
        job_root: Path,
        *,
        timeout_seconds: float,
        maximum_rss_bytes: int | None = None,
        maximum_writable_growth_bytes: int | None = None,
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
            cwd=self.runtime.working_directory,
            environment_allowlist=self.runtime.child_environment_allowlist,
            allowed_working_roots=(self.runtime.working_directory,),
            timeout_seconds=timeout_seconds,
            max_output_bytes=1_048_576,
            resource_policy=ResourcePolicy(
                filesystem_path=self.runtime.filesystem_path,
                staging_root=self.runtime.staging_root,
                writable_roots=(job_root,),
                minimum_free_bytes=self.runtime.minimum_free_bytes,
                maximum_rss_bytes=(maximum_rss_bytes or self.runtime.maximum_rss_bytes),
                maximum_writable_growth_bytes=maximum_writable_growth_bytes,
                sampling_interval_ms=self.runtime.resource_sampling_interval_ms,
                max_observed_files=self.runtime.max_resource_observed_files,
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
                ErrorCode.DECODE_FAILURE,
                f"{definition.name} worker transport failed before a trustworthy response.",
                details={
                    "exit_code": exit_code,
                    "stderr": stderr.decode(errors="replace")[-2_000:],
                },
            )
        try:
            raw = BoundedFileSystem((response_path.parent,)).read_bytes(
                response_path,
                max_bytes=self.runtime.max_response_bytes,
                require_single_link=True,
            )
            envelope = TYPED_WORKER_RESPONSE.validate_json(raw)
        except (OSError, ValidationError, ValueError) as exc:
            raise DomainError(
                ErrorCode.DECODE_FAILURE,
                f"{definition.name} worker response violates the transport contract.",
            ) from exc
        if (
            envelope.request_id != request_id
            or envelope.operation is not definition.operation
            or envelope.implementation != definition.implementation
        ):
            raise DomainError(
                ErrorCode.MISSING_OR_CHANGED_INPUT,
                f"{definition.name} worker response is bound to another request.",
            )
        if isinstance(envelope, WorkerSucceeded):
            if exit_code != 0:
                raise DomainError(
                    ErrorCode.DECODE_FAILURE,
                    f"{definition.name} worker success envelope has a failure exit status.",
                )
            try:
                return definition.response.validate_python(envelope.payload)
            except ValidationError as exc:
                raise DomainError(
                    ErrorCode.DECODE_FAILURE,
                    f"{definition.name} worker success payload violates its exact schema.",
                ) from exc
        if not isinstance(envelope, WorkerFailed) or exit_code != 1:
            raise DomainError(
                ErrorCode.DECODE_FAILURE,
                f"{definition.name} worker failure envelope has an invalid exit status.",
            )
        raise DomainError(
            worker_failure_error_code(envelope.failure.kind),
            envelope.failure.message,
        )
