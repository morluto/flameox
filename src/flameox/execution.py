from __future__ import annotations

import asyncio
import os
import shutil
import signal
import stat
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import IO, Annotated, Any, Literal, cast

import anyio
import psutil
from pydantic import Field, field_validator, model_validator

from flameox.command_binding import ExecutableResolver
from flameox.environment_policy import is_dangerous_environment_name, is_safe_control_override
from flameox.executable_models import ResolvedExecutable
from flameox.models import ContractModel
from flameox.process_models import (
    ProcessCancellationCause,
    ProcessResult,
    ResourcePolicyCancellationCause,
    RuntimeResourceSummary,
    process_termination_from_returncode,
)
from flameox.runtime_errors import DomainError, ErrorCode

INSTALLER_ENVIRONMENT_ALLOWLIST = (
    "PATH",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_INDEX",
    "UV_NATIVE_TLS",
    "UV_OFFLINE",
    "UV_CACHE_DIR",
    "UV_PYTHON_DOWNLOADS",
    "UV_NO_CONFIG",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
)
_SINK_PREVIEW_BYTES = 64 * 1024


class ProcessContainment(StrEnum):
    BROKER = "broker"
    PROCESS = "process"
    PROCESS_GROUP = "process_group"
    SYSTEMD_SCOPE = "systemd_scope"


class ProcessDiscoverySource(StrEnum):
    ROOT = "root"
    ANCESTRY = "ancestry"
    PREVIOUSLY_OBSERVED = "previously_observed"
    CONTAINMENT = "containment"


class ProcessSnapshotPhase(StrEnum):
    RUNNING = "running"
    PRE_CLEANUP = "pre_cleanup"
    POST_CLEANUP = "post_cleanup"
    POST_ROOT_EXIT = "post_root_exit"


class ResourcePolicy(ContractModel):
    filesystem_path: Path
    staging_root: Path | None = None
    writable_roots: tuple[Path, ...] = ()
    minimum_free_bytes: int = Field(ge=0)
    maximum_rss_bytes: int | None = Field(default=None, gt=0)
    sampling_interval_ms: int = Field(default=250, ge=25, le=10_000)
    max_observed_files: int = Field(default=10_000, ge=1, le=1_000_000)
    maximum_writable_growth_bytes: int | None = Field(default=None, gt=0)


class ExecutionRequest(ContractModel):
    argv: tuple[str, ...]
    executable_binding: ResolvedExecutable
    cwd: Path
    stdin_bytes: bytes | None = None
    environment_allowlist: tuple[str, ...] = ("PATH",)
    environment_overrides: dict[str, str] = Field(default_factory=dict)
    allowed_working_roots: tuple[Path, ...]
    timeout_seconds: float | None = Field(default=300, gt=0, le=86_400)
    graceful_shutdown_seconds: float = Field(default=5, ge=0, le=60)
    max_output_bytes: int = Field(default=16_777_216, gt=0)
    diagnostic_bytes: int | None = Field(
        default=None,
        ge=0,
        exclude=True,
        description="Internal bounded per-stream diagnostic prefix size.",
    )
    output_directory: Path | None = Field(
        default=None,
        exclude=True,
        description="Internal request-owned directory for exact stdout/stderr prefixes.",
    )
    output_root: Path | None = Field(
        default=None,
        exclude=True,
        description="Internal owner root containing output_directory.",
    )
    systemd_scope_unit: str | None = None
    resource_policy: ResourcePolicy | None = None
    inherited_directory_fds: tuple[Annotated[int, Field(ge=0)], ...] = Field(
        default=(),
        exclude=True,
        max_length=8,
    )

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("argv must include an executable")
        if not value[0]:
            raise ValueError("argv[0] must identify an executable")
        if any("\x00" in item for item in value):
            raise ValueError("argv entries cannot contain NUL")
        return value

    @model_validator(mode="after")
    def validate_executable_binding(self) -> ExecutionRequest:
        binding = self.executable_binding
        if self.argv[0] not in {
            binding.requested_token,
            str(binding.invocation_path),
            str(binding.canonical_target),
        }:
            raise ValueError("argv[0] must identify the bound executable")
        return self

    @model_validator(mode="after")
    def inherited_descriptors_are_unique_directories(self) -> ExecutionRequest:
        if len(set(self.inherited_directory_fds)) != len(self.inherited_directory_fds):
            raise ValueError("inherited directory descriptors must be unique")
        if self.inherited_directory_fds and os.name != "posix":
            raise ValueError("inherited directory descriptors require a POSIX subprocess")
        for descriptor in self.inherited_directory_fds:
            try:
                metadata = os.fstat(descriptor)
            except OSError as exc:
                raise ValueError("inherited directory descriptor is not open") from exc
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("only directory descriptors can be inherited")
        return self

    @model_validator(mode="after")
    def validate_output_directory(self) -> ExecutionRequest:
        if self.diagnostic_bytes is not None and self.output_directory is not None:
            raise ValueError("diagnostic_bytes cannot be combined with output_directory")
        if self.output_directory is None:
            return self
        if not self.output_directory.is_absolute() or "\x00" in str(self.output_directory):
            raise ValueError("output_directory must be an absolute path without NUL")
        output_directory = self.output_directory.resolve(strict=False)
        if self.output_root is None or not self.output_root.is_absolute():
            raise ValueError("output_root is required and must be absolute with output_directory")
        output_root = self.output_root.resolve(strict=False)
        if self.output_root.exists() and self.output_root.is_symlink():
            raise ValueError("output_root cannot be a symlink")
        if self.output_directory.exists() and self.output_directory.is_symlink():
            raise ValueError("output_directory cannot be a symlink")
        if not (output_directory == output_root or output_directory.is_relative_to(output_root)):
            raise ValueError("output_directory must be beneath output_root")
        return self

    @field_validator("systemd_scope_unit")
    @classmethod
    def validate_scope_unit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.endswith(".scope") or "/" in value or "\\" in value or "\x00" in value:
            raise ValueError("systemd scope unit must be a simple .scope unit name")
        return value


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """Settled execution; stdout/stderr are previews when output_sink is present."""

    process: ProcessResult
    stdout: bytes
    stderr: bytes
    resolved_executable: Path
    containment: ProcessContainment
    executable_binding: ResolvedExecutable
    process_observations: tuple[ProcessObservation, ...] = ()
    output_sink: OutputSink | None = None
    diagnostic_output: DiagnosticOutput | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticOutput:
    """Bounded in-memory console diagnostic accounting, without output content."""

    stdout_observed_bytes: int
    stderr_observed_bytes: int
    stdout_retained_bytes: int
    stderr_retained_bytes: int
    stdout_omitted_bytes: int
    stderr_omitted_bytes: int
    stdout_complete: bool
    stderr_complete: bool

    def as_details(self) -> dict[str, object]:
        return {
            "stdout_observed_bytes": self.stdout_observed_bytes,
            "stderr_observed_bytes": self.stderr_observed_bytes,
            "stdout_retained_bytes": self.stdout_retained_bytes,
            "stderr_retained_bytes": self.stderr_retained_bytes,
            "stdout_omitted_bytes": self.stdout_omitted_bytes,
            "stderr_omitted_bytes": self.stderr_omitted_bytes,
            "stdout_complete": self.stdout_complete,
            "stderr_complete": self.stderr_complete,
        }


@dataclass(frozen=True, slots=True)
class OutputSink:
    """Exact stream prefixes on disk plus the smaller in-memory response previews."""

    directory: Path
    stdout_path: Path
    stderr_path: Path
    stdout_bytes: int
    stderr_bytes: int
    stdout_preview_bytes: int
    stderr_preview_bytes: int
    io_error: bool
    stdout_complete: bool
    stderr_complete: bool

    @property
    def complete(self) -> bool:
        return self.stdout_complete and self.stderr_complete

    def as_details(self) -> dict[str, object]:
        return {
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "stdout_preview_bytes": self.stdout_preview_bytes,
            "stderr_preview_bytes": self.stderr_preview_bytes,
            "stdout_complete": self.stdout_complete,
            "stderr_complete": self.stderr_complete,
            "complete": self.complete,
            "io_error": self.io_error,
        }


class ProcessObservation(ContractModel):
    """Privacy-bounded process evidence captured around broker cleanup."""

    pid: int = Field(gt=0)
    create_time: float | None = None
    parent_pid: int | None = None
    parent_create_time: float | None = None
    discovery_source: ProcessDiscoverySource
    name: str | None = None
    status: str | None = None
    rss_bytes: int | None = None
    cpu_user_seconds: float | None = None
    cpu_system_seconds: float | None = None
    thread_count: int | None = None
    fd_count: int | None = None
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    snapshot_phase: ProcessSnapshotPhase
    alive_before_cleanup: bool | None = None
    cleanup_action: str | None = None
    cleanup_outcome: str | None = None
    failures: tuple[str, ...] = ()


class ProcessExecutionError(DomainError):
    """A process failure with typed internal state and a public wire projection."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        process: ProcessResult,
        process_observations: tuple[ProcessObservation, ...],
        stdout: bytes | None = None,
        stderr: bytes | None = None,
        output_sink: OutputSink | None = None,
        diagnostic_output: DiagnosticOutput | None = None,
        retryable: bool = False,
    ) -> None:
        self.process = process
        self.process_observations = process_observations
        self.stdout = stdout
        self.stderr = stderr
        self.output_sink = output_sink
        self.diagnostic_output = diagnostic_output
        super().__init__(
            code,
            message,
            details={
                "process": process.model_dump(mode="json"),
                "process_observations": [
                    item.model_dump(mode="json") for item in process_observations
                ],
                **({"output_sink": output_sink.as_details()} if output_sink is not None else {}),
                **(
                    {"diagnostic_output": diagnostic_output.as_details()}
                    if diagnostic_output is not None
                    else {}
                ),
            },
            retryable=retryable,
        )


class ProcessCancelledError(asyncio.CancelledError):
    """Cancellation carrying bounded evidence collected before process cleanup."""

    def __init__(
        self,
        *,
        process: ProcessResult,
        process_observations: tuple[ProcessObservation, ...],
        stdout: bytes,
        stderr: bytes,
        output_sink: OutputSink | None = None,
        diagnostic_output: DiagnosticOutput | None = None,
    ) -> None:
        super().__init__("process execution cancelled")
        self.process = process
        self.process_observations = process_observations
        self.stdout = stdout
        self.stderr = stderr
        self.output_sink = output_sink
        self.diagnostic_output = diagnostic_output


class _OutputLimitExceeded(Exception):
    pass


class _OutputSinkFailure(Exception):
    pass


class _ResourcePolicyExceeded(Exception):
    def __init__(
        self,
        summary: RuntimeResourceSummary,
        cause: ResourcePolicyCancellationCause,
    ) -> None:
        self.summary = summary
        self.cause = cause


@dataclass(slots=True)
class _OutputBudget:
    remaining: int

    def consume(self, byte_count: int) -> None:
        self.remaining -= byte_count
        if self.remaining < 0:
            raise _OutputLimitExceeded

    @property
    def exceeded(self) -> bool:
        return self.remaining < 0


@dataclass(slots=True)
class _DiagnosticOutputState:
    limit: int
    observed: dict[str, int]
    retained: dict[str, int]
    complete: dict[str, bool]

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.observed = {"stdout": 0, "stderr": 0}
        self.retained = {"stdout": 0, "stderr": 0}
        self.complete = {"stdout": False, "stderr": False}

    def record(self, stream: Literal["stdout", "stderr"], size: int, retained: int) -> None:
        self.observed[stream] += size
        self.retained[stream] += retained

    def finish(self, stream: Literal["stdout", "stderr"], complete: bool) -> None:
        self.complete[stream] = self.complete[stream] or complete

    def mark_incomplete(self, *streams: Literal["stdout", "stderr"]) -> None:
        for stream in streams:
            self.complete[stream] = False

    def snapshot(self) -> DiagnosticOutput:
        return DiagnosticOutput(
            stdout_observed_bytes=self.observed["stdout"],
            stderr_observed_bytes=self.observed["stderr"],
            stdout_retained_bytes=self.retained["stdout"],
            stderr_retained_bytes=self.retained["stderr"],
            stdout_omitted_bytes=self.observed["stdout"] - self.retained["stdout"],
            stderr_omitted_bytes=self.observed["stderr"] - self.retained["stderr"],
            stdout_complete=self.complete["stdout"],
            stderr_complete=self.complete["stderr"],
        )


class _AsyncOutputSink:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink():
            raise ValueError("output sink directory cannot be a symlink")
        self.stdout_path = directory / "stdout"
        self.stderr_path = directory / "stderr"
        self._streams: dict[str, IO[bytes]] = {}
        try:
            for name, path in (("stdout", self.stdout_path), ("stderr", self.stderr_path)):
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(path, flags, 0o600)
                self._streams[name] = os.fdopen(descriptor, "wb", buffering=0)
        except BaseException:
            for stream in self._streams.values():
                with suppress(OSError):
                    stream.close()
            raise
        self._bytes = {"stdout": 0, "stderr": 0}
        self._preview_bytes = {"stdout": 0, "stderr": 0}
        self._complete = {"stdout": False, "stderr": False}
        self._io_error = False

    def write(self, stream: Literal["stdout", "stderr"], content: bytes) -> None:
        remaining = memoryview(content)
        while remaining:
            written = self._streams[stream].write(remaining)
            if written is None or written <= 0:
                raise OSError("Output sink write made no progress")
            self._bytes[stream] += written
            remaining = remaining[written:]

    @staticmethod
    async def _run_owned_io[T](operation: Callable[[], T]) -> T:
        # A raw asyncio cancellation must join the thread, not merely abandon
        # its future. The child task's AnyIO thread wait is joined by the group.
        result: Future[T] = Future()

        async def execute() -> None:
            try:
                result.set_result(await anyio.to_thread.run_sync(operation))
            except BaseException as error:
                result.set_exception(error)

        async with anyio.create_task_group() as group:
            group.start_soon(execute)
        return result.result()

    async def write_async(self, stream: Literal["stdout", "stderr"], content: bytes) -> None:
        try:
            await self._run_owned_io(lambda: self.write(stream, content))
        except OSError as exc:
            self._io_error = True
            raise _OutputSinkFailure from exc

    def finish(
        self, stream: Literal["stdout", "stderr"], complete: bool, preview_bytes: int
    ) -> None:
        self._complete[stream] = complete
        self._preview_bytes[stream] = preview_bytes

    def mark_incomplete(self) -> None:
        for stream in self._complete:
            self._complete[stream] = False

    def _close(self) -> OutputSink:
        for stream in self._streams.values():
            try:
                stream.close()
            except OSError:
                self._io_error = True
        return OutputSink(
            directory=self.directory,
            stdout_path=self.stdout_path,
            stderr_path=self.stderr_path,
            stdout_bytes=self._bytes["stdout"],
            stderr_bytes=self._bytes["stderr"],
            stdout_preview_bytes=self._preview_bytes["stdout"],
            stderr_preview_bytes=self._preview_bytes["stderr"],
            io_error=self._io_error,
            stdout_complete=self._complete["stdout"] and not self._io_error,
            stderr_complete=self._complete["stderr"] and not self._io_error,
        )

    async def close(self) -> OutputSink:
        with anyio.CancelScope(shield=True):
            result = await self._run_owned_io(self._close)
        return result


class SubprocessBroker:
    _MAX_OBSERVED_PROCESSES = 10_000

    def run_sync(
        self,
        request: ExecutionRequest,
        *,
        on_started: Callable[[int], Awaitable[None]] | None = None,
        on_cleanup: Callable[[bool], Awaitable[None]] | None = None,
    ) -> ExecutionOutcome:
        """Run through the async broker while preserving synchronous adapter APIs."""

        try:
            anyio.from_thread.check_cancelled()
        except RuntimeError:
            # Ordinary synchronous callers have no owning AnyIO request scope.
            pass
        else:
            return self._run_from_anyio_worker(request, on_started, on_cleanup)

        async def execute() -> ExecutionOutcome:
            return await self.run(request, on_started=on_started, on_cleanup=on_cleanup)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(execute())

        result: list[ExecutionOutcome] = []
        failure: list[BaseException] = []

        def run_in_thread() -> None:
            try:
                result.append(asyncio.run(execute()))
            except BaseException as exc:
                failure.append(exc)

        thread = threading.Thread(target=run_in_thread, name="flameox-subprocess", daemon=True)
        thread.start()
        thread.join()
        if failure:
            raise failure[0]
        if not result:
            raise RuntimeError("subprocess broker did not return an outcome")
        return result[0]

    def _run_from_anyio_worker(
        self,
        request: ExecutionRequest,
        on_started: Callable[[int], Awaitable[None]] | None,
        on_cleanup: Callable[[bool], Awaitable[None]] | None,
    ) -> ExecutionOutcome:
        async def execute() -> ExecutionOutcome | ProcessCancelledError:
            try:
                return await self.run(request, on_started=on_started, on_cleanup=on_cleanup)
            except ProcessCancelledError as error:
                # AnyIO translates asyncio cancellation into a concurrent-futures
                # exception across this boundary. Return the settled receipt so its
                # native output and cleanup evidence survive that translation.
                return error

        try:
            result = anyio.from_thread.run(execute)
        except FutureCancelledError:
            # Cancellation before the broker starts has no process receipt.
            raise asyncio.CancelledError from None
        if isinstance(result, ProcessCancelledError):
            raise result
        return result

    async def run(  # noqa: C901
        self,
        request: ExecutionRequest,
        *,
        on_started: Callable[[int], Awaitable[None]] | None = None,
        on_cleanup: Callable[[bool], Awaitable[None]] | None = None,
    ) -> ExecutionOutcome:
        cwd = self._resolve_cwd(request.cwd, request.allowed_working_roots)
        environment = self._build_environment(request)
        binding = self._bound_executable(request)
        executable = binding.invocation_path
        argv = (str(executable), *request.argv[1:])
        started = time.monotonic_ns()
        deadline = (
            None
            if request.timeout_seconds is None
            else asyncio.get_running_loop().time() + request.timeout_seconds
        )
        process_observations: list[ProcessObservation] = []
        policy = request.resource_policy
        initial_sizes = (
            {
                str(root): self._bounded_tree_size(root, max_files=policy.max_observed_files)
                for root in policy.writable_roots
            }
            if policy is not None
            else {}
        )
        initial_staging = (
            self._bounded_tree_size(policy.staging_root, max_files=policy.max_observed_files)
            if policy is not None and policy.staging_root is not None
            else None
        )
        output_sink = (
            _AsyncOutputSink(request.output_directory)
            if request.output_directory is not None
            else None
        )
        diagnostic_state = (
            _DiagnosticOutputState(request.diagnostic_bytes)
            if request.diagnostic_bytes is not None
            else None
        )
        output_sink_metadata: OutputSink | None = None
        try:
            async with asyncio.timeout_at(deadline):
                await anyio.lowlevel.checkpoint_if_cancelled()
                # Acquire the handle before request-scope cancellation can unwind.
                # asyncio's transport startup cleanup is not level-cancellation safe.
                # The broker deadline still bounds this acquisition.
                with anyio.CancelScope(shield=True):
                    process = await asyncio.create_subprocess_exec(
                        *argv,
                        cwd=cwd,
                        env=environment,
                        stdin=(
                            asyncio.subprocess.PIPE
                            if request.stdin_bytes is not None
                            else asyncio.subprocess.DEVNULL
                        ),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        start_new_session=os.name == "posix",
                        pass_fds=request.inherited_directory_fds,
                    )
        except TimeoutError as exc:
            if output_sink is not None:
                output_sink.mark_incomplete()
                output_sink_metadata = await output_sink.close()
            cleanup_complete = True
            if on_cleanup is not None:
                with anyio.CancelScope(shield=True):
                    await on_cleanup(cleanup_complete)
            timeout_process = ProcessResult(
                wall_time_ns=time.monotonic_ns() - started,
                cancellation_cause=ProcessCancellationCause.TIMEOUT,
                cleanup_complete=cleanup_complete,
            )
            raise ProcessExecutionError(
                ErrorCode.EXECUTION_TIMEOUT,
                f"Process exceeded {request.timeout_seconds} seconds.",
                process=timeout_process,
                process_observations=tuple(process_observations),
                output_sink=output_sink_metadata,
                diagnostic_output=diagnostic_state.snapshot() if diagnostic_state else None,
                retryable=True,
            ) from exc
        except asyncio.CancelledError:
            if output_sink is not None:
                output_sink.mark_incomplete()
                output_sink_metadata = await output_sink.close()
            if on_cleanup is not None:
                with anyio.CancelScope(shield=True):
                    await on_cleanup(True)
            raise ProcessCancelledError(
                process=ProcessResult(
                    wall_time_ns=time.monotonic_ns() - started,
                    cancellation_cause=ProcessCancellationCause.CALLER_CANCELLED,
                    cleanup_complete=True,
                ),
                process_observations=(),
                stdout=b"",
                stderr=b"",
                output_sink=output_sink_metadata,
                diagnostic_output=diagnostic_state.snapshot() if diagnostic_state else None,
            ) from None
        except BaseException:
            if output_sink is not None:
                output_sink.mark_incomplete()
                output_sink_metadata = await output_sink.close()
            raise
        assert process.stdout is not None
        assert process.stderr is not None

        output_budget = _OutputBudget(request.max_output_bytes)
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        tracked_descendants: dict[int, float | None] = {}
        stdout_task = asyncio.create_task(
            self._read_bounded(
                process.stdout,
                output_budget,
                output=stdout_buffer,
                sink=output_sink,
                stream_name="stdout",
                diagnostic_limit=request.diagnostic_bytes,
                diagnostic_state=diagnostic_state,
            )
        )
        stderr_task = asyncio.create_task(
            self._read_bounded(
                process.stderr,
                output_budget,
                output=stderr_buffer,
                sink=output_sink,
                stream_name="stderr",
                diagnostic_limit=request.diagnostic_bytes,
                diagnostic_state=diagnostic_state,
            )
        )
        stop_observation = asyncio.Event()
        resource_task = asyncio.create_task(
            self._observe_resources(
                process,
                request.resource_policy,
                tracked_descendants,
                stop_observation,
                initial_sizes,
                initial_staging,
            )
        )
        stdin_task: asyncio.Task[None] | None = None
        if request.stdin_bytes is not None:
            assert process.stdin is not None
            stdin_task = asyncio.create_task(self._write_stdin(process.stdin, request.stdin_bytes))
        try:
            async with asyncio.timeout_at(deadline):
                if on_started is not None:
                    await on_started(process.pid)
                process_observations.extend(
                    self._snapshot_processes(
                        process.pid,
                        ProcessSnapshotPhase.RUNNING,
                        True,
                        None,
                        None,
                        deadline=deadline,
                    )
                )
                self._track_observed_descendants(
                    process.pid,
                    process_observations,
                    tracked_descendants,
                )
                observed_identities = self._observation_identities(process_observations)

                async def wait_for_exit() -> int:
                    await self._wait_root_exit(process)
                    stop_observation.set()
                    if not await self._terminate(process, request, tracked_descendants):
                        raise OSError("Descendant cleanup did not complete")
                    assert process.returncode is not None
                    return process.returncode

                results = await asyncio.gather(
                    asyncio.shield(stdout_task),
                    asyncio.shield(stderr_task),
                    wait_for_exit(),
                    asyncio.shield(resource_task),
                    *([asyncio.shield(stdin_task)] if stdin_task is not None else []),
                )
                stdout = cast(bytes, results[0])
                stderr = cast(bytes, results[1])
                resources = cast(RuntimeResourceSummary | None, results[3])
        except BaseException as exc:
            # Cleanup belongs to the broker, including under AnyIO's repeated
            # cancellation. Callers must not need their own task/shield wrappers.
            with anyio.CancelScope(shield=True):
                try:
                    cleanup_complete = await self._terminate_with_observation(
                        process, request, process_observations, on_cleanup, tracked_descendants
                    )
                finally:
                    stop_observation.set()
                    await self._settle_readers(stdout_task, stderr_task)
                    if diagnostic_state is not None:
                        # A terminated request cannot claim that no inherited
                        # writer remains, even if cleanup happened to produce
                        # EOF while the readers were settling.
                        diagnostic_state.mark_incomplete("stdout", "stderr")
                    self._close_async_process_transport(process)
                    resources = await self._collect_resource(resource_task)
                    await self._settle_task(stdin_task)
            partial_stdout, partial_stderr = bytes(stdout_buffer), bytes(stderr_buffer)
            if output_sink is not None:
                output_sink.mark_incomplete()
                output_sink_metadata = await output_sink.close()
            retryable = False
            if isinstance(exc, TimeoutError):
                cause = ProcessCancellationCause.TIMEOUT
                code = ErrorCode.EXECUTION_TIMEOUT
                message = f"Process exceeded {request.timeout_seconds} seconds."
                retryable = True
            elif isinstance(exc, _OutputLimitExceeded):
                cause = ProcessCancellationCause.OUTPUT_LIMIT
                code = ErrorCode.LIMIT_EXCEEDED
                message = f"Process output exceeded {request.max_output_bytes} bytes."
            elif isinstance(exc, _ResourcePolicyExceeded):
                cause = exc.cause
                resources = exc.summary
                code = ErrorCode.LIMIT_EXCEEDED
                message = (
                    "Runtime writable-byte policy was exceeded."
                    if cause
                    in {
                        ProcessCancellationCause.STORAGE_RESERVE_EXCEEDED,
                        ProcessCancellationCause.WRITABLE_LIMIT_EXCEEDED,
                    }
                    else "Process tree exceeded the configured memory budget."
                )
            elif isinstance(exc, _OutputSinkFailure):
                cause = ProcessCancellationCause.IO_FAILURE
                code = ErrorCode.EXECUTION_FAILURE
                message = "Process output could not be written to the request-owned sink."
            elif isinstance(exc, asyncio.CancelledError):
                cause = ProcessCancellationCause.CALLER_CANCELLED
            else:
                raise
            failure_process = ProcessResult(
                termination=process_termination_from_returncode(process.returncode),
                wall_time_ns=time.monotonic_ns() - started,
                cancellation_cause=cause,
                cleanup_complete=cleanup_complete,
                peak_rss_bytes=resources.peak_rss_bytes if resources is not None else None,
                resources=resources,
                stdout=partial_stdout.decode(errors="replace"),
                stderr=partial_stderr.decode(errors="replace"),
            )
            if isinstance(exc, asyncio.CancelledError):
                raise ProcessCancelledError(
                    process=failure_process,
                    process_observations=tuple(process_observations),
                    stdout=partial_stdout,
                    stderr=partial_stderr,
                    output_sink=output_sink_metadata,
                    diagnostic_output=diagnostic_state.snapshot() if diagnostic_state else None,
                ) from None
            raise ProcessExecutionError(
                code,
                message,
                process=failure_process,
                process_observations=tuple(process_observations),
                stdout=partial_stdout,
                stderr=partial_stderr,
                output_sink=output_sink_metadata,
                diagnostic_output=diagnostic_state.snapshot() if diagnostic_state else None,
                retryable=retryable,
            ) from exc

        finished = time.monotonic_ns()
        process_observations.extend(
            self._snapshot_known_processes(
                observed_identities,
                ProcessSnapshotPhase.POST_ROOT_EXIT,
                False,
                None,
                None,
            )
        )
        result = ProcessResult(
            termination=process_termination_from_returncode(process.returncode),
            wall_time_ns=finished - started,
            cleanup_complete=True,
            peak_rss_bytes=resources.peak_rss_bytes if resources is not None else None,
            resources=resources,
        )
        if output_sink is not None:
            output_sink_metadata = await output_sink.close()
        if output_sink_metadata is not None and output_sink_metadata.io_error:
            raise ProcessExecutionError(
                ErrorCode.EXECUTION_FAILURE,
                "Process output could not be finalized in the request-owned sink.",
                process=result.model_copy(
                    update={"cancellation_cause": ProcessCancellationCause.IO_FAILURE}
                ),
                process_observations=tuple(process_observations),
                stdout=stdout,
                stderr=stderr,
                output_sink=output_sink_metadata,
                diagnostic_output=diagnostic_state.snapshot() if diagnostic_state else None,
            )
        return ExecutionOutcome(
            process=result,
            stdout=stdout,
            stderr=stderr,
            resolved_executable=executable,
            executable_binding=binding,
            containment=(
                ProcessContainment.SYSTEMD_SCOPE
                if request.systemd_scope_unit is not None
                else (
                    ProcessContainment.PROCESS_GROUP
                    if os.name == "posix"
                    else ProcessContainment.PROCESS
                )
            ),
            process_observations=tuple(process_observations),
            output_sink=output_sink_metadata,
            diagnostic_output=diagnostic_state.snapshot() if diagnostic_state else None,
        )

    async def _terminate_with_observation(
        self,
        process: asyncio.subprocess.Process,
        request: ExecutionRequest,
        observations: list[ProcessObservation],
        on_cleanup: Callable[[bool], Awaitable[None]] | None,
        tracked_descendants: dict[int, float | None],
    ) -> bool:
        observations.extend(
            self._snapshot_processes(
                process.pid,
                ProcessSnapshotPhase.PRE_CLEANUP,
                True,
                "terminate",
                None,
            )
        )
        identities = self._observation_identities(observations)
        cleanup_complete = await asyncio.shield(
            self._terminate(process, request, tracked_descendants)
        )
        observations.extend(
            self._snapshot_known_processes(
                identities,
                ProcessSnapshotPhase.POST_CLEANUP,
                False,
                "terminate",
                str(cleanup_complete),
            )
        )
        if on_cleanup is not None:
            await asyncio.shield(on_cleanup(cleanup_complete))
        return cleanup_complete

    def _snapshot_processes(
        self,
        root_pid: int,
        phase: ProcessSnapshotPhase,
        alive_before_cleanup: bool | None,
        cleanup_action: str | None,
        cleanup_outcome: str | None,
        *,
        deadline: float | None = None,
    ) -> tuple[ProcessObservation, ...]:
        started = time.monotonic()
        observation_deadline = min(
            started + 2.0,
            deadline if deadline is not None else started + 2.0,
        )
        processes, truncated = self._enumerate_processes(root_pid, observation_deadline)
        observations = tuple(
            self._observe_process(
                process,
                source,
                phase,
                alive_before_cleanup,
                cleanup_action,
                cleanup_outcome,
                deadline=observation_deadline,
            )
            for process, source in processes
            if time.monotonic() <= observation_deadline
        )
        if truncated and observations:
            observations = (
                observations[0].validated_copy(
                    update={
                        "failures": (
                            *observations[0].failures,
                            "descendant_enumeration_truncated",
                        )
                    }
                ),
                *observations[1:],
            )
        return observations

    def _enumerate_processes(
        self, root_pid: int, deadline: float
    ) -> tuple[list[tuple[psutil.Process, ProcessDiscoverySource]], bool]:
        processes: list[tuple[psutil.Process, ProcessDiscoverySource]] = []
        pending: deque[tuple[psutil.Process, ProcessDiscoverySource]] = deque()
        truncated = False
        try:
            pending.append((psutil.Process(root_pid), ProcessDiscoverySource.ROOT))
            while pending and len(processes) < self._MAX_OBSERVED_PROCESSES:
                if time.monotonic() > deadline:
                    truncated = True
                    break
                process, source = pending.popleft()
                processes.append((process, source))
                try:
                    children = process.children(recursive=False)
                except psutil.Error:
                    children = []
                    truncated = True
                if time.monotonic() > deadline:
                    truncated = True
                    break
                for child in children:
                    if len(processes) + len(pending) >= self._MAX_OBSERVED_PROCESSES:
                        truncated = True
                        break
                    pending.append((child, ProcessDiscoverySource.ANCESTRY))
            truncated = truncated or bool(pending)
        except psutil.Error:
            return [], False
        return processes, truncated

    def _snapshot_known_processes(
        self,
        identities: tuple[tuple[int, float | None, ProcessDiscoverySource], ...],
        phase: ProcessSnapshotPhase,
        alive_before_cleanup: bool | None,
        cleanup_action: str | None,
        cleanup_outcome: str | None,
    ) -> tuple[ProcessObservation, ...]:
        started = time.monotonic()
        observations: list[ProcessObservation] = []
        for pid, create_time, _source in identities:
            if time.monotonic() > started + 2.0:
                break
            try:
                process = psutil.Process(pid)
            except psutil.Error:
                observations.append(
                    ProcessObservation(
                        pid=pid,
                        create_time=create_time,
                        discovery_source=ProcessDiscoverySource.PREVIOUSLY_OBSERVED,
                        snapshot_phase=phase,
                        alive_before_cleanup=alive_before_cleanup,
                        cleanup_action=cleanup_action,
                        cleanup_outcome=cleanup_outcome,
                        failures=("process_unavailable",),
                    )
                )
                continue
            current_create_time: float | None
            try:
                current_create_time = process.create_time()
            except psutil.Error:
                current_create_time = None
            if (
                create_time is not None
                and current_create_time is not None
                and create_time != current_create_time
            ):
                observations.append(
                    ProcessObservation(
                        pid=pid,
                        create_time=current_create_time,
                        discovery_source=ProcessDiscoverySource.PREVIOUSLY_OBSERVED,
                        snapshot_phase=phase,
                        alive_before_cleanup=alive_before_cleanup,
                        cleanup_action=cleanup_action,
                        cleanup_outcome=cleanup_outcome,
                        failures=("pid_reused",),
                    )
                )
                continue
            observations.append(
                self._observe_process(
                    process,
                    ProcessDiscoverySource.PREVIOUSLY_OBSERVED,
                    phase,
                    alive_before_cleanup,
                    cleanup_action,
                    cleanup_outcome,
                    deadline=started + 2.0,
                )
            )
        return tuple(observations)

    @staticmethod
    def _observation_identities(
        observations: list[ProcessObservation],
    ) -> tuple[tuple[int, float | None, ProcessDiscoverySource], ...]:
        return tuple(
            dict.fromkeys(
                (item.pid, item.create_time, item.discovery_source) for item in observations
            )
        )

    @staticmethod
    def _observe_process(
        process: psutil.Process,
        source: ProcessDiscoverySource,
        phase: ProcessSnapshotPhase,
        alive_before_cleanup: bool | None,
        cleanup_action: str | None,
        cleanup_outcome: str | None,
        *,
        deadline: float | None,
    ) -> ProcessObservation:
        failures: list[str] = []

        def read(name: str, default: Any = None) -> Any:
            if deadline is not None and time.monotonic() > deadline:
                failures.append("observation_budget_exceeded")
                return default
            try:
                return getattr(process, name)()
            except (psutil.Error, OSError, PermissionError, AttributeError):
                failures.append(name)
                return default

        create_time = read("create_time")
        parent = read("parent")
        parent_pid = None
        parent_create_time = None
        if parent is not None:
            try:
                parent_pid = parent.pid
                parent_create_time = parent.create_time()
            except psutil.Error:
                failures.append("parent_identity")
        memory = read("memory_info")
        cpu = read("cpu_times")
        fd_count = read("num_fds")
        if fd_count is None:
            fd_count = read("num_handles")
        alive = read("is_running")
        return ProcessObservation(
            pid=process.pid,
            create_time=create_time,
            parent_pid=parent_pid,
            parent_create_time=parent_create_time,
            discovery_source=source,
            name=read("name"),
            status=read("status"),
            rss_bytes=getattr(memory, "rss", None),
            cpu_user_seconds=getattr(cpu, "user", None),
            cpu_system_seconds=getattr(cpu, "system", None),
            thread_count=read("num_threads"),
            fd_count=fd_count,
            snapshot_phase=phase,
            alive_before_cleanup=alive_before_cleanup if alive is not None else None,
            cleanup_action=cleanup_action,
            cleanup_outcome=cleanup_outcome,
            failures=tuple(sorted(set(failures))),
        )

    async def _write_stdin(
        self,
        stream: asyncio.StreamWriter,
        stdin_bytes: bytes,
    ) -> None:
        try:
            stream.write(stdin_bytes)
            await stream.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            stream.close()
            with suppress(Exception):
                await stream.wait_closed()

    async def _observe_resources(
        self,
        process: asyncio.subprocess.Process,
        policy: ResourcePolicy | None,
        tracked_descendants: dict[int, float | None],
        stop: asyncio.Event,
        initial_sizes: dict[str, int | None],
        initial_staging: int | None,
    ) -> RuntimeResourceSummary | None:
        interval = 0.1 if policy is None else policy.sampling_interval_ms / 1_000
        if policy is None:
            while process.returncode is None and not stop.is_set():
                self._track_current_descendants(process.pid, tracked_descendants)
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=interval)
            return None
        minimum_free: int | None = None
        peak_rss = 0
        free_sampled = False
        rss_sampled = False
        unavailable: set[str] = set()
        # Check persistent output even if the child exits before the first sample,
        # and once more when the stop notification arrives between samples.
        while True:
            self._track_current_descendants(process.pid, tracked_descendants)
            try:
                free = shutil.disk_usage(policy.filesystem_path).free
            except OSError:
                unavailable.add("minimum_free_bytes")
                free = None
            else:
                free_sampled = True
                minimum_free = free if minimum_free is None else min(minimum_free, cast(int, free))
            rss = 0
            if process.returncode is None:
                try:
                    parent = psutil.Process(process.pid)
                    processes = (parent, *parent.children(recursive=True))
                    for observed in processes:
                        try:
                            rss += observed.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    rss_sampled = True
                    peak_rss = max(peak_rss, rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    unavailable.add("peak_rss_bytes")
            if free is not None and free < policy.minimum_free_bytes:
                summary = self._resource_summary(
                    policy,
                    initial_sizes=initial_sizes,
                    initial_staging=initial_staging,
                    minimum_free=minimum_free,
                    peak_rss=peak_rss,
                    unavailable=unavailable,
                    termination=ProcessCancellationCause.STORAGE_RESERVE_EXCEEDED,
                )
                raise _ResourcePolicyExceeded(
                    summary,
                    ProcessCancellationCause.STORAGE_RESERVE_EXCEEDED,
                )
            if policy.maximum_rss_bytes is not None and rss > policy.maximum_rss_bytes:
                summary = self._resource_summary(
                    policy,
                    initial_sizes=initial_sizes,
                    initial_staging=initial_staging,
                    minimum_free=minimum_free,
                    peak_rss=peak_rss,
                    unavailable=unavailable,
                    termination=ProcessCancellationCause.MEMORY_LIMIT_EXCEEDED,
                )
                raise _ResourcePolicyExceeded(
                    summary,
                    ProcessCancellationCause.MEMORY_LIMIT_EXCEEDED,
                )
            if policy.maximum_writable_growth_bytes is not None:
                growth = self._writable_growth(policy, initial_sizes)
                if growth is None:
                    unavailable.add("writable_root_growth_bytes")
                    summary = self._resource_summary(
                        policy,
                        initial_sizes=initial_sizes,
                        initial_staging=initial_staging,
                        minimum_free=minimum_free,
                        peak_rss=peak_rss,
                        unavailable=unavailable,
                        termination=ProcessCancellationCause.WRITABLE_LIMIT_EXCEEDED,
                    )
                    raise _ResourcePolicyExceeded(
                        summary,
                        ProcessCancellationCause.WRITABLE_LIMIT_EXCEEDED,
                    )
                elif growth > policy.maximum_writable_growth_bytes:
                    summary = self._resource_summary(
                        policy,
                        initial_sizes=initial_sizes,
                        initial_staging=initial_staging,
                        minimum_free=minimum_free,
                        peak_rss=peak_rss,
                        unavailable=unavailable,
                        termination=ProcessCancellationCause.WRITABLE_LIMIT_EXCEEDED,
                    )
                    raise _ResourcePolicyExceeded(
                        summary,
                        ProcessCancellationCause.WRITABLE_LIMIT_EXCEEDED,
                    )
            if process.returncode is not None or stop.is_set():
                break
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval)
        if not free_sampled:
            unavailable.add("minimum_free_bytes")
        if not rss_sampled or peak_rss == 0:
            unavailable.add("peak_rss_bytes")
        return self._resource_summary(
            policy,
            initial_sizes=initial_sizes,
            initial_staging=initial_staging,
            minimum_free=minimum_free,
            peak_rss=peak_rss,
            unavailable=unavailable,
            termination=None,
        )

    def _writable_growth(
        self,
        policy: ResourcePolicy,
        initial_sizes: dict[str, int | None],
    ) -> int | None:
        growth = 0
        for root in policy.writable_roots:
            initial = initial_sizes[str(root)]
            current = self._bounded_tree_size(root, max_files=policy.max_observed_files)
            if initial is None or current is None:
                return None
            growth += max(0, current - initial)
        return growth

    def _resource_summary(
        self,
        policy: ResourcePolicy,
        *,
        initial_sizes: dict[str, int | None],
        initial_staging: int | None,
        minimum_free: int | None,
        peak_rss: int,
        unavailable: set[str],
        termination: ResourcePolicyCancellationCause | None,
    ) -> RuntimeResourceSummary:
        growth: dict[str, int] = {}
        for root in policy.writable_roots:
            initial = initial_sizes[str(root)]
            final = self._bounded_tree_size(root, max_files=policy.max_observed_files)
            if initial is None or final is None:
                unavailable.add(f"writable_root_growth:{root}")
            else:
                growth[str(root)] = max(0, final - initial)
        staging_growth: int | None = None
        if policy.staging_root is not None:
            final_staging = self._bounded_tree_size(
                policy.staging_root,
                max_files=policy.max_observed_files,
            )
            if initial_staging is None or final_staging is None:
                unavailable.add("staging_growth_bytes")
            else:
                nested_growth = sum(growth.values())
                staging_growth = max(0, final_staging - initial_staging - nested_growth)
        return RuntimeResourceSummary(
            sampling_interval_ms=policy.sampling_interval_ms,
            minimum_free_bytes=minimum_free,
            staging_growth_bytes=staging_growth,
            writable_root_growth_bytes=growth,
            peak_rss_bytes=peak_rss or None,
            peak_rss_backend=("psutil_recursive_polling" if peak_rss else None),
            unavailable_metrics=tuple(sorted(unavailable)),
            policy_termination=termination,
        )

    def _bounded_tree_size(self, root: Path, *, max_files: int) -> int | None:
        total = 0
        observed = 0
        try:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                observed += 1
                if observed > max_files:
                    return None
                total += path.stat().st_size
        except OSError:
            return None
        return total

    async def _read_bounded(
        self,
        reader: asyncio.StreamReader,
        budget: _OutputBudget,
        *,
        drain_on_limit: bool = False,
        diagnostic_limit: int | None = None,
        diagnostic_state: _DiagnosticOutputState | None = None,
        output: bytearray | None = None,
        sink: _AsyncOutputSink | None = None,
        stream_name: Literal["stdout", "stderr"] | None = None,
    ) -> bytes:
        output = output if output is not None else bytearray()
        name = stream_name
        complete = False
        try:
            while chunk := await reader.read(64 * 1024):
                if diagnostic_limit is not None:
                    retained = min(
                        len(chunk),
                        max(diagnostic_limit - len(output), 0),
                    )
                    if diagnostic_state is not None and name is not None:
                        diagnostic_state.record(name, len(chunk), retained)
                    output.extend(chunk[:retained])
                    continue
                retained = min(len(chunk), max(budget.remaining, 0))
                exceeded = False
                try:
                    budget.consume(len(chunk))
                except _OutputLimitExceeded:
                    exceeded = True
                if sink is not None and name is not None:
                    await sink.write_async(name, chunk[:retained])
                preview = (
                    retained
                    if sink is None
                    else min(retained, max(_SINK_PREVIEW_BYTES - len(output), 0))
                )
                output.extend(chunk[:preview])
                if exceeded:
                    if not drain_on_limit:
                        raise _OutputLimitExceeded
                    while await reader.read(64 * 1024):
                        pass
                    break
            complete = True
        finally:
            if diagnostic_state is not None and name is not None:
                diagnostic_state.finish(name, complete)
            if sink is not None and name is not None:
                sink.finish(name, complete, len(output))
        return bytes(output)

    async def _terminate(
        self,
        process: asyncio.subprocess.Process,
        request: ExecutionRequest,
        tracked_descendants: dict[int, float | None],
    ) -> bool:
        descendants = self._tracked_processes(
            (*self._descendants(process.pid),),
            tracked_descendants,
            root_pid=process.pid,
        )
        for descendant in descendants:
            with suppress(psutil.Error):
                descendant.terminate()
        if process.returncode is not None:
            if os.name == "posix":
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            return await self._finish_descendant_cleanup(descendants)
        scope_stopped = True
        if request.systemd_scope_unit is not None:
            scope_stopped = await self._stop_systemd_scope(
                request.systemd_scope_unit,
                timeout_seconds=request.graceful_shutdown_seconds,
            )
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            await self._wait_root_exit(process)
            return scope_stopped and await self._finish_descendant_cleanup(descendants)
        try:
            async with asyncio.timeout(request.graceful_shutdown_seconds):
                await self._wait_root_exit(process)
                return scope_stopped and await self._finish_descendant_cleanup(descendants)
        except TimeoutError:
            pass
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        await self._wait_root_exit(process)
        return scope_stopped and await self._finish_descendant_cleanup(descendants)

    @staticmethod
    async def _wait_root_exit(process: asyncio.subprocess.Process) -> None:
        # Process.wait() also waits for pipe transports. A reader that has hit
        # its output limit can leave a paused pipe, even after the root is reaped.
        # Readers are settled and transports closed by run() after termination.
        while process.returncode is None:
            await asyncio.sleep(0.01)

    @staticmethod
    def _descendants(root_pid: int) -> tuple[psutil.Process, ...]:
        try:
            return tuple(psutil.Process(root_pid).children(recursive=True))
        except psutil.Error:
            return ()

    @staticmethod
    def _track_current_descendants(
        root_pid: int,
        tracked_descendants: dict[int, float | None],
    ) -> None:
        for descendant in SubprocessBroker._descendants(root_pid):
            try:
                tracked_descendants[descendant.pid] = descendant.create_time()
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                tracked_descendants.setdefault(descendant.pid, None)

    @staticmethod
    def _track_observed_descendants(
        root_pid: int,
        observations: list[ProcessObservation],
        tracked_descendants: dict[int, float | None],
    ) -> None:
        for observation in observations:
            if observation.pid != root_pid:
                tracked_descendants.setdefault(observation.pid, observation.create_time)

    @staticmethod
    def _tracked_processes(
        current: tuple[psutil.Process, ...],
        tracked_descendants: dict[int, float | None],
        *,
        root_pid: int,
    ) -> tuple[psutil.Process, ...]:
        processes = {item.pid: item for item in current if item.pid != root_pid}
        for pid, expected_create_time in tracked_descendants.items():
            if pid == root_pid or pid in processes:
                continue
            try:
                candidate = psutil.Process(pid)
                if (
                    expected_create_time is not None
                    and candidate.create_time() != expected_create_time
                ):
                    continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            processes[pid] = candidate
        return tuple(processes.values())

    @staticmethod
    async def _finish_descendant_cleanup(descendants: tuple[psutil.Process, ...]) -> bool:
        for descendant in descendants:
            with suppress(psutil.Error):
                if descendant.is_running() and descendant.status() != psutil.STATUS_ZOMBIE:
                    descendant.kill()
        if descendants:
            await asyncio.to_thread(psutil.wait_procs, descendants, timeout=0.25)
        return all(SubprocessBroker._process_stopped(item) for item in descendants)

    @staticmethod
    def _process_stopped(process: psutil.Process) -> bool:
        try:
            return not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return True
        except psutil.AccessDenied:
            return False

    async def _stop_systemd_scope(
        self,
        unit: str,
        *,
        timeout_seconds: float,
    ) -> bool:
        systemctl_binding = ExecutableResolver().resolve_host_tool("systemctl")
        if systemctl_binding is None:
            return False
        systemctl = systemctl_binding.invocation_path
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                str(Path(systemctl).resolve()),
                "--user",
                "stop",
                unit,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=os.name == "posix",
            )
            async with asyncio.timeout(max(timeout_seconds, 1)):
                return await process.wait() == 0
        except (OSError, TimeoutError):
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            return False

    async def _settle_readers(
        self,
        *tasks: asyncio.Task[bytes],
    ) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _close_async_process_transport(process: asyncio.subprocess.Process) -> None:
        # A cancelled reader can leave an asyncio pipe transport alive after
        # the process has been reaped. Close the owning transport while its
        # event loop is still active, avoiding loop-closed __del__ warnings.
        transport = getattr(process, "_transport", None)
        if transport is not None:
            transport.close()

    async def _collect_readers(
        self,
        stdout_task: asyncio.Task[bytes],
        stderr_task: asyncio.Task[bytes],
    ) -> tuple[bytes, bytes]:
        values = await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        stdout = values[0] if isinstance(values[0], bytes) else b""
        stderr = values[1] if isinstance(values[1], bytes) else b""
        return stdout, stderr

    async def _collect_resource(
        self,
        task: asyncio.Task[RuntimeResourceSummary | None],
    ) -> RuntimeResourceSummary | None:
        values = await asyncio.gather(task, return_exceptions=True)
        value = values[0]
        return value if isinstance(value, RuntimeResourceSummary) else None

    async def _settle_task(self, task: asyncio.Task[Any] | None) -> None:
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _resolve_cwd(self, cwd: Path, allowed_roots: tuple[Path, ...]) -> Path:
        resolved = cwd.resolve()
        if not resolved.is_dir():
            raise DomainError(
                ErrorCode.INVALID_INPUT,
                f"Working directory does not exist: {resolved}",
            )
        for root in allowed_roots:
            try:
                resolved.relative_to(root.resolve())
                return resolved
            except ValueError:
                continue
        raise DomainError(
            ErrorCode.INVALID_INPUT,
            "Working directory is outside the allowed roots.",
        )

    def _build_environment(self, request: ExecutionRequest) -> dict[str, str]:
        environment = {
            name: os.environ[name]
            for name in request.environment_allowlist
            if name in os.environ and not is_dangerous_environment_name(name)
        }
        if request.systemd_scope_unit is not None:
            for name in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
                if name in os.environ:
                    environment[name] = os.environ[name]
        for name, value in request.environment_overrides.items():
            if is_dangerous_environment_name(name) and not is_safe_control_override(name, value):
                raise DomainError(
                    ErrorCode.INVALID_INPUT,
                    f"Environment override {name!r} is blocked by policy.",
                )
            if "\x00" in name or "\x00" in value or "=" in name:
                raise DomainError(
                    ErrorCode.INVALID_INPUT,
                    "Environment overrides contain invalid data.",
                )
            environment[name] = value
        return environment

    def _bound_executable(
        self,
        request: ExecutionRequest,
    ) -> ResolvedExecutable:
        resolver = ExecutableResolver()
        binding = request.executable_binding
        if request.argv[0] not in {
            binding.requested_token,
            str(binding.invocation_path),
            str(binding.canonical_target),
        }:
            raise DomainError(
                ErrorCode.MISSING_OR_CHANGED_INPUT,
                "Execution argv does not match the bound executable.",
            )
        return resolver.revalidate(binding)
