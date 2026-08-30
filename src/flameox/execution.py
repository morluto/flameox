from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import IO, Annotated, Any, Literal, TypeVar, cast

import psutil
from pydantic import Field, field_validator, model_validator

from flameox.command_binding import ExecutableResolver
from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.executables import ResolvedExecutable
from flameox.domain.models import (
    ProcessCancellationCause,
    ProcessResult,
    ResourcePolicyCancellationCause,
    RuntimeResourceSummary,
    process_termination_from_returncode,
)
from flameox.models import ContractModel

_DANGEROUS_ENVIRONMENT = {
    "BASH_ENV",
    "CDPATH",
    "ENV",
    "GIT_ASKPASS",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONPATH",
    "SSH_ASKPASS",
}
_DANGEROUS_ENVIRONMENT_PREFIXES = (
    "DYLD_",
    "GDB_",
    "GIT_CONFIG_",
    "LD_",
    "LLDB_",
)
_DANGEROUS_ENVIRONMENT_NAMES = {
    "JAVA_TOOL_OPTIONS",
    "NODE_EXTRA_CA_CERTS",
    "NODE_OPTIONS",
    "PERL5LIB",
    "PERL5OPT",
    "RUBYLIB",
    "RUBYOPT",
    "GDBINIT",
    "GDBHISTFILE",
    "PYTHONSTARTUP",
}
_CREDENTIAL_ENVIRONMENT = re.compile(
    r"(?:^|_)(?:TOKEN|PASSWORD|PASSWD|SECRET|KEY|CREDENTIALS?|COOKIES?)(?:_|$)"
)
_SAFE_CONTROL_OVERRIDES = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}
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
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
)

_T = TypeVar("_T")


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


def _is_dangerous_environment_name(name: str) -> bool:
    normalized = name.upper()
    return (
        normalized in _DANGEROUS_ENVIRONMENT
        or normalized in _DANGEROUS_ENVIRONMENT_NAMES
        or normalized.startswith(_DANGEROUS_ENVIRONMENT_PREFIXES)
        or _CREDENTIAL_ENVIRONMENT.search(normalized) is not None
    )


def _is_safe_control_override(name: str, value: str) -> bool:
    return _SAFE_CONTROL_OVERRIDES.get(name.upper()) == value


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
    timeout_seconds: float = Field(default=300, gt=0, le=86_400)
    graceful_shutdown_seconds: float = Field(default=5, ge=0, le=60)
    max_output_bytes: int = Field(default=16_777_216, gt=0)
    observation: Literal["child_peak_rss"] | None = None
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
        if any(not item or "\x00" in item for item in value):
            raise ValueError("argv entries must be non-empty and cannot contain NUL")
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
    process: ProcessResult
    stdout: bytes
    stderr: bytes
    resolved_executable: Path
    containment: ProcessContainment
    executable_binding: ResolvedExecutable
    peak_rss_backend: str | None = None
    process_observations: tuple[ProcessObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class ManagedSidecarOutcome:
    process: ProcessResult
    stdout: bytes
    stderr: bytes
    containment: ProcessContainment
    process_observations: tuple[ProcessObservation, ...]
    started_at: datetime
    finished_at: datetime


class ManagedSidecarStartupError(DomainError):
    """Managed sidecar startup failure with its bounded, finalized process evidence."""

    def __init__(self, cause: Exception, outcome: ManagedSidecarOutcome) -> None:
        self.outcome = outcome
        if isinstance(cause, DomainError):
            super().__init__(
                cause.code,
                cause.message,
                retryable=cause.retryable,
                details=cause.details,
                remediation=cause.remediation,
                next_action=cause.next_action,
            )
        else:
            super().__init__(
                ErrorCode.PROCESS_FAILED,
                "Managed sidecar startup failed.",
                details={"exception_type": type(cause).__name__},
            )


class ManagedSidecarStartupCancelled(asyncio.CancelledError):
    def __init__(self, outcome: ManagedSidecarOutcome) -> None:
        super().__init__("managed sidecar startup cancelled")
        self.outcome = outcome


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
        retryable: bool = False,
    ) -> None:
        self.process = process
        self.process_observations = process_observations
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            code,
            message,
            details={
                "process": process.model_dump(mode="json"),
                "process_observations": [
                    item.model_dump(mode="json") for item in process_observations
                ],
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
    ) -> None:
        super().__init__("process execution cancelled")
        self.process = process
        self.process_observations = process_observations
        self.stdout = stdout
        self.stderr = stderr


class _OutputLimitExceeded(Exception):
    pass


class _ResourcePolicyExceeded(Exception):
    def __init__(
        self,
        summary: RuntimeResourceSummary,
        cause: ResourcePolicyCancellationCause,
    ) -> None:
        self.summary = summary
        self.cause = cause


@dataclass(frozen=True, slots=True)
class _ObservedWait:
    returncode: int
    peak_rss_bytes: int | None
    peak_rss_backend: str
    cancellation_cause: ProcessCancellationCause | None = None


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
class _ObservedOutput:
    """Incrementally collect observed-process output under one shared budget."""

    _remaining: int
    _stdout: bytearray
    _stderr: bytearray
    _lock: threading.Lock
    _limit_exceeded: threading.Event
    _io_failed: threading.Event
    _stop: threading.Event

    def __init__(self, max_output_bytes: int) -> None:
        self._remaining = max_output_bytes
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._lock = threading.Lock()
        self._limit_exceeded = threading.Event()
        self._io_failed = threading.Event()
        self._stop = threading.Event()

    def read_stdout(self, stream: IO[bytes]) -> None:
        self._read(stream, self._stdout)

    def read_stderr(self, stream: IO[bytes]) -> None:
        self._read(stream, self._stderr)

    def _read(self, stream: IO[bytes], destination: bytearray) -> None:
        try:
            descriptor = stream.fileno()
            os.set_blocking(descriptor, False)
            while not self._stop.is_set():
                try:
                    chunk = os.read(descriptor, 64 * 1024)
                except (BlockingIOError, InterruptedError):
                    self._stop.wait(0.005)
                    continue
                if not chunk:
                    return
                with self._lock:
                    retained = min(len(chunk), self._remaining)
                    destination.extend(chunk[:retained])
                    self._remaining -= retained
                    if retained < len(chunk):
                        self._limit_exceeded.set()
                        self._stop.set()
                        return
        except (OSError, ValueError):
            if not self._stop.is_set():
                self._io_failed.set()
                self._stop.set()
            return
        finally:
            with suppress(OSError, ValueError):
                stream.close()

    @property
    def limit_exceeded(self) -> bool:
        return self._limit_exceeded.is_set()

    @property
    def io_failed(self) -> bool:
        return self._io_failed.is_set()

    def stop(self) -> None:
        self._stop.set()

    def write_stdin(self, stream: IO[bytes], stdin_bytes: bytes) -> None:
        try:
            descriptor = stream.fileno()
            os.set_blocking(descriptor, False)
            content = memoryview(stdin_bytes)
            offset = 0
            while offset < len(content) and not self._stop.is_set():
                try:
                    written = os.write(descriptor, content[offset:])
                except (BlockingIOError, InterruptedError):
                    self._stop.wait(0.005)
                    continue
                if written == 0:
                    self._stop.wait(0.005)
                    continue
                offset += written
        except (BrokenPipeError, OSError, ValueError):
            return
        finally:
            with suppress(OSError, ValueError):
                stream.close()

    def collect(self) -> tuple[bytes, bytes]:
        with self._lock:
            return bytes(self._stdout), bytes(self._stderr)


class SubprocessBroker:
    _MAX_OBSERVED_PROCESSES = 10_000
    _OBSERVED_IO_JOIN_SECONDS = 0.25
    _OBSERVED_IO_STOP_SECONDS = 0.05

    async def start_toxiproxy(
        self,
        executable: Path,
        *,
        admin_host: str,
        admin_port: int,
        readiness: Callable[[], Awaitable[bool]],
        tool_receipt: object | None = None,
        readiness_timeout_seconds: float = 10.0,
    ) -> ManagedSidecarLease:
        """Start the only long-lived process this broker exposes: Toxiproxy."""

        return await ManagedSidecarLease.start(
            self,
            executable,
            admin_host=admin_host,
            admin_port=admin_port,
            readiness=readiness,
            tool_receipt=tool_receipt,
            readiness_timeout_seconds=readiness_timeout_seconds,
        )

    async def start_inference_server(
        self,
        request: ExecutionRequest,
        *,
        host: str,
        port: int,
        readiness: Callable[[], Awaitable[bool]],
        absolute_deadline: float,
    ) -> ManagedSidecarLease:
        """Start one declared loopback inference server under a managed lease."""

        return await ManagedSidecarLease.start_inference(
            self,
            request,
            host=host,
            port=port,
            readiness=readiness,
            absolute_deadline=absolute_deadline,
        )

    def run_sync(
        self,
        request: ExecutionRequest,
        *,
        on_started: Callable[[int], Awaitable[None]] | None = None,
        on_cleanup: Callable[[bool], Awaitable[None]] | None = None,
    ) -> ExecutionOutcome:
        """Run through the async broker while preserving synchronous adapter APIs."""

        if request.observation == "child_peak_rss":
            if on_started is not None or on_cleanup is not None:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "Child peak-RSS observation does not support async lifecycle callbacks.",
                )
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return self._run_observed_sync(request)

            observed_result: list[ExecutionOutcome] = []
            observed_failure: list[BaseException] = []

            def observe_in_thread() -> None:
                try:
                    observed_result.append(self._run_observed_sync(request))
                except BaseException as exc:
                    observed_failure.append(exc)

            thread = threading.Thread(
                target=observe_in_thread,
                name="flameox-observed-subprocess",
                daemon=True,
            )
            thread.start()
            thread.join()
            if observed_failure:
                raise observed_failure[0]
            if not observed_result:
                raise RuntimeError("observed subprocess did not return an outcome")
            return observed_result[0]

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

    async def run(
        self,
        request: ExecutionRequest,
        *,
        on_started: Callable[[int], Awaitable[None]] | None = None,
        on_cleanup: Callable[[bool], Awaitable[None]] | None = None,
    ) -> ExecutionOutcome:
        if request.observation == "child_peak_rss":
            return await self._run_observed_async(request, on_started, on_cleanup)

        cwd = self._resolve_cwd(request.cwd, request.allowed_working_roots)
        environment = self._build_environment(request)
        binding = self._bound_executable(request)
        executable = binding.invocation_path
        argv = (str(executable), *request.argv[1:])
        started = time.monotonic_ns()
        deadline = asyncio.get_running_loop().time() + request.timeout_seconds
        process_observations: list[ProcessObservation] = []
        try:
            async with asyncio.timeout_at(deadline):
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
            cleanup_complete = True
            if on_cleanup is not None:
                await asyncio.shield(on_cleanup(cleanup_complete))
            timeout_process = ProcessResult(
                wall_time_ns=time.monotonic_ns() - started,
                cancellation_cause=ProcessCancellationCause.TIMEOUT,
                cleanup_complete=cleanup_complete,
            )
            raise ProcessExecutionError(
                ErrorCode.PROCESS_TIMEOUT,
                f"Process exceeded {request.timeout_seconds} seconds.",
                process=timeout_process,
                process_observations=tuple(process_observations),
                retryable=True,
            ) from exc
        except asyncio.CancelledError:
            if on_cleanup is not None:
                await asyncio.shield(on_cleanup(True))
            raise
        assert process.stdout is not None
        assert process.stderr is not None

        output_budget = _OutputBudget(request.max_output_bytes)
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        tracked_descendants: dict[int, float | None] = {}
        stdout_task = asyncio.create_task(
            self._read_bounded(process.stdout, output_budget, output=stdout_buffer)
        )
        stderr_task = asyncio.create_task(
            self._read_bounded(process.stderr, output_budget, output=stderr_buffer)
        )
        resource_task = asyncio.create_task(
            self._observe_resources(process, request.resource_policy, tracked_descendants)
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
                results = await asyncio.gather(
                    asyncio.shield(stdout_task),
                    asyncio.shield(stderr_task),
                    process.wait(),
                    asyncio.shield(resource_task),
                    *([asyncio.shield(stdin_task)] if stdin_task is not None else []),
                )
                stdout = cast(bytes, results[0])
                stderr = cast(bytes, results[1])
                resources = cast(RuntimeResourceSummary | None, results[3])
        except TimeoutError as exc:
            cleanup_complete = await self._terminate_with_observation(
                process,
                request,
                process_observations,
                on_cleanup,
                tracked_descendants,
            )
            await self._settle_readers(stdout_task, stderr_task)
            partial_stdout, partial_stderr = bytes(stdout_buffer), bytes(stderr_buffer)
            if request.resource_policy is None:
                await self._settle_resource(resource_task)
                resources = None
            else:
                resources = await self._collect_resource(resource_task)
            await self._settle_task(stdin_task)
            timeout_process = ProcessResult(
                termination=process_termination_from_returncode(process.returncode),
                wall_time_ns=time.monotonic_ns() - started,
                cancellation_cause=ProcessCancellationCause.TIMEOUT,
                cleanup_complete=cleanup_complete,
                peak_rss_bytes=resources.peak_rss_bytes if resources is not None else None,
                resources=resources,
                stdout=partial_stdout.decode(errors="replace"),
                stderr=partial_stderr.decode(errors="replace"),
            )
            raise ProcessExecutionError(
                ErrorCode.PROCESS_TIMEOUT,
                f"Process exceeded {request.timeout_seconds} seconds.",
                process=timeout_process,
                process_observations=tuple(process_observations),
                stdout=partial_stdout,
                stderr=partial_stderr,
                retryable=True,
            ) from exc
        except _OutputLimitExceeded as exc:
            cleanup_complete = await self._terminate_with_observation(
                process,
                request,
                process_observations,
                on_cleanup,
                tracked_descendants,
            )
            await self._settle_readers(stdout_task, stderr_task)
            partial_stdout, partial_stderr = bytes(stdout_buffer), bytes(stderr_buffer)
            await self._settle_resource(resource_task)
            await self._settle_task(stdin_task)
            output_process = ProcessResult(
                termination=process_termination_from_returncode(process.returncode),
                wall_time_ns=time.monotonic_ns() - started,
                cancellation_cause=ProcessCancellationCause.OUTPUT_LIMIT,
                cleanup_complete=cleanup_complete,
                stdout=partial_stdout.decode(errors="replace"),
                stderr=partial_stderr.decode(errors="replace"),
            )
            raise ProcessExecutionError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Process output exceeded {request.max_output_bytes} bytes.",
                process=output_process,
                process_observations=tuple(process_observations),
                stdout=partial_stdout,
                stderr=partial_stderr,
            ) from exc
        except _ResourcePolicyExceeded as exc:
            cleanup_complete = await self._terminate_with_observation(
                process,
                request,
                process_observations,
                on_cleanup,
                tracked_descendants,
            )
            await self._settle_readers(stdout_task, stderr_task)
            partial_stdout, partial_stderr = bytes(stdout_buffer), bytes(stderr_buffer)
            await self._settle_task(stdin_task)
            storage_cause = exc.cause in {
                ProcessCancellationCause.STORAGE_RESERVE_EXCEEDED,
                ProcessCancellationCause.WRITABLE_LIMIT_EXCEEDED,
            }
            code = (
                ErrorCode.STORAGE_QUOTA_EXCEEDED
                if storage_cause
                else ErrorCode.QUERY_BUDGET_EXCEEDED
            )
            message = (
                "Runtime writable-byte policy was exceeded."
                if storage_cause
                else "Process tree exceeded the configured memory budget."
            )
            policy_process = ProcessResult(
                cancellation_cause=exc.cause,
                cleanup_complete=cleanup_complete,
                peak_rss_bytes=exc.summary.peak_rss_bytes,
                resources=exc.summary,
                stdout=partial_stdout.decode(errors="replace"),
                stderr=partial_stderr.decode(errors="replace"),
            )
            raise ProcessExecutionError(
                code,
                message,
                process=policy_process,
                process_observations=tuple(process_observations),
                stdout=partial_stdout,
                stderr=partial_stderr,
            ) from exc
        except asyncio.CancelledError:
            cleanup_complete = await self._terminate_with_observation(
                process,
                request,
                process_observations,
                on_cleanup,
                tracked_descendants,
            )
            await self._settle_readers(stdout_task, stderr_task)
            partial_stdout, partial_stderr = bytes(stdout_buffer), bytes(stderr_buffer)
            resources = await self._collect_resource(resource_task)
            await self._settle_task(stdin_task)
            raise ProcessCancelledError(
                process=ProcessResult(
                    termination=process_termination_from_returncode(process.returncode),
                    wall_time_ns=time.monotonic_ns() - started,
                    cancellation_cause=ProcessCancellationCause.CALLER_CANCELLED,
                    cleanup_complete=cleanup_complete,
                    peak_rss_bytes=resources.peak_rss_bytes if resources is not None else None,
                    resources=resources,
                    stdout=partial_stdout.decode(errors="replace"),
                    stderr=partial_stderr.decode(errors="replace"),
                ),
                process_observations=tuple(process_observations),
                stdout=partial_stdout,
                stderr=partial_stderr,
            ) from None
        except BaseException:
            await self._terminate_with_observation(
                process,
                request,
                process_observations,
                on_cleanup,
                tracked_descendants,
            )
            await self._settle_readers(stdout_task, stderr_task)
            await self._settle_resource(resource_task)
            await self._settle_task(stdin_task)
            raise

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

    async def _run_observed_async(
        self,
        request: ExecutionRequest,
        on_started: Callable[[int], Awaitable[None]] | None,
        on_cleanup: Callable[[bool], Awaitable[None]] | None,
    ) -> ExecutionOutcome:
        if on_started is not None:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Child peak-RSS observation does not support an on_started callback.",
            )
        cancellation = threading.Event()
        observation = asyncio.create_task(
            asyncio.to_thread(self._run_observed_sync, request, cancellation)
        )
        try:
            return await asyncio.shield(observation)
        except asyncio.CancelledError:
            cancellation.set()
            outcome = await asyncio.shield(observation)
            if on_cleanup is not None:
                await asyncio.shield(on_cleanup(outcome.process.cleanup_complete is True))
            raise

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
        deadline: float,
    ) -> ProcessObservation:
        failures: list[str] = []

        def read(name: str, default: Any = None) -> Any:
            if time.monotonic() > deadline:
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

    def _run_observed_sync(
        self,
        request: ExecutionRequest,
        cancellation: threading.Event | None = None,
    ) -> ExecutionOutcome:
        """Run a workload while retaining wait4's child-observation semantics.

        The normal async path cannot use ``wait4`` because the event loop's child
        watcher owns ``waitpid``.  This explicit path keeps spawning, output capture,
        timeout, and group cleanup inside the broker while reaping the child itself.
        """

        cwd = self._resolve_cwd(request.cwd, request.allowed_working_roots)
        environment = self._build_environment(request)
        binding = self._bound_executable(request)
        executable = binding.invocation_path
        argv = (str(executable), *request.argv[1:])
        started = time.monotonic_ns()
        process: subprocess.Popen[bytes] | None = None
        process_observations: list[ProcessObservation] = []
        cleanup_complete = True
        output = _ObservedOutput(request.max_output_bytes)
        reader_threads: tuple[threading.Thread, ...] = ()
        stdin_thread: threading.Thread | None = None
        deadline = time.monotonic() + request.timeout_seconds
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdin=(subprocess.PIPE if request.stdin_bytes is not None else subprocess.DEVNULL),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=os.name == "posix",
                pass_fds=request.inherited_directory_fds,
            )
            assert process.stdout is not None
            assert process.stderr is not None
            reader_threads = (
                threading.Thread(
                    target=output.read_stdout,
                    args=(process.stdout,),
                    name="flameox-observed-stdout",
                    daemon=True,
                ),
                threading.Thread(
                    target=output.read_stderr,
                    args=(process.stderr,),
                    name="flameox-observed-stderr",
                    daemon=True,
                ),
            )
            for thread in reader_threads:
                thread.start()
            if request.stdin_bytes is not None:
                assert process.stdin is not None
                stdin_thread = threading.Thread(
                    target=output.write_stdin,
                    args=(process.stdin, request.stdin_bytes),
                    name="flameox-observed-stdin",
                    daemon=True,
                )
                stdin_thread.start()

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
            observed = self._wait_observed(
                process,
                deadline=deadline,
                output=output,
                cancellation=cancellation,
                observations=process_observations,
            )
            cleanup_complete = self._terminate_observed_group(process, force=True)
            self._join_observed_io(output, reader_threads, stdin_thread)
            stdout, stderr = output.collect()
        except BaseException:
            if process is not None and process.poll() is None:
                cleanup_complete = self._terminate_observed_group(process, force=True)
                self._reap_observed(process)
            self._join_observed_io(output, reader_threads, stdin_thread)
            raise

        finished = time.monotonic_ns()
        process_observations.extend(
            self._snapshot_known_processes(
                self._observation_identities(process_observations),
                ProcessSnapshotPhase.POST_ROOT_EXIT,
                False,
                None,
                None,
            )
        )
        if observed.cancellation_cause is ProcessCancellationCause.IO_FAILURE or output.io_failed:
            io_process = ProcessResult(
                termination=process_termination_from_returncode(observed.returncode),
                wall_time_ns=finished - started,
                cancellation_cause=ProcessCancellationCause.IO_FAILURE,
                cleanup_complete=cleanup_complete,
            )
            raise ProcessExecutionError(
                ErrorCode.PROCESS_FAILED,
                "Process output could not be drained safely.",
                process=io_process,
                process_observations=tuple(process_observations),
            )
        if (
            observed.cancellation_cause is ProcessCancellationCause.OUTPUT_LIMIT
            or output.limit_exceeded
        ):
            output_process = ProcessResult(
                termination=process_termination_from_returncode(observed.returncode),
                wall_time_ns=finished - started,
                cancellation_cause=ProcessCancellationCause.OUTPUT_LIMIT,
                cleanup_complete=cleanup_complete,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )
            raise ProcessExecutionError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Process output exceeded {request.max_output_bytes} bytes.",
                process=output_process,
                process_observations=tuple(process_observations),
                stdout=stdout,
                stderr=stderr,
            )
        if observed.cancellation_cause is ProcessCancellationCause.TIMEOUT:
            timeout_process = ProcessResult(
                termination=process_termination_from_returncode(observed.returncode),
                wall_time_ns=finished - started,
                cancellation_cause=ProcessCancellationCause.TIMEOUT,
                cleanup_complete=cleanup_complete,
                peak_rss_bytes=observed.peak_rss_bytes,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )
            raise ProcessExecutionError(
                ErrorCode.PROCESS_TIMEOUT,
                f"Process exceeded {request.timeout_seconds} seconds.",
                process=timeout_process,
                process_observations=tuple(process_observations),
                retryable=True,
            )

        process_result = ProcessResult(
            termination=process_termination_from_returncode(observed.returncode),
            wall_time_ns=finished - started,
            cancellation_cause=observed.cancellation_cause,
            cleanup_complete=cleanup_complete,
            peak_rss_bytes=observed.peak_rss_bytes,
        )
        return ExecutionOutcome(
            process=process_result,
            stdout=stdout,
            stderr=stderr,
            resolved_executable=executable,
            executable_binding=binding,
            containment=(
                ProcessContainment.PROCESS_GROUP
                if os.name == "posix"
                else ProcessContainment.PROCESS
            ),
            peak_rss_backend=observed.peak_rss_backend,
            process_observations=tuple(process_observations),
        )

    def _wait_observed(
        self,
        process: subprocess.Popen[bytes],
        *,
        deadline: float,
        output: _ObservedOutput,
        cancellation: threading.Event | None,
        observations: list[ProcessObservation],
    ) -> _ObservedWait:
        terminating: ProcessCancellationCause | None = None
        wait4 = getattr(os, "wait4", None)
        if wait4 is not None:
            while True:
                if terminating is None:
                    if cancellation is not None and cancellation.is_set():
                        terminating = ProcessCancellationCause.CALLER_CANCELLED
                        self._terminate_observed_with_observation(process, observations, force=True)
                    elif output.limit_exceeded:
                        terminating = ProcessCancellationCause.OUTPUT_LIMIT
                        self._terminate_observed_with_observation(process, observations, force=True)
                    elif output.io_failed:
                        terminating = ProcessCancellationCause.IO_FAILURE
                        self._terminate_observed_with_observation(process, observations, force=True)
                    elif time.monotonic() >= deadline:
                        terminating = ProcessCancellationCause.TIMEOUT
                        self._terminate_observed_with_observation(process, observations, force=True)
                waited_pid, status, usage = wait4(process.pid, os.WNOHANG)
                if waited_pid == process.pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
                    peak_rss = int(usage.ru_maxrss)
                    if sys.platform != "darwin":
                        peak_rss *= 1024
                    return _ObservedWait(
                        returncode=process.returncode,
                        peak_rss_bytes=peak_rss or None,
                        peak_rss_backend="wait4_ru_maxrss",
                        cancellation_cause=terminating,
                    )
                time.sleep(0.005)

        peak = 0
        while process.poll() is None:
            peak = max(peak, self._observed_peak_rss(process.pid))
            if terminating is None:
                if cancellation is not None and cancellation.is_set():
                    terminating = ProcessCancellationCause.CALLER_CANCELLED
                    self._terminate_observed_with_observation(process, observations, force=True)
                elif output.limit_exceeded:
                    terminating = ProcessCancellationCause.OUTPUT_LIMIT
                    self._terminate_observed_with_observation(process, observations, force=True)
                elif output.io_failed:
                    terminating = ProcessCancellationCause.IO_FAILURE
                    self._terminate_observed_with_observation(process, observations, force=True)
                elif time.monotonic() >= deadline:
                    terminating = ProcessCancellationCause.TIMEOUT
                    self._terminate_observed_with_observation(process, observations, force=True)
            time.sleep(0.005)
        peak = max(peak, self._observed_peak_rss(process.pid))
        return _ObservedWait(
            returncode=process.returncode if process.returncode is not None else 0,
            peak_rss_bytes=peak or None,
            peak_rss_backend="psutil_polling",
            cancellation_cause=terminating,
        )

    def _terminate_observed_with_observation(
        self,
        process: subprocess.Popen[bytes],
        observations: list[ProcessObservation],
        *,
        force: bool,
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
        cleanup_complete = self._terminate_observed_group(process, force=force)
        observations.extend(
            self._snapshot_known_processes(
                identities,
                ProcessSnapshotPhase.POST_CLEANUP,
                False,
                "terminate",
                str(cleanup_complete),
            )
        )
        return cleanup_complete

    def _observed_peak_rss(self, pid: int) -> int:
        peak = 0
        try:
            processes, _truncated = self._enumerate_processes(pid, time.monotonic() + 0.5)
            peak = sum(
                process.memory_info().rss for process, _source in processes if process.is_running()
            )
        except (psutil.Error, OSError):
            pass
        return peak

    def _terminate_observed_group(
        self,
        process: subprocess.Popen[bytes],
        *,
        force: bool,
    ) -> bool:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return True
            if force:
                time.sleep(0.05)
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            return True

        if process.returncode is not None:
            return True
        if process.poll() is None:
            process.terminate()
            if force:
                time.sleep(0.05)
                if process.poll() is None:
                    process.kill()
        return True

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

    def _join_observed_io(
        self,
        output: _ObservedOutput,
        reader_threads: tuple[threading.Thread, ...],
        stdin_thread: threading.Thread | None,
    ) -> None:
        threads = (*reader_threads, *([stdin_thread] if stdin_thread is not None else []))
        deadline = time.monotonic() + self._OBSERVED_IO_JOIN_SECONDS
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        output.stop()
        stop_deadline = time.monotonic() + self._OBSERVED_IO_STOP_SECONDS
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=max(0.0, stop_deadline - time.monotonic()))

    def _reap_observed(self, process: subprocess.Popen[bytes]) -> None:
        if process.returncode is not None:
            return
        wait4 = getattr(os, "wait4", None)
        if wait4 is not None:
            _, status, _ = wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        else:
            process.wait()

    async def _observe_resources(
        self,
        process: asyncio.subprocess.Process,
        policy: ResourcePolicy | None,
        tracked_descendants: dict[int, float | None],
    ) -> RuntimeResourceSummary | None:
        interval = 0.1 if policy is None else policy.sampling_interval_ms / 1_000
        if policy is None:
            while process.returncode is None:
                self._track_current_descendants(process.pid, tracked_descendants)
                await asyncio.sleep(interval)
            return None
        minimum_free: int | None = None
        peak_rss = 0
        free_sampled = False
        rss_sampled = False
        unavailable: set[str] = set()
        initial_sizes = {
            str(root): self._bounded_tree_size(
                root,
                max_files=policy.max_observed_files,
            )
            for root in policy.writable_roots
        }
        initial_staging = (
            self._bounded_tree_size(policy.staging_root, max_files=policy.max_observed_files)
            if policy.staging_root is not None
            else None
        )
        while process.returncode is None:
            self._track_current_descendants(process.pid, tracked_descendants)
            try:
                free = shutil.disk_usage(policy.filesystem_path).free
            except OSError:
                unavailable.add("minimum_free_bytes")
                free = None
            else:
                free_sampled = True
                previous_free = minimum_free
                if previous_free is None:
                    minimum_free = free
                else:
                    minimum_free = min(previous_free, cast(int, free))
            rss = 0
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
            await asyncio.sleep(interval)
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
        stream: asyncio.StreamReader,
        budget: _OutputBudget,
        *,
        drain_on_limit: bool = False,
        output: bytearray | None = None,
    ) -> bytes:
        output = output if output is not None else bytearray()
        while chunk := await stream.read(64 * 1024):
            retained = min(len(chunk), max(budget.remaining, 0))
            output.extend(chunk[:retained])
            try:
                budget.consume(len(chunk))
            except _OutputLimitExceeded:
                if not drain_on_limit:
                    raise
                while await stream.read(64 * 1024):
                    pass
                break
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
            await process.wait()
            return scope_stopped and await self._finish_descendant_cleanup(descendants)
        try:
            async with asyncio.timeout(request.graceful_shutdown_seconds):
                await process.wait()
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
        await process.wait()
        return scope_stopped and await self._finish_descendant_cleanup(descendants)

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

    async def _settle_resource(
        self,
        task: asyncio.Task[RuntimeResourceSummary | None],
    ) -> None:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

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
                ErrorCode.EXECUTION_REFUSED,
                f"Working directory does not exist: {resolved}",
            )
        for root in allowed_roots:
            try:
                resolved.relative_to(root.resolve())
                return resolved
            except ValueError:
                continue
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            "Working directory is outside the allowed roots.",
        )

    def _build_environment(self, request: ExecutionRequest) -> dict[str, str]:
        environment = {
            name: os.environ[name]
            for name in request.environment_allowlist
            if name in os.environ and not _is_dangerous_environment_name(name)
        }
        if request.systemd_scope_unit is not None:
            for name in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
                if name in os.environ:
                    environment[name] = os.environ[name]
        for name, value in request.environment_overrides.items():
            if _is_dangerous_environment_name(name) and not _is_safe_control_override(name, value):
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    f"Environment override {name!r} is blocked by policy.",
                )
            if "\x00" in name or "\x00" in value or "=" in name:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
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
                ErrorCode.INVALID_CAPTURE_PLAN,
                "Execution argv does not match the bound executable.",
            )
        return resolver.revalidate(binding)


class ManagedSidecarLease:
    """Broker-owned lease for a typed loopback sidecar process."""

    def __init__(
        self,
        broker: SubprocessBroker,
        process: asyncio.subprocess.Process,
        executable_binding: ResolvedExecutable,
        admin_host: str,
        admin_port: int,
        stdout_task: asyncio.Task[bytes],
        stderr_task: asyncio.Task[bytes],
        output_budget: _OutputBudget,
        observations: list[ProcessObservation],
        started_at: datetime,
        *,
        supports_toxiproxy_control: bool = False,
    ) -> None:
        self._broker = broker
        self._process = process
        self._executable_binding = executable_binding
        self.admin_host = admin_host
        self.admin_port = admin_port
        self._stdout_task = stdout_task
        self._stderr_task = stderr_task
        self._output_budget = output_budget
        self._observations = observations
        self._started_at = started_at
        self._supports_toxiproxy_control = supports_toxiproxy_control
        self._tracked_proxies: set[str] = set()
        self._closed = False
        self._outcome: ManagedSidecarOutcome | None = None

    @classmethod
    async def start(
        cls,
        broker: SubprocessBroker,
        executable: Path,
        *,
        admin_host: str,
        admin_port: int,
        readiness: Callable[[], Awaitable[bool]],
        tool_receipt: object | None,
        readiness_timeout_seconds: float,
    ) -> ManagedSidecarLease:
        import ipaddress

        try:
            address = ipaddress.ip_address(admin_host)
        except ValueError as error:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Toxiproxy admin host must be an IP literal.",
            ) from error
        if not address.is_loopback or not 1 <= admin_port <= 65_535:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Toxiproxy admin endpoint must be a loopback address and valid port.",
            )
        resolved = executable.resolve()
        from flameox.adapters.toxiproxy import ToxiproxyToolReceipt

        if not isinstance(tool_receipt, ToxiproxyToolReceipt):
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Managed Toxiproxy startup requires a verified setup receipt.",
            )
        if tool_receipt.executable.resolve() != resolved:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "The Toxiproxy executable does not match its verified setup receipt.",
            )
        if resolved.name not in {"toxiproxy-server", "toxiproxy-server.exe"}:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Managed sidecar leases accept only the pinned Toxiproxy server executable.",
            )
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise DomainError(ErrorCode.CAPABILITY_UNAVAILABLE, "Toxiproxy server is not runnable.")
        executable_binding = ExecutableResolver().require_host_tool(
            str(resolved), cwd=resolved.parent
        )
        request = ExecutionRequest(
            argv=(str(resolved), "-host", admin_host, "-port", str(admin_port)),
            executable_binding=executable_binding,
            cwd=resolved.parent,
            environment_allowlist=("PATH",),
            allowed_working_roots=(resolved.parent,),
            timeout_seconds=86_400,
            graceful_shutdown_seconds=2,
            max_output_bytes=2 * 1024 * 1024,
        )
        environment = broker._build_environment(request)
        process = await asyncio.create_subprocess_exec(
            *request.argv,
            cwd=broker._resolve_cwd(request.cwd, request.allowed_working_roots),
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
            pass_fds=request.inherited_directory_fds,
        )
        assert process.stdout is not None and process.stderr is not None
        observations: list[ProcessObservation] = []
        output_budget = _OutputBudget(request.max_output_bytes)
        lease = cls(
            broker,
            process,
            executable_binding,
            admin_host,
            admin_port,
            asyncio.create_task(
                broker._read_bounded(process.stdout, output_budget, drain_on_limit=True)
            ),
            asyncio.create_task(
                broker._read_bounded(process.stderr, output_budget, drain_on_limit=True)
            ),
            output_budget,
            observations,
            datetime.now(UTC),
            supports_toxiproxy_control=True,
        )
        deadline = time.monotonic() + readiness_timeout_seconds
        try:
            while time.monotonic() < deadline:
                if process.returncode is not None:
                    raise DomainError(
                        ErrorCode.CAPABILITY_UNAVAILABLE,
                        "Toxiproxy exited before its control API became ready.",
                    )
                if await readiness():
                    observations.extend(
                        broker._snapshot_processes(
                            process.pid,
                            ProcessSnapshotPhase.RUNNING,
                            True,
                            None,
                            None,
                        )
                    )
                    return lease
                await asyncio.sleep(0.05)
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "Toxiproxy control API did not become ready before the bounded deadline.",
            )
        except asyncio.CancelledError:
            outcome = await asyncio.shield(lease.close())
            raise ManagedSidecarStartupCancelled(outcome) from None
        except Exception as error:
            outcome = await asyncio.shield(lease.close())
            raise ManagedSidecarStartupError(error, outcome) from error

    @classmethod
    async def start_inference(
        cls,
        broker: SubprocessBroker,
        request: ExecutionRequest,
        *,
        host: str,
        port: int,
        readiness: Callable[[], Awaitable[bool]],
        absolute_deadline: float,
    ) -> ManagedSidecarLease:
        import ipaddress

        try:
            address = ipaddress.ip_address(host)
        except ValueError as error:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Managed inference server host must be an IP literal.",
            ) from error
        if not address.is_loopback or not 1 <= port <= 65_535:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Managed inference servers require a loopback address and valid port.",
            )
        if time.monotonic() >= absolute_deadline:
            raise DomainError(ErrorCode.PROCESS_TIMEOUT, "Inference startup deadline expired.")
        family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        endpoint = (host, port, 0, 0) if address.version == 6 else (host, port)
        with socket.socket(family, socket.SOCK_STREAM) as port_guard:
            try:
                port_guard.bind(endpoint)
            except OSError as error:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "Managed inference server endpoint is already occupied before startup.",
                    details={"host": host, "port": port},
                ) from error
        cwd = broker._resolve_cwd(request.cwd, request.allowed_working_roots)
        environment = broker._build_environment(request)
        executable = broker._bound_executable(request).invocation_path
        argv = (str(executable), *request.argv[1:])
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
            pass_fds=request.inherited_directory_fds,
        )
        assert process.stdout is not None and process.stderr is not None
        observations: list[ProcessObservation] = []
        output_budget = _OutputBudget(request.max_output_bytes)
        lease = cls(
            broker,
            process,
            request.executable_binding,
            host,
            port,
            asyncio.create_task(
                broker._read_bounded(process.stdout, output_budget, drain_on_limit=True)
            ),
            asyncio.create_task(
                broker._read_bounded(process.stderr, output_budget, drain_on_limit=True)
            ),
            output_budget,
            observations,
            datetime.now(UTC),
        )
        try:
            while time.monotonic() < absolute_deadline:
                if process.returncode is not None:
                    raise DomainError(
                        ErrorCode.CAPABILITY_UNAVAILABLE,
                        "Managed inference server exited before readiness.",
                    )
                ready = await cls._await_before_deadline(
                    readiness(),
                    absolute_deadline,
                    "Managed inference server readiness check exceeded the run deadline.",
                )
                if ready:
                    await asyncio.sleep(0)
                    if process.returncode is not None:
                        raise DomainError(
                            ErrorCode.CAPABILITY_UNAVAILABLE,
                            "Managed inference server exited while reporting readiness.",
                        )
                    snapshot = await cls._await_before_deadline(
                        asyncio.to_thread(
                            broker._snapshot_processes,
                            process.pid,
                            ProcessSnapshotPhase.RUNNING,
                            True,
                            None,
                            None,
                        ),
                        absolute_deadline,
                        "Managed inference server observation exceeded the run deadline.",
                    )
                    observations.extend(snapshot)
                    if process.returncode is not None:
                        raise DomainError(
                            ErrorCode.CAPABILITY_UNAVAILABLE,
                            "Managed inference server exited before its lease was established.",
                        )
                    return lease
                await asyncio.sleep(min(0.05, max(0.0, absolute_deadline - time.monotonic())))
            raise DomainError(
                ErrorCode.PROCESS_TIMEOUT,
                "Managed inference server did not become ready before the run deadline.",
            )
        except BaseException:
            await asyncio.shield(lease.close())
            raise

    @staticmethod
    async def _await_before_deadline(
        awaitable: Awaitable[_T],
        absolute_deadline: float,
        timeout_message: str,
    ) -> _T:
        """Await external startup work without allowing cancellation to extend its deadline."""

        task = asyncio.ensure_future(awaitable)

        def consume_result(completed: asyncio.Future[_T]) -> None:
            with suppress(BaseException):
                completed.result()

        remaining = absolute_deadline - time.monotonic()
        try:
            if remaining > 0:
                done, _pending = await asyncio.wait((task,), timeout=remaining)
                if done:
                    return task.result()
        except BaseException:
            task.cancel()
            task.add_done_callback(consume_result)
            raise
        task.cancel()
        task.add_done_callback(consume_result)
        raise DomainError(ErrorCode.PROCESS_TIMEOUT, timeout_message)

    async def __aenter__(self) -> ManagedSidecarLease:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await asyncio.shield(self.close())

    def create_proxy(
        self, *, name: str, listen: str, upstream: str, enabled: bool = True
    ) -> dict[str, Any]:
        self._require_toxiproxy_control()
        self._ensure_open()
        from flameox.adapters.toxiproxy import ToxiproxyClient

        result = ToxiproxyClient(self.base_url).create_proxy(
            name=name, listen=listen, upstream=upstream, enabled=enabled
        )
        self._tracked_proxies.add(name)
        return result

    async def create_proxy_async(
        self,
        *,
        name: str,
        listen: str,
        upstream: str,
        enabled: bool = True,
    ) -> dict[str, Any]:
        self._require_toxiproxy_control()
        self._ensure_open()
        from flameox.adapters.toxiproxy import ToxiproxyClient

        result = await ToxiproxyClient(self.base_url).create_proxy_async(
            name=name,
            listen=listen,
            upstream=upstream,
            enabled=enabled,
        )
        self._tracked_proxies.add(name)
        return result

    def update_proxy(self, name: str, *, enabled: bool) -> dict[str, Any]:
        self._require_toxiproxy_control()
        self._ensure_open()
        from flameox.adapters.toxiproxy import ToxiproxyClient

        self._require_tracked(name)
        return ToxiproxyClient(self.base_url).update_proxy(name, enabled=enabled)

    async def update_proxy_async(self, name: str, *, enabled: bool) -> dict[str, Any]:
        self._require_toxiproxy_control()
        self._ensure_open()
        from flameox.adapters.toxiproxy import ToxiproxyClient

        self._require_tracked(name)
        return await ToxiproxyClient(self.base_url).update_proxy_async(name, enabled=enabled)

    def add_toxic(self, **kwargs: Any) -> dict[str, Any]:
        self._require_toxiproxy_control()
        self._ensure_open()
        from flameox.adapters.toxiproxy import ToxiproxyClient

        proxy = kwargs.get("proxy")
        if not isinstance(proxy, str):
            raise DomainError(ErrorCode.INVALID_CAPTURE_PLAN, "A toxic requires a tracked proxy.")
        self._require_tracked(proxy)
        return ToxiproxyClient(self.base_url).add_toxic(**kwargs)

    async def add_toxic_async(self, **kwargs: Any) -> dict[str, Any]:
        self._require_toxiproxy_control()
        self._ensure_open()
        from flameox.adapters.toxiproxy import ToxiproxyClient

        proxy = kwargs.get("proxy")
        if not isinstance(proxy, str):
            raise DomainError(ErrorCode.INVALID_CAPTURE_PLAN, "A toxic requires a tracked proxy.")
        self._require_tracked(proxy)
        return await ToxiproxyClient(self.base_url).add_toxic_async(**kwargs)

    @property
    def base_url(self) -> str:
        return f"http://{self.admin_host}:{self.admin_port}"

    @property
    def outcome(self) -> ManagedSidecarOutcome | None:
        return self._outcome

    async def close(self) -> ManagedSidecarOutcome:
        if self._outcome is not None:
            return self._outcome
        self._closed = True
        cleanup_failures: list[str] = []
        if self._output_budget.exceeded:
            cleanup_failures.append(
                "sidecar output exceeded the bounded capture budget; excess bytes were discarded"
            )
        if self._supports_toxiproxy_control:
            from flameox.adapters.toxiproxy import ToxiproxyApiError, ToxiproxyClient

            client = ToxiproxyClient(self.base_url, timeout_seconds=1.0)
            for name in tuple(sorted(self._tracked_proxies)):
                try:
                    await client.delete_proxy_async(name)
                except (ToxiproxyApiError, OSError) as error:
                    cleanup_failures.append(f"proxy {name}: {error}")
        request = ExecutionRequest(
            argv=(str(self._executable_binding.invocation_path),),
            executable_binding=self._executable_binding,
            cwd=self._executable_binding.invocation_path.parent,
            allowed_working_roots=(self._executable_binding.invocation_path.parent,),
            graceful_shutdown_seconds=2,
        )
        self._observations.extend(
            self._broker._snapshot_processes(
                self._process.pid,
                ProcessSnapshotPhase.PRE_CLEANUP,
                True,
                "terminate",
                None,
            )
        )
        identities = self._broker._observation_identities(self._observations)
        tracked_descendants: dict[int, float | None] = {}
        self._broker._track_observed_descendants(
            self._process.pid,
            self._observations,
            tracked_descendants,
        )
        cleanup_complete = await asyncio.shield(
            self._broker._terminate(self._process, request, tracked_descendants)
        )
        self._observations.extend(
            self._broker._snapshot_known_processes(
                identities,
                ProcessSnapshotPhase.POST_CLEANUP,
                False,
                "terminate",
                str(cleanup_complete),
            )
        )
        stdout = await self._broker._collect_readers(self._stdout_task, self._stderr_task)
        stdout_bytes, stderr_bytes = stdout
        if cleanup_failures:
            stderr_bytes += ("\n" + "\n".join(cleanup_failures)).encode()
        finished = datetime.now(UTC)
        self._outcome = ManagedSidecarOutcome(
            process=ProcessResult(
                termination=process_termination_from_returncode(self._process.returncode),
                cleanup_complete=cleanup_complete,
                wall_time_ns=max(0, int((finished - self._started_at).total_seconds() * 1e9)),
                stdout=stdout_bytes.decode(errors="replace"),
                stderr=stderr_bytes.decode(errors="replace"),
            ),
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            containment=(
                ProcessContainment.PROCESS_GROUP
                if os.name == "posix"
                else ProcessContainment.PROCESS
            ),
            process_observations=tuple(self._observations),
            started_at=self._started_at,
            finished_at=finished,
        )
        return self._outcome

    def _ensure_open(self) -> None:
        if self._closed:
            raise DomainError(ErrorCode.EXECUTION_REFUSED, "The Toxiproxy sidecar lease is closed.")

    def _require_toxiproxy_control(self) -> None:
        if not self._supports_toxiproxy_control:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "This managed sidecar does not expose Toxiproxy controls.",
            )

    def _require_tracked(self, name: str) -> None:
        if name not in self._tracked_proxies:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "The proxy is outside this sidecar lease.",
            )
