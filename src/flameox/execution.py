from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, cast

import psutil
from pydantic import Field, field_validator

from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.models import ProcessResult, RuntimeResourceSummary
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
    "PYTHON",
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
    sampling_interval_ms: int = Field(default=250, ge=25, le=10_000)
    max_observed_files: int = Field(default=10_000, ge=1, le=1_000_000)


class ExecutionRequest(ContractModel):
    argv: tuple[str, ...]
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

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("argv must include an executable")
        if any(not item or "\x00" in item for item in value):
            raise ValueError("argv entries must be non-empty and cannot contain NUL")
        return value

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
    containment: str
    peak_rss_backend: str | None = None


class _OutputLimitExceeded(Exception):
    pass


class _ResourcePolicyExceeded(Exception):
    def __init__(self, summary: RuntimeResourceSummary) -> None:
        self.summary = summary


@dataclass(frozen=True, slots=True)
class _ObservedWait:
    returncode: int
    peak_rss_bytes: int | None
    peak_rss_backend: str
    cancellation_cause: Literal["timeout", "caller_cancelled", "output_limit"] | None = None


@dataclass(slots=True)
class _OutputBudget:
    remaining: int

    def consume(self, byte_count: int) -> None:
        self.remaining -= byte_count
        if self.remaining < 0:
            raise _OutputLimitExceeded


class SubprocessBroker:
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
        executable = self._resolve_executable(request.argv[0], cwd, environment)
        argv = (str(executable), *request.argv[1:])
        started = time.monotonic_ns()
        spawn = asyncio.create_task(
            asyncio.create_subprocess_exec(
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
            )
        )
        try:
            process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            process = await asyncio.shield(spawn)
            cleanup_complete = await asyncio.shield(self._terminate(process, request))
            if on_cleanup is not None:
                await asyncio.shield(on_cleanup(cleanup_complete))
            raise
        if on_started is not None:
            try:
                await on_started(process.pid)
            except BaseException:
                cleanup_complete = await asyncio.shield(self._terminate(process, request))
                if on_cleanup is not None:
                    await asyncio.shield(on_cleanup(cleanup_complete))
                raise
        assert process.stdout is not None
        assert process.stderr is not None

        if request.stdin_bytes is not None:
            assert process.stdin is not None
            process.stdin.write(request.stdin_bytes)
            await process.stdin.drain()
            process.stdin.close()

        output_budget = _OutputBudget(request.max_output_bytes)
        stdout_task = asyncio.create_task(self._read_bounded(process.stdout, output_budget))
        stderr_task = asyncio.create_task(self._read_bounded(process.stderr, output_budget))
        resource_task = asyncio.create_task(
            self._observe_resources(process, request.resource_policy)
        )
        timed_out = False
        cancellation_cause: str | None = None
        try:
            async with asyncio.timeout(request.timeout_seconds):
                stdout, stderr, _, resources = await asyncio.gather(
                    asyncio.shield(stdout_task),
                    asyncio.shield(stderr_task),
                    process.wait(),
                    asyncio.shield(resource_task),
                )
        except TimeoutError as exc:
            timed_out = True
            cancellation_cause = "timeout"
            cleanup_complete = await asyncio.shield(self._terminate(process, request))
            if on_cleanup is not None:
                await asyncio.shield(on_cleanup(cleanup_complete))
            partial_stdout, partial_stderr = await self._collect_readers(
                stdout_task,
                stderr_task,
            )
            resources = await self._collect_resource(resource_task)
            timeout_process = ProcessResult(
                exit_code=None,
                terminating_signal=(
                    -process.returncode
                    if process.returncode is not None and process.returncode < 0
                    else None
                ),
                wall_time_ns=time.monotonic_ns() - started,
                timed_out=True,
                cancellation_cause=cancellation_cause,
                cleanup_complete=cleanup_complete,
                peak_rss_bytes=resources.peak_rss_bytes if resources is not None else None,
                resources=resources,
                stdout=partial_stdout.decode(errors="replace"),
                stderr=partial_stderr.decode(errors="replace"),
            )
            raise DomainError(
                ErrorCode.PROCESS_TIMEOUT,
                f"Process exceeded {request.timeout_seconds} seconds.",
                details={"process": timeout_process.model_dump(mode="json")},
                retryable=True,
            ) from exc
        except _OutputLimitExceeded as exc:
            cancellation_cause = "output_limit"
            cleanup_complete = await asyncio.shield(self._terminate(process, request))
            if on_cleanup is not None:
                await asyncio.shield(on_cleanup(cleanup_complete))
            await self._settle_readers(stdout_task, stderr_task)
            await self._settle_resource(resource_task)
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Process output exceeded {request.max_output_bytes} bytes.",
            ) from exc
        except _ResourcePolicyExceeded as exc:
            cancellation_cause = "storage_reserve_exceeded"
            cleanup_complete = await asyncio.shield(self._terminate(process, request))
            if on_cleanup is not None:
                await asyncio.shield(on_cleanup(cleanup_complete))
            partial_stdout, partial_stderr = await self._collect_readers(
                stdout_task,
                stderr_task,
            )
            raise DomainError(
                ErrorCode.STORAGE_QUOTA_EXCEEDED,
                "Runtime storage reserve was exceeded.",
                details={
                    "process": ProcessResult(
                        cancellation_cause=cancellation_cause,
                        cleanup_complete=cleanup_complete,
                        peak_rss_bytes=exc.summary.peak_rss_bytes,
                        resources=exc.summary,
                        stdout=partial_stdout.decode(errors="replace"),
                        stderr=partial_stderr.decode(errors="replace"),
                    ).model_dump(mode="json")
                },
            ) from exc
        except asyncio.CancelledError:
            cancellation_cause = "caller_cancelled"
            cleanup_complete = await asyncio.shield(self._terminate(process, request))
            if on_cleanup is not None:
                await asyncio.shield(on_cleanup(cleanup_complete))
            await self._settle_readers(stdout_task, stderr_task)
            await self._settle_resource(resource_task)
            raise

        finished = time.monotonic_ns()
        result = ProcessResult(
            exit_code=(
                process.returncode
                if process.returncode is not None and process.returncode >= 0
                else None
            ),
            terminating_signal=(
                -process.returncode
                if process.returncode is not None and process.returncode < 0
                else None
            ),
            wall_time_ns=finished - started,
            timed_out=timed_out,
            cancellation_cause=cancellation_cause,
            cleanup_complete=True,
            peak_rss_bytes=resources.peak_rss_bytes if resources is not None else None,
            resources=resources,
        )
        return ExecutionOutcome(
            process=result,
            stdout=stdout,
            stderr=stderr,
            resolved_executable=executable,
            containment=(
                "systemd_scope"
                if request.systemd_scope_unit is not None
                else ("process_group" if os.name == "posix" else "process")
            ),
        )

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
        executable = self._resolve_executable(request.argv[0], cwd, environment)
        argv = (str(executable), *request.argv[1:])
        started = time.monotonic_ns()
        process: subprocess.Popen[bytes] | None = None
        cleanup_complete = True
        with tempfile.TemporaryFile() as stdout_stream, tempfile.TemporaryFile() as stderr_stream:
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=environment,
                    stdin=(
                        subprocess.PIPE if request.stdin_bytes is not None else subprocess.DEVNULL
                    ),
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    start_new_session=os.name == "posix",
                )
                if request.stdin_bytes is not None:
                    assert process.stdin is not None
                    try:
                        process.stdin.write(request.stdin_bytes)
                        process.stdin.flush()
                    except BrokenPipeError:
                        pass
                    finally:
                        process.stdin.close()

                observed = self._wait_observed(
                    process,
                    request,
                    stdout_stream=stdout_stream,
                    stderr_stream=stderr_stream,
                    cancellation=cancellation,
                )
                cleanup_complete = self._terminate_observed_group(process, force=True)
                stdout_stream.seek(0)
                stderr_stream.seek(0)
                stdout = stdout_stream.read()
                stderr = stderr_stream.read()
            except BaseException:
                if process is not None and process.poll() is None:
                    cleanup_complete = self._terminate_observed_group(process, force=True)
                    self._reap_observed(process)
                raise

        finished = time.monotonic_ns()
        if observed.cancellation_cause == "output_limit":
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Process output exceeded {request.max_output_bytes} bytes.",
            )
        if observed.cancellation_cause == "timeout":
            timeout_process = ProcessResult(
                exit_code=None,
                terminating_signal=(-observed.returncode if observed.returncode < 0 else None),
                wall_time_ns=finished - started,
                timed_out=True,
                cancellation_cause="timeout",
                cleanup_complete=cleanup_complete,
                peak_rss_bytes=observed.peak_rss_bytes,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )
            raise DomainError(
                ErrorCode.PROCESS_TIMEOUT,
                f"Process exceeded {request.timeout_seconds} seconds.",
                details={"process": timeout_process.model_dump(mode="json")},
                retryable=True,
            )

        process_result = ProcessResult(
            exit_code=observed.returncode if observed.returncode >= 0 else None,
            terminating_signal=(-observed.returncode if observed.returncode < 0 else None),
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
            containment=("process_group" if os.name == "posix" else "process"),
            peak_rss_backend=observed.peak_rss_backend,
        )

    def _wait_observed(
        self,
        process: subprocess.Popen[bytes],
        request: ExecutionRequest,
        *,
        stdout_stream: IO[bytes],
        stderr_stream: IO[bytes],
        cancellation: threading.Event | None,
    ) -> _ObservedWait:
        deadline = time.monotonic() + request.timeout_seconds
        terminating: Literal["timeout", "caller_cancelled", "output_limit"] | None = None
        wait4 = getattr(os, "wait4", None)
        if wait4 is not None:
            while True:
                if terminating is None:
                    if cancellation is not None and cancellation.is_set():
                        terminating = "caller_cancelled"
                        self._terminate_observed_group(process, force=True)
                    elif self._observed_output_exceeded(
                        stdout_stream, stderr_stream, request.max_output_bytes
                    ):
                        terminating = "output_limit"
                        self._terminate_observed_group(process, force=True)
                    elif time.monotonic() >= deadline:
                        terminating = "timeout"
                        self._terminate_observed_group(process, force=True)
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
                    terminating = "caller_cancelled"
                    self._terminate_observed_group(process, force=True)
                elif self._observed_output_exceeded(
                    stdout_stream, stderr_stream, request.max_output_bytes
                ):
                    terminating = "output_limit"
                    self._terminate_observed_group(process, force=True)
                elif time.monotonic() >= deadline:
                    terminating = "timeout"
                    self._terminate_observed_group(process, force=True)
            time.sleep(0.005)
        peak = max(peak, self._observed_peak_rss(process.pid))
        return _ObservedWait(
            returncode=process.returncode if process.returncode is not None else 0,
            peak_rss_bytes=peak or None,
            peak_rss_backend="psutil_polling",
            cancellation_cause=terminating,
        )

    def _observed_peak_rss(self, pid: int) -> int:
        peak = 0
        try:
            parent = psutil.Process(pid)
            processes = (parent, *parent.children(recursive=True))
            peak = sum(
                observed.memory_info().rss for observed in processes if observed.is_running()
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
        if process.returncode is not None:
            return True
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

        if process.poll() is None:
            process.terminate()
            if force:
                time.sleep(0.05)
                if process.poll() is None:
                    process.kill()
        return True

    @staticmethod
    def _observed_output_exceeded(
        stdout_stream: IO[bytes],
        stderr_stream: IO[bytes],
        max_output_bytes: int,
    ) -> bool:
        try:
            size = (
                os.fstat(stdout_stream.fileno()).st_size + os.fstat(stderr_stream.fileno()).st_size
            )
        except OSError:
            return False
        return size > max_output_bytes

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
    ) -> RuntimeResourceSummary | None:
        if policy is None:
            while process.returncode is None:
                await asyncio.sleep(0.1)
            return None
        interval = policy.sampling_interval_ms / 1_000
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
            try:
                parent = psutil.Process(process.pid)
                processes = (parent, *parent.children(recursive=True))
                rss = 0
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
                    termination="storage_reserve_exceeded",
                )
                raise _ResourcePolicyExceeded(summary)
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

    def _resource_summary(
        self,
        policy: ResourcePolicy,
        *,
        initial_sizes: dict[str, int | None],
        initial_staging: int | None,
        minimum_free: int | None,
        peak_rss: int,
        unavailable: set[str],
        termination: Literal["storage_reserve_exceeded"] | None,
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
    ) -> bytes:
        output = bytearray()
        while chunk := await stream.read(64 * 1024):
            budget.consume(len(chunk))
            output.extend(chunk)
        return bytes(output)

    async def _terminate(
        self,
        process: asyncio.subprocess.Process,
        request: ExecutionRequest,
    ) -> bool:
        if process.returncode is not None:
            return True
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
            return scope_stopped
        try:
            async with asyncio.timeout(request.graceful_shutdown_seconds):
                await process.wait()
                return scope_stopped
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
        return scope_stopped

    async def _stop_systemd_scope(
        self,
        unit: str,
        *,
        timeout_seconds: float,
    ) -> bool:
        systemctl = shutil.which("systemctl")
        if systemctl is None:
            return False
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

    def _resolve_executable(
        self,
        value: str,
        cwd: Path,
        environment: dict[str, str],
    ) -> Path:
        if os.sep in value or (os.altsep is not None and os.altsep in value):
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = cwd / candidate
            resolved = candidate.parent.resolve() / candidate.name
        else:
            located = shutil.which(value, path=environment.get("PATH"))
            if located is None:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    f"Executable {value!r} was not found in the allowed PATH.",
                )
            resolved = Path(located).absolute()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                f"Executable is not a runnable file: {resolved}",
            )
        return resolved
