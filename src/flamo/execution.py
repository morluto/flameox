from __future__ import annotations

import asyncio
import os
import shutil
import signal
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flamo.domain.errors import DomainError, ErrorCode
from flamo.domain.models import ProcessResult

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


class ExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    argv: tuple[str, ...]
    cwd: Path
    environment_allowlist: tuple[str, ...] = ("PATH",)
    environment_overrides: dict[str, str] = Field(default_factory=dict)
    allowed_working_roots: tuple[Path, ...]
    timeout_seconds: float = Field(default=300, gt=0, le=86_400)
    graceful_shutdown_seconds: float = Field(default=5, ge=0, le=60)
    max_output_bytes: int = Field(default=16_777_216, gt=0)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("argv must include an executable")
        if any(not item or "\x00" in item for item in value):
            raise ValueError("argv entries must be non-empty and cannot contain NUL")
        return value


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    process: ProcessResult
    stdout: bytes
    stderr: bytes
    resolved_executable: Path
    containment: str


class _OutputLimitExceeded(Exception):
    pass


@dataclass(slots=True)
class _OutputBudget:
    remaining: int

    def consume(self, byte_count: int) -> None:
        self.remaining -= byte_count
        if self.remaining < 0:
            raise _OutputLimitExceeded


class SubprocessBroker:
    async def run(
        self,
        request: ExecutionRequest,
        *,
        on_started: Callable[[int], Awaitable[None]] | None = None,
    ) -> ExecutionOutcome:
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
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        )
        try:
            process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            process = await asyncio.shield(spawn)
            await asyncio.shield(self._terminate(process, request.graceful_shutdown_seconds))
            raise
        if on_started is not None:
            try:
                await on_started(process.pid)
            except BaseException:
                await asyncio.shield(self._terminate(process, request.graceful_shutdown_seconds))
                raise
        assert process.stdout is not None
        assert process.stderr is not None

        output_budget = _OutputBudget(request.max_output_bytes)
        stdout_task = asyncio.create_task(self._read_bounded(process.stdout, output_budget))
        stderr_task = asyncio.create_task(self._read_bounded(process.stderr, output_budget))
        timed_out = False
        cancellation_cause: str | None = None
        try:
            async with asyncio.timeout(request.timeout_seconds):
                stdout, stderr, _ = await asyncio.gather(
                    stdout_task,
                    stderr_task,
                    process.wait(),
                )
        except TimeoutError as exc:
            timed_out = True
            cancellation_cause = "timeout"
            await asyncio.shield(self._terminate(process, request.graceful_shutdown_seconds))
            await self._settle_readers(stdout_task, stderr_task)
            raise DomainError(
                ErrorCode.PROCESS_TIMEOUT,
                f"Process exceeded {request.timeout_seconds} seconds.",
                retryable=True,
            ) from exc
        except _OutputLimitExceeded as exc:
            cancellation_cause = "output_limit"
            await asyncio.shield(self._terminate(process, request.graceful_shutdown_seconds))
            await self._settle_readers(stdout_task, stderr_task)
            raise DomainError(
                ErrorCode.QUERY_BUDGET_EXCEEDED,
                f"Process output exceeded {request.max_output_bytes} bytes.",
            ) from exc
        except asyncio.CancelledError:
            cancellation_cause = "caller_cancelled"
            await asyncio.shield(self._terminate(process, request.graceful_shutdown_seconds))
            await self._settle_readers(stdout_task, stderr_task)
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
        )
        return ExecutionOutcome(
            process=result,
            stdout=stdout,
            stderr=stderr,
            resolved_executable=executable,
            containment="process_group" if os.name == "posix" else "process",
        )

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
        grace_seconds: float,
    ) -> None:
        if process.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            await process.wait()
            return
        try:
            async with asyncio.timeout(grace_seconds):
                await process.wait()
                return
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

    async def _settle_readers(
        self,
        *tasks: asyncio.Task[bytes],
    ) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

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
            if name in os.environ and name not in _DANGEROUS_ENVIRONMENT
        }
        for name, value in request.environment_overrides.items():
            if name in _DANGEROUS_ENVIRONMENT:
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
