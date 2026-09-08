from __future__ import annotations

import asyncio
import os
import signal
import sys
from contextlib import suppress
from pathlib import Path

import anyio
import psutil
import pytest
from pydantic import ValidationError

from flameox.command_binding import ExecutableResolver
from flameox.execution import (
    ExecutionRequest,
    ProcessCancelledError,
    SubprocessBroker,
)
from flameox.process_models import ProcessCancellationCause

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]

_PYTHON_BINDING = ExecutableResolver().require_host_tool(sys.executable)


def request(tmp_path: Path, *arguments: str, **overrides: object) -> ExecutionRequest:
    values: dict[str, object] = {
        "argv": (sys.executable, *arguments),
        "executable_binding": _PYTHON_BINDING,
        "cwd": tmp_path,
        "allowed_working_roots": (tmp_path,),
        "environment_allowlist": ("PATH",),
        "max_output_bytes": 1_000,
    }
    values.update(overrides)
    return ExecutionRequest.model_validate(values)


def process_is_alive(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def test_optional_deadline_is_unbounded_only_when_explicitly_none(tmp_path: Path) -> None:
    assert request(tmp_path, "-c", "pass").timeout_seconds == 300
    assert request(tmp_path, "-c", "pass", timeout_seconds=None).timeout_seconds is None
    with pytest.raises(ValidationError):
        request(tmp_path, "-c", "pass", timeout_seconds=0)
    with pytest.raises(ValidationError):
        request(tmp_path, "-c", "pass", timeout_seconds=86_400.1)


@pytest.mark.anyio
async def test_ordinary_execution_with_no_deadline_can_complete(tmp_path: Path) -> None:
    outcome = await SubprocessBroker().run(
        request(
            tmp_path,
            "-c",
            "import time; time.sleep(0.15); print('finished', flush=True)",
            timeout_seconds=None,
        )
    )
    assert outcome.stdout == b"finished\n"
    assert outcome.process.cancellation_cause is None


@pytest.mark.anyio
async def test_raw_task_cancel_with_no_deadline_settles_inherited_pipe_writer(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "grandchild.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "print('started', flush=True); time.sleep(60)"
    )
    task = asyncio.create_task(
        SubprocessBroker().run(
            request(tmp_path, "-c", code, str(child_pid_path), timeout_seconds=None)
        )
    )
    try:
        for _ in range(200):
            if child_pid_path.exists():
                break
            await asyncio.sleep(0.01)
        assert child_pid_path.exists()
        child_pid = int(child_pid_path.read_text())
        task.cancel()
        with pytest.raises(ProcessCancelledError) as cancelled:
            await asyncio.wait_for(task, timeout=2)
        assert (
            cancelled.value.process.cancellation_cause is ProcessCancellationCause.CALLER_CANCELLED
        )
        assert cancelled.value.process.cleanup_complete is True
        assert not process_is_alive(child_pid)
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if child_pid_path.exists():
            child_pid = int(child_pid_path.read_text())
            if process_is_alive(child_pid):
                os.kill(child_pid, signal.SIGKILL)


@pytest.mark.anyio
async def test_observed_no_deadline_cancellation_keeps_cleanup_contract(tmp_path: Path) -> None:
    pid_path = tmp_path / "observed.pid"
    with anyio.fail_after(5), anyio.CancelScope() as scope:

        async def cancel_started_child() -> None:
            while not pid_path.exists():
                await anyio.sleep(0.01)
            scope.cancel()

        async with anyio.create_task_group() as group:
            group.start_soon(cancel_started_child)
            with pytest.raises(ProcessCancelledError) as cancelled:
                await SubprocessBroker().run(
                    request(
                        tmp_path,
                        "-c",
                        "import os, pathlib, time; "
                        "pathlib.Path('observed.pid').write_text(str(os.getpid())); "
                        "time.sleep(60)",
                        timeout_seconds=None,
                        observation="child_peak_rss",
                    )
                )
    assert cancelled.value.process.cancellation_cause is ProcessCancellationCause.CALLER_CANCELLED
    assert cancelled.value.process.cleanup_complete is True
    assert not process_is_alive(int(pid_path.read_text()))
