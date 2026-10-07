from __future__ import annotations

import sys
from pathlib import Path

import anyio
import pytest

from flameox.command_binding import ExecutableResolver
from flameox.execution import (
    ExecutionRequest,
    ProcessCancelledError,
    SubprocessBroker,
)
from flameox.process_models import ProcessCancellationCause
from tests.support.processes import process_is_alive, wait_for_pid_file

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
async def test_observed_no_deadline_cancellation_keeps_cleanup_contract(tmp_path: Path) -> None:
    pid_path = tmp_path / "observed.pid"
    with anyio.fail_after(5), anyio.CancelScope() as scope:

        async def cancel_started_child() -> None:
            await wait_for_pid_file(pid_path)
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
