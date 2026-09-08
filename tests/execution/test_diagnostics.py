from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import anyio
import pytest

from flameox.command_binding import ExecutableResolver
from flameox.execution import (
    ExecutionRequest,
    ProcessCancelledError,
    ProcessExecutionError,
    SubprocessBroker,
)
from flameox.process_models import ExitedProcessTermination
from flameox.runtime_errors import ErrorCode

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]

_BINDING = ExecutableResolver().require_host_tool(sys.executable)


def request(tmp_path: Path, *arguments: str, **overrides: object) -> ExecutionRequest:
    values: dict[str, object] = {
        "argv": (sys.executable, *arguments),
        "executable_binding": _BINDING,
        "cwd": tmp_path,
        "allowed_working_roots": (tmp_path,),
        "environment_allowlist": ("PATH",),
        "timeout_seconds": 5,
        "max_output_bytes": 1_024,
    }
    values.update(overrides)
    return ExecutionRequest.model_validate(values)


@pytest.mark.anyio
async def test_diagnostics_drains_large_both_streams_without_shared_limit(tmp_path: Path) -> None:
    size = 17 * 1024 * 1024
    code = (
        "import os; n = int(__import__('sys').argv[1]); "
        "[os.write(1, b'a' * 65536) for _ in range(n // 65536)]; "
        "[os.write(2, b'b' * 65536) for _ in range(n // 65536)]"
    )
    outcome = await SubprocessBroker().run(
        request(tmp_path, "-c", code, str(size), diagnostic_bytes=4_096)
    )

    assert isinstance(outcome.process.termination, ExitedProcessTermination)
    assert outcome.process.termination.exit_code == 0
    assert outcome.stdout == b"a" * 4_096
    assert outcome.stderr == b"b" * 4_096
    assert outcome.diagnostic_output is not None
    assert outcome.diagnostic_output.stdout_observed_bytes == size
    assert outcome.diagnostic_output.stderr_observed_bytes == size
    assert outcome.diagnostic_output.stdout_retained_bytes == 4_096
    assert outcome.diagnostic_output.stderr_retained_bytes == 4_096
    assert outcome.diagnostic_output.stdout_omitted_bytes == size - 4_096
    assert outcome.diagnostic_output.stderr_omitted_bytes == size - 4_096
    assert outcome.diagnostic_output.stdout_complete
    assert outcome.diagnostic_output.stderr_complete


@pytest.mark.anyio
async def test_diagnostics_zero_retention_still_counts_complete_streams(tmp_path: Path) -> None:
    outcome = await SubprocessBroker().run(
        request(
            tmp_path,
            "-c",
            "import sys; sys.stdout.write('abc'); sys.stderr.write('de')",
            diagnostic_bytes=0,
        )
    )
    assert outcome.stdout == b""
    assert outcome.stderr == b""
    assert outcome.diagnostic_output is not None
    assert outcome.diagnostic_output.stdout_observed_bytes == 3
    assert outcome.diagnostic_output.stderr_observed_bytes == 2
    assert outcome.diagnostic_output.stdout_retained_bytes == 0
    assert outcome.diagnostic_output.stderr_retained_bytes == 0
    assert outcome.diagnostic_output.stdout_omitted_bytes == 3
    assert outcome.diagnostic_output.stderr_omitted_bytes == 2
    assert outcome.diagnostic_output.stdout_complete
    assert outcome.diagnostic_output.stderr_complete


def test_diagnostics_rejects_disk_sink_and_rss_observation(tmp_path: Path) -> None:
    for overrides in (
        {
            "diagnostic_bytes": 4_096,
            "output_directory": tmp_path / "sink",
            "output_root": tmp_path,
        },
        {"diagnostic_bytes": 4_096, "observation": "child_peak_rss"},
    ):
        with pytest.raises(ValueError):
            request(tmp_path, "-c", "pass", **overrides)


@pytest.mark.anyio
async def test_diagnostics_timeout_reports_incomplete_streams(tmp_path: Path) -> None:
    emitted = tmp_path / "emitted"
    with pytest.raises(ProcessExecutionError) as raised:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import pathlib,subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(30)'], stdout=sys.stdout, stderr=sys.stderr); "
                "sys.stdout.write('x'*8192); sys.stdout.flush(); pathlib.Path('emitted').touch(); "
                "time.sleep(30)",
                diagnostic_bytes=4_096,
                timeout_seconds=2,
            )
        )
    assert emitted.exists()
    error = raised.value
    assert error.code is ErrorCode.EXECUTION_TIMEOUT
    assert error.diagnostic_output is not None
    assert error.diagnostic_output.stdout_retained_bytes == 4_096
    assert error.diagnostic_output.stdout_omitted_bytes >= 4_096
    assert not error.diagnostic_output.stdout_complete


@pytest.mark.anyio
async def test_diagnostics_cancellation_does_not_fabricate_future_bytes(tmp_path: Path) -> None:
    emitted = tmp_path / "emitted"
    task = asyncio.create_task(
        SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import pathlib,sys,time; sys.stdout.write('x'*4096); sys.stdout.flush(); "
                "pathlib.Path('emitted').touch(); time.sleep(30)",
                diagnostic_bytes=1_024,
            )
        )
    )
    with anyio.fail_after(5):
        while not emitted.exists():
            await anyio.sleep(0.01)
    task.cancel()
    with pytest.raises(ProcessCancelledError) as raised:
        await task
    error = raised.value
    assert error.process.cleanup_complete
    assert error.stdout == b"x" * 1_024
    assert error.diagnostic_output is not None
    assert error.diagnostic_output.stdout_retained_bytes == 1_024
    assert not error.diagnostic_output.stdout_complete


@pytest.mark.anyio
async def test_diagnostics_task_cancellation_with_inherited_pipe_writer_is_incomplete(
    tmp_path: Path,
) -> None:
    emitted = tmp_path / "emitted"
    task = asyncio.create_task(
        SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import pathlib,subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(30)'], stdout=sys.stdout, stderr=sys.stderr); "
                "pathlib.Path('emitted').touch(); time.sleep(30)",
                diagnostic_bytes=128,
            )
        )
    )
    with anyio.fail_after(5):
        while not emitted.exists():
            await anyio.sleep(0.01)
    task.cancel()
    with pytest.raises(ProcessCancelledError) as raised:
        await task
    error = raised.value
    assert error.process.cleanup_complete
    assert error.diagnostic_output is not None
    assert not error.diagnostic_output.stdout_complete
    assert not error.diagnostic_output.stderr_complete


@pytest.mark.anyio
async def test_normal_mode_still_terminates_on_combined_output_cap(tmp_path: Path) -> None:
    with pytest.raises(ProcessExecutionError) as raised:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import os; os.write(1, b'a'*2048); os.write(2, b'b'*2048)",
            )
        )
    assert raised.value.code is ErrorCode.LIMIT_EXCEEDED
    assert raised.value.diagnostic_output is None
