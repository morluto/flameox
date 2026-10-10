from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import anyio
import pytest

from flameox.command_binding import ExecutableResolver
from flameox.execution import (
    ExecutionRequest,
    ProcessCancelledError,
    ProcessExecutionError,
    ResourcePolicy,
    SubprocessBroker,
)
from flameox.process_models import ExitedProcessTermination, ProcessCancellationCause
from flameox.runtime_errors import DomainError, ErrorCode
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
        "timeout_seconds": 5,
        "max_output_bytes": 1_000,
    }
    values.update(overrides)
    return ExecutionRequest.model_validate(values)


@pytest.mark.anyio
async def test_sync_worker_bridge_cancellation_settles_child_and_retains_receipt(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "bridge.pid"
    receipts: list[ProcessCancelledError] = []
    execution = request(
        tmp_path,
        "-c",
        "import os, pathlib, time; print('before cancellation', flush=True); "
        "pathlib.Path('bridge.pid').write_text(str(os.getpid())); time.sleep(30)",
        timeout_seconds=3,
    )

    def execute() -> None:
        try:
            SubprocessBroker().run_sync(execution)
        except ProcessCancelledError as error:
            receipts.append(error)
            raise

    with anyio.fail_after(6), anyio.CancelScope() as scope:

        async def cancel_started_child() -> None:
            await wait_for_pid_file(pid_path)
            scope.cancel()

        async with anyio.create_task_group() as group:
            group.start_soon(cancel_started_child)
            await anyio.to_thread.run_sync(execute)

    assert len(receipts) == 1
    assert receipts[0].stdout == b"before cancellation\n"
    assert receipts[0].process.cleanup_complete
    assert receipts[0].process.cancellation_cause is ProcessCancellationCause.CALLER_CANCELLED
    assert not process_is_alive(int(pid_path.read_text()))


@pytest.mark.anyio
async def test_cancellation_preserves_raw_output_without_waiting_for_inherited_pipes(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "sys.stdout.buffer.write(b'\\xffbefore cancel'); sys.stdout.flush(); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    )
    task = asyncio.create_task(
        SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                code,
                str(child_pid_path),
                timeout_seconds=None,
            )
        )
    )
    child_pid: int | None = None
    try:
        child_pid = await wait_for_pid_file(child_pid_path)
        task.cancel()
        with pytest.raises(ProcessCancelledError) as cancelled:
            await asyncio.wait_for(task, timeout=2)
        assert cancelled.value.stdout == b"\xffbefore cancel"
        assert cancelled.value.process.cleanup_complete is True
        assert not process_is_alive(child_pid)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if child_pid is not None and process_is_alive(child_pid):
            os.kill(child_pid, signal.SIGKILL)


@pytest.mark.anyio
async def test_shell_metacharacters_remain_literal_arguments(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"
    literal = f"$(touch {marker})"

    outcome = await SubprocessBroker().run(
        request(
            tmp_path,
            "-c",
            "import sys; print(sys.argv[1])",
            literal,
            timeout_seconds=None,
        )
    )

    assert outcome.stdout.decode().strip() == literal
    assert not marker.exists()
    assert outcome.process.termination == ExitedProcessTermination(exit_code=0)
    assert any(item.snapshot_phase == "running" for item in outcome.process_observations)
    assert any(item.snapshot_phase == "post_root_exit" for item in outcome.process_observations)


@pytest.mark.anyio
async def test_broker_passes_bounded_stdin_to_jsonc_style_helpers(tmp_path: Path) -> None:
    outcome = await SubprocessBroker().run(
        request(
            tmp_path,
            "-c",
            "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())",
            stdin_bytes=b'{"operation":"modify"}',
        )
    )

    assert outcome.stdout == b'{"operation":"modify"}'
    assert outcome.process.termination == ExitedProcessTermination(exit_code=0)


@pytest.mark.anyio
async def test_broker_bounds_stdin_transfer_when_child_does_not_read(tmp_path: Path) -> None:
    with anyio.fail_after(5), pytest.raises(ProcessExecutionError) as error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import time; time.sleep(30)",
                stdin_bytes=b"x" * 10_000_000,
                timeout_seconds=0.1,
            )
        )

    assert error.value.code is ErrorCode.EXECUTION_TIMEOUT
    assert error.value.process.cleanup_complete is True


@pytest.mark.anyio
async def test_output_limit_interrupts_a_slow_resource_observer(tmp_path: Path) -> None:
    with anyio.fail_after(5), pytest.raises(ProcessExecutionError) as error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import sys,time; time.sleep(0.1); "
                "sys.stdout.write('x' * 100_000); sys.stdout.flush(); time.sleep(30)",
                max_output_bytes=1_000,
                timeout_seconds=None,
                resource_policy=ResourcePolicy(
                    filesystem_path=tmp_path,
                    minimum_free_bytes=0,
                    sampling_interval_ms=10_000,
                ),
            )
        )

    assert error.value.code is ErrorCode.LIMIT_EXCEEDED
    assert isinstance(error.value.details["process"], dict)
    assert error.value.details["process"]["cancellation_cause"] == "output_limit"
    assert error.value.details["process_observations"]
    assert error.value.stdout == b"x" * 1_000
    assert error.value.stderr == b""


def test_run_cleans_up_descendants_after_parent_exits(tmp_path: Path) -> None:
    pid_path = tmp_path / "observed-parent-exit.pid"
    code = (
        "import pathlib, subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))"
    )

    outcome = SubprocessBroker().run_sync(request(tmp_path, "-c", code, str(pid_path)))

    assert outcome.process.termination == ExitedProcessTermination(exit_code=0)
    assert pid_path.is_file()
    child_pid = int(pid_path.read_text())
    assert not process_is_alive(child_pid)


@pytest.mark.anyio
async def test_no_deadline_cancellation_keeps_cleanup_contract(tmp_path: Path) -> None:
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
                    )
                )
    assert cancelled.value.process.cancellation_cause is ProcessCancellationCause.CALLER_CANCELLED
    assert cancelled.value.process.cleanup_complete is True
    assert not process_is_alive(int(pid_path.read_text()))


@pytest.mark.anyio
async def test_environment_is_allowlisted_and_dangerous_overrides_fail(
    tmp_path: Path,
) -> None:
    outcome = await SubprocessBroker().run(
        request(
            tmp_path,
            "-c",
            "import os; print(os.getenv('SECRET_FOR_TEST'))",
        )
    )
    assert outcome.stdout == b"None\n"

    with pytest.raises(DomainError) as error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "pass",
                environment_overrides={"PYTHONPATH": str(tmp_path / "unsafe")},
            )
        )
    assert error.value.code is ErrorCode.INVALID_INPUT


@pytest.mark.anyio
async def test_timeout_and_output_budget_terminate_process(tmp_path: Path) -> None:
    with pytest.raises(DomainError) as timeout_error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import time; time.sleep(10)",
                timeout_seconds=0.05,
            )
        )
    with pytest.raises(ProcessExecutionError) as output_error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import os; os.write(1, b'a' * 64); os.write(2, b'b' * 64)",
                max_output_bytes=100,
            )
        )

    assert timeout_error.value.code is ErrorCode.EXECUTION_TIMEOUT
    assert output_error.value.code is ErrorCode.LIMIT_EXCEEDED
    assert len(output_error.value.stdout or b"") + len(output_error.value.stderr or b"") == 100
    assert output_error.value.diagnostic_output is None


@pytest.mark.anyio
async def test_resource_policy_records_descendant_rss_disk_floor_and_root_growth(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    code = (
        "import pathlib, subprocess, sys; "
        "pathlib.Path(sys.argv[1]).write_bytes(b'x' * 4096); "
        "subprocess.run([sys.executable, '-c', "
        "'import time; value = bytearray(4000000); time.sleep(0.2)'], check=True)"
    )
    outcome = await SubprocessBroker().run(
        request(
            tmp_path,
            "-c",
            code,
            str(output / "build.bin"),
            resource_policy=ResourcePolicy(
                filesystem_path=tmp_path,
                writable_roots=(output,),
                minimum_free_bytes=0,
                sampling_interval_ms=25,
            ),
        )
    )

    resources = outcome.process.resources
    assert resources is not None
    assert resources.minimum_free_bytes is not None
    assert resources.peak_rss_bytes is not None
    assert resources.peak_rss_bytes > 0
    assert resources.peak_rss_backend == "psutil_recursive_polling"
    assert resources.writable_root_growth_bytes[str(output)] == 4096
    assert resources.policy_termination is None
    assert outcome.process.peak_rss_bytes == resources.peak_rss_bytes


@pytest.mark.anyio
async def test_resource_policy_terminates_process_tree_above_memory_limit(
    tmp_path: Path,
) -> None:
    with pytest.raises(DomainError) as error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import time; value = bytearray(20_000_000); time.sleep(10)",
                resource_policy=ResourcePolicy(
                    filesystem_path=tmp_path,
                    writable_roots=(tmp_path,),
                    minimum_free_bytes=0,
                    maximum_rss_bytes=1_000_000,
                    sampling_interval_ms=25,
                ),
            )
        )

    assert error.value.code is ErrorCode.LIMIT_EXCEEDED
    process = error.value.details["process"]
    assert process["cancellation_cause"] == "memory_limit_exceeded"
    assert process["resources"]["policy_termination"] == "memory_limit_exceeded"


@pytest.mark.anyio
async def test_resource_policy_terminates_process_tree_above_writable_growth_limit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bounded-output"
    output.mkdir()
    code = (
        "import pathlib,sys,time; "
        "pathlib.Path(sys.argv[1]).write_bytes(b'x' * 1000000); time.sleep(10)"
    )

    with pytest.raises(DomainError) as error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                code,
                str(output / "oversized.bin"),
                resource_policy=ResourcePolicy(
                    filesystem_path=tmp_path,
                    writable_roots=(output,),
                    minimum_free_bytes=0,
                    maximum_writable_growth_bytes=1024,
                    sampling_interval_ms=25,
                ),
            )
        )

    assert error.value.code is ErrorCode.LIMIT_EXCEEDED
    process = error.value.details["process"]
    assert process["cancellation_cause"] == "writable_limit_exceeded"
    assert process["resources"]["policy_termination"] == "writable_limit_exceeded"


@pytest.mark.anyio
async def test_resource_policy_terminates_scope_with_structured_storage_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = 0

    def disk_usage(_path: object) -> SimpleNamespace:
        nonlocal checks
        checks += 1
        free = 100 if checks == 1 else 1
        return SimpleNamespace(total=100, used=100 - free, free=free)

    monkeypatch.setattr(
        "flameox.execution.shutil.disk_usage",
        disk_usage,
    )
    with pytest.raises(DomainError) as exceeded:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import time; print('partial', flush=True); time.sleep(10)",
                resource_policy=ResourcePolicy(
                    filesystem_path=tmp_path,
                    minimum_free_bytes=2,
                    sampling_interval_ms=25,
                ),
            )
        )

    assert exceeded.value.code is ErrorCode.LIMIT_EXCEEDED
    process = exceeded.value.details["process"]
    assert isinstance(process, dict)
    assert process["cancellation_cause"] == "storage_reserve_exceeded"
    assert process["cleanup_complete"] is True
    assert process["stdout"] == "partial\n"
    assert process["resources"]["policy_termination"] == "storage_reserve_exceeded"


@pytest.mark.anyio
async def test_systemd_scope_cancellation_terminates_escaped_descendants(
    tmp_path: Path,
) -> None:
    systemd_run = shutil.which("systemd-run")
    if systemd_run is None:
        pytest.skip("systemd-run is not installed")
    probe = subprocess.run(
        (
            systemd_run,
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            "/usr/bin/true",
        ),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode != 0:
        pytest.skip("A systemd user manager is not available")
    unit = f"flameox-test-{uuid4().hex}.scope"
    pid_path = tmp_path / "descendant.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(30)'], "
        "start_new_session=True); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    cleanup: list[bool] = []

    async def record_cleanup(complete: bool) -> None:
        cleanup.append(complete)

    task = asyncio.create_task(
        SubprocessBroker().run(
            ExecutionRequest(
                argv=(
                    systemd_run,
                    "--user",
                    "--scope",
                    "--quiet",
                    "--collect",
                    "--expand-environment=no",
                    f"--unit={unit}",
                    "--property=KillMode=control-group",
                    "--",
                    sys.executable,
                    "-c",
                    code,
                    str(pid_path),
                ),
                executable_binding=ExecutableResolver().require_host_tool(
                    systemd_run, cwd=tmp_path
                ),
                cwd=tmp_path,
                allowed_working_roots=(tmp_path,),
                environment_allowlist=("PATH",),
                timeout_seconds=5,
                max_output_bytes=1_000,
                systemd_scope_unit=unit,
            ),
            on_cleanup=record_cleanup,
        )
    )
    for _ in range(100):
        if pid_path.is_file():
            break
        await asyncio.sleep(0.02)
    assert pid_path.is_file()
    descendant_pid = int(pid_path.read_text())
    assert os.path.exists(f"/proc/{descendant_pid}")

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(100):
        if not os.path.exists(f"/proc/{descendant_pid}"):
            break
        await asyncio.sleep(0.02)

    assert cleanup == [True]
    assert not os.path.exists(f"/proc/{descendant_pid}")
