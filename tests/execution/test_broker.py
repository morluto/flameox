from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import psutil
import pytest

from flameox.command_binding import ExecutableResolver
from flameox.execution import (
    ExecutionRequest,
    ProcessCancelledError,
    ProcessExecutionError,
    ResourcePolicy,
    SubprocessBroker,
)
from flameox.process_models import ExitedProcessTermination
from flameox.runtime_errors import DomainError, ErrorCode

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


def _process_is_alive(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


@pytest.mark.anyio
async def test_cancellation_preserves_raw_output_without_waiting_for_inherited_pipes(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "sys.stdout.buffer.write(b'\\xffbefore cancel'); sys.stdout.flush(); time.sleep(60)"
    )
    task = asyncio.create_task(
        SubprocessBroker().run(request(tmp_path, "-c", code, str(child_pid_path)))
    )
    for _ in range(100):
        if child_pid_path.exists():
            break
        await asyncio.sleep(0.01)
    child_pid = int(child_pid_path.read_text())

    task.cancel()
    try:
        with pytest.raises(ProcessCancelledError) as cancelled:
            await asyncio.wait_for(task, timeout=2)
        assert cancelled.value.stdout == b"\xffbefore cancel"
        assert cancelled.value.process.cleanup_complete is True
        assert not _process_is_alive(child_pid)
    finally:
        if _process_is_alive(child_pid):
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
    started = time.monotonic()
    with pytest.raises(DomainError) as error:
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
    assert time.monotonic() - started < 2


def test_observed_run_preserves_streams_exit_status_and_peak_rss(tmp_path: Path) -> None:
    outcome = SubprocessBroker().run_sync(
        request(
            tmp_path,
            "-c",
            (
                "import sys; value = bytearray(2_000_000); "
                "print('observed stdout', flush=True); "
                "print('observed stderr', file=sys.stderr, flush=True); "
                "raise SystemExit(7)"
            ),
            observation="child_peak_rss",
        )
    )

    assert outcome.stdout == b"observed stdout\n"
    assert outcome.stderr == b"observed stderr\n"
    assert outcome.process.termination == ExitedProcessTermination(exit_code=7)
    assert outcome.process.peak_rss_bytes is not None
    assert any(item.snapshot_phase == "running" for item in outcome.process_observations)
    assert any(item.snapshot_phase == "post_root_exit" for item in outcome.process_observations)
    assert outcome.peak_rss_backend == (
        "wait4_ru_maxrss" if hasattr(os, "wait4") else "psutil_polling"
    )


def test_observed_output_budget_terminates_the_process(tmp_path: Path) -> None:
    with pytest.raises(ProcessExecutionError) as error:
        SubprocessBroker().run_sync(
            request(
                tmp_path,
                "-c",
                "import sys; sys.stdout.write('x' * 100_000); sys.stdout.flush()",
                observation="child_peak_rss",
                max_output_bytes=1_000,
            )
        )

    assert error.value.code is ErrorCode.LIMIT_EXCEEDED
    assert isinstance(error.value.details["process"], dict)
    assert error.value.details["process"]["cancellation_cause"] == "output_limit"
    assert error.value.details["process_observations"]
    assert error.value.stdout == b"x" * 1_000
    assert error.value.stderr == b""


def test_observed_output_budget_stops_a_burst_before_it_completes(tmp_path: Path) -> None:
    completed = tmp_path / "burst-completed"
    code = (
        "import pathlib, sys; "
        "sys.stdout.buffer.write(b'x' * (5 * 1024 * 1024)); "
        "sys.stdout.flush(); "
        "pathlib.Path(sys.argv[1]).write_text('completed')"
    )

    with pytest.raises(DomainError) as error:
        SubprocessBroker().run_sync(
            request(
                tmp_path,
                "-c",
                code,
                str(completed),
                observation="child_peak_rss",
                max_output_bytes=1_024,
            )
        )

    assert error.value.code is ErrorCode.LIMIT_EXCEEDED
    assert not completed.exists()


def test_observed_timeout_includes_large_stdin_transfer(tmp_path: Path) -> None:
    started = time.monotonic()

    with pytest.raises(DomainError) as error:
        SubprocessBroker().run_sync(
            request(
                tmp_path,
                "-c",
                "import sys, time; time.sleep(0.5); sys.stdin.buffer.read()",
                stdin_bytes=b"x" * (2 * 1024 * 1024),
                observation="child_peak_rss",
                timeout_seconds=0.05,
            )
        )

    assert error.value.code is ErrorCode.EXECUTION_TIMEOUT
    assert time.monotonic() - started < 0.5


def test_observed_timeout_cleans_up_the_process_group(tmp_path: Path) -> None:
    pid_path = tmp_path / "observed.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "time.sleep(30)"
    )

    with pytest.raises(DomainError) as error:
        SubprocessBroker().run_sync(
            request(
                tmp_path,
                "-c",
                code,
                str(pid_path),
                observation="child_peak_rss",
                timeout_seconds=0.1,
            )
        )

    assert error.value.code is ErrorCode.EXECUTION_TIMEOUT
    details = error.value.details["process"]
    assert isinstance(details, dict)
    assert details["cleanup_complete"] is True
    observations = error.value.details["process_observations"]
    assert isinstance(observations, list)
    assert {item["snapshot_phase"] for item in observations} >= {
        "pre_cleanup",
        "post_cleanup",
    }
    assert all(
        not set(item).intersection(
            {"cmdline", "environment", "cwd", "exe", "open_files", "connections"}
        )
        for item in observations
    )
    assert pid_path.is_file()
    child_pid = int(pid_path.read_text())
    for _ in range(100):
        if not _process_is_alive(child_pid):
            break
        time.sleep(0.01)
    assert not _process_is_alive(child_pid)


def test_observed_run_cleans_up_descendants_after_parent_exits(tmp_path: Path) -> None:
    pid_path = tmp_path / "observed-parent-exit.pid"
    code = (
        "import pathlib, subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))"
    )

    outcome = SubprocessBroker().run_sync(
        request(tmp_path, "-c", code, str(pid_path), observation="child_peak_rss")
    )

    assert outcome.process.termination == ExitedProcessTermination(exit_code=0)
    assert pid_path.is_file()
    child_pid = int(pid_path.read_text())
    for _ in range(100):
        if not _process_is_alive(child_pid):
            break
        time.sleep(0.01)
    assert not _process_is_alive(child_pid)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process sessions")
def test_observed_run_does_not_wait_for_an_escaped_output_writer(tmp_path: Path) -> None:
    pid_path = tmp_path / "escaped-output-writer.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(0.75)'], "
        "start_new_session=True); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(0.15)"
    )
    started = time.monotonic()

    child_pid: int | None = None
    try:
        outcome = SubprocessBroker().run_sync(
            request(tmp_path, "-c", code, str(pid_path), observation="child_peak_rss")
        )
        child_pid = int(pid_path.read_text())

        assert outcome.process.termination == ExitedProcessTermination(exit_code=0)
        assert time.monotonic() - started < 0.7
        assert not any(
            thread.name.startswith("flameox-observed-") for thread in threading.enumerate()
        )
    finally:
        if child_pid is not None:
            with suppress(ProcessLookupError):
                os.killpg(child_pid, signal.SIGKILL)


@pytest.mark.anyio
async def test_observed_cancellation_cleans_up_before_propagating(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "observed-cancel.pid"
    code = (
        "import pathlib, sys, time; "
        "pathlib.Path(sys.argv[1]).write_text(str(__import__('os').getpid())); "
        "time.sleep(30)"
    )
    task = asyncio.create_task(
        SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                code,
                str(pid_path),
                observation="child_peak_rss",
            )
        )
    )
    for _ in range(100):
        if pid_path.is_file():
            break
        await asyncio.sleep(0.01)
    assert pid_path.is_file()
    process_pid = int(pid_path.read_text())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(100):
        if not _process_is_alive(process_pid):
            break
        await asyncio.sleep(0.01)
    assert not _process_is_alive(process_pid)


@pytest.mark.anyio
async def test_observed_cancellation_interrupts_blocked_stdin(tmp_path: Path) -> None:
    task = asyncio.create_task(
        SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import sys, time; time.sleep(0.75); sys.stdin.buffer.read()",
                stdin_bytes=b"x" * (2 * 1024 * 1024),
                observation="child_peak_rss",
            )
        )
    )
    await asyncio.sleep(0.05)
    started = time.monotonic()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert time.monotonic() - started < 0.5


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
async def test_benign_python_runtime_controls_can_be_overridden(tmp_path: Path) -> None:
    outcome = await SubprocessBroker().run(
        request(
            tmp_path,
            "-c",
            "import os; print([os.environ[name] for name in ('PYTHONHASHSEED', "
            "'PYTHONUNBUFFERED', 'PYTHONIOENCODING')])",
            environment_overrides={
                "PYTHONHASHSEED": "random",
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
            },
        )
    )

    assert outcome.stdout == b"['random', '1', 'utf-8']\n"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "name",
    [
        "LD_AUDIT",
        "DYLD_INSERT_LIBRARIES",
        "NODE_OPTIONS",
        "PYTHONSTARTUP",
        "GDBINIT",
        "AWS_SECRET_ACCESS_KEY",
    ],
)
async def test_pattern_dangerous_environment_overrides_fail(
    tmp_path: Path,
    name: str,
) -> None:
    with pytest.raises(DomainError) as error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "pass",
                environment_overrides={name: "unsafe"},
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
    with pytest.raises(DomainError) as output_error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "print('x' * 10000)",
                max_output_bytes=100,
            )
        )

    assert timeout_error.value.code is ErrorCode.EXECUTION_TIMEOUT
    assert output_error.value.code is ErrorCode.LIMIT_EXCEEDED


@pytest.mark.anyio
async def test_timeout_terminates_descendants_outside_the_root_process_group(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "timeout-descendant.pid"
    child_options = ", start_new_session=True" if os.name == "posix" else ""
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']"
        f"{child_options}); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
    )

    with pytest.raises(DomainError) as error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                code,
                str(pid_path),
                timeout_seconds=0.3,
            )
        )

    assert error.value.code is ErrorCode.EXECUTION_TIMEOUT
    assert pid_path.is_file()
    assert not _process_is_alive(int(pid_path.read_text()))


@pytest.mark.anyio
@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process sessions")
async def test_timeout_terminates_observed_descendant_after_parent_exits(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "exited-parent-descendant.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(0.15)"
    )

    with pytest.raises(DomainError) as error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                code,
                str(pid_path),
                timeout_seconds=0.3,
            )
        )

    assert error.value.code is ErrorCode.EXECUTION_TIMEOUT
    assert pid_path.is_file()
    assert not _process_is_alive(int(pid_path.read_text()))


@pytest.mark.anyio
async def test_timeout_includes_startup_callback_and_cleans_up_child(tmp_path: Path) -> None:
    child_pid: int | None = None

    async def slow_started(pid: int) -> None:
        nonlocal child_pid
        child_pid = pid
        await asyncio.sleep(0.25)

    started = time.monotonic()
    with pytest.raises(DomainError) as error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import time; time.sleep(10)",
                timeout_seconds=0.05,
            ),
            on_started=slow_started,
        )

    assert error.value.code is ErrorCode.EXECUTION_TIMEOUT
    assert time.monotonic() - started < 0.2
    assert child_pid is not None
    assert not _process_is_alive(child_pid)


@pytest.mark.anyio
async def test_timeout_does_not_reawait_a_stalled_subprocess_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stalled_spawn(*_arguments: object, **_options: object) -> object:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", stalled_spawn)

    started = time.monotonic()
    with pytest.raises(DomainError) as error:
        await asyncio.wait_for(
            SubprocessBroker().run(request(tmp_path, "-c", "pass", timeout_seconds=0.05)),
            timeout=0.2,
        )

    assert error.value.code is ErrorCode.EXECUTION_TIMEOUT
    assert time.monotonic() - started < 0.2


@pytest.mark.anyio
async def test_output_budget_is_shared_between_stdout_and_stderr(
    tmp_path: Path,
) -> None:
    with pytest.raises(DomainError) as error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import sys; print('x' * 60); print('y' * 60, file=sys.stderr)",
                max_output_bytes=100,
            )
        )

    assert error.value.code is ErrorCode.LIMIT_EXCEEDED


@pytest.mark.anyio
async def test_cancellation_performs_cleanup_before_propagating(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    process_ids: list[int] = []
    cleanup: list[bool] = []

    async def record_started(process_id: int) -> None:
        process_ids.append(process_id)
        started.set()

    async def record_cleanup(complete: bool) -> None:
        cleanup.append(complete)

    task = asyncio.create_task(
        SubprocessBroker().run(
            request(tmp_path, "-c", "import time; time.sleep(10)"),
            on_started=record_started,
            on_cleanup=record_cleanup,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleanup == [True]
    assert len(process_ids) == 1
    assert not _process_is_alive(process_ids[0])


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
async def test_resource_policy_fails_closed_when_writable_growth_exceeds_file_scan_bound(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bounded-output"
    output.mkdir()
    code = (
        "import pathlib,sys,time; root=pathlib.Path(sys.argv[1]); "
        "[(root / str(i)).write_bytes(b'x') for i in range(3)]; time.sleep(10)"
    )

    with pytest.raises(DomainError) as error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                code,
                str(output),
                resource_policy=ResourcePolicy(
                    filesystem_path=tmp_path,
                    writable_roots=(output,),
                    minimum_free_bytes=0,
                    maximum_writable_growth_bytes=1024,
                    max_observed_files=2,
                    sampling_interval_ms=25,
                ),
            )
        )

    assert error.value.code is ErrorCode.LIMIT_EXCEEDED
    process = error.value.details["process"]
    assert process["cancellation_cause"] == "writable_limit_exceeded"
    assert "writable_root_growth_bytes" in process["resources"]["unavailable_metrics"]


@pytest.mark.anyio
async def test_resource_policy_marks_short_process_samples_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_disk_usage(_path: object) -> SimpleNamespace:
        raise OSError("disk usage unavailable")

    def unavailable_process(_pid: int) -> object:
        raise psutil.NoSuchProcess(_pid)

    monkeypatch.setattr("flameox.execution.shutil.disk_usage", unavailable_disk_usage)
    monkeypatch.setattr("flameox.execution.psutil.Process", unavailable_process)

    outcome = await SubprocessBroker().run(
        request(
            tmp_path,
            "-c",
            "import time; time.sleep(0.08)",
            resource_policy=ResourcePolicy(
                filesystem_path=tmp_path,
                minimum_free_bytes=0,
                sampling_interval_ms=25,
            ),
        )
    )

    resources = outcome.process.resources
    assert resources is not None
    assert resources.minimum_free_bytes is None
    assert resources.peak_rss_bytes is None
    assert resources.peak_rss_backend is None
    assert set(resources.unavailable_metrics) >= {
        "minimum_free_bytes",
        "peak_rss_bytes",
    }


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
