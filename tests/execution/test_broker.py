from __future__ import annotations

import asyncio
import io
import os
import shutil
import signal
import subprocess
import sys
import threading
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from uuid import uuid4

import anyio
import psutil
import pytest

from flameox import execution as execution_module
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
@pytest.mark.parametrize("phase", ["before_start", "protocol_setup", "transport_handshake"])
async def test_startup_cancellation_retains_receipt_and_settles_acquired_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    pids: list[int] = []
    receipts: list[ProcessCancelledError] = []
    cleanup: list[bool] = []
    scope = anyio.CancelScope()
    original = asyncio.subprocess.SubprocessStreamProtocol.connection_made
    original_init = asyncio.subprocess.SubprocessStreamProtocol.__init__

    def initialize(
        protocol: asyncio.subprocess.SubprocessStreamProtocol,
        limit: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        original_init(protocol, limit, loop)
        if phase == "protocol_setup":
            scope.cancel()

    def connected(
        protocol: asyncio.subprocess.SubprocessStreamProtocol, transport: asyncio.BaseTransport
    ) -> None:
        original(protocol, transport)
        pids.append(cast(asyncio.SubprocessTransport, transport).get_pid())
        if phase == "transport_handshake":
            scope.cancel()

    async def cleaned(complete: bool) -> None:
        await anyio.sleep(0)
        cleanup.append(complete)

    monkeypatch.setattr(asyncio.subprocess.SubprocessStreamProtocol, "connection_made", connected)
    monkeypatch.setattr(asyncio.subprocess.SubprocessStreamProtocol, "__init__", initialize)
    try:
        with anyio.fail_after(5), scope:
            if phase == "before_start":
                scope.cancel()
            try:
                await SubprocessBroker().run(
                    request(tmp_path, "-c", "import time; time.sleep(30)"), on_cleanup=cleaned
                )
            except ProcessCancelledError as error:
                receipts.append(error)
                raise
        assert len(receipts) == 1
        assert cleanup == [True]
        assert receipts[0].process.cleanup_complete is True
        assert receipts[0].process.cancellation_cause is ProcessCancellationCause.CALLER_CANCELLED
        assert len(pids) == (0 if phase == "before_start" else 1)
        assert all(not process_is_alive(pid) for pid in pids)
    finally:
        for pid in pids:
            if process_is_alive(pid):
                os.kill(pid, signal.SIGKILL)
        await anyio.sleep(0.1)


@pytest.mark.anyio
@pytest.mark.parametrize("observation", [None, "child_peak_rss"])
async def test_sync_worker_bridge_cancellation_settles_child_and_retains_receipt(
    tmp_path: Path, observation: str | None
) -> None:
    pid_path = tmp_path / "bridge.pid"
    receipts: list[ProcessCancelledError] = []
    execution = request(
        tmp_path,
        "-c",
        "import os, pathlib, time; print('before cancellation', flush=True); "
        "pathlib.Path('bridge.pid').write_text(str(os.getpid())); time.sleep(30)",
        observation=observation,
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
async def test_sync_worker_bridge_callbacks_run_on_owning_loop(tmp_path: Path) -> None:
    owning_loop = asyncio.get_running_loop()
    callback_loops: list[asyncio.AbstractEventLoop] = []

    async def on_started(pid: int) -> None:
        assert process_is_alive(pid)
        callback_loops.append(asyncio.get_running_loop())

    def execute() -> bytes:
        outcome = SubprocessBroker().run_sync(
            request(tmp_path, "-c", "print('bridge output')"), on_started=on_started
        )
        return outcome.stdout

    assert await anyio.to_thread.run_sync(execute) == b"bridge output\n"
    assert callback_loops == [owning_loop]


@pytest.mark.anyio
async def test_sync_worker_bridge_does_not_retry_callback_failure(tmp_path: Path) -> None:
    started_pids: list[int] = []

    async def on_started(pid: int) -> None:
        started_pids.append(pid)
        raise RuntimeError("callback failure")

    def execute() -> None:
        SubprocessBroker().run_sync(
            request(tmp_path, "-c", "import time; time.sleep(30)"), on_started=on_started
        )

    with pytest.raises(RuntimeError, match="callback failure"):
        await anyio.to_thread.run_sync(execute)
    assert len(started_pids) == 1
    assert not process_is_alive(started_pids[0])


@pytest.mark.anyio
@pytest.mark.parametrize("inherited_pipes", [False, True])
async def test_output_limit_wakes_resource_observer_before_its_next_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inherited_pipes: bool
) -> None:
    child_pid_path = tmp_path / "writer.pid"
    original_disk_usage = shutil.disk_usage

    def signal_first_sample(path: Any) -> Any:
        usage = original_disk_usage(path)
        (tmp_path / "emit").touch()
        return usage

    monkeypatch.setattr(shutil, "disk_usage", signal_first_sample)
    writer = (
        "import pathlib,time\n"
        "while not pathlib.Path('emit').exists(): time.sleep(0.01)\n"
        "print('x' * 2000, flush=True); time.sleep(30)"
    )
    code = writer
    if inherited_pipes:
        code = (
            "import pathlib, subprocess, sys; "
            f"child = subprocess.Popen([sys.executable, '-c', {writer!r}], "
            "start_new_session=True); "
            "pathlib.Path('writer.pid').write_text(str(child.pid))"
        )
    try:
        # The observer's next sample is ten seconds away. Output completion
        # must wake it, even with no workload deadline to end the observation.
        with anyio.fail_after(5), pytest.raises(ProcessExecutionError) as failure:
            await SubprocessBroker().run(
                request(
                    tmp_path,
                    "-c",
                    code,
                    timeout_seconds=None,
                    resource_policy=ResourcePolicy(
                        filesystem_path=tmp_path,
                        sampling_interval_ms=10_000,
                        minimum_free_bytes=0,
                    ),
                )
            )
        assert failure.value.code is ErrorCode.LIMIT_EXCEEDED
        assert failure.value.process.resources is not None
    finally:
        if child_pid_path.exists():
            child_pid = int(child_pid_path.read_text())
            if process_is_alive(child_pid):
                os.kill(child_pid, signal.SIGKILL)
            await asyncio.sleep(0.1)


@pytest.mark.anyio
async def test_observed_level_cancellation_settles_thread_and_cleanup(tmp_path: Path) -> None:
    pid_path = tmp_path / "observed.pid"
    cleanup: list[bool] = []

    async def on_cleanup(complete: bool) -> None:
        await anyio.sleep(0)
        cleanup.append(complete)

    with anyio.fail_after(5), anyio.CancelScope() as scope:

        async def cancel_started_child() -> None:
            await wait_for_pid_file(pid_path)
            scope.cancel()

        async with anyio.create_task_group() as group:
            group.start_soon(cancel_started_child)
            await SubprocessBroker().run(
                request(
                    tmp_path,
                    "-c",
                    "import os, pathlib, time; "
                    "pathlib.Path('observed.pid').write_text(str(os.getpid())); time.sleep(30)",
                    observation="child_peak_rss",
                ),
                on_cleanup=on_cleanup,
            )
    assert cleanup == [True]
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
        SubprocessBroker().run(request(tmp_path, "-c", code, str(child_pid_path)))
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
    with pytest.raises(ProcessExecutionError) as error:
        SubprocessBroker().run_sync(
            request(
                tmp_path,
                "-c",
                "import pathlib, sys, time; time.sleep(30); "
                "pathlib.Path('stdin-read-started').touch(); sys.stdin.buffer.read()",
                stdin_bytes=b"x" * (2 * 1024 * 1024),
                observation="child_peak_rss",
                timeout_seconds=0.05,
            )
        )

    assert error.value.code is ErrorCode.EXECUTION_TIMEOUT
    assert error.value.process.cleanup_complete is True
    assert not (tmp_path / "stdin-read-started").exists()


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
                timeout_seconds=2,
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
    assert not process_is_alive(child_pid)


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
    assert not process_is_alive(child_pid)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process sessions")
def test_observed_run_does_not_wait_for_an_escaped_output_writer(tmp_path: Path) -> None:
    pid_path = tmp_path / "escaped-output-writer.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(30)'], "
        "start_new_session=True); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(0.15)"
    )
    child_pid: int | None = None
    try:
        outcome = SubprocessBroker().run_sync(
            request(tmp_path, "-c", code, str(pid_path), observation="child_peak_rss")
        )
        child_pid = int(pid_path.read_text())

        assert outcome.process.termination == ExitedProcessTermination(exit_code=0)
        # Returning while the escaped writer still owns the pipe proves that
        # completion did not wait for its EOF, independently of host startup time.
        assert process_is_alive(child_pid)
        assert not any(
            thread.name.startswith("flameox-observed-") for thread in threading.enumerate()
        )
    finally:
        if child_pid is None and pid_path.exists():
            child_pid = int(pid_path.read_text())
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
    process_pid = await wait_for_pid_file(pid_path)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not process_is_alive(process_pid)


@pytest.mark.anyio
async def test_observed_cancellation_interrupts_blocked_stdin(tmp_path: Path) -> None:
    pid_path = tmp_path / "stdin-writer.pid"
    task = asyncio.create_task(
        SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import os, pathlib, sys, time; "
                "pathlib.Path('stdin-writer.pid').write_text(str(os.getpid())); "
                "time.sleep(30); sys.stdin.buffer.read()",
                stdin_bytes=b"x" * (2 * 1024 * 1024),
                observation="child_peak_rss",
            )
        )
    )
    try:
        process_pid = await wait_for_pid_file(pid_path)
        assert process_is_alive(process_pid)
        task.cancel()
        with anyio.fail_after(5), pytest.raises(ProcessCancelledError) as cancelled:
            await task
        assert cancelled.value.process.cleanup_complete is True
        assert not process_is_alive(process_pid)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


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
    assert not process_is_alive(int(pid_path.read_text()))


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
    assert not process_is_alive(int(pid_path.read_text()))


@pytest.mark.anyio
async def test_timeout_includes_startup_callback_and_cleans_up_child(tmp_path: Path) -> None:
    child_pid: int | None = None

    async def slow_started(pid: int) -> None:
        nonlocal child_pid
        child_pid = pid
        await anyio.sleep_forever()

    # Prove that the workload deadline interrupts startup, independently of the
    # host time needed to acquire the process and finish descendant cleanup.
    with anyio.fail_after(5), pytest.raises(DomainError) as error:
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
    assert child_pid is not None
    assert not process_is_alive(child_pid)


@pytest.mark.anyio
async def test_timeout_does_not_reawait_a_stalled_subprocess_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stalled_spawn(*_arguments: object, **_options: object) -> object:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", stalled_spawn)

    with anyio.fail_after(5), pytest.raises(DomainError) as error:
        await SubprocessBroker().run(request(tmp_path, "-c", "pass", timeout_seconds=0.05))

    assert error.value.code is ErrorCode.EXECUTION_TIMEOUT


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
async def test_async_output_sink_retains_exact_bounded_streams(tmp_path: Path) -> None:
    sink = tmp_path / "sink"
    outcome = await SubprocessBroker().run(
        request(
            tmp_path,
            "-c",
            "import sys; sys.stdout.buffer.write(b'o' * 1200); sys.stderr.buffer.write(b'e' * 700)",
            max_output_bytes=4_000,
            output_directory=sink,
            output_root=tmp_path,
        )
    )

    assert outcome.output_sink is not None
    assert outcome.output_sink.complete is True
    assert outcome.output_sink.stdout_bytes == 1_200
    assert outcome.output_sink.stderr_bytes == 700
    assert outcome.output_sink.stdout_path.read_bytes() == b"o" * 1_200
    assert outcome.output_sink.stderr_path.read_bytes() == b"e" * 700


@pytest.mark.anyio
async def test_async_output_sink_keeps_large_stream_out_of_preview_memory(
    tmp_path: Path,
) -> None:
    sink = tmp_path / "sink"
    size = 17 * 1024 * 1024
    outcome = await SubprocessBroker().run(
        request(
            tmp_path,
            "-c",
            "import os, threading; "
            f"a=threading.Thread(target=lambda: os.write(1, b'o' * {size})); "
            f"a.start(); os.write(2, b'e' * {size}); a.join()",
            max_output_bytes=2 * size + 1,
            timeout_seconds=15,
            output_directory=sink,
            output_root=tmp_path,
        )
    )

    assert outcome.output_sink is not None
    assert outcome.output_sink.stdout_bytes == size
    assert outcome.output_sink.complete is True
    assert len(outcome.stdout) == 64 * 1024
    assert outcome.output_sink.stdout_path.stat().st_size == size
    assert outcome.output_sink.stderr_bytes == size
    assert len(outcome.stderr) == 64 * 1024
    for path, expected in (
        (outcome.output_sink.stdout_path, b"o"),
        (outcome.output_sink.stderr_path, b"e"),
    ):
        with path.open("rb") as stream:
            while chunk := stream.read(64 * 1024):
                assert chunk == expected * len(chunk)


@pytest.mark.anyio
async def test_async_output_sink_marks_partial_prefix_on_limit(tmp_path: Path) -> None:
    sink = tmp_path / "sink"
    with pytest.raises(ProcessExecutionError) as failure:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import sys; sys.stdout.buffer.write(b'o' * 1000); "
                "sys.stderr.buffer.write(b'e' * 1000)",
                max_output_bytes=1_000,
                output_directory=sink,
                output_root=tmp_path,
            )
        )

    assert failure.value.output_sink is not None
    assert failure.value.output_sink.complete is False
    assert failure.value.output_sink.stdout_bytes + failure.value.output_sink.stderr_bytes == 1_000
    sink_metadata = failure.value.output_sink
    for path, byte_count, preview, preview_count in (
        (
            sink_metadata.stdout_path,
            sink_metadata.stdout_bytes,
            failure.value.stdout,
            sink_metadata.stdout_preview_bytes,
        ),
        (
            sink_metadata.stderr_path,
            sink_metadata.stderr_bytes,
            failure.value.stderr,
            sink_metadata.stderr_preview_bytes,
        ),
    ):
        native = path.read_bytes()
        assert len(native) == byte_count
        assert preview is not None
        assert native.startswith(preview)
        assert len(preview) == preview_count
    assert failure.value.details["output_sink"]["complete"] is False


@pytest.mark.anyio
async def test_async_output_sink_settles_inherited_writer_on_cancellation(tmp_path: Path) -> None:
    sink = tmp_path / "sink"
    child_pid_path = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "sys.stdout.write('before cancel'); sys.stdout.flush(); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    )
    task = asyncio.create_task(
        SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                code,
                str(child_pid_path),
                output_directory=sink,
                output_root=tmp_path,
            )
        )
    )
    child_pid: int | None = None
    try:
        child_pid = await wait_for_pid_file(child_pid_path)
        task.cancel()
        with pytest.raises(ProcessCancelledError) as cancelled:
            await asyncio.wait_for(task, timeout=3)
        assert cancelled.value.output_sink is not None
        assert cancelled.value.output_sink.complete is False
        assert cancelled.value.output_sink.stdout_path.read_bytes() == b"before cancel"
        assert not process_is_alive(child_pid)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if child_pid is not None and process_is_alive(child_pid):
            os.kill(child_pid, signal.SIGKILL)


@pytest.mark.anyio
async def test_async_output_sink_does_not_clobber_existing_hardlink(tmp_path: Path) -> None:
    sink = tmp_path / "sink"
    sink.mkdir()
    victim = tmp_path / "victim"
    victim.write_bytes(b"protected")
    os.link(victim, sink / "stdout")

    with pytest.raises(FileExistsError):
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "print('must not launch')",
                output_directory=sink,
                output_root=tmp_path,
            )
        )
    assert victim.read_bytes() == b"protected"


@pytest.mark.anyio
async def test_async_output_sink_does_not_overwrite_existing_regular_file(tmp_path: Path) -> None:
    sink = tmp_path / "sink"
    sink.mkdir()
    existing = sink / "stdout"
    existing.write_bytes(b"protected")

    with pytest.raises(FileExistsError):
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "print('must not launch')",
                output_directory=sink,
                output_root=tmp_path,
            )
        )
    assert existing.read_bytes() == b"protected"


@pytest.mark.anyio
async def test_async_output_sink_cancellation_waits_for_blocked_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = tmp_path / "sink"
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    loop = asyncio.get_running_loop()
    release = threading.Event()
    original = execution_module._AsyncOutputSink.write

    def blocked_write(
        output_sink: execution_module._AsyncOutputSink,
        stream: Literal["stdout", "stderr"],
        content: bytes,
    ) -> None:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10), "Test did not release the blocked output write"
        original(output_sink, stream, content)

    async def on_cleanup(complete: bool) -> None:
        assert complete
        cleaned.set()

    monkeypatch.setattr(execution_module._AsyncOutputSink, "write", blocked_write)
    task = asyncio.create_task(
        SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import sys, time; print('blocked', flush=True); time.sleep(30)",
                output_directory=sink,
                output_root=tmp_path,
            ),
            on_cleanup=on_cleanup,
        )
    )
    try:
        with anyio.fail_after(5):
            await entered.wait()
            task.cancel()
            await cleaned.wait()
            await anyio.wait_all_tasks_blocked()
            assert not task.done()
            release.set()
            with pytest.raises(ProcessCancelledError) as cancelled:
                await task
        assert cancelled.value.output_sink is not None
        assert cancelled.value.output_sink.stdout_path.read_bytes() == b"blocked\n"
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_async_output_sink_close_failure_retains_typed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CloseFailure(io.FileIO):
        def close(self) -> None:
            super().close()
            raise OSError("injected close failure")

    def open_with_close_failure(descriptor: int, mode: str, buffering: int) -> io.FileIO:
        return CloseFailure(descriptor, mode=mode)

    monkeypatch.setattr(os, "fdopen", open_with_close_failure)
    with pytest.raises(ProcessExecutionError) as failure:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "print('retained', flush=True)",
                output_directory=tmp_path / "sink",
                output_root=tmp_path,
            )
        )
    assert failure.value.process.cancellation_cause is ProcessCancellationCause.IO_FAILURE
    sink = failure.value.output_sink
    assert sink is not None
    assert sink.io_error and not sink.complete
    assert sink.stdout_path.read_bytes() == b"retained\n"


@pytest.mark.anyio
async def test_async_output_sink_write_failure_is_typed_and_preserves_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = tmp_path / "sink"

    def failing_write(output_sink: object, stream: str, content: bytes) -> None:
        raise OSError("injected sink failure")

    monkeypatch.setattr(execution_module._AsyncOutputSink, "write", failing_write)
    with pytest.raises(ProcessExecutionError) as failure:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "print('retained?', flush=True)",
                output_directory=sink,
                output_root=tmp_path,
            )
        )

    assert failure.value.code is ErrorCode.EXECUTION_FAILURE
    assert failure.value.output_sink is not None
    assert failure.value.output_sink.complete is False
    assert failure.value.details["output_sink"]["io_error"] is True


def test_observed_output_sink_is_rejected_before_launch(tmp_path: Path) -> None:
    sink = tmp_path / "sink"
    with pytest.raises(ValueError, match="unsupported"):
        request(
            tmp_path,
            "-c",
            "import sys; sys.stdout.buffer.write(b'o' * 1200); sys.stderr.buffer.write(b'e' * 700)",
            max_output_bytes=4_000,
            output_directory=sink,
            output_root=tmp_path,
            observation="child_peak_rss",
        )


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
    assert not process_is_alive(process_ids[0])


@pytest.mark.anyio
async def test_level_cancellation_waits_for_broker_cleanup(tmp_path: Path) -> None:
    process_ids: list[int] = []
    cleanup: list[bool] = []
    scope = anyio.CancelScope()

    async def started(pid: int) -> None:
        process_ids.append(pid)
        scope.cancel()

    async def cleaned(complete: bool) -> None:
        await anyio.sleep(0)
        cleanup.append(complete)

    try:
        with scope:
            await SubprocessBroker().run(
                request(tmp_path, "-c", "import time; time.sleep(10)"),
                on_started=started,
                on_cleanup=cleaned,
            )
        assert cleanup == [True]
        assert not process_is_alive(process_ids[0])
    finally:
        for pid in process_ids:
            if process_is_alive(pid):
                os.kill(pid, signal.SIGKILL)
        await asyncio.sleep(0.1)


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
@pytest.mark.parametrize("exit_before_observation", [False, True])
async def test_writable_growth_baseline_precedes_subprocess_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_before_observation: bool
) -> None:
    output = tmp_path / "early-output"
    output.mkdir()
    (output / "existing.bin").write_bytes(b"x" * 200)
    completed_write = tmp_path / "write-complete"
    original = asyncio.create_subprocess_exec

    async def return_after_write(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await original(*args, **kwargs)
        while not completed_write.exists():
            await anyio.sleep(0.01)
        if exit_before_observation:
            await process.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", return_after_write)
    with pytest.raises(ProcessExecutionError) as error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import pathlib,time; "
                "pathlib.Path('early-output/new.bin').write_bytes(b'x' * 1024); "
                "pathlib.Path('write-complete').touch(); "
                + ("pass" if exit_before_observation else "time.sleep(30)"),
                timeout_seconds=2,
                resource_policy=ResourcePolicy(
                    filesystem_path=tmp_path,
                    writable_roots=(output,),
                    minimum_free_bytes=0,
                    maximum_writable_growth_bytes=64,
                    sampling_interval_ms=25,
                ),
            )
        )
    assert error.value.code is ErrorCode.LIMIT_EXCEEDED
    assert error.value.process.cleanup_complete is True
    assert error.value.process.resources is not None
    assert error.value.process.resources.writable_root_growth_bytes == {str(output): 1024}


@pytest.mark.anyio
async def test_final_resource_sample_retains_observed_rss_availability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sampled = anyio.Event()
    original = shutil.disk_usage

    def sample_disk(path: Any) -> Any:
        result = original(path)
        sampled.set()
        return result

    async def release_after_sample(_pid: int) -> None:
        # The resource observer samples RSS before yielding after disk_usage.
        # Keep the child alive until that first observation has happened.
        await sampled.wait()
        (tmp_path / "finish").touch()

    monkeypatch.setattr(shutil, "disk_usage", sample_disk)
    outcome = await SubprocessBroker().run(
        request(
            tmp_path,
            "-c",
            "import pathlib,time\nwhile not pathlib.Path('finish').exists(): time.sleep(0.01)\n",
            resource_policy=ResourcePolicy(
                filesystem_path=tmp_path,
                minimum_free_bytes=0,
                sampling_interval_ms=25,
            ),
        ),
        on_started=release_after_sample,
    )
    assert outcome.process.resources is not None
    assert outcome.process.resources.peak_rss_bytes is not None
    assert outcome.process.resources.peak_rss_bytes > 0
    assert "peak_rss_bytes" not in outcome.process.resources.unavailable_metrics


@pytest.mark.anyio
async def test_writable_growth_is_checked_after_exit_between_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "late-output"
    output.mkdir()
    sampled = anyio.Event()
    original = shutil.disk_usage

    def sample_disk(path: Any) -> Any:
        result = original(path)
        sampled.set()
        return result

    async def release_writer(_pid: int) -> None:
        await sampled.wait()
        (tmp_path / "write-now").touch()

    monkeypatch.setattr(shutil, "disk_usage", sample_disk)
    with pytest.raises(ProcessExecutionError) as error:
        await SubprocessBroker().run(
            request(
                tmp_path,
                "-c",
                "import pathlib,time\n"
                "while not pathlib.Path('write-now').exists(): time.sleep(0.01)\n"
                "pathlib.Path('late-output/new.bin').write_bytes(b'x' * 1024)\n",
                timeout_seconds=1,
                resource_policy=ResourcePolicy(
                    filesystem_path=tmp_path,
                    writable_roots=(output,),
                    minimum_free_bytes=0,
                    maximum_writable_growth_bytes=64,
                    sampling_interval_ms=3000,
                ),
            ),
            on_started=release_writer,
        )
    assert error.value.code is ErrorCode.LIMIT_EXCEEDED
    assert error.value.process.resources is not None
    assert error.value.process.resources.writable_root_growth_bytes == {str(output): 1024}
    assert error.value.process.cleanup_complete is True


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
