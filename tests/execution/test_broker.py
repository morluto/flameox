from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from flameox.domain import DomainError, ErrorCode
from flameox.execution import ExecutionRequest, ResourcePolicy, SubprocessBroker


def request(tmp_path: Path, *arguments: str, **overrides: object) -> ExecutionRequest:
    values: dict[str, object] = {
        "argv": (sys.executable, *arguments),
        "cwd": tmp_path,
        "allowed_working_roots": (tmp_path,),
        "environment_allowlist": ("PATH",),
        "timeout_seconds": 5,
        "max_output_bytes": 1_000,
    }
    values.update(overrides)
    return ExecutionRequest.model_validate(values)


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
    assert outcome.process.exit_code == 0


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
                environment_overrides={"PYTHONPATH": "/tmp/unsafe"},
            )
        )
    assert error.value.code is ErrorCode.EXECUTION_REFUSED


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

    assert timeout_error.value.code is ErrorCode.PROCESS_TIMEOUT
    assert output_error.value.code is ErrorCode.QUERY_BUDGET_EXCEEDED


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

    assert error.value.code is ErrorCode.QUERY_BUDGET_EXCEEDED


@pytest.mark.anyio
async def test_cancellation_performs_cleanup_before_propagating(
    tmp_path: Path,
) -> None:
    task = asyncio.create_task(
        SubprocessBroker().run(request(tmp_path, "-c", "import time; time.sleep(10)"))
    )
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


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
    assert resources.writable_root_growth_bytes[str(output)] == 4096
    assert resources.policy_termination is None
    assert outcome.process.peak_rss_bytes == resources.peak_rss_bytes


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

    assert exceeded.value.code is ErrorCode.STORAGE_QUOTA_EXCEEDED
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
