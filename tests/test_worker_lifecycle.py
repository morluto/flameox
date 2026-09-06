from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Annotated

import anyio
import psutil
import pytest
from pydantic import Field, TypeAdapter, ValidationError

from flameox.workers.harness import IsolatedWorkerHarness, WorkerRuntimeConfig
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId


def worker_definition() -> WorkerDefinition[int, int]:
    return WorkerDefinition(
        operation=WorkerOperationId.PSTATS_PARSE,
        module="lifecycle_worker",
        request=TypeAdapter(Annotated[int, Field(gt=0)]),
        response=TypeAdapter(int),
        name="lifecycle",
        implementation="test",
        timeout_seconds=10,
    )


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["sync", "async", "session"])
def test_invalid_worker_request_releases_job_directory(tmp_path: Path, mode: str) -> None:
    harness = IsolatedWorkerHarness(WorkerRuntimeConfig(tmp_path, tmp_path, tmp_path))

    async def exercise() -> None:
        if mode == "session":
            await harness.run_typed_session(worker_definition(), 0, consume=lambda value, _: value)
        else:
            await harness.run_typed(worker_definition(), 0)

    with pytest.raises(ValidationError):
        if mode == "sync":
            harness.run_typed_sync(worker_definition(), 0)
        else:
            anyio.run(exercise)
    assert not list((tmp_path / "artifact-workers").iterdir())


@pytest.mark.process
@pytest.mark.serial
@pytest.mark.parametrize("cancelled", [False, True])
def test_worker_session_settles_child_before_releasing_staging(
    tmp_path: Path, cancelled: bool
) -> None:
    pid_path = tmp_path / "child.pid"
    (tmp_path / "lifecycle_worker.py").write_text(
        "import os, pathlib, time\n"
        "pathlib.Path('child.pid').write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    harness = IsolatedWorkerHarness(WorkerRuntimeConfig(tmp_path, tmp_path, tmp_path))

    async def exercise() -> None:
        with anyio.CancelScope() as scope:

            async def heartbeat(root: Path) -> None:
                if pid_path.exists():
                    if cancelled:
                        scope.cancel()
                        await anyio.sleep(0)
                    raise RuntimeError("heartbeat failed")

            if cancelled:
                await harness.run_typed_session(
                    worker_definition(), 1, consume=lambda value, _: value, heartbeat=heartbeat
                )
            else:
                with pytest.raises(RuntimeError, match="heartbeat failed"):
                    await harness.run_typed_session(
                        worker_definition(),
                        1,
                        consume=lambda value, _: value,
                        heartbeat=heartbeat,
                    )
        try:
            pid = int(pid_path.read_text())
            assert (
                not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
            )
            assert not list((tmp_path / "artifact-workers").iterdir())
        finally:
            if pid_path.exists() and psutil.pid_exists(int(pid_path.read_text())):
                os.kill(int(pid_path.read_text()), signal.SIGKILL)
            await asyncio.sleep(0.1)

    anyio.run(exercise)
