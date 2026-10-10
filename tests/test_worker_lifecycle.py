from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Annotated

import anyio
import psutil
import pytest
from pydantic import Field, TypeAdapter

from flameox.execution import ProcessCancelledError
from flameox.runtime import AnalysisRuntime
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


@pytest.mark.process
@pytest.mark.serial
@pytest.mark.parametrize("cancelled", [False, True])
def test_worker_session_settles_child_before_releasing_staging(
    tmp_path: Path, cancelled: bool
) -> None:
    pid_path = tmp_path / "child.pid"
    (tmp_path / "lifecycle_worker.py").write_text(
        "import json, os, pathlib, sys, time\n"
        "pathlib.Path('child.pid').write_text(str(os.getpid()))\n"
        "request = json.loads(pathlib.Path(sys.argv[2]).read_text())\n"
        "if request['payload'] == 1: time.sleep(30)\n"
        "request['kind'] = 'success'\n"
        "pathlib.Path(sys.argv[4]).write_text(json.dumps(request))\n"
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    harness = IsolatedWorkerHarness(
        WorkerRuntimeConfig(tmp_path, tmp_path, tmp_path), broker=runtime.broker
    )
    receipts: list[ProcessCancelledError] = []

    def consume() -> None:
        try:
            with harness.run_typed_sync_session(worker_definition(), 1 if cancelled else 2) as (
                result,
                root,
            ):
                assert result == 2
                assert root.is_dir()
                raise RuntimeError("consumer failed")
        except ProcessCancelledError as error:
            receipts.append(error)
            raise

    async def exercise() -> None:
        try:
            with anyio.fail_after(10), anyio.CancelScope() as scope:

                async def cancel_started_child() -> None:
                    while not pid_path.exists():
                        await anyio.sleep(0.01)
                    scope.cancel()

                if cancelled:
                    async with anyio.create_task_group() as group:
                        group.start_soon(cancel_started_child)
                        await runtime.run_in_request(consume)
                else:
                    with pytest.raises(RuntimeError, match="consumer failed"):
                        await runtime.run_in_request(consume)
            if cancelled:
                assert len(receipts) == 1
                assert receipts[0].process.cleanup_complete
            pid = int(pid_path.read_text())
            assert (
                not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
            )
            assert not list((tmp_path / "artifact-workers").iterdir())
        finally:
            if pid_path.exists() and psutil.pid_exists(int(pid_path.read_text())):
                os.kill(int(pid_path.read_text()), signal.SIGKILL)
            runtime.close()

    anyio.run(exercise)
