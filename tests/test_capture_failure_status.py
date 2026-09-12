from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import anyio
import psutil
import pytest

from flameox.execution import ExecutionOutcome, ExecutionRequest, SubprocessBroker
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    CaptureTarget,
    ExperimentCase,
    ExperimentDesign,
    RequestLimits,
)


class ExitedBeforeCollectionBroker(SubprocessBroker):
    """Exercise the valid schedule where a short producer exits before collection."""

    async def run(
        self,
        request: ExecutionRequest,
        *,
        on_started: Callable[[int], Awaitable[None]] | None = None,
        on_cleanup: Callable[[bool], Awaitable[None]] | None = None,
    ) -> ExecutionOutcome:
        async def wait_for_exit(pid: int) -> None:
            if on_started is not None:
                await on_started(pid)
            with anyio.fail_after(5):
                while psutil.pid_exists(pid):
                    try:
                        if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                            break
                    except psutil.NoSuchProcess:
                        break
                    await anyio.sleep(0.01)
            await anyio.sleep(0)

        return await super().run(request, on_started=wait_for_exit, on_cleanup=on_cleanup)


@pytest.mark.process
@pytest.mark.parametrize("noisy_oracle", [False, True])
def test_zero_exit_output_limit_is_preserved_as_failed_capture(
    tmp_path: Path, noisy_oracle: bool
) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        runtime.broker = ExitedBeforeCollectionBroker()
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[
                        sys.executable,
                        "-c",
                        "print('ok')" if noisy_oracle else "import os; os.write(1, b'x' * 2048)",
                    ],
                    cwd=str(tmp_path),
                    provider_id="direct",
                    console_output="full",
                ),
                "artifact.preview",
                limits=RequestLimits(max_output_bytes=1024),
                preserve=True,
                experiment=ExperimentDesign(
                    cases=[ExperimentCase(name="baseline"), ExperimentCase(name="candidate")],
                    blocks=1,
                    seed=7,
                    metric="wall_time_ns",
                    estimand="median_difference",
                    practical_threshold=0,
                    semantic_oracle=[sys.executable, "-c", "import os; os.write(1, b'x' * 2048)"],
                ),
            )
            for execution in result["capture"]["executions"]:
                assert execution["returncode"] == 0
                assert execution["status"] == "failed"
                if noisy_oracle:
                    assert execution["failure_code"] == "SEMANTIC_ORACLE_FAILED"
                    oracle = execution["semantic_oracle"]
                    assert oracle["returncode"] == 0
                    assert oracle["failure_code"] == "LIMIT_EXCEEDED"
                    assert oracle["status"] == "failed"
                else:
                    assert execution["failure_code"] == "LIMIT_EXCEEDED"
                    assert execution["semantic_oracle"] is None
            evidence_id = result["preserved"]["evidence_id"]
        finally:
            runtime.close()
        reopened = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        try:
            manifest = reopened.read_evidence(evidence_id)
            stored = manifest["body"]["capture_request"]["executions"]
            assert stored == result["capture"]["executions"]
        finally:
            reopened.close()

    anyio.run(exercise)
