from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import anyio
import psutil
import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    CaptureTarget,
    ExperimentCase,
    ExperimentDesign,
    RequestLimits,
    WorkloadBudget,
)


def _target(
    tmp_path: Path,
    code: str,
    *,
    budget: WorkloadBudget | None = None,
    provider_id: str = "direct",
) -> CaptureTarget:
    options: dict[str, Any] = {}
    if budget is not None:
        options["budget"] = budget
    return CaptureTarget(
        argv=[sys.executable, "-c", code],
        cwd=str(tmp_path),
        provider_id=provider_id,
        **options,
    )


@pytest.mark.process
def test_decoder_limits_do_not_bound_an_unbudgeted_workload(tmp_path: Path) -> None:
    async def exercise() -> Any:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        try:
            result = await runtime.capture_and_analyze(
                _target(
                    tmp_path,
                    "import time; data = bytearray(64 * 1024 * 1024); print('ready'); "
                    "time.sleep(0.1)",
                ),
                "artifact.preview",
                limits=RequestLimits(timeout_seconds=0.01, max_memory_bytes=16 * 1024**2),
            )
            return result["capture"]["executions"][0]
        finally:
            runtime.close()

    execution = anyio.run(exercise)
    assert execution["status"] == "succeeded"
    assert execution["failure_code"] is None


@pytest.mark.process
def test_explicit_workload_timeout_is_attributed_and_reopened(tmp_path: Path) -> None:
    async def exercise() -> tuple[str, Any]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        try:
            result = await runtime.capture_and_analyze(
                _target(
                    tmp_path,
                    "import time; print('before-timeout', flush=True); time.sleep(2)",
                    budget=WorkloadBudget(timeout_seconds=0.1),
                ),
                "artifact.preview",
                limits=RequestLimits(timeout_seconds=10, max_memory_bytes=16 * 1024**2),
                preserve=True,
            )
            execution = result["capture"]["executions"][0]
            return str(result["preserved"]["evidence_id"]), execution
        finally:
            runtime.close()

    evidence_id, execution = anyio.run(exercise)
    assert execution["status"] == "failed"
    assert execution["failure_code"] == "EXECUTION_TIMEOUT"
    assert execution["limit"]["kind"] == "timeout"
    assert execution["limit"]["configured"] == 0.1
    assert execution["limit"]["unit"] == "seconds"

    reopened = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    try:
        manifest = reopened.read_evidence(evidence_id)
        projection = reopened.read_evidence_agent_projection(evidence_id)
    finally:
        reopened.close()
    assert manifest["body"]["capture_request"]["target"]["budget"] == {
        "timeout_seconds": 0.1,
        "max_memory_bytes": None,
    }
    stored = manifest["body"]["capture_request"]["executions"][0]
    assert stored["limit"]["configured"] == 0.1
    assert projection["body"]["capture_request"]["target"]["budget"] == {
        "timeout_seconds": 0.1,
        "max_memory_bytes": None,
    }


@pytest.mark.process
def test_explicit_workload_memory_budget_is_attributed(tmp_path: Path) -> None:
    async def exercise() -> Any:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        try:
            result = await runtime.capture_and_analyze(
                _target(
                    tmp_path,
                    "data = bytearray(64 * 1024 * 1024); print('allocated'); "
                    "import time; time.sleep(2)",
                    budget=WorkloadBudget(max_memory_bytes=32 * 1024**2),
                ),
                "artifact.preview",
                limits=RequestLimits(timeout_seconds=10, max_memory_bytes=16 * 1024**2),
            )
            return result["capture"]["executions"][0]
        finally:
            runtime.close()

    execution = anyio.run(exercise)
    assert execution["status"] == "failed"
    assert execution["failure_code"] == "LIMIT_EXCEEDED"
    assert execution["limit"]["kind"] == "memory_limit_exceeded"
    assert execution["limit"]["configured"] == 32 * 1024**2
    assert execution["limit"]["unit"] == "bytes"


@pytest.mark.process
def test_workload_budget_also_bounds_semantic_oracle(tmp_path: Path) -> None:
    async def exercise() -> Any:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        try:
            result = await runtime.capture_and_analyze(
                _target(tmp_path, "print('workload')", budget=WorkloadBudget(timeout_seconds=0.2)),
                "artifact.preview",
                mode="experiment",
                experiment=ExperimentDesign(
                    cases=[ExperimentCase(name="baseline"), ExperimentCase(name="candidate")],
                    blocks=1,
                    seed=1,
                    metric="wall_time_ns",
                    estimand="median_difference",
                    practical_threshold=0,
                    semantic_oracle=[
                        sys.executable,
                        "-c",
                        "import time; time.sleep(2)",
                    ],
                ),
            )
            return result["capture"]["executions"][0]
        finally:
            runtime.close()

    execution = anyio.run(exercise)
    assert execution["status"] == "failed"
    assert execution["failure_code"] == "SEMANTIC_ORACLE_FAILED"
    oracle = execution["semantic_oracle"]
    assert oracle["status"] == "failed"
    assert oracle["failure_code"] == "EXECUTION_TIMEOUT"
    assert oracle["limit"]["kind"] == "timeout"
    assert oracle["limit"]["configured"] == 0.2
    assert oracle["limit"]["unit"] == "seconds"


@pytest.mark.process
@pytest.mark.requires_memray
def test_unbudgeted_memray_capture_keeps_native_artifact_when_worker_times_out(
    tmp_path: Path,
) -> None:
    pytest.importorskip("memray")

    async def exercise() -> tuple[dict[str, Any], dict[str, Any]]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        try:
            result = await runtime.capture_and_analyze(
                _target(
                    tmp_path,
                    "retained = [bytearray(1024) for _ in range(64)]; print('captured')",
                    provider_id="memray",
                ),
                "memory.hotspots",
                limits=RequestLimits(timeout_seconds=0.01, max_memory_bytes=16 * 1024**2),
                preserve=True,
            )
            return result, runtime.read_evidence(result["preserved"]["evidence_id"])
        finally:
            runtime.close()

    result, manifest = anyio.run(exercise)
    execution = result["capture"]["executions"][0]
    assert execution["status"] == "succeeded"
    assert result["analysis_failure"] is not None
    assert result["analysis_failure"]["code"] == "EXECUTION_FAILURE"
    assert "0.01 seconds" in result["analysis_failure"]["message"]
    assert result["preserved"]["evidence_id"]
    roles = {item["role"] for item in manifest["body"]["artifacts"]}
    assert "capture-0001/memory" in roles


@pytest.mark.process
def test_unbudgeted_capture_cancellation_cleans_child_and_scratch(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "child.pid"

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        task = asyncio.create_task(
            runtime.capture_and_analyze(
                _target(
                    tmp_path,
                    "import pathlib, subprocess, sys, time; "
                    "child=subprocess.Popen([sys.executable, '-c', "
                    "'import time; time.sleep(30)']); "
                    f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid)); "
                    "time.sleep(30)",
                ),
                "artifact.preview",
            )
        )
        try:
            for _ in range(200):
                if child_pid_file.is_file():
                    break
                await asyncio.sleep(0.01)
            assert child_pid_file.is_file()
            child_pid = int(child_pid_file.read_text())
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            for _ in range(200):
                if not psutil.pid_exists(child_pid):
                    break
                try:
                    if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                        break
                except psutil.NoSuchProcess:
                    break
                await asyncio.sleep(0.01)
            assert not psutil.pid_exists(child_pid) or (
                psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
            )
            assert list(runtime.scratch.glob("capture-*")) == []
        finally:
            if not task.done():
                task.cancel()
            runtime.close()

    anyio.run(exercise)
