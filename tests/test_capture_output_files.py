from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import anyio
import pytest

from flameox.runtime_contracts import (
    CaptureTarget,
    ExperimentCase,
    ExperimentDesign,
    RequestLimits,
    WorkloadBudget,
)
from flameox.stateless import AnalysisRuntime


def _experiment(oracle: list[str]) -> ExperimentDesign:
    return ExperimentDesign(
        cases=[ExperimentCase(name="baseline"), ExperimentCase(name="candidate")],
        blocks=1,
        seed=7,
        metric="wall_time_ns",
        estimand="median_difference",
        practical_threshold=0,
        semantic_oracle=oracle,
    )


def _preserved_data_files(runtime: AnalysisRuntime, evidence_id: str) -> list[Path]:
    del evidence_id
    bundle = runtime.repository.root / "artifacts" / "sha256"
    return [path for path in bundle.rglob("payload") if path.is_file()]


@pytest.mark.process
def test_capture_streams_preserve_exact_large_native_bytes_across_restart(tmp_path: Path) -> None:
    stdout = b"out-" * 40_000
    stderr = b"err-" * 40_000

    async def exercise() -> tuple[str, dict[str, object]]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[
                        sys.executable,
                        "-c",
                        "import os; os.write(1, b'out-' * 40000); os.write(2, b'err-' * 40000)",
                    ],
                    cwd=str(tmp_path),
                    provider_id="direct",
                ),
                "artifact.preview",
                limits=RequestLimits(max_output_bytes=stdout.__len__() + stderr.__len__() + 1),
                preserve=True,
            )
            execution = result["capture"]["executions"][0]
            assert execution["status"] == "succeeded"
            assert execution["output_streams"] == {
                "stdout_bytes": len(stdout),
                "stderr_bytes": len(stderr),
                "stdout_complete": True,
                "stderr_complete": True,
                "io_error": False,
            }
            return result["preserved"]["evidence_id"], execution
        finally:
            runtime.close()

    evidence_id, _execution = anyio.run(exercise)
    reopened = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    try:
        manifest = reopened.read_evidence(evidence_id)
        assert manifest["body"]["capture_request"]["executions"][0]["output_streams"][
            "stdout_bytes"
        ] == len(stdout)
        files = _preserved_data_files(reopened, evidence_id)
        assert stdout in [path.read_bytes() for path in files]
        assert stderr in [path.read_bytes() for path in files]
    finally:
        reopened.close()


@pytest.mark.process
def test_semantic_oracle_reads_full_capture_and_preserves_its_large_logs(tmp_path: Path) -> None:
    oracle_stdout = b"oracle-out-" * 8_000
    oracle_stderr = b"oracle-err-" * 8_000
    oracle = [
        sys.executable,
        "-c",
        (
            "import os, pathlib, sys; "
            "assert pathlib.Path(os.environ['FLAMEOX_CAPTURE_STDOUT']).read_bytes() "
            "== b'out-' * 40000; "
            "assert pathlib.Path(os.environ['FLAMEOX_CAPTURE_STDERR']).read_bytes() "
            "== b'err-' * 40000; "
            "os.write(1, b'oracle-out-' * 8000); os.write(2, b'oracle-err-' * 8000)"
        ),
    ]

    async def exercise() -> str:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[
                        sys.executable,
                        "-c",
                        "import os; os.write(1, b'out-' * 40000); os.write(2, b'err-' * 40000)",
                    ],
                    cwd=str(tmp_path),
                    provider_id="direct",
                    console_output="full",
                ),
                "artifact.preview",
                experiment=_experiment(oracle),
                mode="experiment",
                limits=RequestLimits(max_output_bytes=1_000_000),
                preserve=True,
            )
            execution = result["capture"]["executions"][0]
            assert execution["status"] == "succeeded"
            assert execution["semantic_oracle"]["status"] == "passed"
            assert execution["semantic_oracle"]["output_streams"] == {
                "stdout_bytes": len(oracle_stdout),
                "stderr_bytes": len(oracle_stderr),
                "stdout_complete": True,
                "stderr_complete": True,
                "io_error": False,
            }
            return str(result["preserved"]["evidence_id"])
        finally:
            runtime.close()

    evidence_id = anyio.run(exercise)
    reopened = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    try:
        files = _preserved_data_files(reopened, evidence_id)
        contents = [path.read_bytes() for path in files]
        assert oracle_stdout in contents
        assert oracle_stderr in contents
    finally:
        reopened.close()


@pytest.mark.process
@pytest.mark.parametrize("failure", ["timeout", "output_limit"])
def test_capture_failure_preserves_stream_prefix_and_marks_sink_incomplete(
    tmp_path: Path, failure: str
) -> None:
    async def exercise() -> tuple[str, bytes]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        prefix = b"prefix-" * 20_000 if failure == "timeout" else b"x" * 100_000
        try:
            code = (
                "import os, time; os.write(1, b'prefix-' * 20000); time.sleep(5)"
                if failure == "timeout"
                else "import os; os.write(1, b'x' * 200000)"
            )
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", code],
                    cwd=str(tmp_path),
                    provider_id="direct",
                    budget=WorkloadBudget(timeout_seconds=0.2 if failure == "timeout" else 10),
                ),
                "artifact.preview",
                limits=RequestLimits(
                    max_output_bytes=len(prefix) if failure == "output_limit" else 1_000_000,
                ),
                preserve=True,
            )
            execution = result["capture"]["executions"][0]
            assert execution["status"] == "failed"
            assert execution["failure_code"] == (
                "EXECUTION_TIMEOUT" if failure == "timeout" else "LIMIT_EXCEEDED"
            )
            streams = execution["output_streams"]
            assert streams["stdout_bytes"] == len(prefix)
            assert streams["stdout_complete"] is False
            assert streams["stderr_complete"] is False
            return result["preserved"]["evidence_id"], prefix
        finally:
            runtime.close()

    evidence_id, prefix = anyio.run(exercise)
    reopened = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    try:
        files = _preserved_data_files(reopened, evidence_id)
        assert prefix in [path.read_bytes() for path in files]
    finally:
        reopened.close()


@pytest.mark.process
def test_cancelled_capture_removes_request_scratch(tmp_path: Path) -> None:
    started = tmp_path / "started"

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        task = asyncio.create_task(
            runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; import time; "
                        f"Path({str(started)!r}).touch(); time.sleep(30)",
                    ],
                    cwd=str(tmp_path),
                    provider_id="direct",
                ),
                "artifact.preview",
            )
        )
        try:
            with anyio.fail_after(5):
                while not started.exists():
                    await anyio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not list(runtime.scratch.glob("capture-*"))
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            runtime.close()

    anyio.run(exercise)
