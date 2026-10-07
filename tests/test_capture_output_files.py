from __future__ import annotations

import sys
from pathlib import Path

import anyio
import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    CaptureTarget,
    ExperimentCase,
    ExperimentDesign,
    RequestLimits,
    WorkloadBudget,
)


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


def _preserved_data_files(runtime: AnalysisRuntime, evidence_id: str) -> dict[str, Path]:
    manifest = runtime.read_evidence(evidence_id)
    bundle = runtime.repository.root / "artifacts" / "sha256"
    return {
        artifact["role"]: bundle / artifact["sha256"][:2] / artifact["sha256"] / "payload"
        for artifact in manifest["body"]["artifacts"]
    }


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
                limits=RequestLimits(max_output_bytes=len(stdout) + len(stderr) + 1),
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
        assert stdout in [path.read_bytes() for path in files.values()]
        assert stderr in [path.read_bytes() for path in files.values()]
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
        contents = [path.read_bytes() for path in files.values()]
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
            retained = streams["stdout_bytes"]
            if failure == "output_limit":
                assert retained == len(prefix)
            else:
                # A deadline can interrupt collection partway through a write.
                # Verify the exact retained prefix after reopening, not bytes
                # the workload intended to emit before the deadline.
                assert 0 <= retained <= len(prefix)
            assert streams["stdout_complete"] is False
            assert streams["stderr_complete"] is False
            return result["preserved"]["evidence_id"], prefix[:retained]
        finally:
            runtime.close()

    evidence_id, prefix = anyio.run(exercise)
    reopened = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    try:
        files = _preserved_data_files(reopened, evidence_id)
        assert files["capture-0001/stdout"].read_bytes() == prefix
    finally:
        reopened.close()
