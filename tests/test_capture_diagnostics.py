from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import CaptureTarget, ExperimentCase, ExperimentDesign, RequestLimits


def _workload(*, size: int) -> str:
    return (
        "import os\n"
        "def work():\n"
        "    return sum(range(20))\n"
        "work()\n"
        f"os.write(1, b'o' * {size}); os.write(2, b'e' * {size})"
    )


def _capture_roles(manifest: dict[str, object]) -> set[str]:
    body = manifest["body"]
    assert isinstance(body, dict)
    artifacts = body["artifacts"]
    assert isinstance(artifacts, list)
    return {item["role"] for item in artifacts if isinstance(item, dict)}


@pytest.mark.process
def test_capture_default_diagnostics_bound_noisy_native_capture_and_preservation(
    tmp_path: Path,
) -> None:
    size = 17 * 1024 * 1024
    workload = tmp_path / "workload.py"
    workload.write_text(_workload(size=size), encoding="utf-8")

    async def exercise() -> str:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, str(workload)],
                    cwd=str(tmp_path),
                    provider_id="coverage",
                ),
                "coverage.summary",
                limits=RequestLimits(max_output_bytes=1024 * 1024),
                preserve=True,
            )
            execution = result["capture"]["executions"][0]
            assert execution["status"] == "succeeded"
            diagnostics = execution["console_diagnostics"]
            assert diagnostics["stdout_observed_bytes"] == size
            assert diagnostics["stderr_observed_bytes"] == size
            assert (
                diagnostics["stdout_retained_bytes"] + diagnostics["stdout_omitted_bytes"] == size
            )
            assert (
                diagnostics["stderr_retained_bytes"] + diagnostics["stderr_omitted_bytes"] == size
            )
            assert diagnostics["stdout_omitted_bytes"] > 0
            assert diagnostics["stderr_omitted_bytes"] > 0
            assert diagnostics["stdout_complete"] is True
            assert diagnostics["stderr_complete"] is True
            assert "output_streams" not in execution
            preserved = result["preserved"]
            assert isinstance(preserved, dict)
            return str(preserved["evidence_id"])
        finally:
            runtime.close()

    evidence_id = anyio.run(exercise)
    reopened = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    try:
        manifest = reopened.read_evidence(evidence_id)
        roles = _capture_roles(manifest)
        assert "capture-0001/coverage" in roles
        assert "capture-0001/stdout" not in roles
        assert "capture-0001/stderr" not in roles
        projection = reopened.read_evidence_agent_projection(evidence_id)
        counts = projection["body"]["capture_request"]["executions"][0]["console_diagnostics"]
        assert counts["stdout_observed_bytes"] == size
        assert counts["stdout_omitted_bytes"] == size - counts["stdout_retained_bytes"]
        assert "stdout" not in counts
        assert "stderr" not in counts
    finally:
        reopened.close()


@pytest.mark.process
def test_native_capture_can_explicitly_preserve_full_console_output(tmp_path: Path) -> None:
    stdout, stderr = b"out-" * 8_000, b"err-" * 8_000
    workload = tmp_path / "workload.py"
    workload.write_text(
        "import os; os.write(1, b'out-' * 8000); os.write(2, b'err-' * 8000)",
        encoding="utf-8",
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    try:

        async def capture() -> dict[str, Any]:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, str(workload)],
                    cwd=str(tmp_path),
                    provider_id="coverage",
                    console_output="full",
                ),
                "coverage.summary",
                limits=RequestLimits(max_output_bytes=1_000_000),
                preserve=True,
            )

        result = anyio.run(capture)
        execution = result["capture"]["executions"][0]
        assert execution["output_streams"] == {
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_complete": True,
            "stderr_complete": True,
            "io_error": False,
        }
        payloads = [
            path.read_bytes()
            for path in (runtime.repository.root / "artifacts" / "sha256").rglob("payload")
        ]
        assert stdout in payloads
        assert stderr in payloads
    finally:
        runtime.close()


@pytest.mark.process
def test_capture_oracle_gets_full_workload_logs_but_keeps_its_own_diagnostics(
    tmp_path: Path,
) -> None:
    oracle_stdout = b"oracle-out-" * 2_000
    oracle_stderr = b"oracle-err-" * 2_000
    oracle = [
        sys.executable,
        "-c",
        (
            "import os, pathlib; "
            "assert pathlib.Path(os.environ['FLAMEOX_CAPTURE_STDOUT']).read_bytes() "
            "== b'workload-out-' * 2000; "
            "assert pathlib.Path(os.environ['FLAMEOX_CAPTURE_STDERR']).read_bytes() "
            "== b'workload-err-' * 2000; "
            "os.write(1, b'oracle-out-' * 2000); os.write(2, b'oracle-err-' * 2000)"
        ),
    ]
    experiment = ExperimentDesign(
        cases=[ExperimentCase(name="baseline"), ExperimentCase(name="candidate")],
        blocks=1,
        seed=3,
        metric="wall_time_ns",
        estimand="median_difference",
        practical_threshold=0,
        semantic_oracle=oracle,
    )

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[
                        sys.executable,
                        "-c",
                        "import os; os.write(1, b'workload-out-' * 2000); "
                        "os.write(2, b'workload-err-' * 2000)",
                    ],
                    cwd=str(tmp_path),
                    provider_id="direct",
                ),
                "artifact.preview",
                experiment=experiment,
                limits=RequestLimits(max_output_bytes=1_000_000),
                preserve=True,
            )
            execution = result["capture"]["executions"][0]
            assert execution["status"] == "succeeded"
            oracle_result = execution["semantic_oracle"]
            assert oracle_result["status"] == "passed"
            oracle_diagnostics = oracle_result["console_diagnostics"]
            assert oracle_diagnostics["stdout_observed_bytes"] == len(oracle_stdout)
            assert oracle_diagnostics["stderr_observed_bytes"] == len(oracle_stderr)
            assert oracle_diagnostics["stdout_retained_bytes"] + oracle_diagnostics[
                "stdout_omitted_bytes"
            ] == len(oracle_stdout)
            assert oracle_diagnostics["stderr_retained_bytes"] + oracle_diagnostics[
                "stderr_omitted_bytes"
            ] == len(oracle_stderr)
            assert oracle_diagnostics["stdout_omitted_bytes"] > 0
            assert oracle_diagnostics["stderr_omitted_bytes"] > 0
            assert oracle_diagnostics["stdout_complete"] is True
            assert oracle_diagnostics["stderr_complete"] is True
            return result
        finally:
            runtime.close()

    result = anyio.run(exercise)
    execution = result["capture"]["executions"][0]
    assert execution["output_streams"] is not None
    assert execution.get("console_diagnostics") is None
    preserved = result["preserved"]
    assert isinstance(preserved, dict)
    reopened = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    try:
        manifest = reopened.read_evidence(str(preserved["evidence_id"]))
        roles = _capture_roles(manifest)
        assert "capture-0001/oracle_stdout" not in roles
        assert "capture-0001/oracle_stderr" not in roles
    finally:
        reopened.close()


@pytest.mark.process
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("console_output", ["diagnostics", "full"])
@pytest.mark.parametrize(
    ("provider_id", "capability_id", "analysis_arguments", "missing_role"),
    [
        ("observations", "failures.summary", {}, "observations"),
        ("benchmark-samples", "benchmark.scaling", {"input_dimension": "size"}, "benchmark"),
    ],
)
def test_missing_native_artifact_keeps_capture_diagnostics_for_recovery(
    tmp_path: Path,
    preserve: bool,
    console_output: Literal["diagnostics", "full"],
    provider_id: str,
    capability_id: str,
    analysis_arguments: dict[str, Any],
    missing_role: str,
) -> None:
    workload = tmp_path / "missing_observations.py"
    workload.write_text(
        "import os, sys; os.write(1, b'failure-output'); "
        "os.write(2, b'failure-error'); sys.exit(7)",
        encoding="utf-8",
    )

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, str(workload)],
                    cwd=str(tmp_path),
                    provider_id=provider_id,
                    console_output=console_output,
                    analysis_arguments=analysis_arguments,
                ),
                capability_id,
                limits=RequestLimits(max_output_bytes=64 * 1024),
                preserve=preserve,
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    execution = result["capture"]["executions"][0]
    assert result["capability_id"] == capability_id
    assert execution["status"] == "failed"
    assert result["analysis_failure"]["code"] == "EXECUTION_FAILURE"
    assert result["analysis_failure"]["details"]["analysis_source_count"] == 0
    assert execution["returncode"] == 7
    assert execution["failure_code"] is None
    assert execution["missing_artifact_roles"] == [missing_role]
    if console_output == "diagnostics":
        diagnostics = execution["console_diagnostics"]
        assert diagnostics["stdout_observed_bytes"] == len(b"failure-output")
        assert diagnostics["stderr_observed_bytes"] == len(b"failure-error")
        assert diagnostics["stdout"] == "failure-output"
        assert diagnostics["stderr"] == "failure-error"
        assert result["inputs"] == []
    else:
        assert execution["output_streams"]["stdout_bytes"] == len(b"failure-output")
        assert execution["output_streams"]["stderr_bytes"] == len(b"failure-error")
        assert len(result["inputs"]) == 2
        assert {item["format"] for item in result["inputs"]} == {"text"}
    if preserve:
        preserved = result["preserved"]
        assert isinstance(preserved, dict)
        assert preserved["evidence_id"]
        reopened = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
        try:
            manifest = reopened.read_evidence(str(preserved["evidence_id"]))
            assert manifest["body"]["analysis_request"]["inputs"] == []
            assert manifest["body"]["source_layout"]["analysis_sources"] == []
            stored = manifest["body"]["capture_request"]["executions"][0]
            if console_output == "diagnostics":
                assert stored["console_diagnostics"]["stdout"] == "failure-output"
                assert stored["console_diagnostics"]["stderr"] == "failure-error"
            else:
                payloads = [
                    path.read_bytes()
                    for path in (reopened.repository.root / "artifacts" / "sha256").rglob("payload")
                ]
                assert sorted(payloads) == [b"failure-error", b"failure-output"]
        finally:
            reopened.close()
    else:
        assert "preserved" not in result
