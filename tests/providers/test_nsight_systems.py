from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import anyio
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flameox.runtime_contracts import CaptureTarget, PathSource
from flameox.stateless import AnalysisRuntime


@pytest.mark.process
def test_nsight_systems_capture_preserves_native_report_and_exports_parquetdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / "kernels.parquet"
    pq.write_table(pa.table({"start_ns": [1], "kernel": ["captured"]}), template)
    calls = tmp_path / "calls.txt"
    executable = tmp_path / "bin" / "nsys"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, shutil, sys\n"
        "arguments = sys.argv[1:]\n"
        f"calls = pathlib.Path({str(calls)!r})\n"
        "with calls.open('a') as stream: stream.write(' '.join(arguments) + '\\n')\n"
        "if arguments[0] == 'profile':\n"
        "    stem = pathlib.Path(arguments[arguments.index('--output') + 1])\n"
        "    stem.with_suffix('.nsys-rep').write_bytes(b'native-report')\n"
        "else:\n"
        "    output = pathlib.Path(arguments[arguments.index('--output') + 1])\n"
        "    destination = output.with_suffix('.parquetdir')\n"
        "    destination.mkdir(parents=True, exist_ok=True)\n"
        f"    shutil.copyfile(pathlib.Path({str(template)!r}), "
        "destination / 'CUDA_GPU_KERN_SUM.parquet')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass"],
                    cwd=str(tmp_path),
                    provider_id="nsight-systems",
                    capture_arguments={"trace": ["cuda", "nvtx"]},
                ),
                "gpu.launches",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    assert result["provider"]["id"] == "nsight-systems-parquetdir"
    assert result["capture"]["executions"][0]["capture_argv"][0] == "nsys"
    assert result["blocks"][1]["rows"][0]["kernel"] == "captured"
    profile, export = calls.read_text().splitlines()
    assert "profile --trace=cuda,nvtx" in profile
    assert "--export=sqlite" not in profile
    assert "export --type parquetdir" in export


def test_nsight_systems_projects_native_uint64_identifiers_losslessly(tmp_path: Path) -> None:
    parquetdir = tmp_path / "report.parquetdir"
    parquetdir.mkdir()
    native_id = 18_302_628_885_633_695_744
    pq.write_table(
        pa.table({"correlationId": pa.array([native_id], type=pa.uint64())}),
        parquetdir / "CUDA_GPU_KERN_SUM.parquet",
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "gpu.launches",
            [PathSource(path=str(parquetdir), format="nsys-parquet")],
            {},
        )
        preserved = runtime.preserve_evidence(result["analysis_id"])
    finally:
        runtime.close()

    assert result["blocks"][1]["rows"][0]["correlationId"] == str(native_id)
    assert preserved["evidence_id"]


def test_nsight_systems_cuda_api_only_is_negative_accelerator_evidence(tmp_path: Path) -> None:
    parquetdir = tmp_path / "report.parquetdir"
    parquetdir.mkdir()
    pq.write_table(
        pa.table({"name": ["cudaGetDeviceCount"]}),
        parquetdir / "CUDA_API_TRACE.parquet",
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "gpu.launches",
            [PathSource(path=str(parquetdir), format="nsys-parquet")],
            {},
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"] == {
        "table_count": 0,
        "row_count": 0,
        "accelerator_activity_observed": False,
    }
    assert result["blocks"][1]["rows"] == []
    assert "no_accelerator_activity_observed" in result["limitations"]


def test_nsight_systems_trace_projections_select_semantic_table_families(
    tmp_path: Path,
) -> None:
    parquetdir = tmp_path / "report.parquetdir"
    parquetdir.mkdir()
    pq.write_table(
        pa.table({"name": ["cudaLaunchKernel"], "duration_ns": [10]}),
        parquetdir / "CUDA_API_TRACE.parquet",
    )
    pq.write_table(
        pa.table({"name": ["poll"], "duration_ns": [5]}),
        parquetdir / "OSRT_API.parquet",
    )
    pq.write_table(
        pa.table({"process_id": [7], "event": ["started"]}),
        parquetdir / "PROCESS_LIFECYCLE.parquet",
    )
    pq.write_table(
        pa.table({"kernel": ["ignored"]}),
        parquetdir / "CUDA_GPU_KERN_SUM.parquet",
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    source = [PathSource(path=str(parquetdir), format="nsys-parquet")]
    try:
        summary = runtime.analyze("trace.summary", source, {})
        operations = runtime.analyze("trace.operations", source, {})
        lifecycle = runtime.analyze("trace.lifecycle", source, {})
    finally:
        runtime.close()

    assert {row["table"] for row in summary["blocks"][1]["rows"]} == {
        "CUDA_API_TRACE",
        "CUDA_GPU_KERN_SUM",
        "OSRT_API",
        "PROCESS_LIFECYCLE",
    }
    assert {row["table"] for row in operations["blocks"][1]["rows"]} == {
        "CUDA_API_TRACE",
        "OSRT_API",
    }
    assert {row["table"] for row in lifecycle["blocks"][1]["rows"]} == {"PROCESS_LIFECYCLE"}


@pytest.mark.process
def test_cpu_only_nsight_capture_is_typed_negative_accelerator_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / "os-runtime.parquet"
    pq.write_table(pa.table({"start_ns": [1], "operation": ["poll"]}), template)
    executable = tmp_path / "bin" / "nsys"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, shutil, sys\n"
        "arguments = sys.argv[1:]\n"
        "if arguments[0] == 'profile':\n"
        "    stem = pathlib.Path(arguments[arguments.index('--output') + 1])\n"
        "    stem.with_suffix('.nsys-rep').write_bytes(b'cpu-only-native-report')\n"
        "else:\n"
        "    output = pathlib.Path(arguments[arguments.index('--output') + 1])\n"
        "    destination = output.with_suffix('.parquetdir')\n"
        "    destination.mkdir(parents=True, exist_ok=True)\n"
        f"    shutil.copyfile(pathlib.Path({str(template)!r}), "
        "destination / 'OS_RUNTIME_SUM.parquet')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])

    async def exercise() -> tuple[dict[str, Any], dict[str, Any]]:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass"],
                    cwd=str(tmp_path),
                    provider_id="nsight-systems",
                ),
                "gpu.launches",
                preserve=True,
            )
            manifest = runtime.read_evidence(result["preserved"]["evidence_id"])
            return result, manifest
        finally:
            runtime.close()

    result, manifest = anyio.run(exercise)
    assert result["coverage"] == {"rows_returned": 0, "rows_observed": 0, "complete": True}
    assert result["blocks"][0]["values"]["accelerator_activity_observed"] is False
    assert "no_accelerator_activity_observed" in result["limitations"]
    assert result["analysis_failure"] is None
    assert any(item["format"] == "nsys-rep" for item in manifest["body"]["artifacts"])
