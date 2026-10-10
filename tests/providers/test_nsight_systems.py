from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    PathSource,
    RequestLimits,
    RuntimeFailure,
)


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
    for table in (
        "CUDA_API_TRACE",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "CUPTI_ACTIVITY_KIND_DRIVER",
    ):
        pq.write_table(
            pa.table({"name": ["cudaGetDeviceCount"]}),
            parquetdir / f"{table}.parquet",
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


def test_nsight_native_cuda_tables_separate_host_operations_from_device_activity(
    tmp_path: Path,
) -> None:
    # Native table names and columns from Nsight Systems 2026.1 parquetdir exports.
    # Runtime/driver APIs can exist without a device launch; CUPTI is not a GPU-only family.
    parquetdir = tmp_path / "report.parquetdir"
    parquetdir.mkdir()
    for table in (
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "CUPTI_ACTIVITY_KIND_DRIVER",
        "CUPTI_ACTIVITY_KIND_OVERHEAD",
        "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
    ):
        pq.write_table(
            pa.table({"start": [1, 3], "end": [2, 4], "correlationId": [7, 8]}),
            parquetdir / f"{table}.parquet",
        )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    sources = [PathSource(path=str(parquetdir), format="nsys-parquet")]
    try:
        summary = runtime.analyze("trace.summary", sources, {})
        operations = runtime.analyze("trace.operations", sources, {})
        launches = runtime.analyze("gpu.launches", sources, {})
        bounded = runtime.analyze("gpu.launches", sources, {}, limits=RequestLimits(max_rows=1))
    finally:
        runtime.close()

    assert summary["coverage"]["rows_observed"] == 14
    assert {row["table"] for row in operations["blocks"][1]["rows"]} == {
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "CUPTI_ACTIVITY_KIND_DRIVER",
        "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    }
    assert {row["table"] for row in launches["blocks"][1]["rows"]} == {
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
    }
    assert launches["coverage"] == {
        "rows_returned": 6,
        "rows_observed": 6,
        "complete": True,
    }
    assert bounded["coverage"] == {
        "rows_returned": 1,
        "rows_observed": 6,
        "complete": False,
    }
    assert bounded["blocks"][1]["rows"][0]["table"] == "CUPTI_ACTIVITY_KIND_KERNEL"


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


@pytest.mark.golden
def test_nsight_parquetdir_is_analyzed_without_sqlite_or_repository(tmp_path: Path) -> None:
    export = tmp_path / "report.parquetdir"
    export.mkdir()
    pq.write_table(
        pa.table({"start_ns": [1, 2, 3], "kernel": ["a", "b", "c"]}),
        export / "CUDA_GPU_KERN_SUM.parquet",
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "gpu.launches",
            [PathSource(path=str(export), format="nsys-parquet")],
            {},
            limits=RequestLimits(max_rows=2),
        )
    finally:
        runtime.close()

    assert result["provider"]["id"] == "nsight-systems-parquetdir"
    assert result["coverage"] == {"rows_returned": 2, "rows_observed": 3, "complete": False}
    assert result["continuation"]
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.parametrize(
    "capability,table",
    [
        ("gpu.launches", "CUDA_GPU_KERN_SUM"),
        ("trace.summary", "CUDA_GPU_KERN_SUM"),
        ("trace.operations", "CUDA_API_TRACE"),
        ("trace.lifecycle", "PROCESS"),
    ],
)
def test_nsight_systems_rejects_malformed_selected_tables(
    tmp_path: Path, capability: str, table: str
) -> None:
    report = tmp_path / "report.parquetdir"
    report.mkdir()
    native = report / f"{table}.parquet"
    native.write_bytes(b"not parquet")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as caught:
            runtime.analyze(capability, [PathSource(path=str(report), format="nsys-parquet")], {})
        assert caught.value.code == "DECODE_FAILURE"
    finally:
        runtime.close()
    assert native.read_bytes() == b"not parquet"
