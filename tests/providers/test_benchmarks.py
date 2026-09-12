from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
import pyperf
import pytest

from flameox.providers.benchmark_scaling import scaling_projection
from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    CaptureTarget,
    EvidenceSource,
    PathSource,
    RequestLimits,
)


def _write_pyperf_suite(path: Path, values: list[float]) -> None:
    run = pyperf.Run(
        values,
        metadata={"name": "workload", "unit": "second", "loops": 1},
        collect_metadata=False,
    )
    pyperf.BenchmarkSuite([pyperf.Benchmark([run])]).dump(str(path), replace=True)


def _write_benchmark_samples(path: Path, values: list[int]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "flameox.benchmark-samples.v1",
                "producer": "example-benchmark",
                "producer_version": "1.0",
                "benchmarks": [
                    {
                        "name": "operation",
                        "unit": "ns",
                        "measurement_clock": "host_monotonic",
                        "synchronization": "not_required",
                        "samples": values,
                    }
                ],
            }
        )
    )


def _write_scaling_samples(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "flameox.benchmark-samples.v1",
                "producer": "example-benchmark",
                "producer_version": "1.0",
                "benchmarks": [
                    {
                        "name": "operation",
                        "unit": "ns",
                        "measurement_clock": "host_monotonic",
                        "synchronization": "not_required",
                        "dimensions": {"elements": str(elements)},
                        "samples": [duration, duration],
                    }
                    for elements, duration in ((10, 100), (20, 400), (40, 1_600))
                ],
            }
        )
    )


@pytest.mark.process
def test_pyperf_summary_uses_native_isolated_reader(tmp_path: Path) -> None:
    artifact = tmp_path / "benchmark.json"
    _write_pyperf_suite(artifact, [0.010, 0.012])
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "benchmark.summary",
            [PathSource(path=str(artifact), format="pyperf")],
            {},
        )
    finally:
        runtime.close()

    assert result["provider"]["id"] == "pyperf"
    assert result["blocks"][0]["values"]["measurement_count"] == 2
    assert [row["value_int"] for row in result["blocks"][1]["rows"]] == [
        10_000_000,
        12_000_000,
    ]
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.process
def test_pyperf_compare_reads_explicit_artifacts_directly(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_pyperf_suite(baseline, [0.010, 0.012])
    _write_pyperf_suite(candidate, [0.005, 0.006])
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "benchmark.compare",
            [
                PathSource(path=str(baseline), format="pyperf"),
                PathSource(path=str(candidate), format="pyperf"),
            ],
            {"metric": "workload", "baseline_index": 0},
        )
        assert not (tmp_path / ".flameox").exists()
        preserved = runtime.preserve_evidence(result["analysis_id"])
        reanalyzed = runtime.analyze(
            "benchmark.compare",
            [
                EvidenceSource(
                    kind="evidence",
                    evidence_id=preserved["evidence_id"],
                    artifact_role="input-0001",
                ),
                EvidenceSource(
                    kind="evidence",
                    evidence_id=preserved["evidence_id"],
                    artifact_role="input-0002",
                ),
            ],
            {"metric": "workload", "baseline_index": 0},
        )
    finally:
        runtime.close()

    row = result["blocks"][1]["rows"][0]
    assert row["benchmark"] == "workload"
    assert row["baseline_mean"] == 11_000_000
    assert row["candidate_mean"] == 5_500_000
    assert row["ratio"] == 0.5
    assert reanalyzed["blocks"][1]["rows"][0]["ratio"] == 0.5
    assert (tmp_path / ".flameox" / "repository.json").is_file()


@pytest.mark.process
def test_pyperf_compare_aggregates_beyond_the_sample_row_ceiling(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_pyperf_suite(baseline, [0.010] * 1_002)
    _write_pyperf_suite(candidate, [0.005] * 1_002)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "benchmark.compare",
            [
                PathSource(path=str(baseline), format="pyperf"),
                PathSource(path=str(candidate), format="pyperf"),
            ],
            {"metric": "workload"},
        )
    finally:
        runtime.close()

    assert result["blocks"][1]["rows"][0]["ratio"] == 0.5


@pytest.mark.process
def test_structured_benchmark_samples_are_isolated_and_comparable(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.samples.json"
    candidate = tmp_path / "candidate.samples.json"
    _write_benchmark_samples(baseline, [10, 12])
    _write_benchmark_samples(candidate, [5, 6])
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        summary = runtime.analyze(
            "benchmark.summary",
            [PathSource(path=str(baseline), format="samples")],
            {},
        )
        comparison = runtime.analyze(
            "benchmark.compare",
            [
                PathSource(path=str(baseline), format="samples"),
                PathSource(path=str(candidate), format="samples"),
            ],
            {"metric": "operation"},
        )
    finally:
        runtime.close()

    assert summary["provider"]["id"] == "benchmark-samples"
    assert summary["blocks"][0]["values"]["measurement_count"] == 2
    assert comparison["blocks"][1]["rows"][0]["ratio"] == 0.5
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.process
def test_benchmark_compare_retains_series_dimensions_and_timing_protocol(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.samples.json"
    candidate = tmp_path / "candidate.samples.json"

    def write(path: Path, small: int, large: int, *, clock: str = "host_monotonic") -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "flameox.benchmark-samples.v1",
                    "producer": "example-benchmark",
                    "benchmarks": [
                        {
                            "name": "operation",
                            "unit": "ns",
                            "measurement_clock": clock,
                            "synchronization": "not_required",
                            "dimensions": {"size": size},
                            "samples": [value],
                        }
                        for size, value in (("small", small), ("large", large))
                    ],
                }
            )
        )

    write(baseline, 1, 100)
    write(candidate, 2, 50)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        comparison = runtime.analyze(
            "benchmark.compare",
            [
                PathSource(path=str(baseline), format="samples"),
                PathSource(path=str(candidate), format="samples"),
            ],
            {"metric": "operation"},
        )
        write(candidate, 2, 50, clock="cuda_event")
        incompatible = runtime.analyze(
            "benchmark.compare",
            [
                PathSource(path=str(baseline), format="samples"),
                PathSource(path=str(candidate), format="samples"),
            ],
            {"metric": "operation"},
        )
    finally:
        runtime.close()

    rows = comparison["blocks"][1]["rows"]
    assert {row["dimensions"]["size"]: row["ratio"] for row in rows} == {
        "small": 2.0,
        "large": 0.5,
    }
    assert all(row["dimensions"]["measurement_clock"] == "host_monotonic" for row in rows)
    assert incompatible["blocks"][1]["rows"] == []
    assert incompatible["blocks"][0]["values"]["unmatched_identity_count"] == 4


@pytest.mark.process
def test_benchmark_scaling_estimates_power_law_from_declared_numeric_dimension(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "scaling.samples.json"
    _write_scaling_samples(artifact)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "benchmark.scaling",
            [PathSource(path=str(artifact), format="samples")],
            {"input_dimension": "elements", "metric": "operation"},
        )
    finally:
        runtime.close()

    metrics = result["blocks"][0]["values"]
    assert metrics["scaling_status"] == "estimated"
    assert metrics["input_dimension"] == "elements"
    row = result["blocks"][1]["rows"][0]
    assert row["benchmark"] == "operation"
    assert row["point_count"] == 3
    assert row["exponent"] == pytest.approx(2.0)
    assert row["r_squared"] == pytest.approx(1.0)


@pytest.mark.process
def test_benchmark_compare_aggregates_beyond_the_sample_row_ceiling(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.samples.json"
    candidate = tmp_path / "candidate.samples.json"
    _write_benchmark_samples(baseline, [10] * 1_002)
    _write_benchmark_samples(candidate, [5] * 1_002)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "benchmark.compare",
            [
                PathSource(path=str(baseline), format="samples"),
                PathSource(path=str(candidate), format="samples"),
            ],
            {"metric": "operation"},
        )
    finally:
        runtime.close()

    row = result["blocks"][1]["rows"][0]
    assert row["baseline_mean"] == 10
    assert row["candidate_mean"] == 5
    assert row["ratio"] == 0.5


@pytest.mark.process
def test_benchmark_scaling_aggregates_beyond_the_sample_row_ceiling(tmp_path: Path) -> None:
    artifact = tmp_path / "scaling.samples.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "flameox.benchmark-samples.v1",
                "producer": "example-benchmark",
                "benchmarks": [
                    {
                        "name": "operation",
                        "unit": "ns",
                        "measurement_clock": "host_monotonic",
                        "synchronization": "not_required",
                        "dimensions": {"elements": str(elements)},
                        "samples": [duration] * 1_002,
                    }
                    for elements, duration in ((10, 100), (20, 400))
                ],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "benchmark.scaling",
            [PathSource(path=str(artifact), format="samples")],
            {"input_dimension": "elements", "metric": "operation"},
        )
    finally:
        runtime.close()

    row = result["blocks"][1]["rows"][0]
    assert row["point_count"] == 2
    assert row["exponent"] == pytest.approx(2.0)


@pytest.mark.process
def test_benchmark_scaling_excludes_nonpositive_samples_from_series_aggregates(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "scaling.samples.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "flameox.benchmark-samples.v1",
                "producer": "example-benchmark",
                "benchmarks": [
                    {
                        "name": "operation",
                        "unit": "ns",
                        "measurement_clock": "host_monotonic",
                        "synchronization": "not_required",
                        "dimensions": {"elements": str(elements)},
                        "samples": samples,
                    }
                    for elements, samples in ((10, [10, -10]), (20, [40, -1]))
                ],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "benchmark.scaling",
            [PathSource(path=str(artifact), format="samples")],
            {"input_dimension": "elements", "metric": "operation"},
        )
    finally:
        runtime.close()

    row = result["blocks"][1]["rows"][0]
    assert row["point_count"] == 2
    assert row["exponent"] == pytest.approx(2.0)


@pytest.mark.process
def test_benchmark_scaling_reports_inconclusive_without_declared_dimension_values(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "benchmark.json"
    _write_pyperf_suite(artifact, [0.01, 0.02])
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "benchmark.scaling",
            [PathSource(path=str(artifact), format="pyperf")],
            {"input_dimension": "elements"},
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"]["scaling_status"] == "inconclusive"
    assert result["blocks"][1]["rows"][0]["status"] == "inconclusive"
    assert result["blocks"][1]["rows"][0]["exponent"] is None
    assert any("numeric 'elements'" in item for item in result["limitations"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("inputs", "measurements", "reason"),
    [
        (
            (1_000_000_000_000_000, 1_000_000_000_000_001),
            (1_000.0, 2_000.0),
            "insufficient log-space input separation",
        ),
        ((1_024, 1_025), (2_000.0, 1_000.0), "finite numeric range"),
    ],
)
def test_benchmark_scaling_reports_numerical_limits_without_crashing(
    inputs: tuple[int, int], measurements: tuple[float, float], reason: str
) -> None:
    result = scaling_projection(
        [
            {
                "benchmark": "operation",
                "unit": "ns",
                "dimensions": {"elements": str(input_value)},
                "value_float": measurement,
            }
            for input_value, measurement in zip(inputs, measurements, strict=True)
        ],
        {"input_dimension": "elements"},
        provider_id="test",
        provider_version="1",
        max_rows=10,
    )

    row = result.blocks[1]["rows"][0]
    assert row["status"] == "inconclusive"
    assert row["exponent"] is None
    assert reason in row["reason"]


@pytest.mark.process
def test_benchmark_scaling_keeps_non_axis_dimensions_as_distinct_series(tmp_path: Path) -> None:
    artifact = tmp_path / "variants.samples.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "flameox.benchmark-samples.v1",
                "producer": "example-benchmark",
                "producer_version": "1.0",
                "benchmarks": [
                    {
                        "name": "operation",
                        "unit": "ns",
                        "measurement_clock": "host_monotonic",
                        "synchronization": "not_required",
                        "dimensions": {"elements": str(size), "dtype": dtype},
                        "samples": [duration],
                    }
                    for dtype, size, duration in (
                        ("float32", 10, 100),
                        ("float32", 20, 400),
                        ("float64", 10, 1_000),
                        ("float64", 20, 2_000),
                    )
                ],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "benchmark.scaling",
            [PathSource(path=str(artifact), format="samples")],
            {"input_dimension": "elements"},
        )
    finally:
        runtime.close()

    rows = result["blocks"][1]["rows"]
    assert {row["dimensions"]["dtype"] for row in rows} == {"float32", "float64"}
    assert sorted(row["exponent"] for row in rows) == pytest.approx([1.0, 2.0])


@pytest.mark.process
def test_pyperf_capture_binds_native_output_before_analysis(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(
            evidence_directory=tmp_path / ".flameox", limits=RequestLimits(timeout_seconds=20)
        )
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass"],
                    cwd=str(tmp_path),
                    provider_id="pyperf",
                    capture_arguments={
                        "processes": 1,
                        "values": 1,
                        "warmups": 0,
                        "loops": 1,
                        "min_time": 0.001,
                        "name": "startup",
                    },
                ),
                "benchmark.summary",
            )
            assert result["provider"]["id"] == "pyperf"
            assert result["blocks"][0]["values"]["measurement_count"] == 1
            assert result["inputs"][0]["format"] == "pyperf"
            assert result["capture"]["executions"][0]["capture_argv"][1:4] == [
                "-m",
                "pyperf",
                "command",
            ]
            preserved = runtime.preserve_evidence(result["analysis_id"])
            assert list(runtime.scratch.glob("capture-*")) == []
            manifest = runtime.read_evidence(preserved["evidence_id"])
            assert {item["role"] for item in manifest["body"]["artifacts"]} == {
                "capture-0001/benchmark",
            }
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_pyperf_capture_preserves_multiline_target_argv(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(
            evidence_directory=tmp_path / ".flameox", limits=RequestLimits(timeout_seconds=20)
        )
        try:
            code = "value = 1\nassert value == 1"
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", code],
                    cwd=str(tmp_path),
                    provider_id="pyperf",
                    capture_arguments={
                        "processes": 1,
                        "values": 1,
                        "warmups": 0,
                        "loops": 1,
                        "min_time": 0.001,
                        "name": "multiline",
                    },
                ),
                "benchmark.summary",
            )
            assert result["provider"]["id"] == "pyperf"
            assert result["capture"]["executions"][0]["argv"] == [sys.executable, "-c", code]
            assert result["capture"]["executions"][0]["status"] == "succeeded"
        finally:
            runtime.close()

    anyio.run(exercise)


def test_composed_evidence_namespaces_colliding_source_roles(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_pyperf_suite(first_path, [0.01])
    _write_pyperf_suite(second_path, [0.02])
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        evidence_ids = []
        for path in (first_path, second_path):
            analysis = runtime.analyze(
                "benchmark.summary", [PathSource(path=str(path), format="pyperf")], {}
            )
            evidence_ids.append(runtime.preserve_evidence(analysis["analysis_id"])["evidence_id"])
        composed = runtime.analyze(
            "benchmark.scaling",
            [EvidenceSource(kind="evidence", evidence_id=value) for value in evidence_ids],
            {"input_dimension": "elements"},
        )
        preserved = runtime.preserve_evidence(composed["analysis_id"])
        manifest = runtime.read_evidence(preserved["evidence_id"])
    finally:
        runtime.close()

    assert {item["role"] for item in manifest["body"]["artifacts"]} == {
        "source-0001/input",
        "source-0002/input",
    }
    assert [item["role"] for item in manifest["body"]["inputs"]] == ["input", "input"]


@pytest.mark.process
def test_failed_provider_capture_returns_preservable_diagnostics(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(
            evidence_directory=tmp_path / ".flameox", limits=RequestLimits(timeout_seconds=20)
        )
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[
                        sys.executable,
                        "-c",
                        "import sys; print('before-failure'); raise SystemExit(7)",
                    ],
                    cwd=str(tmp_path),
                    provider_id="pyperf",
                    capture_arguments={
                        "processes": 1,
                        "values": 1,
                        "warmups": 0,
                        "loops": 1,
                        "min_time": 0.001,
                    },
                ),
                "benchmark.summary",
            )
            assert result["capture"]["requested_capability_id"] == "benchmark.summary"
            assert result["capture"]["executions"][0]["status"] == "failed"
            assert result["capability_id"] == "benchmark.summary"
            assert result["analysis_failure"] is not None
            assert result["blocks"][1]["rows"] == []
            diagnostics = result["capture"]["executions"][0]["console_diagnostics"]
            assert diagnostics["stderr_observed_bytes"] > 0
            preserved = runtime.preserve_evidence(result["analysis_id"])
            manifest = runtime.read_evidence(preserved["evidence_id"])
            assert manifest["body"]["artifacts"] == []
            stored = manifest["body"]["capture_request"]["executions"][0]
            assert stored["console_diagnostics"] == diagnostics
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_unpreserved_capture_analysis_failure_can_be_preserved_later(tmp_path: Path) -> None:
    code = "import os,pathlib; pathlib.Path(os.environ['FLAMEOX_BENCHMARK_OUTPUT']).write_text('{')"

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", code],
                    cwd=str(tmp_path),
                    provider_id="benchmark-samples",
                ),
                "benchmark.summary",
                preserve=False,
            )
            assert result["analysis_failure"]["code"] == "DECODE_FAILURE"
            assert result["capture"]["outcome"]["status"] == "succeeded"
            assert list(runtime.scratch.glob("capture-*"))
            assert not (tmp_path / ".flameox").exists()
            preserved = runtime.preserve_evidence(result["analysis_id"])
            manifest = runtime.read_evidence(preserved["evidence_id"])
            assert manifest["body"]["analysis_request"]["failure"]["code"] == "DECODE_FAILURE"
            assert list(runtime.scratch.glob("capture-*")) == []
        finally:
            runtime.close()

    anyio.run(exercise)
