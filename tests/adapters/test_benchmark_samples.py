from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters import BenchmarkSamplesExtractor
from flameox.application import EvidenceQueryService, ImportArtifactRequest, ImportService
from flameox.domain import ArtifactKind, DomainError, ErrorCode
from flameox.storage import Workspace


def _write_samples(path: Path, *, synchronization: str = "event_synchronize") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "flameox.benchmark-samples.v1",
                "producer": "decode-benchmark",
                "producer_version": "git:abc123",
                "benchmarks": [
                    {
                        "name": "decode.token_latency",
                        "unit": "ns",
                        "measurement_clock": "cuda_event",
                        "synchronization": synchronization,
                        "scope": "workload",
                        "phase": "steady_state",
                        "loop_count": 100,
                        "device": {"type": "cuda", "index": 0, "stream": "7"},
                        "dimensions": {
                            "batch": "1",
                            "dtype": "bfloat16",
                            "mode": "cuda_graph",
                        },
                        "warmups": [45_000],
                        "samples": [42_100, 41_900, 42_400],
                    }
                ],
            }
        )
    )


def _import(workspace: Workspace, path: Path, *, producer: str | None = None) -> str:
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=path,
            kind=ArtifactKind.BENCHMARK_SAMPLES,
            producer=producer,
        )
    )
    return imported.run.run_id


def test_structured_benchmark_extraction_preserves_samples_and_timing_semantics(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "accelerator-benchmark.json"
    _write_samples(source)
    run_id = _import(workspace, source)

    result = BenchmarkSamplesExtractor(workspace).extract(run_id)
    repeated = BenchmarkSamplesExtractor(workspace).extract(run_id)

    assert result.measurement_count == 3
    assert result.warmup_count == 1
    assert result.producer == "decode-benchmark"
    assert result.limitations == ()
    assert repeated.corpus_commit_id == result.corpus_commit_id
    queried = EvidenceQueryService(workspace).measurements(
        run_id=run_id,
        name_prefix="decode.",
        include_warmups=True,
    )
    assert all(
        item.value is not None and item.value.kind == "integer" for item in queried.measurements
    )
    assert {
        item.value.value if item.value is not None else None for item in queried.measurements
    } == {
        45_000,
        42_100,
        41_900,
        42_400,
    }
    assert all(
        item.dimensions
        == {
            "batch": "1",
            "dtype": "bfloat16",
            "mode": "cuda_graph",
            "measurement_clock": "cuda_event",
            "synchronization": "event_synchronize",
            "producer": "decode-benchmark",
            "producer_version": "git:abc123",
            "device.type": "cuda",
            "device.index": "0",
            "device.stream": "7",
        }
        for item in queried.measurements
    )
    assert [
        item.value.value if item.value is not None else None
        for item in queried.measurements
        if item.is_warmup
    ] == [45_000]
    assert all(item.unit == "ns" for item in queried.measurements)
    assert queried.total == 4


def test_structured_benchmark_reports_unknown_device_synchronization(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "accelerator-benchmark.json"
    _write_samples(source, synchronization="unknown")
    run_id = _import(workspace, source)

    result = BenchmarkSamplesExtractor(workspace).extract(run_id)

    assert result.limitations == (
        "decode.token_latency uses asynchronous device timing with synchronization=unknown.",
    )


def test_structured_benchmark_reports_missing_producer_version(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "accelerator-benchmark.json"
    _write_samples(source)
    payload = json.loads(source.read_text())
    del payload["producer_version"]
    source.write_text(json.dumps(payload))
    run_id = _import(workspace, source)

    result = BenchmarkSamplesExtractor(workspace).extract(run_id)

    assert result.producer_version is None
    assert result.limitations == ("The benchmark producer version was not declared.",)


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {
            "schema_version": "flameox.benchmark-samples.v1",
            "producer": "fixture",
            "benchmarks": [
                {
                    "name": "decode.latency",
                    "unit": "ns",
                    "measurement_clock": "cuda_event",
                    "synchronization": "event_synchronize",
                    "samples": [1.5],
                }
            ],
        },
        {
            "schema_version": "flameox.benchmark-samples.v1",
            "producer": "fixture",
            "benchmarks": [
                {
                    "name": "decode.latency",
                    "unit": "ns",
                    "measurement_clock": "cuda_event",
                    "synchronization": "event_synchronize",
                    "samples": [True],
                }
            ],
        },
        {
            "schema_version": "flameox.benchmark-samples.v1",
            "producer": "fixture",
            "benchmarks": [
                {
                    "name": "decode.latency",
                    "unit": "ns",
                    "measurement_clock": "cuda_event",
                    "synchronization": "event_synchronize",
                    "samples": ["1"],
                }
            ],
        },
    ),
)
def test_structured_benchmark_rejects_malformed_or_inexact_samples(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "accelerator-benchmark.json"
    source.write_text(json.dumps(payload))
    run_id = _import(workspace, source)

    with pytest.raises(DomainError) as error:
        BenchmarkSamplesExtractor(workspace).extract(run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED


def test_structured_benchmark_rejects_conflicting_registered_producer(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    source = tmp_path / "accelerator-benchmark.json"
    _write_samples(source)
    run_id = _import(workspace, source, producer="different-producer")

    with pytest.raises(DomainError) as error:
        BenchmarkSamplesExtractor(workspace).extract(run_id)

    assert error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
    assert error.value.details["document_producer"] == "decode-benchmark"

    payload = json.loads(source.read_text())
    payload["benchmarks"][0]["dimensions"]["measurement_clock"] = "host_monotonic"
    source.write_text(json.dumps(payload))
    conflicting_dimension_run = _import(workspace, source)

    with pytest.raises(DomainError) as dimension_error:
        BenchmarkSamplesExtractor(workspace).extract(conflicting_dimension_run)

    assert dimension_error.value.code is ErrorCode.ARTIFACT_PARSE_FAILED
