from __future__ import annotations

import asyncio
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import anyio
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import pyperf
import pytest
from coverage import CoverageData
from mcp import Client, StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import MCPError
from mcp_types import TextResourceContents
from pydantic import ValidationError

from flameox import __version__
from flameox.canonical import canonical_bytes
from flameox.mcp import create_server
from flameox.providers.contracts import ProviderAnalysis
from flameox.repository import AGENT_EVIDENCE_MEDIA_TYPE, EvidenceRepository
from flameox.runtime_contracts import (
    MAX_ROWS,
    CaptureTarget,
    EvidenceSource,
    ExperimentCase,
    ExperimentDesign,
    PathSource,
    RequestLimits,
    RuntimeFailure,
)
from flameox.setup import ExternalRequirement, ProviderPreparation, ProviderSelectionFailure
from flameox.stateless import AnalysisRuntime


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory",
    [
        lambda: ExperimentCase(name="case", argv=["bad\x00argument"]),
        lambda: ExperimentCase(name="case", environment={"BAD=NAME": "value"}),
        lambda: ExperimentDesign(
            cases=[ExperimentCase(name="a"), ExperimentCase(name="b")],
            blocks=1,
            seed=1,
            metric="wall_time_ns",
            estimand="median_difference",
            practical_threshold=0,
            semantic_oracle=["bad\x00command"],
        ),
    ],
)
def test_experiment_commands_use_the_same_strict_validation_as_capture_targets(
    factory: Any,
) -> None:
    with pytest.raises(ValidationError):
        factory()


@pytest.mark.unit
def test_typed_capability_never_falls_back_to_generic_rows(tmp_path: Path) -> None:
    pyperf_artifact = tmp_path / "benchmark.json"
    samples_artifact = tmp_path / "benchmark.samples.json"
    pyperf_artifact.write_text("{}")
    samples_artifact.write_text("{}")
    runtime = AnalysisRuntime(tmp_path)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "benchmark.summary",
                [
                    PathSource(path=str(pyperf_artifact), format="pyperf"),
                    PathSource(path=str(samples_artifact), format="samples"),
                ],
                {},
            )
    finally:
        runtime.close()

    assert failure.value.code == "UNSUPPORTED_FORMAT"


@pytest.mark.unit
def test_capture_rejects_declared_provider_capability_mismatch_before_execution(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "executed"

    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path)
        try:
            with pytest.raises(RuntimeFailure) as failure:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[
                            sys.executable,
                            "-c",
                            f"from pathlib import Path; Path({str(marker)!r}).touch()",
                        ],
                        provider_id="pyperf",
                    ),
                    "failures.summary",
                )
        finally:
            runtime.close()
        assert failure.value.code == "UNSUPPORTED_FORMAT"
        assert failure.value.details["output_formats"] == ["pyperf"]
        assert failure.value.details["compatible_capture_providers"] == [
            "observations",
            "pytest",
        ]

    anyio.run(exercise)
    assert not marker.exists()


def test_special_file_sources_are_rejected_before_decoding(tmp_path: Path) -> None:
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)
    runtime = AnalysisRuntime(tmp_path)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("artifact.preview", [PathSource(path=str(fifo))], {})
    finally:
        runtime.close()

    assert failure.value.code == "INVALID_INPUT"


@pytest.mark.process
def test_experiment_environment_limit_applies_after_overrides_are_merged(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path)
        try:
            with pytest.raises(RuntimeFailure) as failure:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[sys.executable, "-c", "pass"],
                        provider_id="direct",
                        environment={f"TARGET_{index}": "value" for index in range(32)},
                    ),
                    "artifact.preview",
                    mode="experiment",
                    experiment=ExperimentDesign(
                        cases=[
                            ExperimentCase(
                                name="baseline",
                                environment={f"CASE_{index}": "value" for index in range(32)},
                            ),
                            ExperimentCase(name="candidate"),
                        ],
                        blocks=1,
                        seed=1,
                        metric="wall_time_ns",
                        estimand="median_difference",
                        practical_threshold=0,
                    ),
                )
        finally:
            runtime.close()

        assert failure.value.code == "INVALID_INPUT"

    anyio.run(exercise)


@pytest.mark.process
def test_capture_rejects_unbounded_durable_provenance_before_execution(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path)
        try:
            before = list(runtime.scratch.iterdir())
            with pytest.raises(RuntimeFailure) as failure:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[sys.executable, "-c", "x" * 8_000],
                        provider_id="direct",
                    ),
                    "artifact.preview",
                    limits=RequestLimits(max_provenance_bytes=4 * 1024),
                )
            after = list(runtime.scratch.iterdir())
        finally:
            runtime.close()

        assert failure.value.code == "LIMIT_EXCEEDED"
        assert before == after == []

    anyio.run(exercise)


@pytest.mark.unit
def test_analysis_is_bounded_deterministic_and_does_not_change_input(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text(json.dumps([{"value": value} for value in range(4)]))
    before = artifact.read_bytes()
    request = [PathSource(path=str(artifact))]
    runtime = AnalysisRuntime(tmp_path)
    try:
        limits = RequestLimits(max_rows=2)
        first = runtime.analyze("artifact.preview", request, {}, limits=limits)
        second = runtime.analyze("artifact.preview", request, {}, limits=limits)
    finally:
        runtime.close()

    assert first == second
    assert first["analysis_id"] == second["analysis_id"]
    assert first["inputs"][0]["sha256"] == hashlib.sha256(before).hexdigest()
    assert first["coverage"] == {"rows_returned": 2, "rows_observed": 3, "complete": False}
    assert first["continuation"]
    assert first["truncation"] == {"reason": "row_limit", "next_offset": 2}
    assert artifact.read_bytes() == before


@pytest.mark.unit
def test_continuation_pages_have_distinct_preservable_analysis_ids(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text(json.dumps([{"value": value} for value in range(4)]))
    runtime = AnalysisRuntime(tmp_path)
    try:
        first = runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=2),
        )
        second = runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=2),
            continuation=first["continuation"],
        )

        assert first["analysis_id"] != second["analysis_id"]
        assert [row["value"] for row in first["blocks"][1]["rows"]] == [0, 1]
        assert [row["value"] for row in second["blocks"][1]["rows"]] == [2, 3]
        assert runtime.preserve_evidence(first["analysis_id"])["evidence_id"]
        assert runtime.preserve_evidence(second["analysis_id"])["evidence_id"]
    finally:
        runtime.close()


@pytest.mark.unit
def test_capability_arguments_reject_unknown_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")

    runtime = AnalysisRuntime(tmp_path)
    with pytest.raises(ValidationError):
        runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact))],
            {"unsupported": True},
        )
    runtime.close()


@pytest.mark.unit
def test_request_limits_can_only_lower_explicit_startup_bounds() -> None:
    startup = RequestLimits(
        max_rows=20,
        timeout_seconds=10,
        max_output_bytes=4096,
        max_memory_bytes=512 * 1024 * 1024,
    )

    effective = RequestLimits(max_rows=5, max_memory_bytes=256 * 1024 * 1024).lowered_against(
        startup
    )

    assert effective.max_rows == 5
    assert effective.timeout_seconds == 10
    assert effective.max_output_bytes == 4096
    assert effective.max_memory_bytes == 256 * 1024 * 1024
    with pytest.raises(RuntimeFailure) as failure:
        RequestLimits(timeout_seconds=11).lowered_against(startup)
    assert failure.value.code == "LIMIT_EXCEEDED"
    with pytest.raises(RuntimeFailure) as failure:
        RequestLimits(max_memory_bytes=1024**3).lowered_against(startup)
    assert failure.value.code == "LIMIT_EXCEEDED"


@pytest.mark.unit
def test_explicit_inputs_fail_at_byte_and_file_bounds(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.txt"
    oversized.write_bytes(b"x" * 2048)
    directory = tmp_path / "inputs"
    directory.mkdir()
    (directory / "one.txt").write_text("one")
    (directory / "two.txt").write_text("two")
    runtime = AnalysisRuntime(tmp_path)
    try:
        with pytest.raises(RuntimeFailure) as byte_failure:
            runtime.analyze(
                "artifact.preview",
                [PathSource(path=str(oversized))],
                {},
                limits=RequestLimits(max_input_bytes=1024),
            )
        assert byte_failure.value.code == "LIMIT_EXCEEDED"

        with pytest.raises(RuntimeFailure) as file_failure:
            runtime.analyze(
                "artifact.preview",
                [PathSource(path=str(directory))],
                {},
                limits=RequestLimits(max_input_files=1),
            )
        assert file_failure.value.code == "LIMIT_EXCEEDED"
    finally:
        runtime.close()


@pytest.mark.unit
def test_input_limits_apply_across_all_explicit_sources(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"a" * 600)
    second.write_bytes(b"b" * 600)
    sources = [PathSource(path=str(first)), PathSource(path=str(second))]
    runtime = AnalysisRuntime(tmp_path)
    try:
        with pytest.raises(RuntimeFailure) as byte_failure:
            runtime.analyze(
                "artifact.preview",
                sources,
                {},
                limits=RequestLimits(max_input_bytes=1024),
            )
        assert byte_failure.value.code == "LIMIT_EXCEEDED"

        with pytest.raises(RuntimeFailure) as file_failure:
            runtime.analyze(
                "artifact.preview",
                sources,
                {},
                limits=RequestLimits(max_input_files=1),
            )
        assert file_failure.value.code == "LIMIT_EXCEEDED"
    finally:
        runtime.close()


@pytest.mark.unit
def test_digest_mismatch_fails_before_decoding(tmp_path: Path) -> None:
    artifact = tmp_path / "invalid.json"
    artifact.write_text("not json")

    runtime = AnalysisRuntime(tmp_path)
    with pytest.raises(RuntimeFailure, match="SHA-256 mismatch") as failure:
        runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact), expected_sha256="0" * 64)],
            {},
        )
    assert failure.value.code == "MISSING_OR_CHANGED_INPUT"
    runtime.close()


@pytest.mark.unit
def test_json_object_sequence_is_streamed_with_bounded_continuation(tmp_path: Path) -> None:
    artifact = tmp_path / "report.json"
    artifact.write_text(json.dumps({"metadata": {"ignored": True}, "results": list(range(20))}))
    runtime = AnalysisRuntime(tmp_path)
    try:
        first = runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=3),
        )
        second = runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=3),
            continuation=first["continuation"],
        )
    finally:
        runtime.close()

    assert [row["value"] for row in first["blocks"][1]["rows"]] == [0, 1, 2]
    assert [row["value"] for row in second["blocks"][1]["rows"]] == [3, 4, 5]


@pytest.mark.unit
def test_json_object_preview_includes_all_arrays_and_root_scalars(tmp_path: Path) -> None:
    artifact = tmp_path / "sections.json"
    artifact.write_text(json.dumps({"first": [1, 2], "label": "kept", "second": [{"value": 3}]}))
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze(
            "artifact.preview",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=10),
        )
    finally:
        runtime.close()

    rows = result["blocks"][1]["rows"]
    assert rows == [
        {"section": "first", "value": 1, "input_sha256": result["inputs"][0]["sha256"]},
        {"section": "first", "value": 2, "input_sha256": result["inputs"][0]["sha256"]},
        {"key": "label", "value": "kept", "input_sha256": result["inputs"][0]["sha256"]},
        {
            "section": "second",
            "value": 3,
            "input_sha256": result["inputs"][0]["sha256"],
        },
    ]


@pytest.mark.unit
def test_preview_source_digest_wins_over_user_row_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "rows.json"
    artifact.write_text(json.dumps([{"input_sha256": "f" * 64, "value": 1}]))
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
    finally:
        runtime.close()

    assert result["blocks"][1]["rows"][0]["input_sha256"] == result["inputs"][0]["sha256"]


@pytest.mark.unit
def test_provider_pages_use_one_stable_projection_limit(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(tmp_path)
    limits_seen: list[int] = []

    def analyze(*_args: Any, max_rows: int, **_kwargs: Any) -> ProviderAnalysis:
        limits_seen.append(max_rows)
        rows = [{"index": index} for index in range(max_rows)]
        return ProviderAnalysis(
            provider_id="test",
            provider_version="1",
            blocks=[{"type": "metrics", "values": {}}, {"type": "table", "rows": rows}],
            rows_observed=max_rows + 1,
            complete=False,
            limitations=[],
        )

    runtime.benchmarks.analyze = analyze  # type: ignore[method-assign]
    try:
        first = runtime.analyze(
            "benchmark.summary",
            [PathSource(path=str(artifact), format="samples")],
            {},
            limits=RequestLimits(max_rows=3),
        )
        second = runtime.analyze(
            "benchmark.summary",
            [PathSource(path=str(artifact), format="samples")],
            {},
            limits=RequestLimits(max_rows=3),
            continuation=first["continuation"],
        )
    finally:
        runtime.close()

    assert limits_seen == [1001, 1001]
    assert first["blocks"][1]["rows"] == [{"index": 0}, {"index": 1}, {"index": 2}]
    assert second["blocks"][1]["rows"] == [{"index": 3}, {"index": 4}, {"index": 5}]


@pytest.mark.process
def test_cpu_profile_uses_explicit_isolated_worker_without_repository(tmp_path: Path) -> None:
    profile = tmp_path / "cpu.cpuprofile"
    profile.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": 1,
                        "callFrame": {
                            "functionName": "main",
                            "url": (tmp_path / "index.js").as_uri(),
                            "scriptId": "1",
                            "lineNumber": 1,
                            "columnNumber": 0,
                        },
                        "hitCount": 2,
                        "children": [],
                    }
                ],
                "samples": [1, 1],
                "startTime": 0,
                "endTime": 100,
            }
        )
    )
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze(
            "cpu.hotspots",
            [PathSource(path=str(profile), format="cpuprofile")],
            {},
        )
    finally:
        runtime.close()

    assert result["provider"]["id"] == "v8-cpu-profile"
    assert result["blocks"][0]["values"]["sample_count"] == 2
    assert result["blocks"][1]["rows"]
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.process
def test_coverage_data_uses_isolated_reader_and_continuation(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("one = 1\ntwo = 2\nthree = 3\n")
    artifact = tmp_path / ".coverage"
    data = CoverageData(basename=str(artifact))
    data.add_lines({str(source): {1, 2, 3}})
    data.write()
    runtime = AnalysisRuntime(tmp_path)
    try:
        first = runtime.analyze(
            "coverage.summary",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=2),
        )
        second = runtime.analyze(
            "coverage.summary",
            [PathSource(path=str(artifact))],
            {},
            limits=RequestLimits(max_rows=2),
            continuation=first["continuation"],
        )
    finally:
        runtime.close()

    assert first["provider"]["id"] == "coverage.py"
    assert first["blocks"][0]["values"] == {
        "file_count": 1,
        "line_count": 3,
        "arc_count": 0,
    }
    assert [row["line_from"] for row in first["blocks"][1]["rows"]] == [1, 2]
    assert [row["line_from"] for row in second["blocks"][1]["rows"]] == [3]
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.process
def test_provider_continuation_stops_at_a_truthfully_reported_bounded_prefix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_text("\n" * 1_205)
    artifact = tmp_path / ".coverage"
    data = CoverageData(basename=str(artifact))
    data.add_lines({str(source): set(range(1, 1_206))})
    data.write()
    runtime = AnalysisRuntime(tmp_path)
    pages: list[dict[str, Any]] = []
    continuation: str | None = None
    try:
        while True:
            page = runtime.analyze(
                "coverage.summary",
                [PathSource(path=str(artifact))],
                {},
                limits=RequestLimits(max_rows=100),
                continuation=continuation,
            )
            pages.append(page)
            continuation = page["continuation"]
            if continuation is None:
                break
    finally:
        runtime.close()

    rows = [row for page in pages for row in page["blocks"][1]["rows"]]
    assert len(rows) == MAX_ROWS + 1
    assert [row["line_from"] for row in rows] == list(range(1, MAX_ROWS + 2))
    assert pages[-1]["coverage"] == {
        "rows_returned": 1,
        "rows_observed": 1_205,
        "complete": False,
    }
    assert pages[-1]["truncation"] == {"reason": "provider_limit", "next_offset": 1_001}
    assert any("truncated to 1001" in item for item in pages[0]["limitations"])


@pytest.mark.unit
def test_sarif_candidates_are_scoped_to_project_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "report.sarif"
    artifact.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {"driver": {"name": "scanner", "version": "1.2.3"}},
                        "results": [
                            {
                                "ruleId": "slow-loop",
                                "level": "warning",
                                "message": {"text": "Loop is a performance candidate"},
                                "locations": [
                                    {
                                        "physicalLocation": {
                                            "artifactLocation": {"uri": "src/slow.py"},
                                            "region": {"startLine": 7},
                                        }
                                    }
                                ],
                            },
                            {
                                "ruleId": "ignored",
                                "message": {"text": "Excluded candidate"},
                                "locations": [
                                    {
                                        "physicalLocation": {
                                            "artifactLocation": {"uri": "tests/test_slow.py"},
                                            "region": {"startLine": 3},
                                        }
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        )
    )
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze(
            "static.performance_candidates",
            [PathSource(path=str(artifact))],
            {"include_paths": ["src/*"]},
        )
    finally:
        runtime.close()

    assert result["provider"] == {"id": "sarif", "version": "2.1.0"}
    assert result["blocks"][0]["values"]["result_count"] == 2
    assert result["blocks"][0]["values"]["excluded_count"] == 1
    assert result["blocks"][1]["rows"][0]["relative_path"] == "src/slow.py"
    assert not (tmp_path / ".flameox").exists()


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


@pytest.mark.process
def test_pyperf_summary_uses_native_isolated_reader(tmp_path: Path) -> None:
    artifact = tmp_path / "benchmark.json"
    _write_pyperf_suite(artifact, [0.010, 0.012])
    runtime = AnalysisRuntime(tmp_path)
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
def test_pyperf_compare_reads_explicit_artifacts_without_catalog(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_pyperf_suite(baseline, [0.010, 0.012])
    _write_pyperf_suite(candidate, [0.005, 0.006])
    runtime = AnalysisRuntime(tmp_path)
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
def test_structured_benchmark_samples_are_isolated_and_comparable(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.samples.json"
    candidate = tmp_path / "candidate.samples.json"
    _write_benchmark_samples(baseline, [10, 12])
    _write_benchmark_samples(candidate, [5, 6])
    runtime = AnalysisRuntime(tmp_path)
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
def test_pyperf_capture_binds_native_output_before_analysis(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path, limits=RequestLimits(timeout_seconds=20))
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass"],
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
            manifest = runtime.read_evidence(preserved["evidence_id"])
            assert {item["role"] for item in manifest["body"]["artifacts"]} == {
                "capture-0001/stdout",
                "capture-0001/stderr",
                "capture-0001/benchmark",
            }
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_pyperf_capture_preserves_multiline_target_argv(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path, limits=RequestLimits(timeout_seconds=20))
        try:
            code = "value = 1\nassert value == 1"
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", code],
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
    runtime = AnalysisRuntime(tmp_path)
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
            {},
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
def test_failed_provider_capture_returns_preservable_stream_evidence(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path, limits=RequestLimits(timeout_seconds=20))
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[
                        sys.executable,
                        "-c",
                        "import sys; print('before-failure'); raise SystemExit(7)",
                    ],
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
            assert result["capability_id"] == "artifact.preview"
            assert result["blocks"][1]["rows"]
            preserved = runtime.preserve_evidence(result["analysis_id"])
            manifest = runtime.read_evidence(preserved["evidence_id"])
            assert {item["role"] for item in manifest["body"]["artifacts"]} == {
                "capture-0001/stdout",
                "capture-0001/stderr",
            }
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_coverage_capture_uses_explicit_empty_config_and_native_data(tmp_path: Path) -> None:
    script = tmp_path / "covered.py"
    script.write_text("value = 1\nprint(value)\n")

    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path, limits=RequestLimits(timeout_seconds=20))
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, str(script)],
                    provider_id="coverage",
                    capture_arguments={"branch": True, "source": [str(tmp_path)]},
                ),
                "coverage.summary",
            )
            execution = result["capture"]["executions"][0]
            assert execution["status"] == "succeeded"
            assert "--rcfile" in execution["capture_argv"]
            assert result["provider"]["id"] == "coverage.py"
            assert result["blocks"][0]["values"]["line_count"] >= 2
            assert result["inputs"][0]["format"] == "coverage"
            assert not (tmp_path / ".coverage").exists()
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_coverage_capture_rejects_workload_interpreter_without_provider(tmp_path: Path) -> None:
    workload_python = tmp_path / "python"
    workload_python.write_text("#!/bin/sh\nexit 7\n")
    workload_python.chmod(0o755)

    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path, limits=RequestLimits(timeout_seconds=20))
        try:
            with pytest.raises(RuntimeFailure) as raised:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[str(workload_python), "workload.py"],
                        provider_id="coverage",
                    ),
                    "coverage.summary",
                )
        finally:
            runtime.close()

        assert raised.value.code == "UNAVAILABLE_CAPABILITY"
        assert "workload interpreter" in raised.value.message

    anyio.run(exercise)


@pytest.mark.unit
def test_nsight_parquetdir_is_analyzed_without_sqlite_or_repository(tmp_path: Path) -> None:
    export = tmp_path / "report.parquetdir"
    export.mkdir()
    pq.write_table(
        pa.table({"start_ns": [1, 2, 3], "kernel": ["a", "b", "c"]}),
        export / "CUDA_GPU_KERN_SUM.parquet",
    )
    runtime = AnalysisRuntime(tmp_path)
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


@pytest.mark.process
def test_native_nsight_report_uses_cached_parquetdir_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / "template.parquet"
    pq.write_table(pa.table({"start_ns": [1], "kernel": ["cached"]}), template)
    counter = tmp_path / "exports.txt"
    executable = tmp_path / "bin" / "nsys"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, shutil, sys\n"
        "arguments = sys.argv[1:]\n"
        "output = pathlib.Path(arguments[arguments.index('--output') + 1])\n"
        "destination = output.with_suffix('.parquetdir')\n"
        "destination.mkdir(parents=True, exist_ok=True)\n"
        f"shutil.copyfile(pathlib.Path({str(template)!r}), "
        "destination / 'CUDA_GPU_KERN_SUM.parquet')\n"
        f"with pathlib.Path({str(counter)!r}).open('a') as stream: stream.write('1\\n')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"native-nsight-report")

    runtime = AnalysisRuntime(tmp_path)
    try:
        first = runtime.analyze("gpu.launches", [PathSource(path=str(report))], {})
        runtime.analyses.clear()
        second = runtime.analyze("gpu.launches", [PathSource(path=str(report))], {})
    finally:
        runtime.close()

    assert first["blocks"][1]["rows"][0]["kernel"] == "cached"
    assert second["blocks"][1]["rows"] == first["blocks"][1]["rows"]
    assert counter.read_text().splitlines() == ["1"]
    assert not (tmp_path / ".flameox").exists()


def _write_otlp_trace(path: Path) -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    request = ExportTraceServiceRequest()
    resource_spans = request.resource_spans.add()
    scope_spans = resource_spans.scope_spans.add()
    scope_spans.scope.name = "test-scope"
    for index, start in enumerate((100, 200, 300), 1):
        span = scope_spans.spans.add()
        span.trace_id = bytes.fromhex("01" * 16)
        span.span_id = index.to_bytes(8, "big")
        span.name = f"span-{index}"
        span.start_time_unix_nano = start
        span.end_time_unix_nano = start + 10
    path.write_bytes(request.SerializeToString())


@pytest.mark.process
def test_otlp_partial_rows_continue_without_legacy_application_service(tmp_path: Path) -> None:
    trace = tmp_path / "trace.otlp"
    _write_otlp_trace(trace)
    runtime = AnalysisRuntime(tmp_path)
    try:
        first = runtime.analyze(
            "trace.summary",
            [PathSource(path=str(trace), format="otlp")],
            {},
            limits=RequestLimits(max_rows=3),
        )
        second = runtime.analyze(
            "trace.summary",
            [PathSource(path=str(trace), format="otlp")],
            {},
            limits=RequestLimits(max_rows=3),
            continuation=first["continuation"],
        )
    finally:
        runtime.close()

    assert first["provider"]["id"] == "otlp"
    assert first["coverage"] == {"rows_returned": 3, "rows_observed": 5, "complete": False}
    assert [row["name"] for row in second["blocks"][1]["rows"]] == ["span-2", "span-3"]
    assert second["coverage"]["complete"] is True
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.process
def test_otlp_window_filters_inside_isolated_parser(tmp_path: Path) -> None:
    trace = tmp_path / "trace.otlp"
    _write_otlp_trace(trace)
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze(
            "trace.window",
            [PathSource(path=str(trace), format="otlp")],
            {"start_ns": 150, "end_ns": 250},
        )
    finally:
        runtime.close()

    span_rows = [row for row in result["blocks"][1]["rows"] if row["table"] == "spans"]
    assert [row["name"] for row in span_rows] == ["span-2"]


@pytest.mark.unit
def test_pytest_stream_has_typed_summary_and_bounded_rows(tmp_path: Path) -> None:
    events = tmp_path / "pytest.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"event": "run_started", "run_started_at_ns": 1},
                {"event": "test_collected", "nodeid": "test_ok"},
                {
                    "event": "test_phase",
                    "nodeid": "test_ok",
                    "phase": "call",
                    "outcome": "passed",
                    "duration_ns": 10,
                    "worker_id": "main",
                },
                {"event": "test_collected", "nodeid": "test_missing"},
                {"event": "run_finished"},
            )
        )
        + "\n"
    )
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze(
            "failures.summary",
            [PathSource(path=str(events), format="pytest")],
            {},
            limits=RequestLimits(max_rows=2),
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"] == {
        "completion": "complete",
        "collected": 2,
        "executed": 1,
        "unexecuted": 1,
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "errored": 0,
    }
    assert result["coverage"] == {"rows_returned": 1, "rows_observed": 1, "complete": True}
    assert result["blocks"][1]["rows"][0]["classification"] == "unexecuted"


@pytest.mark.unit
def test_pytest_failure_identities_precede_large_successful_population(tmp_path: Path) -> None:
    events = tmp_path / "pytest.jsonl"
    nodeids = [f"tests/test_large.py::test_{index:04d}" for index in range(1_472)]
    failing = set(nodeids[-15:])
    payloads = [{"event": "test_collected", "nodeid": nodeid} for nodeid in nodeids]
    payloads.extend(
        {
            "event": "test_phase",
            "nodeid": nodeid,
            "phase": "call",
            "outcome": "failed" if nodeid in failing else "passed",
        }
        for nodeid in nodeids
    )
    payloads.append({"event": "run_finished"})
    events.write_text("\n".join(json.dumps(item) for item in payloads) + "\n")
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze(
            "failures.summary",
            [PathSource(path=str(events), format="pytest")],
            {},
            limits=RequestLimits(max_rows=10),
        )
        second = runtime.analyze(
            "failures.summary",
            [PathSource(path=str(events), format="pytest")],
            {},
            limits=RequestLimits(max_rows=10),
            continuation=result["continuation"],
        )
    finally:
        runtime.close()

    rows = [*result["blocks"][1]["rows"], *second["blocks"][1]["rows"]]
    assert {row["nodeid"] for row in rows} == failing
    assert all(row["classification"] == "failed" for row in rows)
    assert second["coverage"]["complete"] is True


@pytest.mark.process
def test_pytest_collection_failure_identity_is_reported(tmp_path: Path) -> None:
    broken = tmp_path / "test_broken.py"
    broken.write_text("raise RuntimeError('collection failed')\n")

    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path)
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-m", "pytest", str(broken), "-q"],
                    provider_id="pytest",
                ),
                "failures.summary",
            )
        finally:
            runtime.close()

        assert result["blocks"][0]["values"]["errored"] == 1
        assert result["blocks"][1]["rows"] == [
            {
                "index": 1,
                "nodeid": "test_broken.py",
                "classification": "errored",
                "failing_phase": "collection",
                "phase_outcomes": {"collection": "failed"},
            }
        ]

    anyio.run(exercise)


@pytest.mark.process
def test_pytest_capture_produces_analyzable_session_evidence(tmp_path: Path) -> None:
    (tmp_path / "local_module.py").write_text("VALUE = 7\n")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "conftest.py").write_text("from local_module import VALUE\nassert VALUE == 7\n")
    test_file = tests_dir / "test_sample.py"
    test_file.write_text(
        "from local_module import VALUE\n\n"
        "def test_passes():\n"
        "    assert VALUE == 7\n\n"
        "def test_skips():\n"
        "    import pytest\n"
        "    pytest.skip('bounded example')\n"
    )

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(tmp_path)
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-m", "pytest", "-q", str(test_file)],
                    provider_id="pytest",
                ),
                "failures.summary",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    assert result["provider"] == {"id": "pytest", "version": "event-stream-v1"}
    assert result["blocks"][0]["values"] == {
        "completion": "complete",
        "collected": 2,
        "executed": 2,
        "unexecuted": 0,
        "passed": 1,
        "failed": 0,
        "skipped": 1,
        "errored": 0,
    }
    assert result["capture"]["executions"][0]["status"] == "succeeded"
    assert result["inputs"][0]["format"] == "pytest"
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.unit
def test_pytest_interruption_is_not_overwritten_by_session_finish(tmp_path: Path) -> None:
    events = tmp_path / "pytest.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"event": "run_started", "run_started_at_ns": 1},
                {"event": "interrupted"},
                {"event": "run_finished", "exitstatus": 2},
            )
        )
        + "\n"
    )
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze(
            "failures.summary", [PathSource(path=str(events), format="pytest")], {}
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"]["completion"] == "interrupted"


@pytest.mark.unit
def test_semantic_observations_reject_unknown_fields(tmp_path: Path) -> None:
    events = tmp_path / "observations.jsonl"
    events.write_text(
        json.dumps(
            {
                "name": "phase",
                "phase": None,
                "monotonic_ns": 1,
                "values": {},
                "unexpected": True,
            }
        )
        + "\n"
    )
    runtime = AnalysisRuntime(tmp_path)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "failures.summary",
                [PathSource(path=str(events), format="observations")],
                {},
            )
    finally:
        runtime.close()

    assert failure.value.code == "DECODE_FAILURE"


@pytest.mark.integration
@pytest.mark.requires_memray
def test_memray_capture_is_analyzed_without_workspace_or_repository(tmp_path: Path) -> None:
    memray = pytest.importorskip("memray")
    capture = tmp_path / "memory.bin"
    with memray.Tracker(str(capture)):
        retained = [bytearray(1_024) for _ in range(8)]
    assert retained

    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze(
            "memory.hotspots",
            [PathSource(path=str(capture), format="memray", producer="memray")],
            {},
            limits=RequestLimits(max_rows=10),
        )
    finally:
        runtime.close()

    assert result["provider"] == {"id": "memray", "version": memray.__version__}
    assert result["blocks"][0]["values"]["peak_memory_bytes"] > 0
    assert result["blocks"][1]["rows"]
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.process
@pytest.mark.requires_memray
def test_direct_memray_capture_uses_typed_argv_and_preserves_native_output(
    tmp_path: Path,
) -> None:
    pytest.importorskip("memray")

    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path)
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "retained = [bytearray(4096) for _ in range(8)]"],
                    provider_id="memray",
                    capture_arguments={"native": False},
                ),
                "memory.hotspots",
                limits=RequestLimits(max_rows=10),
            )
            preserved = runtime.preserve_evidence(result["analysis_id"])
            manifest = runtime.read_evidence(preserved["evidence_id"])
        finally:
            runtime.close()

        execution = result["capture"]["executions"][0]
        assert execution["capture_argv"][1:4] == ["-m", "memray", "run"]
        assert result["provider"]["id"] == "memray"
        assert {item["role"] for item in manifest["body"]["artifacts"]} == {
            "capture-0001/stdout",
            "capture-0001/stderr",
            "capture-0001/memory",
        }

    asyncio.run(exercise())


@pytest.mark.process
def test_memray_capture_rejects_workload_interpreter_without_provider(tmp_path: Path) -> None:
    workload_python = tmp_path / "python"
    workload_python.write_text("#!/bin/sh\nexit 7\n")
    workload_python.chmod(0o755)

    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path, limits=RequestLimits(timeout_seconds=20))
        try:
            with pytest.raises(RuntimeFailure) as raised:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[str(workload_python), "workload.py"],
                        provider_id="memray",
                    ),
                    "memory.hotspots",
                )
        finally:
            runtime.close()

        assert raised.value.code == "UNAVAILABLE_CAPABILITY"
        assert "memray >=1.17" in raised.value.message

    anyio.run(exercise)


@pytest.mark.process
def test_aiperf_export_is_projected_without_prompts_or_repository(tmp_path: Path) -> None:
    pytest.importorskip("aiperf")
    export = tmp_path / "profile_export.jsonl"
    export.write_text(
        json.dumps(
            {
                "metadata": {
                    "session_num": 7,
                    "x_request_id": "request-7",
                    "conversation_id": "conversation-a",
                    "turn_index": 2,
                    "request_start_ns": 125,
                    "request_end_ns": 10_000_125,
                    "worker_id": "worker-0",
                    "record_processor_id": "processor-0",
                    "benchmark_phase": "profiling",
                    "was_cancelled": False,
                },
                "metrics": {
                    "input_sequence_length": {"value": 20, "unit": "tokens"},
                    "output_sequence_length": {"value": 3, "unit": "tokens"},
                    "time_to_first_token": {"value": 2, "unit": "ms"},
                    "request_latency": {"value": 10, "unit": "ms"},
                },
                "error": None,
                "raw_prompt": "must never leave the isolated reader",
            }
        )
        + "\n"
    )
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze(
            "inference.summary",
            [PathSource(path=str(export), format="aiperf", producer="aiperf")],
            {},
        )
    finally:
        runtime.close()

    assert result["provider"]["id"] == "aiperf"
    assert result["blocks"][0]["values"]["median_ttft_ns"] == 2_000_000
    assert result["blocks"][1]["rows"][0]["source_request_id"] == "conversation-a:2"
    assert "raw_prompt" not in json.dumps(result)
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.unit
def test_inspection_exposes_strict_validation_schema(tmp_path: Path) -> None:
    runtime = AnalysisRuntime(tmp_path)
    details = runtime.inspect_capabilities(["trace.window", "benchmark.summary"])["capabilities"]
    runtime.close()
    detail = details[0]
    schema = detail["argument_schema"]

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["start_ns", "end_ns"]
    assert detail["source_modes"] == ["path", "evidence"]
    pyperf_capture = details[1]["capture_providers"][0]
    assert pyperf_capture["id"] == "pyperf"
    assert pyperf_capture["capture_argument_schema"]["additionalProperties"] is False
    assert "processes" in pyperf_capture["capture_argument_schema"]["properties"]


@pytest.mark.unit
def test_discovery_reports_external_remediation_without_setup_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flameox.stateless.shutil.which", lambda _name: None)
    runtime = AnalysisRuntime(Path.cwd())
    result = runtime.discover_capabilities(
        "gpu launches",
        [PathSource(path="/artifact.nsys-rep", format="nsys-rep")],
        include_unavailable=True,
    )
    runtime.close()
    match = next(item for item in result["capabilities"] if item["id"] == "gpu.launches")

    assert match["available"] is False
    assert match["providers"][0]["missing_executable"] == "nsys"
    assert "system/vendor package manager" in match["remediation"][0]
    assert "setup" not in json.dumps(match).lower()


@pytest.mark.integration
def test_discovery_uses_the_selected_evidence_artifact_format(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value": 1}]')
    runtime = AnalysisRuntime(tmp_path)
    try:
        analysis = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(analysis["analysis_id"])
        discovered = runtime.discover_capabilities(
            sources=[EvidenceSource(kind="evidence", evidence_id=preserved["evidence_id"])],
        )
    finally:
        runtime.close()

    preview = next(item for item in discovered["capabilities"] if item["id"] == "artifact.preview")
    assert discovered["sniffed_sources"][0]["format"] == "json"
    assert preview["available"] is True


@pytest.mark.unit
def test_discovery_separates_builtin_readers_from_capture_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("flameox.stateless.shutil.which", lambda _name: None)
    runtime = AnalysisRuntime(tmp_path)
    monkeypatch.setattr("flameox.stateless.sys.executable", str(tmp_path / "python"))
    discovered = runtime.discover_capabilities(
        sources=[PathSource(path="/profile.json", format="py-spy")],
        include_unavailable=True,
    )
    inspected = runtime.inspect_capabilities(["cpu.hotspots"])
    runtime.close()

    cpu = next(item for item in discovered["capabilities"] if item["id"] == "cpu.hotspots")
    assert cpu["available"] is True
    assert cpu["providers"][0]["id"] == "py-spy-speedscope"
    pyspy_capture = next(
        item for item in inspected["capabilities"][0]["capture_providers"] if item["id"] == "py-spy"
    )
    assert pyspy_capture["availability"]["available"] is False
    assert pyspy_capture["availability"]["missing_executable"] == "py-spy"


@pytest.mark.unit
def test_py_spy_capture_discovers_executable_in_managed_tool_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    managed_bin = tmp_path / "managed" / "bin"
    managed_bin.mkdir(parents=True)
    managed_python = managed_bin / "python"
    managed_python.write_text("#!/bin/sh\nexit 0\n")
    managed_python.chmod(0o755)
    managed_pyspy = managed_bin / "py-spy"
    managed_pyspy.write_text("#!/bin/sh\nexit 0\n")
    managed_pyspy.chmod(0o755)
    monkeypatch.setattr("flameox.stateless.sys.executable", str(managed_python))
    monkeypatch.setattr("flameox.stateless.shutil.which", lambda _name: None)

    runtime = AnalysisRuntime(tmp_path)
    try:
        inspected = runtime.inspect_capabilities(["cpu.hotspots"])
    finally:
        runtime.close()

    capture = next(
        item for item in inspected["capabilities"][0]["capture_providers"] if item["id"] == "py-spy"
    )
    assert capture["availability"]["available"] is True
    assert capture["availability"]["executable"] == str(managed_pyspy)


@pytest.mark.process
def test_py_spy_capture_executes_managed_tool_when_request_path_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload_python = sys.executable
    managed_bin = tmp_path / "managed" / "bin"
    managed_bin.mkdir(parents=True)
    managed_python = managed_bin / "python"
    managed_python.symlink_to(workload_python)
    managed_pyspy = managed_bin / "py-spy"
    managed_pyspy.write_text(
        f"#!{workload_python}\n"
        "import json, sys\n"
        "output = sys.argv[sys.argv.index('--output') + 1]\n"
        "document = {\n"
        "    'shared': {'frames': [{'name': 'work', 'file': 'work.py', 'line': 1}]},\n"
        "    'profiles': [{'type': 'sampled', 'samples': [[0]], 'weights': [1.0]}],\n"
        "}\n"
        "with open(output, 'w') as stream:\n"
        "    json.dump(document, stream)\n"
    )
    managed_pyspy.chmod(0o755)
    monkeypatch.setattr("flameox.stateless.sys.executable", str(managed_python))

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(tmp_path)
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[workload_python, "-c", "sum(range(100))"],
                    environment={"PATH": ""},
                    provider_id="py-spy",
                ),
                "cpu.hotspots",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    execution = result["capture"]["executions"][0]
    assert execution["capture_argv"][0] == str(managed_pyspy)
    assert execution["status"] == "succeeded"
    assert result["provider"]["id"] == "py-spy-speedscope"


@pytest.mark.unit
def test_discovery_reports_missing_and_unsupported_python_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_find_spec = importlib.util.find_spec

    def missing_memray(name: str) -> Any:
        return None if name == "memray" else original_find_spec(name)

    monkeypatch.setattr("flameox.stateless.importlib.util.find_spec", missing_memray)
    runtime = AnalysisRuntime(tmp_path)
    missing = runtime.discover_capabilities(
        sources=[PathSource(path="/artifact.bin", format="memray")],
        include_unavailable=True,
    )
    runtime.close()
    memory = next(item for item in missing["capabilities"] if item["id"] == "memory.hotspots")
    assert memory["available"] is False
    assert memory["providers"][0]["missing_package"] == "memray"
    assert "flameox setup --provider memray" in memory["remediation"][0]

    def installed_memray(name: str) -> Any:
        if name == "memray":
            return importlib.machinery.ModuleSpec(name, loader=None)
        return original_find_spec(name)

    monkeypatch.setattr("flameox.stateless.importlib.util.find_spec", installed_memray)
    monkeypatch.setattr("flameox.stateless.importlib.metadata.version", lambda _name: "1.0")
    runtime = AnalysisRuntime(tmp_path)
    unsupported = runtime.discover_capabilities(
        sources=[PathSource(path="/artifact.bin", format="memray")],
        include_unavailable=True,
    )
    runtime.close()
    memory = next(item for item in unsupported["capabilities"] if item["id"] == "memory.hotspots")
    assert memory["available"] is False
    assert memory["providers"][0]["unsupported_version"] is True


@pytest.mark.unit
def test_in_process_capture_availability_is_workload_interpreter_dependent(
    tmp_path: Path,
) -> None:
    runtime = AnalysisRuntime(tmp_path)
    try:
        inspected = runtime.inspect_capabilities(["coverage.summary", "memory.hotspots"])
    finally:
        runtime.close()

    for capability in inspected["capabilities"]:
        capture = capability["capture_providers"][0]
        assert capture["availability"]["available"] is None
        assert capture["availability"]["status"] == "target_dependent"
        assert capture["availability"]["environment_scope"] == "workload_interpreter"


@pytest.mark.unit
def test_discovery_reports_wrong_platform_and_executable_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host_platform = sys.platform
    monkeypatch.setattr("flameox.stateless.sys.platform", "darwin")
    monkeypatch.setattr("flameox.stateless.shutil.which", lambda _name: "/usr/bin/nsys")
    runtime = AnalysisRuntime(tmp_path)
    wrong_platform = runtime.discover_capabilities(
        sources=[PathSource(path="/artifact.nsys-rep", format="nsys-rep")],
        include_unavailable=True,
    )
    runtime.close()
    launches = next(item for item in wrong_platform["capabilities"] if item["id"] == "gpu.launches")
    assert launches["available"] is False
    assert "not supported on darwin" in launches["limitations"][0]

    processor = tmp_path / "trace_processor_shell"
    processor.write_text("binary")
    monkeypatch.setenv("FLAMEOX_TRACE_PROCESSOR", str(processor))
    monkeypatch.setattr("flameox.stateless.sys.platform", host_platform)
    runtime = AnalysisRuntime(tmp_path)
    monkeypatch.setattr("flameox.stateless.os.access", lambda _path, _mode: False)
    permission = runtime.discover_capabilities(
        sources=[PathSource(path="/artifact.pftrace", format="perfetto")],
        include_unavailable=True,
    )
    runtime.close()
    trace = next(item for item in permission["capabilities"] if item["id"] == "trace.summary")
    assert trace["available"] is False
    assert trace["providers"][0]["permission_limited"] is True


@pytest.mark.unit
def test_discovery_actively_reports_perf_event_permission_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("flameox.stateless.shutil.which", lambda name: "/usr/bin/perf")
    monkeypatch.setattr(
        "flameox.stateless.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 255, stderr="Access denied by perf_event_paranoid"
        ),
    )
    runtime = AnalysisRuntime(tmp_path)
    try:
        inspected = runtime.inspect_capabilities(["cpu.hotspots"])
    finally:
        runtime.close()

    perf = next(
        provider
        for provider in inspected["capabilities"][0]["capture_providers"]
        if provider["id"] == "perf"
    )
    assert perf["availability"]["available"] is False
    assert perf["availability"]["permission_limited"] is True
    assert "CAP_PERFMON" in perf["availability"]["remediation"][0]


@pytest.mark.unit
def test_discovery_requires_nsight_compute_official_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "ncu"
    executable.write_text("binary")
    executable.chmod(0o755)
    monkeypatch.setattr(
        "flameox.stateless.shutil.which",
        lambda name: str(executable) if name == "ncu" else None,
    )
    monkeypatch.setattr("flameox.stateless.find_report_interface", lambda _path: None)
    runtime = AnalysisRuntime(tmp_path)
    try:
        discovered = runtime.discover_capabilities(
            sources=[PathSource(path="/artifact.ncu-rep", format="nsight-compute")],
            include_unavailable=True,
        )
    finally:
        runtime.close()

    metrics = next(
        item for item in discovered["capabilities"] if item["id"] == "gpu.kernel_metrics"
    )
    assert metrics["available"] is False
    assert metrics["providers"][0]["missing_resource"] == "ncu_report.py"
    assert "extras/python interface" in metrics["remediation"][0]


@pytest.mark.unit
def test_mcp_catalog_is_exact_typed_and_has_one_resource_template() -> None:
    async def inspect() -> None:
        server = create_server()
        tools = await server.list_tools()
        templates = await server.list_resource_templates()

        assert [tool.name for tool in tools] == [
            "discover_capabilities",
            "inspect_capabilities",
            "prepare_capabilities",
            "analyze",
            "capture_and_analyze",
            "preserve_evidence",
            "query_evidence",
        ]
        assert all(tool.output_schema is not None for tool in tools)
        prepare = next(tool for tool in tools if tool.name == "prepare_capabilities")
        assert prepare.annotations and prepare.annotations.open_world_hint is True
        assert prepare.annotations.read_only_hint is False
        assert prepare.annotations.destructive_hint is False
        assert all(
            tool.annotations and tool.annotations.open_world_hint is False
            for tool in tools
            if tool.name != "prepare_capabilities"
        )
        assert await server.list_resources() == []
        assert [item.uri_template for item in templates] == ["flameox://evidence/{evidence_id}"]
        assert templates[0].mime_type == AGENT_EVIDENCE_MEDIA_TYPE

    anyio.run(inspect)


@pytest.mark.unit
def test_mcp_prepares_managed_providers_and_only_guides_host_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparation_calls: list[list[str]] = []

    def prepare(
        provider_ids: list[str], project_root: Path, timeout_seconds: int
    ) -> ProviderPreparation:
        assert timeout_seconds == 1_800
        if provider_ids == ["unknown-provider"]:
            raise ProviderSelectionFailure("Unknown provider 'unknown-provider'")
        preparation_calls.append(provider_ids)
        return ProviderPreparation(
            ["memray", "nsight-compute"],
            ["memray"],
            [
                ExternalRequirement(
                    "nsight-compute",
                    "Install NVIDIA Nsight Compute with its extras/python interface.",
                )
            ],
            ["/usr/bin/uvx", "--from", f"flameox[memory]=={__version__}", "--version"],
            "uvx",
            [
                "--python",
                "3.12",
                "--from",
                f"flameox[memory]=={__version__}",
                "flameox",
                "mcp",
                "serve",
                "--project-root",
                str(project_root),
            ],
        )

    monkeypatch.setattr("flameox.mcp.server.prepare_providers", prepare)

    async def exercise() -> None:
        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            result = await client.call_tool(
                "prepare_capabilities",
                {"provider_ids": ["memray", "nsight-compute", "memray"]},
            )
            invalid = await client.call_tool(
                "prepare_capabilities",
                {"provider_ids": ["unknown-provider"]},
            )

        assert result.is_error is False
        assert result.structured_content["requested_providers"] == [
            "memray",
            "nsight-compute",
        ]
        assert result.structured_content["prepared_managed_providers"] == ["memray"]
        assert result.structured_content["external_requirements"] == [
            {
                "provider_id": "nsight-compute",
                "guidance": "Install NVIDIA Nsight Compute with its extras/python interface.",
            }
        ]
        assert result.structured_content["preparation"]["status"] == "prepared"
        assert result.structured_content["restart_required"] is True
        assert result.structured_content["launcher"]["args"][3] == (
            f"flameox[memory]=={__version__}"
        )
        assert result.structured_content["launcher"]["args"][-5:] == [
            "flameox",
            "mcp",
            "serve",
            "--project-root",
            str(tmp_path),
        ]

        assert invalid.is_error is True
        assert invalid.structured_content["code"] == "INVALID_INPUT"

    anyio.run(exercise)
    assert preparation_calls == [["memray", "nsight-compute", "memray"]]


@pytest.mark.process
@pytest.mark.serial
def test_real_stdio_initialize_and_catalog_match_the_stateless_contract(tmp_path: Path) -> None:
    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "flameox",
                "mcp",
                "serve",
                "--project-root",
                str(tmp_path),
            ],
            cwd=tmp_path,
        )
        async with stdio_client(parameters) as streams, ClientSession(*streams) as session:
            initialized = await session.initialize()
            tools = await session.list_tools()
            resources = await session.list_resources()
            templates = await session.list_resource_templates()
            inspected = await session.call_tool(
                "inspect_capabilities", {"capability_ids": ["cpu.hotspots"]}
            )
            invalid = await session.call_tool(
                "inspect_capabilities", {"capability_ids": ["unknown.capability"]}
            )
            await session.validate_tool_result("inspect_capabilities", inspected)
            await session.validate_tool_result("inspect_capabilities", invalid)

        assert initialized.server_info.version == __version__
        assert [tool.name for tool in tools.tools] == [
            "discover_capabilities",
            "inspect_capabilities",
            "prepare_capabilities",
            "analyze",
            "capture_and_analyze",
            "preserve_evidence",
            "query_evidence",
        ]
        assert all(tool.output_schema is not None for tool in tools.tools)
        assert invalid.is_error is True
        assert resources.resources == []
        assert [item.uri_template for item in templates.resource_templates] == [
            "flameox://evidence/{evidence_id}"
        ]

    anyio.run(exercise)


@pytest.mark.integration
def test_analysis_preservation_query_resource_and_restart(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1},{"value":2}]')

    async def exercise() -> None:
        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            analyzed = await client.call_tool(
                "analyze",
                {
                    "capability_id": "artifact.preview",
                    "sources": [{"kind": "path", "path": str(artifact)}],
                    "arguments": {},
                },
            )
            assert analyzed.is_error is False
            analysis_id = analyzed.structured_content["analysis_id"]
            preserved = await client.call_tool("preserve_evidence", {"analysis_id": analysis_id})
            assert preserved.is_error is False
            evidence_id = preserved.structured_content["evidence_id"]
            assert any(block.type == "resource_link" for block in preserved.content)
            queried = await client.call_tool("query_evidence", {"limit": 10})
            assert queried.structured_content["evidence"][0]["evidence_id"] == evidence_id
            resource = await client.read_resource(f"flameox://evidence/{evidence_id}")
            assert resource.contents[0].mime_type == AGENT_EVIDENCE_MEDIA_TYPE

        async with Client(create_server(tmp_path), raise_exceptions=True) as restarted:
            reanalyzed = await restarted.call_tool(
                "analyze",
                {
                    "capability_id": "artifact.preview",
                    "sources": [
                        {
                            "kind": "evidence",
                            "evidence_id": evidence_id,
                            "artifact_role": "input",
                        }
                    ],
                    "arguments": {},
                },
            )
            assert reanalyzed.is_error is False
            assert reanalyzed.structured_content["blocks"][1]["rows"][0]["value"] == 1
            expired = await restarted.call_tool("preserve_evidence", {"analysis_id": analysis_id})
            assert expired.is_error is True
            assert expired.structured_content["code"] == "EXPIRED_SESSION_ANALYSIS"
            resource = await restarted.read_resource(f"flameox://evidence/{evidence_id}")
            content = resource.contents[0]
            assert isinstance(content, TextResourceContents)
            assert json.loads(content.text)["evidence_id"] == evidence_id
            with pytest.raises(MCPError):
                await restarted.read_resource(f"flameox://evidence/{'0' * 64}")

    anyio.run(exercise)


@pytest.mark.integration
def test_mcp_evidence_resource_redacts_capture_provenance(tmp_path: Path) -> None:
    secret_argument = "known-safe-argument-placeholder"
    secret_environment = "known-safe-environment-placeholder"
    secret_path = str(tmp_path.resolve())

    async def exercise() -> None:
        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            captured = await client.call_tool(
                "capture_and_analyze",
                {
                    "target": {
                        "argv": [sys.executable, "-c", "print('ok')", secret_argument],
                        "cwd": ".",
                        "environment": {"FLAMEOX_TEST_MARKER": secret_environment},
                        "provider_id": "direct",
                    },
                    "capability_id": "artifact.preview",
                },
            )
            assert captured.is_error is False
            preserved = await client.call_tool(
                "preserve_evidence",
                {"analysis_id": captured.structured_content["analysis_id"]},
            )
            evidence_id = preserved.structured_content["evidence_id"]
            resource = await client.read_resource(f"flameox://evidence/{evidence_id}")
            content = resource.contents[0]
            assert isinstance(content, TextResourceContents)
            projection = content.text
            assert secret_argument not in projection
            assert secret_environment not in projection
            assert secret_path not in projection
            assert '"argv"' not in projection
            assert '"capture_argv"' not in projection
            assert '"cwd"' not in projection
            assert '"environment"' not in projection

            # The MCP projection must not weaken the canonical local provenance record.
            canonical = AnalysisRuntime(tmp_path)
            try:
                manifest = json.dumps(canonical.read_evidence(evidence_id))
            finally:
                canonical.close()
            assert secret_argument in manifest
            assert secret_environment in manifest
            assert secret_path in manifest

    anyio.run(exercise)


@pytest.mark.integration
def test_mcp_validation_unknown_capability_and_failed_execution_are_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")

    async def exercise() -> None:
        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            invalid = await client.call_tool(
                "analyze",
                {
                    "capability_id": "artifact.preview",
                    "sources": [{"kind": "path", "path": str(artifact)}],
                    "arguments": {"unexpected": True},
                },
            )
            assert invalid.is_error is True
            assert invalid.structured_content["code"] == "INVALID_INPUT"

            unknown = await client.call_tool(
                "analyze",
                {
                    "capability_id": "unknown.capability",
                    "sources": [{"kind": "path", "path": str(artifact)}],
                    "arguments": {},
                },
            )
            assert unknown.is_error is True
            assert unknown.structured_content["code"] == "UNKNOWN_CAPABILITY"

            empty_path = tmp_path / "empty-bin"
            empty_path.mkdir()
            unmanaged_python = empty_path / "python"
            unmanaged_python.symlink_to(sys.executable)
            monkeypatch.setattr("flameox.stateless.sys.executable", str(unmanaged_python))
            unavailable = await client.call_tool(
                "capture_and_analyze",
                {
                    "target": {
                        "argv": [sys.executable, "-c", "pass"],
                        "environment": {"PATH": str(empty_path)},
                        "provider_id": "py-spy",
                    },
                    "capability_id": "cpu.hotspots",
                },
            )
            assert unavailable.is_error is True
            assert unavailable.structured_content["code"] == "UNAVAILABLE_CAPABILITY"

            failed = await client.call_tool(
                "capture_and_analyze",
                {
                    "target": {
                        "argv": [sys.executable, "-c", "raise SystemExit(7)"],
                        "provider_id": "direct",
                    }
                },
            )
            assert failed.is_error is True
            assert failed.structured_content["code"] == "EXECUTION_FAILURE"
            partial = failed.structured_content["details"]["partial_evidence"]
            assert partial["analysis_id"]
            assert partial["capture"]["executions"][0]["returncode"] == 7

    anyio.run(exercise)


@pytest.mark.process
def test_direct_capture_reports_progress_and_preserves_native_output(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path, limits=RequestLimits(timeout_seconds=10))
        updates: list[tuple[int, int]] = []

        async def progress(current: int, total: int, _message: str) -> None:
            updates.append((current, total))

        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "print('captured')"],
                    provider_id="direct",
                ),
                "artifact.preview",
                progress=progress,
            )
            assert result["blocks"][1]["rows"][0]["text"] == "captured"
            assert updates == [(0, 1), (1, 1)]
            provider_id = result["provider"]["id"]
            result["provider"]["id"] = "caller-mutated"
            preserved = runtime.preserve_evidence(result["analysis_id"])
            manifest = runtime.read_evidence(preserved["evidence_id"])
            assert {item["role"] for item in manifest["body"]["artifacts"]} == {
                "capture-0001/stdout",
                "capture-0001/stderr",
            }
            bundle = (
                tmp_path
                / ".flameox"
                / "evidence"
                / "sha256"
                / preserved["evidence_id"][:2]
                / preserved["evidence_id"]
            )
            durable = json.loads((bundle / "data" / "analysis.json").read_text())
            assert durable["provider"]["id"] == provider_id
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_timed_out_capture_returns_preservable_partial_evidence(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path)
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[
                        sys.executable,
                        "-c",
                        "import time; print('before-timeout', flush=True); time.sleep(5)",
                    ],
                    provider_id="direct",
                ),
                "artifact.preview",
                limits=RequestLimits(timeout_seconds=0.2),
            )
            execution = result["capture"]["executions"][0]
            assert execution["status"] == "failed"
            assert execution["failure_code"] == "EXECUTION_TIMEOUT"
            assert execution["limit"]["kind"] == "timeout"
            assert execution["limit"]["configured"] == 0.2
            assert execution["limit"]["unit"] == "seconds"
            assert result["blocks"][1]["rows"][0]["text"] == "before-timeout"
            assert runtime.preserve_evidence(result["analysis_id"])["evidence_id"]
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_cancelled_capture_cleans_up_descendants(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "child.pid"

    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path)
        task = asyncio.create_task(
            runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[
                        sys.executable,
                        "-c",
                        (
                            "import pathlib, subprocess, sys, time; "
                            "child=subprocess.Popen([sys.executable, '-c', 'import time; "
                            "time.sleep(30)']); "
                            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid)); "
                            "time.sleep(30)"
                        ),
                    ],
                    provider_id="direct",
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
        finally:
            if not task.done():
                task.cancel()
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_experiment_runs_bounded_cases_and_semantic_oracle(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path)
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "print('base')"],
                    provider_id="direct",
                ),
                "artifact.preview",
                mode="experiment",
                experiment=ExperimentDesign(
                    cases=[
                        ExperimentCase(
                            name="baseline",
                            argv=[sys.executable, "-c", "print('baseline')"],
                        ),
                        ExperimentCase(
                            name="candidate",
                            argv=[sys.executable, "-c", "print('candidate')"],
                        ),
                    ],
                    blocks=1,
                    seed=7,
                    metric="wall_time_ns",
                    estimand="median_difference",
                    practical_threshold=0,
                    semantic_oracle=[
                        sys.executable,
                        "-c",
                        (
                            "import os, pathlib; "
                            "raise SystemExit(not pathlib.Path("
                            "os.environ['FLAMEOX_CAPTURE_STDOUT']).read_text().strip())"
                        ),
                    ],
                ),
            )
            executions = result["capture"]["executions"]
            assert {item["case"] for item in executions} == {"baseline", "candidate"}
            assert all(item["semantic_oracle"]["status"] == "passed" for item in executions)
            assert all(item["status"] == "succeeded" for item in executions)
            comparison = result["blocks"][-1]["rows"][0]
            assert comparison["baseline_case"] == "baseline"
            assert comparison["candidate_case"] == "candidate"
            assert comparison["metric"] == "wall_time_ns"
            assert comparison["estimand"] == "median_difference"
            assert comparison["paired_blocks"] == 1
            assert comparison["decision"] in {
                "practically_improved",
                "practically_regressed",
                "within_threshold",
            }
            preserved = runtime.preserve_evidence(result["analysis_id"])
            manifest = runtime.read_evidence(preserved["evidence_id"])
            assert manifest["body"]["limitations"] == result["limitations"]
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_capture_result_applies_byte_bound_after_provenance(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path)
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass", "x" * 1_000],
                    provider_id="direct",
                ),
                "artifact.preview",
                mode="experiment",
                experiment=ExperimentDesign(
                    cases=[ExperimentCase(name=f"case-{index}") for index in range(16)],
                    blocks=1,
                    seed=7,
                    metric="wall_time_ns",
                    estimand="median_difference",
                    practical_threshold=0,
                ),
                limits=RequestLimits(max_result_bytes=16_384),
            )
        finally:
            runtime.close()

        assert len(canonical_bytes(result)) <= 16_384
        assert result["capture"]["execution_count"] == 16
        assert result["capture"]["executions_truncated"] >= 0

    anyio.run(exercise)


@pytest.mark.integration
def test_missing_repository_metadata_does_not_hide_preserved_evidence(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1}]')
    runtime = AnalysisRuntime(tmp_path)
    try:
        analysis = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(analysis["analysis_id"])
        (tmp_path / ".flameox" / "repository.json").unlink()

        with pytest.raises(RuntimeFailure) as query_failure:
            runtime.query_evidence()
        assert query_failure.value.code == "REPOSITORY_CORRUPTION"
        with pytest.raises(RuntimeFailure) as read_failure:
            runtime.read_evidence(preserved["evidence_id"])
        assert read_failure.value.code == "REPOSITORY_CORRUPTION"
        second = runtime.analyze(
            "artifact.preview", [PathSource(path=str(artifact))], {"offset": 1}
        )
        with pytest.raises(RuntimeFailure) as preserve_failure:
            runtime.preserve_evidence(second["analysis_id"])
        assert preserve_failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.unit
def test_experiment_zero_effect_is_within_zero_threshold() -> None:
    experiment = ExperimentDesign(
        cases=[ExperimentCase(name="baseline"), ExperimentCase(name="candidate")],
        blocks=1,
        seed=1,
        metric="wall_time_ns",
        estimand="median_difference",
        practical_threshold=0,
    )
    blocks, _limitations = AnalysisRuntime._experiment_blocks(
        experiment,
        [
            {"case": "baseline", "block": 1, "status": "succeeded", "wall_time_ns": 10},
            {"case": "candidate", "block": 1, "status": "succeeded", "wall_time_ns": 10},
        ],
    )

    assert blocks[-1]["rows"][0]["estimate"] == 0
    assert blocks[-1]["rows"][0]["decision"] == "within_threshold"


@pytest.mark.integration
def test_concurrent_identical_publication_reuses_complete_bundle(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1}]')
    runtimes = [AnalysisRuntime(tmp_path) for _ in range(8)]
    analyses = [
        runtime.analyze(
            "artifact.preview", [PathSource(path=str(artifact))], {}, limits=RequestLimits()
        )
        for runtime in runtimes
    ]
    # Freeze the episode timestamp so every publication has the same content identity.
    episode = runtimes[0].analyses[analyses[0]["analysis_id"]].manifest_body["episode"]
    for runtime, analysis in zip(runtimes[1:], analyses[1:], strict=True):
        runtime.analyses[analysis["analysis_id"]].manifest_body["episode"] = episode

    try:
        with ThreadPoolExecutor(max_workers=len(runtimes)) as executor:
            results = list(
                executor.map(
                    lambda pair: pair[0].preserve_evidence(pair[1]["analysis_id"]),
                    zip(runtimes, analyses, strict=True),
                )
            )
        assert len({result["evidence_id"] for result in results}) == 1
        assert (
            runtimes[0].read_evidence(results[0]["evidence_id"])["evidence_id"]
            == results[0]["evidence_id"]
        )
    finally:
        for runtime in runtimes:
            runtime.close()


@pytest.mark.integration
def test_interrupted_evidence_publication_never_exposes_partial_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1}]')
    runtime = AnalysisRuntime(tmp_path)
    result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
    original = EvidenceRepository._publish_directory

    def interrupt_evidence(repository: EvidenceRepository, stage: Path, destination: Path) -> None:
        if "evidence" in destination.parts:
            raise OSError("injected evidence publication interruption")
        original(repository, stage, destination)

    monkeypatch.setattr(EvidenceRepository, "_publish_directory", interrupt_evidence)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(result["analysis_id"])
        assert failure.value.code == "REPOSITORY_IO_FAILURE"
        assert runtime.query_evidence()["evidence"] == []
        assert not list((tmp_path / ".flameox" / "evidence").rglob("manifest.json"))
    finally:
        runtime.close()


@pytest.mark.integration
def test_interrupted_artifact_publication_never_exposes_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text('[{"value":1}]')
    runtime = AnalysisRuntime(tmp_path)
    result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
    original = EvidenceRepository._publish_directory

    def interrupt_artifact(repository: EvidenceRepository, stage: Path, destination: Path) -> None:
        if "artifacts" in destination.parts:
            raise OSError("injected artifact publication interruption")
        original(repository, stage, destination)

    monkeypatch.setattr(EvidenceRepository, "_publish_directory", interrupt_artifact)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(result["analysis_id"])
        assert failure.value.code == "REPOSITORY_IO_FAILURE"
        artifact_root = tmp_path / ".flameox" / "artifacts" / "sha256"
        assert not list(artifact_root.glob("*/*"))
        assert runtime.query_evidence()["evidence"] == []
    finally:
        runtime.close()


@pytest.mark.integration
def test_abandoned_staging_cleanup_removes_only_proven_dead_owner(tmp_path: Path) -> None:
    runtime = AnalysisRuntime(tmp_path)
    try:
        runtime.repository.initialize()
        dead = tmp_path / ".flameox" / ".staging" / "999999999-dead" / "publication"
        unknown = tmp_path / ".flameox" / ".staging" / "unknown" / "publication"
        dead.mkdir(parents=True)
        unknown.mkdir(parents=True)

        runtime.repository.cleanup_abandoned_staging()

        assert not dead.parent.exists()
        assert unknown.parent.exists()
    finally:
        runtime.close()


@pytest.mark.integration
def test_unsupported_repository_format_is_not_read(tmp_path: Path) -> None:
    repository = tmp_path / ".flameox"
    repository.mkdir()
    (repository / "repository.json").write_text(
        json.dumps({"format_version": "999", "created_at": "2026-08-31T00:00:00+00:00"})
    )
    runtime = AnalysisRuntime(tmp_path)
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.query_evidence()
        assert failure.value.code == "UNSUPPORTED_REPOSITORY_FORMAT"
    finally:
        runtime.close()


@pytest.mark.integration
def test_preservation_rejects_symlinked_repository_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / ".flameox").symlink_to(outside, target_is_directory=True)
    artifact = project / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(project)
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})

        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(result["analysis_id"])

        assert failure.value.code == "REPOSITORY_CORRUPTION"
        assert not (outside / "repository.json").exists()
    finally:
        runtime.close()


@pytest.mark.unit
def test_unpreserved_operations_create_no_durable_state(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(tmp_path)
    try:
        runtime.discover_capabilities(sources=[PathSource(path=str(artifact))])
        runtime.inspect_capabilities(["artifact.preview"])
        runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
    finally:
        runtime.close()

    assert not (tmp_path / ".flameox").exists()
    assert not (tmp_path / ".diagnostics").exists()
    assert not list(tmp_path.rglob("*.sqlite*"))
    assert not list(tmp_path.rglob("*.duckdb"))


@pytest.mark.integration
def test_preservation_rejects_input_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        artifact.write_text('[{"changed":true}]')
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(result["analysis_id"])
        assert failure.value.code == "MISSING_OR_CHANGED_INPUT"
    finally:
        runtime.close()


@pytest.mark.integration
def test_corrupt_manifest_and_missing_data_are_not_returned(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(result["analysis_id"])
        evidence_id = preserved["evidence_id"]
        bundle = tmp_path / ".flameox" / "evidence" / "sha256" / evidence_id[:2] / evidence_id
        (bundle / "data" / "analysis.json").unlink()

        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence(evidence_id)
        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_repository_rejects_symlinked_evidence_data(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(result["analysis_id"])
        evidence_id = preserved["evidence_id"]
        bundle = tmp_path / ".flameox" / "evidence" / "sha256" / evidence_id[:2] / evidence_id
        outside = tmp_path / "outside-data"
        (bundle / "data").rename(outside)
        (bundle / "data").symlink_to(outside, target_is_directory=True)

        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence(evidence_id)

        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_query_rejects_symlinked_inventory_prefix(tmp_path: Path) -> None:
    runtime = AnalysisRuntime(tmp_path)
    try:
        runtime.repository.initialize()
        outside = tmp_path / "outside-inventory"
        outside.mkdir()
        evidence_root = tmp_path / ".flameox" / "evidence" / "sha256"
        (evidence_root / "aa").symlink_to(outside, target_is_directory=True)

        with pytest.raises(RuntimeFailure) as failure:
            runtime.query_evidence()

        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_repository_rejects_extra_metadata_and_missing_native_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(result["analysis_id"])
        evidence_id = preserved["evidence_id"]
        repository_metadata = tmp_path / ".flameox" / "repository.json"
        metadata = json.loads(repository_metadata.read_text())
        repository_metadata.write_text(json.dumps({**metadata, "mutable_head": "forbidden"}))

        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence(evidence_id)
        assert failure.value.code == "REPOSITORY_CORRUPTION"

        repository_metadata.write_text(json.dumps(metadata))
        manifest = runtime.read_evidence(evidence_id)
        digest = manifest["body"]["artifacts"][0]["sha256"]
        payload = tmp_path / ".flameox" / "artifacts" / "sha256" / digest[:2] / digest / "payload"
        payload.unlink()

        with pytest.raises(RuntimeFailure) as failure:
            runtime.read_evidence(evidence_id)
        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_repository_rejects_self_consistent_manifest_with_invalid_body_shape(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        preserved = runtime.preserve_evidence(result["analysis_id"])
        evidence_id = preserved["evidence_id"]
        bundle = tmp_path / ".flameox" / "evidence" / "sha256" / evidence_id[:2] / evidence_id
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["body"]["inputs"] = ["not-an-input"]
        malformed_id = hashlib.sha256(canonical_bytes(manifest["body"])).hexdigest()
        manifest["evidence_id"] = malformed_id
        malformed_bundle = bundle.parent.parent / malformed_id[:2] / malformed_id
        malformed_bundle.parent.mkdir()
        bundle.rename(malformed_bundle)
        (malformed_bundle / "manifest.json").write_bytes(canonical_bytes(manifest))

        with pytest.raises(RuntimeFailure) as failure:
            runtime.query_evidence()

        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_repeated_preservation_revalidates_bundle_and_returns_defensive_reference(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        first = runtime.preserve_evidence(result["analysis_id"])
        evidence_id = first["evidence_id"]
        first["evidence_id"] = "0" * 64

        assert runtime.preserve_evidence(result["analysis_id"])["evidence_id"] == evidence_id

        bundle = tmp_path / ".flameox" / "evidence" / "sha256" / evidence_id[:2] / evidence_id
        (bundle / "data" / "analysis.json").write_text("corrupt")
        with pytest.raises(RuntimeFailure) as failure:
            runtime.preserve_evidence(result["analysis_id"])

        assert failure.value.code == "REPOSITORY_CORRUPTION"
    finally:
        runtime.close()


@pytest.mark.integration
def test_query_pagination_is_deterministic_and_inventory_bound(tmp_path: Path) -> None:
    runtime = AnalysisRuntime(tmp_path)
    try:
        for index in range(3):
            artifact = tmp_path / f"samples-{index}.json"
            artifact.write_text(json.dumps([{"value": index}]))
            result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
            runtime.preserve_evidence(result["analysis_id"])

        first = runtime.query_evidence(limit=2)
        second = runtime.query_evidence(limit=2, cursor=first["continuation"])
        ids = [item["evidence_id"] for item in first["evidence"] + second["evidence"]]
        assert ids == sorted(ids)
        assert len(ids) == 3
        assert second["continuation"] is None

        earliest = runtime.read_evidence(ids[0])
        input_digest = earliest["body"]["inputs"][0]["sha256"]
        filtered = runtime.query_evidence(input_sha256=input_digest, limit=1)
        assert [item["evidence_id"] for item in filtered["evidence"]] == [ids[0]]
        assert filtered["continuation"] is None
    finally:
        runtime.close()


@pytest.mark.integration
def test_first_preservation_updates_only_repository_local_git_exclude(tmp_path: Path) -> None:
    git_info = tmp_path / ".git" / "info"
    git_info.mkdir(parents=True)
    exclude = git_info / "exclude"
    exclude.write_text("existing-pattern\n")
    tracked_ignore = tmp_path / ".gitignore"
    tracked_ignore.write_text("tracked-pattern\n")
    artifact = tmp_path / "samples.json"
    artifact.write_text("[]")
    runtime = AnalysisRuntime(tmp_path)
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        runtime.preserve_evidence(result["analysis_id"])
        runtime.preserve_evidence(result["analysis_id"])
    finally:
        runtime.close()

    assert exclude.read_text().splitlines() == ["existing-pattern", ".flameox/"]
    assert tracked_ignore.read_text() == "tracked-pattern\n"


@pytest.mark.integration
def test_first_preservation_handles_git_worktree_indirection(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    common = tmp_path / "repository.git"
    worktree_git = common / "worktrees" / "project"
    worktree_git.mkdir(parents=True)
    (worktree_git / "commondir").write_text("../..\n")
    (project / ".git").write_text(f"gitdir: {worktree_git}\n")
    artifact = project / "samples.json"
    artifact.write_text("[]")

    runtime = AnalysisRuntime(project)
    try:
        result = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
        runtime.preserve_evidence(result["analysis_id"])
    finally:
        runtime.close()

    assert (common / "info" / "exclude").read_text() == ".flameox/\n"
