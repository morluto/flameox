from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

import pytest

from flameox.runtime_contracts import (
    CaptureTarget,
    EvidenceSource,
    PathSource,
    RuntimeFailure,
)
from flameox.stateless import AnalysisRuntime


def _bundle(root: Path, samples: list[float], *, elements: int = 65_536) -> Path:
    root.mkdir()
    sidecar = root / "results.json-bin" / "0.bin"
    sidecar.parent.mkdir()
    sidecar.write_bytes(struct.pack(f"<{len(samples)}f", *samples))
    (root / "results.json").write_text(
        json.dumps(
            {
                "meta": {
                    "version": {
                        "json": {"major": 1, "minor": 0, "patch": 0},
                        "nvbench": {"major": 0, "minor": 1, "patch": 0, "string": "0.1.0"},
                    }
                },
                "benchmarks": [
                    {
                        "name": "cub.scan",
                        "states": [
                            {
                                "name": f"elements={elements}",
                                "device": 0,
                                "is_skipped": False,
                                "summaries": [
                                    {
                                        "tag": "nv/json/bin:sample_times",
                                        "hint": "file/sample_times",
                                        "data": [
                                            {
                                                "name": "filename",
                                                "type": "string",
                                                "value": "results.json-bin/0.bin",
                                            },
                                            {
                                                "name": "size",
                                                "type": "int64",
                                                "value": str(len(samples)),
                                            },
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    )
    return root


def test_nvbench_directory_preserves_native_sample_values_and_compares(tmp_path: Path) -> None:
    baseline = _bundle(tmp_path / "baseline", [0.004, 0.006])
    candidate = _bundle(tmp_path / "candidate", [0.002, 0.003])
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        summary = runtime.analyze(
            "benchmark.summary",
            [PathSource(path=str(baseline), format="nvbench")],
            {},
        )
        comparison = runtime.analyze(
            "benchmark.compare",
            [
                PathSource(path=str(baseline), format="nvbench"),
                PathSource(path=str(candidate), format="nvbench"),
            ],
            {"metric": "cub.scan.sample_times"},
        )
        preserved = runtime.preserve_evidence(summary["analysis_id"])
        manifest = runtime.read_evidence(preserved["evidence_id"])
    finally:
        runtime.close()

    assert summary["provider"] == {"id": "nvbench", "version": "0.1.0"}
    assert summary["blocks"][1]["rows"][0]["value_float"] == pytest.approx(0.004)
    assert comparison["blocks"][1]["rows"][0]["ratio"] == pytest.approx(0.5)
    assert {item["role"] for item in manifest["body"]["artifacts"]} == {
        "input:results.json",
        "input:results.json-bin/0.bin",
    }


def test_nvbench_compare_does_not_pair_different_states(tmp_path: Path) -> None:
    baseline = _bundle(tmp_path / "baseline", [1.0], elements=16)
    candidate = _bundle(tmp_path / "candidate", [2.0], elements=1_024)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "benchmark.compare",
            [
                PathSource(path=str(baseline), format="nvbench"),
                PathSource(path=str(candidate), format="nvbench"),
            ],
            {"metric": "cub.scan.sample_times"},
        )
    finally:
        runtime.close()

    assert result["blocks"][1]["rows"] == []
    assert result["blocks"][0]["values"] == {
        "input_count": 2,
        "compatible_metric_count": 0,
        "unmatched_identity_count": 2,
    }


def test_nvbench_scaling_uses_numeric_state_dimensions(tmp_path: Path) -> None:
    sources = [
        PathSource(
            path=str(_bundle(tmp_path / f"size-{elements}", [duration], elements=elements)),
            format="nvbench",
        )
        for elements, duration in ((10, 1.0), (20, 4.0), (40, 16.0))
    ]
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "benchmark.scaling",
            sources,
            {"input_dimension": "elements", "metric": "cub.scan.sample_times"},
        )
    finally:
        runtime.close()

    row = result["blocks"][1]["rows"][0]
    assert row["status"] == "estimated"
    assert row["point_count"] == 3
    assert row["exponent"] == pytest.approx(2.0)


def test_nvbench_scaling_aggregates_beyond_the_sample_row_ceiling(tmp_path: Path) -> None:
    sources = [
        PathSource(
            path=str(
                _bundle(
                    tmp_path / f"size-{elements}",
                    [duration] * 1_002,
                    elements=elements,
                )
            ),
            format="nvbench",
        )
        for elements, duration in ((10, 1.0), (20, 4.0))
    ]
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "benchmark.scaling",
            sources,
            {"input_dimension": "elements", "metric": "cub.scan.sample_times"},
        )
    finally:
        runtime.close()

    row = result["blocks"][1]["rows"][0]
    assert row["point_count"] == 2
    assert row["exponent"] == pytest.approx(2.0)


def test_nvbench_rejects_unbound_sidecars_and_nonfinite_samples(tmp_path: Path) -> None:
    standalone = tmp_path / "results.json"
    standalone.write_text("{}")
    invalid = _bundle(tmp_path / "invalid", [float("nan")])
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        with pytest.raises(RuntimeFailure) as unbound:
            runtime.analyze(
                "benchmark.summary",
                [PathSource(path=str(standalone), format="nvbench")],
                {},
            )
        with pytest.raises(RuntimeFailure) as malformed:
            runtime.analyze(
                "benchmark.summary",
                [PathSource(path=str(invalid), format="nvbench")],
                {},
            )
    finally:
        runtime.close()

    assert unbound.value.code == "UNSUPPORTED_FORMAT"
    assert malformed.value.code == "DECODE_FAILURE"


def test_nvbench_capture_analyzes_and_preserves_the_json_bin_directory(tmp_path: Path) -> None:
    executable = tmp_path / "nvbench-fixture"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import struct
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--jsonbin") + 1])
sidecar = output.parent / f"{output.name}-bin" / "0.bin"
sidecar.parent.mkdir()
sidecar.write_bytes(struct.pack("<2f", 0.004, 0.006))
output.write_text(json.dumps({
    "meta": {"version": {
        "json": {"major": 1, "minor": 0, "patch": 0},
        "nvbench": {"major": 0, "minor": 1, "patch": 0, "string": "0.1.0"}
    }},
    "benchmarks": [{"name": "cub.scan", "states": [{
        "name": "elements=65536", "device": 0, "is_skipped": False,
        "summaries": [{
            "tag": "nv/json/bin:sample_times", "hint": "file/sample_times",
            "data": [
                {"name": "filename", "type": "string", "value": "results.json-bin/0.bin"},
                {"name": "size", "type": "int64", "value": "2"}
            ]
        }]
    }]}]
}))
"""
    )
    executable.chmod(0o755)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = asyncio.run(
            runtime.capture_and_analyze(
                CaptureTarget(argv=[str(executable)], cwd=str(tmp_path), provider_id="nvbench"),
                "benchmark.summary",
            )
        )
        preserved = runtime.preserve_evidence(result["analysis_id"])
        manifest = runtime.read_evidence(preserved["evidence_id"])
        reanalyzed = runtime.analyze(
            "benchmark.summary",
            [EvidenceSource(kind="evidence", evidence_id=preserved["evidence_id"])],
            {},
        )
    finally:
        runtime.close()

    assert result["provider"] == {"id": "nvbench", "version": "0.1.0"}
    assert result["capture"]["executions"][0]["capture_argv"][-2] == "--jsonbin"
    assert result["blocks"][1]["rows"][1]["value_float"] == pytest.approx(0.006)
    assert reanalyzed["provider"] == {"id": "nvbench", "version": "0.1.0"}
    assert {item["role"] for item in manifest["body"]["artifacts"]} == {
        "capture-0001/stdout",
        "capture-0001/stderr",
        "capture-0001/benchmark:results.json",
        "capture-0001/benchmark:results.json-bin/0.bin",
    }
