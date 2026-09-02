from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.runtime_contracts import PathSource, RuntimeFailure
from flameox.stateless import AnalysisRuntime


def _kernel_document(value: float, *, status: str = "pass") -> dict[str, object]:
    return {
        "schema_version": "flameox.kernel-validation.v2",
        "producer": "kernel-tests",
        "producer_version": "1.0",
        "status": status,
        "coverage_complete": True,
        "cases": [
            {
                "case_id": "square-fp32-128",
                "dimensions": {"size": 128},
                "seed": 42,
                "device": "cuda:0-sm86",
                "status": status,
                "outputs": [
                    {
                        "name": "result",
                        "dtype": "float32",
                        "shape": [128, 128],
                        "status": status,
                        "metrics": [
                            {
                                "name": "max_abs_error",
                                "value": {"kind": "finite", "value": value},
                                "comparator": "<=",
                                "threshold": 0.001,
                                "unit": "absolute",
                                "status": status,
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _triton_event() -> dict[str, object]:
    first = {
        "kwargs": {"BLOCK": 128},
        "num_warps": 4,
        "num_stages": 1,
        "num_ctas": 1,
        "maxnreg": None,
        "ir_override": None,
    }
    winner = {**first, "kwargs": {"BLOCK": 256}, "num_warps": 8}
    return {
        "function_name": "workload.kernel",
        "key_digest": "sha256:" + "1" * 64,
        "cache_hit": False,
        "duration_ms": 32.0,
        "winner": winner,
        "candidates": [
            {"config": first, "timings_ms": [2.0, 1.8, 2.2]},
            {"config": winner, "timings_ms": [1.0, 0.9, 1.1]},
        ],
    }


def test_kernel_validation_summary_and_comparison_use_typed_rows(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_kernel_document(0.0001)))
    candidate.write_text(json.dumps(_kernel_document(0.0002)))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        summary = runtime.analyze(
            "kernel.validation",
            [PathSource(path=str(baseline), format="kernel-validation")],
            {},
        )
        comparison = runtime.analyze(
            "kernel.compare",
            [
                PathSource(path=str(baseline), format="kernel-validation"),
                PathSource(path=str(candidate), format="kernel-validation"),
            ],
            {"metric": "max_abs_error"},
        )
    finally:
        runtime.close()

    assert summary["provider"]["id"] == "kernel-validation"
    assert summary["blocks"][0]["values"]["status"] == "pass"
    assert summary["blocks"][1]["rows"][0]["evidence_kind"] == "measurement"
    assert comparison["blocks"][1]["rows"][0]["ratio"] == 2.0
    assert not (tmp_path / ".flameox").exists()


def test_kernel_validation_rejects_an_unknown_native_schema(tmp_path: Path) -> None:
    artifact = tmp_path / "validation.json"
    document = _kernel_document(0.0)
    document["schema_version"] = "flameox.kernel-validation.v1"
    artifact.write_text(json.dumps(document))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "kernel.validation",
                [PathSource(path=str(artifact), format="kernel-validation")],
                {},
            )
    finally:
        runtime.close()

    assert failure.value.code == "UNSUPPORTED_FORMAT"


def test_triton_autotune_stream_reports_provider_selection(tmp_path: Path) -> None:
    artifact = tmp_path / "triton.jsonl"
    artifact.write_text(json.dumps(_triton_event()) + "\n")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "triton.autotune",
            [PathSource(path=str(artifact), format="triton")],
            {},
        )
    finally:
        runtime.close()

    assert result["provider"]["id"] == "triton-autotune"
    assert result["blocks"][0]["values"] == {"selection_count": 1, "cache_hit_count": 0}
    row = result["blocks"][1]["rows"][0]
    assert row["function_name"] == "workload.kernel"
    assert row["winner_config_id"] in {item["config_id"] for item in row["candidates"]}
