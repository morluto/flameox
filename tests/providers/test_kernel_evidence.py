from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.runtime_contracts import PathSource, RequestLimits, RuntimeFailure
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


def test_kernel_compare_requires_complete_semantic_identity(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_kernel_document(0.0001)))
    changed = _kernel_document(0.0002)
    changed_output = changed["cases"][0]["outputs"][0]  # type: ignore[index]
    changed_output["shape"] = [1_024]
    changed_output["dtype"] = "int8"
    changed_output["metrics"][0]["unit"] = "percent"
    candidate.write_text(json.dumps(changed))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "kernel.compare",
            [
                PathSource(path=str(baseline), format="kernel-validation"),
                PathSource(path=str(candidate), format="kernel-validation"),
            ],
            {"metric": "max_abs_error"},
        )
    finally:
        runtime.close()

    assert result["blocks"][1]["rows"] == []
    assert result["blocks"][0]["values"] == {
        "status": "consistent",
        "input_count": 2,
        "compatible_metric_count": 0,
        "unmatched_identity_count": 2,
        "consistency_failure_count": 0,
        "consistency_failures": [],
    }


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


def test_kernel_validation_rejects_coerced_coverage_and_duplicate_metrics(
    tmp_path: Path,
) -> None:
    coverage = _kernel_document(0.0)
    coverage["coverage_complete"] = "false"
    duplicate = _kernel_document(0.0)
    duplicate_case = duplicate["cases"][0]  # type: ignore[index]
    duplicate_output = duplicate_case["outputs"][0]
    duplicate_output["metrics"].append(dict(duplicate_output["metrics"][0]))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        for name, document in (("coverage", coverage), ("duplicate", duplicate)):
            artifact = tmp_path / f"{name}.json"
            artifact.write_text(json.dumps(document))
            with pytest.raises(RuntimeFailure) as failure:
                runtime.analyze(
                    "kernel.validation",
                    [PathSource(path=str(artifact), format="kernel-validation")],
                    {},
                )
            assert failure.value.code == "DECODE_FAILURE"
    finally:
        runtime.close()


def test_kernel_validation_preserves_outputless_cases(tmp_path: Path) -> None:
    artifact = tmp_path / "outputless.json"
    document = _kernel_document(0.0, status="fail")
    case = document["cases"][0]  # type: ignore[index]
    case["status"] = "unsupported"
    case["outputs"] = []
    artifact.write_text(json.dumps(document))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "kernel.validation",
            [PathSource(path=str(artifact), format="kernel-validation")],
            {},
        )
    finally:
        runtime.close()

    assert result["coverage"] == {"rows_returned": 1, "rows_observed": 1, "complete": True}
    assert result["blocks"][1]["rows"] == [
        {
            "evidence_kind": "case",
            "case_id": "square-fp32-128",
            "case_status": "unsupported",
            "dimensions": {"size": 128},
            "seed": 42,
            "device": "cuda:0-sm86",
        }
    ]


def test_kernel_validation_defaults_omitted_coverage_to_incomplete(tmp_path: Path) -> None:
    artifact = tmp_path / "validation.json"
    document = _kernel_document(0.0)
    del document["coverage_complete"]
    artifact.write_text(json.dumps(document))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "kernel.validation",
            [PathSource(path=str(artifact), format="kernel-validation")],
            {},
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"]["coverage_complete"] is False


def test_kernel_validation_marks_producer_contradictions_inconclusive(tmp_path: Path) -> None:
    artifact = tmp_path / "contradictory.json"
    document = _kernel_document(10.0)
    artifact.write_text(json.dumps(document))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "kernel.validation",
            [PathSource(path=str(artifact), format="kernel-validation")],
            {},
        )
    finally:
        runtime.close()

    metrics = result["blocks"][0]["values"]
    assert metrics["status"] == "inconclusive"
    assert metrics["producer_status"] == "pass"
    assert metrics["consistency_failure_count"] == 1
    assert metrics["consistency_failures"][0]["rule"] == "numeric_comparator"

    comparison = AnalysisRuntime(evidence_directory=tmp_path / "comparison-store")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_kernel_document(0.0)))
    try:
        compared = comparison.analyze(
            "kernel.compare",
            [
                PathSource(path=str(baseline), format="kernel-validation"),
                PathSource(path=str(artifact), format="kernel-validation"),
            ],
            {},
        )
    finally:
        comparison.close()
    compared_metrics = compared["blocks"][0]["values"]
    assert compared_metrics["status"] == "inconclusive"
    assert compared_metrics["consistency_failure_count"] == 1
    assert compared_metrics["consistency_failures"][0]["input_index"] == 1


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


def test_native_triton_cache_preserves_quantiles_and_derives_lexicographic_winner(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "scatter.autotune.json"
    artifact.write_text(
        json.dumps(
            {
                "key": [256, "torch.bfloat16"],
                "configs_timings": [
                    [{"kwargs": {"BLOCK_D": 128}}, [1, 0.9, 9]],
                    [{"kwargs": {"BLOCK_D": 256}}, [2, 0.1, 2]],
                    [{"kwargs": {"BLOCK_D": 512}}, [float("inf")] * 3],
                ],
            }
        )
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    try:
        result = runtime.analyze("triton.autotune", [PathSource(path=str(artifact))], {})
    finally:
        runtime.close()
    rows = result["blocks"][1]["rows"]
    assert result["provider"]["id"] == "triton-autotune-cache"
    assert result["blocks"][0]["values"]["derived_best_config_id"] == rows[0]["config_id"]
    assert rows[0]["timing_values"] == [1, 0.9, 9]
    assert rows[2]["timing_values"] == ["positive_infinity"] * 3
    assert "mean_ms" not in rows[0]
    assert result["coverage"] == {"rows_observed": 3, "rows_returned": 3, "complete": True}


@pytest.mark.parametrize("timing", [-1, float("nan"), True, []])
def test_native_triton_cache_rejects_invalid_timings(tmp_path: Path, timing: object) -> None:
    artifact = tmp_path / "invalid.autotune.json"
    artifact.write_text(json.dumps({"key": [1], "configs_timings": [[{"kwargs": {}}, timing]]}))
    runtime = AnalysisRuntime()
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("triton.autotune", [PathSource(path=str(artifact))], {})
        assert failure.value.code == "DECODE_FAILURE"
    finally:
        runtime.close()


@pytest.mark.parametrize("smaller", [2**53, 10**400])
def test_native_triton_cache_preserves_integer_selection_order(
    tmp_path: Path, smaller: int
) -> None:
    artifact = tmp_path / "integer.autotune.json"
    artifact.write_text(
        json.dumps(
            {
                "key": [1],
                "configs_timings": [
                    [{"kwargs": {"BLOCK": 128}}, smaller + 1],
                    [{"kwargs": {"BLOCK": 256}}, smaller],
                ],
            }
        )
    )
    runtime = AnalysisRuntime()
    try:
        result = runtime.analyze("triton.autotune", [PathSource(path=str(artifact))], {})
    finally:
        runtime.close()
    rows = result["blocks"][1]["rows"]
    assert rows[0]["timing_values"] == [str(smaller + 1)]
    assert rows[1]["timing_values"] == [str(smaller)]
    assert result["blocks"][0]["values"]["derived_best_config_id"] == rows[1]["config_id"]


@pytest.mark.parametrize("max_rows", [1, 10])
def test_triton_cache_bundle_keeps_distinct_native_paths_and_global_counts(
    tmp_path: Path, max_rows: int
) -> None:
    bundle = tmp_path / "cache"
    bundle.mkdir()
    for name in ("first", "second"):
        member = bundle / name
        member.mkdir()
        (member / "scatter.autotune.json").write_text(
            json.dumps(
                {"key": [128], "configs_timings": [[{"kwargs": {"BLOCK": 128}}, [1, 0.5, 2]]]}
            )
        )
    (bundle / "compiled.cubin").write_bytes(b"not an analysis record")
    runtime = AnalysisRuntime()
    try:
        result = runtime.analyze(
            "triton.autotune",
            [PathSource(path=str(bundle), format="triton-cache")],
            {},
            limits=RequestLimits(max_rows=max_rows),
        )
    finally:
        runtime.close()
    assert result["blocks"][0]["values"] == {"cache_count": 2, "candidate_count": 2}
    rows = result["blocks"][1]["rows"]
    assert rows[0]["cache_path"] == "first/scatter.autotune.json"
    assert rows[0]["derived_best"] is True
    assert result["coverage"]["rows_observed"] == 2
    assert result["coverage"]["complete"] == (max_rows >= 2)
    if max_rows >= 2:
        assert rows[1]["cache_path"] == "second/scatter.autotune.json"


def test_triton_compilation_without_autotuning_is_not_complete_negative_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "compiled.cubin").write_bytes(b"compiled")
    runtime = AnalysisRuntime()
    try:
        with pytest.raises(RuntimeFailure, match="No native Triton autotune caches"):
            runtime.analyze(
                "triton.autotune", [PathSource(path=str(tmp_path), format="triton-cache")], {}
            )
    finally:
        runtime.close()
