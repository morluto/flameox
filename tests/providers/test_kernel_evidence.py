from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import PathSource, RequestLimits, RuntimeFailure


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


@pytest.mark.parametrize("seed", [42, 2**63 + 42])
def test_kernel_validation_summary_and_comparison_use_typed_rows(tmp_path: Path, seed: int) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    for path, value in ((baseline, 0.0001), (candidate, 0.0002)):
        document = _kernel_document(value)
        document["cases"][0]["seed"] = seed  # type: ignore[index]
        path.write_text(json.dumps(document))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        summary = runtime.analyze(
            "inspect_kernel_validation",
            [PathSource(path=str(baseline), format="kernel-validation")],
            {},
        )
        comparison = runtime.analyze(
            "compare_kernel_validation",
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
    assert comparison["blocks"][1]["rows"][0]["seed"] == (seed if seed == 42 else str(seed))
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.parametrize("baseline_value", [1e-308, -1e308])
def test_kernel_comparison_rejects_unrepresentable_derived_values(
    tmp_path: Path, baseline_value: float
) -> None:
    sources = []
    for name, value in (("baseline", baseline_value), ("candidate", 1e308)):
        artifact = tmp_path / f"{name}.json"
        artifact.write_text(json.dumps(_kernel_document(value)))
        sources.append(PathSource(path=str(artifact), format="kernel-validation"))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("compare_kernel_validation", sources, {})
        assert failure.value.code == "LIMIT_EXCEEDED"
    finally:
        runtime.close()


@pytest.mark.parametrize("difference", ["output", "wide_integer_type", "reserved_integer_tag"])
def test_kernel_compare_requires_complete_semantic_identity(
    tmp_path: Path, difference: str
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    original = _kernel_document(0.0001)
    changed = _kernel_document(0.0002)
    if difference == "output":
        changed_output = changed["cases"][0]["outputs"][0]  # type: ignore[index]
        changed_output["shape"] = [1_024]
        changed_output["dtype"] = "int8"
        changed_output["metrics"][0]["unit"] = "percent"
    else:
        wide_integer = 2**63 + 42
        original["cases"][0]["dimensions"] = {"size": wide_integer}  # type: ignore[index]
        distinct: object = (
            str(wide_integer)
            if difference == "wide_integer_type"
            else {"$flameox.integer": str(wide_integer)}
        )
        changed["cases"][0]["dimensions"] = {"size": distinct}  # type: ignore[index]
    baseline.write_text(json.dumps(original))
    candidate.write_text(json.dumps(changed))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "compare_kernel_validation",
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
                "inspect_kernel_validation",
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
    invalid_status = _kernel_document(0.0)
    invalid_status["status"] = []
    invalid_case = _kernel_document(0.0)
    invalid_case["cases"][0]["status"] = {}  # type: ignore[index]
    invalid_comparator = _kernel_document(0.0)
    invalid_comparator["cases"][0]["outputs"][0]["metrics"][0]["comparator"] = []  # type: ignore[index]
    overflow = _kernel_document(10**400)
    null_outputs = _kernel_document(0.0)
    null_outputs["cases"][0]["outputs"] = None  # type: ignore[index]
    null_metrics = _kernel_document(0.0)
    null_metrics["cases"][0]["outputs"][0]["metrics"] = None  # type: ignore[index]
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        for name, document, code in (
            ("coverage", coverage, "DECODE_FAILURE"),
            ("duplicate", duplicate, "DECODE_FAILURE"),
            ("status", invalid_status, "DECODE_FAILURE"),
            ("case", invalid_case, "DECODE_FAILURE"),
            ("comparator", invalid_comparator, "DECODE_FAILURE"),
            ("overflow", overflow, "DECODE_FAILURE"),
            ("null_outputs", null_outputs, "LIMIT_EXCEEDED"),
            ("null_metrics", null_metrics, "LIMIT_EXCEEDED"),
        ):
            artifact = tmp_path / f"{name}.json"
            artifact.write_text(json.dumps(document))
            source = PathSource(path=str(artifact), format="kernel-validation")
            for operation, sources in (
                ("inspect_kernel_validation", [source]),
                ("compare_kernel_validation", [source, source]),
            ):
                with pytest.raises(RuntimeFailure) as failure:
                    runtime.analyze(operation, sources, {})
                assert failure.value.code == code
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
            "inspect_kernel_validation",
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
            "inspect_kernel_validation",
            [PathSource(path=str(artifact), format="kernel-validation")],
            {},
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"]["coverage_complete"] is False


@pytest.mark.parametrize("kind", ["finite", "infinite_pass", "infinite_fail"])
def test_kernel_validation_marks_producer_contradictions_inconclusive(
    tmp_path: Path, kind: str
) -> None:
    artifact = tmp_path / "contradictory.json"
    document = _kernel_document(10.0, status="fail" if kind == "infinite_fail" else "pass")
    if kind != "finite":
        native: Any = document
        metric = native["cases"][0]["outputs"][0]["metrics"][0]
        metric["name"] = "psnr"
        metric["value"] = {"kind": "positive_infinity", "reason": "zero_mse_exact_agreement"}
        metric["comparator"] = ">=" if kind == "infinite_fail" else "<="
        metric["threshold"] = 30
    artifact.write_text(json.dumps(document))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "inspect_kernel_validation",
            [PathSource(path=str(artifact), format="kernel-validation")],
            {},
        )
    finally:
        runtime.close()

    metrics = result["blocks"][0]["values"]
    assert metrics["status"] == "inconclusive"
    assert metrics["producer_status"] == ("fail" if kind == "infinite_fail" else "pass")
    assert metrics["consistency_failure_count"] == 1
    assert metrics["consistency_failures"][0]["rule"] == "numeric_comparator"

    comparison = AnalysisRuntime(evidence_directory=tmp_path / "comparison-store")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_kernel_document(0.0)))
    try:
        compared = comparison.analyze(
            "compare_kernel_validation",
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


@pytest.mark.parametrize("block", [256, 2**63 + 42])
def test_triton_autotune_stream_reports_provider_selection(tmp_path: Path, block: int) -> None:
    artifact = tmp_path / "triton.jsonl"
    event = _triton_event()
    event["winner"]["kwargs"]["BLOCK"] = block  # type: ignore[index]
    artifact.write_text(json.dumps(event) + "\n")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "inspect_triton_autotune",
            [PathSource(path=str(artifact), format="triton")],
            {},
        )
    finally:
        runtime.close()

    assert result["provider"]["id"] == "triton-autotune"
    assert result["blocks"][0]["values"] == {"selection_count": 1, "cache_hit_count": 0}
    row = result["blocks"][1]["rows"][0]
    assert row["function_name"] == "workload.kernel"
    winner = next(
        item for item in row["candidates"] if item["config_id"] == row["winner_config_id"]
    )
    event["winner"]["kwargs"]["BLOCK"] = block if block == 256 else str(block)  # type: ignore[index]
    assert winner["config"] == event["winner"]


@pytest.mark.parametrize("timing", [1e308, 1.7976931348623157e308])
def test_triton_listener_means_remain_finite_when_timing_sums_overflow(
    tmp_path: Path, timing: float
) -> None:
    artifact = tmp_path / "triton.jsonl"
    event = _triton_event()
    event["candidates"][0]["timings_ms"] = [timing, timing]  # type: ignore[index]
    artifact.write_text(json.dumps(event) + "\n")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            "inspect_triton_autotune", [PathSource(path=str(artifact), format="triton")], {}
        )
    finally:
        runtime.close()
    assert result["blocks"][1]["rows"][0]["candidates"][0]["mean_ms"] == timing
    assert result["coverage"]["complete"]


def test_triton_listener_counts_cover_events_beyond_the_returned_population(tmp_path: Path) -> None:
    artifact = tmp_path / "triton.jsonl"
    event = {**_triton_event(), "cache_hit": True, "duration_ms": None}
    artifact.write_text((json.dumps(event) + "\n") * 1_002)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        first = runtime.analyze(
            "inspect_triton_autotune",
            [PathSource(path=str(artifact), format="triton")],
            {},
            limits=RequestLimits(max_rows=1),
        )
        second = runtime.analyze(
            "inspect_triton_autotune",
            [PathSource(path=str(artifact), format="triton")],
            {},
            limits=RequestLimits(max_rows=1),
            continuation=first["continuation"],
        )
    finally:
        runtime.close()
    assert first["blocks"][0]["values"] == {
        "selection_count": 1_002,
        "cache_hit_count": 1_002,
    }
    assert second["blocks"][0] == first["blocks"][0]
    assert first["coverage"] == {"rows_returned": 1, "rows_observed": 1_002, "complete": False}


@pytest.mark.parametrize(
    "unusable_record",
    ["{bad json}", json.dumps({"listener_unavailable": "Listener was unavailable"})],
)
def test_triton_listener_retains_valid_selections_without_claiming_complete_coverage(
    tmp_path: Path, unusable_record: str
) -> None:
    artifact = tmp_path / "triton.jsonl"
    artifact.write_text(json.dumps(_triton_event()) + "\n" + unusable_record + "\n")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            "inspect_triton_autotune", [PathSource(path=str(artifact), format="triton")], {}
        )
    finally:
        runtime.close()
    assert result["blocks"][0]["values"] == {"selection_count": 1, "cache_hit_count": 0}
    assert result["blocks"][1]["rows"][0]["function_name"] == "workload.kernel"
    assert result["coverage"] == {"rows_returned": 1, "rows_observed": 1, "complete": False}
    assert result["continuation"] is None
    assert result["limitations"]


@pytest.mark.parametrize(
    ("native", "expected_code"),
    [
        ("{bad json}\n", "DECODE_FAILURE"),
        (json.dumps({"listener_unavailable": "Listener was unavailable"}), "UNSUPPORTED_FORMAT"),
        ("", "UNSUPPORTED_FORMAT"),
    ],
)
def test_triton_listener_without_usable_selections_reports_typed_failure(
    tmp_path: Path, native: str, expected_code: str
) -> None:
    artifact = tmp_path / "triton.jsonl"
    artifact.write_text(native)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "inspect_triton_autotune", [PathSource(path=str(artifact), format="triton")], {}
            )
        assert failure.value.code == expected_code
        assert "no usable autotune selections" in failure.value.message
        assert artifact.read_text() == native
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "native",
    ["x" * (64 * 1024 + 1) + "\n", "{}\n" * 100_001],
    ids=["oversized-line", "invalid-event-ceiling"],
)
def test_triton_listener_enforces_native_limits_before_semantic_filtering(
    tmp_path: Path, native: str
) -> None:
    artifact = tmp_path / "triton.jsonl"
    artifact.write_text(native)
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "inspect_triton_autotune",
                [PathSource(path=str(artifact), format="triton")],
                {},
                limits=RequestLimits(max_rows=1),
            )
        assert failure.value.code == "LIMIT_EXCEEDED"
        assert artifact.read_text() == native
    finally:
        runtime.close()


@pytest.mark.parametrize("key_value", [256, 2**63 + 42])
def test_native_triton_cache_preserves_quantiles_and_derives_lexicographic_winner(
    tmp_path: Path,
    key_value: int,
) -> None:
    artifact = tmp_path / "scatter.autotune.json"
    artifact.write_text(
        json.dumps(
            {
                "key": [key_value, "torch.bfloat16"],
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
        result = runtime.analyze("inspect_triton_autotune", [PathSource(path=str(artifact))], {})
    finally:
        runtime.close()
    rows = result["blocks"][1]["rows"]
    assert result["provider"]["id"] == "triton-autotune-cache"
    assert result["blocks"][0]["values"]["derived_best_config_id"] == rows[0]["config_id"]
    assert rows[0]["timing_values"] == [1, 0.9, 9]
    assert rows[2]["timing_values"] == ["positive_infinity"] * 3
    assert "mean_ms" not in rows[0]
    assert result["coverage"] == {"rows_observed": 3, "rows_returned": 3, "complete": True}


def test_native_triton_cache_rejects_invalid_timing(tmp_path: Path) -> None:
    artifact = tmp_path / "invalid.autotune.json"
    artifact.write_text(
        json.dumps({"key": [1], "configs_timings": [[{"kwargs": {}}, float("nan")]]})
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze("inspect_triton_autotune", [PathSource(path=str(artifact))], {})
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
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze("inspect_triton_autotune", [PathSource(path=str(artifact))], {})
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
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        result = runtime.analyze(
            "inspect_triton_autotune",
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
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure, match="No native Triton autotune caches"):
            runtime.analyze(
                "inspect_triton_autotune",
                [PathSource(path=str(tmp_path), format="triton-cache")],
                {},
            )
    finally:
        runtime.close()
