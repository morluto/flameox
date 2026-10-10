from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    PathSource,
    RequestLimits,
    RuntimeFailure,
)


def _vllm_payload(*, throughput: float = 5.0, input_tokens: int = 20) -> dict[str, object]:
    return {
        "metrics": {
            "completed": 2,
            "total_input": input_tokens,
            "total_output": 8,
            "request_throughput": throughput,
            "request_goodput": throughput,
            "output_throughput": 10.0,
            "total_token_throughput": 20.0,
            "mean_ttft_ms": 2.0,
            "median_ttft_ms": 2.0,
            "std_ttft_ms": 0.1,
            "percentiles_ttft_ms": [[95.0, 3.0]],
            "mean_tpot_ms": 1.0,
            "median_tpot_ms": 1.0,
            "std_tpot_ms": 0.1,
            "percentiles_tpot_ms": [],
            "mean_itl_ms": 1.0,
            "median_itl_ms": 1.0,
            "std_itl_ms": 0.1,
            "percentiles_itl_ms": [],
            "mean_e2el_ms": 4.0,
            "median_e2el_ms": 4.0,
            "std_e2el_ms": 0.2,
            "percentiles_e2el_ms": [],
        },
        "successful_requests": 2,
        "failed_requests": 0,
        "total_requests": 2,
        "actual_duration": 1.0,
        "time_scale": 1.0,
        "raw_prompts": ["must not escape"],
        "error_log": "private endpoint",
    }


def test_vllm_summary_and_comparison_are_prompt_free(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_vllm_payload(throughput=5.0, input_tokens=2**53 + 1)))
    candidate.write_text(json.dumps(_vllm_payload(throughput=10.0, input_tokens=2**53 + 1)))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        summary = runtime.analyze(
            "inference.summary",
            [PathSource(path=str(baseline), format="vllm-benchmark")],
            {},
        )
        comparison = runtime.analyze(
            "inference.compare",
            [
                PathSource(path=str(baseline), format="vllm-benchmark"),
                PathSource(path=str(candidate), format="vllm-benchmark"),
            ],
            {"metric": "vllm.request_throughput"},
        )
    finally:
        runtime.close()

    assert summary["provider"]["id"] == "vllm-benchmark"
    assert next(
        row["value"]
        for row in summary["blocks"][1]["rows"]
        if row["name"] == "vllm.total_input_tokens"
    ) == str(2**53 + 1)
    assert "must not escape" not in json.dumps(summary)
    assert "private endpoint" not in json.dumps(summary)
    assert comparison["blocks"][1]["rows"][0]["ratio"] == 2.0
    assert comparison["blocks"][1]["rows"][0]["compatibility"] == "partial"
    assert comparison["blocks"][1]["rows"][0]["identity_unavailable"] == [
        "system.backend",
        "system.model",
        "system.tokenizer",
        "workload.dataset_name",
        "workload.max_concurrency",
        "workload.num_prompts",
        "workload.request_rate",
    ]


def test_inference_comparison_rejects_unrepresentable_derived_ratio(tmp_path: Path) -> None:
    paths = [tmp_path / "baseline.json", tmp_path / "candidate.json"]
    for path, value in zip(paths, (1e-308, 1e308), strict=True):
        path.write_text(json.dumps(_vllm_payload(throughput=value)))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "inference.compare",
                [PathSource(path=str(path), format="vllm-benchmark") for path in paths],
                {},
            )
        assert failure.value.code == "LIMIT_EXCEEDED"
        for path in paths:
            assert runtime.analyze(
                "inference.summary", [PathSource(path=str(path), format="vllm-benchmark")], {}
            )["coverage"]["complete"]
    finally:
        runtime.close()


@pytest.mark.parametrize("format_name", ["vllm-benchmark", "sglang-benchmark"])
def test_inference_rejects_unrepresentable_measured_values(
    tmp_path: Path, format_name: str
) -> None:
    artifact = tmp_path / "metrics.json"
    if format_name == "vllm-benchmark":
        payload = _vllm_payload()
        payload["metrics"]["percentiles_ttft_ms"] = [[95, 10**400]]  # type: ignore[index]
    else:
        payload = {
            "duration": 1,
            "completed": 1,
            "total_input_tokens": 2,
            "total_output_tokens": 1,
            "request_throughput": 10**400,
        }
    artifact.write_text(json.dumps(payload))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "inference.summary", [PathSource(path=str(artifact), format=format_name)], {}
            )
        assert failure.value.code == "DECODE_FAILURE"
    finally:
        runtime.close()


def test_inference_compare_rejects_known_differences_unless_explicit(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline_payload = _vllm_payload(throughput=5.0)
    baseline_payload.update({"model": "model-a", "dataset_name": "set-a"})
    candidate_payload = _vllm_payload(throughput=10.0)
    candidate_payload.update({"model": "model-b", "dataset_name": "set-b"})
    baseline.write_text(json.dumps(baseline_payload))
    candidate.write_text(json.dumps(candidate_payload))
    sources = [
        PathSource(path=str(baseline), format="vllm-benchmark"),
        PathSource(path=str(candidate), format="vllm-benchmark"),
    ]
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "inference.compare",
                sources,
                {"metric": "vllm.request_throughput"},
            )
        exploratory = runtime.analyze(
            "inference.compare",
            sources,
            {"metric": "vllm.request_throughput", "allow_heterogeneous": True},
        )
    finally:
        runtime.close()

    assert failure.value.code == "INVALID_INPUT"
    assert failure.value.details == {"differing_fields": ["system.model", "workload.dataset_name"]}
    row = exploratory["blocks"][1]["rows"][0]
    assert row["ratio"] == 2.0
    assert row["compatibility"] == "heterogeneous"
    assert set(row["identity_differences"]) == {"system.model", "workload.dataset_name"}
    assert "model-a" not in json.dumps(exploratory)
    assert "model-b" not in json.dumps(exploratory)


def test_inference_compare_treats_one_sided_optional_identity_as_partial(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline_payload = _vllm_payload(throughput=5.0)
    baseline_payload["model"] = "model-a"
    candidate_payload = _vllm_payload(throughput=10.0)
    candidate_payload.update({"model": "model-a", "backend": "ray"})
    baseline.write_text(json.dumps(baseline_payload))
    candidate.write_text(json.dumps(candidate_payload))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "inference.compare",
            [
                PathSource(path=str(baseline), format="vllm-benchmark"),
                PathSource(path=str(candidate), format="vllm-benchmark"),
            ],
            {"metric": "vllm.request_throughput"},
        )
    finally:
        runtime.close()

    row = result["blocks"][1]["rows"][0]
    assert row["compatibility"] == "partial"
    assert row["identity_differences"] == {}
    assert "system.backend" in row["identity_unavailable"]


def test_sglang_rejects_detailed_output_and_projects_scalars(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.jsonl"
    aggregate.write_text(
        json.dumps(
            {
                "duration": 1.0,
                "completed": 2,
                "total_input_tokens": 2**53 + 1,
                "total_output_tokens": 4,
                "request_throughput": 2.0,
                "p95_ttft_ms": 9.0,
                "unknown_metric": 999,
            }
        )
        + "\n"
    )
    detailed = tmp_path / "detailed.jsonl"
    detailed.write_text(
        json.dumps(
            {
                "duration": 1.0,
                "completed": 1,
                "total_input_tokens": 4,
                "total_output_tokens": 2,
                "generated_texts": ["secret"],
            }
        )
        + "\n"
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "inference.summary",
            [PathSource(path=str(aggregate), format="sglang-benchmark")],
            {},
        )
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "inference.summary",
                [PathSource(path=str(detailed), format="sglang-benchmark")],
                {},
            )
    finally:
        runtime.close()

    names = {row["name"] for row in result["blocks"][1]["rows"]}
    assert "sglang.p95_ttft_ms" in names
    assert "sglang.unknown_metric" not in names
    assert next(
        row["value"]
        for row in result["blocks"][1]["rows"]
        if row["name"] == "sglang.total_input_tokens"
    ) == str(2**53 + 1)
    assert failure.value.code == "DECODE_FAILURE"


def test_mooncake_trace_is_streamed_without_sensitive_payloads(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "timestamp": 0,
                    "input_length": 2**53 + 1,
                    "output_length": 2,
                    "hash_ids": [1, 2],
                    "messages": [{"content": "secret prompt"}],
                },
                {
                    "timestamp": 5,
                    "input_length": 20,
                    "output_length": 4,
                    "hash_ids": [3],
                },
            )
        )
        + "\n"
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "inference.summary",
            [PathSource(path=str(trace), format="mooncake-trace")],
            {},
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"]["request_count"] == 2
    assert result["blocks"][1]["rows"][0]["prefix_hash_count"] == 2
    assert result["blocks"][1]["rows"][0]["input_length"] == str(2**53 + 1)
    assert "secret prompt" not in json.dumps(result)
    assert not (tmp_path / ".flameox").exists()


def test_mooncake_summary_aggregates_beyond_returned_rows(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": index,
                    "input_length": input_length,
                    "output_length": input_length // 2,
                }
            )
            for index, input_length in enumerate((10, 20, 999))
        )
        + "\n"
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "inference.summary",
            [PathSource(path=str(trace), format="mooncake-trace")],
            {},
            limits=RequestLimits(max_rows=1),
        )
    finally:
        runtime.close()

    assert result["blocks"][0]["values"]["request_count"] == 3
    assert result["blocks"][0]["values"]["max_input_length"] == 999
    assert result["coverage"]["complete"] is False


@pytest.mark.optional
@pytest.mark.process
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_aiperf_retains_empty_metric_failures_without_inventing_counts(
    tmp_path: Path, cancelled: bool, mixed: bool
) -> None:
    pytest.importorskip("aiperf")
    export = tmp_path / "native.jsonl"
    metadata = {
        "session_num": 7,
        "request_start_ns": 125,
        "request_end_ns": 250,
        "worker_id": "worker-0",
        "record_processor_id": "processor-0",
        "benchmark_phase": "profiling",
        "was_cancelled": cancelled,
    }
    records: list[dict[str, Any]] = [
        {
            "metadata": metadata,
            "metrics": {},
            "error": None
            if cancelled
            else {"type": "TimeoutError", "code": 408, "message": "PRIVATE_ERROR_TEXT"},
            "raw_prompt": "PRIVATE_PROMPT",
        }
    ]
    if mixed:
        records.append(
            {
                "metadata": {**metadata, "session_num": 8, "was_cancelled": False},
                "metrics": {
                    "input_sequence_length": {"value": 20, "unit": "tokens"},
                    "output_sequence_length": {"value": 3, "unit": "tokens"},
                    "request_latency": {"value": 10, "unit": "ms"},
                },
                "error": None,
            }
        )
    export.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        sources = [PathSource(path=str(export), format="aiperf")]
        result = runtime.analyze("inference.summary", sources, {})
        metrics = result["blocks"][0]["values"]
        assert metrics["request_count"] == len(records)
        assert metrics["successful_requests"] == int(mixed)
        assert metrics["input_tokens"] is None
        assert metrics["output_tokens"] is None
        assert metrics["requests_missing_input_tokens"] == 1
        assert metrics["comparison_identity"] == {}
        assert set(metrics["comparison_identity_unavailable"]) == {"system", "workload"}
        row = result["blocks"][1]["rows"][0]
        assert row["outcome"] == ("cancelled" if cancelled else "failed")
        assert row["input_tokens"] is None and row["output_tokens"] is None
        assert row["latency_ns"] is None and row["tpot_ns"] is None
        assert "PRIVATE_" not in json.dumps(result)
        if mixed:
            compared = runtime.analyze("inference.compare", sources * 2, {})
            comparison = compared["blocks"][1]["rows"][0]
            assert comparison["ratio"] == 1
            assert comparison["baseline_samples"] == 1
            assert comparison["compatibility"] == "partial"
            assert "workload" in comparison["identity_unavailable"]
    finally:
        runtime.close()


@pytest.mark.optional
@pytest.mark.process
def test_aiperf_runtime_comparison_uses_prompt_free_request_metrics(tmp_path: Path) -> None:
    pytest.importorskip("aiperf")

    def write_export(path: Path, latencies_ms: tuple[int, int]) -> None:
        records = []
        for index, latency_ms in enumerate(latencies_ms):
            records.append(
                {
                    "metadata": {
                        "session_num": index + 1,
                        "x_request_id": f"request-{index}",
                        "conversation_id": f"conversation-{index}",
                        "turn_index": index,
                        "request_start_ns": 1_000 + index,
                        "request_end_ns": 1_000_000 + latency_ms * 1_000_000,
                        "worker_id": "worker-0",
                        "record_processor_id": "processor-0",
                        "benchmark_phase": "profiling",
                        "was_cancelled": False,
                    },
                    "metrics": {
                        "input_sequence_length": {"value": 20, "unit": "tokens"},
                        "output_sequence_length": {"value": 3, "unit": "tokens"},
                        "time_to_first_token": {"value": 2, "unit": "ms"},
                        "request_latency": {"value": latency_ms, "unit": "ms"},
                    },
                    "error": None,
                    "raw_prompt": "must never leave the isolated reader",
                }
            )
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    baseline = tmp_path / "baseline.aiperf.jsonl"
    candidate = tmp_path / "candidate.aiperf.jsonl"
    write_export(baseline, (10, 14))
    write_export(candidate, (5, 7))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "inference.compare",
            [
                PathSource(path=str(baseline), format="aiperf", producer="aiperf"),
                PathSource(path=str(candidate), format="aiperf", producer="aiperf"),
            ],
            {"metric": "latency_ns"},
        )
    finally:
        runtime.close()

    assert result["provider"]["id"] == "aiperf"
    assert result["blocks"][1]["rows"] == [
        {
            "metric": "latency_ns",
            "baseline_index": 0,
            "candidate_index": 1,
            "baseline_mean": 12_000_000,
            "candidate_mean": 6_000_000,
            "ratio": 0.5,
            "baseline_samples": 2,
            "candidate_samples": 2,
            "compatibility": "partial",
            "identity_differences": {},
            "identity_unavailable": ["system"],
        }
    ]
    assert "must never leave the isolated reader" not in json.dumps(result)
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.process
def test_aiperf_analysis_reports_missing_optional_package_as_unavailable(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("aiperf") is not None:
        pytest.skip("AIPerf is installed; its successful analysis path is covered separately")

    export = tmp_path / "missing-reader.aiperf.jsonl"
    export.write_text(
        json.dumps(
            {
                "metadata": {"session_num": 1, "request_start_ns": 1},
                "metrics": {
                    "input_sequence_length": {"value": 20, "unit": "tokens"},
                    "output_sequence_length": {"value": 3, "unit": "tokens"},
                    "request_latency": {"value": 10, "unit": "ms"},
                },
                "error": None,
            }
        )
        + "\n"
    )
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "inference.summary",
                [PathSource(path=str(export), format="aiperf", producer="aiperf")],
                {},
            )
    finally:
        runtime.close()

    assert failure.value.code == "UNAVAILABLE_CAPABILITY"
    assert "uv sync --extra inference" in failure.value.message


@pytest.mark.optional
@pytest.mark.process
@pytest.mark.parametrize("mode", ["exact", "scaled_overflow", "giant_tokens", "giant_duration"])
def test_aiperf_export_is_projected_without_prompts_or_repository(
    tmp_path: Path, mode: str
) -> None:
    pytest.importorskip("aiperf")
    export = tmp_path / "profile_export.jsonl"
    payload: dict[str, Any] = {
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
    metrics = payload["metrics"]
    if mode == "exact":
        metrics["input_sequence_length"]["value"] = 2**53 + 1
        metrics["output_sequence_length"]["value"] = 2
        metrics["request_latency"] = {"value": 2**53 + 1, "unit": "ns"}
    elif mode == "scaled_overflow":
        metrics["request_latency"] = {"value": 1e308, "unit": "s"}
    elif mode == "giant_tokens":
        metrics["input_sequence_length"]["value"] = 10**400
    else:
        metrics["request_latency"] = {"value": 10**400, "unit": "ns"}
    export.write_text(json.dumps(payload) + "\n")
    native = export.read_bytes()
    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        if mode in {"scaled_overflow", "giant_duration"}:
            with pytest.raises(RuntimeFailure) as failure:
                runtime.analyze(
                    "inference.summary", [PathSource(path=str(export), format="aiperf")], {}
                )
            assert failure.value.code == "DECODE_FAILURE"
            assert export.read_bytes() == native
            assert not (tmp_path / ".flameox").exists()
            return
        result = runtime.analyze(
            "inference.summary",
            [PathSource(path=str(export), format="aiperf", producer="aiperf")],
            {},
        )
        if mode == "giant_tokens":
            with pytest.raises(RuntimeFailure) as failure:
                runtime.analyze(
                    "inference.compare",
                    [PathSource(path=str(export), format="aiperf")] * 2,
                    {"metric": "input_tokens"},
                )
            assert failure.value.code == "LIMIT_EXCEEDED"
    finally:
        runtime.close()

    row = result["blocks"][1]["rows"][0]
    assert row["input_tokens"] == str(2**53 + 1 if mode == "exact" else 10**400)
    if mode == "exact":
        assert row["latency_ns"] == str(2**53 + 1)
        assert row["tpot_ns"] == 2**53 + 1 - 2_000_000
    assert export.read_bytes() == native
    assert result["provider"]["id"] == "aiperf"
    assert result["blocks"][0]["values"]["median_ttft_ns"] == 2_000_000
    assert result["blocks"][1]["rows"][0]["source_request_id"] == "conversation-a:2"
    assert "raw_prompt" not in json.dumps(result)
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.parametrize("format_name", ["vllm-benchmark", "sglang-benchmark"])
def test_inference_rejects_invalid_unicode_identity_without_leaking_native_text(
    tmp_path: Path, format_name: str
) -> None:
    payload: dict[str, object] = (
        _vllm_payload()
        if format_name == "vllm-benchmark"
        else {"duration": 1, "completed": 1, "total_input_tokens": 1, "total_output_tokens": 1}
    )
    payload["model"] = "\ud800"
    artifact = tmp_path / "native.json"
    artifact.write_text(json.dumps(payload))
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "inference.summary", [PathSource(path=str(artifact), format=format_name)], {}
            )
        assert failure.value.code == "DECODE_FAILURE"
    finally:
        runtime.close()


@pytest.mark.process
@pytest.mark.parametrize("format_name,line_limit", [("mooncake", 65_536), ("aiperf", 1_048_576)])
@pytest.mark.parametrize("prefix", ["", "\n", " "])
def test_inference_line_bounds_apply_before_whitespace_skipping(
    tmp_path: Path, format_name: str, line_limit: int, prefix: str
) -> None:
    if format_name == "aiperf":
        pytest.importorskip("aiperf")
        native = {
            "metadata": {"session_num": 0, "was_cancelled": False},
            "metrics": {},
            "error": {"type": "TimeoutError", "message": "timeout"},
        }
    else:
        native = {"timestamp": 0, "input_length": 1, "output_length": 1}
    artifact = tmp_path / "native.jsonl"
    artifact.write_text(prefix + " " * (line_limit + 1) + json.dumps(native) + "\n")
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        with pytest.raises(RuntimeFailure) as failure:
            runtime.analyze(
                "inference.summary",
                [
                    PathSource(
                        path=str(artifact),
                        format="aiperf" if format_name == "aiperf" else "mooncake-trace",
                    )
                ],
                {},
            )
        assert failure.value.code == "DECODE_FAILURE"
    finally:
        runtime.close()
