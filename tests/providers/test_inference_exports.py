from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.runtime_contracts import PathSource, RequestLimits, RuntimeFailure
from flameox.stateless import AnalysisRuntime


def _vllm_payload(*, throughput: float = 5.0) -> dict[str, object]:
    return {
        "metrics": {
            "completed": 2,
            "total_input": 20,
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
    baseline.write_text(json.dumps(_vllm_payload(throughput=5.0)))
    candidate.write_text(json.dumps(_vllm_payload(throughput=10.0)))
    runtime = AnalysisRuntime(tmp_path)
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
    assert "must not escape" not in json.dumps(summary)
    assert "private endpoint" not in json.dumps(summary)
    assert comparison["blocks"][1]["rows"][0]["ratio"] == 2.0


def test_sglang_rejects_detailed_output_and_projects_scalars(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate.jsonl"
    aggregate.write_text(
        json.dumps(
            {
                "duration": 1.0,
                "completed": 2,
                "total_input_tokens": 8,
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
    runtime = AnalysisRuntime(tmp_path)
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
    assert failure.value.code == "DECODE_FAILURE"


def test_mooncake_trace_is_streamed_without_sensitive_payloads(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "timestamp": 0,
                    "input_length": 10,
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
    runtime = AnalysisRuntime(tmp_path)
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
    runtime = AnalysisRuntime(tmp_path)
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
