from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters.inference import SglangResultParser
from flameox.domain import DomainError

pytestmark = pytest.mark.unit


def test_sglang_result_parser_rejects_sensitive_detailed_output(tmp_path: Path) -> None:
    result_path = tmp_path / "result.jsonl"
    result_path.write_text(
        json.dumps(
            {
                "duration": 1.0,
                "completed": 1,
                "total_input_tokens": 4,
                "total_output_tokens": 2,
                "request_throughput": 1.0,
                "generated_texts": ["secret"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DomainError, match="aggregate SGLang JSONL"):
        SglangResultParser().parse(result_path)


def test_sglang_result_parser_emits_safe_optional_scalar_metrics(tmp_path: Path) -> None:
    result_path = tmp_path / "result.jsonl"
    result_path.write_text(
        json.dumps(
            {
                "duration": 1.0,
                "completed": 2,
                "total_input_tokens": 8,
                "total_output_tokens": 4,
                "request_throughput": 2.0,
                "input_throughput": 8.0,
                "output_throughput": 4.0,
                "total_throughput": 12.0,
                "accept_length": 3.5,
                "concurrency": 1.5,
                "mean_ttft_ms": 4.0,
                "p95_ttft_ms": 9.0,
                "new_throughput": 999.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _document, rows = SglangResultParser().parse(result_path)

    values = {row.name: row.value_float for row in rows}
    assert values["sglang.accept_length"] == 3.5
    assert values["sglang.request_throughput"] == 2.0
    assert values["sglang.total_throughput"] == 12.0
    assert "sglang.new_throughput" not in values
    assert all(row.dimensions["producer"] == "sglang.benchmark.serving" for row in rows)
    by_name = {row.name: row for row in rows}
    assert by_name["sglang.request_throughput"].unit == "requests/sec"
    assert by_name["sglang.input_throughput"].unit == "tokens/sec"
    assert by_name["sglang.output_throughput"].unit == "tokens/sec"
    assert by_name["sglang.total_input_tokens"].unit == "tokens"
    assert by_name["sglang.total_output_tokens"].unit == "tokens"
    assert by_name["sglang.accept_length"].unit == "tokens"
    assert by_name["sglang.concurrency"].unit == "dimensionless"
    assert by_name["sglang.mean_ttft_ms"].aggregation == "mean"
    assert by_name["sglang.p95_ttft_ms"].aggregation == "percentile"
    assert by_name["sglang.p95_ttft_ms"].dimensions["percentile"] == "95"


def test_sglang_every_supported_metric_has_an_explicit_descriptor(tmp_path: Path) -> None:
    payload: dict[str, float | int] = {
        "duration": 1.0,
        "completed": 2,
        "total_input_tokens": 8,
        "total_output_tokens": 4,
        "request_throughput": 2.0,
        "input_throughput": 8.0,
        "output_throughput": 4.0,
        "total_throughput": 12.0,
        "accept_length": 3.5,
        "concurrency": 1.5,
    }
    expected = {
        "sglang.duration": ("s", "aggregate", "duration"),
        "sglang.completed": ("requests", "count", "request_count"),
        "sglang.total_input_tokens": ("tokens", "sum", "token_count"),
        "sglang.total_output_tokens": ("tokens", "sum", "token_count"),
        "sglang.request_throughput": ("requests/sec", "rate", "request_rate"),
        "sglang.input_throughput": ("tokens/sec", "rate", "token_rate"),
        "sglang.output_throughput": ("tokens/sec", "rate", "token_rate"),
        "sglang.total_throughput": ("tokens/sec", "rate", "token_rate"),
        "sglang.accept_length": ("tokens", "mean", "speculative_accept_length"),
        "sglang.concurrency": ("dimensionless", "mean", "concurrency"),
    }
    for family in ("e2e_latency", "ttft", "tpot", "itl"):
        for statistic, aggregation in (
            ("mean", "mean"),
            ("median", "median"),
            ("std", "std"),
            ("p90", "percentile"),
            ("p95", "percentile"),
            ("p99", "percentile"),
        ):
            field = f"{statistic}_{family}_ms"
            payload[field] = 1.0
            expected[f"sglang.{field}"] = ("ms", aggregation, "latency")
    result_path = tmp_path / "all-metrics.jsonl"
    result_path.write_text(json.dumps(payload) + "\n")

    _document, rows = SglangResultParser().parse(result_path)

    assert {
        row.name: (row.unit, row.aggregation, row.dimensions["semantic_type"]) for row in rows
    } == expected


@pytest.mark.parametrize("payload", ["", "{}\n{}\n", '{"duration": NaN}\n'])
def test_sglang_result_parser_rejects_malformed_or_multiple_records(
    tmp_path: Path, payload: str
) -> None:
    result_path = tmp_path / "result.jsonl"
    result_path.write_text(payload, encoding="utf-8")

    with pytest.raises(DomainError, match="aggregate SGLang JSONL"):
        SglangResultParser().parse(result_path)
