from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters import (
    SglangResultParser,
)
from flameox.domain import DomainError


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
                "total_token_throughput": 12.0,
                "accept_length": 3.5,
                "mean_ttft_ms": 4.0,
                "p95_ttft_ms": 9.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _document, rows = SglangResultParser().parse(result_path)

    values = {row.name: row.value_float for row in rows}
    assert values["sglang.accept_length"] == 3.5
    assert values["sglang.request_throughput"] == 2.0
    assert values["sglang.total_token_throughput"] == 12.0
    assert all(row.dimensions["producer"] == "sglang.bench_serving" for row in rows)
    by_name = {row.name: row for row in rows}
    assert by_name["sglang.mean_ttft_ms"].aggregation == "mean"
    assert by_name["sglang.p95_ttft_ms"].aggregation == "percentile"
    assert by_name["sglang.p95_ttft_ms"].dimensions["percentile"] == "95"


@pytest.mark.parametrize("payload", ["", "{}\n{}\n", '{"duration": NaN}\n'])
def test_sglang_result_parser_rejects_malformed_or_multiple_records(
    tmp_path: Path, payload: str
) -> None:
    result_path = tmp_path / "result.jsonl"
    result_path.write_text(payload, encoding="utf-8")

    with pytest.raises(DomainError, match="aggregate SGLang JSONL"):
        SglangResultParser().parse(result_path)
