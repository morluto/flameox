from __future__ import annotations

import pytest

from flameox.providers.aiperf import AIPerfProvider
from flameox.providers.contracts import ProviderAnalysis


def _analysis(*latencies: int) -> ProviderAnalysis:
    rows = [
        {
            "outcome": "succeeded",
            "latency_ns": latency,
            "input_tokens": 10,
            "output_tokens": 2,
        }
        for latency in latencies
    ]
    return ProviderAnalysis(
        provider_id="aiperf",
        provider_version="0.12.0",
        blocks=[{"type": "metrics", "values": {}}, {"type": "table", "rows": rows}],
        rows_observed=len(rows),
        complete=True,
        limitations=[],
    )


def test_aiperf_comparison_uses_complete_prompt_free_request_metrics() -> None:
    result = AIPerfProvider.compare(
        [_analysis(10, 14), _analysis(5, 7)],
        {"metric": "latency_ns", "baseline_index": 0},
        max_rows=10,
    )

    assert result.provider_id == "aiperf"
    assert result.blocks[1]["rows"][0]["baseline_mean"] == pytest.approx(12)
    assert result.blocks[1]["rows"][0]["ratio"] == pytest.approx(0.5)
