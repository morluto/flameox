from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from flameox.providers.perfetto import PerfettoProvider
from flameox.workers.perfetto_contract import (
    PerfettoCallGraphRow,
    PerfettoExtractResult,
    PerfettoSliceRow,
)


class _Harness:
    def __init__(self, response: PerfettoExtractResult) -> None:
        self.response = response
        self.requests: list[Any] = []

    def run_typed_sync(self, _worker: Any, request: Any, **_kwargs: Any) -> PerfettoExtractResult:
        self.requests.append(request)
        return self.response


def _slice(
    identifier: int,
    name: str,
    *,
    category: str | None = None,
    parent_id: int | None = None,
) -> PerfettoSliceRow:
    return PerfettoSliceRow(
        id=identifier,
        parent_id=parent_id,
        name=name,
        ts=identifier * 10,
        dur=5,
        track_id=1,
        category=category,
        thread_name="main",
        process_name="python",
        filename=None,
        line=None,
        input_shapes=None,
        allocation_bytes=None,
        phase=None,
        correlation_id=None,
        device=None,
        stream=None,
    )


def test_pytorch_projection_excludes_generic_perfetto_slices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = PerfettoExtractResult(
        truncated=False,
        rows=(
            _slice(1, "event_loop", category="python"),
            _slice(2, "aten::matmul", category="cpu_op"),
            _slice(3, "ProfilerStep#1", category="pytorch"),
        ),
    )
    harness = _Harness(response)
    provider = PerfettoProvider(harness)  # type: ignore[arg-type]
    binary = tmp_path / "trace_processor"
    binary.write_bytes(b"binary")
    monkeypatch.setattr(PerfettoProvider, "_binary", staticmethod(lambda: binary))
    monkeypatch.setattr(PerfettoProvider, "_identity", staticmethod(lambda _path: "test"))

    summary = provider.analyze(
        "trace.summary",
        tmp_path / "trace.json",
        {},
        max_rows=10,
        timeout_seconds=1,
        maximum_rss_bytes=1024,
        maximum_output_bytes=1024,
    )
    pytorch = provider.analyze(
        "trace.pytorch",
        tmp_path / "trace.json",
        {},
        max_rows=10,
        timeout_seconds=1,
        maximum_rss_bytes=1024,
        maximum_output_bytes=1024,
    )

    assert [row["name"] for row in summary.blocks[1]["rows"]] == [
        "event_loop",
        "aten::matmul",
        "ProfilerStep#1",
    ]
    assert [row["name"] for row in pytorch.blocks[1]["rows"]] == [
        "aten::matmul",
        "ProfilerStep#1",
    ]
    assert pytorch.blocks[0]["values"]["pytorch_event_count"] == 2
    assert harness.requests[-1].projection == "pytorch"


@pytest.mark.parametrize("total", [1, 111])
def test_call_graph_reports_edge_population_instead_of_zero_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, total: int
) -> None:
    harness = _Harness(
        PerfettoExtractResult(
            truncated=total > 1,
            rows=(),
            call_graph_rows=(
                PerfettoCallGraphRow(
                    parent="parent", child="child", sample_count=1, inclusive_duration_ns=5
                ),
            ),
            projected_total=total,
        )
    )
    provider = PerfettoProvider(harness)  # type: ignore[arg-type]
    monkeypatch.setattr(PerfettoProvider, "_binary", staticmethod(lambda: tmp_path / "reader"))
    monkeypatch.setattr(PerfettoProvider, "_identity", staticmethod(lambda _path: "test"))
    result = provider.analyze(
        "trace.call_graph",
        tmp_path / "trace.json",
        {},
        max_rows=1,
        timeout_seconds=1,
        maximum_rss_bytes=1024,
        maximum_output_bytes=1024,
    )
    assert result.blocks[0]["values"] == {"edge_count": total}
    assert result.rows_observed == total
    assert result.complete is (total == 1)
    assert result.blocks[1]["rows"] == [
        {
            "parent": "parent",
            "child": "child",
            "sample_count": 1,
            "inclusive_duration_ns": 5,
        }
    ]
