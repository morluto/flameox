from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from flameox.providers.perfetto import PerfettoProvider
from flameox.workers.perfetto_contract import PerfettoExtractResult, PerfettoSliceRow


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


def test_pytorch_projection_does_not_replace_existing_call_graph_projection() -> None:
    rows = [
        _slice(1, "parent").model_dump(mode="json"),
        _slice(2, "child", parent_id=1).model_dump(mode="json"),
    ]

    assert PerfettoProvider._project("trace.call_graph", rows) == [
        {
            "parent": "parent",
            "child": "child",
            "sample_count": 1,
            "inclusive_duration_ns": 5,
        }
    ]
