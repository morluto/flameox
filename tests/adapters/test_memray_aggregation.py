from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest

from flameox.workers import memray as memray_worker


@dataclass(frozen=True)
class _AllocationRecord:
    size: int
    n_allocations: int
    stack: tuple[tuple[str, str, int], ...]

    def stack_trace(self) -> tuple[tuple[str, str, int], ...]:
        return self.stack


def test_memray_frame_identity_is_computed_once_across_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized: list[str] = []

    def normalize(filename: str, project_root: Path) -> str:
        assert project_root == tmp_path
        normalized.append(filename)
        return filename

    monkeypatch.setattr(memray_worker, "_normalize", normalize)
    stack = (("allocate", "agent.py", 10), ("main", "runner.py", 20))
    records = [_AllocationRecord(size=5, n_allocations=1, stack=stack) for _ in range(100)]
    frame_rows: dict[str, dict[str, Any]] = {}
    frame_cache: dict[tuple[str, str, int], str] = {}
    aggregates: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"self": 0, "inclusive": 0, "samples": 0}
    )

    for metric in ("memory.high_watermark", "memory.retained_end"):
        memray_worker._aggregate(
            records,
            metric=metric,
            project_root=tmp_path,
            frame_rows=frame_rows,
            frame_cache=frame_cache,
            aggregates=aggregates,
            artifact_id="artifact",
        )

    assert normalized == ["agent.py", "runner.py"]
    assert len(frame_cache) == len(frame_rows) == 2
    assert len(aggregates) == 4


@pytest.mark.performance
def test_memray_repeated_frame_aggregation_budget(tmp_path: Path) -> None:
    stack = (("allocate", "agent.py", 10), ("main", "runner.py", 20))
    record = _AllocationRecord(size=5, n_allocations=1, stack=stack)
    records = [record] * 500_000
    frame_rows: dict[str, dict[str, Any]] = {}
    frame_cache: dict[tuple[str, str, int], str] = {}
    aggregates: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"self": 0, "inclusive": 0, "samples": 0}
    )

    started = perf_counter()
    memray_worker._aggregate(
        records,
        metric="memory.high_watermark",
        project_root=tmp_path,
        frame_rows=frame_rows,
        frame_cache=frame_cache,
        aggregates=aggregates,
        artifact_id="artifact",
    )
    elapsed = perf_counter() - started

    assert len(frame_cache) == len(frame_rows) == len(aggregates) == 2
    assert elapsed < 5
