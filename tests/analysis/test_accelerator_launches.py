from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.analysis import RecipeService
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace
from tests.support.analysis import run_row

pytestmark = pytest.mark.integration


def _event(
    run_id: str,
    observation_id: str,
    name: str,
    *,
    category: str,
    start_ns: int,
    duration_ns: int,
    phase: str | None = "decode",
    correlation_id: str | None = None,
    stream: str | None = None,
    track_id: int | None = 7,
    thread: int | None = None,
    device: str = "cuda:0",
    context: str | None = None,
) -> dict[str, object]:
    value = {
        "category": category,
        "start_ns": start_ns,
        "duration_ns": duration_ns,
        "track_id": track_id,
        "thread": thread,
        "phase": phase,
        "correlation_id": correlation_id,
        "device": device,
        "context": context,
        "stream": stream,
    }
    return {
        "observation_id": observation_id,
        "run_id": run_id,
        "artifact_id": None,
        "kind": "trace.event",
        "name": name,
        "value_json": json.dumps(value, sort_keys=True),
        "file": None,
        "line_from": None,
        "line_to": None,
        "context": phase,
        "evidence_level": "observed",
    }


def test_accelerator_launches_counts_graphs_kernels_and_per_stream_gaps(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row("decode-run")],
            "observations": [
                _event(
                    "decode-run",
                    "direct",
                    "cudaLaunchKernel",
                    category="cuda_runtime",
                    start_ns=0,
                    duration_ns=3,
                    correlation_id="41",
                ),
                _event(
                    "decode-run",
                    "graph",
                    "cudaGraphLaunch",
                    category="cuda_runtime",
                    start_ns=4,
                    duration_ns=2,
                ),
                _event(
                    "decode-run",
                    "annotation-named-like-direct-launch",
                    "cudaLaunchKernel",
                    category="user_annotation",
                    start_ns=7,
                    duration_ns=1,
                ),
                _event(
                    "decode-run",
                    "annotation-named-like-graph-launch",
                    "cudaGraphLaunch",
                    category="user_annotation",
                    start_ns=8,
                    duration_ns=1,
                ),
                _event(
                    "decode-run",
                    "kernel-a",
                    "projection_kernel",
                    category="kernel",
                    start_ns=10,
                    duration_ns=5,
                    correlation_id="41",
                    stream="9",
                ),
                _event(
                    "decode-run",
                    "kernel-b",
                    "projection_kernel",
                    category="kernel",
                    start_ns=20,
                    duration_ns=4,
                    correlation_id="42",
                    stream="9",
                ),
                _event(
                    "decode-run",
                    "warmup-kernel",
                    "warmup_kernel",
                    category="kernel",
                    start_ns=1,
                    duration_ns=100,
                    phase="warmup",
                    stream="9",
                ),
            ],
        },
        publisher="accelerator-fixture",
        publisher_version="1",
    )

    result = RecipeService(workspace).accelerator_launches(
        "decode-run",
        phase="decode",
    )

    assert result.total == 1
    assert result.coverage == {
        "runtime_launches": True,
        "accelerator_kernels": True,
        "phase_annotations": True,
        "correlation_ids": True,
        "host_to_device_correlation": True,
        "stream_identity": True,
    }
    region = result.regions[0]
    assert region.region == "decode"
    assert region.direct_launch_count == 1
    assert region.graph_launch_count == 1
    assert region.kernel_count == 2
    assert region.kernel_duration_ns == 9
    assert region.correlated_kernel_count == 1
    assert region.idle_gap_count == 1
    assert region.idle_gap_total_ns == 5
    assert region.idle_gap_max_ns == 5
    assert region.stream_count == 1
    assert region.streams[0].device == "cuda:0"
    assert region.streams[0].stream == "9"
    assert region.streams[0].idle_gap_total_ns == 5
    assert not region.streams_truncated
    assert [(item.name, item.count) for item in region.kernel_names] == [("projection_kernel", 2)]
    assert "streams_truncated" not in type(region).model_fields
    assert "streams_truncated" in type(region).model_json_schema()["properties"]

    missing_phase = RecipeService(workspace).accelerator_launches(
        "decode-run",
        phase="missing",
    )
    assert missing_phase.total == 0
    assert missing_phase.coverage["phase_annotations"]
    assert any("matched phase 'missing'" in item for item in missing_phase.limitations)


def test_accelerator_launch_comparison_reports_descriptive_region_deltas(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row("eager"), run_row("graph")],
            "observations": [
                _event(
                    "eager",
                    f"eager-{index}",
                    "cudaLaunchKernel",
                    category="cuda_runtime",
                    start_ns=index * 10,
                    duration_ns=2,
                    correlation_id=f"eager-{index}",
                )
                for index in range(3)
            ]
            + [
                _event(
                    "eager",
                    f"eager-kernel-{index}",
                    "projection_kernel",
                    category="kernel",
                    start_ns=100 + index * 10,
                    duration_ns=5,
                    correlation_id=f"eager-{index}",
                    stream="9",
                )
                for index in range(3)
            ]
            + [
                _event(
                    "graph",
                    "graph-launch",
                    "cudaGraphLaunch",
                    category="cuda_runtime",
                    start_ns=0,
                    duration_ns=3,
                    correlation_id="graph",
                )
            ]
            + [
                _event(
                    "graph",
                    f"graph-kernel-{index}",
                    "projection_kernel",
                    category="kernel",
                    start_ns=100 + index * 10,
                    duration_ns=5,
                    correlation_id="graph",
                    stream="9",
                )
                for index in range(3)
            ],
        },
        publisher="accelerator-comparison-fixture",
        publisher_version="1",
    )

    result = RecipeService(workspace).accelerator_launches(
        "eager",
        comparison_input_id="graph",
    )

    assert len(result.comparisons) == 1
    comparison = result.comparisons[0]
    assert comparison.region == "decode"
    assert comparison.direct_launch_count_delta == -3
    assert comparison.graph_launch_count_delta == 1
    assert comparison.kernel_count_delta == 0
    assert result.comparison_coverage is not None
    assert result.comparison_coverage["accelerator_kernels"]
    assert result.evidence.status == "available"
    assert any("descriptive" in limitation for limitation in result.limitations)


def test_accelerator_launches_keep_nsight_thread_and_device_streams_separate(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row("nsight-run")],
            "observations": [
                _event(
                    "nsight-run",
                    "runtime-thread-7",
                    "cudaLaunchKernel",
                    category="cuda_runtime",
                    start_ns=0,
                    duration_ns=2,
                    track_id=None,
                    thread=7,
                ),
                _event(
                    "nsight-run",
                    "runtime-thread-8",
                    "cudaLaunchKernel",
                    category="cuda_runtime",
                    start_ns=10,
                    duration_ns=2,
                    track_id=None,
                    thread=8,
                ),
                _event(
                    "nsight-run",
                    "kernel-device-0",
                    "projection_kernel",
                    category="kernel",
                    start_ns=20,
                    duration_ns=5,
                    stream="9",
                    device="0",
                    context="1",
                ),
                _event(
                    "nsight-run",
                    "kernel-device-1",
                    "projection_kernel",
                    category="kernel",
                    start_ns=30,
                    duration_ns=5,
                    stream="9",
                    device="1",
                    context="1",
                ),
            ],
        },
        publisher="nsight-identity-fixture",
        publisher_version="1",
    )

    region = RecipeService(workspace).accelerator_launches("nsight-run").regions[0]

    assert region.runtime_launch_gap_count == 0
    assert region.idle_gap_count == 0
    assert region.stream_count == 2
    assert {(item.device, item.context, item.stream) for item in region.streams} == {
        ("0", "1", "9"),
        ("1", "1", "9"),
    }


def test_accelerator_launches_marks_missing_runtime_or_kernel_tracks_partial(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row("partial-run")],
            "observations": [
                _event(
                    "partial-run",
                    "graph-only",
                    "cudaGraphLaunch",
                    category="cuda_runtime",
                    start_ns=0,
                    duration_ns=2,
                    phase="graph-only",
                ),
                _event(
                    "partial-run",
                    "kernel-only",
                    "projection_kernel",
                    category="kernel",
                    start_ns=10,
                    duration_ns=3,
                    phase="kernel-only",
                    stream="9",
                ),
            ],
        },
        publisher="partial-accelerator-fixture",
        publisher_version="1",
    )

    graph_only = RecipeService(workspace).accelerator_launches(
        "partial-run",
        phase="graph-only",
    )
    kernel_only = RecipeService(workspace).accelerator_launches(
        "partial-run",
        phase="kernel-only",
    )

    assert graph_only.evidence.status == "partial"
    assert graph_only.coverage["runtime_launches"]
    assert not graph_only.coverage["accelerator_kernels"]
    assert kernel_only.evidence.status == "partial"
    assert not kernel_only.coverage["runtime_launches"]
    assert kernel_only.coverage["accelerator_kernels"]


def test_accelerator_launches_reports_unscoped_zero_duration_evidence(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row("unscoped-run")],
            "observations": [
                _event(
                    "unscoped-run",
                    "runtime",
                    "cudaLaunchKernel",
                    category="cuda_runtime",
                    start_ns=0,
                    duration_ns=0,
                    phase=None,
                ),
                _event(
                    "unscoped-run",
                    "kernel",
                    "projection_kernel",
                    category="kernel",
                    start_ns=0,
                    duration_ns=0,
                    phase=None,
                    stream="9",
                ),
            ],
        },
        publisher="unscoped-accelerator-fixture",
        publisher_version="1",
    )

    result = RecipeService(workspace).accelerator_launches("unscoped-run")

    assert result.regions[0].region == "<unscoped>"
    assert result.regions[0].kernel_duration_ns == 0
    assert not result.coverage["phase_annotations"]
    assert not result.coverage["correlation_ids"]
    assert any("grouped as <unscoped>" in item for item in result.limitations)
