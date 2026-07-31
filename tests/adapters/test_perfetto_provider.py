from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters import PerfettoExtractor
from flameox.analysis import RecipeService
from flameox.application import (
    ImportArtifactRequest,
    ImportService,
)
from flameox.domain import ArtifactKind
from flameox.storage import Workspace
from tests.support.providers import require_trace_processor


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_perfetto
async def test_torch_profiler_trace_preserves_shapes_memory_phases_and_sync(
    tmp_path: Path,
) -> None:
    binary = require_trace_processor()
    trace = tmp_path / "torch-trace.json"
    events = [
        {
            "name": "aten::add",
            "cat": "cpu_op",
            "ph": "X",
            "ts": index * 10,
            "dur": 5,
            "pid": 1,
            "tid": 1,
            "args": {
                "Input Shapes": "[[4, 8]]",
                "Bytes": 4096,
                "phase": "warmup",
            },
        }
        for index in range(4)
    ]
    events.extend(
        [
            {
                "name": "cudaDeviceSynchronize",
                "cat": "cuda_runtime",
                "ph": "X",
                "ts": 50,
                "dur": 7,
                "pid": 1,
                "tid": 1,
                "args": {},
            },
            {
                "name": "Torch-Compiled Region",
                "cat": "cpu_op",
                "ph": "X",
                "ts": 60,
                "dur": 11,
                "pid": 1,
                "tid": 1,
                "args": {"phase": "compile"},
            },
        ]
    )
    trace.write_text(json.dumps({"traceEvents": events}))
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.model_copy(
        update={
            "analysis": workspace.config.analysis.model_copy(
                update={"trace_processor_path": str(binary)}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.EXECUTION_TRACE,
            producer="torch.profiler",
        )
    )

    await PerfettoExtractor(workspace).extract(imported.run.run_id)
    result = RecipeService(workspace).pytorch(imported.run.run_id)

    by_name = {operator.operator: operator for operator in result.operators}
    assert by_name["aten::add"].input_shapes == ("[[4, 8]]",)
    assert by_name["aten::add"].allocation_bytes == 4 * 4096
    assert by_name["aten::add"].warmup is True
    assert result.coverage["input_shapes"]
    assert result.coverage["memory_allocations"]
    assert result.synchronization_time_ns == 7_000
    assert result.compilation_time_ns == 11_000
    assert result.repeated_small_operations[0].operator == "aten::add"


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_perfetto
async def test_pytorch_analysis_keeps_same_named_frames_and_mixed_phases_separate(
    tmp_path: Path,
) -> None:
    binary = require_trace_processor()
    trace = tmp_path / "torch-frames.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "aten::add",
                        "cat": "cpu_op",
                        "ph": "X",
                        "ts": 0,
                        "dur": 5,
                        "pid": 1,
                        "tid": 1,
                        "args": {
                            "filename": "model.py",
                            "line": 10,
                            "Input Shapes": "[[2, 2]]",
                            "Bytes": 100,
                            "phase": "warmup",
                        },
                    },
                    {
                        "name": "aten::add",
                        "cat": "cpu_op",
                        "ph": "X",
                        "ts": 10,
                        "dur": 7,
                        "pid": 1,
                        "tid": 1,
                        "args": {
                            "filename": "model.py",
                            "line": 10,
                            "Input Shapes": "[[2, 2]]",
                            "Bytes": 100,
                            "phase": "steady_state",
                        },
                    },
                    {
                        "name": "aten::add",
                        "cat": "cpu_op",
                        "ph": "X",
                        "ts": 20,
                        "dur": 11,
                        "pid": 1,
                        "tid": 1,
                        "args": {
                            "filename": "model.py",
                            "line": 20,
                            "Input Shapes": "[[8, 8]]",
                            "Bytes": 700,
                            "phase": "steady_state",
                        },
                    },
                ]
            }
        )
    )
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.model_copy(
        update={
            "analysis": workspace.config.analysis.model_copy(
                update={"trace_processor_path": str(binary)}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    imported = ImportService(workspace).import_artifact(
        ImportArtifactRequest(
            path=trace,
            kind=ArtifactKind.EXECUTION_TRACE,
            producer="torch.profiler",
        )
    )

    await PerfettoExtractor(workspace).extract(imported.run.run_id)
    result = RecipeService(workspace).pytorch(imported.run.run_id)

    by_shapes = {operator.input_shapes: operator for operator in result.operators}
    mixed = by_shapes[("[[2, 2]]",)]
    steady = by_shapes[("[[8, 8]]",)]
    assert mixed.allocation_bytes == 200
    assert mixed.warmup is None
    assert steady.allocation_bytes == 700
    assert steady.warmup is False
    assert result.warmup_time_ns == 5_000
