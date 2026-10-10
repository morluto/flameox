from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import PathSource, RequestLimits


def _trace_processor() -> str | None:
    configured = os.environ.get("FLAMEOX_TRACE_PROCESSOR")
    if configured:
        return shutil.which(configured)
    return shutil.which("trace_processor_shell") or shutil.which("trace_processor")


@pytest.mark.golden
@pytest.mark.process
@pytest.mark.optional
def test_chrome_trace_projects_native_perfetto_evidence(tmp_path: Path) -> None:
    pytest.importorskip("perfetto", reason="Perfetto Python package is not installed")
    binary = _trace_processor()
    if binary is None:
        pytest.skip("Trace Processor executable not found on PATH or FLAMEOX_TRACE_PROCESSOR")

    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "name": "request",
                        "cat": "python",
                        "ph": "X",
                        "ts": 100,
                        "dur": 600,
                        "pid": 1,
                        "tid": 7,
                    },
                    {
                        "name": "load_model",
                        "cat": "python",
                        "ph": "X",
                        "ts": 150,
                        "dur": 300,
                        "pid": 1,
                        "tid": 7,
                    },
                    {
                        "name": "aten::matmul",
                        "cat": "cpu_op",
                        "ph": "X",
                        "ts": 200,
                        "dur": 100,
                        "pid": 1,
                        "tid": 7,
                        "args": {"Input Shapes": "[2, 3]", "filename": "model.py", "line": 42},
                    },
                    {
                        "name": "event_loop",
                        "cat": "python",
                        "ph": "X",
                        "ts": 800,
                        "dur": 50,
                        "pid": 1,
                        "tid": 7,
                    },
                ]
            }
        )
    )

    runtime = AnalysisRuntime(
        evidence_directory=tmp_path / ".flameox", limits=RequestLimits(max_rows=1_000)
    )
    source = [PathSource(path=str(trace), format="chrome-trace")]
    try:
        summary = runtime.analyze("summarize_trace", source, {})
        call_graph = runtime.analyze("inspect_trace_call_graph", source, {})
        pytorch = runtime.analyze("summarize_pytorch_trace", source, {})
        window = runtime.analyze(
            "inspect_trace_window", source, {"start_ns": 240_000, "end_ns": 250_000}
        )
        large_trace = tmp_path / "many-edges.json"
        large_trace.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {"name": "root", "ph": "X", "ts": 0, "dur": 3_000, "pid": 1, "tid": 1},
                        *[
                            {
                                "name": f"child-{index}",
                                "ph": "X",
                                "ts": index * 2 + 1,
                                "dur": 1,
                                "pid": 1,
                                "tid": 1,
                            }
                            for index in range(1_005)
                        ],
                    ]
                }
            )
        )
        limited = runtime.analyze(
            "inspect_trace_call_graph",
            [PathSource(path=str(large_trace), format="chrome-trace")],
            {},
            limits=RequestLimits(max_rows=1_000),
        )
    finally:
        runtime.close()

    summary_rows = summary["blocks"][1]["rows"]
    assert [row["name"] for row in summary_rows] == [
        "request",
        "load_model",
        "aten::matmul",
        "event_loop",
    ]
    assert summary_rows[2]["input_shapes"] == "[2, 3]"
    assert summary_rows[2]["filename"] == "model.py"
    assert summary_rows[2]["line"] == 42
    assert call_graph["blocks"][1]["rows"] == [
        {
            "parent": "request",
            "child": "load_model",
            "sample_count": 1,
            "inclusive_duration_ns": 300_000,
        },
        {
            "parent": "load_model",
            "child": "aten::matmul",
            "sample_count": 1,
            "inclusive_duration_ns": 100_000,
        },
    ]
    assert [row["name"] for row in pytorch["blocks"][1]["rows"]] == ["aten::matmul"]
    assert [row["name"] for row in window["blocks"][1]["rows"]] == [
        "request",
        "load_model",
        "aten::matmul",
    ]
    assert window["blocks"][0]["values"]["matching_slice_count"] == 3
    assert limited["coverage"] == {
        "rows_observed": 1_005,
        "rows_returned": 1_000,
        "complete": False,
    }
    assert limited["blocks"][0]["values"]["edge_count"] == 1_005
    assert not (tmp_path / ".flameox").exists()
