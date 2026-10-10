from __future__ import annotations

import sys
from pathlib import Path

import anyio
import pytest
from pydantic import ValidationError

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import CaptureTarget


def test_rocprof_capture_requires_at_least_one_trace_domain(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass"],
                    cwd=str(tmp_path),
                    provider_id="rocprofv3",
                    capture_arguments={
                        "hip_trace": False,
                        "kernel_trace": False,
                        "memory_copy_trace": False,
                        "memory_allocation_trace": False,
                        "scratch_memory_trace": False,
                        "marker_trace": False,
                    },
                ),
                "summarize_trace",
            )
        finally:
            runtime.close()

    with pytest.raises(ValidationError, match="at least one ROCprof trace domain"):
        anyio.run(exercise)
