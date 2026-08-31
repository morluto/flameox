from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from pydantic import ValidationError

from flameox.providers.contracts import ProviderAnalysis
from flameox.runtime_contracts import CaptureTarget
from flameox.stateless import AnalysisRuntime


@pytest.mark.process
def test_rocprof_capture_produces_native_pftrace_for_perfetto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "bin" / "rocprofv3"
    executable.parent.mkdir()
    calls = tmp_path / "calls.txt"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "arguments = sys.argv[1:]\n"
        f"pathlib.Path({str(calls)!r}).write_text(' '.join(arguments))\n"
        "root = pathlib.Path(arguments[arguments.index('-d') + 1])\n"
        "(root / 'rocprofv3_results.pftrace').write_bytes(b'perfetto-trace')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])

    async def exercise() -> dict[str, Any]:
        runtime = AnalysisRuntime(tmp_path)
        monkeypatch.setattr(
            runtime.perfetto,
            "analyze",
            lambda _capability, path, _arguments, **_kwargs: ProviderAnalysis(
                provider_id="perfetto",
                provider_version="test",
                blocks=[
                    {"type": "metrics", "values": {"trace_size": path.stat().st_size}},
                    {"type": "table", "rows": [{"name": "kernel"}]},
                ],
                rows_observed=1,
                complete=True,
                limitations=[],
            ),
        )
        try:
            return await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass"],
                    provider_id="rocprofv3",
                    capture_arguments={
                        "kernel_trace": True,
                        "memory_copy_trace": True,
                    },
                ),
                "trace.summary",
            )
        finally:
            runtime.close()

    result = anyio.run(exercise)
    assert result["provider"] == {"id": "perfetto", "version": "test"}
    assert result["blocks"][1]["rows"] == [{"name": "kernel"}]
    invocation = calls.read_text()
    assert "--output-format pftrace" in invocation
    assert "--kernel-trace --memory-copy-trace" in invocation
    assert " -- " in invocation


def test_rocprof_capture_requires_at_least_one_trace_domain(tmp_path: Path) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(tmp_path)
        try:
            await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "pass"],
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
                "trace.summary",
            )
        finally:
            runtime.close()

    with pytest.raises(ValidationError, match="at least one ROCprof trace domain"):
        anyio.run(exercise)
