from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import anyio
import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    CaptureTarget,
    PathSource,
    RequestLimits,
    RuntimeFailure,
)


@pytest.mark.integration
@pytest.mark.requires_memray
def test_memray_capture_is_analyzed_without_creating_repository(tmp_path: Path) -> None:
    memray = pytest.importorskip("memray")
    capture = tmp_path / "memory.bin"
    with memray.Tracker(str(capture)):
        retained = [bytearray(1_024) for _ in range(8)]
    assert retained

    runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
    try:
        result = runtime.analyze(
            "memory.hotspots",
            [PathSource(path=str(capture), format="memray", producer="memray")],
            {},
            limits=RequestLimits(max_rows=10),
        )
    finally:
        runtime.close()

    assert result["provider"] == {"id": "memray", "version": memray.__version__}
    assert result["blocks"][0]["values"]["peak_memory_bytes"] > 0
    assert result["blocks"][1]["rows"]
    assert not (tmp_path / ".flameox").exists()


@pytest.mark.process
@pytest.mark.requires_memray
def test_direct_memray_capture_uses_typed_argv_and_preserves_native_output(
    tmp_path: Path,
) -> None:
    pytest.importorskip("memray")

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / ".flameox")
        try:
            result = await runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "retained = [bytearray(4096) for _ in range(8)]"],
                    cwd=str(tmp_path),
                    provider_id="memray",
                    capture_arguments={"native": False},
                ),
                "memory.hotspots",
                limits=RequestLimits(max_rows=10),
            )
            preserved = runtime.preserve_evidence(result["analysis_id"])
            manifest = runtime.read_evidence(preserved["evidence_id"])
        finally:
            runtime.close()

        execution = result["capture"]["executions"][0]
        assert execution["capture_argv"][1:4] == ["-m", "memray", "run"]
        assert result["provider"]["id"] == "memray"
        assert {item["role"] for item in manifest["body"]["artifacts"]} == {
            "capture-0001/memory",
        }

    asyncio.run(exercise())


@pytest.mark.process
def test_memray_capture_rejects_workload_interpreter_without_provider(tmp_path: Path) -> None:
    workload_python = tmp_path / "python"
    workload_python.write_text("#!/bin/sh\nexit 7\n")
    workload_python.chmod(0o755)

    async def exercise() -> None:
        runtime = AnalysisRuntime(
            evidence_directory=tmp_path / ".flameox", limits=RequestLimits(timeout_seconds=20)
        )
        try:
            with pytest.raises(RuntimeFailure) as raised:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[str(workload_python), "workload.py"],
                        cwd=str(tmp_path),
                        provider_id="memray",
                    ),
                    "memory.hotspots",
                )
        finally:
            runtime.close()

        assert raised.value.code == "UNAVAILABLE_CAPABILITY"
        assert "memray >=1.17" in raised.value.message

    anyio.run(exercise)
