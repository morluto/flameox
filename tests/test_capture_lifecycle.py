from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import anyio
import psutil
import pytest

from flameox.runtime import AnalysisRuntime
from flameox.runtime_contracts import (
    CaptureTarget,
    EvidenceSource,
    PathSource,
    RequestLimits,
    RuntimeFailure,
)


@pytest.mark.integration
@pytest.mark.parametrize("cancelled", [False, True])
def test_capture_unwinds_scratch_when_progress_fails(tmp_path: Path, cancelled: bool) -> None:
    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")

        async def fail_progress(current: int, total: int, message: str) -> None:
            if cancelled:
                raise asyncio.CancelledError()
            raise RuntimeError("progress callback failed")

        try:
            with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[sys.executable, "-c", "print(1)"],
                        cwd=str(tmp_path),
                        provider_id="direct",
                    ),
                    "preview_artifact",
                    progress=fail_progress,
                )
            assert not list(runtime.scratch.glob("capture-*"))
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
@pytest.mark.serial
def test_cancelled_capture_settles_child_and_removes_scratch(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        task = asyncio.create_task(
            runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[
                        sys.executable,
                        "-c",
                        "import os, pathlib, time; "
                        "pathlib.Path('child.pid').write_text(str(os.getpid())); time.sleep(30)",
                    ],
                    cwd=str(tmp_path),
                    provider_id="direct",
                ),
                "preview_artifact",
            )
        )
        try:
            with anyio.fail_after(5):
                while not pid_path.exists():
                    await anyio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            pid = int(pid_path.read_text())
            assert (
                not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
            )
            assert not list(runtime.scratch.glob("capture-*"))
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
def test_live_capture_keeps_capacity_reserved_until_unwind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("flameox.runtime.MAX_SESSION_SCRATCH_BYTES", 1024)

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        started = asyncio.Event()
        release = asyncio.Event()
        target = CaptureTarget(
            argv=[sys.executable, "-c", "print(1)"], cwd=str(tmp_path), provider_id="direct"
        )
        limits = RequestLimits(max_output_bytes=1024)

        async def hold(current: int, total: int, message: str) -> None:
            started.set()
            await release.wait()

        first = asyncio.create_task(
            runtime.capture_and_analyze(target, "preview_artifact", limits=limits, progress=hold)
        )
        try:
            await asyncio.wait_for(started.wait(), 2)
            with pytest.raises(RuntimeFailure) as failure:
                await runtime.capture_and_analyze(target, "preview_artifact", limits=limits)
            assert failure.value.code == "LIMIT_EXCEEDED"
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            result = await runtime.capture_and_analyze(target, "preview_artifact", limits=limits)
            assert result["capture"]["outcome"]["status"] == "succeeded"
        finally:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            runtime.close()

    anyio.run(exercise)


@pytest.mark.integration
def test_evidence_materialization_cannot_consume_live_capture_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "data.txt").write_text("x" * 2048)
    publisher = AnalysisRuntime(evidence_directory=tmp_path / "store")
    try:
        analysis = publisher.analyze("preview_artifact", [PathSource(path=str(bundle))], {})
        preserved = publisher.preserve_evidence(analysis["analysis_id"])
        projection = publisher.read_evidence_agent_projection(preserved["evidence_id"])
        source = EvidenceSource.model_validate(projection["analysis_sources"][0])
    finally:
        publisher.close()
    monkeypatch.setattr("flameox.runtime.MAX_SESSION_SCRATCH_BYTES", 2048)

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        started = asyncio.Event()
        release = asyncio.Event()

        async def hold(current: int, total: int, message: str) -> None:
            started.set()
            await release.wait()

        capture = asyncio.create_task(
            runtime.capture_and_analyze(
                CaptureTarget(
                    argv=[sys.executable, "-c", "print(1)"],
                    cwd=str(tmp_path),
                    provider_id="direct",
                ),
                "preview_artifact",
                limits=RequestLimits(max_output_bytes=1024),
                progress=hold,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 5)
            with pytest.raises(RuntimeFailure) as failure:
                runtime.analyze("preview_artifact", [source], {})
            assert failure.value.code == "LIMIT_EXCEEDED"
            assert not [path for path in runtime.scratch.rglob("*") if path.is_file()]
        finally:
            capture.cancel()
            await asyncio.gather(capture, return_exceptions=True)
            runtime.close()

    anyio.run(exercise)
