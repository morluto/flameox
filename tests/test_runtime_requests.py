from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any

import anyio
import psutil
import pytest
from mcp import Client

from flameox.command_binding import ExecutableResolver
from flameox.executable_models import ResolvedExecutable
from flameox.execution import ExecutionRequest
from flameox.mcp import create_server
from flameox.runtime_contracts import CaptureTarget, RuntimeFailure
from flameox.stateless import AnalysisRuntime


@pytest.mark.unit
@pytest.mark.anyio
async def test_request_boundary_preserves_values_and_domain_errors(tmp_path: Path) -> None:
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")

    def fail() -> None:
        raise RuntimeFailure("INVALID_INPUT", "invalid request")

    def nothing() -> object:
        return None

    try:
        assert await runtime.run_in_request(lambda: 42) == 42
        assert await runtime.run_in_request(nothing) is None
        with pytest.raises(RuntimeFailure, match="invalid request"):
            await runtime.run_in_request(fail)
        assert await runtime.run_in_request(lambda: "usable") == "usable"
    finally:
        runtime.close()


@pytest.mark.unit
@pytest.mark.anyio
async def test_mcp_catalog_remains_available_during_blocking_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.txt"
    source.write_text("evidence\n")
    started = threading.Event()
    release = threading.Event()
    original = AnalysisRuntime.analyze

    def gated(runtime: AnalysisRuntime, *args: Any, **kwargs: Any) -> dict[str, Any]:
        started.set()
        if not release.wait(3):
            raise TimeoutError("Catalog did not run while analysis was blocked")
        return original(runtime, *args, **kwargs)

    monkeypatch.setattr(AnalysisRuntime, "analyze", gated)
    try:
        async with Client(create_server(evidence_directory=tmp_path / "evidence")) as client:
            analysis = asyncio.create_task(
                client.call_tool(
                    "preview_artifact",
                    {"sources": [{"kind": "path", "path": str(source)}]},
                )
            )
            assert await anyio.to_thread.run_sync(started.wait, 3)
            try:
                catalog = await client.list_tools()
                assert catalog
                assert not analysis.done()
            finally:
                release.set()
            result = await analysis
            assert not result.is_error
    finally:
        release.set()


@pytest.mark.integration
@pytest.mark.anyio
async def test_provider_publication_waits_for_active_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    replacement = tmp_path / "replacement-py-spy"
    replacement.write_text(f"#!{sys.executable}\nprint('py-spy 0.4.2')\n")
    replacement.chmod(0o755)
    resolver = ExecutableResolver()
    first = resolver.require_host_tool(sys.executable)
    second = resolver.require_host_tool(str(replacement))
    replacement_ready = anyio.Event()
    bindings = iter((first, second))

    async def prepare(_timeout: int) -> tuple[ResolvedExecutable, list[str]]:
        binding = next(bindings)
        if binding is second:
            replacement_ready.set()
        return binding, []

    monkeypatch.setattr(runtime.dependencies, "_prepare_py_spy", prepare)
    started = threading.Event()
    release = threading.Event()

    def admission() -> str | None:
        executable = runtime.dependencies.py_spy_executable()
        started.set()
        if not release.wait(3):
            raise TimeoutError("Admission was not released")
        runtime.dependencies.verify_capture_binding("py-spy", first)
        return executable

    tasks: list[asyncio.Task[Any]] = []
    try:
        await runtime.dependencies.prepare(["py-spy"])
        active = asyncio.create_task(runtime.run_in_request(admission))
        tasks.append(active)
        assert await anyio.to_thread.run_sync(started.wait, 3)
        publication = asyncio.create_task(runtime.dependencies.prepare(["py-spy"]))
        tasks.append(publication)
        await replacement_ready.wait()
        try:
            assert not publication.done()
        finally:
            release.set()
        assert await active == str(first.invocation_path)
        await publication
        assert runtime.dependencies.py_spy_executable() == str(second.invocation_path)
    finally:
        release.set()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(*tasks, return_exceptions=True)
        runtime.close()


@pytest.mark.integration
@pytest.mark.process
@pytest.mark.anyio
@pytest.mark.parametrize("cancellation", ["task", "scope"])
@pytest.mark.parametrize("request_kind", ["analysis", "capture_finalization"])
async def test_request_cancellation_joins_worker_before_releasing_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancellation: str, request_kind: str
) -> None:
    runtime = AnalysisRuntime(evidence_directory=tmp_path / "evidence")
    settled = threading.Event()
    scopes: list[anyio.CancelScope] = []
    pid_path = tmp_path / "child.pid"
    execution = ExecutionRequest(
        argv=(
            sys.executable,
            "-c",
            "import os,pathlib,time; "
            "pathlib.Path('child.pid').write_text(str(os.getpid())); time.sleep(30)",
        ),
        executable_binding=ExecutableResolver().require_host_tool(sys.executable),
        cwd=tmp_path,
        allowed_working_roots=(tmp_path,),
        timeout_seconds=5,
        max_output_bytes=1024,
    )

    def operation() -> None:
        try:
            runtime.broker.run_sync(execution)
        finally:
            settled.set()

    def analyze(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        operation()
        raise AssertionError("Expected cancellation before analysis completed")

    if request_kind == "capture_finalization":
        monkeypatch.setattr(runtime, "analyze", analyze)

    async def request() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            if request_kind == "analysis":
                await runtime.run_in_request(operation)
            else:
                await runtime.capture_and_analyze(
                    CaptureTarget(
                        argv=[sys.executable, "-c", "print('captured')"],
                        cwd=str(tmp_path),
                        provider_id="direct",
                    ),
                    "artifact.preview",
                )

    active = asyncio.create_task(request())
    try:
        with anyio.fail_after(10):
            while not pid_path.exists():
                await anyio.sleep(0.01)
            following = asyncio.create_task(runtime.run_in_request(settled.is_set))
            if cancellation == "task":
                active.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await active
            else:
                scopes[0].cancel()
                await active
            assert settled.is_set()
            assert await following is True
            assert not psutil.pid_exists(int(pid_path.read_text()))
            assert not list(runtime.scratch.glob("capture-*"))
    finally:
        active.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(active, return_exceptions=True)
        runtime.close()
