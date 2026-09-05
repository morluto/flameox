from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from mcp import Client
from mcp_types import TextContent

from flameox import __version__
from flameox.mcp import create_server
from flameox.providers.preparation import PY_SPY_VERSION, ProviderDependencies
from flameox.runtime_contracts import PathSource, RuntimeFailure
from flameox.setup import SetupFailure, active_provider_status
from flameox.stateless import AnalysisRuntime


@pytest.mark.unit
@pytest.mark.parametrize(
    ("release", "installed", "expected"),
    [
        (__version__, "2.0", "ready"),
        (__version__, "1.0", "restart_required"),
        ("0.0.0", "2.0", "unknown"),
    ],
)
def test_active_provider_contract_checks_release_and_requirements(
    monkeypatch: pytest.MonkeyPatch,
    release: str,
    installed: str,
    expected: str,
) -> None:
    distribution = SimpleNamespace(version=release, requires=['memray>=2; extra == "memory"'])
    monkeypatch.setattr(
        "flameox.providers.environment.importlib.metadata.distribution", lambda _: distribution
    )
    monkeypatch.setattr(
        "flameox.providers.environment.importlib.metadata.version", lambda _: installed
    )
    assert active_provider_status(["memray"]) == expected


@pytest.mark.integration
@pytest.mark.parametrize("status", ["ready", "restart_required", "unknown"])
def test_mcp_preparation_reports_verified_or_conditional_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    monkeypatch.setattr("flameox.providers.preparation.active_provider_status", lambda _: status)

    async def run(self: ProviderDependencies, argv: list[str], timeout: int) -> bytes:
        return __version__.encode()

    monkeypatch.setattr(ProviderDependencies, "_run", run)

    async def exercise() -> None:
        async with Client(create_server(evidence_directory=tmp_path / "store")) as client:
            result = await client.call_tool(
                "prepare_providers", {"provider_ids": ["memray", "perf"]}
            )
            assert not result.is_error
            value = result.structured_content
            assert value["activation_status"] == status
            if status == "ready":
                assert value["next_action"] is None
            else:
                assert value["next_action"]["necessity"] == (
                    "required" if status == "restart_required" else "conditional"
                )
                assert "Preserve" in value["next_action"]["message"]
            assert isinstance(result.content[0], TextContent)
            assert "host readiness has not been verified" in result.content[0].text

    anyio.run(exercise)


def fake_collector_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    interpreter = sys.executable
    server = tmp_path / "server"
    server.mkdir()
    server_python = server / "python"
    server_python.symlink_to(interpreter)
    monkeypatch.setattr("flameox.providers.preparation.sys.executable", str(server_python))
    collector = tmp_path / "collector"
    collector.write_text(f"#!{interpreter}\nprint('py-spy {PY_SPY_VERSION}')\n")
    collector.chmod(0o755)
    uvx = tmp_path / "uvx"
    receipt = json.dumps({"version": PY_SPY_VERSION, "executable": str(collector)})
    uvx.write_text(f"#!{interpreter}\nprint({receipt!r})\n")
    uvx.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    return collector


@pytest.mark.process
@pytest.mark.skipif(os.name == "nt", reason="POSIX executable fixtures")
def test_prepare_collector_keeps_session_and_reuses_verified_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = fake_collector_environment(tmp_path, monkeypatch)

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        artifact = tmp_path / "input.txt"
        artifact.write_text("retain session")
        try:
            analysis = runtime.analyze("artifact.preview", [PathSource(path=str(artifact))], {})
            first = await runtime.dependencies.prepare(["py-spy", "perf"])
            assert first.activation_status == "ready"
            assert first.restart_required is False
            assert first.preparation_command[5] == f"py-spy=={PY_SPY_VERSION}"
            assert runtime._require_managed_executable("py-spy", "py-spy") == str(collector)
            second = await runtime.dependencies.prepare(["py-spy"])
            assert second.preparation_command == []
            host = await runtime.dependencies.prepare(["perf"])
            assert host.activation_status == "not_applicable"
            assert runtime.dependencies.py_spy_executable() == str(collector)
            assert runtime.preserve_evidence(analysis["analysis_id"])["artifact_count"] == 1
            collector.write_text(collector.read_text() + "# changed\n")
            with pytest.raises(RuntimeFailure, match="missing or changed"):
                runtime.dependencies.py_spy_executable()
        finally:
            runtime.close()

    anyio.run(exercise)


@pytest.mark.process
@pytest.mark.skipif(os.name == "nt", reason="POSIX executable fixtures")
def test_preparation_cancellation_cleans_up_and_does_not_activate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpreter = sys.executable
    fake_collector_environment(tmp_path, monkeypatch)
    uvx = tmp_path / "uvx"
    uvx.write_text(f"#!{interpreter}\nimport time\ntime.sleep(60)\n")

    async def exercise() -> None:
        runtime = AnalysisRuntime(evidence_directory=tmp_path / "store")
        try:
            with anyio.move_on_after(0.2) as scope:
                await runtime.dependencies.prepare(["py-spy"])
            assert scope.cancel_called
            assert runtime.dependencies.py_spy_executable() is None
            with pytest.raises(SetupFailure, match=r"timeout|complete"):
                await runtime.dependencies.prepare(["py-spy"], timeout_seconds=1)
            assert runtime.dependencies.py_spy_executable() is None
        finally:
            runtime.close()

    anyio.run(exercise)
