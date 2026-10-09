from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anyio
import pytest
from mcp import Client

from flameox.mcp import create_server
from flameox.providers.preparation import PY_SPY_VERSION


@pytest.mark.integration
def test_host_provider_preparation_reports_requirements_without_installing(tmp_path: Path) -> None:
    async def exercise() -> None:
        async with Client(create_server(evidence_directory=tmp_path / "store")) as client:
            result = await client.call_tool("prepare_providers", {"provider_ids": ["xctrace"]})
            assert not result.is_error
            value = result.structured_content
            assert value["preparation"]["status"] == "not_applicable"
            assert value["prepared_managed_providers"] == []
            assert value["external_requirements"][0]["provider_id"] == "xctrace"
            assert "Xcode" in value["external_requirements"][0]["guidance"]
            assert value["next_action"] is None

    anyio.run(exercise)
    assert not (tmp_path / "store").exists()


@pytest.mark.integration
@pytest.mark.process
@pytest.mark.skipif(os.name == "nt", reason="POSIX executable fixtures")
def test_prepare_activates_verified_collector_for_the_live_session_and_reuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = sys.executable
    server = tmp_path / "server"
    server.mkdir()
    server_python = server / "python"
    server_python.symlink_to(interpreter)
    monkeypatch.setattr("flameox.providers.preparation.sys.executable", str(server_python))

    collector = tmp_path / "py-spy"
    collector.write_text(f"#!{interpreter}\nprint('py-spy {PY_SPY_VERSION}')\n")
    collector.chmod(0o755)
    receipt = json.dumps({"version": PY_SPY_VERSION, "executable": str(collector)})
    invocations = tmp_path / "uvx-invocations"
    environment_records = tmp_path / "installer-environments"
    uvx = tmp_path / "uvx"
    uvx.write_text(
        f"#!{interpreter}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"Path({str(invocations)!r}).open('a').write('called\\n')\n"
        f"with Path({str(environment_records)!r}).open('a') as output: "
        "output.write(json.dumps(dict(os.environ)) + '\\n')\n"
        "if '--version' in sys.argv and any('flameox[memray]' in arg for arg in sys.argv):\n"
        "    raise SystemExit(7)\n"
        f"print({receipt!r})\n"
    )
    uvx.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_NO_CONFIG", "1")
    monkeypatch.setenv("PYTHONPATH", "/untrusted-import-path")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-forward")
    monkeypatch.setattr("flameox.providers.preparation.active_provider_status", lambda _: "unknown")

    async def exercise() -> None:
        async with Client(create_server(evidence_directory=tmp_path / "store")) as client:
            rejected = await client.call_tool(
                "prepare_providers", {"provider_ids": ["py-spy", "memray"]}
            )
            assert rejected.is_error
            assert rejected.structured_content["code"] == "SETUP_FAILURE"

            unavailable = await client.call_tool(
                "capture_cpu_hotspots",
                {
                    "target": {"argv": [sys.executable, "-c", "pass"], "cwd": str(tmp_path)},
                    "provider": {"kind": "py-spy"},
                },
            )
            assert not unavailable.is_error
            assert unavailable.structured_content["status"] == "retryable"
            assert unavailable.structured_content["code"] == "UNAVAILABLE_CAPABILITY"

            prepared = await client.call_tool("prepare_providers", {"provider_ids": ["py-spy"]})
            assert not prepared.is_error
            value = prepared.structured_content
            assert value["activation_status"] == "ready"
            assert value["next_action"] is None
            assert value["prepared_managed_providers"] == ["py-spy"]
            assert len(invocations.read_text().splitlines()) == 3
            environments = [
                json.loads(line) for line in environment_records.read_text().splitlines()
            ]
            assert all(item.get("UV_OFFLINE") == "1" for item in environments)
            assert all(item.get("UV_NO_CONFIG") == "1" for item in environments)
            assert all("PYTHONPATH" not in item for item in environments)
            assert all("UNRELATED_SECRET" not in item for item in environments)

            repeated = await client.call_tool("prepare_providers", {"provider_ids": ["py-spy"]})
            assert not repeated.is_error
            assert len(invocations.read_text().splitlines()) == 3

            collector.write_text(collector.read_text() + "# changed\n")
            failed = await client.call_tool(
                "capture_cpu_hotspots",
                {
                    "target": {"argv": [sys.executable, "-c", "pass"], "cwd": str(tmp_path)},
                    "provider": {"kind": "py-spy"},
                },
            )
            assert not failed.is_error
            assert failed.structured_content["status"] == "retryable"
            assert failed.structured_content["code"] == "UNAVAILABLE_CAPABILITY"
            assert failed.structured_content["next_action"]["tool"] == "prepare_providers"

    anyio.run(exercise)
