from __future__ import annotations

import sys
from pathlib import Path

import anyio
import pytest
from mcp import Client
from mcp_types import TextContent

from flameox.mcp.server import FlameoxServer
from flameox.repository import EvidenceRepository


@pytest.mark.integration
def test_mcp_unavailable_provider_names_preparation_and_capture_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def exercise() -> None:
        async with Client(
            FlameoxServer(evidence_directory=tmp_path / ".flameox"), raise_exceptions=True
        ) as client:
            empty_path = tmp_path / "empty-bin"
            empty_path.mkdir()
            unmanaged_python = empty_path / "python"
            unmanaged_python.symlink_to(sys.executable)
            monkeypatch.setattr("flameox.runtime.sys.executable", str(unmanaged_python))
            unavailable = await client.call_tool(
                "capture_cpu_hotspots",
                {
                    "target": {
                        "argv": [sys.executable, "-c", "pass"],
                        "cwd": str(tmp_path),
                        "environment": {"PATH": str(empty_path)},
                    },
                    "provider": {"kind": "py-spy"},
                },
            )
            assert unavailable.is_error is False
            assert unavailable.structured_content["status"] == "retryable"
            assert unavailable.structured_content["code"] == "UNAVAILABLE_CAPABILITY"
            assert unavailable.structured_content["details"] == {
                "provider_id": "py-spy",
            }
            next_action = unavailable.structured_content["next_action"]
            assert next_action["tool"] == "prepare_providers"
            assert next_action["arguments"] == {"provider_ids": ["py-spy"]}
            assert next_action["then_retry"] == "capture_cpu_hotspots"

    anyio.run(exercise)


@pytest.mark.integration
@pytest.mark.process
def test_mcp_oracle_failure_does_not_become_a_workload_failure(tmp_path: Path) -> None:
    async def exercise() -> None:
        async with Client(FlameoxServer(evidence_directory=tmp_path / "store")) as client:
            result = await client.call_tool(
                "capture_artifact_preview",
                {
                    "provider": {"kind": "direct"},
                    "target": {
                        "argv": [sys.executable, "-c", "print('captured')"],
                        "cwd": str(tmp_path),
                    },
                    "experiment": {
                        "cases": [{"name": "baseline"}, {"name": "candidate"}],
                        "blocks": 1,
                        "seed": 7,
                        "metric": "wall_time_ns",
                        "estimand": "median_difference",
                        "practical_threshold": 0,
                        "semantic_oracle": [sys.executable, "-c", "raise SystemExit(1)"],
                    },
                },
            )

        assert result.is_error is False
        value = result.structured_content
        assert value["status"] == "partial"
        assert value["capture"]["outcome"]["status"] == "failed"
        assert value["capture"]["workload_status"] == "succeeded"
        assert value["capture"]["mode"] == "experiment"
        assert {item["case"] for item in value["capture"]["executions"]} == {
            "baseline",
            "candidate",
        }
        assert all(item["workload_returncode"] == 0 for item in value["capture"]["executions"])

    anyio.run(exercise)


@pytest.mark.process
def test_failed_capture_returns_full_provenance_once(tmp_path: Path) -> None:
    directory = tmp_path / "store"
    arguments = [f"native-argument-{index}:".ljust(16_384, "x") for index in range(8)]

    async def exercise() -> str:
        async with Client(
            FlameoxServer(
                evidence_directory=directory,
            )
        ) as client:
            result = await client.call_tool(
                "capture_artifact_preview",
                {
                    "target": {
                        "argv": [sys.executable, "-c", "raise SystemExit(7)", *arguments],
                        "cwd": str(tmp_path),
                    },
                    "provider": {"kind": "direct"},
                },
            )
            assert not result.is_error
            value = result.structured_content
            assert value is not None, result.content
            assert value["status"] == "partial"
            summary = result.content[0]
            assert isinstance(summary, TextContent)
            assert "native-argument-" not in summary.text
            assert value["capture"]["executions"][0]["argv"][3:] == arguments
            preserved = await client.call_tool(
                "preserve_evidence", {"analysis_id": value["analysis_id"]}
            )
            assert not preserved.is_error
            return str(preserved.structured_content["evidence_id"])

    evidence_id = anyio.run(exercise)
    manifest = EvidenceRepository(directory, "reader").read(evidence_id)
    execution = manifest["body"]["capture_request"]["executions"][0]
    assert execution["argv"][3:] == arguments
    assert execution["returncode"] == 7
    assert execution["status"] == "failed"
