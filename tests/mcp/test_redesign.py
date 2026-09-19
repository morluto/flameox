from __future__ import annotations

from pathlib import Path
from typing import cast

import anyio
import pytest
from mcp import Client

from flameox.mcp import create_server
from flameox.repository import EvidenceRepository


@pytest.mark.integration
def test_capability_discovery_returns_exact_contract_only_on_drill_down(tmp_path: Path) -> None:
    async def exercise() -> None:
        async with Client(create_server(evidence_directory=tmp_path / "store")) as client:
            compact = await client.call_tool(
                "inspect_capabilities", {"mode": "list", "artifact_format": "pstats"}
            )
            exact = await client.call_tool(
                "inspect_capabilities", {"mode": "get", "capability_id": "cpu.hotspots"}
            )
            comparison = await client.call_tool(
                "inspect_capabilities", {"mode": "get", "capability_id": "benchmark.compare"}
            )

        assert compact.is_error is False
        assert {item["capability_id"] for item in compact.structured_content["capabilities"]} == {
            "cpu.hotspots",
            "cpu.callers",
        }
        assert all(
            "analysis_option_schema" not in item
            for item in compact.structured_content["capabilities"]
        )
        capability = exact.structured_content["capabilities"][0]
        assert capability["accepted_formats"] == [
            "cpuprofile",
            "pstats",
            "py-spy",
            "perf",
            "perf-data",
        ]
        assert capability["analysis_option_schema"]["properties"]["metric"]
        assert {item["id"] for item in capability["capture_providers"]} == {
            "node-cpu-profile",
            "perf",
            "py-spy",
        }
        assert all(item["option_schema"] is not None for item in capability["capture_providers"])
        assert (
            len(
                comparison.structured_content["capabilities"][0]["analysis_example"]["request"][
                    "sources"
                ]
            )
            == 2
        )

    anyio.run(exercise)


@pytest.mark.integration
def test_query_accepts_historical_capability_ids(tmp_path: Path) -> None:
    async def exercise() -> None:
        async with Client(create_server(evidence_directory=tmp_path / "store")) as client:
            result = await client.call_tool(
                "query_evidence", {"capability_id": "retired.capability"}
            )

        assert result.is_error is False
        assert result.structured_content["inventory_status"] == "absent"

    anyio.run(exercise)


@pytest.mark.integration
def test_capability_discovery_requires_an_actionable_mode(tmp_path: Path) -> None:
    async def exercise() -> None:
        async with Client(create_server(evidence_directory=tmp_path / "store")) as client:
            result = await client.call_tool(
                "inspect_capabilities", {"capability_id": "cpu.hotspots"}
            )

        assert result.is_error is True
        assert result.structured_content["code"] == "INVALID_REQUEST"
        assert result.structured_content["field_path"] == ["mode"]
        assert result.structured_content["accepted_values"] == ["list", "get"]

    anyio.run(exercise)


@pytest.mark.integration
def test_query_distinguishes_unavailable_empty_and_no_matches(tmp_path: Path) -> None:
    unavailable_store = tmp_path / "unavailable"
    empty_store = tmp_path / "empty"
    EvidenceRepository(empty_store, "test").initialize()

    async def query(directory: Path) -> dict[str, object]:
        async with Client(create_server(evidence_directory=directory)) as client:
            result = await client.call_tool("query_evidence", {})
            assert result.is_error is False
            return cast(dict[str, object], result.structured_content)

    async def populated_no_matches(directory: Path) -> dict[str, object]:
        artifact = tmp_path / "artifact.json"
        artifact.write_text("[1]")
        async with Client(create_server(evidence_directory=directory)) as client:
            analyzed = await client.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "artifact.preview",
                        "sources": [{"kind": "path", "path": str(artifact)}],
                    }
                },
            )
            await client.call_tool(
                "preserve_evidence",
                {"analysis_id": analyzed.structured_content["analysis_id"]},
            )
            result = await client.call_tool("query_evidence", {"capability_id": "cpu.hotspots"})
            return cast(dict[str, object], result.structured_content)

    unavailable = anyio.run(query, unavailable_store)
    empty = anyio.run(query, empty_store)
    no_matches = anyio.run(populated_no_matches, tmp_path / "populated")
    assert unavailable["inventory_status"] == "absent"
    assert unavailable["inventory_size"] == 0
    assert empty["inventory_status"] == "empty"
    assert empty["inventory_size"] == 0
    assert empty["match_status"] == "no_matches"
    assert no_matches["inventory_status"] == "available"
    assert no_matches["inventory_size"] == 1
    assert no_matches["match_status"] == "no_matches"


@pytest.mark.integration
def test_explicit_incompatible_format_fails_before_path_resolution(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.json"

    async def exercise() -> None:
        async with Client(create_server(evidence_directory=tmp_path / "store")) as client:
            result = await client.call_tool(
                "analyze",
                {
                    "request": {
                        "capability_id": "cpu.hotspots",
                        "sources": [{"kind": "path", "path": str(missing), "format": "json"}],
                    }
                },
            )

        assert result.is_error is True
        assert result.structured_content["code"] == "UNSUPPORTED_FORMAT"
        assert result.structured_content["field_path"] == [
            "request",
            "sources",
            0,
            "format",
        ]
        assert "pstats" in result.structured_content["accepted_values"]
        assert result.structured_content["next_action"]["kind"] == "adjust_request"
        assert result.structured_content["details"]["source_index"] == 0
        assert "pstats" in result.structured_content["details"]["accepted_formats"]

    anyio.run(exercise)
