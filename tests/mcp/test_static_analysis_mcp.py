from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp import Client
from mcp_types import TextResourceContents

from flameox.mcp import create_server
from flameox.storage import Workspace

pytestmark = [pytest.mark.integration, pytest.mark.serial]


def _write_sarif(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {"driver": {"name": "example-analyzer", "version": "1"}},
                        "results": [
                            {
                                "ruleId": "example.first",
                                "message": {"text": "first candidate"},
                                "locations": [
                                    {
                                        "physicalLocation": {
                                            "artifactLocation": {"uri": "first.py"},
                                            "region": {"startLine": 1},
                                        }
                                    }
                                ],
                            },
                            {
                                "ruleId": "example.second",
                                "message": {"text": "second candidate"},
                                "locations": [
                                    {
                                        "physicalLocation": {
                                            "artifactLocation": {"uri": "second.py"},
                                            "region": {"startLine": 2},
                                        }
                                    }
                                ],
                            },
                        ],
                    }
                ],
            }
        )
    )


def _write_many_run_sarif(path: Path, count: int = 24) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {
                            "driver": {
                                "name": f"analyzer-{index}-{'x' * 5_000}",
                                "version": "v" * 5_000,
                            }
                        },
                        "results": [
                            {
                                "ruleId": f"example.{index}",
                                "message": {"text": f"candidate {index}"},
                                "locations": [
                                    {
                                        "physicalLocation": {
                                            "artifactLocation": {"uri": "app.py"},
                                            "region": {"startLine": 1},
                                        }
                                    }
                                ],
                            }
                        ],
                    }
                    for index in range(count)
                ],
            }
        )
    )


@pytest.mark.anyio
async def test_mcp_static_analysis_import_and_cursor_resource_workflow(tmp_path: Path) -> None:
    (tmp_path / "first.py").write_text("pass\n")
    (tmp_path / "second.py").write_text("pass\n")
    _write_sarif(tmp_path / "analysis.sarif")
    Workspace.initialize(tmp_path)

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
        imported = await client.call_tool(
            "import_static_analysis",
            {"path": "analysis.sarif", "source_root": "."},
        )
        assert imported.structured_content is not None
        import_result = imported.structured_content["result"]
        run_id = import_result["run_id"]
        candidates_uri = import_result["candidates_resource_uri"]
        first_page = await client.call_tool(
            "query_static_candidates",
            {"run_id": run_id, "limit": 1},
        )
        assert first_page.structured_content is not None
        cursor = first_page.structured_content["result"]["next_cursor"]
        assert cursor is not None
        second_page = await client.call_tool(
            "query_static_candidates",
            {"run_id": run_id, "limit": 1, "cursor": cursor},
        )
        resource = await client.read_resource(candidates_uri)

    assert {"import_static_analysis", "query_static_candidates"} <= tools
    assert import_result["coverage"]["normalized_count"] == 2
    assert import_result["semantics"]["source_root"] == "."
    assert import_result["semantics"]["source_root_truncated"] is False
    assert import_result["semantics"]["analyzers"] == [{"name": "example-analyzer", "version": "1"}]
    assert import_result["source_state_id"] is not None
    assert {item.uri for item in imported.content if item.type == "resource_link"} == {
        import_result["run_resource_uri"],
        import_result["artifact_resource_uri"],
        candidates_uri,
    }
    assert first_page.structured_content["result"]["total"] == 2
    assert second_page.structured_content is not None
    assert second_page.structured_content["result"]["next_cursor"] is None
    contents = resource.contents[0]
    assert isinstance(contents, TextResourceContents)
    assert '"total": 2' in contents.text


@pytest.mark.anyio
async def test_mcp_static_analysis_projection_bounds_many_analyzers(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("pass\n")
    _write_many_run_sarif(tmp_path / "analysis.sarif")
    Workspace.initialize(tmp_path)

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        imported = await client.call_tool(
            "import_static_analysis",
            {"path": "analysis.sarif", "source_root": "."},
        )

    assert imported.structured_content is not None
    result = imported.structured_content["result"]
    semantics = result["semantics"]
    analyzers = semantics["analyzers"]
    assert len(analyzers) == 16
    assert all(len(analyzer["name"]) <= 256 for analyzer in analyzers)
    assert all(len(analyzer["version"]) <= 256 for analyzer in analyzers)
    assert result["coverage"]["normalized_count"] == 24
    assert any("provenance record(s) were omitted" in item for item in result["limitations"])
    assert any("fields truncated" in item for item in result["limitations"])
    assert len(json.dumps(semantics)) < 12_000
