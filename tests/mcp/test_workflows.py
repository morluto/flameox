from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from mcp import Client
from mcp_types import TextResourceContents
from typer.testing import CliRunner

from flameox.application import workspace_status
from flameox.catalog import Catalog
from flameox.cli import app
from flameox.mcp import create_server
from flameox.storage import Workspace


@pytest.mark.anyio
async def test_unknown_declared_workflow_routes_back_to_discovery(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text("schema_version = 1\n")
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool(
            "get_declared_workflow",
            {"kind": "workload", "name": "missing"},
        )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["error"]["recovery"]["kind"] == "discover_workflows"
    assert result.structured_content["error"]["recovery"]["next_tool"] == "list_declared_workflows"


@pytest.mark.anyio
async def test_missing_workload_configuration_routes_to_configure_workload(
    tmp_path: Path,
) -> None:
    Workspace.initialize(tmp_path)
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        status = await client.call_tool("workload_configuration_status", {})
        missing = await client.call_tool(
            "list_declared_workflows",
            {},
        )
        configured = await client.call_tool(
            "configure_workload",
            {
                "name": "probe",
                "operation": "create",
                "argv": ["python", "-c", "print('ok')"],
            },
        )
        discovered = await client.call_tool(
            "list_declared_workflows",
            {},
        )

    assert status.is_error is False
    assert status.structured_content is not None
    assert status.structured_content["result"] == {
        "schema_version": 1,
        "status": "missing",
        "config_path": "flameox.toml",
        "configuration_id": None,
        "workload_names": [],
        "diagnostics": ["No named workload configuration exists yet."],
        "next_tool": "configure_workload",
    }
    assert missing.is_error is True
    assert missing.structured_content is not None
    assert missing.structured_content["error"]["recovery"] == {
        "kind": "configure_workload",
        "safe_to_repeat_same_call": False,
        "retry_after_ms": None,
        "next_tool": "configure_workload",
    }
    assert configured.is_error is False
    assert configured.structured_content is not None
    assert configured.structured_content["result"]["action"] == "created"
    assert configured.structured_content["result"]["configuration_source"] == "agent"
    assert configured.structured_content["result"]["changed_paths"] == [
        "flameox.toml",
    ]
    assert discovered.is_error is False
    assert discovered.structured_content is not None
    assert discovered.structured_content["result"]["workflows"][0]["name"] == "probe"


@pytest.mark.anyio
async def test_invalid_capture_adapter_returns_bounded_workflow_recovery(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.probe]
argv = ["python", "-c", "print('ok')"]
"""
    )
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool(
            "plan_capture",
            {
                "workload_name": "probe",
                "adapter": "not-a-capture-adapter",
                "parameters": {},
            },
        )

    assert result.is_error is True
    assert result.structured_content is not None
    error = result.structured_content["error"]
    assert error["details"]["allowed_adapters"]
    assert error["recovery"]["next_tool"] == "get_declared_workflow"


@pytest.mark.anyio
async def test_cli_json_and_mcp_result_are_same_domain_model(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    expected = workspace_status(workspace).model_dump(mode="json")
    cli = CliRunner().invoke(
        app,
        ["status", "--workspace", str(workspace.paths.root), "--json"],
    )
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        mcp = await client.call_tool("workspace_status", {})

    assert cli.exit_code == 0, cli.output
    assert mcp.structured_content is not None
    assert json.loads(cli.stdout) == expected
    assert mcp.structured_content["result"] == expected


@pytest.mark.anyio
async def test_mcp_inspect_instructions_match_initialize_metadata(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        await client.list_tools()
        initialize_instructions = client.instructions

    cli = await asyncio.to_thread(
        CliRunner().invoke,
        app,
        ["mcp", "inspect", "--project-root", str(tmp_path), "--json"],
    )

    assert cli.exit_code == 0, cli.output
    inspected = json.loads(cli.stdout)
    assert inspected["schema_version"] == 1
    assert inspected["instructions"] == initialize_instructions
    assert len(inspected["tools"]) == 66


@pytest.mark.anyio
async def test_mcp_analysis_reports_named_query_phases(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    recorded_progress: list[tuple[float, float | None, str | None]] = []

    async def record(
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        recorded_progress.append((progress, total, message))

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool(
            "analyze_failures",
            {"limit": 10},
            progress_callback=record,
        )

    assert result.is_error is False
    assert [item[0] for item in recorded_progress] == [0, 1, 2]
    assert {item[1] for item in recorded_progress} == {2}
    assert all(item[2] for item in recorded_progress)


@pytest.mark.anyio
async def test_mcp_import_list_get_and_resource_workflow(tmp_path: Path) -> None:
    artifact = tmp_path / "profile.json"
    artifact.write_text('{"samples": []}')
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        imported = await client.call_tool(
            "import_artifact",
            {
                "path": "profile.json",
                "kind": "collector_metadata",
                "sensitivity": "internal",
            },
        )
        assert imported.structured_content is not None
        result = imported.structured_content["result"]
        run_id = result["run_id"]
        artifact_id = result["artifact_id"]
        listed = await client.call_tool("list_runs", {"limit": 10})
        fetched = await client.call_tool("get_run", {"run_id": run_id})
        resource = await client.read_resource(f"flameox://runs/{run_id}")
        artifact_resource = await client.read_resource(f"flameox://artifacts/{artifact_id}")

    assert "run" not in result
    assert {item.uri for item in imported.content if item.type == "resource_link"} == {
        f"flameox://runs/{run_id}",
        f"flameox://artifacts/{artifact_id}",
    }
    assert listed.structured_content is not None
    assert listed.structured_content["result"]["runs"][0]["run_id"] == run_id
    assert listed.structured_content["result"]["runs"][0]["artifact_kinds"] == [
        "collector_metadata"
    ]
    assert fetched.structured_content is not None
    assert fetched.structured_content["result"]["run_id"] == run_id
    contents = resource.contents[0]
    assert isinstance(contents, TextResourceContents)
    assert run_id in contents.text
    artifact_contents = artifact_resource.contents[0]
    assert isinstance(artifact_contents, TextResourceContents)
    assert artifact_id in artifact_contents.text


@pytest.mark.anyio
async def test_mcp_run_discovery_filters_pages_and_rejects_stale_cursor(tmp_path: Path) -> None:
    for name in ("one.json", "two.json", "three.json"):
        (tmp_path / name).write_text(f'{{"name": "{name}"}}')
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        for name in ("one.json", "two.json"):
            imported = await client.call_tool(
                "import_artifact",
                {
                    "path": name,
                    "kind": "collector_metadata",
                    "sensitivity": "internal",
                },
            )
            assert imported.is_error is False
        first = await client.call_tool(
            "list_runs",
            {
                "limit": 1,
                "filter": {
                    "execution_status": ["not_applicable"],
                    "validation_status": ["not_requested"],
                },
            },
        )
        assert first.structured_content is not None
        first_result = first.structured_content["result"]
        cursor = first_result["next_cursor"]
        second = await client.call_tool(
            "list_runs",
            {
                "limit": 1,
                "cursor": cursor,
                "filter": {
                    "execution_status": ["not_applicable"],
                    "validation_status": ["not_requested"],
                },
            },
        )
        empty = await client.call_tool(
            "list_runs",
            {"limit": 10, "filter": {"environment_id": "sha256:" + "0" * 64}},
        )
        await client.call_tool(
            "import_artifact",
            {
                "path": "three.json",
                "kind": "collector_metadata",
                "sensitivity": "internal",
            },
        )
        stale = await client.call_tool(
            "list_runs",
            {
                "limit": 1,
                "cursor": cursor,
                "filter": {
                    "execution_status": ["not_applicable"],
                    "validation_status": ["not_requested"],
                },
            },
        )

    assert cursor is not None
    assert first_result["coverage"]["filters_applied"] == [
        "execution_status",
        "validation_status",
    ]
    assert second.structured_content is not None
    assert (
        second.structured_content["result"]["runs"][0]["run_id"]
        != first_result["runs"][0]["run_id"]
    )
    assert empty.structured_content is not None
    assert empty.structured_content["result"]["total"] == 0
    assert stale.is_error is True
    assert stale.structured_content is not None
    assert stale.structured_content["error"]["code"] == "STALE_CURSOR"


@pytest.mark.anyio
async def test_mcp_import_accepts_an_absolute_path_inside_the_explicit_temp_root(
    tmp_path: Path,
) -> None:
    external = tmp_path.parent / "flameox-agent-trace.json"
    external.write_text('{"traceEvents": []}')
    Workspace.initialize(tmp_path)
    try:
        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            imported = await client.call_tool(
                "import_artifact",
                {
                    "path": str(external),
                    "source_root": "temp",
                    "kind": "execution_trace",
                    "sensitivity": "normal",
                },
            )
        assert imported.is_error is False
    finally:
        external.unlink(missing_ok=True)
