from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_types import TextResourceContents
from typer.testing import CliRunner

from flameox.application import WorkloadService, workspace_status
from flameox.catalog import Catalog
from flameox.cli import app
from flameox.mcp import create_server
from flameox.storage import RunStore, Workspace


@pytest.mark.anyio
async def test_mcp_tools_use_explicit_envelopes_and_annotations(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        tools = await client.list_tools()
        by_name = {tool.name: tool for tool in tools.tools}
        result = await client.call_tool("workspace_status", {})

    assert by_name["workspace_status"].annotations is not None
    assert by_name["workspace_status"].annotations.read_only_hint is True
    assert by_name["import_artifact"].annotations is not None
    assert by_name["import_artifact"].annotations.read_only_hint is False
    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["ok"] is True
    assert result.structured_content["result"]["workspace_id"] == workspace.identity.workspace_id
    assert len(result.content) == 1


@pytest.mark.anyio
async def test_every_mcp_tool_has_bounded_object_schemas_and_annotations(
    tmp_path: Path,
) -> None:
    Workspace.initialize(tmp_path)
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        tools = (await client.list_tools()).tools

    assert len(tools) >= 40
    for tool in tools:
        assert tool.input_schema["type"] == "object"
        assert tool.output_schema is not None
        assert tool.output_schema["type"] == "object"
        assert {"ok", "result", "error"} <= set(tool.output_schema["properties"])
        assert tool.annotations is not None
        if tool.annotations.read_only_hint:
            assert tool.annotations.destructive_hint is False


@pytest.mark.anyio
async def test_mcp_domain_errors_remain_structured(tmp_path: Path) -> None:
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool("workspace_status", {})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["ok"] is False
    assert result.structured_content["error"]["code"] == "WORKSPACE_NOT_FOUND"


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
        run_id = imported.structured_content["result"]["run"]["run_id"]
        listed = await client.call_tool("list_runs", {"limit": 10})
        fetched = await client.call_tool("get_run", {"run_id": run_id})
        resource = await client.read_resource(f"flameox://runs/{run_id}")

    assert listed.structured_content is not None
    assert listed.structured_content["result"]["runs"][0]["run_id"] == run_id
    assert fetched.structured_content is not None
    assert fetched.structured_content["result"]["run_id"] == run_id
    contents = resource.contents[0]
    assert isinstance(contents, TextResourceContents)
    assert run_id in contents.text


@pytest.mark.anyio
async def test_real_stdio_server_keeps_protocol_on_stdout(tmp_path: Path) -> None:
    (tmp_path / "profile.json").write_text('{"samples": []}')
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "flameox",
            "mcp",
            "serve",
            "--project-root",
            str(tmp_path),
            "--init",
        ],
        cwd=tmp_path,
    )

    async with Client(stdio_client(parameters), raise_exceptions=True) as client:
        tools = await client.list_tools()
        status = await client.call_tool("workspace_status", {})
        capabilities = await client.call_tool("list_capabilities", {})
        imported = await client.call_tool(
            "import_artifact",
            {
                "path": "profile.json",
                "kind": "collector_metadata",
                "sensitivity": "internal",
            },
        )
        assert imported.structured_content is not None
        run_id = imported.structured_content["result"]["run"]["run_id"]
        resource = await client.read_resource(f"flameox://runs/{run_id}")
        structured_error = await client.call_tool(
            "get_run",
            {"run_id": "missing-run"},
        )

    assert "workspace_status" in {tool.name for tool in tools.tools}
    assert status.is_error is False
    assert status.structured_content is not None
    assert status.structured_content["result"]["workspace_id"]
    assert capabilities.is_error is False
    assert isinstance(resource.contents[0], TextResourceContents)
    assert run_id in resource.contents[0].text
    assert structured_error.is_error is True
    assert structured_error.structured_content is not None
    assert structured_error.structured_content["error"]["code"] == "WORKSPACE_INVALID"


@pytest.mark.anyio
async def test_real_stdio_server_reports_progress_and_propagates_cancellation(
    tmp_path: Path,
) -> None:
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.wait]
argv = ["python", "-c", "import time; time.sleep(30)"]
cwd = "."
timeout_seconds = 60
"""
    )
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    WorkloadService(workspace).approve("wait")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "flameox",
            "mcp",
            "serve",
            "--project-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
    )
    recorded_progress: list[tuple[float, float | None, str | None]] = []

    async def record_progress(
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        progress_events = (progress, total, message)
        recorded_progress.append(progress_events)

    async with Client(stdio_client(parameters), raise_exceptions=True) as client:
        analyzed = await client.call_tool(
            "analyze_failures",
            {"limit": 10},
            progress_callback=record_progress,
        )
        planned = await client.call_tool(
            "plan_capture",
            {"workload_name": "wait", "adapter": "command", "parameters": {}},
        )
        assert planned.structured_content is not None
        task = asyncio.create_task(
            client.call_tool(
                "execute_capture_plan",
                {"plan_id": planned.structured_content["result"]["plan_id"]},
            )
        )
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert analyzed.is_error is False
    assert [item[0] for item in recorded_progress] == [0, 1, 2]
    runs = [
        RunStore(workspace).read(path.name)
        for path in workspace.paths.runs.iterdir()
        if path.is_dir()
    ]
    assert len(runs) == 1
    assert runs[0].execution_status.value == "cancelled"


@pytest.mark.anyio
async def test_mcp_capture_requires_approval_and_plan_tokens_are_single_use(
    tmp_path: Path,
) -> None:
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.echo]
argv = ["python", "-c", "print('captured')"]
cwd = "."
timeout_seconds = 5
"""
    )
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    Catalog(workspace).rebuild()

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        refused = await client.call_tool(
            "plan_capture",
            {"workload_name": "echo", "adapter": "command", "parameters": {}},
        )
    assert refused.is_error is True
    assert refused.structured_content is not None
    assert refused.structured_content["error"]["code"] == "EXECUTION_REFUSED", (
        refused.structured_content
    )

    WorkloadService(workspace).approve("echo")
    recorded_progress: list[tuple[float, float | None, str | None]] = []

    async def record_progress(
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        recorded_progress.append((progress, total, message))

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        planned = await client.call_tool(
            "plan_capture",
            {"workload_name": "echo", "adapter": "command", "parameters": {}},
        )
        assert planned.is_error is False, planned.structured_content
        assert planned.structured_content is not None
        plan_id = planned.structured_content["result"]["plan_id"]
        executed = await client.call_tool(
            "execute_capture_plan",
            {"plan_id": plan_id},
            progress_callback=record_progress,
        )
        replayed = await client.call_tool(
            "execute_capture_plan",
            {"plan_id": plan_id},
        )

    assert executed.is_error is False
    assert executed.structured_content is not None
    assert executed.structured_content["result"]["run"]["run_id"]
    assert replayed.is_error is True
    assert replayed.structured_content is not None
    assert replayed.structured_content["error"]["code"] == "INVALID_CAPTURE_PLAN"
    assert [item[0] for item in recorded_progress] == list(range(9))
    assert {item[1] for item in recorded_progress} == {8}
    assert all(item[2] for item in recorded_progress)


@pytest.mark.anyio
async def test_mcp_cancellation_propagates_to_terminal_run_state(
    tmp_path: Path,
) -> None:
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.wait]
argv = ["python", "-c", "import time; time.sleep(30)"]
cwd = "."
timeout_seconds = 60
"""
    )
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    WorkloadService(workspace).approve("wait")

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        planned = await client.call_tool(
            "plan_capture",
            {"workload_name": "wait", "adapter": "command", "parameters": {}},
        )
        assert planned.structured_content is not None
        task = asyncio.create_task(
            client.call_tool(
                "execute_capture_plan",
                {"plan_id": planned.structured_content["result"]["plan_id"]},
            )
        )
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    runs = [
        RunStore(workspace).read(path.name)
        for path in workspace.paths.runs.iterdir()
        if path.is_dir()
    ]
    assert len(runs) == 1
    assert runs[0].execution_status.value == "cancelled"
    assert runs[0].process is not None
    assert runs[0].process.cleanup_complete is True
