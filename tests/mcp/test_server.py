from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_types import TextResourceContents

from flamo.application import WorkloadService
from flamo.catalog import Catalog
from flamo.mcp import create_server
from flamo.storage import RunStore, Workspace


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
async def test_mcp_domain_errors_remain_structured(tmp_path: Path) -> None:
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool("workspace_status", {})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["ok"] is False
    assert result.structured_content["error"]["code"] == "WORKSPACE_NOT_FOUND"


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
        resource = await client.read_resource(f"flamo://runs/{run_id}")

    assert listed.structured_content is not None
    assert listed.structured_content["result"]["runs"][0]["run_id"] == run_id
    assert fetched.structured_content is not None
    assert fetched.structured_content["result"]["run_id"] == run_id
    contents = resource.contents[0]
    assert isinstance(contents, TextResourceContents)
    assert run_id in contents.text


@pytest.mark.anyio
async def test_real_stdio_server_keeps_protocol_on_stdout(tmp_path: Path) -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "flamo",
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

    assert "workspace_status" in {tool.name for tool in tools.tools}
    assert status.is_error is False
    assert status.structured_content is not None
    assert status.structured_content["result"]["workspace_id"]
    assert capabilities.is_error is False


@pytest.mark.anyio
async def test_mcp_capture_requires_approval_and_plan_tokens_are_single_use(
    tmp_path: Path,
) -> None:
    (tmp_path / "flamo.toml").write_text(
        """
schema_version = 1
[workloads.echo]
argv = ["python", "-c", "print('captured')"]
cwd = "."
timeout_seconds = 5
"""
    )
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        refused = await client.call_tool(
            "plan_capture",
            {"workload_name": "echo", "adapter": "command", "parameters": {}},
        )
    assert refused.is_error is True
    assert refused.structured_content is not None
    assert refused.structured_content["error"]["code"] == "EXECUTION_REFUSED"

    WorkloadService(workspace).approve("echo")
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        planned = await client.call_tool(
            "plan_capture",
            {"workload_name": "echo", "adapter": "command", "parameters": {}},
        )
        assert planned.structured_content is not None
        plan_id = planned.structured_content["result"]["plan_id"]
        executed = await client.call_tool(
            "execute_capture_plan",
            {"plan_id": plan_id},
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


@pytest.mark.anyio
async def test_mcp_cancellation_propagates_to_terminal_run_state(
    tmp_path: Path,
) -> None:
    (tmp_path / "flamo.toml").write_text(
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
