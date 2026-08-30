from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp_types import TextContent, TextResourceContents
from typer.testing import CliRunner

from flameox import __version__
from flameox.application.capabilities import managed_setup_adapter_names
from flameox.cli import app
from flameox.domain import RunManifest
from flameox.storage import RunStore, Workspace

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


@pytest.mark.anyio
@pytest.mark.process
@pytest.mark.serial
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
        instructions = client.instructions
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
        run_id = imported.structured_content["result"]["run_id"]
        resource = await client.read_resource(f"flameox://runs/{run_id}")
        structured_error = await client.call_tool(
            "get_run",
            {"run_id": "missing-run"},
        )

    assert "workspace_status" in {tool.name for tool in tools.tools}
    assert "start_capability_setup" in {tool.name for tool in tools.tools}
    setup_tool = next(tool for tool in tools.tools if tool.name == "start_capability_setup")
    adapter_schema = setup_tool.input_schema["properties"]["adapters"]
    adapter_definition = setup_tool.input_schema["$defs"][
        adapter_schema["items"]["$ref"].removeprefix("#/$defs/")
    ]
    assert tuple(adapter_definition["enum"]) == managed_setup_adapter_names()
    assert "aiperf" in adapter_definition["enum"]
    version_schema = setup_tool.input_schema["properties"]["memray_reader_version"]
    string_schema = next(item for item in version_schema["anyOf"] if item.get("type") == "string")
    assert string_schema["minLength"] == 1
    assert string_schema["maxLength"] == 100
    assert instructions is not None
    assert "get_declared_workflow" in instructions
    assert "start_capability_setup" in instructions
    inspected = await asyncio.to_thread(
        CliRunner().invoke,
        app,
        ["mcp", "inspect", "--project-root", str(tmp_path), "--json"],
    )
    assert inspected.exit_code == 0, inspected.output
    assert json.loads(inspected.stdout)["instructions"] == instructions
    assert status.is_error is False
    assert status.structured_content is not None
    assert status.structured_content["result"]["workspace_id"]
    assert capabilities.is_error is False
    assert isinstance(resource.contents[0], TextResourceContents)
    assert run_id in resource.contents[0].text
    assert structured_error.is_error is True
    assert structured_error.structured_content is not None
    assert structured_error.structured_content["error"]["code"] == "RUN_NOT_FOUND"
    assert structured_error.structured_content["error"]["remediation"] == [
        "Call list_runs to choose an existing run."
    ]
    assert structured_error.structured_content["error"]["recovery"] == {
        "kind": "tool",
        "safe_to_repeat_same_call": False,
        "retry_after_ms": None,
        "tool_name": "list_runs",
        "arguments": {"limit": 50},
    }


@pytest.mark.anyio
@pytest.mark.process
@pytest.mark.serial
async def test_sdk_client_validates_native_tool_output_schemas(tmp_path: Path) -> None:
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

    async with stdio_client(parameters) as streams, ClientSession(*streams) as session:
        initialized = await session.initialize()
        result = await session.list_tools()
        status = await session.call_tool("workspace_status", {})

    assert len(result.tools) >= 40
    assert initialized.server_info.version == __version__
    assert all(tool.output_schema is not None for tool in result.tools)
    assert status.is_error is False
    assert status.structured_content is not None
    assert status.structured_content["ok"] is True


@pytest.mark.anyio
@pytest.mark.process
@pytest.mark.serial
async def test_fresh_stdio_connections_negotiate_the_flameox_runtime_version(
    tmp_path: Path,
) -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "flameox", "mcp", "serve", "--project-root", str(tmp_path)],
        cwd=tmp_path,
    )

    for _ in range(2):
        async with stdio_client(parameters) as streams, ClientSession(*streams) as session:
            initialized = await session.initialize()
            tools = await session.list_tools()

        assert initialized.server_info.version == __version__
        assert initialized.protocol_version
        assert tools.tools


@pytest.mark.anyio
@pytest.mark.process
@pytest.mark.serial
async def test_real_stdio_discovers_then_plans_and_executes_declared_workload(
    tmp_path: Path,
) -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "flameox", "mcp", "serve", "--project-root", str(tmp_path)],
        cwd=tmp_path,
    )

    async with Client(stdio_client(parameters), raise_exceptions=True) as client:
        tools = await client.list_tools()
        initialized = await client.call_tool("initialize_workspace", {})
        assert initialized.is_error is False
        workspace = Workspace.discover(tmp_path)
        config = workspace.config.validated_copy(
            update={
                "execution": workspace.config.execution.validated_copy(
                    update={"containment": "disabled"}
                )
            }
        )
        workspace.paths.config.write_text(config.to_toml())
        status = await client.call_tool("workload_configuration_status", {})
        configured = await client.call_tool(
            "configure_workload",
            {
                "name": "probe",
                "operation": "create",
                "argv": [sys.executable, "-c", "print('{value}')"],
                "timeout_seconds": 5,
                "parameters": {"value": ["baseline", "candidate"]},
            },
        )
        assert RunStore(workspace).list() == ()
        declared = await client.call_tool(
            "list_declared_workflows",
            {"kind": "workload", "limit": 10},
        )
        assert declared.structured_content is not None
        assert declared.structured_content["result"]["workflows"][0]["name"] == "probe"
        detail = await client.call_tool(
            "get_declared_workflow",
            {"kind": "workload", "name": "probe"},
        )
        assert detail.structured_content is not None
        assert detail.structured_content["result"]["allowed_parameters"] == {
            "value": ["baseline", "candidate"]
        }
        planned = await client.call_tool(
            "plan_capture",
            {
                "workload_name": "probe",
                "adapter": "command",
                "parameters": {"value": "candidate"},
            },
        )
        assert planned.structured_content is not None
        executed = await client.call_tool(
            "execute_capture_plan",
            {"plan_token": planned.structured_content["result"]["plan_token"]},
        )

    assert {tool.name for tool in tools.tools} >= {
        "initialize_workspace",
        "workload_configuration_status",
        "configure_workload",
    }
    assert status.is_error is False
    assert status.structured_content is not None
    assert status.structured_content["result"]["status"] == "missing"
    assert configured.is_error is False
    assert configured.structured_content is not None
    assert planned.structured_content["result"]["execution_policy"] == "trusted_local"
    assert any(
        "Trusted-local execution selected" in warning
        for warning in planned.structured_content["result"]["warnings"]
    )
    assert any(
        item["code"] == "trusted_local_execution"
        for item in planned.structured_content["result"]["limitation_details"]
    )
    assert executed.is_error is False
    assert executed.structured_content is not None
    assert executed.structured_content["result"]["execution_status"] == "succeeded"


@pytest.mark.anyio
@pytest.mark.process
@pytest.mark.serial
async def test_real_stdio_server_returns_sdk_schema_validation_errors(
    tmp_path: Path,
) -> None:
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
        result = await client.call_tool("list_runs", {"limit": 0})
        unmanaged = await client.call_tool(
            "start_capability_setup",
            {"adapters": ["perf"]},
        )
        invalid_kind = await client.call_tool(
            "import_artifact",
            {"path": "profile.json", "kind": "normal", "sensitivity": "normal"},
        )

    assert result.is_error is True
    assert result.structured_content is None
    assert isinstance(result.content[0], TextContent)
    assert "limit" in result.content[0].text
    assert "greater than or equal to 1" in result.content[0].text
    assert unmanaged.is_error is True
    assert unmanaged.structured_content is None
    assert invalid_kind.is_error is True
    assert isinstance(invalid_kind.content[0], TextContent)
    assert "execution_trace" in invalid_kind.content[0].text


@pytest.mark.anyio
@pytest.mark.process
@pytest.mark.serial
async def test_real_stdio_server_reports_progress_and_propagates_cancellation(
    tmp_path: Path,
) -> None:
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.wait]
argv = ["python", "-c", "import time; time.sleep(30)"]
cwd = "."
timeout_seconds = 60
"""
    )
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
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

    def read_runs() -> list[RunManifest]:
        runs: list[RunManifest] = []
        runs.extend(RunStore(workspace).list())
        return runs

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
                {"plan_token": planned.structured_content["result"]["plan_token"]},
            )
        )
        for _ in range(200):
            runs = read_runs()
            if runs and runs[0].execution_status.value == "running":
                break
            await asyncio.sleep(0.05)
        assert runs and runs[0].execution_status.value == "running"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(100):
            runs = read_runs()
            if runs and runs[0].execution_status.value == "cancelled":
                break
            await asyncio.sleep(0.05)
        assert len(runs) == 1
        assert runs[0].execution_status.value == "cancelled"

    assert analyzed.is_error is False
    assert [item[0] for item in recorded_progress] == [0, 1, 2]
    assert runs[0].execution_status.value == "cancelled"
