from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcp import Client

from flameox.catalog import Catalog
from flameox.mcp import create_server
from flameox.storage import RunStore, Workspace


@pytest.mark.anyio
@pytest.mark.process
@pytest.mark.serial
async def test_mcp_capture_uses_current_workload_and_plan_tokens_are_single_use(
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
        assert executed.structured_content is not None
        run_id = executed.structured_content["result"]["run_id"]
        fetched = await client.call_tool("get_run", {"run_id": run_id})
        replayed = await client.call_tool(
            "execute_capture_plan",
            {"plan_id": plan_id},
        )

    assert executed.is_error is False
    assert executed.structured_content is not None
    assert executed.structured_content["result"]["resource_uri"] == f"flameox://runs/{run_id}"
    assert "run" not in executed.structured_content["result"]
    assert any(
        item.type == "resource_link" and item.uri == f"flameox://runs/{run_id}"
        for item in executed.content
    )
    assert replayed.is_error is True
    assert replayed.structured_content is not None
    assert replayed.structured_content["error"]["code"] == "INVALID_CAPTURE_PLAN"
    assert replayed.structured_content["error"]["recovery"] == {
        "kind": "replan_capture",
        "safe_to_repeat_same_call": False,
        "retry_after_ms": None,
        "next_tool": "plan_capture",
    }
    assert fetched.structured_content is not None
    assert fetched.structured_content["result"]["run_id"] == run_id
    assert [item[0] for item in recorded_progress] == list(range(9))
    assert {item[1] for item in recorded_progress} == {8}
    assert all(item[2] for item in recorded_progress)


@pytest.mark.anyio
async def test_mcp_plan_capture_rejects_adhoc_argv_and_cwd_inputs(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.probe]
argv = ["python", "-c", "print('ok')"]
"""
    )

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        argv = await client.call_tool(
            "plan_capture",
            {
                "workload_name": "probe",
                "adapter": "command",
                "parameters": {},
                "argv": ["echo", "unsafe"],
            },
        )
        cwd = await client.call_tool(
            "plan_capture",
            {
                "workload_name": "probe",
                "adapter": "command",
                "parameters": {},
                "cwd": ".",
            },
        )

    assert workspace.project_root == tmp_path.resolve()
    for result, field in ((argv, "argv"), (cwd, "cwd")):
        assert result.is_error is True
        assert result.structured_content is not None
        assert result.structured_content["error"]["code"] == "INVALID_ARGUMENTS"
        assert result.structured_content["error"]["details"]["fields"][0]["field"] == field


@pytest.mark.anyio
async def test_mcp_plan_missing_dependency_routes_to_managed_preparation(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.probe]
argv = ["python", "-c", "print('ok')"]
[workloads.probe.requirements]
python_distributions = ["flameox-agent-fixture>=99"]
"""
    )

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool(
            "plan_capture",
            {"workload_name": "probe", "adapter": "command", "parameters": {}},
        )

    assert result.is_error is True
    assert result.structured_content is not None
    error = result.structured_content["error"]
    assert error["code"] == "CAPABILITY_UNAVAILABLE"
    assert error["details"]["next_tool"] == "prepare_workload_dependencies"
    assert error["details"]["missing_python_distributions"] == ["flameox-agent-fixture>=99"]
    assert error["recovery"] == {
        "kind": "prepare_workload_dependencies",
        "safe_to_repeat_same_call": True,
        "retry_after_ms": None,
        "next_tool": "prepare_workload_dependencies",
    }


@pytest.mark.anyio
@pytest.mark.process
@pytest.mark.serial
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
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        planned = await client.call_tool(
            "plan_capture",
            {"workload_name": "wait", "adapter": "command", "parameters": {}},
        )
        assert planned.structured_content is not None
        collector_started = asyncio.Event()

        async def record_progress(
            progress: float,
            total: float | None,
            message: str | None,
        ) -> None:
            del total, message
            if progress == 4:
                collector_started.set()

        task = asyncio.create_task(
            client.call_tool(
                "execute_capture_plan",
                {"plan_id": planned.structured_content["result"]["plan_id"]},
                progress_callback=record_progress,
            )
        )
        await asyncio.wait_for(collector_started.wait(), timeout=5)
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


@pytest.mark.anyio
@pytest.mark.process
@pytest.mark.serial
async def test_mcp_detached_capture_start_reconnect_and_cancellation(
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
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        planned = await client.call_tool(
            "plan_capture",
            {"workload_name": "wait", "adapter": "command", "parameters": {}},
        )
        assert planned.structured_content is not None
        arguments = {
            "plan_id": planned.structured_content["result"]["plan_id"],
            "idempotency_key": "transport-retry-001",
        }
        started = await client.call_tool("start_detached_capture", arguments)
        repeated = await client.call_tool("start_detached_capture", arguments)
        assert started.structured_content is not None
        assert repeated.structured_content is not None
        run_id = started.structured_content["result"]["run_id"]
        reconnected = await client.call_tool(
            "get_detached_capture",
            {"run_id": run_id},
        )
        cancelled = await client.call_tool(
            "cancel_detached_capture",
            {"run_id": run_id},
        )

    assert tools["start_detached_capture"].annotations is not None
    assert tools["start_detached_capture"].annotations.idempotent_hint is True
    assert repeated.structured_content["result"]["run_id"] == run_id
    assert reconnected.structured_content is not None
    assert reconnected.structured_content["result"]["state"] == "running"
    assert cancelled.structured_content is not None
    assert cancelled.structured_content["result"]["state"] == "terminal"
    assert cancelled.structured_content["result"]["execution_status"] == "cancelled"
