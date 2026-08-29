from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import pytest
from mcp import Client

from flameox.catalog import Catalog
from flameox.mcp import create_server
from flameox.storage import RunStore, Workspace

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


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
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
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
        plan_token = planned.structured_content["result"]["plan_token"]
        executed = await client.call_tool(
            "execute_capture_plan",
            {"plan_token": plan_token},
            progress_callback=record_progress,
        )
        assert executed.structured_content is not None
        run_id = executed.structured_content["result"]["run_id"]
        fetched = await client.call_tool("get_run", {"run_id": run_id})
        replayed = await client.call_tool(
            "execute_capture_plan",
            {"plan_token": plan_token},
        )

    assert executed.is_error is False
    assert executed.structured_content is not None
    assert executed.structured_content["result"]["resource_uri"] == f"flameox://runs/{run_id}"
    semantics = executed.structured_content["result"]["semantics"]
    assert semantics["adapter"] == "command"
    assert semantics["semantic_id"].startswith("sha256:")
    assert semantics["bounds"] == {}
    assert executed.structured_content["result"]["artifact_count"] == len(
        executed.structured_content["result"]["artifact_ids"]
    )
    assert executed.structured_content["result"]["artifacts_truncated"] is False
    assert executed.structured_content["result"]["limitations_truncated"] is False
    assert "run" not in executed.structured_content["result"]
    assert any(
        item.type == "resource_link" and item.uri == f"flameox://runs/{run_id}"
        for item in executed.content
    )
    assert replayed.is_error is True
    assert replayed.structured_content is not None
    assert replayed.structured_content["error"]["code"] == "INVALID_CAPTURE_PLAN"
    assert replayed.structured_content["error"]["recovery"] == {
        "kind": "manual",
        "safe_to_repeat_same_call": False,
        "retry_after_ms": None,
        "next_tool": None,
        "action": {
            "kind": "manual",
            "instruction": (
                "Inspect the declared workload and supply a complete capture plan request."
            ),
            "suggested_action": "capture.plan",
            "missing_arguments": ["workload_name", "adapter", "parameters"],
        },
    }
    assert fetched.structured_content is not None
    assert fetched.structured_content["result"]["run_id"] == run_id
    assert [item[0] for item in recorded_progress] == list(range(9))
    assert {item[1] for item in recorded_progress} == {8}
    assert all(item[2] for item in recorded_progress)


@pytest.mark.anyio
async def test_mcp_plan_capture_ignores_adhoc_arguments_without_overriding_workload(
    tmp_path: Path,
) -> None:
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
    for result in (argv, cwd):
        assert result.is_error is False
        assert result.structured_content is not None
        command = result.structured_content["result"]["workload_instance"]["command"]
        assert command["argv"][-2:] == ["-c", "print('ok')"]
        assert "unsafe" not in command["argv"]
        assert command["cwd"] == str(tmp_path.resolve())


@pytest.mark.anyio
async def test_mcp_plan_capture_binds_compute_sanitizer_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "compute-sanitizer"
    executable.write_text(
        '#!/bin/sh\nif [ "$1" = "--version" ]; then echo \'Version 2026.2.1\'; exit 0; fi\nexit 1\n'
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    (tmp_path / "sanitizer.supp").write_text("# fixture\n")
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.probe]
argv = ["/bin/true"]
cwd = "."
"""
    )
    Workspace.initialize(tmp_path)

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool(
            "plan_capture",
            {
                "workload_name": "probe",
                "adapter": "compute-sanitizer",
                "parameters": {},
                "capture_mode": "trusted_local",
                "preflight_mode": "passive",
                "compute_sanitizer_options": {
                    "tool": "racecheck",
                    "launch_skip": 2,
                    "launch_count": 3,
                    "suppression_file": "sanitizer.supp",
                },
            },
        )

    assert result.is_error is False, result.structured_content
    assert result.structured_content is not None
    options = result.structured_content["result"]["adapter_options"]
    assert options["tool"] == "racecheck"
    assert options["launch_skip"] == 2
    assert options["launch_count"] == 3
    assert options["suppression_file"] == "sanitizer.supp"
    assert options["suppression_digest"].startswith("sha256:")


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
    assert error["details"]["missing_python_distributions"] == ["flameox-agent-fixture>=99"]
    assert error["recovery"] == {
        "kind": "manual",
        "safe_to_repeat_same_call": False,
        "retry_after_ms": None,
        "action": {
            "kind": "manual",
            "instruction": (
                "Install the missing distributions in the workload's declared Python "
                "environment or select another workload, then plan capture again."
            ),
            "suggested_action": "workflow.get",
            "missing_arguments": [],
        },
        "next_tool": None,
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
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
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
                {"plan_token": planned.structured_content["result"]["plan_token"]},
                progress_callback=record_progress,
            )
        )
        await asyncio.wait_for(collector_started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    runs = list(RunStore(workspace).list())
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
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
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
            "plan_token": planned.structured_content["result"]["plan_token"],
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
