from __future__ import annotations

import errno
from pathlib import Path
from unittest.mock import patch

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from mcp import Client
from mcp_types import TextContent

from flameox.application import DetachedCaptureManager
from flameox.catalog import Catalog
from flameox.mcp import create_server
from flameox.storage import Workspace


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
        instructions = client.instructions

    assert by_name["workspace_status"].annotations is not None
    assert by_name["workspace_status"].annotations.read_only_hint is True
    assert by_name["import_artifact"].annotations is not None
    assert by_name["import_artifact"].annotations.read_only_hint is False
    assert by_name["workload_configuration_status"].annotations is not None
    assert by_name["workload_configuration_status"].annotations.read_only_hint is True
    assert by_name["configure_workload"].annotations is not None
    assert by_name["configure_workload"].annotations.read_only_hint is False
    assert by_name["configure_workload"].annotations.destructive_hint is False
    assert by_name["configure_workload"].annotations.idempotent_hint is True
    assert by_name["configure_workload"].annotations.open_world_hint is False
    assert by_name["configure_workload"].input_schema["required"] == [
        "name",
        "operation",
        "argv",
    ]
    assert "never executes" in (by_name["configure_workload"].description or "")
    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["ok"] is True
    assert result.structured_content["result"]["workspace_id"] == workspace.identity.workspace_id
    assert len(result.content) == 1
    assert instructions is not None
    assert "list_declared_workflows" in instructions
    assert "consumed capture plan" in instructions
    assert "WORKSPACE_NOT_FOUND" in instructions
    assert "workload_configuration_status" in instructions
    assert "configure_workload" in instructions
    assert "never executes it" in instructions
    assert "capture_mode='auto'" in instructions
    assert "execute_capture_plan" in instructions
    assert "producer" in by_name["import_artifact"].input_schema["properties"]
    assert "torch.profiler" in by_name["import_artifact"].input_schema["properties"]["producer"][
        "enum"
    ]
    assert "temp" in by_name["import_artifact"].input_schema["properties"]["source_root"][
        "enum"
    ]
    assert "prepare_adapter" in by_name
    assert "prepare_workload_dependencies" in by_name
    assert "exact installed package identity" in (by_name["prepare_adapter"].description or "")
    assert "never executes" in (
        by_name["prepare_workload_dependencies"].description or ""
    )


@pytest.mark.anyio
async def test_every_mcp_tool_has_bounded_object_schemas_and_annotations(
    tmp_path: Path,
) -> None:
    Workspace.initialize(tmp_path)
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        tools = (await client.list_tools()).tools

    by_name = {tool.name: tool for tool in tools}
    assert len(tools) >= 40
    for tool in tools:
        assert tool.input_schema["type"] == "object"
        assert tool.input_schema["additionalProperties"] is False
        assert tool.output_schema is not None
        validator = Draft202012Validator(tool.output_schema)
        assert list(validator.iter_errors({"ok": True}))
        assert list(validator.iter_errors({"ok": False}))
        assert tool.annotations is not None
        if tool.annotations.read_only_hint:
            assert tool.annotations.destructive_hint is False

    assert "run_or_artifact" in by_name["analyze_hotspots"].input_schema["properties"]
    assert "input_id" not in by_name["analyze_hotspots"].input_schema["properties"]
    assert (
        by_name["analyze_execution"].input_schema["properties"]["comparison_run_or_artifact"][
            "description"
        ]
        == "Optional second run or artifact ID for a compatible comparison."
    )
    assert by_name["get_evidence"].description is not None
    assert "ref_type and its ID separately" in by_name["get_evidence"].description


@pytest.mark.anyio
async def test_mcp_domain_errors_remain_structured(tmp_path: Path) -> None:
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool("workspace_status", {})
        configuration = await client.call_tool("workload_configuration_status", {})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["ok"] is False
    assert result.structured_content["error"]["code"] == "WORKSPACE_NOT_FOUND"
    assert result.structured_content["error"]["remediation"] == [
        "Verify the server's fixed project root is the intended checkout, then call "
        "initialize_workspace."
    ]
    assert result.structured_content["error"]["recovery"] == {
        "kind": "initialize_workspace",
        "safe_to_repeat_same_call": False,
        "retry_after_ms": None,
        "next_tool": "initialize_workspace",
    }
    assert configuration.is_error is True
    assert configuration.structured_content is not None
    assert configuration.structured_content["error"]["code"] == "WORKSPACE_NOT_FOUND"
    assert configuration.structured_content["error"]["recovery"]["next_tool"] == (
        "initialize_workspace"
    )


@pytest.mark.anyio
async def test_mcp_workspace_initialization_failures_remain_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "simulated quota exhaustion")

    monkeypatch.setattr("flameox.storage.workspace.atomic_write_json", fail_write)

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool("initialize_workspace", {})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["error"]["code"] == "STORAGE_QUOTA_EXCEEDED"
    assert result.structured_content["error"]["remediation"] == [
        "Free local storage or increase the filesystem quota, then retry initialization."
    ]


@pytest.mark.anyio
async def test_mcp_invalid_configuration_exposes_typed_recovery_context(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n[experiments.broken]\nworkload = 'missing'\nvariants = ['a', 'b']\n"
    )

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        status = await client.call_tool("workload_configuration_status", {})
        failed = await client.call_tool("list_declared_workflows", {"kind": "workload"})

    assert status.structured_content is not None
    assert status.structured_content["result"]["next_tool"] == "configure_workload"
    assert failed.is_error is True
    assert failed.structured_content is not None
    assert failed.structured_content["error"]["recovery"] == {
        "kind": "configure_workload",
        "safe_to_repeat_same_call": False,
        "retry_after_ms": None,
        "next_tool": "configure_workload",
        "context": {
            "kind": "configure_workload",
            "operation": "create",
            "config_path": "flameox.toml",
        },
    }


@pytest.mark.anyio
async def test_mcp_invalid_configuration_falls_back_to_typed_manual_recovery(
    tmp_path: Path,
) -> None:
    Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text("schema_version =\n")

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        failed = await client.call_tool(
            "configure_workload",
            {
                "name": "probe",
                "operation": "create",
                "argv": ["python", "-c", "print('ok')"],
            },
        )

    assert failed.is_error is True
    assert failed.structured_content is not None
    recovery = failed.structured_content["error"]["recovery"]
    assert recovery == {
        "kind": "manual",
        "safe_to_repeat_same_call": False,
        "retry_after_ms": None,
        "next_tool": None,
        "context": {
            "kind": "manual_configuration",
            "config_path": "flameox.toml",
            "diagnostic": recovery["context"]["diagnostic"],
            "verification_tool": "workload_configuration_status",
        },
    }
    assert recovery["context"]["diagnostic"]
    assert len(recovery["context"]["diagnostic"]) <= 500


@pytest.mark.anyio
async def test_mcp_capability_reports_include_provisioning_and_setup_verification(
    tmp_path: Path,
) -> None:
    Workspace.initialize(tmp_path)

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool("list_capabilities", {"mode": "passive"})

    assert result.structured_content is not None
    capabilities = result.structured_content["result"]["capabilities"]
    assert all("provisioning" in item for item in capabilities)
    assert all("setup_verification" in item for item in capabilities)


@pytest.mark.anyio
async def test_mcp_initialize_is_idempotent_without_replacing_capture_manager(
    tmp_path: Path,
) -> None:
    with patch(
        "flameox.mcp.server.DetachedCaptureManager",
        wraps=DetachedCaptureManager,
    ) as constructor:
        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            first = await client.call_tool("initialize_workspace", {})
            second = await client.call_tool("initialize_workspace", {})

    assert first.is_error is False
    assert second.is_error is False
    assert constructor.call_count == 1
    assert first.structured_content is not None
    assert second.structured_content is not None
    assert (
        first.structured_content["result"]["workspace_id"]
        == second.structured_content["result"]["workspace_id"]
    )


@pytest.mark.anyio
async def test_mcp_modes_make_capability_and_integrity_choices_explicit(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        passive = await client.call_tool("list_capabilities", {"mode": "passive"})
        invalid = await client.call_tool("list_capabilities", {"refresh": True})
        standard = await client.call_tool("validate_workspace", {"mode": "standard"})

    capability_mode = tools["list_capabilities"].input_schema["properties"]["mode"]
    integrity_mode = tools["validate_workspace"].input_schema["properties"]["mode"]
    assert capability_mode["enum"] == ["passive", "active_cached", "active_refresh"]
    assert integrity_mode["enum"] == ["standard", "full"]
    assert passive.is_error is False
    assert invalid.is_error is True
    assert isinstance(invalid.content[0], TextContent)
    assert "refresh" in invalid.content[0].text
    assert invalid.structured_content is not None
    assert invalid.structured_content["error"]["code"] == "INVALID_ARGUMENTS"
    assert invalid.structured_content["error"]["details"]["fields"] == [
        {
            "field": "refresh",
            "message": "Unknown argument field.",
            "type": "extra_forbidden",
        }
    ]
    assert standard.is_error is False


@pytest.mark.anyio
async def test_mcp_trace_window_rejects_reversed_time_bounds(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool(
            "get_trace_window",
            {"artifact_id": "trace", "start_ns": 10, "end_ns": 10, "limit": 1},
        )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["error"]["code"] == "INVALID_ARGUMENTS"
    assert result.structured_content["error"]["details"]["fields"] == [
        {
            "field": "end_ns",
            "message": "must be greater than start_ns",
            "type": "greater_than",
        }
    ]
