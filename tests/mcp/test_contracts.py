from __future__ import annotations

import errno
import sys
from pathlib import Path
from typing import Literal

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from mcp import Client
from mcp_types import TextContent

from flameox.catalog import Catalog
from flameox.domain import ErrorCode
from flameox.mcp import create_server
from flameox.storage import Workspace

pytestmark = [pytest.mark.integration, pytest.mark.serial]


def _open_object_schema_paths(schema: object, path: str = "$") -> list[str]:
    if isinstance(schema, dict):
        paths = (
            [path]
            if path != "$"
            and schema.get("type") == "object"
            and schema.get("additionalProperties") is not False
            else []
        )
        for key, value in schema.items():
            if key != "additionalProperties":
                paths.extend(_open_object_schema_paths(value, f"{path}.{key}"))
        return paths
    if isinstance(schema, list):
        return [
            nested_path
            for index, value in enumerate(schema)
            for nested_path in _open_object_schema_paths(value, f"{path}[{index}]")
        ]
    return []


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
    output_schema = by_name["workspace_status"].output_schema
    assert output_schema is not None
    assert output_schema["type"] == "object"
    assert output_schema["discriminator"]["propertyName"] == "ok"
    definitions = output_schema["$defs"]
    success_name = next(name for name in definitions if name.startswith("SuccessPayload_"))
    assert definitions[success_name]["required"] == [
        "ok",
        "result",
        "error",
    ]
    assert definitions[success_name]["properties"]["ok"]["const"] is True
    assert definitions["FailurePayload"]["required"] == [
        "ok",
        "result",
        "error",
    ]
    assert definitions["FailurePayload"]["properties"]["ok"]["const"] is False
    assert set(definitions["ErrorCode"]["enum"]) == {code.value for code in ErrorCode}
    assert definitions["RecoveryAction"]["discriminator"]["propertyName"] == "kind"
    assert "context" not in definitions["ToolActionRecovery"]["properties"]
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
    assert (
        "torch.profiler"
        in by_name["import_artifact"].input_schema["properties"]["producer"]["enum"]
    )
    assert "temp" in by_name["import_artifact"].input_schema["properties"]["source_root"]["enum"]
    assert "prepare_adapter" in by_name
    assert "prepare_workload_dependencies" in by_name
    assert by_name["prepare_workload_dependencies"].annotations is not None
    assert by_name["prepare_workload_dependencies"].annotations.read_only_hint is True
    assert by_name["prepare_workload_dependencies"].annotations.destructive_hint is False
    assert "exact installed package identity" in (by_name["prepare_adapter"].description or "")
    assert "never executes" in (by_name["prepare_workload_dependencies"].description or "")


@pytest.mark.anyio
async def test_every_mcp_tool_uses_sdk_generated_schemas_and_annotations(
    tmp_path: Path,
) -> None:
    Workspace.initialize(tmp_path)
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        tools = (await client.list_tools()).tools

    by_name = {tool.name: tool for tool in tools}
    assert len(tools) >= 40
    for tool in tools:
        assert tool.input_schema["type"] == "object"
        assert tool.output_schema is not None
        assert tool.output_schema["type"] == "object"
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
    execution_output = by_name["analyze_execution"].output_schema
    assert execution_output is not None
    definitions = execution_output["$defs"]
    success = definitions["SuccessPayload_ExecutionAnalysisResult_"]
    execution_result = success["properties"]["result"]
    assert execution_result["properties"]["returned"]["readOnly"] is True
    assert execution_result["properties"]["truncated"]["readOnly"] is True
    assert {"returned", "truncated"} <= set(execution_result["required"])


@pytest.mark.anyio
async def test_mcp_nested_models_only_advertise_intentional_flexible_object_maps(
    tmp_path: Path,
) -> None:
    Workspace.initialize(tmp_path)
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        tools = (await client.list_tools()).tools

    open_object_tools = {
        tool.name for tool in tools if _open_object_schema_paths(tool.input_schema)
    }
    assert open_object_tools == {
        "compare_kernel_validation",  # bounded named case dimensions and inputs
        "compare_run_sets",  # bounded measurement-series dimensions
        "configure_workload",  # environment and declared parameter names
        "freeze_run_set",  # persisted comparison-selection projection
        "plan_capture",  # names declared by the selected workload
        "plan_experiment",  # names declared by the selected experiment
        "plan_fault_experiment",  # names declared by the selected fault experiment
        "plan_reduction",  # names accepted by the declared predicate workload
        "record_finding",  # bounded JSON handoff to a human investigation
        "record_comparison",  # bounded measurement-series dimensions
        "record_kernel_validation_comparison",  # bounded named case dimensions and inputs
    }


@pytest.mark.anyio
async def test_accelerator_tools_advertise_bounded_v2_schemas(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    launches = tools["analyze_accelerator_launches"].input_schema["properties"]
    assert launches["run_or_artifact"]["minLength"] == 1
    assert launches["run_or_artifact"]["maxLength"] == 200
    assert launches["limit"] == {
        "description": "Maximum regions and kernel names to return.",
        "maximum": 1000,
        "minimum": 1,
        "title": "Limit",
        "type": "integer",
    }
    for name in ("extract_benchmark_samples", "extract_nsight_systems"):
        run_id = tools[name].input_schema["properties"]["run_id"]
        assert run_id["minLength"] == 1
        assert run_id["maxLength"] == 200

    perfetto = tools["extract_perfetto"].input_schema["properties"]
    assert perfetto["run_id"]["minLength"] == 1
    assert perfetto["artifact_id"]["anyOf"][0]["minLength"] == 1
    profiler = tools["plan_capture"].input_schema["$defs"]["TorchProfilerCaptureOptions"]
    assert profiler["oneOf"] == [
        {"$ref": "#/$defs/WholeEntrypointTorchProfilerOptions"},
        {"$ref": "#/$defs/SdkTorchProfilerOptions"},
    ]
    for variant in ("WholeEntrypointTorchProfilerOptions", "SdkTorchProfilerOptions"):
        assert tools["plan_capture"].input_schema["$defs"][variant]["additionalProperties"] is False
    schedule = tools["plan_capture"].input_schema["$defs"]["TorchProfilerSchedule"]
    assert schedule["additionalProperties"] is False
    assert schedule["properties"]["repeat"]["maximum"] == 100
    sanitizer = tools["plan_capture"].input_schema["$defs"]["ComputeSanitizerCaptureOptions"]
    assert sanitizer["additionalProperties"] is False
    assert set(sanitizer["properties"]) == {
        "tool",
        "launch_skip",
        "launch_count",
        "target_processes",
        "target_processes_filter",
        "kernel_name",
        "demangle",
        "suppression_file",
        "finding_exit_code",
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_fields"),
    [
        (
            "analyze_accelerator_launches",
            {"run_or_artifact": "run", "limit": True},
            ("limit",),
        ),
        ("extract_benchmark_samples", {"run_id": ""}, ("run_id",)),
        ("extract_nsight_systems", {"run_id": ""}, ("run_id",)),
        ("extract_perfetto", {"run_id": "run", "artifact_id": ""}, ("artifact_id",)),
        (
            "plan_capture",
            {
                "workload_name": "workload",
                "adapter": "torch.profiler",
                "parameters": {},
                "torch_profiler_options": {
                    "mode": "sdk",
                    "record_shapes": 1,
                    "schedule": {"active": True},
                },
            },
            (
                "torch_profiler_options.sdk.record_shapes",
                "torch_profiler_options.sdk.schedule.active",
            ),
        ),
        (
            "plan_capture",
            {
                "workload_name": "workload",
                "adapter": "compute-sanitizer",
                "parameters": {},
                "compute_sanitizer_options": {"launch_skip": True},
            },
            ("compute_sanitizer_options.launch_skip",),
        ),
    ],
)
async def test_sdk_rejects_invalid_accelerator_tool_values(
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, object],
    expected_fields: tuple[str, ...],
) -> None:
    Workspace.initialize(tmp_path)
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool(tool_name, arguments)

    assert result.is_error is True
    assert result.structured_content is None
    assert isinstance(result.content[0], TextContent)
    for field in expected_fields:
        assert field in result.content[0].text


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_field"),
    [
        ("list_runs", {"limit": True}, "limit"),
        ("list_runs", {"limit": "1"}, "limit"),
        (
            "configure_workload",
            {
                "name": "probe",
                "operation": "create",
                "argv": ["python", "-c", "pass"],
                "timeout_seconds": True,
            },
            "timeout_seconds",
        ),
        (
            "configure_inference_scenario",
            {
                "name": "probe",
                "operation": "create",
                "server_name": "server",
                "provider": "aiperf",
                "streaming": 1,
            },
            "streaming",
        ),
    ],
)
async def test_sdk_strict_scalars_reject_json_coercions(
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, object],
    expected_field: str,
) -> None:
    Workspace.initialize(tmp_path)
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool(tool_name, arguments)

    assert result.is_error is True
    assert result.structured_content is None
    assert isinstance(result.content[0], TextContent)
    assert expected_field in result.content[0].text


@pytest.mark.anyio
async def test_sdk_strict_float_accepts_integer_json_numbers(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool(
            "configure_workload",
            {
                "name": "probe",
                "operation": "create",
                "argv": [sys.executable, "-c", "pass"],
                "timeout_seconds": 5,
            },
        )

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["ok"] is True


@pytest.mark.anyio
async def test_native_sdk_ignores_unknown_top_level_tool_arguments(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        tool = {item.name: item for item in (await client.list_tools()).tools}["workspace_status"]
        result = await client.call_tool("workspace_status", {"unknown": True})

    assert "additionalProperties" not in tool.input_schema
    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["ok"] is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("mode", "expected_protocol"),
    [
        ("legacy", "2025-11-25"),
        ("2026-07-28", "2026-07-28"),
    ],
)
async def test_mcp_contract_is_native_in_both_supported_protocol_eras(
    tmp_path: Path,
    mode: Literal["legacy", "2026-07-28"],
    expected_protocol: str,
) -> None:
    Workspace.initialize(tmp_path)
    async with Client(
        create_server(tmp_path),
        mode=mode,
        raise_exceptions=True,
    ) as client:
        assert client.protocol_version == expected_protocol
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        success = await client.call_tool("workspace_status", {})
        domain_failure = await client.call_tool("get_run", {"run_id": "missing"})
        sdk_failure = await client.call_tool("list_runs", {"limit": True})

    assert tools["workspace_status"].output_schema is not None
    assert tools["workspace_status"].output_schema["type"] == "object"
    assert success.is_error is False
    assert success.structured_content is not None
    assert success.structured_content["ok"] is True
    assert domain_failure.is_error is True
    assert domain_failure.structured_content is not None
    assert domain_failure.structured_content["error"]["code"] == "RUN_NOT_FOUND"
    assert sdk_failure.is_error is True
    assert sdk_failure.structured_content is None


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
        "kind": "tool_action",
        "safe_to_repeat_same_call": False,
        "retry_after_ms": None,
        "action": {
            "kind": "tool",
            "action": "workspace.initialize",
            "arguments": {},
        },
        "next_tool": "initialize_workspace",
        "next_arguments": {},
    }
    assert configuration.is_error is True
    assert configuration.structured_content is not None
    assert configuration.structured_content["error"]["code"] == "WORKSPACE_NOT_FOUND"
    assert configuration.structured_content["error"]["recovery"]["next_tool"] == (
        "initialize_workspace"
    )


@pytest.mark.anyio
async def test_extractors_name_missing_artifact_and_require_state_change(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    (tmp_path / "unrelated.log").write_text("not extractor input\n")

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        imported = await client.call_tool(
            "import_artifact",
            {
                "path": "unrelated.log",
                "kind": "process_output",
                "sensitivity": "normal",
            },
        )
        assert imported.structured_content is not None
        run_id = imported.structured_content["result"]["run_id"]
        calls: tuple[tuple[str, dict[str, str]], ...] = (
            ("extract_perfetto", {}),
            ("extract_pyperf", {}),
            ("extract_python_startup", {}),
            ("extract_pytest", {}),
            ("extract_observations", {}),
            ("extract_nsight_systems", {}),
            ("extract_benchmark_samples", {}),
            ("extract_inference_result", {"provider": "aiperf"}),
            ("extract_inference_trace", {}),
            ("extract_otlp_trace", {}),
        )
        results = [
            (name, await client.call_tool(name, {"run_id": run_id, **arguments}))
            for name, arguments in calls
        ]
        unknown = await client.call_tool("extract_pyperf", {"run_id": "unknown-run"})
        malformed_import = await client.call_tool(
            "import_artifact",
            {
                "path": "unrelated.log",
                "kind": "benchmark_samples",
                "sensitivity": "normal",
                "producer": "pyperf",
            },
        )
        assert malformed_import.structured_content is not None
        malformed = await client.call_tool(
            "extract_pyperf",
            {"run_id": malformed_import.structured_content["result"]["run_id"]},
        )
        otlp_import = await client.call_tool(
            "import_artifact",
            {
                "path": "unrelated.log",
                "kind": "otlp_trace",
                "media_type": "application/json",
                "sensitivity": "normal",
            },
        )
        assert otlp_import.structured_content is not None
        otlp_artifact_id = otlp_import.structured_content["result"]["artifact_id"]
        invalid_otlp_selection = await client.call_tool(
            "extract_otlp_trace",
            {
                "run_id": otlp_import.structured_content["result"]["run_id"],
                "artifact_id": "wrong-artifact",
            },
        )

    for name, result in results:
        assert result.is_error is True, name
        assert result.structured_content is not None
        error = result.structured_content["error"]
        assert error["code"] == "ARTIFACT_NOT_FOUND", name
        assert error["run_id"] == run_id
        assert error["details"]["required_artifact_kinds"]
        assert (
            error["details"]["compatible_capture_adapters"]
            or error["details"]["compatible_import_producers"]
        )
        assert error["recovery"]["safe_to_repeat_same_call"] is False
        assert error["recovery"]["action"]["suggested_action"] in {
            "capture.detached.start",
            "artifact.import",
        }

    assert unknown.structured_content is not None
    assert unknown.structured_content["error"]["code"] == "RUN_NOT_FOUND"
    assert malformed.structured_content is not None
    assert malformed.structured_content["error"]["code"] == "ARTIFACT_PARSE_FAILED"
    assert invalid_otlp_selection.structured_content is not None
    selection_error = invalid_otlp_selection.structured_content["error"]
    assert selection_error["code"] == "INVALID_ARGUMENTS"
    assert selection_error["details"]["available_artifact_ids"] == [otlp_artifact_id]
    assert selection_error["recovery"]["safe_to_repeat_same_call"] is False
    assert selection_error["recovery"]["action"]["missing_arguments"] == ["artifact_id"]


@pytest.mark.anyio
async def test_import_profile_returns_validated_run_semantics_inline(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    (tmp_path / "pyspy.json").write_text(
        '[{"args":{"filename":"scan.py","line":12},"cat":"py-spy",'
        '"name":"scan","ph":"B","pid":1,"tid":2,"ts":3},'
        '{"args":{"filename":"scan.py","line":12},"cat":"py-spy",'
        '"name":"scan","ph":"E","pid":1,"tid":2,"ts":4}]'
    )

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool(
            "import_artifact",
            {
                "path": "pyspy.json",
                "kind": "sample_profile",
                "sensitivity": "normal",
                "producer": "py-spy",
                "producer_version": "0.4.2",
                "profile": "py-spy-chrometrace",
            },
        )

    assert result.is_error is False
    assert result.structured_content is not None
    semantics = result.structured_content["result"]["semantics"]
    assert semantics["origin"] == "import"
    assert semantics["adapter"] == "py-spy"
    assert semantics["adapter_version"] is None
    assert semantics["unavailable_fields"] == ["adapter_version", "scope"]

    imported = result.structured_content["result"]
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        qualified = await client.call_tool(
            "qualify_artifact_import",
            {
                "source_run_id": imported["run_id"],
                "artifact_id": imported["artifact_id"],
                "profile": "py-spy-chrometrace",
            },
        )
    assert qualified.is_error is False
    assert qualified.structured_content is not None
    assert qualified.structured_content["result"]["artifact_id"] == imported["artifact_id"]


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
async def test_mcp_invalid_configuration_exposes_missing_action_inputs(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        "schema_version = 1\n[experiments.broken]\nworkload = 'missing'\n"
        "treatment_factor = 'mode'\n[experiments.broken.factors]\nmode = ['a', 'b']\n"
    )

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        status = await client.call_tool("workload_configuration_status", {})
        failed = await client.call_tool("list_declared_workflows", {"kind": "workload"})

    assert status.structured_content is not None
    assert status.structured_content["result"]["next_action"] == {
        "kind": "manual",
        "instruction": "Supply a complete named workload definition before continuing.",
        "suggested_action": "workload.configure",
        "missing_arguments": ["name", "operation", "argv"],
    }
    assert failed.is_error is True
    assert failed.structured_content is not None
    assert failed.structured_content["error"]["recovery"] == {
        "kind": "manual",
        "safe_to_repeat_same_call": False,
        "retry_after_ms": None,
        "next_tool": None,
        "action": {
            "kind": "manual",
            "instruction": "Supply a complete named workload definition before continuing.",
            "suggested_action": "workload.configure",
            "missing_arguments": ["name", "operation", "argv"],
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
        "action": {
            "kind": "manual",
            "instruction": "Repair flameox.toml manually, then verify its status.",
            "suggested_action": "workload.configuration.inspect",
            "missing_arguments": [],
        },
    }
    diagnostic = failed.structured_content["error"]["details"]["diagnostic"]
    assert diagnostic
    assert len(diagnostic) <= 500


@pytest.mark.anyio
async def test_mcp_capability_reports_include_provisioning_and_setup_verification(
    tmp_path: Path,
) -> None:
    Workspace.initialize(tmp_path)

    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        result = await client.call_tool("list_capabilities", {"mode": "passive"})

    assert result.structured_content is not None
    capabilities = result.structured_content["result"]["capabilities"]
    assert {"command", "pyperf", "torch.profiler"} <= {item["adapter"] for item in capabilities}
    for capability in capabilities:
        assert "provisioning" in capability, capability["adapter"]
        assert "setup_verification" in capability, capability["adapter"]


@pytest.mark.anyio
@pytest.mark.process
@pytest.mark.serial
async def test_mcp_initialize_is_idempotent_without_discarding_capture_plans(
    tmp_path: Path,
) -> None:
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        first = await client.call_tool("initialize_workspace", {})
        configured = await client.call_tool(
            "configure_workload",
            {
                "name": "probe",
                "operation": "create",
                "argv": [sys.executable, "-c", "print('capture-plan-survived')"],
            },
        )
        planned = await client.call_tool(
            "plan_capture",
            {"workload_name": "probe", "adapter": "command", "parameters": {}},
        )
        assert planned.structured_content is not None
        second = await client.call_tool("initialize_workspace", {})
        executed = await client.call_tool(
            "execute_capture_plan",
            {"plan_token": planned.structured_content["result"]["plan_token"]},
        )
        status = await client.call_tool("workspace_status", {})

    assert first.is_error is False
    assert configured.is_error is False
    assert planned.is_error is False
    assert second.is_error is False
    assert executed.is_error is False
    assert status.is_error is False
    assert first.structured_content is not None
    assert second.structured_content is not None
    assert executed.structured_content is not None
    assert status.structured_content is not None
    assert executed.structured_content["result"]["execution_status"] == "succeeded"
    assert (
        first.structured_content["result"]["workspace_id"]
        == second.structured_content["result"]["workspace_id"]
        == status.structured_content["result"]["workspace_id"]
        == Workspace.discover(tmp_path).identity.workspace_id
    )


@pytest.mark.anyio
async def test_mcp_modes_make_capability_and_integrity_choices_explicit(tmp_path: Path) -> None:
    Workspace.initialize(tmp_path)
    async with Client(create_server(tmp_path), raise_exceptions=True) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        passive = await client.call_tool("list_capabilities", {"mode": "passive"})
        scoped = await client.call_tool(
            "list_capabilities",
            {"mode": "passive", "adapter": "torch.profiler"},
        )
        standard = await client.call_tool("validate_workspace", {"mode": "standard"})

    capability_mode = tools["list_capabilities"].input_schema["properties"]["mode"]
    assert "adapter" in tools["list_capabilities"].input_schema["properties"]
    integrity_mode = tools["validate_workspace"].input_schema["properties"]["mode"]
    assert capability_mode["enum"] == ["passive", "active_cached", "active_refresh"]
    assert integrity_mode["enum"] == ["standard", "full"]
    assert passive.is_error is False
    assert passive.structured_content is not None
    assert passive.structured_content["result"]["setup_adapters"] == []
    assert passive.structured_content["result"]["next_action"] is None
    assert scoped.is_error is False
    assert scoped.structured_content is not None
    assert scoped.structured_content["result"]["recommendation_scope"] == "torch.profiler"
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
