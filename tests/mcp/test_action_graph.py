from __future__ import annotations

from pathlib import Path

import pytest
from jsonschema.validators import validator_for  # type: ignore[import-untyped]

from flameox.action_graph import (
    ACTION_REGISTRY,
    ActionId,
    ManualAction,
    ToolAction,
    next_action_for_action,
    tool_action,
)
from flameox.domain.errors import DomainError, ErrorCode
from flameox.mcp import create_server

pytestmark = pytest.mark.integration


def _representative_actions() -> dict[ActionId, ToolAction]:
    """Return one registry-validated, executable payload for every workflow edge."""

    return {
        ActionId.INITIALIZE_WORKSPACE: tool_action(ActionId.INITIALIZE_WORKSPACE),
        ActionId.INSPECT_WORKLOAD_CONFIGURATION: tool_action(
            ActionId.INSPECT_WORKLOAD_CONFIGURATION
        ),
        ActionId.CONFIGURE_WORKLOAD: tool_action(
            ActionId.CONFIGURE_WORKLOAD,
            name="demo",
            operation="create",
            argv=["python", "-m", "demo"],
        ),
        ActionId.INSPECT_CAPABILITIES: tool_action(ActionId.INSPECT_CAPABILITIES),
        ActionId.START_CAPABILITY_SETUP: tool_action(
            ActionId.START_CAPABILITY_SETUP,
            adapters=["memray"],
            idempotency_key="setup-1",
        ),
        ActionId.GET_CAPABILITY_SETUP: tool_action(
            ActionId.GET_CAPABILITY_SETUP,
            operation_id="op-1",
        ),
        ActionId.PREPARE_ADAPTER: tool_action(
            ActionId.PREPARE_ADAPTER,
            adapter="third.party",
            distribution="third-party",
        ),
        ActionId.PREPARE_WORKLOAD_DEPENDENCIES: tool_action(
            ActionId.PREPARE_WORKLOAD_DEPENDENCIES,
            workload_name="demo",
        ),
        ActionId.LIST_DECLARED_WORKFLOWS: tool_action(ActionId.LIST_DECLARED_WORKFLOWS),
        ActionId.GET_DECLARED_WORKFLOW: tool_action(
            ActionId.GET_DECLARED_WORKFLOW,
            kind="workload",
            name="demo",
        ),
        ActionId.PLAN_CAPTURE: tool_action(
            ActionId.PLAN_CAPTURE,
            workload_name="demo",
            adapter="py-spy",
            parameters={},
        ),
        ActionId.START_DETACHED_CAPTURE: tool_action(
            ActionId.START_DETACHED_CAPTURE,
            plan_token="plan-token",
            idempotency_key="capture-1",
        ),
        ActionId.GET_DETACHED_CAPTURE: tool_action(
            ActionId.GET_DETACHED_CAPTURE,
            run_id="run-1",
        ),
        ActionId.LIST_RUNS: tool_action(ActionId.LIST_RUNS),
        ActionId.LIST_ARTIFACTS: tool_action(ActionId.LIST_ARTIFACTS),
        ActionId.PREVIEW_ARTIFACT: tool_action(
            ActionId.PREVIEW_ARTIFACT,
            artifact_id="sha256:" + "0" * 64,
            offset=0,
            max_bytes=4_096,
            max_lines=80,
        ),
        ActionId.IMPORT_ARTIFACT: tool_action(
            ActionId.IMPORT_ARTIFACT,
            path="trace.json",
            kind="execution_trace",
            sensitivity="normal",
        ),
        ActionId.IMPORT_XCTRACE: tool_action(
            ActionId.IMPORT_XCTRACE,
            path="metal.trace",
        ),
        ActionId.EXTRACT_PERFETTO: tool_action(
            ActionId.EXTRACT_PERFETTO,
            run_id="run-1",
        ),
        ActionId.EXTRACT_MEMRAY: tool_action(
            ActionId.EXTRACT_MEMRAY,
            run_id="run-1",
        ),
        ActionId.EXTRACT_NSIGHT_SYSTEMS: tool_action(
            ActionId.EXTRACT_NSIGHT_SYSTEMS,
            run_id="run-1",
        ),
        ActionId.CONFIGURE_INFERENCE_SERVER: tool_action(
            ActionId.CONFIGURE_INFERENCE_SERVER,
            name="local",
            operation="create",
            mode="existing_local",
            model="demo/model",
        ),
        ActionId.LIST_INFERENCE_CONFIGURATIONS: tool_action(ActionId.LIST_INFERENCE_CONFIGURATIONS),
        ActionId.PLAN_INFERENCE_SCENARIO: tool_action(
            ActionId.PLAN_INFERENCE_SCENARIO,
            scenario_name="smoke",
        ),
        ActionId.EXECUTE_REDUCTION: tool_action(
            ActionId.EXECUTE_REDUCTION,
            plan_id="plan-1",
        ),
        ActionId.GET_REDUCTION: tool_action(
            ActionId.GET_REDUCTION,
            reduction_id="reduction-1",
        ),
    }


@pytest.mark.anyio
async def test_registered_actions_match_live_mcp_tools_and_accept_executable_payloads(
    tmp_path: Path,
) -> None:
    tools = await create_server(tmp_path).list_tools()
    by_name = {tool.name: tool for tool in tools}
    examples = _representative_actions()

    assert set(ACTION_REGISTRY) == set(ActionId) == set(examples)
    assert len(tools) == len(by_name)

    for action_id, descriptor in ACTION_REGISTRY.items():
        tool = by_name[descriptor.tool_name.value]
        arguments = examples[action_id].arguments
        validator_type = validator_for(tool.input_schema)
        validator_type.check_schema(tool.input_schema)
        validator_type(tool.input_schema).validate(arguments)

        annotations = tool.annotations
        assert annotations is not None
        assert annotations.read_only_hint is descriptor.annotations.read_only
        assert annotations.destructive_hint is descriptor.annotations.destructive
        assert annotations.idempotent_hint is descriptor.annotations.idempotent
        assert annotations.open_world_hint is descriptor.annotations.open_world


def test_incomplete_recovery_is_explicitly_manual() -> None:
    action = next_action_for_action(
        ActionId.CONFIGURE_WORKLOAD,
        context={"operation": "create"},
        instruction="Declare the missing workload inputs before continuing.",
    )

    assert isinstance(action, ManualAction)
    assert action.suggested_action is ActionId.CONFIGURE_WORKLOAD
    assert action.missing_arguments == ("name", "argv")


def test_domain_error_preserves_validated_recovery_action() -> None:
    error = DomainError(
        ErrorCode.ARTIFACT_PARSE_FAILED,
        "Perfetto evidence has not been extracted.",
        details={"run_id": "run-1"},
        remediation=("Extract the trace before querying it.",),
        next_action=tool_action(ActionId.EXTRACT_PERFETTO, run_id="run-1"),
    )

    assert isinstance(error.next_action, ToolAction)
    assert error.next_action.action is ActionId.EXTRACT_PERFETTO
    assert error.next_action.arguments == {"run_id": "run-1"}
    assert "next_tool" not in error.details


def test_domain_error_rejects_legacy_recovery_fields() -> None:
    with pytest.raises(ValueError, match="Legacy recovery fields are not accepted"):
        DomainError(
            ErrorCode.INTERNAL_ERROR,
            "Recovery metadata is invalid.",
            details={"next_tool": "extract_perfetto"},
        )
