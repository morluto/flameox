from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import pytest
from mcp import Client

from flameox.mcp import create_server
from flameox.storage import Workspace


class TraceCase(TypedDict):
    name: str
    allowed_paths: list[str]
    expected_invalid_calls: int
    expected_recovery: bool


@dataclass(frozen=True, slots=True)
class TraceMetrics:
    tool_calls: int
    invalid_calls: int
    repeated_calls: int
    response_bytes: int
    writes_outside_permitted: tuple[str, ...]


def _cases() -> tuple[TraceCase, ...]:
    path = Path(__file__).parent / "fixtures" / "agent-workflows.json"
    return tuple(cast(list[TraceCase], json.loads(path.read_text())))


def _snapshot(root: Path) -> frozenset[str]:
    return frozenset(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )


def _outside_permitted(
    before: frozenset[str],
    after: frozenset[str],
    allowed_paths: list[str],
) -> tuple[str, ...]:
    changed = after - before
    return tuple(
        sorted(
            path
            for path in changed
            if not any(path == allowed or path.startswith(allowed) for allowed in allowed_paths)
        )
    )


async def _run_case(case: TraceCase, root: Path) -> tuple[TraceMetrics, bool]:
    workspace = Workspace.initialize(root)
    if case["name"] == "add_workload":
        (root / "flameox.toml").write_text(
            'schema_version = 1\n[workloads.existing]\nargv = ["python", "-c", "pass"]\n'
        )
    elif case["name"] == "replace_stale_workload":
        (root / "flameox.toml").write_text(
            'schema_version = 1\n[workloads.probe]\nargv = ["python", "-c", "pass"]\n'
        )
    elif case["name"] == "invalid_configuration":
        (root / "flameox.toml").write_text(
            'schema_version = 1\n[experiments.broken]\nworkload = "missing"\n'
        )
    elif case["name"] == "attempted_adhoc_capture":
        (root / "flameox.toml").write_text(
            'schema_version = 1\n[workloads.probe]\nargv = ["python", "-c", "pass"]\n'
        )
    elif case["name"] == "normal_configure_discover_plan_execute":
        config = workspace.config.model_copy(
            update={
                "execution": workspace.config.execution.model_copy(
                    update={"containment": "disabled"}
                )
            }
        )
        workspace.paths.config.write_text(config.to_toml())
    before = _snapshot(root)
    calls: list[tuple[str, dict[str, Any]]] = []
    response_bytes = 0
    invalid_calls = 0
    repeated_calls = 0
    previous: tuple[str, dict[str, Any]] | None = None

    async with Client(create_server(root), raise_exceptions=True) as client:

        async def call(name: str, arguments: dict[str, Any]) -> Any:
            nonlocal response_bytes, invalid_calls, repeated_calls, previous
            current = (name, arguments)
            if current == previous:
                repeated_calls += 1
            previous = current
            calls.append(current)
            result = await client.call_tool(name, arguments)
            response_bytes += len(json.dumps(result.structured_content, sort_keys=True))
            if result.is_error and result.structured_content is not None:
                error = result.structured_content.get("error")
                if isinstance(error, dict) and error.get("code") == "INVALID_ARGUMENTS":
                    invalid_calls += 1
            return result

        if case["name"] == "fresh_repository":
            status = await call("workload_configuration_status", {})
            await call("workload_configuration_status", {})
            configured = await call(
                "configure_workload",
                {"name": "probe", "operation": "create", "argv": [sys.executable, "-c", "pass"]},
            )
            recovery_complete = (
                status.structured_content is not None
                and status.structured_content["result"]["next_tool"] == "configure_workload"
                and configured.is_error is False
            )
        elif case["name"] == "add_workload":
            status = await call("workload_configuration_status", {})
            configured = await call(
                "configure_workload",
                {"name": "added", "operation": "create", "argv": ["python", "-c", "pass"]},
            )
            recovery_complete = (
                status.structured_content is not None
                and status.structured_content["result"]["next_tool"] == "list_declared_workflows"
                and configured.is_error is False
            )
        elif case["name"] == "replace_stale_workload":
            status = await call("workload_configuration_status", {})
            stale = await call(
                "configure_workload",
                {
                    "name": "probe",
                    "operation": "replace",
                    "argv": ["python", "-c", "print(1)"],
                    "expected_configuration_id": "sha256:" + "0" * 64,
                },
            )
            current_id = status.structured_content["result"]["configuration_id"]
            updated = await call(
                "configure_workload",
                {
                    "name": "probe",
                    "operation": "replace",
                    "argv": ["python", "-c", "print(1)"],
                    "expected_configuration_id": current_id,
                },
            )
            recovery_complete = stale.is_error and updated.is_error is False
        elif case["name"] == "invalid_configuration":
            status = await call("workload_configuration_status", {})
            recovery_complete = (
                status.is_error is False
                and status.structured_content is not None
                and status.structured_content["result"]["status"] == "invalid"
                and status.structured_content["result"]["next_tool"] == "configure_workload"
            )
        elif case["name"] == "attempted_adhoc_capture":
            result = await call(
                "plan_capture",
                {
                    "workload_name": "probe",
                    "adapter": "command",
                    "parameters": {},
                    "argv": ["echo", "not-declared"],
                },
            )
            recovery_complete = result.is_error
        else:
            status = await call("workload_configuration_status", {})
            configured = await call(
                "configure_workload",
                {
                    "name": "probe",
                    "operation": "create",
                    "argv": [sys.executable, "-c", "print('trace')"],
                },
            )
            configured_without_execution = not any(workspace.paths.runs.iterdir())
            listed = await call("list_declared_workflows", {"kind": "workload"})
            inspected = await call("get_declared_workflow", {"kind": "workload", "name": "probe"})
            capabilities = await call("list_capabilities", {"mode": "passive"})
            planned = await call(
                "plan_capture",
                {"workload_name": "probe", "adapter": "command", "parameters": {}},
            )
            executed = await call(
                "execute_capture_plan",
                {"plan_id": planned.structured_content["result"]["plan_id"]},
            )
            recovery_complete = (
                all(
                    result.is_error is False
                    for result in (
                        status,
                        configured,
                        listed,
                        inspected,
                        capabilities,
                        planned,
                        executed,
                    )
                )
                and configured_without_execution
            )

    after = _snapshot(root)
    return (
        TraceMetrics(
            tool_calls=len(calls),
            invalid_calls=invalid_calls,
            repeated_calls=repeated_calls,
            response_bytes=response_bytes,
            writes_outside_permitted=_outside_permitted(before, after, case["allowed_paths"]),
        ),
        recovery_complete,
    )


@pytest.mark.anyio
@pytest.mark.process
@pytest.mark.serial
@pytest.mark.parametrize("case", _cases(), ids=lambda item: item["name"])
async def test_offline_agent_workflow_traces_measure_contract_behavior(
    case: TraceCase,
    tmp_path: Path,
) -> None:
    metrics, recovery_complete = await _run_case(case, tmp_path)

    assert recovery_complete is case["expected_recovery"]
    assert metrics.invalid_calls == case["expected_invalid_calls"]
    assert metrics.tool_calls > 0
    assert metrics.response_bytes > 0
    assert metrics.writes_outside_permitted == ()
    if case["name"] == "fresh_repository":
        assert metrics.repeated_calls == 1
