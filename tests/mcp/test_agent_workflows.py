from __future__ import annotations

import hashlib
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from mcp import Client

from flameox.mcp import create_server
from flameox.storage import Workspace

ToolCall = Callable[[str, dict[str, Any]], Awaitable[Any]]
Setup = Callable[[Workspace], None]
Exercise = Callable[[ToolCall, Workspace], Awaitable[None]]
Snapshot = dict[str, str]


@dataclass(frozen=True, slots=True)
class WorkflowCase:
    name: str
    allowed_paths: tuple[str, ...]
    expected_tool_calls: int
    expected_invalid_calls: int
    expected_repeated_calls: int
    setup: Setup
    exercise: Exercise


@dataclass(frozen=True, slots=True)
class TraceMetrics:
    tool_calls: int
    invalid_calls: int
    repeated_calls: int
    writes_outside_permitted: tuple[str, ...]


def _structured(result: Any) -> dict[str, Any]:
    assert result.structured_content is not None
    return cast(dict[str, Any], result.structured_content)


def _snapshot(root: Path) -> Snapshot:
    snapshot: Snapshot = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = f"symlink:{path.readlink()}"
        elif path.is_dir():
            snapshot[relative] = "directory"
        elif path.is_file():
            snapshot[relative] = f"file:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        else:
            snapshot[relative] = "other"
    return snapshot


def _path_is_permitted(path: str, allowed_paths: tuple[str, ...]) -> bool:
    return any(
        path == allowed.removesuffix("/") or path.startswith(allowed)
        if allowed.endswith("/")
        else path == allowed
        for allowed in allowed_paths
    )


def _outside_permitted(
    before: Snapshot,
    after: Snapshot,
    allowed_paths: tuple[str, ...],
) -> tuple[str, ...]:
    changed = {path for path in before.keys() | after.keys() if before.get(path) != after.get(path)}
    return tuple(sorted(path for path in changed if not _path_is_permitted(path, allowed_paths)))


def _no_setup(workspace: Workspace) -> None:
    del workspace


def _write_workload(workspace: Workspace, argv: str = "pass") -> None:
    (workspace.project_root / "flameox.toml").write_text(
        f'schema_version = 1\n[workloads.probe]\nargv = ["python", "-c", "{argv}"]\n'
    )


def _setup_existing_workload(workspace: Workspace) -> None:
    (workspace.project_root / "flameox.toml").write_text(
        'schema_version = 1\n[workloads.existing]\nargv = ["python", "-c", "pass"]\n'
    )


def _setup_probe_workload(workspace: Workspace) -> None:
    _write_workload(workspace)


def _setup_invalid_configuration(workspace: Workspace) -> None:
    (workspace.project_root / "flameox.toml").write_text(
        'schema_version = 1\n[experiments.broken]\nworkload = "missing"\n'
    )


def _setup_trusted_local_execution(workspace: Workspace) -> None:
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())


async def _exercise_fresh_repository(call: ToolCall, workspace: Workspace) -> None:
    del workspace
    status = await call("workload_configuration_status", {})
    repeated = await call("workload_configuration_status", {})
    configured = await call(
        "configure_workload",
        {"name": "probe", "operation": "create", "argv": [sys.executable, "-c", "pass"]},
    )

    assert status.is_error is False
    assert _structured(status)["result"]["next_tool"] == "configure_workload"
    assert _structured(repeated) == _structured(status)
    assert configured.is_error is False
    assert _structured(configured)["result"]["action"] == "created"


async def _exercise_add_workload(call: ToolCall, workspace: Workspace) -> None:
    del workspace
    status = await call("workload_configuration_status", {})
    configured = await call(
        "configure_workload",
        {"name": "added", "operation": "create", "argv": ["python", "-c", "pass"]},
    )

    assert status.is_error is False
    assert _structured(status)["result"]["next_tool"] == "list_declared_workflows"
    assert configured.is_error is False
    assert _structured(configured)["result"]["action"] == "created"


async def _exercise_replace_stale_workload(call: ToolCall, workspace: Workspace) -> None:
    del workspace
    status = await call("workload_configuration_status", {})
    assert status.is_error is False
    stale = await call(
        "configure_workload",
        {
            "name": "probe",
            "operation": "replace",
            "argv": ["python", "-c", "print(1)"],
            "expected_configuration_id": "sha256:" + "0" * 64,
        },
    )
    current_id = _structured(status)["result"]["configuration_id"]
    updated = await call(
        "configure_workload",
        {
            "name": "probe",
            "operation": "replace",
            "argv": ["python", "-c", "print(1)"],
            "expected_configuration_id": current_id,
        },
    )

    assert stale.is_error is True
    assert _structured(stale)["error"]["code"] == "REVISION_CONFLICT"
    assert updated.is_error is False
    assert _structured(updated)["result"]["action"] == "updated"


async def _exercise_invalid_configuration(call: ToolCall, workspace: Workspace) -> None:
    del workspace
    status = await call("workload_configuration_status", {})

    assert status.is_error is False
    assert _structured(status)["result"]["status"] == "invalid"
    assert _structured(status)["result"]["next_tool"] == "configure_workload"


async def _exercise_ignored_adhoc_capture_arguments(
    call: ToolCall,
    workspace: Workspace,
) -> None:
    result = await call(
        "plan_capture",
        {
            "workload_name": "probe",
            "adapter": "command",
            "parameters": {},
            "argv": ["echo", "not-declared"],
        },
    )

    assert result.is_error is False
    command = _structured(result)["result"]["workload_instance"]["command"]
    assert command["argv"][-1] == "pass"
    assert "not-declared" not in command["argv"]
    assert command["cwd"] == str(workspace.project_root)


async def _exercise_normal_workflow(call: ToolCall, workspace: Workspace) -> None:
    status = await call("workload_configuration_status", {})
    assert status.is_error is False
    assert _structured(status)["result"]["status"] == "missing"

    configured = await call(
        "configure_workload",
        {
            "name": "probe",
            "operation": "create",
            "argv": [sys.executable, "-c", "print('trace')"],
        },
    )
    assert configured.is_error is False
    assert _structured(configured)["result"]["action"] == "created"
    assert not any(workspace.paths.runs.iterdir())

    listed = await call("list_declared_workflows", {"kind": "workload"})
    assert listed.is_error is False
    assert [item["name"] for item in _structured(listed)["result"]["workflows"]] == ["probe"]

    inspected = await call("get_declared_workflow", {"kind": "workload", "name": "probe"})
    assert inspected.is_error is False
    assert _structured(inspected)["result"]["summary"]["name"] == "probe"

    capabilities = await call("list_capabilities", {"mode": "passive"})
    assert capabilities.is_error is False
    assert _structured(capabilities)["result"]["capabilities"]

    planned = await call(
        "plan_capture",
        {"workload_name": "probe", "adapter": "command", "parameters": {}},
    )
    assert planned.is_error is False
    assert _structured(planned)["result"]["workload_name"] == "probe"

    executed = await call(
        "execute_capture_plan",
        {"plan_id": _structured(planned)["result"]["plan_id"]},
    )
    assert executed.is_error is False
    assert _structured(executed)["result"]["execution_status"] == "succeeded"


CASES = (
    WorkflowCase(
        name="fresh_repository",
        allowed_paths=("flameox.toml",),
        expected_tool_calls=3,
        expected_invalid_calls=0,
        expected_repeated_calls=1,
        setup=_no_setup,
        exercise=_exercise_fresh_repository,
    ),
    WorkflowCase(
        name="add_workload",
        allowed_paths=("flameox.toml",),
        expected_tool_calls=2,
        expected_invalid_calls=0,
        expected_repeated_calls=0,
        setup=_setup_existing_workload,
        exercise=_exercise_add_workload,
    ),
    WorkflowCase(
        name="replace_stale_workload",
        allowed_paths=("flameox.toml",),
        expected_tool_calls=3,
        expected_invalid_calls=0,
        expected_repeated_calls=0,
        setup=_setup_probe_workload,
        exercise=_exercise_replace_stale_workload,
    ),
    WorkflowCase(
        name="invalid_configuration",
        allowed_paths=(),
        expected_tool_calls=1,
        expected_invalid_calls=0,
        expected_repeated_calls=0,
        setup=_setup_invalid_configuration,
        exercise=_exercise_invalid_configuration,
    ),
    WorkflowCase(
        name="ignored_adhoc_capture_arguments_cannot_override_workload",
        allowed_paths=(),
        expected_tool_calls=1,
        expected_invalid_calls=0,
        expected_repeated_calls=0,
        setup=_setup_probe_workload,
        exercise=_exercise_ignored_adhoc_capture_arguments,
    ),
    WorkflowCase(
        name="normal_configure_discover_plan_execute",
        allowed_paths=("flameox.toml", ".diagnostics/"),
        expected_tool_calls=7,
        expected_invalid_calls=0,
        expected_repeated_calls=0,
        setup=_setup_trusted_local_execution,
        exercise=_exercise_normal_workflow,
    ),
)


async def _run_case(case: WorkflowCase, root: Path) -> TraceMetrics:
    workspace = Workspace.initialize(root)
    case.setup(workspace)
    before = _snapshot(root)
    calls: list[tuple[str, dict[str, Any]]] = []
    invalid_calls = 0
    repeated_calls = 0
    previous: tuple[str, dict[str, Any]] | None = None

    async with Client(create_server(root), raise_exceptions=True) as client:

        async def call(name: str, arguments: dict[str, Any]) -> Any:
            nonlocal invalid_calls, repeated_calls, previous
            current = (name, arguments)
            if current == previous:
                repeated_calls += 1
            previous = current
            calls.append(current)
            result = await client.call_tool(name, arguments)
            if result.is_error and result.structured_content is not None:
                error = result.structured_content.get("error")
                if isinstance(error, dict) and error.get("code") == "INVALID_ARGUMENTS":
                    invalid_calls += 1
            return result

        await case.exercise(call, workspace)

    return TraceMetrics(
        tool_calls=len(calls),
        invalid_calls=invalid_calls,
        repeated_calls=repeated_calls,
        writes_outside_permitted=_outside_permitted(before, _snapshot(root), case.allowed_paths),
    )


def test_write_audit_detects_modifications_deletions_and_prefix_collisions(tmp_path: Path) -> None:
    (tmp_path / "flameox.toml").write_text("before")
    (tmp_path / "edited.txt").write_text("before")
    (tmp_path / "deleted.txt").write_text("before")
    (tmp_path / "deleted-empty").mkdir()
    allowed_directory = tmp_path / "allowed"
    allowed_directory.mkdir()
    (allowed_directory / "existing.txt").write_text("before")
    before = _snapshot(tmp_path)

    (tmp_path / "flameox.toml").write_text("after")
    (tmp_path / "flameox.toml.backup").write_text("unexpected")
    (tmp_path / "edited.txt").write_text("after")
    (tmp_path / "deleted.txt").unlink()
    (tmp_path / "deleted-empty").rmdir()
    (tmp_path / "created-empty").mkdir()
    (allowed_directory / "existing.txt").write_text("after")

    assert _outside_permitted(
        before,
        _snapshot(tmp_path),
        ("flameox.toml", "allowed/"),
    ) == (
        "created-empty",
        "deleted-empty",
        "deleted.txt",
        "edited.txt",
        "flameox.toml.backup",
    )


@pytest.mark.anyio
@pytest.mark.process
@pytest.mark.serial
@pytest.mark.parametrize("case", CASES, ids=lambda item: item.name)
async def test_offline_agent_workflow_traces_measure_contract_behavior(
    case: WorkflowCase,
    tmp_path: Path,
) -> None:
    metrics = await _run_case(case, tmp_path)

    assert metrics.tool_calls == case.expected_tool_calls
    assert metrics.invalid_calls == case.expected_invalid_calls
    assert metrics.repeated_calls == case.expected_repeated_calls
    assert metrics.writes_outside_permitted == ()
