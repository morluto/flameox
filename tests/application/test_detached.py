from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from flameox.application import (
    CapturePlanRegistry,
    CaptureService,
    DetachedCaptureManager,
    ExecutionPolicy,
)
from flameox.catalog import Catalog
from flameox.domain import DomainError, ErrorCode, ExecutionStatus
from flameox.storage import Workspace


def _workspace(tmp_path: Path, command: str, *, timeout: float = 30) -> Workspace:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.detached]
argv = [{json.dumps(sys.executable)}, "-c", {json.dumps(command)}]
timeout_seconds = {timeout}
"""
    )
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    return workspace


def _manager(workspace: Workspace) -> tuple[CaptureService, DetachedCaptureManager]:
    plans = CapturePlanRegistry()
    captures = CaptureService(workspace, plans=plans)
    return captures, DetachedCaptureManager(workspace, captures)


@pytest.mark.anyio
async def test_detached_start_is_idempotent_and_reconnects_by_run_id(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path, "import time; time.sleep(10)")
    captures, manager = _manager(workspace)
    plan = await captures.plan(
        workload_name="detached",
        adapter="command",
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )

    started = await manager.start(plan.plan_id, "review-attempt-001")
    repeated = await manager.start(plan.plan_id, "review-attempt-001")
    reconnected = manager.status(started.run_id)

    assert started.run_id == plan.run_id
    assert repeated.run_id == started.run_id
    assert reconnected.state == "running"
    assert reconnected.execution_status is ExecutionStatus.RUNNING
    assert reconnected.progress

    cancelled, raced_cancel = await asyncio.gather(
        manager.cancel(started.run_id),
        manager.cancel(started.run_id),
    )
    repeated_cancel = await manager.cancel(started.run_id)

    assert cancelled.state == "terminal"
    assert cancelled.execution_status is ExecutionStatus.CANCELLED
    assert raced_cancel.execution_status is ExecutionStatus.CANCELLED
    assert repeated_cancel.execution_status is ExecutionStatus.CANCELLED


@pytest.mark.anyio
async def test_detached_idempotency_key_cannot_authorize_another_plan(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path, "import time; time.sleep(10)")
    captures, manager = _manager(workspace)
    first = await captures.plan(
        workload_name="detached",
        adapter="command",
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )
    second = await captures.plan(
        workload_name="detached",
        adapter="command",
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )
    await manager.start(first.plan_id, "review-attempt-002")

    with pytest.raises(DomainError) as reused:
        await manager.start(second.plan_id, "review-attempt-002")

    assert reused.value.code is ErrorCode.INVALID_CAPTURE_PLAN
    await manager.cancel(first.run_id)


@pytest.mark.anyio
async def test_lost_start_caller_does_not_cancel_or_orphan_owned_capture(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path, "import time; time.sleep(10)")
    captures, manager = _manager(workspace)
    plan = await captures.plan(
        workload_name="detached",
        adapter="command",
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )
    start_call = asyncio.create_task(manager.start(plan.plan_id, "review-disconnect-001"))
    for _ in range(200):
        try:
            manager.status(plan.run_id)
            break
        except DomainError:
            await asyncio.sleep(0.005)
    start_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_call

    for _ in range(200):
        try:
            status = manager.status(plan.run_id)
        except DomainError:
            await asyncio.sleep(0.01)
            continue
        if status.state == "running":
            break
        await asyncio.sleep(0.01)

    assert status.state == "running"
    assert status.execution_status is ExecutionStatus.RUNNING
    await manager.cancel(plan.run_id)


@pytest.mark.anyio
async def test_detached_timeout_is_terminal_and_published_once(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, "import time; time.sleep(10)", timeout=0.1)
    captures, manager = _manager(workspace)
    plan = await captures.plan(
        workload_name="detached",
        adapter="command",
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )

    await manager.start(plan.plan_id, "review-timeout-001")
    for _ in range(200):
        status = manager.status(plan.run_id)
        if status.state == "terminal":
            break
        await asyncio.sleep(0.025)

    assert status.execution_status is ExecutionStatus.TIMED_OUT
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute(
            "SELECT count(*) FROM runs WHERE run_id = ?",
            (plan.run_id,),
        ).fetchone() == (1,)


@pytest.mark.anyio
async def test_detached_output_limit_failure_remains_inspectable(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, "print('x' * 10000)")
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(
                update={"containment": "disabled", "max_output_bytes": 100}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    captures, manager = _manager(workspace)
    plan = await captures.plan(
        workload_name="detached",
        adapter="command",
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )

    await manager.start(plan.plan_id, "review-output-001")
    for _ in range(200):
        status = manager.status(plan.run_id)
        if status.state == "terminal":
            break
        await asyncio.sleep(0.025)

    assert status.execution_status is ExecutionStatus.FAILED
    assert status.failure_code == ErrorCode.QUERY_BUDGET_EXCEEDED.value
    assert "output exceeded" in (status.failure_message or "").lower()


@pytest.mark.anyio
async def test_restart_reconnect_is_truthfully_read_only(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, "import time; time.sleep(10)")
    captures, manager = _manager(workspace)
    plan = await captures.plan(
        workload_name="detached",
        adapter="command",
        execution_policy=ExecutionPolicy.APPROVED_AGENT,
    )
    await manager.start(plan.plan_id, "review-restart-001")

    replacement_captures = CaptureService(workspace)
    replacement = DetachedCaptureManager(workspace, replacement_captures)
    repeated = await replacement.start(plan.plan_id, "review-restart-001")
    status = replacement.status(plan.run_id)

    assert repeated.run_id == plan.run_id
    assert status.state == "unmanaged_after_restart"
    assert status.limitations
    with pytest.raises(DomainError) as cancellation:
        await replacement.cancel(plan.run_id)
    assert cancellation.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    await manager.cancel(plan.run_id)
