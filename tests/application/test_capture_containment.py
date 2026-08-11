from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from flameox.application import (
    CaptureService,
    ExecutionPolicy,
    WorkloadService,
)
from flameox.catalog import Catalog
from flameox.config import ContainmentPolicy
from flameox.domain import (
    DomainError,
    ErrorCode,
    ProcessCancellationCause,
)
from flameox.storage import RunStore, Workspace
from tests.support.capture import disable_containment, write_workload


def test_approved_agent_parses_containment_mode_before_policy_decision() -> None:
    configured_mode = ContainmentPolicy.REQUIRED_FOR_MCP.value

    assert ExecutionPolicy.APPROVED_AGENT.requires_containment(configured_mode) is True


@pytest.mark.parametrize(
    "writable_path",
    [
        pytest.param("../outside", id="parent-traversal"),
        pytest.param("escape", id="symlink-escape"),
        pytest.param(".git", id="git-metadata"),
        pytest.param(".diagnostics/build", id="workspace-metadata"),
    ],
)
def test_writable_roots_reject_traversal_and_symlink_escape(
    tmp_path: Path,
    writable_path: str,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.build]
argv = ["python", "-c", "print('build')"]
writable_paths = [{json.dumps(writable_path)}]
"""
    )

    with pytest.raises(DomainError) as refused:
        WorkloadService(workspace).writable_targets("build")

    assert refused.value.code is ErrorCode.EXECUTION_REFUSED


@pytest.mark.anyio
async def test_trusted_local_capture_is_uncontained_when_containment_tools_are_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    service = CaptureService(workspace)
    real_which = shutil.which

    def without_containment_tools(name: str) -> str | None:
        if name in {"bwrap", "systemd-run"}:
            return None
        return real_which(name)

    monkeypatch.setattr("flameox.application.capture.shutil.which", without_containment_tools)

    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    assert plan.containment == "uncontained"
    assert any("runs directly" in warning for warning in plan.warnings)
    assert plan.systemd_scope_unit is None


@pytest.mark.anyio
async def test_capture_preserves_runtime_storage_policy_termination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.wait]
argv = ["python", "-c", "import time; time.sleep(10)"]
timeout_seconds = 20
"""
    )
    disable_containment(workspace)
    monkeypatch.setattr(
        "flameox.application.capture.StorageQuota.require_capacity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "flameox.execution.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=10_000, used=9_999, free=1),
    )
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="wait",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    with pytest.raises(DomainError) as terminated:
        await service.execute(plan.plan_id)

    run = RunStore(workspace).read(plan.run_id)
    assert terminated.value.code is ErrorCode.STORAGE_QUOTA_EXCEEDED
    assert terminated.value.run_id == plan.run_id
    assert run.process is not None
    assert run.process.cancellation_cause == "storage_reserve_exceeded"
    assert run.process.cleanup_complete is True
    assert run.process.resources is not None
    assert (
        run.process.resources.policy_termination
        is ProcessCancellationCause.STORAGE_RESERVE_EXCEEDED
    )
    with Catalog(workspace).open_snapshot() as snapshot:
        resource_row = snapshot.execute(
            "SELECT policy_termination FROM runtime_resource_summaries WHERE run_id = ?",
            (plan.run_id,),
        ).fetchone()
    assert resource_row == ("storage_reserve_exceeded",)
