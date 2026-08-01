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
from flameox.domain import (
    DomainError,
    ErrorCode,
)
from flameox.storage import RunStore, Workspace
from tests.support.capture import disable_containment, write_workload


def test_writable_roots_reject_traversal_and_symlink_escape(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    for value in ("../outside", "escape", ".git", ".diagnostics/build"):
        (tmp_path / "flameox.toml").write_text(
            f"""
schema_version = 1
[workloads.build]
argv = ["python", "-c", "print('build')"]
writable_paths = [{json.dumps(value)}]
"""
        )
        with pytest.raises(DomainError) as refused:
            WorkloadService(workspace).writable_targets("build")
        assert refused.value.code is ErrorCode.EXECUTION_REFUSED


@pytest.mark.anyio
async def test_trusted_local_capture_does_not_require_systemd_user_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    service = CaptureService(workspace)
    real_which = shutil.which

    def available_executable(name: str) -> str | None:
        if name in {"bwrap", "systemd-run"}:
            return "/usr/bin/true"
        return real_which(name)

    async def unexpected_user_manager_probe(_systemd_run: str) -> bool:
        raise AssertionError("trusted-local execution must not probe containment")

    monkeypatch.setattr("flameox.application.capture.shutil.which", available_executable)
    monkeypatch.setattr(service, "_systemd_user_scope_available", unexpected_user_manager_probe)

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
    assert run.process.resources.policy_termination == "storage_reserve_exceeded"
    with Catalog(workspace).open_snapshot() as snapshot:
        resource_row = snapshot.execute(
            "SELECT policy_termination FROM runtime_resource_summaries WHERE run_id = ?",
            (plan.run_id,),
        ).fetchone()
    assert resource_row == ("storage_reserve_exceeded",)
