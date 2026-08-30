from __future__ import annotations

import sys
from pathlib import Path

import pytest

from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.provider_runtime import ProviderRuntimeManager
from flameox.domain import ExecutionStatus, ValidationStatus
from flameox.storage import ArtifactStore, Workspace
from tests.support.capture import disable_containment

pytestmark = [
    pytest.mark.integration,
    pytest.mark.optional,
    pytest.mark.process,
    pytest.mark.requires_memray,
    pytest.mark.serial,
]


@pytest.mark.anyio
async def test_memray_sdk_capture_preserves_one_native_region_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("memray", reason="optional provider unavailable: install memray")
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "workload.py").write_text(
        "from flameox.sdk import memray_region\n"
        "setup = bytearray(4_000_000)\n"
        "with memray_region('steady_step'):\n"
        "    measured = bytearray(128_000)\n"
        "print(len(setup) + len(measured))\n"
    )
    (tmp_path / "flameox.toml").write_text(
        f"""
[workloads.memory]
argv = [{sys.executable!r}, "workload.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    disable_containment(workspace)
    monkeypatch.setattr(
        ProviderRuntimeManager,
        "find_distribution",
        lambda _self, **_kwargs: object(),
    )
    service = CaptureService(workspace)

    plan = await service.plan(
        workload_name="memory",
        adapter="memray",
        adapter_options={"mode": "sdk", "region": "steady_step", "warmup_count": 2},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_token)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert plan.semantics.scope.mode is not None
    assert plan.semantics.scope.mode.value == "sdk"
    assert plan.semantics.scope.bounds == {"warmup_count": 2}
    assert plan.semantics.scope.filters == {
        "region": "steady_step",
        "thread_scope": "all_threads",
    }
    assert plan.semantics.scope.process_scope is not None
    assert plan.semantics.scope.process_scope.value == "workload_process"
    assert plan.workload_definition_id
    assert plan.workload_instance.workload_instance_id
    assert plan.workload_instance.command.timeout_seconds == 30
    assert plan.collector_environment["FLAMEOX_MEMRAY_OUTPUT"].startswith(
        str(workspace.paths.staging)
    )
    assert plan.execution_limits.max_artifact_bytes > 0
    profile = next(item for item in result.run.artifacts if item.kind.value == "memory_profile")
    assert ArtifactStore(workspace).get(profile.artifact_id).payload_path.suffix == ".bin"
    assert result.run.semantics == plan.semantics
    assert result.run.source_state_id is not None
    assert result.run.validation_status is ValidationStatus.NOT_REQUESTED
