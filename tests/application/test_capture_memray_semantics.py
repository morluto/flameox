from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.provider_runtime import ProviderRuntimeManager
from flameox.domain import ArtifactKind, CaptureStatus, ExecutionStatus, ValidationStatus
from flameox.storage import ArtifactStore, Workspace
from tests.support.capture import disable_containment

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


def _fake_memray_interpreter(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import pathlib
if 'FLAMEOX_MEMRAY_OUTPUT' in os.environ:
    output = pathlib.Path(os.environ['FLAMEOX_MEMRAY_OUTPUT'])
    output.write_bytes(b'partial memray evidence')
    observations = pathlib.Path(os.environ['FLAMEOX_OBSERVATIONS_PATH'])
    observations.write_text(json.dumps({
        'name': 'flameox.memray.region.start',
        'values': {'region': 'steady_step', 'warmup_count': 2},
    }) + '\\n')
print('{"memray":"1.20.0"}')
"""
    )
    path.chmod(0o755)


@pytest.mark.anyio
async def test_memray_unclosed_region_marks_validation_error_and_preserves_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    interpreter = tmp_path / "workload-env" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    _fake_memray_interpreter(interpreter)
    (tmp_path / "flameox.toml").write_text(
        f"""
[workloads.memory]
argv = ["{interpreter}", "workload.py"]
timeout_seconds = 5
"""
    )
    monkeypatch.setattr(
        ProviderRuntimeManager,
        "find_distribution",
        lambda _self, **_kwargs: object(),
    )

    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="memory",
        adapter="memray",
        adapter_options={
            "mode": "sdk",
            "region": "steady_step",
            "warmup_count": 2,
        },
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    run = (await service.execute(plan.plan_token)).run

    assert run.execution_status is ExecutionStatus.FAILED
    assert run.capture_status is CaptureStatus.REGISTERED
    assert run.validation_status is ValidationStatus.ERROR
    assert any(detail.code == "memray_region_unclosed" for detail in run.limitation_details)
    profile = next(item for item in run.artifacts if item.kind is ArtifactKind.MEMORY_PROFILE)
    assert ArtifactStore(workspace).get(profile.artifact_id).payload_path.read_bytes() == (
        b"partial memray evidence"
    )
    assert any(item.kind is ArtifactKind.SEMANTIC_OBSERVATIONS for item in run.artifacts)
