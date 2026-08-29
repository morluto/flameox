from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from flameox.adapters.nvbench import NvbenchExtractor
from flameox.application import CaptureService, ExecutionPolicy
from flameox.domain import ArtifactKind, CaptureStatus, ExecutionStatus
from flameox.storage import Workspace
from tests.support.capture import disable_containment

pytestmark = [
    pytest.mark.integration,
    pytest.mark.optional,
    pytest.mark.process,
    pytest.mark.serial,
    pytest.mark.requires_nvbench,
]


@pytest.mark.requires_nvbench
@pytest.mark.anyio
async def test_configured_nvbench_executable_emits_native_output(tmp_path: Path) -> None:
    executable = os.environ.get("FLAMEOX_NVBENCH_EXECUTABLE")
    assert executable is not None
    workspace = Workspace.initialize(tmp_path)
    disable_containment(workspace)
    (tmp_path / "flameox.toml").write_text(
        f"""
schema_version = 1
[workloads.bench]
argv = [{json.dumps(executable)}]
timeout_seconds = 120
execution_protocol = "nvbench"
"""
    )

    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="bench",
        adapter="nvbench",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_token)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    assert any(
        registration.kind is ArtifactKind.BENCHMARK_SAMPLES and registration.role == "primary"
        for registration in result.run.artifacts
    )
    sidecars = [
        registration
        for registration in result.run.artifacts
        if registration.kind is ArtifactKind.BENCHMARK_SAMPLES
        and registration.role == "nvbench_sidecar"
    ]
    assert sidecars

    extracted = NvbenchExtractor(workspace).extract(result.run.run_id)
    repeated = NvbenchExtractor(workspace).extract(result.run.run_id)
    assert extracted.benchmark_count > 0
    assert extracted.measurement_count > 0
    assert repeated.corpus_commit_id == extracted.corpus_commit_id
