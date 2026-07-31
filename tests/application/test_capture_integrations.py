from __future__ import annotations

from pathlib import Path

import pytest

from flameox.adapters import PyPerfExtractor
from flameox.application import CaptureService, ExecutionPolicy
from flameox.domain import ExecutionStatus
from flameox.storage import Workspace


@pytest.mark.anyio
@pytest.mark.process
async def test_pyperf_capture_preserves_native_worker_hierarchy(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "scan.py").write_text(
        "values = list(range(2000))\n"
        "total = 0\n"
        "for value in reversed(values):\n"
        "    total += value\n"
        "assert total > 0\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.scan]
argv = ["python", "scan.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="scan",
        adapter="pyperf",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    captured = await service.execute(plan.plan_id)
    extracted = PyPerfExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.execution_status is ExecutionStatus.SUCCEEDED
    separator = captured.plan.collector_argv.index("--")
    assert Path(captured.plan.collector_argv[separator + 1]).name.startswith("python")
    assert captured.plan.collector_argv[separator + 2 :] == ("scan.py",)
    assert extracted.measurement_count == 9
    assert extracted.warmup_count >= 3
    primary = next(artifact for artifact in captured.run.artifacts if artifact.role == "primary")
    assert captured.plan.adapter_version is not None
    assert primary.producer == "pyperf"
    assert primary.producer_version == captured.plan.adapter_version
    assert extracted.limitations == ()
