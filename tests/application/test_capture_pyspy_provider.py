from __future__ import annotations

from pathlib import Path

import pytest

from flameox.adapters import PerfettoExtractor
from flameox.application import CaptureService, ExecutionPolicy
from flameox.domain import ExecutionStatus
from flameox.storage import Workspace
from tests.support.providers import require_trace_processor

pytestmark = [
    pytest.mark.integration,
    pytest.mark.optional,
    pytest.mark.process,
    pytest.mark.serial,
    pytest.mark.requires_perfetto,
    pytest.mark.requires_pyspy,
]


@pytest.mark.anyio
@pytest.mark.optional
@pytest.mark.requires_perfetto
@pytest.mark.requires_pyspy
@pytest.mark.process
async def test_py_spy_capture_round_trips_through_perfetto(
    tmp_path: Path,
) -> None:
    binary = require_trace_processor()
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "busy.py").write_text(
        "import time\n"
        "total = 0\n"
        "deadline = time.monotonic() + 1.0\n"
        "while time.monotonic() < deadline:\n"
        "    total += sum(i * i for i in range(5000))\n"
        "print(total)\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.busy]
argv = ["python", "busy.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            ),
            "analysis": workspace.config.analysis.validated_copy(
                update={"trace_processor_path": str(binary)}
            ),
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="busy",
        adapter="py-spy",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    captured = await service.execute(plan.plan_token)
    extracted = await PerfettoExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.process is not None
    expected_status = (
        ExecutionStatus.SUCCEEDED if captured.run.process.exit_code == 0 else ExecutionStatus.FAILED
    )
    assert captured.run.execution_status is expected_status
    assert extracted.slice_count > 0
    assert extracted.frame_count > 0
