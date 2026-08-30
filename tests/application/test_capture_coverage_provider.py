from __future__ import annotations

from pathlib import Path

import pytest

from flameox.adapters.coverage import qualified_control_coverage_reader_version
from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.domain import DomainError, ErrorCode, ExecutionStatus
from flameox.storage import Workspace
from tests.support.capture import disable_containment

pytestmark = [
    pytest.mark.integration,
    pytest.mark.process,
    pytest.mark.serial,
]


@pytest.mark.anyio
@pytest.mark.process
async def test_coverage_capture_uses_supported_launcher_and_registers_native_data(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "workload.py").write_text("value = sum(range(10))\nprint(value)\n")
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.script]
argv = ["python", "workload.py"]
cwd = "."
timeout_seconds = 10
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="script",
        adapter="coverage",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_token)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    registration = next(
        item for item in result.run.artifacts if item.kind.value == "execution_coverage"
    )
    assert registration.producer == "coverage"
    assert registration.producer_version == plan.adapter_version
    assert (
        plan.semantics.configuration["reader_version"]
        == qualified_control_coverage_reader_version()
    )


@pytest.mark.anyio
async def test_coverage_plan_requires_a_qualified_control_reader_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flameox.application.capture as capture_module

    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "workload.py").write_text("print('never launched')\n")
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.script]
argv = ["python", "workload.py"]
cwd = "."
timeout_seconds = 10
"""
    )

    def reject_reader() -> str:
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            "The Flameox control environment has no coverage.py reader.",
        )

    monkeypatch.setattr(capture_module, "qualified_control_coverage_reader_version", reject_reader)

    with pytest.raises(DomainError) as error:
        await CaptureService(workspace).plan(
            workload_name="script",
            adapter="coverage",
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )

    assert error.value.code is ErrorCode.CAPABILITY_UNAVAILABLE
    assert not (workspace.paths.staging / "captures").exists()
