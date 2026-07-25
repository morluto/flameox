from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from flamo.adapters import PerfettoExtractor, PyPerfExtractor
from flamo.application import CaptureService, ExecutionPolicy, WorkloadService
from flamo.config import WorkspaceConfig
from flamo.domain import CaptureStatus, DomainError, ErrorCode, ExecutionStatus
from flamo.storage import ArtifactStore, RunStore, Workspace


def write_workload(project: Path, *, message: str = "hello") -> None:
    (project / "flamo.toml").write_text(
        f"""
schema_version = 1

[workloads.echo]
argv = ["python", "-c", "print('{{message}}')"]
cwd = "."
timeout_seconds = 5

[workloads.echo.parameters]
message = ["{message}", "candidate"]
"""
    )


def disable_containment(workspace: Workspace) -> None:
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
        }
    )
    assert isinstance(config, WorkspaceConfig)
    workspace.paths.config.write_text(config.to_toml())


def test_workload_approval_is_bound_to_canonical_definition(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    service = WorkloadService(workspace)

    definition = service.definition("echo")
    assert definition.approved_definition_digest is None
    with pytest.raises(DomainError) as unapproved:
        service.resolve("echo", require_approval=True)

    approved = service.approve("echo")
    instance = service.resolve(
        "echo",
        {"message": "candidate"},
        require_approval=True,
    )
    write_workload(tmp_path, message="changed")

    assert unapproved.value.code is ErrorCode.EXECUTION_REFUSED
    assert approved.approved_definition_digest == approved.workload_definition_id
    assert instance.command.argv[-1] == "print('candidate')"
    assert service.definition("echo").approved_definition_digest is None


@pytest.mark.anyio
async def test_capture_plan_is_single_use_and_publishes_process_evidence(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        parameters={"message": "candidate"},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.execute(plan.plan_id)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.capture_status is CaptureStatus.REGISTERED
    assert result.run.source_state_id is not None
    assert result.run.artifacts
    payload = ArtifactStore(workspace).get(result.run.artifacts[0].artifact_id)
    assert payload.payload_path.read_text().strip() == "candidate"
    with pytest.raises(DomainError) as replay:
        await service.execute(plan.plan_id)
    assert replay.value.code is ErrorCode.INVALID_CAPTURE_PLAN


@pytest.mark.anyio
async def test_capture_cancellation_leaves_terminal_run_revision(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flamo.toml").write_text(
        """
schema_version = 1
[workloads.wait]
argv = ["python", "-c", "import time; time.sleep(10)"]
cwd = "."
timeout_seconds = 30
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="wait",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    task = asyncio.create_task(service.execute(plan.plan_id))
    await asyncio.sleep(0.1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    runs = [path.name for path in workspace.paths.runs.iterdir() if path.is_dir()]
    assert len(runs) == 1
    terminal = RunStore(workspace).read(runs[0])
    assert terminal.execution_status is ExecutionStatus.CANCELLED
    assert terminal.capture_status is CaptureStatus.CANCELLED
    assert terminal.finished_at is not None


@pytest.mark.anyio
async def test_coverage_capture_uses_supported_launcher_and_registers_native_data(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "workload.py").write_text("value = sum(range(10))\nprint(value)\n")
    (tmp_path / "flamo.toml").write_text(
        """
schema_version = 1
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
    result = await service.execute(plan.plan_id)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert any(item.kind.value == "execution_coverage" for item in result.run.artifacts)


@pytest.mark.anyio
async def test_py_spy_capture_round_trips_through_perfetto(
    tmp_path: Path,
) -> None:
    binaries = sorted(
        (Path.home() / ".local" / "share" / "perfetto" / "prebuilts").glob(
            "trace_processor_shell-*"
        )
    )
    if not binaries:
        pytest.skip("A local Trace Processor binary is not installed.")
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "busy.py").write_text(
        "import time\n"
        "total = 0\n"
        "deadline = time.monotonic() + 1.0\n"
        "while time.monotonic() < deadline:\n"
        "    total += sum(i * i for i in range(5000))\n"
        "print(total)\n"
    )
    (tmp_path / "flamo.toml").write_text(
        """
schema_version = 1
[workloads.busy]
argv = ["python", "busy.py"]
cwd = "."
timeout_seconds = 30
"""
    )
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"}),
            "analysis": workspace.config.analysis.model_copy(
                update={"trace_processor_path": str(binaries[-1])}
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
    captured = await service.execute(plan.plan_id)
    extracted = await PerfettoExtractor(workspace).extract(captured.run.run_id)

    assert captured.run.process is not None
    expected_status = (
        ExecutionStatus.SUCCEEDED if captured.run.process.exit_code == 0 else ExecutionStatus.FAILED
    )
    assert captured.run.execution_status is expected_status
    assert extracted.slice_count > 0
    assert extracted.frame_count > 0


@pytest.mark.anyio
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
    (tmp_path / "flamo.toml").write_text(
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
    assert extracted.measurement_count == 9
    assert extracted.warmup_count >= 3
