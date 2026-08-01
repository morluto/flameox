from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, cast

import pytest

from flameox.analysis import RecipeService
from flameox.application import (
    CaptureService,
    ExecutionPolicy,
)
from flameox.catalog import Catalog
from flameox.domain import (
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ExternalExecutionContext,
    ValidationStatus,
)
from flameox.storage import ArtifactStore, RunStore, Workspace
from tests.support.capture import disable_containment, write_workload


@pytest.mark.anyio
@pytest.mark.process
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
    assert any(item.source == "adapter" for item in result.run.limitation_details)
    assert any(item.message in result.run.limitations for item in result.run.limitation_details)
    payload = ArtifactStore(workspace).get(result.run.artifacts[0].artifact_id)
    assert payload.payload_path.read_text().strip() == "candidate"
    with Catalog(workspace).open_snapshot() as snapshot:
        resource_row = snapshot.execute(
            "SELECT run_id, sampling_interval_ms, minimum_free_bytes, "
            "staging_growth_bytes, peak_rss_bytes, unavailable_metrics "
            "FROM runtime_resource_summaries WHERE run_id = ?",
            (result.run.run_id,),
        ).fetchone()
    assert resource_row is not None
    assert resource_row[0] == result.run.run_id
    assert resource_row[1] > 0
    assert isinstance(resource_row[5], list)
    memory = RecipeService(workspace).memory(result.run.run_id)
    assert memory.runtime_resource_totals is not None
    assert memory.runtime_resource_totals.run_count == 1
    assert memory.truncated is memory.runtime_resources_truncated
    hotspots = RecipeService(workspace).hotspots(result.run.run_id)
    assert hotspots.evidence_status == "unavailable"
    assert hotspots.unavailable_reason == "no_profile_artifact"
    events = [
        json.loads(line)
        for line in workspace.paths.operation_log.read_text().splitlines()
        if json.loads(line)["operation"] == "capture.execute"
    ]
    assert events[-1]["phase"] == "Evidence publication complete"
    assert "candidate" not in workspace.paths.operation_log.read_text()
    with pytest.raises(DomainError) as replay:
        await service.execute(plan.plan_id)
    assert replay.value.code is ErrorCode.INVALID_CAPTURE_PLAN


@pytest.mark.anyio
@pytest.mark.process
async def test_capture_publishes_project_relative_writable_root_growth(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "target").mkdir()
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.build]
argv = ["python", "-c", "from pathlib import Path; Path('target/output.bin').write_bytes(b'build')"]
writable_paths = ["target"]
"""
    )
    disable_containment(workspace)

    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="build",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    result = await service.execute(plan.plan_id)

    with Catalog(workspace).open_snapshot() as snapshot:
        row = snapshot.execute(
            "SELECT writable_root_identity, target_path, growth_bytes, available, "
            "unavailable_reason FROM runtime_writable_root_growth WHERE run_id = ?",
            (result.run.run_id,),
        ).fetchone()
    assert row is not None
    assert row[0]
    assert row[1] == "target"
    assert row[2] >= 0
    assert row[3] is True
    assert row[4] is None
    assert str(tmp_path) not in " ".join(str(item) for item in row)


@pytest.mark.anyio
@pytest.mark.process
async def test_structured_oracle_receipt_is_managed_preserved_and_projected(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "oracle.py").write_text(
        """import json
import os
from pathlib import Path

Path(os.environ[\"FLAMEOX_ORACLE_RECEIPT\"]).write_text(json.dumps({
    \"schema_version\": \"flameox.oracle-receipt.v1\",
    \"status\": \"fail\",
    \"reason\": \"contract_mismatch\",
    \"case_id\": \"candidate-mismatch\",
    \"output_field\": \"forward\",
    \"coordinate\": [0],
    \"expected\": {\"kind\": \"scalar\", \"value\": 1.0},
    \"observed\": {\"kind\": \"scalar\", \"value\": 2.0},
    \"absolute_error\": 1.0,
    \"tolerance\": {\"absolute\": 0.01},
}))
print(\"oracle diagnostic\")
"""
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.semantic]
argv = ["python", "-c", "print('workload')"]
[workloads.semantic.oracle]
strength = "contract_check"
argv = ["python", "oracle.py"]
receipt_schema = "flameox.oracle-receipt.v1"
"""
    )
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="semantic",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.execute(plan.plan_id)

    assert result.run.validation_status is ValidationStatus.FAILED
    assert result.run.oracle_receipt is not None
    assert result.run.oracle_receipt.receipt.output_field == "forward"
    receipt_registration = next(
        item for item in result.run.artifacts if item.role == "validation_receipt"
    )
    assert receipt_registration.artifact_id == result.run.oracle_receipt.receipt_artifact_id
    receipt_payload = ArtifactStore(workspace).get(receipt_registration.artifact_id)
    assert json.loads(receipt_payload.payload_path.read_text())["reason"] == "contract_mismatch"
    assert {item.role for item in result.run.artifacts} >= {
        "validation_receipt",
        "validation_contract_check",
    }


@pytest.mark.anyio
@pytest.mark.process
async def test_measurement_protocol_identity_binds_collector_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    identity = {"digest": "first"}
    monkeypatch.setattr(
        service,
        "_executable_identity",
        lambda _executable: dict(identity),
    )

    first_plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    first = await service.execute(first_plan.plan_id)
    identity["digest"] = "second"
    second_plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    second = await service.execute(second_plan.plan_id)

    assert first.run.measurement_protocol_id != second.run.measurement_protocol_id


@pytest.mark.anyio
@pytest.mark.process
async def test_concurrent_captures_execute_while_publications_serialize(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    first, second = await asyncio.gather(
        service.plan(
            workload_name="echo",
            adapter="command",
            parameters={"message": "hello"},
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        ),
        service.plan(
            workload_name="echo",
            adapter="command",
            parameters={"message": "candidate"},
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        ),
    )

    results = await asyncio.gather(
        service.execute(first.plan_id),
        service.execute(second.plan_id),
    )

    assert {result.run.execution_status for result in results} == {ExecutionStatus.SUCCEEDED}
    assert len({result.run.run_id for result in results}) == 2
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute("SELECT count(DISTINCT run_id) FROM runs").fetchone() == (2,)


@pytest.mark.anyio
@pytest.mark.process
async def test_capture_cancellation_leaves_terminal_run_revision(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
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
    context = ExternalExecutionContext(
        orchestrator="crabbox",
        provider="runpod",
        lease_id="lease-cancelled",
        worker_id="worker-cancelled",
        orchestration_run_id="orchestration-cancelled",
        sensitivity="sensitive",
    )
    plan = await service.plan(
        workload_name="wait",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        external_context=context,
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
    assert terminal.external_context == context


@pytest.mark.anyio
@pytest.mark.process
async def test_artifact_registration_failure_is_terminal_and_cleans_staging(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    config = workspace.config.model_copy(
        update={
            "capture": workspace.config.capture.model_copy(update={"max_artifact_bytes": 1}),
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"}),
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    with pytest.raises(DomainError) as error:
        await service.execute(plan.plan_id)

    terminal = RunStore(workspace).read(plan.run_id)
    assert error.value.code is ErrorCode.ARTIFACT_TOO_LARGE
    assert terminal.execution_status is ExecutionStatus.FAILED
    assert terminal.capture_status is CaptureStatus.FAILED
    assert not any((workspace.paths.staging / "captures").iterdir())


@pytest.mark.anyio
@pytest.mark.process
async def test_publication_failure_is_terminal_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    original = cast(Any, service.publisher.publish_rows)

    def fail_evidence_publication(
        rows: dict[str, list[dict[str, object]]],
        *args: object,
        **kwargs: object,
    ) -> object:
        if "artifact_registrations" in rows:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "simulated publication failure",
            )
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(service.publisher, "publish_rows", fail_evidence_publication)

    with pytest.raises(DomainError, match="simulated publication failure"):
        await service.execute(plan.plan_id)

    terminal = RunStore(workspace).read(plan.run_id)
    assert terminal.execution_status is ExecutionStatus.FAILED
    assert terminal.capture_status is CaptureStatus.FAILED
    assert not any((workspace.paths.staging / "captures").iterdir())


@pytest.mark.parametrize("cancel_at_phase", (1, 2, 3, 4, 5, 6, 7))
@pytest.mark.anyio
@pytest.mark.process
async def test_capture_cancellation_at_awaited_phase_is_terminal(
    tmp_path: Path,
    cancel_at_phase: int,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    async def cancel(
        completed: float,
        _total: float,
        _message: str,
    ) -> None:
        if completed == cancel_at_phase:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await service.execute(plan.plan_id, progress=cancel)

    terminal = RunStore(workspace).read(plan.run_id)
    assert terminal.execution_status is ExecutionStatus.CANCELLED
    assert terminal.capture_status is CaptureStatus.CANCELLED


@pytest.mark.anyio
@pytest.mark.process
async def test_cancellation_during_atomic_publication_finishes_then_marks_run_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    started = threading.Event()
    release = threading.Event()
    original = cast(Any, service.publisher.publish_rows)

    def delayed(*args: object, **kwargs: object) -> object:
        started.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(service.publisher, "publish_rows", delayed)
    task = asyncio.create_task(service.execute(plan.plan_id))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    terminal = RunStore(workspace).read(plan.run_id)
    assert terminal.execution_status is ExecutionStatus.CANCELLED
    assert terminal.capture_status is CaptureStatus.CANCELLED
    with Catalog(workspace).open_snapshot() as snapshot:
        latest = snapshot.execute(
            "SELECT execution_status FROM runs WHERE run_id = ? ORDER BY published_at DESC LIMIT 1",
            (plan.run_id,),
        ).fetchone()
    assert latest == ("cancelled",)
