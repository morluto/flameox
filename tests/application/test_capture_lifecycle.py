from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from flameox.analysis import RecipeService
from flameox.application import (
    CaptureService,
    ExecutionPolicy,
    RunProjectionService,
)
from flameox.application.capture_admission import CaptureAdmissionService
from flameox.application.proc import read_boot_id
from flameox.catalog import Catalog
from flameox.domain import (
    CaptureLease,
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ExternalExecutionContext,
    ProjectionState,
    Sensitivity,
    ValidationStatus,
)
from flameox.domain.models import utc_now
from flameox.storage import (
    ArtifactStore,
    CaptureAdmissionRecord,
    CaptureAdmissionStore,
    ProjectionIntentStore,
    RunStore,
    Workspace,
)
from tests.support.capture import disable_containment, write_workload

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


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

    result = await service.execute(plan.plan_token)

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
            "staging_growth_bytes, peak_rss_bytes, peak_rss_backend, unavailable_metrics "
            "FROM runtime_resource_summaries WHERE run_id = ?",
            (result.run.run_id,),
        ).fetchone()
    assert resource_row is not None
    assert resource_row[0] == result.run.run_id
    assert resource_row[1] > 0
    assert resource_row[5] in {None, "psutil_recursive_polling"}
    assert isinstance(resource_row[6], list)
    memory = RecipeService(workspace).memory(result.run.run_id)
    assert memory.runtime_resource_totals is not None
    assert memory.runtime_resource_totals.run_count == 1
    assert memory.evidence.status in {"available", "partial"}
    if not memory.measurements and not memory.hotspots:
        assert memory.evidence.status == "partial"
        assert memory.evidence.reason == "no_memory_profile_artifact_runtime_evidence_present"
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
        await service.execute(plan.plan_token)
    assert replay.value.code is ErrorCode.PLAN_TOKEN_CONSUMED


@pytest.mark.anyio
@pytest.mark.process
async def test_capture_expected_plan_identity_fails_before_consumption(tmp_path: Path) -> None:
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

    with pytest.raises(DomainError) as mismatch:
        await service.execute(plan.plan_token, expected_plan_id="sha256:" + "0" * 64)

    assert mismatch.value.code is ErrorCode.PLAN_ID_MISMATCH
    assert "expected intent" in mismatch.value.message
    result = await service.execute(plan.plan_token, expected_plan_id=plan.plan_id)
    assert result.run.execution_status is ExecutionStatus.SUCCEEDED


@pytest.mark.anyio
@pytest.mark.process
async def test_progress_delivery_failure_cannot_fail_a_healthy_capture(
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

    async def failed_notification(completed: float, total: float, message: str) -> None:
        raise RuntimeError("progress transport closed")

    result = await service.execute(plan.plan_token, progress=failed_notification)

    assert result.run.execution_status is ExecutionStatus.SUCCEEDED
    assert result.run.capture_status is CaptureStatus.REGISTERED


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
    result = await service.execute(plan.plan_token)

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

    result = await service.execute(plan.plan_token)

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
    bindir = tmp_path / "bin"
    bindir.mkdir()
    collector = bindir / "python"
    collector.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    collector.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    first_plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    first = await service.execute(first_plan.plan_token)
    original = collector.read_bytes()
    collector.write_bytes(original + b"\n# identity changed\n")
    second_plan = await service.plan(
        workload_name="echo",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    second = await service.execute(second_plan.plan_token)
    collector.write_bytes(original)

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
        service.execute(first.plan_token),
        service.execute(second.plan_token),
    )

    assert {result.run.execution_status for result in results} == {ExecutionStatus.SUCCEEDED}
    assert len({result.run.run_id for result in results}) == 2
    with Catalog(workspace).open_snapshot() as snapshot:
        assert snapshot.execute("SELECT count(DISTINCT run_id) FROM runs").fetchone() == (2,)


@pytest.mark.anyio
@pytest.mark.process
async def test_capture_admission_limit_is_shared_by_independent_services(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.wait]
argv = ["python", "-c", "import time; time.sleep(0.75)"]
cwd = "."
timeout_seconds = 30
"""
    )
    config = workspace.config.validated_copy(
        update={
            "capture": workspace.config.capture.validated_copy(update={"max_parallel_captures": 1}),
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            ),
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    first_service = CaptureService(workspace)
    second_service = CaptureService(workspace)
    first_plan = await first_service.plan(
        workload_name="wait",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    second_plan = await second_service.plan(
        workload_name="wait",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    first_admitted = asyncio.Event()
    active = 0
    maximum_active = 0

    async def observe(completed: float, total: float, message: str) -> None:
        nonlocal active, maximum_active
        if completed == 4:
            active += 1
            maximum_active = max(maximum_active, active)
            first_admitted.set()
        elif completed == 5:
            active -= 1

    first_task = asyncio.create_task(first_service.execute(first_plan.plan_token, progress=observe))
    await asyncio.wait_for(first_admitted.wait(), timeout=5)
    second_task = asyncio.create_task(
        second_service.execute(second_plan.plan_token, progress=observe)
    )
    queued = None
    for _ in range(200):
        try:
            queued = RunStore(workspace).read(second_plan.run_id)
        except DomainError:
            await asyncio.sleep(0.01)
            continue
        if queued.revision >= 1:
            break
        await asyncio.sleep(0.01)

    assert queued is not None
    assert queued.execution_status is ExecutionStatus.PLANNED
    assert queued.capture_status is CaptureStatus.PENDING
    results = await asyncio.gather(first_task, second_task)

    assert maximum_active == 1
    assert active == 0
    assert all(result.run.execution_status is ExecutionStatus.SUCCEEDED for result in results)


@pytest.mark.anyio
async def test_capture_admission_reclaims_only_a_proven_dead_exact_owner(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    observed = utc_now()
    dead = CaptureAdmissionRecord(
        run_id="dead-run",
        owner_id="dead-owner",
        process_lease=CaptureLease(
            process_id=2_000_000_000,
            process_start_identity="dead-process-start",
            boot_id=read_boot_id(),
            heartbeat_monotonic_ns=time.monotonic_ns(),
            observed_at=observed,
            expires_at=observed + timedelta(seconds=60),
        ),
        acquired_at=observed,
    )
    store = CaptureAdmissionStore(workspace)
    assert store.try_acquire(dead, limit=1)

    admission = await CaptureAdmissionService(workspace, limit=1).acquire("replacement-run")
    try:
        assert [record.run_id for record in store.list()] == ["replacement-run"]
    finally:
        admission.release()


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
argv = ["python", "-c", "import os,time;os.write(1,bytes([255])+b'x');time.sleep(10)"]
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
        sensitivity=Sensitivity.SENSITIVE,
    )
    plan = await service.plan(
        workload_name="wait",
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        external_context=context,
    )
    collector_started = asyncio.Event()

    async def record_progress(completed: float, _total: float, _message: str) -> None:
        if completed == 4:
            collector_started.set()

    task = asyncio.create_task(service.execute(plan.plan_token, progress=record_progress))
    await asyncio.wait_for(collector_started.wait(), timeout=5)
    await asyncio.sleep(0.2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    runs = [run.run_id for run in RunStore(workspace).list()]
    assert len(runs) == 1
    terminal = RunStore(workspace).read(runs[0])
    assert terminal.execution_status is ExecutionStatus.CANCELLED
    assert terminal.capture_status is CaptureStatus.CANCELLED
    assert terminal.finished_at is not None
    assert terminal.external_context == context
    stdout = next(item for item in terminal.artifacts if item.role == "stdout")
    assert ArtifactStore(workspace).get(stdout.artifact_id).payload_path.read_bytes() == (b"\xffx")
    with Catalog(workspace).open_snapshot() as snapshot:
        registered = snapshot.execute(
            "SELECT artifact_id FROM artifact_registrations "
            "WHERE run_id = ? AND kind = 'process_output' AND role = 'stdout'",
            (terminal.run_id,),
        ).fetchall()
    assert registered == [(stdout.artifact_id,)]
    projection = RunProjectionService(workspace).get(terminal.run_id)
    assert [action.tool_name for action in projection.recovery_actions] == ["preview_artifact"]
    assert projection.recovery_actions[0].arguments == {
        "artifact_id": stdout.artifact_id,
        "offset": 0,
        "max_bytes": 4_096,
        "max_lines": 80,
    }


@pytest.mark.anyio
async def test_startup_identity_failure_is_terminal_and_cleans_staging(
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

    async def fail_source_identity(*_args: object, **_kwargs: object) -> None:
        raise DomainError(ErrorCode.INTERNAL_ERROR, "simulated source identity failure")

    monkeypatch.setattr(
        "flameox.application.capture.collect_source_state",
        fail_source_identity,
    )

    with pytest.raises(DomainError, match="simulated source identity failure"):
        await service.execute(plan.plan_token)

    terminal = RunStore(workspace).read(plan.run_id)
    assert terminal.execution_status is ExecutionStatus.FAILED
    assert terminal.capture_status is CaptureStatus.FAILED
    assert terminal.finished_at is not None
    assert terminal.process is not None
    assert terminal.process.cancellation_cause == "process_error"
    assert not (workspace.paths.staging / "captures" / plan.run_id).exists()


@pytest.mark.anyio
async def test_startup_identity_collection_records_recoverable_owner_lease(
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
    collecting = asyncio.Event()

    async def wait_during_source_identity(*_args: object, **_kwargs: object) -> None:
        collecting.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "flameox.application.capture.collect_source_state",
        wait_during_source_identity,
    )
    task = asyncio.create_task(service.execute(plan.plan_token))
    await asyncio.wait_for(collecting.wait(), timeout=2)

    starting = RunStore(workspace).read(plan.run_id)

    assert starting.execution_status is ExecutionStatus.PLANNED
    assert starting.capture_status is CaptureStatus.PENDING
    assert starting.lease is not None
    assert starting.lease.process_id == os.getpid()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_startup_lease_failure_terminalizes_consumed_capture_plan(
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

    def fail_lease(_process_id: int) -> object:
        raise DomainError(ErrorCode.PROCESS_FAILED, "simulated lease identity failure")

    monkeypatch.setattr(service, "_lease", fail_lease)

    with pytest.raises(DomainError, match="simulated lease identity failure") as error:
        await service.execute(plan.plan_token)

    terminal = RunStore(workspace).read(plan.run_id)
    assert error.value.run_id == plan.run_id
    assert terminal.execution_status is ExecutionStatus.FAILED
    assert terminal.capture_status is CaptureStatus.FAILED
    assert terminal.finished_at is not None


@pytest.mark.anyio
@pytest.mark.process
async def test_artifact_registration_failure_is_terminal_and_cleans_staging(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    write_workload(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "capture": workspace.config.capture.validated_copy(update={"max_artifact_bytes": 1}),
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            ),
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
        await service.execute(plan.plan_token)

    terminal = RunStore(workspace).read(plan.run_id)
    assert error.value.code is ErrorCode.ARTIFACT_TOO_LARGE
    assert terminal.execution_status is ExecutionStatus.FAILED
    assert terminal.capture_status is CaptureStatus.FAILED
    assert not any((workspace.paths.staging / "captures").iterdir())


@pytest.mark.anyio
@pytest.mark.process
async def test_publication_failure_preserves_succeeded_run_and_cleans_staging(
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
    original = cast(Any, service.projections.publisher.publish_rows_idempotent)

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

    monkeypatch.setattr(
        service.projections.publisher,
        "publish_rows_idempotent",
        fail_evidence_publication,
    )

    with pytest.raises(DomainError, match="simulated publication failure"):
        await service.execute(plan.plan_token)

    terminal = RunStore(workspace).read(plan.run_id)
    assert terminal.execution_status is ExecutionStatus.SUCCEEDED
    assert terminal.capture_status is CaptureStatus.REGISTERED
    intent = ProjectionIntentStore(workspace).latest(
        domain_kind="run",
        domain_id=plan.run_id,
        projection_kind="run.core",
    )
    assert intent is not None
    assert intent.state is ProjectionState.FAILED
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
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        await service.execute(plan.plan_token, progress=cancel)

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
    task = asyncio.create_task(service.execute(plan.plan_token))
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
