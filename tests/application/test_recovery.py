from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from flameox.analysis import RecipeService
from flameox.application import RecoveryService
from flameox.catalog import Catalog
from flameox.domain import (
    CaptureLease,
    CaptureStatus,
    ExecutionStatus,
    RunManifest,
    RunType,
    ValidationStatus,
)
from flameox.storage import RunStore, Workspace

DIGEST = "sha256:" + ("a" * 64)


def test_recovery_closes_only_disappeared_exact_process_lease(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    observed = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    run = RunManifest(
        run_id="abandoned-run",
        run_type=RunType.EXECUTION,
        execution_status=ExecutionStatus.RUNNING,
        capture_status=CaptureStatus.RUNNING,
        validation_status=ValidationStatus.PENDING,
        environment_id=DIGEST,
        lease=CaptureLease(
            process_id=2_147_483_647,
            process_start_identity="never-existed",
            boot_id="wrong-boot",
            heartbeat_monotonic_ns=0,
            observed_at=observed,
            expires_at=observed + timedelta(minutes=1),
        ),
    )
    RunStore(workspace).create(run)

    before = RecoveryService(workspace).inspect()
    result = RecoveryService(workspace).recover()

    assert before.recoverable_run_ids == ("abandoned-run",)
    assert result.recovered_runs[0].execution_status is ExecutionStatus.CANCELLED
    assert result.recovered_runs[0].process is not None
    assert result.recovered_runs[0].process.cancellation_cause == "crash_recovery"
    assert result.inspection.recoverable_run_ids == ()
    with Catalog(workspace).open_snapshot() as snapshot:
        normalized = snapshot.execute(
            "SELECT execution_status FROM runs WHERE run_id = ? ORDER BY published_at DESC LIMIT 1",
            ("abandoned-run",),
        ).fetchone()
    assert normalized is not None
    assert normalized[0] == "cancelled"
    population = RecipeService(workspace).failures()
    assert population.total_clusters == 1
    assert population.failures[0].run_count == 1
    selected = RecipeService(workspace).failures(
        environment_id=DIGEST,
        execution_status=("cancelled",),
    )
    excluded = RecipeService(workspace).failures(
        environment_id="sha256:" + ("b" * 64),
    )
    assert selected.total_clusters == 1
    assert selected.filters_applied == ("environment_id", "execution_status")
    assert selected.cohort_id != population.cohort_id
    assert excluded.total_clusters == 0
