from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import flameox.application.proc as proc
from flameox.analysis import RecipeService
from flameox.application import RecoveryService
from flameox.catalog import Catalog
from flameox.domain import (
    CaptureLease,
    CaptureStatus,
    ExecutionStatus,
    RunManifest,
    ValidationStatus,
)
from flameox.domain.models import ExecutionRunManifest, utc_now
from flameox.storage import RunStore, Workspace

DIGEST = "sha256:" + ("a" * 64)


def _running_run(*, run_id: str, boot_id: str, start_identity: str) -> RunManifest:
    observed = utc_now()
    return ExecutionRunManifest(
        run_id=run_id,
        started_at=observed,
        execution_status=ExecutionStatus.RUNNING,
        capture_status=CaptureStatus.RUNNING,
        validation_status=ValidationStatus.PENDING,
        environment_id=DIGEST,
        lease=CaptureLease(
            process_id=12345,
            process_start_identity=start_identity,
            boot_id=boot_id,
            heartbeat_monotonic_ns=0,
            observed_at=observed,
            expires_at=observed + timedelta(minutes=1),
        ),
    )


def _stat_line(pid: int, comm: str, starttime: int) -> str:
    fields_before_starttime = " ".join("0" for _ in range(19))
    return f"{pid} ({comm}) {fields_before_starttime} {starttime}\n"


def _write_proc_identity(root: Path, *, boot_id: str, stat_text: str | None) -> Path:
    boot_id_path = root / "sys/kernel/random/boot_id"
    boot_id_path.parent.mkdir(parents=True)
    boot_id_path.write_text(f"{boot_id}\n")
    stat_path = root / "12345" / "stat"
    if stat_text is not None:
        stat_path.parent.mkdir(parents=True)
        stat_path.write_text(stat_text)
    return stat_path


def test_recovery_closes_only_disappeared_exact_process_lease(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    observed = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    run = ExecutionRunManifest(
        run_id="abandoned-run",
        started_at=observed,
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
        execution_status=(ExecutionStatus.CANCELLED,),
    )
    excluded = RecipeService(workspace).failures(
        environment_id="sha256:" + ("b" * 64),
    )
    assert selected.total_clusters == 1
    assert selected.filters_applied == ("environment_id", "execution_status")
    assert selected.cohort_id != population.cohort_id
    assert excluded.total_clusters == 0


def test_recovery_skips_a_run_that_changes_after_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    run = _running_run(
        run_id="raced-run",
        boot_id="wrong-boot",
        start_identity="never-existed",
    )
    RunStore(workspace).create(run)
    service = RecoveryService(workspace)
    append = service.runs.append
    raced = False

    def append_after_concurrent_update(
        manifest: RunManifest,
        *,
        expected_revision: int,
    ) -> RunManifest:
        nonlocal raced
        if not raced:
            raced = True
            current = service.runs.read(manifest.run_id)
            append(
                current.model_copy(update={"revision": current.revision + 1}),
                expected_revision=current.revision,
            )
        return append(manifest, expected_revision=expected_revision)

    monkeypatch.setattr(service.runs, "append", append_after_concurrent_update)

    result = service.recover()

    assert result.recovered_runs == ()
    assert result.inspection.recoverable_run_ids == ("raced-run",)
    assert service.runs.read("raced-run").revision == 1


def test_recovery_keeps_live_lease_when_process_name_contains_spaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    RunStore(workspace).create(
        _running_run(run_id="live-run", boot_id="boot-id", start_identity="118")
    )
    proc_root = tmp_path / "proc"
    _write_proc_identity(
        proc_root,
        boot_id="boot-id",
        stat_text=_stat_line(12345, "Web Content", 118),
    )
    monkeypatch.setattr(proc, "PROC_ROOT", proc_root)

    inspection = RecoveryService(workspace).inspect()

    assert inspection.active_run_ids == ("live-run",)
    assert inspection.recoverable_run_ids == ()
    assert inspection.indeterminate_run_ids == ()


def test_recovery_does_not_equate_unreadable_proc_state_with_process_death(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    RunStore(workspace).create(
        _running_run(run_id="unknown-run", boot_id="boot-id", start_identity="118")
    )
    proc_root = tmp_path / "proc"
    stat_path = _write_proc_identity(proc_root, boot_id="boot-id", stat_text=None)
    stat_path.mkdir(parents=True)
    monkeypatch.setattr(proc, "PROC_ROOT", proc_root)

    inspection = RecoveryService(workspace).inspect()

    assert inspection.active_run_ids == ()
    assert inspection.recoverable_run_ids == ()
    assert inspection.indeterminate_run_ids == ("unknown-run",)


def test_recovery_treats_changed_boot_id_as_conclusive_before_reading_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    RunStore(workspace).create(
        _running_run(run_id="pre-reboot-run", boot_id="old-boot-id", start_identity="118")
    )
    proc_root = tmp_path / "proc"
    _write_proc_identity(proc_root, boot_id="new-boot-id", stat_text=None)
    monkeypatch.setattr(proc, "PROC_ROOT", proc_root)
    monkeypatch.setattr(
        "flameox.application.recovery.read_proc_stat_start_identity",
        lambda _process_id: pytest.fail(
            "a changed boot identity makes the process read unnecessary"
        ),
    )

    inspection = RecoveryService(workspace).inspect()

    assert inspection.recoverable_run_ids == ("pre-reboot-run",)
    assert inspection.indeterminate_run_ids == ()


def test_recovery_reconciles_planned_capture_from_exact_startup_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    startup = _running_run(
        run_id="starting-run",
        boot_id="boot-id",
        start_identity="118",
    ).model_copy(
        update={
            "started_at": None,
            "execution_status": ExecutionStatus.PLANNED,
            "capture_status": CaptureStatus.PENDING,
        }
    )
    RunStore(workspace).create(startup)
    proc_root = tmp_path / "proc"
    stat_path = _write_proc_identity(
        proc_root,
        boot_id="boot-id",
        stat_text=_stat_line(12345, "flameox owner", 118),
    )
    monkeypatch.setattr(proc, "PROC_ROOT", proc_root)

    live = RecoveryService(workspace).inspect()
    assert live.active_run_ids == ("starting-run",)

    stat_path.unlink()

    vanished = RecoveryService(workspace).inspect()
    assert vanished.recoverable_run_ids == ("starting-run",)
