from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import flameox.application.proc as proc
from flameox.application import RecoveryService
from flameox.domain import (
    CaptureLease,
    CaptureStatus,
    ExecutionRunManifest,
    ExecutionStatus,
    RunManifest,
    ValidationStatus,
)
from flameox.storage import RunStore, Workspace

DIGEST = "sha256:" + ("a" * 64)
OBSERVED_AT = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)


def _stat_line(pid: int, comm: str, starttime: str) -> str:
    fields_3_through_21 = " ".join("0" for _ in range(19))
    return f"{pid} ({comm}) {fields_3_through_21} {starttime} 0\n"


def _write_proc_record(root: Path, pid: int, comm: str, starttime: str) -> None:
    stat_path = root / str(pid) / "stat"
    stat_path.parent.mkdir(parents=True)
    stat_path.write_text(_stat_line(pid, comm, starttime))


def _running_run(run_id: str, pid: int, starttime: str, boot_id: str) -> RunManifest:
    return ExecutionRunManifest(
        run_id=run_id,
        execution_status=ExecutionStatus.RUNNING,
        capture_status=CaptureStatus.RUNNING,
        validation_status=ValidationStatus.PENDING,
        environment_id=DIGEST,
        lease=CaptureLease(
            process_id=pid,
            process_start_identity=starttime,
            boot_id=boot_id,
            heartbeat_monotonic_ns=0,
            observed_at=OBSERVED_AT,
            expires_at=OBSERVED_AT + timedelta(minutes=1),
        ),
    )


@pytest.mark.parametrize(
    ("comm", "starttime"),
    [
        pytest.param("python", "118", id="single-word-comm"),
        pytest.param("Web Content", "999", id="spaced-comm"),
        pytest.param("worker) thread", "555", id="closing-paren-comm"),
    ],
)
def test_proc_stat_reader_returns_field_22(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    comm: str,
    starttime: str,
) -> None:
    monkeypatch.setattr(proc, "PROC_ROOT", tmp_path)
    _write_proc_record(tmp_path, 12345, comm, starttime)

    assert proc.read_proc_stat_start_identity(12345) == starttime


@pytest.mark.parametrize(
    "stat_text",
    [
        pytest.param("12345 python 0 0", id="missing-comm-delimiters"),
        pytest.param("12345 (python) 0 0", id="missing-starttime"),
    ],
)
def test_proc_stat_parser_rejects_incomplete_records(stat_text: str) -> None:
    with pytest.raises(ValueError, match=r"comm field|starttime field"):
        proc.parse_proc_stat_start_identity(stat_text)


def test_proc_stat_reader_rejects_oversized_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proc, "PROC_ROOT", tmp_path)
    stat_path = tmp_path / "12345" / "stat"
    stat_path.parent.mkdir()
    stat_path.write_bytes(b"x" * (proc.MAX_STAT_BYTES + 1))

    with pytest.raises(ValueError, match="exceeds 8192 bytes"):
        proc.read_proc_stat_start_identity(12345)


def test_recovery_classifies_exact_process_identity_with_spaced_comm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    proc_root = tmp_path / "proc"
    boot_id_path = proc_root / "sys/kernel/random/boot_id"
    boot_id_path.parent.mkdir(parents=True)
    boot_id_path.write_text("boot-id\n")
    _write_proc_record(proc_root, 12345, "Web Content", "118")
    monkeypatch.setattr(proc, "PROC_ROOT", proc_root)

    runs = RunStore(workspace)
    runs.create(_running_run("active", 12345, "118", "boot-id"))
    runs.create(_running_run("reused-pid", 12345, "117", "boot-id"))
    runs.create(_running_run("previous-boot", 12345, "118", "old-boot-id"))
    runs.create(_running_run("missing-process", 54321, "118", "boot-id"))

    inspection = RecoveryService(workspace).inspect()

    assert inspection.active_run_ids == ("active",)
    assert inspection.recoverable_run_ids == (
        "missing-process",
        "previous-boot",
        "reused-pid",
    )
    assert inspection.indeterminate_run_ids == ()


def test_recovery_keeps_unreadable_process_identities_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path / "workspace")
    proc_root = tmp_path / "proc"
    boot_id_path = proc_root / "sys/kernel/random/boot_id"
    boot_id_path.parent.mkdir(parents=True)
    boot_id_path.write_text("boot-id\n")
    malformed_stat = proc_root / "12345" / "stat"
    malformed_stat.parent.mkdir(parents=True)
    malformed_stat.write_text("12345 (malformed) 0 0\n")
    unreadable_stat = proc_root / "23456" / "stat"
    unreadable_stat.mkdir(parents=True)
    monkeypatch.setattr(proc, "PROC_ROOT", proc_root)

    runs = RunStore(workspace)
    runs.create(_running_run("malformed-stat", 12345, "118", "boot-id"))
    runs.create(_running_run("unreadable-stat", 23456, "118", "boot-id"))
    service = RecoveryService(workspace)

    inspection = service.inspect()
    result = service.recover()

    assert inspection.active_run_ids == ()
    assert inspection.recoverable_run_ids == ()
    assert inspection.indeterminate_run_ids == ("malformed-stat", "unreadable-stat")
    assert result.recovered_runs == ()
    assert result.inspection.indeterminate_run_ids == ("malformed-stat", "unreadable-stat")
    assert runs.read("malformed-stat").execution_status is ExecutionStatus.RUNNING
    assert runs.read("unreadable-stat").execution_status is ExecutionStatus.RUNNING
