from __future__ import annotations

from enum import StrEnum

from flameox.application.gc import GarbageCollector
from flameox.application.proc import read_boot_id, read_proc_stat_start_identity
from flameox.application.quarantine import QuarantineService
from flameox.application.run_rows import run_row
from flameox.domain import (
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ProcessCancellationCause,
    ProcessResult,
    RunManifest,
)
from flameox.domain.models import utc_now
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.storage import RunStore, Workspace


class LeaseState(StrEnum):
    ACTIVE = "active"
    RECOVERABLE = "recoverable"
    INDETERMINATE = "indeterminate"


class RecoveryInspection(ContractModel):
    schema_version: int = 1
    active_run_ids: tuple[str, ...]
    recoverable_run_ids: tuple[str, ...]
    indeterminate_run_ids: tuple[str, ...]
    staging_paths: tuple[str, ...]


class RecoveryResult(ContractModel):
    schema_version: int = 1
    recovered_runs: tuple[RunManifest, ...]
    resumed_trash_manifests: tuple[str, ...] = ()
    resumed_quarantine_manifests: tuple[str, ...] = ()
    inspection: RecoveryInspection


class RecoveryService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)
        self.publisher = GenerationPublisher(workspace)
        self.garbage = GarbageCollector(workspace)
        self.quarantine = QuarantineService(workspace)

    def inspect(self) -> RecoveryInspection:
        active: list[str] = []
        recoverable: list[str] = []
        indeterminate: list[str] = []
        for run in self.runs.list():
            recoverable_lifecycle = run.execution_status is ExecutionStatus.RUNNING or (
                run.execution_status is ExecutionStatus.PLANNED
                and run.capture_status is CaptureStatus.PENDING
            )
            if not recoverable_lifecycle:
                continue
            lease_state = self._lease_state(run)
            if lease_state is LeaseState.ACTIVE:
                active.append(run.run_id)
            elif lease_state is LeaseState.RECOVERABLE:
                recoverable.append(run.run_id)
            else:
                indeterminate.append(run.run_id)
        staging_paths = tuple(
            path.relative_to(self.workspace.paths.root).as_posix()
            for path in sorted(self.workspace.paths.staging.rglob("*"))
            if path.is_dir() and path != self.workspace.paths.staging / "captures"
        )
        return RecoveryInspection(
            active_run_ids=tuple(sorted(active)),
            recoverable_run_ids=tuple(sorted(recoverable)),
            indeterminate_run_ids=tuple(sorted(indeterminate)),
            staging_paths=staging_paths,
        )

    def recover(self) -> RecoveryResult:
        resumed_trash = self.garbage.moving_manifests()
        for manifest_id in resumed_trash:
            self.garbage.resume(manifest_id)
        resumed_quarantine = self.quarantine.moving_manifests()
        for quarantine_id in resumed_quarantine:
            self.quarantine.resume(quarantine_id)
        inspection = self.inspect()
        recovered: list[RunManifest] = []
        for run_id in inspection.recoverable_run_ids:
            current = self.runs.read(run_id)
            terminal = current.validated_copy(
                update={
                    "revision": current.revision + 1,
                    "finished_at": utc_now(),
                    "execution_status": ExecutionStatus.CANCELLED,
                    "capture_status": CaptureStatus.CANCELLED,
                    "process": ProcessResult(
                        cancellation_cause=ProcessCancellationCause.CRASH_RECOVERY,
                        cleanup_complete=None,
                    ),
                    "limitations": (
                        *current.limitations,
                        "Recovered after the exact leased process identity disappeared.",
                    ),
                }
            )
            try:
                recovered.append(self.runs.append(terminal, expected_revision=current.revision))
            except DomainError as error:
                if error.code is not ErrorCode.REVISION_CONFLICT:
                    raise
        if recovered:
            self.publisher.publish_rows(
                {"runs": [run_row(run) for run in recovered]},
                publisher="flameox.recovery",
                publisher_version="1",
                input_run_ids=tuple(run.run_id for run in recovered),
            )
        return RecoveryResult(
            recovered_runs=tuple(recovered),
            resumed_trash_manifests=resumed_trash,
            resumed_quarantine_manifests=resumed_quarantine,
            inspection=self.inspect(),
        )

    def _lease_state(self, run: RunManifest) -> LeaseState:
        if run.lease is None:
            return LeaseState.INDETERMINATE
        try:
            boot_id = read_boot_id()
        except (OSError, ValueError):
            return LeaseState.INDETERMINATE
        if boot_id != run.lease.boot_id:
            return LeaseState.RECOVERABLE
        try:
            start_identity = read_proc_stat_start_identity(run.lease.process_id)
        except FileNotFoundError:
            return LeaseState.RECOVERABLE
        except (OSError, ValueError):
            return LeaseState.INDETERMINATE
        if start_identity == run.lease.process_start_identity:
            return LeaseState.ACTIVE
        return LeaseState.RECOVERABLE
