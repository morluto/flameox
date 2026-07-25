from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from flamo.application.run_rows import run_row
from flamo.domain import (
    CaptureStatus,
    ExecutionStatus,
    ProcessResult,
    RunManifest,
)
from flamo.domain.models import utc_now
from flamo.evidence import GenerationPublisher
from flamo.storage import RunStore, Workspace


class RecoveryInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    active_run_ids: tuple[str, ...]
    recoverable_run_ids: tuple[str, ...]
    indeterminate_run_ids: tuple[str, ...]
    staging_paths: tuple[str, ...]


class RecoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    recovered_runs: tuple[RunManifest, ...]
    inspection: RecoveryInspection


class RecoveryService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)
        self.publisher = GenerationPublisher(workspace)

    def inspect(self) -> RecoveryInspection:
        active: list[str] = []
        recoverable: list[str] = []
        indeterminate: list[str] = []
        for projection in sorted(self.workspace.paths.runs.glob("*/manifest.json")):
            run = self.runs.read(projection.parent.name)
            if run.execution_status is not ExecutionStatus.RUNNING:
                continue
            if run.lease is None:
                indeterminate.append(run.run_id)
            elif self._lease_is_live(run):
                active.append(run.run_id)
            else:
                recoverable.append(run.run_id)
        staging_paths = tuple(
            path.relative_to(self.workspace.paths.root).as_posix()
            for path in sorted(self.workspace.paths.staging.rglob("*"))
            if path.is_dir() and path != self.workspace.paths.staging / "captures"
        )
        return RecoveryInspection(
            active_run_ids=tuple(active),
            recoverable_run_ids=tuple(recoverable),
            indeterminate_run_ids=tuple(indeterminate),
            staging_paths=staging_paths,
        )

    def recover(self) -> RecoveryResult:
        inspection = self.inspect()
        recovered: list[RunManifest] = []
        for run_id in inspection.recoverable_run_ids:
            current = self.runs.read(run_id)
            terminal = current.model_copy(
                update={
                    "revision": current.revision + 1,
                    "finished_at": utc_now(),
                    "execution_status": ExecutionStatus.CANCELLED,
                    "capture_status": CaptureStatus.CANCELLED,
                    "process": ProcessResult(
                        cancellation_cause="crash_recovery",
                        cleanup_complete=None,
                    ),
                    "limitations": (
                        *current.limitations,
                        "Recovered after the exact leased process identity disappeared.",
                    ),
                }
            )
            recovered.append(self.runs.append(terminal, expected_revision=current.revision))
        if recovered:
            self.publisher.publish_rows(
                {"runs": [run_row(run) for run in recovered]},
                publisher="flamo.recovery",
                publisher_version="1",
                input_run_ids=tuple(run.run_id for run in recovered),
            )
        return RecoveryResult(
            recovered_runs=tuple(recovered),
            inspection=self.inspect(),
        )

    def _lease_is_live(self, run: RunManifest) -> bool:
        assert run.lease is not None
        try:
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            fields = Path("/proc").joinpath(str(run.lease.process_id), "stat").read_text().split()
            start_identity = fields[21]
        except (OSError, IndexError):
            return False
        return boot_id == run.lease.boot_id and start_identity == run.lease.process_start_identity
