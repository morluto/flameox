from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from flameox.application.quarantine import QuarantineManifest, QuarantineService
from flameox.domain import DomainError, ErrorCode, RunManifest, digest_model
from flameox.models import ContractModel
from flameox.storage import Workspace
from flameox.storage.atomic import atomic_write_bytes

logger = logging.getLogger(__name__)


class RepairEntry(ContractModel):
    path: str
    action: Literal["rebuild_run_projection"]
    reason: str
    recovery_source: str


class RepairPlan(ContractModel):
    schema_version: int = 1
    plan_id: str
    corpus_commit_id: str
    entries: tuple[RepairEntry, ...]
    unresolved_paths: tuple[str, ...]


class RepairResult(ContractModel):
    schema_version: int = 1
    plan_id: str
    repaired_paths: tuple[str, ...]
    quarantine: tuple[QuarantineManifest, ...]


class RepairService:
    """Preview and apply only repairs with an immutable, validated recovery source."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.quarantine = QuarantineService(workspace)

    def plan(self) -> RepairPlan:
        head = self.workspace.corpus.read_head()
        entries: list[RepairEntry] = []
        unresolved: list[str] = []
        for projection in sorted(self.workspace.paths.runs.glob("*/manifest.json")):
            try:
                RunManifest.model_validate_json(projection.read_text())
                continue
            except (OSError, ValueError) as exc:
                logger.warning("run projection %s is unreadable: %s", projection, exc)
            recovery_source = self._latest_valid_revision(projection.parent)
            relative = projection.relative_to(self.workspace.paths.root).as_posix()
            if recovery_source is None:
                unresolved.append(relative)
                continue
            entries.append(
                RepairEntry(
                    path=relative,
                    action="rebuild_run_projection",
                    reason="The mutable run projection is invalid.",
                    recovery_source=(
                        recovery_source.relative_to(self.workspace.paths.root).as_posix()
                    ),
                )
            )
        content = {
            "corpus_commit_id": head.commit_id,
            "entries": [entry.model_dump(mode="json") for entry in entries],
            "unresolved_paths": unresolved,
        }
        return RepairPlan(
            plan_id=digest_model(content),
            corpus_commit_id=head.commit_id,
            entries=tuple(entries),
            unresolved_paths=tuple(unresolved),
        )

    def apply(self, plan: RepairPlan) -> RepairResult:
        quarantined: list[QuarantineManifest] = []
        repaired: list[str] = []
        with (
            self.workspace.write_locked(),
            self.workspace.retention_locked(shared=False),
        ):
            current = self.plan()
            if current != plan:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "Repair inputs changed after planning.",
                    retryable=True,
                )
            if self.workspace.corpus.read_head().commit_id != plan.corpus_commit_id:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "Corpus HEAD changed before repair.",
                    retryable=True,
                )
            for entry in plan.entries:
                source = self._resolve(entry.path)
                recovery_source = self._resolve(entry.recovery_source)
                recovered = RunManifest.model_validate_json(recovery_source.read_text())
                quarantined.append(
                    self.quarantine.quarantine_locked(
                        source,
                        reason=entry.reason,
                        operation=f"repair:{plan.plan_id}",
                        expected_format="RunManifest JSON",
                        actual_format="invalid JSON or schema",
                        originating_run_id=recovered.run_id,
                    )
                )
                atomic_write_bytes(source, recovery_source.read_bytes())
                repaired.append(entry.path)
        return RepairResult(
            plan_id=plan.plan_id,
            repaired_paths=tuple(repaired),
            quarantine=tuple(quarantined),
        )

    def _resolve(self, relative: str) -> Path:
        path = (self.workspace.paths.root / relative).resolve()
        try:
            path.relative_to(self.workspace.paths.root)
        except ValueError as exc:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Repair plan contains a path outside the workspace.",
            ) from exc
        return path

    @staticmethod
    def _latest_valid_revision(run_root: Path) -> Path | None:
        for path in sorted((run_root / "revisions").glob("*.json"), reverse=True):
            try:
                RunManifest.model_validate_json(path.read_text())
            except (OSError, ValueError):
                continue
            return path
        return None
