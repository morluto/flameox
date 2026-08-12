from __future__ import annotations

from flameox.adapters import (
    CoverageExtractor,
    MemrayExtractor,
    ObservationExtractor,
    PerfettoExtractor,
    PyPerfExtractor,
)
from flameox.application.capabilities import CapabilityService
from flameox.application.integrity import IntegrityLevel, IntegrityService
from flameox.application.quarantine import QuarantineService
from flameox.application.recovery import RecoveryService
from flameox.catalog import Catalog
from flameox.domain import CapabilityStatus, DomainError, ExecutionStatus
from flameox.models import ContractModel
from flameox.storage import RunStore, Workspace, tree_bytes


class WorkspaceStatus(ContractModel):
    schema_version: int = 1
    workspace_id: str
    project_root: str
    corpus_commit_id: str
    catalog_exists: bool
    workspace_valid: bool
    catalog_valid: bool
    catalog_fresh: bool | None
    last_catalog_rebuild_at: str | None
    run_count: int
    artifact_object_count: int
    storage_bytes: int
    storage_by_artifact_kind: dict[str, int]
    active_captures: int
    stale_run_ids: tuple[str, ...]
    quarantined_run_ids: tuple[str, ...]
    extractor_versions: dict[str, str]
    capability_warnings: tuple[str, ...]
    warnings: tuple[str, ...]


def workspace_status(workspace: Workspace) -> WorkspaceStatus:
    catalog_exists = workspace.paths.catalog.is_file()
    catalog_fresh: bool | None = None
    catalog_valid = False
    last_rebuild: str | None = None
    storage_by_kind: dict[str, int] = {}
    if catalog_exists:
        try:
            catalog_status = Catalog(workspace).status()
            catalog_fresh = bool(catalog_status["fresh"])
            catalog_valid = True
            last_rebuild = str(catalog_status["built_at"])
            catalog = Catalog(workspace)
            with catalog.open_snapshot(catalog.pin()) as snapshot:
                rows = snapshot.execute(
                    "WITH unique_objects AS ("
                    "SELECT kind, artifact_id, max(byte_length) AS byte_length "
                    "FROM artifact_registrations GROUP BY kind, artifact_id"
                    ") SELECT kind, sum(byte_length) FROM unique_objects "
                    "GROUP BY kind ORDER BY kind"
                ).fetchall()
            storage_by_kind = {str(row[0]): int(row[1]) for row in rows}
        except DomainError:
            catalog_valid = False
    runs = RunStore(workspace).list()
    active = 0
    warnings: list[str] = []
    for run in runs:
        if run.execution_status is ExecutionStatus.RUNNING:
            active += 1
    storage_bytes = tree_bytes(workspace.paths.root)
    if catalog_fresh is False:
        warnings.append("The rebuildable catalog is stale.")
    integrity = IntegrityService(workspace).validate(IntegrityLevel.QUICK)
    warnings.extend(issue.message for issue in integrity.issues)
    recovery = RecoveryService(workspace).inspect()
    quarantined_run_ids = tuple(
        sorted(
            {
                manifest.originating_run_id
                for manifest in QuarantineService(workspace).list_manifests()
                if manifest.state == "quarantined" and manifest.originating_run_id is not None
            }
        )
    )
    capabilities = CapabilityService(workspace).list().capabilities
    capability_warnings = tuple(
        f"{item.adapter}: " + "; ".join((*item.limitations, *item.remediation)[:2])
        for item in capabilities
        if item.status is not CapabilityStatus.AVAILABLE or item.limitations or item.remediation
    )
    return WorkspaceStatus(
        workspace_id=workspace.identity.workspace_id,
        project_root=str(workspace.project_root),
        corpus_commit_id=workspace.corpus.read_head().commit_id,
        catalog_exists=catalog_exists,
        workspace_valid=integrity.valid,
        catalog_valid=catalog_valid,
        catalog_fresh=catalog_fresh,
        last_catalog_rebuild_at=last_rebuild,
        run_count=len(runs),
        artifact_object_count=sum(1 for _ in workspace.paths.artifacts.glob("*/*/artifact.json")),
        storage_bytes=storage_bytes,
        storage_by_artifact_kind=storage_by_kind,
        active_captures=active,
        stale_run_ids=tuple(
            sorted(
                {
                    *recovery.recoverable_run_ids,
                    *recovery.indeterminate_run_ids,
                }
            )
        ),
        quarantined_run_ids=quarantined_run_ids,
        extractor_versions={
            extractor.name: extractor.version
            for extractor in (
                PyPerfExtractor,
                PerfettoExtractor,
                MemrayExtractor,
                CoverageExtractor,
                ObservationExtractor,
            )
        },
        capability_warnings=capability_warnings,
        warnings=tuple(warnings),
    )
