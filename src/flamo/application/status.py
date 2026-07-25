from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from flamo.catalog import Catalog
from flamo.domain import DomainError, ExecutionStatus
from flamo.storage import RunStore, Workspace


class WorkspaceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    workspace_id: str
    project_root: str
    corpus_commit_id: str
    catalog_exists: bool
    catalog_fresh: bool | None
    run_count: int
    artifact_object_count: int
    storage_bytes: int
    active_captures: int
    warnings: tuple[str, ...]


def workspace_status(workspace: Workspace) -> WorkspaceStatus:
    catalog_exists = workspace.paths.catalog.is_file()
    catalog_fresh: bool | None = None
    if catalog_exists:
        catalog_fresh = bool(Catalog(workspace).status()["fresh"])
    run_directories = [path for path in workspace.paths.runs.iterdir() if path.is_dir()]
    active = 0
    warnings: list[str] = []
    for path in run_directories:
        try:
            run = RunStore(workspace).read(path.name)
        except DomainError:
            warnings.append(f"Run projection is unreadable: {path.name}")
            continue
        if run.execution_status is ExecutionStatus.RUNNING:
            active += 1
    storage_bytes = sum(
        path.stat().st_size for path in workspace.paths.root.rglob("*") if path.is_file()
    )
    if catalog_fresh is False:
        warnings.append("The rebuildable catalog is stale.")
    return WorkspaceStatus(
        workspace_id=workspace.identity.workspace_id,
        project_root=str(workspace.project_root),
        corpus_commit_id=workspace.corpus.read_head().commit_id,
        catalog_exists=catalog_exists,
        catalog_fresh=catalog_fresh,
        run_count=len(run_directories),
        artifact_object_count=sum(1 for _ in workspace.paths.artifacts.glob("*/*/artifact.json")),
        storage_bytes=storage_bytes,
        active_captures=active,
        warnings=tuple(warnings),
    )
