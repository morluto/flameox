from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from flamo.catalog import Catalog
from flamo.domain import DomainError, ErrorCode, digest_model
from flamo.storage import GenerationManifest, Workspace
from flamo.storage.atomic import atomic_write_json


class GarbageEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    kind: Literal["staging", "generation", "artifact"]
    byte_length: int
    reason: str


class GarbagePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    plan_id: str
    corpus_commit_id: str
    cutoff: datetime
    entries: tuple[GarbageEntry, ...]
    total_bytes: int
    recoverable: bool = True


class GarbageApplyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    trash_manifest_id: str
    moved: tuple[GarbageEntry, ...]
    total_bytes: int
    trash_root: str


class GarbageCollector:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def plan(self, *, minimum_age_hours: int = 24) -> GarbagePlan:
        head = self.workspace.corpus.read_head()
        cutoff = datetime.now(UTC) - timedelta(hours=minimum_age_hours)
        referenced_generations = {
            GenerationManifest.model_validate_json(
                (self.workspace.paths.root / relative).read_text()
            ).generation_id
            for relative in head.generation_manifests
        }
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            referenced_artifacts = {
                row[0]
                for row in snapshot.execute(
                    "SELECT DISTINCT artifact_id FROM artifact_registrations"
                ).fetchall()
            }
        entries: list[GarbageEntry] = []
        for path in sorted(self.workspace.paths.staging.iterdir()):
            if path.name == "captures" and path.is_dir():
                for capture_path in sorted(path.iterdir()):
                    self._candidate(
                        entries,
                        capture_path,
                        kind="staging",
                        reason=("Capture staging data is older than the recovery window."),
                        cutoff=cutoff,
                    )
                continue
            self._candidate(
                entries,
                path,
                kind="staging",
                reason="Unpublished staging data older than the recovery window.",
                cutoff=cutoff,
            )
        for path in sorted(self.workspace.paths.generations.iterdir()):
            if path.name not in referenced_generations:
                self._candidate(
                    entries,
                    path,
                    kind="generation",
                    reason="Generation is not reachable from corpus HEAD.",
                    cutoff=cutoff,
                )
        for metadata in sorted(self.workspace.paths.artifacts.glob("*/*/artifact.json")):
            artifact_id = f"sha256:{metadata.parent.name}"
            if artifact_id not in referenced_artifacts:
                self._candidate(
                    entries,
                    metadata.parent,
                    kind="artifact",
                    reason="Artifact has no registration in the active corpus.",
                    cutoff=cutoff,
                )
        content = {
            "corpus_commit_id": head.commit_id,
            "cutoff": cutoff.isoformat(),
            "entries": [entry.model_dump(mode="json") for entry in entries],
        }
        return GarbagePlan(
            plan_id=digest_model(content),
            corpus_commit_id=head.commit_id,
            cutoff=cutoff,
            entries=tuple(entries),
            total_bytes=sum(entry.byte_length for entry in entries),
        )

    def apply(self, plan: GarbagePlan) -> GarbageApplyResult:
        current = self.plan(
            minimum_age_hours=max(
                0,
                round((datetime.now(UTC) - plan.cutoff).total_seconds() / 3600),
            )
        )
        if current.corpus_commit_id != plan.corpus_commit_id or current.entries != plan.entries:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "Garbage-collection inputs changed after planning.",
                retryable=True,
            )
        manifest_id = str(uuid4())
        trash_root = self.workspace.paths.trash / manifest_id
        with (
            self.workspace.retention_locked(shared=False),
            self.workspace.write_locked(),
        ):
            if self.workspace.corpus.read_head().commit_id != plan.corpus_commit_id:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "Corpus HEAD changed before garbage collection.",
                    retryable=True,
                )
            for entry in plan.entries:
                source = (self.workspace.paths.root / entry.path).resolve()
                try:
                    relative = source.relative_to(self.workspace.paths.root)
                except ValueError as exc:
                    raise DomainError(
                        ErrorCode.EXECUTION_REFUSED,
                        "Garbage plan escapes the workspace.",
                    ) from exc
                if not source.exists():
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        f"Garbage candidate disappeared: {entry.path}",
                    )
                destination = trash_root / "objects" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
            atomic_write_json(
                trash_root / "manifest.json",
                {
                    "schema_version": 1,
                    "trash_manifest_id": manifest_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "source_corpus_commit_id": plan.corpus_commit_id,
                    "recoverable": True,
                    "entries": [entry.model_dump(mode="json") for entry in plan.entries],
                },
            )
        return GarbageApplyResult(
            trash_manifest_id=manifest_id,
            moved=plan.entries,
            total_bytes=plan.total_bytes,
            trash_root=str(trash_root),
        )

    def _candidate(
        self,
        entries: list[GarbageEntry],
        path: Path,
        *,
        kind: Literal["staging", "generation", "artifact"],
        reason: str,
        cutoff: datetime,
    ) -> None:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if modified > cutoff:
            return
        byte_length = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        entries.append(
            GarbageEntry(
                path=path.relative_to(self.workspace.paths.root).as_posix(),
                kind=kind,
                byte_length=byte_length,
                reason=reason,
            )
        )
