from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

from flameox.application.recoverable_move import (
    move_path,
    resume_move,
    validate_manifest_id,
)
from flameox.atomic import atomic_write_json, fsync_directory
from flameox.catalog import Catalog
from flameox.domain import DomainError, ErrorCode, digest_model
from flameox.models import ContractModel
from flameox.storage import GenerationManifest, Workspace, tree_bytes


class GarbageEntry(ContractModel):
    path: str
    kind: Literal["staging", "generation", "artifact"]
    byte_length: int
    reason: str


class GarbagePlan(ContractModel):
    schema_version: int = 1
    plan_id: str
    corpus_commit_id: str
    cutoff: datetime
    root_corpus_commit_ids: tuple[str, ...]
    root_generation_ids: tuple[str, ...]
    root_artifact_ids: tuple[str, ...]
    entries: tuple[GarbageEntry, ...]
    total_bytes: int
    recoverable: bool = True


class GarbageApplyResult(ContractModel):
    schema_version: int = 1
    trash_manifest_id: str
    moved: tuple[GarbageEntry, ...]
    total_bytes: int
    trash_root: str


class TrashManifest(ContractModel):
    schema_version: int = 1
    trash_manifest_id: str
    operation: Literal["garbage_collection"] = "garbage_collection"
    state: Literal["moving", "recoverable", "restoring", "restored"]
    created_at: datetime
    expires_at: datetime
    source_corpus_commit_id: str
    recoverable: bool
    entries: tuple[GarbageEntry, ...]
    moved_paths: tuple[str, ...]


class GarbagePurgeResult(ContractModel):
    schema_version: int = 1
    trash_manifest_id: str
    purged_entries: int
    purged_bytes: int


class GarbageRestoreResult(ContractModel):
    schema_version: int = 1
    trash_manifest_id: str
    restored: tuple[GarbageEntry, ...]


class GarbageCollector:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def plan(self, *, minimum_age_hours: int = 24) -> GarbagePlan:
        head = self.workspace.corpus.read_head()
        cutoff = datetime.now(UTC) - timedelta(hours=minimum_age_hours)
        root_commit_ids = {head.commit_id}
        referenced_generations: set[str] = set()
        referenced_artifacts: set[str] = set()
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            root_commit_ids.update(
                str(row[0])
                for row in snapshot.execute(
                    "SELECT DISTINCT corpus_commit_id FROM analyses "
                    "UNION SELECT DISTINCT corpus_commit_id FROM run_sets"
                ).fetchall()
            )
            referenced_generations.update(
                str(row[0])
                for row in snapshot.execute(
                    "SELECT DISTINCT ref_id FROM evidence_refs "
                    "WHERE ref_type = 'generation' "
                    "UNION SELECT DISTINCT unnest(input_generation_ids) FROM analyses"
                ).fetchall()
                if row[0] is not None
            )
            referenced_artifacts.update(
                str(row[0])
                for row in snapshot.execute(
                    "SELECT DISTINCT ref_id FROM evidence_refs "
                    "WHERE ref_type = 'artifact' "
                    "UNION SELECT DISTINCT unnest(input_artifact_ids) FROM analyses"
                ).fetchall()
                if row[0] is not None
            )
        for commit_id in sorted(root_commit_ids):
            commit = self.workspace.corpus.read_commit(commit_id)
            for relative in commit.generation_manifests:
                manifest = GenerationManifest.model_validate_json(
                    (self.workspace.paths.root / relative).read_text()
                )
                referenced_generations.add(manifest.generation_id)
                referenced_artifacts.update(manifest.input_artifact_ids)
            with Catalog(self.workspace).open_snapshot(commit_id) as snapshot:
                referenced_artifacts.update(
                    str(row[0])
                    for row in snapshot.execute(
                        "SELECT DISTINCT artifact_id FROM artifact_registrations"
                    ).fetchall()
                )
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
                    reason="Generation is not reachable from a retained corpus commit.",
                    cutoff=cutoff,
                )
        for metadata in sorted(self.workspace.paths.artifacts.glob("*/*/artifact.json")):
            artifact_id = f"sha256:{metadata.parent.name}"
            if artifact_id not in referenced_artifacts:
                self._candidate(
                    entries,
                    metadata.parent,
                    kind="artifact",
                    reason="Artifact is not reachable from retained evidence.",
                    cutoff=cutoff,
                )
        content = {
            "corpus_commit_id": head.commit_id,
            "cutoff": cutoff.isoformat(),
            "root_corpus_commit_ids": sorted(root_commit_ids),
            "root_generation_ids": sorted(referenced_generations),
            "root_artifact_ids": sorted(referenced_artifacts),
            "entries": [entry.model_dump(mode="json") for entry in entries],
        }
        return GarbagePlan(
            plan_id=digest_model(content),
            corpus_commit_id=head.commit_id,
            cutoff=cutoff,
            root_corpus_commit_ids=tuple(sorted(root_commit_ids)),
            root_generation_ids=tuple(sorted(referenced_generations)),
            root_artifact_ids=tuple(sorted(referenced_artifacts)),
            entries=tuple(entries),
            total_bytes=sum(entry.byte_length for entry in entries),
        )

    def apply(
        self,
        plan: GarbagePlan,
        *,
        recovery_window_hours: int = 24,
    ) -> GarbageApplyResult:
        if recovery_window_hours < 0:
            raise ValueError("recovery_window_hours must not be negative")
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
        created_at = datetime.now(UTC)
        manifest = TrashManifest(
            trash_manifest_id=manifest_id,
            state="moving",
            created_at=created_at,
            expires_at=created_at + timedelta(hours=recovery_window_hours),
            source_corpus_commit_id=plan.corpus_commit_id,
            recoverable=True,
            entries=plan.entries,
            moved_paths=(),
        )
        with (
            self.workspace.write_locked(),
            self.workspace.retention_locked(shared=False),
        ):
            if self.workspace.corpus.read_head().commit_id != plan.corpus_commit_id:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "Corpus HEAD changed before garbage collection.",
                    retryable=True,
                )
            self._write_manifest(trash_root, manifest)
            moved_paths: list[str] = []
            for entry in plan.entries:
                source, relative = self._source(entry)
                if not source.exists():
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        f"Garbage candidate disappeared: {entry.path}",
                    )
                destination = trash_root / "objects" / relative
                move_path(source, destination)
                moved_paths.append(entry.path)
                manifest = manifest.model_copy(update={"moved_paths": tuple(moved_paths)})
                self._write_manifest(trash_root, manifest)
            manifest = manifest.model_copy(
                update={
                    "state": "recoverable",
                    "moved_paths": tuple(entry.path for entry in plan.entries),
                }
            )
            self._write_manifest(trash_root, manifest)
        return GarbageApplyResult(
            trash_manifest_id=manifest_id,
            moved=plan.entries,
            total_bytes=plan.total_bytes,
            trash_root=str(trash_root),
        )

    def resume(self, trash_manifest_id: str) -> TrashManifest:
        """Finish an explicitly requested GC move interrupted by process death."""
        with (
            self.workspace.write_locked(),
            self.workspace.retention_locked(shared=False),
        ):
            trash_root, manifest = self._read_manifest(trash_manifest_id)
            if manifest.state == "restoring":
                return self._resume_restore_locked(trash_root, manifest)
            if manifest.state != "moving":
                return manifest
            moved_paths = set(manifest.moved_paths)
            for entry in manifest.entries:
                source, relative = self._source(entry)
                destination = trash_root / "objects" / relative
                resume_move(
                    source,
                    destination,
                    subject=f"garbage candidate {entry.path}",
                )
                moved_paths.add(entry.path)
                manifest = manifest.model_copy(update={"moved_paths": tuple(sorted(moved_paths))})
                self._write_manifest(trash_root, manifest)
            manifest = manifest.model_copy(update={"state": "recoverable"})
            self._write_manifest(trash_root, manifest)
            return manifest

    def restore(self, trash_manifest_id: str) -> GarbageRestoreResult:
        """Restore one recoverable trash manifest without guessing its scope."""
        with (
            self.workspace.write_locked(),
            self.workspace.retention_locked(shared=False),
        ):
            trash_root, manifest = self._read_manifest(trash_manifest_id)
            if manifest.state != "recoverable":
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"Trash manifest is not recoverable (state={manifest.state}).",
                )
            manifest = manifest.model_copy(update={"state": "restoring"})
            self._write_manifest(trash_root, manifest)
            manifest = self._resume_restore_locked(trash_root, manifest)
            return GarbageRestoreResult(
                trash_manifest_id=trash_manifest_id,
                restored=manifest.entries,
            )

    def purge(self, trash_manifest_id: str) -> GarbagePurgeResult:
        """Permanently delete one expired, recoverable trash manifest."""
        with (
            self.workspace.write_locked(),
            self.workspace.retention_locked(shared=False),
        ):
            trash_root, manifest = self._read_manifest(trash_manifest_id)
            if manifest.state != "recoverable" or not manifest.recoverable:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "Only a complete recoverable trash manifest can be purged.",
                )
            if datetime.now(UTC) < manifest.expires_at:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "The trash recovery window has not expired.",
                    details={"expires_at": manifest.expires_at.isoformat()},
                )
            entry_count = len(manifest.entries)
            total_bytes = sum(entry.byte_length for entry in manifest.entries)
            # ``shutil.rmtree`` is not atomic; a partial failure would leave
            # the trash directory half-deleted with the manifest gone, giving
            # the operator no recovery path. Capture the failure and re-raise
            # as a retryable domain error so the caller can re-attempt the
            # purge once the underlying filesystem issue is resolved.
            try:
                shutil.rmtree(trash_root)
            except OSError as exc:
                raise DomainError(
                    ErrorCode.INTERNAL_ERROR,
                    "Purge failed while deleting the trash directory.",
                    retryable=True,
                    details={"trash_manifest_id": trash_manifest_id},
                ) from exc
            fsync_directory(self.workspace.paths.trash)
            return GarbagePurgeResult(
                trash_manifest_id=trash_manifest_id,
                purged_entries=entry_count,
                purged_bytes=total_bytes,
            )

    def moving_manifests(self) -> tuple[str, ...]:
        manifests: list[str] = []
        for path in sorted(self.workspace.paths.trash.glob("*/manifest.json")):
            try:
                manifest = TrashManifest.model_validate_json(path.read_text())
            except (OSError, ValueError):
                continue
            if manifest.state in {"moving", "restoring"}:
                manifests.append(manifest.trash_manifest_id)
        return tuple(manifests)

    def _resume_restore_locked(
        self,
        trash_root: Path,
        manifest: TrashManifest,
    ) -> TrashManifest:
        pending = set(manifest.moved_paths)
        for entry in reversed(manifest.entries):
            destination, relative = self._source(entry)
            source = trash_root / "objects" / relative
            if not resume_move(
                source,
                destination,
                subject=f"trash restore {entry.path}",
            ):
                pending.discard(entry.path)
                continue
            pending.discard(entry.path)
            manifest = manifest.model_copy(update={"moved_paths": tuple(sorted(pending))})
            self._write_manifest(trash_root, manifest)
        manifest = manifest.model_copy(
            update={"state": "restored", "recoverable": False, "moved_paths": ()}
        )
        self._write_manifest(trash_root, manifest)
        return manifest

    def _read_manifest(self, trash_manifest_id: str) -> tuple[Path, TrashManifest]:
        validate_manifest_id(trash_manifest_id, kind="trash manifest")
        trash_root = self.workspace.paths.trash / trash_manifest_id
        try:
            manifest = TrashManifest.model_validate_json((trash_root / "manifest.json").read_text())
        except (OSError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Trash manifest {trash_manifest_id!r} is missing or invalid.",
            ) from exc
        if manifest.trash_manifest_id != trash_manifest_id:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Trash manifest identity does not match its directory.",
            )
        return trash_root, manifest

    @staticmethod
    def _write_manifest(trash_root: Path, manifest: TrashManifest) -> None:
        atomic_write_json(
            trash_root / "manifest.json",
            manifest.model_dump(mode="json"),
        )

    def _source(self, entry: GarbageEntry) -> tuple[Path, Path]:
        source = (self.workspace.paths.root / entry.path).resolve()
        try:
            relative = source.relative_to(self.workspace.paths.root)
        except ValueError as exc:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Garbage plan escapes the workspace.",
            ) from exc
        if relative.parts and relative.parts[0] in {"trash", "quarantine"}:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Garbage plan targets protected recovery storage.",
            )
        return source, relative

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
        byte_length = tree_bytes(path)
        entries.append(
            GarbageEntry(
                path=path.relative_to(self.workspace.paths.root).as_posix(),
                kind=kind,
                byte_length=byte_length,
                reason=reason,
            )
        )
