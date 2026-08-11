from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import Field, TypeAdapter, model_validator

from flameox.application.recoverable_move import (
    lexical_path_beneath,
    move_path,
    resume_move,
    validate_manifest_id,
)
from flameox.atomic import atomic_write_json, fsync_directory
from flameox.catalog import Catalog
from flameox.domain import DomainError, ErrorCode, digest_model
from flameox.models import ContractModel
from flameox.storage import GenerationManifest, Workspace, tree_bytes


class GarbageEntryKind(StrEnum):
    STAGING = "staging"
    GENERATION = "generation"
    EVIDENCE = "evidence"
    ARTIFACT = "artifact"


class GarbageEntry(ContractModel):
    path: str
    kind: GarbageEntryKind
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
    recoverable: Literal[True] = True


class GarbageApplyResult(ContractModel):
    schema_version: int = 1
    trash_manifest_id: str
    moved: tuple[GarbageEntry, ...]
    total_bytes: int
    trash_root: str


class _TrashManifest(ContractModel):
    schema_version: int = 1
    trash_manifest_id: str
    operation: Literal["garbage_collection"] = "garbage_collection"
    created_at: datetime
    expires_at: datetime
    source_corpus_commit_id: str
    entries: tuple[GarbageEntry, ...]
    moved_paths: tuple[str, ...]

    @model_validator(mode="after")
    def moved_paths_reference_manifest_entries(self) -> Self:
        moved_paths = set(self.moved_paths)
        if len(moved_paths) != len(self.moved_paths):
            raise ValueError("moved_paths must not contain duplicates")
        entry_paths = {entry.path for entry in self.entries}
        if not moved_paths <= entry_paths:
            raise ValueError("moved_paths must reference manifest entries")
        return self


class MovingTrashManifest(_TrashManifest):
    state: Literal["moving"] = "moving"
    recoverable: Literal[True] = True

    def record_moved_paths(self, moved_paths: tuple[str, ...]) -> Self:
        return self.__class__.model_validate(
            {**self.model_dump(mode="python"), "moved_paths": moved_paths}
        )

    def complete(self) -> RecoverableTrashManifest:
        return RecoverableTrashManifest.model_validate(
            {
                **self.model_dump(mode="python"),
                "state": "recoverable",
                "moved_paths": tuple(entry.path for entry in self.entries),
            }
        )


class RecoverableTrashManifest(_TrashManifest):
    state: Literal["recoverable"] = "recoverable"
    recoverable: Literal[True] = True

    @model_validator(mode="after")
    def all_entries_are_moved(self) -> Self:
        if set(self.moved_paths) != {entry.path for entry in self.entries}:
            raise ValueError("a recoverable manifest must contain every entry in moved_paths")
        return self

    def restoring(self) -> RestoringTrashManifest:
        return RestoringTrashManifest.model_validate(
            {**self.model_dump(mode="python"), "state": "restoring"}
        )


class RestoringTrashManifest(_TrashManifest):
    state: Literal["restoring"] = "restoring"
    recoverable: Literal[True] = True

    def record_pending_paths(self, moved_paths: tuple[str, ...]) -> Self:
        return self.__class__.model_validate(
            {**self.model_dump(mode="python"), "moved_paths": moved_paths}
        )

    def restored(self) -> RestoredTrashManifest:
        return RestoredTrashManifest.model_validate(
            {
                **self.model_dump(mode="python"),
                "state": "restored",
                "recoverable": False,
                "moved_paths": (),
            }
        )


class RestoredTrashManifest(_TrashManifest):
    state: Literal["restored"] = "restored"
    recoverable: Literal[False] = False
    moved_paths: tuple[()] = ()


type TrashManifest = Annotated[
    MovingTrashManifest | RecoverableTrashManifest | RestoringTrashManifest | RestoredTrashManifest,
    Field(discriminator="state"),
]

_TRASH_MANIFEST: TypeAdapter[TrashManifest] = TypeAdapter(TrashManifest)


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
                        kind=GarbageEntryKind.STAGING,
                        reason=("Capture staging data is older than the recovery window."),
                        cutoff=cutoff,
                    )
                continue
            self._candidate(
                entries,
                path,
                kind=GarbageEntryKind.STAGING,
                reason="Unpublished staging data older than the recovery window.",
                cutoff=cutoff,
            )
        for path in sorted(self.workspace.paths.generations.iterdir()):
            if path.name not in referenced_generations:
                self._candidate(
                    entries,
                    path,
                    kind=GarbageEntryKind.GENERATION,
                    reason="Generation is not reachable from a retained corpus commit.",
                    cutoff=cutoff,
                )
        for path in sorted(self.workspace.paths.evidence.glob("*/generation=*")):
            generation_id = path.name.removeprefix("generation=")
            if path.is_dir() and generation_id not in referenced_generations:
                self._candidate(
                    entries,
                    path,
                    kind=GarbageEntryKind.EVIDENCE,
                    reason="Evidence is not reachable from a retained corpus commit.",
                    cutoff=cutoff,
                )
        for metadata in sorted(self.workspace.paths.artifacts.glob("*/*/artifact.json")):
            artifact_id = f"sha256:{metadata.parent.name}"
            if artifact_id not in referenced_artifacts:
                self._candidate(
                    entries,
                    metadata.parent,
                    kind=GarbageEntryKind.ARTIFACT,
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
        manifest = MovingTrashManifest(
            trash_manifest_id=manifest_id,
            created_at=created_at,
            expires_at=created_at + timedelta(hours=recovery_window_hours),
            source_corpus_commit_id=plan.corpus_commit_id,
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
            for entry in plan.entries:
                self._source(entry)
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
                manifest = manifest.record_moved_paths(tuple(moved_paths))
                self._write_manifest(trash_root, manifest)
            recoverable = manifest.complete()
            self._write_manifest(trash_root, recoverable)
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
            if isinstance(manifest, RestoringTrashManifest):
                return self._resume_restore_locked(trash_root, manifest)
            if not isinstance(manifest, MovingTrashManifest):
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
                manifest = manifest.record_moved_paths(tuple(sorted(moved_paths)))
                self._write_manifest(trash_root, manifest)
            recoverable = manifest.complete()
            self._write_manifest(trash_root, recoverable)
            return recoverable

    def restore(self, trash_manifest_id: str) -> GarbageRestoreResult:
        """Restore one recoverable trash manifest without guessing its scope."""
        with (
            self.workspace.write_locked(),
            self.workspace.retention_locked(shared=False),
        ):
            trash_root, manifest = self._read_manifest(trash_manifest_id)
            if not isinstance(manifest, RecoverableTrashManifest):
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"Trash manifest is not recoverable (state={manifest.state}).",
                )
            restoring = manifest.restoring()
            self._write_manifest(trash_root, restoring)
            restored = self._resume_restore_locked(trash_root, restoring)
            return GarbageRestoreResult(
                trash_manifest_id=trash_manifest_id,
                restored=restored.entries,
            )

    def purge(self, trash_manifest_id: str) -> GarbagePurgeResult:
        """Permanently delete one expired, recoverable trash manifest."""
        with (
            self.workspace.write_locked(),
            self.workspace.retention_locked(shared=False),
        ):
            trash_root, manifest = self._read_manifest(trash_manifest_id)
            if not isinstance(manifest, RecoverableTrashManifest):
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
                manifest = _TRASH_MANIFEST.validate_json(path.read_text())
            except (OSError, ValueError):
                continue
            if manifest.state in {"moving", "restoring"}:
                manifests.append(manifest.trash_manifest_id)
        return tuple(manifests)

    def _resume_restore_locked(
        self,
        trash_root: Path,
        manifest: RestoringTrashManifest,
    ) -> RestoredTrashManifest:
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
            manifest = manifest.record_pending_paths(tuple(sorted(pending)))
            self._write_manifest(trash_root, manifest)
        restored = manifest.restored()
        self._write_manifest(trash_root, restored)
        return restored

    def _read_manifest(self, trash_manifest_id: str) -> tuple[Path, TrashManifest]:
        validate_manifest_id(trash_manifest_id, kind="trash manifest")
        trash_root = self.workspace.paths.trash / trash_manifest_id
        try:
            manifest = _TRASH_MANIFEST.validate_json((trash_root / "manifest.json").read_text())
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
        source, relative = self._workspace_path(entry.path)
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
        kind: GarbageEntryKind,
        reason: str,
        cutoff: datetime,
    ) -> None:
        self._workspace_path(path.relative_to(self.workspace.paths.root).as_posix())
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

    def _workspace_path(self, value: str) -> tuple[Path, Path]:
        return lexical_path_beneath(
            self.workspace.paths.root,
            value,
            subject="Garbage path",
            error_code=ErrorCode.ARTIFACT_INTEGRITY_FAILED,
        )
