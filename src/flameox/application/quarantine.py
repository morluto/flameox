from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from flameox.application.recoverable_move import (
    move_path,
    resume_move,
    validate_manifest_id,
)
from flameox.domain import DomainError, ErrorCode
from flameox.domain.models import utc_now
from flameox.models import ContractModel
from flameox.storage import Workspace
from flameox.storage.atomic import atomic_write_json


class QuarantineManifest(ContractModel):
    schema_version: int = 1
    quarantine_id: str
    operation: str
    state: Literal["moving", "quarantined", "restoring", "restored"]
    detected_at: datetime
    original_path: str
    stored_path: str
    reason: str
    expected_format: str | None
    actual_format: str | None
    originating_run_id: str | None
    sha256: str
    recovery_options: tuple[Literal["restore"], ...] = ("restore",)


class QuarantineRestoreResult(ContractModel):
    schema_version: int = 1
    quarantine_id: str
    restored_path: str


class QuarantineService:
    """Recoverable, manifest-backed isolation for incomplete or invalid files."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def quarantine(
        self,
        source: Path,
        *,
        reason: str,
        operation: str,
        expected_format: str | None = None,
        actual_format: str | None = None,
        originating_run_id: str | None = None,
    ) -> QuarantineManifest:
        with (
            self.workspace.write_locked(),
            self.workspace.retention_locked(shared=False),
        ):
            return self.quarantine_locked(
                source,
                reason=reason,
                operation=operation,
                expected_format=expected_format,
                actual_format=actual_format,
                originating_run_id=originating_run_id,
            )

    def quarantine_locked(
        self,
        source: Path,
        *,
        reason: str,
        operation: str,
        expected_format: str | None = None,
        actual_format: str | None = None,
        originating_run_id: str | None = None,
    ) -> QuarantineManifest:
        source, relative = self._workspace_path(source)
        if not source.exists():
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                f"Quarantine source no longer exists: {relative.as_posix()}",
            )
        quarantine_id = str(uuid4())
        quarantine_root = self.workspace.paths.quarantine / quarantine_id
        stored_relative = Path("object") / relative
        stored_path = quarantine_root / stored_relative
        manifest = QuarantineManifest(
            quarantine_id=quarantine_id,
            operation=operation,
            state="moving",
            detected_at=utc_now(),
            original_path=relative.as_posix(),
            stored_path=stored_relative.as_posix(),
            reason=reason,
            expected_format=expected_format,
            actual_format=actual_format,
            originating_run_id=originating_run_id,
            sha256=_tree_digest(source),
        )
        self._write_manifest(quarantine_root, manifest)
        move_path(source, stored_path)
        manifest = manifest.model_copy(update={"state": "quarantined"})
        self._write_manifest(quarantine_root, manifest)
        return manifest

    def restore(self, quarantine_id: str) -> QuarantineRestoreResult:
        with (
            self.workspace.write_locked(),
            self.workspace.retention_locked(shared=False),
        ):
            quarantine_root, manifest = self._read_manifest(quarantine_id)
            if manifest.state != "quarantined":
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"Quarantine item is not restorable (state={manifest.state}).",
                )
            destination, source = self._manifest_paths(quarantine_root, manifest)
            if destination.exists():
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"Restore destination already exists: {manifest.original_path}",
                )
            if not source.exists() or _tree_digest(source) != manifest.sha256:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "Quarantined content is missing or differs from its manifest.",
                )
            manifest = manifest.model_copy(update={"state": "restoring"})
            self._write_manifest(quarantine_root, manifest)
            move_path(source, destination)
            manifest = manifest.model_copy(update={"state": "restored"})
            self._write_manifest(quarantine_root, manifest)
            return QuarantineRestoreResult(
                quarantine_id=quarantine_id,
                restored_path=manifest.original_path,
            )

    def resume(self, quarantine_id: str) -> QuarantineManifest:
        """Resolve a quarantine move interrupted between its two atomic steps."""
        with (
            self.workspace.write_locked(),
            self.workspace.retention_locked(shared=False),
        ):
            quarantine_root, manifest = self._read_manifest(quarantine_id)
            if manifest.state == "restoring":
                return self._resume_restore_locked(quarantine_root, manifest)
            if manifest.state != "moving":
                return manifest
            source, destination = self._manifest_paths(quarantine_root, manifest)
            resume_move(source, destination, subject="quarantine move")
            self._verify_digest(destination, manifest)
            manifest = manifest.model_copy(update={"state": "quarantined"})
            self._write_manifest(quarantine_root, manifest)
            return manifest

    def moving_manifests(self) -> tuple[str, ...]:
        result: list[str] = []
        for path in sorted(self.workspace.paths.quarantine.glob("*/manifest.json")):
            try:
                manifest = QuarantineManifest.model_validate_json(path.read_text())
            except (OSError, ValueError):
                continue
            if manifest.state in {"moving", "restoring"}:
                result.append(manifest.quarantine_id)
        return tuple(result)

    def _resume_restore_locked(
        self,
        quarantine_root: Path,
        manifest: QuarantineManifest,
    ) -> QuarantineManifest:
        destination, source = self._manifest_paths(quarantine_root, manifest)
        resume_move(source, destination, subject="quarantine restore")
        self._verify_digest(destination, manifest)
        manifest = manifest.model_copy(update={"state": "restored"})
        self._write_manifest(quarantine_root, manifest)
        return manifest

    def list_manifests(self) -> tuple[QuarantineManifest, ...]:
        result: list[QuarantineManifest] = []
        for path in sorted(self.workspace.paths.quarantine.glob("*/manifest.json")):
            try:
                result.append(QuarantineManifest.model_validate_json(path.read_text()))
            except (OSError, ValueError):
                continue
        return tuple(result)

    def _read_manifest(
        self,
        quarantine_id: str,
    ) -> tuple[Path, QuarantineManifest]:
        validate_manifest_id(quarantine_id, kind="quarantine")
        root = self.workspace.paths.quarantine / quarantine_id
        try:
            manifest = QuarantineManifest.model_validate_json((root / "manifest.json").read_text())
        except (OSError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Quarantine manifest {quarantine_id!r} is missing or invalid.",
            ) from exc
        if manifest.quarantine_id != quarantine_id:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Quarantine manifest identity does not match its directory.",
            )
        self._manifest_paths(root, manifest)
        return root, manifest

    def _manifest_paths(
        self,
        quarantine_root: Path,
        manifest: QuarantineManifest,
    ) -> tuple[Path, Path]:
        original = (self.workspace.paths.root / manifest.original_path).resolve()
        stored = (quarantine_root / manifest.stored_path).resolve()
        try:
            original.relative_to(self.workspace.paths.root)
            stored.relative_to(quarantine_root)
        except ValueError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Quarantine manifest contains a path outside recovery storage.",
            ) from exc
        return original, stored

    def _workspace_path(self, source: Path) -> tuple[Path, Path]:
        workspace_root = self.workspace.paths.root.resolve()
        lexical = source.absolute()
        try:
            lexical_relative = lexical.relative_to(workspace_root)
        except ValueError as exc:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Quarantine source escapes the workspace.",
            ) from exc
        candidate = workspace_root
        for part in lexical_relative.parts:
            candidate /= part
            if candidate.is_symlink():
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "Quarantine does not follow symbolic links.",
                )
        resolved = source.resolve()
        try:
            relative = resolved.relative_to(self.workspace.paths.root)
        except ValueError as exc:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Quarantine source escapes the workspace.",
            ) from exc
        if not relative.parts or relative.parts[0] in {"quarantine", "trash"}:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Quarantine source targets protected recovery storage.",
            )
        return resolved, relative

    @staticmethod
    def _verify_digest(path: Path, manifest: QuarantineManifest) -> None:
        if not path.exists() or _tree_digest(path) != manifest.sha256:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Quarantined content is missing or differs from its manifest.",
            )

    @staticmethod
    def _write_manifest(root: Path, manifest: QuarantineManifest) -> None:
        atomic_write_json(root / "manifest.json", manifest.model_dump(mode="json"))


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_symlink():
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            "Quarantine does not follow symbolic links.",
        )
    if path.is_file():
        digest.update(b"F")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            "Only regular files and directories can be quarantined.",
        )
    digest.update(b"D")
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Quarantine does not follow symbolic links.",
            )
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        if item.is_file():
            digest.update(b"F")
            with item.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        elif item.is_dir():
            digest.update(b"D")
        else:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Quarantine content must contain only regular files and directories.",
            )
    return digest.hexdigest()
