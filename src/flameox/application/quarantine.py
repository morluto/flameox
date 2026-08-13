from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, TypeAdapter

from flameox.application.recoverable_move import (
    lexical_path_beneath,
    move_path,
    resume_move,
    validate_manifest_id,
)
from flameox.atomic import atomic_write_json
from flameox.domain import DomainError, ErrorCode
from flameox.domain.models import utc_now
from flameox.models import ContractModel
from flameox.storage import Workspace
from flameox.storage.locks import RETENTION_EXCLUSIVE, WRITE_EXCLUSIVE


class _QuarantineManifest(ContractModel):
    schema_version: Literal[2] = 2
    quarantine_id: str
    operation: str
    detected_at: datetime
    original_path: str
    stored_path: str
    reason: str
    expected_format: str | None
    actual_format: str | None
    adapter: str | None = None
    originating_run_id: str | None = None
    sha256: str


class MovingQuarantineManifest(_QuarantineManifest):
    state: Literal["moving"] = "moving"
    recovery_options: tuple[()] = ()

    def quarantined(self) -> QuarantinedManifest:
        return QuarantinedManifest.model_validate(
            {
                **self.model_dump(mode="python"),
                "state": "quarantined",
                "recovery_options": ("restore",),
            }
        )


class QuarantinedManifest(_QuarantineManifest):
    state: Literal["quarantined"] = "quarantined"
    recovery_options: tuple[Literal["restore"]] = ("restore",)

    def restoring(self) -> RestoringQuarantineManifest:
        return RestoringQuarantineManifest.model_validate(
            {
                **self.model_dump(mode="python"),
                "state": "restoring",
                "recovery_options": (),
            }
        )


class RestoringQuarantineManifest(_QuarantineManifest):
    state: Literal["restoring"] = "restoring"
    recovery_options: tuple[()] = ()

    def restored(self) -> RestoredQuarantineManifest:
        return RestoredQuarantineManifest.model_validate(
            {**self.model_dump(mode="python"), "state": "restored"}
        )


class RestoredQuarantineManifest(_QuarantineManifest):
    state: Literal["restored"] = "restored"
    recovery_options: tuple[()] = ()


type QuarantineManifest = Annotated[
    MovingQuarantineManifest
    | QuarantinedManifest
    | RestoringQuarantineManifest
    | RestoredQuarantineManifest,
    Field(discriminator="state"),
]

_QUARANTINE_MANIFEST: TypeAdapter[QuarantineManifest] = TypeAdapter(QuarantineManifest)


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
        adapter: str | None = None,
        originating_run_id: str | None = None,
    ) -> QuarantineManifest:
        with self.workspace.locked(
            WRITE_EXCLUSIVE,
            RETENTION_EXCLUSIVE,
            phase="quarantine move",
        ):
            return self.quarantine_locked(
                source,
                reason=reason,
                operation=operation,
                expected_format=expected_format,
                actual_format=actual_format,
                adapter=adapter,
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
        adapter: str | None = None,
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
        manifest = MovingQuarantineManifest(
            quarantine_id=quarantine_id,
            operation=operation,
            detected_at=utc_now(),
            original_path=relative.as_posix(),
            stored_path=stored_relative.as_posix(),
            reason=reason,
            expected_format=expected_format,
            actual_format=actual_format,
            adapter=adapter,
            originating_run_id=originating_run_id,
            sha256=_tree_digest(source),
        )
        self._write_manifest(quarantine_root, manifest)
        move_path(source, stored_path)
        self._verify_digest(stored_path, manifest)
        quarantined = manifest.quarantined()
        self._write_manifest(quarantine_root, quarantined)
        return quarantined

    def restore(self, quarantine_id: str) -> QuarantineRestoreResult:
        with self.workspace.locked(
            WRITE_EXCLUSIVE,
            RETENTION_EXCLUSIVE,
            phase="quarantine restore",
        ):
            quarantine_root, manifest = self._read_manifest(quarantine_id)
            if not isinstance(manifest, QuarantinedManifest):
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
            manifest = manifest.restoring()
            self._write_manifest(quarantine_root, manifest)
            move_path(source, destination)
            manifest = manifest.restored()
            self._write_manifest(quarantine_root, manifest)
            return QuarantineRestoreResult(
                quarantine_id=quarantine_id,
                restored_path=manifest.original_path,
            )

    def resume(self, quarantine_id: str) -> QuarantineManifest:
        """Resolve a quarantine move interrupted between its two atomic steps."""
        with self.workspace.locked(
            WRITE_EXCLUSIVE,
            RETENTION_EXCLUSIVE,
            phase="quarantine resume",
        ):
            quarantine_root, manifest = self._read_manifest(quarantine_id)
            if isinstance(manifest, RestoringQuarantineManifest):
                return self._resume_restore_locked(quarantine_root, manifest)
            if not isinstance(manifest, MovingQuarantineManifest):
                return manifest
            source, destination = self._manifest_paths(quarantine_root, manifest)
            resume_move(source, destination, subject="quarantine move")
            self._verify_digest(destination, manifest)
            manifest = manifest.quarantined()
            self._write_manifest(quarantine_root, manifest)
            return manifest

    def moving_manifests(self) -> tuple[str, ...]:
        result: list[str] = []
        for path in sorted(self.workspace.paths.quarantine.glob("*/manifest.json")):
            try:
                manifest = _QUARANTINE_MANIFEST.validate_json(path.read_text())
            except (OSError, ValueError):
                continue
            if manifest.state in {"moving", "restoring"}:
                result.append(manifest.quarantine_id)
        return tuple(result)

    def _resume_restore_locked(
        self,
        quarantine_root: Path,
        manifest: RestoringQuarantineManifest,
    ) -> RestoredQuarantineManifest:
        destination, source = self._manifest_paths(quarantine_root, manifest)
        resume_move(source, destination, subject="quarantine restore")
        self._verify_digest(destination, manifest)
        restored = manifest.restored()
        self._write_manifest(quarantine_root, restored)
        return restored

    def list_manifests(self) -> tuple[QuarantineManifest, ...]:
        result: list[QuarantineManifest] = []
        for path in sorted(self.workspace.paths.quarantine.glob("*/manifest.json")):
            try:
                result.append(_QUARANTINE_MANIFEST.validate_json(path.read_text()))
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
            manifest = _QUARANTINE_MANIFEST.validate_json((root / "manifest.json").read_text())
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
        original, original_relative = lexical_path_beneath(
            self.workspace.paths.root,
            manifest.original_path,
            subject="Quarantine original path",
            error_code=ErrorCode.ARTIFACT_INTEGRITY_FAILED,
        )
        stored, stored_relative = lexical_path_beneath(
            quarantine_root,
            manifest.stored_path,
            subject="Quarantine stored path",
            error_code=ErrorCode.ARTIFACT_INTEGRITY_FAILED,
        )
        if original_relative.parts[0] in {"quarantine", "trash"}:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Quarantine manifest targets protected recovery storage.",
            )
        if len(stored_relative.parts) < 2 or stored_relative.parts[0] != "object":
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Quarantine manifest contains an invalid stored path.",
            )
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
        candidate, relative = lexical_path_beneath(
            workspace_root,
            lexical_relative.as_posix(),
            subject="Quarantine source",
            error_code=ErrorCode.EXECUTION_REFUSED,
        )
        if not relative.parts or relative.parts[0] in {"quarantine", "trash"}:
            raise DomainError(
                ErrorCode.EXECUTION_REFUSED,
                "Quarantine source targets protected recovery storage.",
            )
        return candidate, relative

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
