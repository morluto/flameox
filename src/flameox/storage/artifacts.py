from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from flameox.atomic import atomic_write_json, fsync_directory
from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.models import ArtifactContent, Integrity, utc_now
from flameox.filesystem import BoundedFileSystem
from flameox.storage.quotas import StorageQuota
from flameox.storage.workspace import Workspace

_SAFE_EXTENSION = re.compile(r"^\.[A-Za-z0-9][A-Za-z0-9._-]{0,15}$")
_SAFE_PAYLOAD_NAME = re.compile(r"^payload(?:\.[A-Za-z0-9][A-Za-z0-9._-]{0,15})?$")


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    content: ArtifactContent
    payload_path: Path


@dataclass(frozen=True, slots=True)
class ArtifactSnapshot:
    payload_path: Path
    byte_length: int
    sha256: str


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]


class ArtifactStore:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def import_path(
        self,
        source: Path,
        *,
        allowed_roots: tuple[Path, ...],
        max_bytes: int,
    ) -> StoredArtifact:
        source = source.absolute()
        with self.temporary_snapshot(
            source,
            allowed_roots=allowed_roots,
            max_bytes=max_bytes,
        ) as snapshot:
            hexadecimal = snapshot.sha256
            artifact_id = f"sha256:{hexadecimal}"
            extension = source.suffix if _SAFE_EXTENSION.fullmatch(source.suffix) else ""
            payload_name = f"payload{extension}"
            object_root = self.workspace.paths.artifacts / hexadecimal[:2] / hexadecimal
            metadata_path = object_root / "artifact.json"
            with self.workspace.write_locked():
                StorageQuota(self.workspace).require_capacity(staging=True)
                if metadata_path.exists():
                    content = self._read_metadata(metadata_path)
                    if (
                        content.artifact_id != artifact_id
                        or content.byte_length != snapshot.byte_length
                    ):
                        raise DomainError(
                            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                            "Existing content-addressed artifact metadata conflicts.",
                        )
                    payload_path = object_root / content.payload_name
                    if not payload_path.is_file():
                        raise DomainError(
                            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                            "Existing artifact payload is missing.",
                        )
                else:
                    object_root.mkdir(parents=True, exist_ok=True)
                    payload_path = object_root / payload_name
                    os.replace(snapshot.payload_path, payload_path)
                    fsync_directory(object_root)
                    content = ArtifactContent(
                        artifact_id=artifact_id,
                        byte_length=snapshot.byte_length,
                        payload_name=payload_name,
                        integrity=Integrity(sha256=hexadecimal, hashed_at=utc_now()),
                    )
                    atomic_write_json(metadata_path, content.model_dump(mode="json"))
        return StoredArtifact(content=content, payload_path=payload_path)

    @contextmanager
    def temporary_snapshot(
        self,
        source: Path,
        *,
        allowed_roots: tuple[Path, ...],
        max_bytes: int,
    ) -> Iterator[ArtifactSnapshot]:
        """Yield an immutable bounded copy without committing it to the CAS."""
        source = source.absolute()
        self._require_allowed_parent(source, allowed_roots)
        source_fd = self._open_source(source, allowed_roots)
        staging_root = self.workspace.paths.staging / f"snapshot-{uuid4().hex}"
        staging_root.mkdir(parents=True, exist_ok=False)
        staged_path = staging_root / "payload"
        staged_fd = os.open(staged_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        total = 0
        copied = False
        try:
            before = os.fstat(source_fd)
            if not stat.S_ISREG(before.st_mode):
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "Artifact imports must be regular files.",
                )
            if before.st_nlink != 1:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "Artifact imports cannot use mutable hard-link sources.",
                )
            StorageQuota(self.workspace).require_capacity(
                additional_bytes=before.st_size,
                staging=True,
            )
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise DomainError(
                        ErrorCode.ARTIFACT_TOO_LARGE,
                        f"Artifact exceeds the {max_bytes}-byte import limit.",
                    )
                digest.update(chunk)
                _write_all(staged_fd, chunk)
            os.fsync(staged_fd)
            after = os.fstat(source_fd)
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if identity_after != identity_before or total != before.st_size:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "Artifact import source changed while it was read.",
                    retryable=True,
                )
            copied = True
        finally:
            os.close(source_fd)
            os.close(staged_fd)
            if not copied:
                shutil.rmtree(staging_root, ignore_errors=True)
        try:
            yield ArtifactSnapshot(
                payload_path=staged_path,
                byte_length=total,
                sha256=digest.hexdigest(),
            )
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    def _open_source(
        self,
        source: Path,
        allowed_roots: tuple[Path, ...],
    ) -> int:
        return BoundedFileSystem(allowed_roots).open_descriptor(source)

    def get(self, artifact_id: str) -> StoredArtifact:
        hexadecimal = artifact_id.removeprefix("sha256:")
        if len(hexadecimal) != 64 or any(
            character not in "0123456789abcdef" for character in hexadecimal
        ):
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Invalid artifact identifier.")
        artifacts_root = self.workspace.paths.artifacts
        shard_root = artifacts_root / hexadecimal[:2]
        object_root = shard_root / hexadecimal
        metadata_path = object_root / "artifact.json"
        for path in (artifacts_root, shard_root, object_root, metadata_path):
            if path.is_symlink():
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "Artifact storage paths must not contain symbolic links.",
                )
        try:
            object_root.resolve().relative_to(artifacts_root.resolve())
        except ValueError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Artifact storage path escapes the artifact root.",
            ) from exc
        content = self._read_metadata(metadata_path)
        if _SAFE_PAYLOAD_NAME.fullmatch(content.payload_name) is None:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Artifact metadata contains an invalid payload name.",
            )
        payload_path = object_root / content.payload_name
        # Open with O_NOFOLLOW to prevent TOCTOU symlink attacks between
        # the is_symlink() check above and the actual open().
        # On Windows O_NOFOLLOW is absent, so we explicitly reject reparse
        # points (symlinks, junctions) before the open.
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if os.name == "nt" and payload_path.is_symlink():
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Artifact payload must not be a symbolic link.",
            )
        try:
            # O_NONBLOCK prevents blocking on FIFOs; combined with O_RDONLY
            # it returns immediately for regular files and fails on FIFOs
            # on most POSIX systems.
            nonblock = getattr(os, "O_NONBLOCK", 0)
            payload_fd = os.open(payload_path, os.O_RDONLY | nofollow | nonblock)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "Artifact payload must not be a symbolic link.",
                ) from exc
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Artifact payload is missing on disk.",
            ) from exc
        try:
            stat_info = os.fstat(payload_fd)
            if not stat.S_ISREG(stat_info.st_mode):
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "Artifact payload must be a regular file.",
                )
            # Defense-in-depth: re-verify the stored payload against the recorded
            # integrity digest and length so that corruption or tampering that
            # occurred after import is detected on retrieval rather than silently
            # returned to the caller.
            actual_size = stat_info.st_size
            if actual_size != content.byte_length:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "Artifact payload length does not match recorded metadata.",
                    details={"expected": content.byte_length, "actual": actual_size},
                )
            digest = hashlib.sha256()
            while True:
                chunk = os.read(payload_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            if digest.hexdigest() != content.integrity.sha256:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "Artifact payload digest does not match recorded metadata.",
                )
        finally:
            os.close(payload_fd)
        return StoredArtifact(
            content=content,
            payload_path=payload_path,
        )

    def _read_metadata(self, path: Path) -> ArtifactContent:
        try:
            return ArtifactContent.model_validate(json.loads(path.read_text()))
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                f"Artifact metadata is missing or invalid: {path}",
            ) from exc

    def _require_allowed_parent(
        self,
        source: Path,
        allowed_roots: tuple[Path, ...],
    ) -> None:
        parent = source.parent.resolve()
        for allowed_root in allowed_roots:
            try:
                parent.relative_to(allowed_root.resolve())
                return
            except ValueError:
                continue
        raise DomainError(
            ErrorCode.EXECUTION_REFUSED,
            "Artifact import source is outside the allowed roots.",
        )
