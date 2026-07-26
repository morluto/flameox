from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from flamo.domain.errors import DomainError, ErrorCode
from flamo.domain.models import ArtifactContent, Integrity, utc_now
from flamo.storage.atomic import atomic_write_json, fsync_directory
from flamo.storage.quotas import StorageQuota
from flamo.storage.workspace import Workspace

_SAFE_EXTENSION = re.compile(r"^\.[A-Za-z0-9][A-Za-z0-9._-]{0,15}$")


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    content: ArtifactContent
    payload_path: Path


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
        self._require_allowed_parent(source, allowed_roots)
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            source_fd = os.open(source, flags)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOENT}:
                raise DomainError(
                    ErrorCode.EXECUTION_REFUSED,
                    "Artifact import source is missing or is a symbolic link.",
                ) from exc
            raise

        staging_root = self.workspace.paths.staging / f"import-{uuid4().hex}"
        staging_root.mkdir(parents=True, exist_ok=False)
        staged_path = staging_root / "payload"
        staged_fd = os.open(staged_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        total = 0
        staged_complete = False
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
            staged_complete = True
        finally:
            os.close(source_fd)
            os.close(staged_fd)
            if not staged_complete:
                shutil.rmtree(staging_root, ignore_errors=True)

        hexadecimal = digest.hexdigest()
        artifact_id = f"sha256:{hexadecimal}"
        extension = source.suffix if _SAFE_EXTENSION.fullmatch(source.suffix) else ""
        payload_name = f"payload{extension}"
        object_root = self.workspace.paths.artifacts / hexadecimal[:2] / hexadecimal
        metadata_path = object_root / "artifact.json"

        try:
            with self.workspace.write_locked():
                StorageQuota(self.workspace).require_capacity(staging=True)
                if metadata_path.exists():
                    content = self._read_metadata(metadata_path)
                    if content.artifact_id != artifact_id or content.byte_length != total:
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
                    staged_path.unlink(missing_ok=True)
                else:
                    object_root.mkdir(parents=True, exist_ok=True)
                    payload_path = object_root / payload_name
                    os.replace(staged_path, payload_path)
                    fsync_directory(object_root)
                    content = ArtifactContent(
                        artifact_id=artifact_id,
                        byte_length=total,
                        payload_name=payload_name,
                        integrity=Integrity(sha256=hexadecimal, hashed_at=utc_now()),
                    )
                    atomic_write_json(metadata_path, content.model_dump(mode="json"))
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

        return StoredArtifact(content=content, payload_path=payload_path)

    def get(self, artifact_id: str) -> StoredArtifact:
        hexadecimal = artifact_id.removeprefix("sha256:")
        if len(hexadecimal) != 64 or any(
            character not in "0123456789abcdef" for character in hexadecimal
        ):
            raise DomainError(ErrorCode.WORKSPACE_INVALID, "Invalid artifact identifier.")
        object_root = self.workspace.paths.artifacts / hexadecimal[:2] / hexadecimal
        content = self._read_metadata(object_root / "artifact.json")
        return StoredArtifact(
            content=content,
            payload_path=object_root / content.payload_name,
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
