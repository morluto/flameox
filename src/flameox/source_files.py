"""Native input identity and bounded filesystem traversal shared by runtime and storage."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flameox.runtime_contracts import RuntimeFailure


@dataclass(slots=True)
class NativeSource:
    path: Path
    sha256: str
    size_bytes: int
    format: str
    producer: str | None
    role: str

    def public(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "format": self.format,
            "producer": self.producer,
            "role": self.role,
        }


def bundle_digest(members: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for relative, sha256 in sorted(members, key=lambda item: Path(item[0])):
        digest.update(relative.encode() + bytes.fromhex(sha256))
    return digest.hexdigest()


def sha256_file(path: Path, *, max_bytes: int | None = None) -> tuple[str, int]:
    if not path.is_file():
        raise RuntimeFailure("INVALID_INPUT", "Source must be a regular file")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise RuntimeFailure("LIMIT_EXCEEDED", "Input exceeds max_input_bytes")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(
            1024 * 1024 if max_bytes is None else min(1024 * 1024, max_bytes - size + 1)
        ):
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise RuntimeFailure("LIMIT_EXCEEDED", "Input exceeds max_input_bytes")
            digest.update(chunk)
    return digest.hexdigest(), size


def copy_verified_file(source: NativeSource, target: Path) -> None:
    """Copy no more than the admitted bytes, rejecting changed native inputs."""
    digest = hashlib.sha256()
    remaining = source.size_bytes
    with source.path.open("rb") as reader, target.open("xb") as writer:
        while remaining:
            chunk = reader.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            writer.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if remaining or reader.read(1) or digest.hexdigest() != source.sha256:
            raise RuntimeFailure("MISSING_OR_CHANGED_INPUT", "Input changed during copying")


def directory_files(path: Path, *, max_files: int | None = None) -> list[Path]:
    files: list[Path] = []
    for item in path.rglob("*"):
        if item.is_symlink():
            raise RuntimeFailure(
                "INVALID_INPUT", f"Directory sources cannot contain symlinks: {item}"
            )
        if not item.is_file() and not item.is_dir():
            raise RuntimeFailure(
                "INVALID_INPUT", f"Directory sources cannot contain special files: {item}"
            )
        if item.is_file():
            files.append(item)
            if max_files is not None and len(files) > max_files:
                raise RuntimeFailure("LIMIT_EXCEEDED", "Input exceeds max_input_files")
    return sorted(files)


def hash_path(
    path: Path, *, max_bytes: int | None = None, max_files: int | None = None
) -> tuple[str, int, int]:
    if not path.is_file() and not path.is_dir():
        raise RuntimeFailure("INVALID_INPUT", f"Source is not a regular file or directory: {path}")
    if path.is_file():
        if max_files is not None and max_files < 1:
            raise RuntimeFailure("LIMIT_EXCEEDED", "Input exceeds max_input_files")
        digest, size = sha256_file(path, max_bytes=max_bytes)
        return digest, size, 1
    members: list[tuple[str, str]] = []
    size = 0
    files = directory_files(path, max_files=max_files)
    for item in files:
        digest, item_size = sha256_file(
            item, max_bytes=None if max_bytes is None else max_bytes - size
        )
        members.append((item.relative_to(path).as_posix(), digest))
        size += item_size
    return bundle_digest(members), size, len(files)
