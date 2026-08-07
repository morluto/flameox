from __future__ import annotations

import os
from pathlib import Path

from flameox.atomic import fsync_directory
from flameox.domain import DomainError, ErrorCode


def lexical_path_beneath(
    root: Path,
    value: str,
    *,
    subject: str,
    error_code: ErrorCode,
) -> tuple[Path, Path]:
    """Validate a root-relative path without resolving symbolic links.

    Returning the lexical path is deliberate: if the final component is swapped
    to a symbolic link after validation, ``os.replace`` moves the link itself
    instead of the object it points to.
    """
    if not value or "\x00" in value or "\\" in value or ":" in value:
        raise DomainError(error_code, f"{subject} is not a valid root-relative path.")
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise DomainError(error_code, f"{subject} is not a valid root-relative path.")
    candidate = root
    if candidate.is_symlink() or bool(getattr(candidate, "is_junction", lambda: False)()):
        raise DomainError(
            error_code,
            f"{subject} must not contain symbolic links or junctions.",
        )
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink() or bool(getattr(candidate, "is_junction", lambda: False)()):
            raise DomainError(
                error_code,
                f"{subject} must not contain symbolic links or junctions.",
            )
    return candidate, relative


def validate_manifest_id(value: str, *, kind: str) -> None:
    if value in {"", ".", ".."} or "/" in value or "\\" in value or ":" in value or "\x00" in value:
        raise DomainError(ErrorCode.EXECUTION_REFUSED, f"Invalid {kind} ID.")


def move_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    fsync_directory(source.parent)
    fsync_directory(destination.parent)


def resume_move(
    source: Path,
    destination: Path,
    *,
    subject: str,
) -> bool:
    """Finish one journaled move, returning whether this call moved the path."""

    source_exists = source.exists()
    destination_exists = destination.exists()
    if source_exists and destination_exists:
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            f"Both source and destination exist for {subject}.",
        )
    if not source_exists and not destination_exists:
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            f"Neither source nor destination exists for {subject}.",
        )
    if destination_exists:
        return False
    move_path(source, destination)
    return True
