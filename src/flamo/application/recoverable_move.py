from __future__ import annotations

import os
from pathlib import Path

from flamo.domain import DomainError, ErrorCode
from flamo.storage.atomic import fsync_directory


def validate_manifest_id(value: str, *, kind: str) -> None:
    if not value or "/" in value or "\\" in value or "\x00" in value:
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
