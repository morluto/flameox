from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import portalocker
from pydantic import Field, ValidationError, model_validator

from flameox.atomic import atomic_write_json
from flameox.domain import DomainError, ErrorCode
from flameox.filesystem import BoundedFileSystem
from flameox.http_transport import (
    BoundedHttpClient,
    BoundedHttpError,
    DownloadProgress,
    HttpFailureKind,
    ManagedDownloadRequest,
)
from flameox.models import ContractModel

_RECEIPT_LIMIT_BYTES = 64 * 1024
_MINIMUM_TRANSFER_BYTES_PER_SECOND = 32 * 1024
_DOWNLOAD_STARTUP_SECONDS = 30
_PROGRESS_INTERVAL_BYTES = 1024 * 1024
_PROGRESS_INTERVAL_SECONDS = 1.0


class ManagedToolAsset(ContractModel):
    """Reviewable identity of one authorized upstream artifact and executable."""

    manifest_revision: str = Field(min_length=1, max_length=200)
    tool: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,99}$")
    version: str = Field(min_length=1, max_length=100)
    platform: str = Field(min_length=1, max_length=50)
    machine: str = Field(min_length=1, max_length=50)
    asset_name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,200}$")
    url: str
    allowed_origins: tuple[str, ...] = Field(min_length=1, max_length=8)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(gt=0, le=1024 * 1024 * 1024)
    max_bytes: int = Field(gt=0, le=1024 * 1024 * 1024)
    max_redirects: int = Field(ge=0, le=4, default=0)
    executable_member: str | None = Field(default=None, max_length=300)
    executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def expected_asset_fits_download_budget(self) -> ManagedToolAsset:
        if self.byte_length > self.max_bytes:
            raise ValueError("managed-tool asset length exceeds its download budget")
        return self


class ManagedToolReceipt(ContractModel):
    manifest_revision: str
    asset_sha256: str
    executable_name: str
    executable_sha256: str


class _PartialDownload(ContractModel):
    asset_sha256: str
    expected_bytes: int
    received_bytes: int
    prefix_sha256: str
    validator: str | None = None


def acquire_verified_asset(
    asset: ManagedToolAsset,
    storage_root: Path,
    *,
    http_client: BoundedHttpClient,
    cancel_check: Callable[[], None] | None = None,
    progress: Callable[[DownloadProgress], None] | None = None,
) -> Path:
    """Acquire one authenticated asset with workspace-owned resumable partial state."""

    downloads_root = storage_root / "managed-downloads"
    if downloads_root.is_symlink():
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            "Managed download storage root is a symbolic link.",
        )
    downloads_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not downloads_root.resolve().is_relative_to(storage_root.resolve()):
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            "Managed download storage escaped its workspace root.",
        )
    identity_root = downloads_root / asset.sha256
    if identity_root.is_symlink():
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            "Managed download storage identity is a symbolic link.",
        )
    identity_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not identity_root.resolve().is_relative_to(downloads_root.resolve()):
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            "Managed download storage escaped its trusted root.",
        )
    try:
        with _acquisition_lock(identity_root, cancel_check=cancel_check):
            if cancel_check is not None:
                cancel_check()
            return _acquire_verified_asset_locked(
                asset,
                identity_root,
                http_client=http_client,
                cancel_check=cancel_check,
                progress=progress,
            )
    except portalocker.exceptions.LockException as error:
        raise DomainError(
            ErrorCode.CAPABILITY_UNAVAILABLE,
            f"Another process is acquiring the managed {asset.tool} asset.",
            retryable=True,
            details={"tool": asset.tool, "asset": asset.asset_name},
            remediation=("Retry setup after the active managed download finishes.",),
        ) from error


@contextmanager
def _acquisition_lock(
    directory: Path,
    *,
    cancel_check: Callable[[], None] | None,
) -> Iterator[None]:
    flags = os.O_CLOEXEC | os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    directory_descriptor = -1
    try:
        if os.open in os.supports_dir_fd and hasattr(os, "O_DIRECTORY"):
            directory_flags = os.O_CLOEXEC | os.O_DIRECTORY | os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                directory_flags |= os.O_NOFOLLOW
            directory_descriptor = os.open(directory, directory_flags)
            descriptor = os.open(
                ".acquire.lock",
                flags,
                0o600,
                dir_fd=directory_descriptor,
            )
        else:
            path = directory / ".acquire.lock"
            if path.is_symlink():
                raise OSError("acquisition lock is a symbolic link")
            descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            "Managed download acquisition lock is not a trusted regular file.",
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "Managed download acquisition lock is not a trusted regular file.",
            )
        with os.fdopen(descriptor, "a+") as stream:
            descriptor = -1
            deadline = time.monotonic() + 30
            while True:
                if cancel_check is not None:
                    cancel_check()
                try:
                    portalocker.lock(stream, portalocker.LOCK_EX | portalocker.LOCK_NB)
                    break
                except portalocker.exceptions.LockException:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
            try:
                yield
            finally:
                portalocker.unlock(stream)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)


def _acquire_verified_asset_locked(
    asset: ManagedToolAsset,
    identity_root: Path,
    *,
    http_client: BoundedHttpClient,
    cancel_check: Callable[[], None] | None,
    progress: Callable[[DownloadProgress], None] | None,
) -> Path:
    destination = identity_root / asset.asset_name
    partial = identity_root / f"{asset.asset_name}.partial"
    state_path = identity_root / f"{asset.asset_name}.partial.json"
    if _verified_asset_file(destination, asset, identity_root=identity_root):
        return destination
    if _verified_asset_file(partial, asset, identity_root=identity_root):
        os.replace(partial, destination)
        state_path.unlink(missing_ok=True)
        return destination

    state = _read_partial_download(state_path, identity_root=identity_root)
    if not _verified_partial(partial, state, asset, identity_root=identity_root):
        partial.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        state = None
    if state is not None and state.validator is None:
        partial.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        state = None

    latest: DownloadProgress | None = None
    reported_bytes = 0
    reported_elapsed = 0.0

    def report(update: DownloadProgress) -> None:
        nonlocal latest, reported_bytes, reported_elapsed
        latest = update
        should_report = (
            reported_bytes == 0
            or update.received_bytes == update.expected_bytes
            or update.received_bytes - reported_bytes >= _PROGRESS_INTERVAL_BYTES
            or update.elapsed_seconds - reported_elapsed >= _PROGRESS_INTERVAL_SECONDS
        )
        if progress is not None and should_report:
            progress(
                DownloadProgress(
                    received_bytes=update.received_bytes,
                    expected_bytes=update.expected_bytes,
                    elapsed_seconds=update.elapsed_seconds,
                    resume_possible=False,
                    validator=None,
                )
            )
            reported_bytes = update.received_bytes
            reported_elapsed = update.elapsed_seconds

    started = time.monotonic()
    deadline = (
        started
        + _DOWNLOAD_STARTUP_SECONDS
        + (asset.byte_length / _MINIMUM_TRANSFER_BYTES_PER_SECOND)
    )
    segment = identity_root / f"{asset.asset_name}.response"
    segment.unlink(missing_ok=True)
    durable_received = state.received_bytes if state is not None else 0
    durable_resume = state is not None
    try:
        with segment.open("w+b") as stream:
            try:
                receipt = http_client.download(
                    ManagedDownloadRequest(
                        url=asset.url,
                        allowed_origins=asset.allowed_origins,
                        deadline_monotonic=deadline,
                        max_response_bytes=asset.byte_length,
                        max_redirects=asset.max_redirects,
                        resume_from=state.received_bytes if state is not None else 0,
                        if_range=state.validator if state is not None else None,
                    ),
                    stream,
                    cancel_check=cancel_check,
                    progress=report,
                )
            except BaseException:
                stream.flush()
                os.fsync(stream.fileno())
                durable_received, durable_resume = _preserve_interrupted_response(
                    partial,
                    state_path,
                    segment,
                    asset,
                    state,
                    latest,
                    identity_root=identity_root,
                )
                raise
            stream.flush()
            os.fsync(stream.fileno())
    except BoundedHttpError as error:
        code = (
            ErrorCode.ARTIFACT_TOO_LARGE
            if error.kind is HttpFailureKind.RESPONSE_TOO_LARGE
            else ErrorCode.CAPABILITY_UNAVAILABLE
        )
        raise DomainError(
            code,
            f"The authorized {asset.tool} asset could not be downloaded within policy.",
            retryable=code is ErrorCode.CAPABILITY_UNAVAILABLE,
            details={
                "tool": asset.tool,
                "asset": asset.asset_name,
                "failure_category": (
                    "download_limit" if code is ErrorCode.ARTIFACT_TOO_LARGE else "network"
                ),
                "received_bytes": durable_received,
                "expected_bytes": asset.byte_length,
                "elapsed_seconds": round(max(0.0, time.monotonic() - started), 3),
                "resume_possible": durable_resume,
            },
            remediation=(
                "Retry capability setup; Flameox will resume the authenticated partial transfer."
                if durable_resume
                else "Retry capability setup; the origin did not provide a safe range validator, "
                "so Flameox will restart the transfer explicitly.",
            ),
        ) from error
    _merge_completed_response(
        partial,
        segment,
        state,
        receipt.response_start,
    )
    try:
        actual_sha256, actual_bytes = hash_regular_file(
            partial,
            trusted_root=identity_root,
            max_bytes=asset.max_bytes,
        )
    except DomainError:
        actual_sha256, actual_bytes = "unavailable", 0
    if actual_sha256 != asset.sha256 or actual_bytes != asset.byte_length:
        partial.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            f"The downloaded {asset.tool} asset did not match its trusted manifest.",
            details={
                "tool": asset.tool,
                "asset": asset.asset_name,
                "expected_sha256": asset.sha256,
                "actual_sha256": actual_sha256,
                "expected_byte_length": asset.byte_length,
                "actual_byte_length": actual_bytes,
            },
        )
    os.replace(partial, destination)
    state_path.unlink(missing_ok=True)
    return destination


def _verified_asset_file(
    path: Path,
    asset: ManagedToolAsset,
    *,
    identity_root: Path,
) -> bool:
    try:
        digest, size = hash_regular_file(
            path,
            trusted_root=identity_root,
            max_bytes=asset.max_bytes,
        )
    except DomainError:
        return False
    return digest == asset.sha256 and size == asset.byte_length


def _read_partial_download(path: Path, *, identity_root: Path) -> _PartialDownload | None:
    try:
        raw = BoundedFileSystem((identity_root,)).read_bytes(
            path,
            max_bytes=_RECEIPT_LIMIT_BYTES,
            require_single_link=True,
        )
        return _PartialDownload.model_validate_json(raw)
    except (DomainError, OSError, ValueError, ValidationError):
        return None


def _verified_partial(
    path: Path,
    state: _PartialDownload | None,
    asset: ManagedToolAsset,
    *,
    identity_root: Path,
) -> bool:
    if (
        state is None
        or state.asset_sha256 != asset.sha256
        or state.expected_bytes != asset.byte_length
        or not 0 < state.received_bytes < asset.byte_length
    ):
        return False
    try:
        digest, size = hash_regular_file(
            path,
            trusted_root=identity_root,
            max_bytes=asset.max_bytes,
        )
    except DomainError:
        return False
    return size == state.received_bytes and digest == state.prefix_sha256


def _preserve_interrupted_response(
    partial: Path,
    state_path: Path,
    segment: Path,
    asset: ManagedToolAsset,
    previous: _PartialDownload | None,
    latest: DownloadProgress | None,
    *,
    identity_root: Path,
) -> tuple[int, bool]:
    if latest is None:
        segment.unlink(missing_ok=True)
        return (previous.received_bytes, True) if previous is not None else (0, False)
    if not latest.resume_possible or latest.validator is None:
        segment.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return 0, False
    previous_bytes = previous.received_bytes if previous is not None else 0
    segment_bytes = segment.stat().st_size if segment.exists() else 0
    if latest.received_bytes != previous_bytes + segment_bytes:
        segment.unlink(missing_ok=True)
        return (previous.received_bytes, True) if previous is not None else (0, False)
    if previous is not None and latest.validator != previous.validator:
        segment.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return 0, False
    if previous is None:
        os.replace(segment, partial)
    else:
        with partial.open("ab") as destination, segment.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        segment.unlink(missing_ok=True)
    digest, size = hash_regular_file(
        partial,
        trusted_root=identity_root,
        max_bytes=asset.max_bytes,
    )
    if size == asset.byte_length and digest == asset.sha256:
        state_path.unlink(missing_ok=True)
        return size, True
    if size <= 0 or size >= asset.byte_length:
        partial.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return 0, False
    atomic_write_json(
        state_path,
        _PartialDownload(
            asset_sha256=asset.sha256,
            expected_bytes=asset.byte_length,
            received_bytes=size,
            prefix_sha256=digest,
            validator=latest.validator if latest.resume_possible else None,
        ).model_dump(mode="json"),
    )
    return size, True


def _merge_completed_response(
    partial: Path,
    segment: Path,
    previous: _PartialDownload | None,
    response_start: int,
) -> None:
    if response_start == 0:
        os.replace(segment, partial)
        return
    if previous is None or previous.received_bytes != response_start:
        segment.unlink(missing_ok=True)
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            "Managed download response did not match its durable partial state.",
        )
    with partial.open("ab") as destination, segment.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            destination.write(chunk)
        destination.flush()
        os.fsync(destination.fileno())
    segment.unlink(missing_ok=True)


def build_managed_tool_receipt(
    asset: ManagedToolAsset,
    executable: Path,
    *,
    trusted_root: Path,
    installed_name: str | None = None,
) -> ManagedToolReceipt:
    digest, _size = hash_regular_file(
        executable,
        trusted_root=trusted_root,
        max_bytes=asset.max_bytes,
    )
    if digest != asset.executable_sha256:
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            f"The authenticated {asset.tool} asset produced an unexpected executable.",
            details={
                "tool": asset.tool,
                "asset": asset.asset_name,
                "expected_executable_sha256": asset.executable_sha256,
                "actual_executable_sha256": digest,
            },
        )
    return ManagedToolReceipt(
        manifest_revision=asset.manifest_revision,
        asset_sha256=asset.sha256,
        executable_name=installed_name or executable.name,
        executable_sha256=digest,
    )


def write_managed_tool_receipt(path: Path, receipt: ManagedToolReceipt) -> None:
    atomic_write_json(path, receipt.model_dump(mode="json"))


def read_verified_tool_receipt(
    path: Path,
    executable: Path,
    asset: ManagedToolAsset,
    *,
    trusted_root: Path,
) -> ManagedToolReceipt | None:
    """Return a receipt only when immutable manifest facts and installed bytes agree."""

    try:
        raw = BoundedFileSystem((path.parent,)).read_bytes(
            path,
            max_bytes=_RECEIPT_LIMIT_BYTES,
            require_single_link=True,
        )
        receipt = ManagedToolReceipt.model_validate(json.loads(raw))
        expected = ManagedToolReceipt(
            manifest_revision=asset.manifest_revision,
            asset_sha256=asset.sha256,
            executable_name=executable.name,
            executable_sha256=asset.executable_sha256,
        )
        if receipt != expected:
            return None
        digest, _size = hash_regular_file(
            executable,
            trusted_root=trusted_root,
            max_bytes=asset.max_bytes,
        )
    except (DomainError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError):
        return None
    return receipt if digest == asset.executable_sha256 else None


def hash_regular_file(
    path: Path,
    *,
    trusted_root: Path,
    max_bytes: int,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    filesystem = BoundedFileSystem((trusted_root,))
    with filesystem.open_regular(
        path,
        max_bytes=max_bytes,
        require_single_link=True,
    ) as descriptor:
        while chunk := os.read(descriptor, 1024 * 1024):
            total += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), total
