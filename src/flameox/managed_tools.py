from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

from pydantic import Field, ValidationError, model_validator

from flameox.atomic import atomic_write_json
from flameox.domain import DomainError, ErrorCode
from flameox.filesystem import BoundedFileSystem
from flameox.http_transport import (
    BinarySink,
    BoundedHttpClient,
    BoundedHttpError,
    DownloadReceipt,
    HttpFailureKind,
    ManagedDownloadRequest,
)
from flameox.models import ContractModel

_RECEIPT_LIMIT_BYTES = 64 * 1024


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
    schema_version: int = Field(default=1, ge=1, le=1)
    manifest_revision: str
    tool: str
    version: str
    platform: str
    machine: str
    asset_name: str
    asset_sha256: str
    asset_byte_length: int
    executable_name: str
    executable_sha256: str


def download_verified_asset(
    asset: ManagedToolAsset,
    destination: BinarySink,
    *,
    http_client: BoundedHttpClient,
    deadline_monotonic: float,
    cancel_check: Callable[[], None] | None = None,
) -> DownloadReceipt:
    """Download once and authenticate bytes against checked-in manifest facts."""

    try:
        receipt = http_client.download(
            ManagedDownloadRequest(
                url=asset.url,
                allowed_origins=asset.allowed_origins,
                deadline_monotonic=deadline_monotonic,
                max_response_bytes=asset.max_bytes,
                max_redirects=asset.max_redirects,
            ),
            destination,
            cancel_check=cancel_check,
        )
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
            },
        ) from error
    if receipt.sha256 != asset.sha256 or receipt.byte_length != asset.byte_length:
        raise DomainError(
            ErrorCode.ARTIFACT_INTEGRITY_FAILED,
            f"The downloaded {asset.tool} asset did not match its trusted manifest.",
            details={
                "tool": asset.tool,
                "asset": asset.asset_name,
                "expected_sha256": asset.sha256,
                "actual_sha256": receipt.sha256,
                "expected_byte_length": asset.byte_length,
                "actual_byte_length": receipt.byte_length,
            },
        )
    return receipt


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
        tool=asset.tool,
        version=asset.version,
        platform=asset.platform,
        machine=asset.machine,
        asset_name=asset.asset_name,
        asset_sha256=asset.sha256,
        asset_byte_length=asset.byte_length,
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
        expected = (
            asset.manifest_revision,
            asset.tool,
            asset.version,
            asset.platform,
            asset.machine,
            asset.asset_name,
            asset.sha256,
            asset.byte_length,
            executable.name,
            asset.executable_sha256,
        )
        actual = (
            receipt.manifest_revision,
            receipt.tool,
            receipt.version,
            receipt.platform,
            receipt.machine,
            receipt.asset_name,
            receipt.asset_sha256,
            receipt.asset_byte_length,
            receipt.executable_name,
            receipt.executable_sha256,
        )
        if actual != expected:
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
