from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import httpx
import pytest

from flameox.domain import DomainError, ErrorCode
from flameox.http_transport import BoundedHttpClient
from flameox.managed_tools import (
    ManagedToolAsset,
    build_managed_tool_receipt,
    download_verified_asset,
    read_verified_tool_receipt,
    write_managed_tool_receipt,
)

pytestmark = pytest.mark.unit


def _asset(payload: bytes, *, executable: bytes | None = None) -> ManagedToolAsset:
    executable_payload = executable if executable is not None else payload
    return ManagedToolAsset(
        manifest_revision="upstream-manifest-commit",
        tool="test-tool",
        version="1.0",
        platform="linux",
        machine="x86_64",
        asset_name="test-tool.bin",
        url="https://downloads.example.com/test-tool.bin",
        allowed_origins=("https://downloads.example.com",),
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_length=len(payload),
        max_bytes=1024,
        executable_sha256=hashlib.sha256(executable_payload).hexdigest(),
    )


def test_download_authenticates_manifest_digest_and_exact_length() -> None:
    payload = b"authorized bytes"
    asset = _asset(payload)
    destination = io.BytesIO()
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, stream=httpx.ByteStream(payload))
    )

    with BoundedHttpClient(sync_transport=transport) as client:
        receipt = download_verified_asset(
            asset,
            destination,
            http_client=client,
            deadline_monotonic=time.monotonic() + 5,
        )

    assert destination.getvalue() == payload
    assert receipt.sha256 == asset.sha256
    assert receipt.byte_length == asset.byte_length


def test_substituted_download_is_rejected_by_manifest_before_use() -> None:
    asset = _asset(b"authorized bytes")
    substituted = b"substituted data"
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, stream=httpx.ByteStream(substituted))
    )

    with (
        BoundedHttpClient(sync_transport=transport) as client,
        pytest.raises(DomainError) as caught,
    ):
        download_verified_asset(
            asset,
            io.BytesIO(),
            http_client=client,
            deadline_monotonic=time.monotonic() + 5,
        )

    assert caught.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_extracted_executable_must_match_independent_manifest_identity(tmp_path: Path) -> None:
    asset = _asset(b"authenticated archive", executable=b"expected executable")
    executable = tmp_path / "tool"
    executable.write_bytes(b"unexpected executable")

    with pytest.raises(DomainError) as caught:
        build_managed_tool_receipt(asset, executable, trusted_root=tmp_path)

    assert caught.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED


def test_mutating_executable_and_receipt_cannot_override_manifest(tmp_path: Path) -> None:
    executable = tmp_path / "tool"
    executable.write_bytes(b"expected executable")
    asset = _asset(b"authenticated archive", executable=executable.read_bytes())
    receipt_path = tmp_path / "receipt.json"
    receipt = build_managed_tool_receipt(asset, executable, trusted_root=tmp_path)
    write_managed_tool_receipt(receipt_path, receipt)

    assert (
        read_verified_tool_receipt(
            receipt_path,
            executable,
            asset,
            trusted_root=tmp_path,
        )
        == receipt
    )

    executable.write_bytes(b"mutated executable")
    forged = receipt.validated_copy(
        update={"executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}
    )
    write_managed_tool_receipt(receipt_path, forged)

    assert (
        read_verified_tool_receipt(
            receipt_path,
            executable,
            asset,
            trusted_root=tmp_path,
        )
        is None
    )


def test_direct_managed_download_requests_are_owned_by_attestation_boundary() -> None:
    source_root = Path(__file__).parents[1] / "src" / "flameox"
    violations = [
        str(path.relative_to(source_root))
        for path in source_root.rglob("*.py")
        if path.name not in {"http_transport.py", "managed_tools.py"}
        and "ManagedDownloadRequest" in path.read_text()
    ]

    assert violations == []
