from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import portalocker
import pytest

from flameox.domain import DomainError, ErrorCode
from flameox.http_transport import BoundedHttpClient, DownloadProgress
from flameox.managed_tools import (
    ManagedToolAsset,
    acquire_verified_asset,
    build_managed_tool_receipt,
    read_verified_tool_receipt,
    write_managed_tool_receipt,
)

pytestmark = pytest.mark.unit


class _InterruptedStream(httpx.SyncByteStream):
    def __init__(self, prefix: bytes) -> None:
        self.prefix = prefix

    def __iter__(self) -> Iterator[bytes]:
        yield self.prefix
        raise httpx.ReadTimeout("interrupted")


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
        max_bytes=max(1024, len(payload)),
        executable_sha256=hashlib.sha256(executable_payload).hexdigest(),
    )


def test_download_authenticates_manifest_digest_and_exact_length(tmp_path: Path) -> None:
    payload = b"authorized bytes"
    asset = _asset(payload)
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            headers={"content-length": str(len(payload)), "etag": '"fixture"'},
            stream=httpx.ByteStream(payload),
        )
    )

    with BoundedHttpClient(sync_transport=transport) as client:
        acquired = acquire_verified_asset(
            asset,
            tmp_path,
            http_client=client,
        )

    assert acquired.read_bytes() == payload
    assert acquired.parent.name == asset.sha256


def test_substituted_download_is_rejected_by_manifest_before_use(tmp_path: Path) -> None:
    asset = _asset(b"authorized bytes")
    substituted = b"substituted data"
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, stream=httpx.ByteStream(substituted))
    )

    with (
        BoundedHttpClient(sync_transport=transport) as client,
        pytest.raises(DomainError) as caught,
    ):
        acquire_verified_asset(
            asset,
            tmp_path,
            http_client=client,
        )

    assert caught.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert not tuple(tmp_path.rglob("*.partial"))
    assert not tuple(tmp_path.rglob("*.partial.json"))


def test_interrupted_download_resumes_from_bound_prefix(tmp_path: Path) -> None:
    prefix = b"a" * (64 * 1024)
    payload = prefix + b"authorized resumable tail"
    asset = _asset(payload)
    requests: list[httpx.Request] = []
    progress: list[DownloadProgress] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={"content-length": str(len(payload)), "etag": '"stable"'},
                stream=_InterruptedStream(prefix),
            )
        assert request.headers["range"] == f"bytes={len(prefix)}-"
        assert request.headers["if-range"] == '"stable"'
        return httpx.Response(
            206,
            headers={
                "content-length": str(len(payload) - len(prefix)),
                "content-range": f"bytes {len(prefix)}-{len(payload) - 1}/{len(payload)}",
                "etag": '"stable"',
            },
            stream=httpx.ByteStream(payload[len(prefix) :]),
        )

    with BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DomainError) as interrupted:
            acquire_verified_asset(
                asset,
                tmp_path,
                http_client=client,
                progress=progress.append,
            )
        acquired = acquire_verified_asset(asset, tmp_path, http_client=client)

    assert interrupted.value.details["received_bytes"] == len(prefix)
    assert interrupted.value.details["expected_bytes"] == len(payload)
    assert interrupted.value.details["resume_possible"] is True
    assert progress
    assert all(not update.resume_possible for update in progress)
    assert acquired.read_bytes() == payload
    assert len(requests) == 2


def test_resume_restarts_when_origin_returns_a_full_representation(tmp_path: Path) -> None:
    prefix = b"a" * (64 * 1024)
    payload = prefix + b"authorized replacement tail"
    asset = _asset(payload)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={"etag": '"old"'},
                stream=_InterruptedStream(prefix),
            )
        assert request.headers["range"] == f"bytes={len(prefix)}-"
        return httpx.Response(
            200,
            headers={"content-length": str(len(payload)), "etag": '"new"'},
            stream=httpx.ByteStream(payload),
        )

    with BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DomainError):
            acquire_verified_asset(asset, tmp_path, http_client=client)
        acquired = acquire_verified_asset(asset, tmp_path, http_client=client)

    assert acquired.read_bytes() == payload
    assert len(requests) == 2


def test_non_range_origin_restarts_without_claiming_resumability(tmp_path: Path) -> None:
    prefix = b"a" * (64 * 1024)
    payload = prefix + b"authorized non-range tail"
    asset = _asset(payload)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, stream=_InterruptedStream(prefix))
        assert "range" not in request.headers
        return httpx.Response(200, stream=httpx.ByteStream(payload))

    with BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DomainError) as interrupted:
            acquire_verified_asset(asset, tmp_path, http_client=client)
        acquired = acquire_verified_asset(asset, tmp_path, http_client=client)

    assert interrupted.value.details["resume_possible"] is False
    assert acquired.read_bytes() == payload
    assert len(requests) == 2


def test_cancelled_download_records_resumable_partial_state(tmp_path: Path) -> None:
    prefix = b"a" * (64 * 1024)
    payload = prefix + b"b" * (64 * 1024) + b"tail"
    asset = _asset(payload)
    requests: list[httpx.Request] = []
    cancel_checks = 0

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        offset = int(request.headers.get("range", "bytes=0-")[6:-1])
        status = 206 if offset else 200
        headers = {"etag": '"stable"'}
        if offset:
            headers["content-range"] = f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
        return httpx.Response(status, headers=headers, stream=httpx.ByteStream(payload[offset:]))

    def cancel() -> None:
        nonlocal cancel_checks
        cancel_checks += 1
        if cancel_checks == 4:
            raise DomainError(ErrorCode.PROCESS_CANCELLED, "cancelled")

    with BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DomainError) as cancelled:
            acquire_verified_asset(asset, tmp_path, http_client=client, cancel_check=cancel)
        acquired = acquire_verified_asset(asset, tmp_path, http_client=client)

    assert cancelled.value.code is ErrorCode.PROCESS_CANCELLED
    assert requests[1].headers["range"] == f"bytes={len(prefix)}-"
    assert acquired.read_bytes() == payload


def test_cancellation_before_first_resumed_byte_preserves_durable_partial(
    tmp_path: Path,
) -> None:
    prefix = b"a" * (64 * 1024)
    payload = prefix + b"trusted tail"
    asset = _asset(payload)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={"etag": '"stable"'},
                stream=_InterruptedStream(prefix),
            )
        return httpx.Response(
            206,
            headers={
                "content-range": f"bytes {len(prefix)}-{len(payload) - 1}/{len(payload)}",
                "etag": '"stable"',
            },
            stream=httpx.ByteStream(payload[len(prefix) :]),
        )

    cancel_checks = 0

    def cancel() -> None:
        nonlocal cancel_checks
        cancel_checks += 1
        if cancel_checks == 3:
            raise DomainError(ErrorCode.PROCESS_CANCELLED, "cancelled")

    with BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DomainError):
            acquire_verified_asset(asset, tmp_path, http_client=client)
        with pytest.raises(DomainError, match="cancelled"):
            acquire_verified_asset(asset, tmp_path, http_client=client, cancel_check=cancel)
        acquired = acquire_verified_asset(asset, tmp_path, http_client=client)

    expected_range = f"bytes={len(prefix)}-"
    assert requests[1].headers["range"] == expected_range
    assert requests[2].headers["range"] == expected_range
    assert acquired.read_bytes() == payload


def test_contradictory_content_range_does_not_mutate_durable_partial(tmp_path: Path) -> None:
    prefix = b"a" * (64 * 1024)
    payload = prefix + b"trusted tail"
    asset = _asset(payload)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={"etag": '"stable"'},
                stream=_InterruptedStream(prefix),
            )
        headers = {
            "content-range": f"bytes {len(prefix)}-{len(prefix)}/{len(payload)}",
            "etag": '"stable"',
        }
        if len(requests) == 2:
            return httpx.Response(
                206,
                headers=headers,
                stream=httpx.ByteStream(payload[len(prefix) :]),
            )
        return httpx.Response(
            206,
            headers={
                "content-range": f"bytes {len(prefix)}-{len(payload) - 1}/{len(payload)}",
                "etag": '"stable"',
            },
            stream=httpx.ByteStream(payload[len(prefix) :]),
        )

    with BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DomainError):
            acquire_verified_asset(asset, tmp_path, http_client=client)
        with pytest.raises(DomainError) as contradictory:
            acquire_verified_asset(asset, tmp_path, http_client=client)
        acquired = acquire_verified_asset(asset, tmp_path, http_client=client)

    assert contradictory.value.details["received_bytes"] == len(prefix)
    assert contradictory.value.details["resume_possible"] is True
    assert requests[2].headers["range"] == f"bytes={len(prefix)}-"
    assert acquired.read_bytes() == payload


def test_modified_partial_prefix_is_discarded_before_retry(tmp_path: Path) -> None:
    prefix = b"a" * (64 * 1024)
    payload = prefix + b"trusted tail"
    asset = _asset(payload)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={"etag": '"stable"'},
                stream=_InterruptedStream(prefix),
            )
        assert "range" not in request.headers
        return httpx.Response(200, stream=httpx.ByteStream(payload))

    with BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DomainError):
            acquire_verified_asset(asset, tmp_path, http_client=client)
        partial = next(tmp_path.rglob("*.partial"))
        partial.write_bytes(b"changed")
        acquired = acquire_verified_asset(asset, tmp_path, http_client=client)

    assert acquired.read_bytes() == payload


def test_changed_range_validator_discards_the_partial_response(tmp_path: Path) -> None:
    prefix = b"a" * (64 * 1024)
    payload = prefix + b"trusted tail"
    asset = _asset(payload)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={"etag": '"old"'},
                stream=_InterruptedStream(prefix),
            )
        if len(requests) == 2:
            return httpx.Response(
                206,
                headers={
                    "content-range": f"bytes {len(prefix)}-{len(payload) - 1}/{len(payload)}",
                    "etag": '"changed"',
                },
                stream=httpx.ByteStream(payload[len(prefix) :]),
            )
        assert "range" not in request.headers
        return httpx.Response(200, stream=httpx.ByteStream(payload))

    with BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DomainError):
            acquire_verified_asset(asset, tmp_path, http_client=client)
        with pytest.raises(DomainError) as changed:
            acquire_verified_asset(asset, tmp_path, http_client=client)
        acquired = acquire_verified_asset(asset, tmp_path, http_client=client)

    assert changed.value.details["resume_possible"] is False
    assert not tuple(tmp_path.rglob("*.partial.json"))
    assert acquired.read_bytes() == payload


@pytest.mark.skipif(os.name == "nt", reason="symlink creation is not generally available")
def test_managed_download_rejects_a_symlinked_storage_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "managed-downloads").symlink_to(outside, target_is_directory=True)
    asset = _asset(b"trusted")
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, content=b"trusted")

    with (
        BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client,
        pytest.raises(DomainError) as refused,
    ):
        acquire_verified_asset(asset, storage, http_client=client)

    assert refused.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert called is False


@pytest.mark.skipif(os.name == "nt", reason="symlink creation is not generally available")
def test_managed_download_rejects_a_symlinked_acquisition_lock(tmp_path: Path) -> None:
    asset = _asset(b"trusted")
    identity_root = tmp_path / "managed-downloads" / asset.sha256
    identity_root.mkdir(parents=True)
    outside = tmp_path / "outside.lock"
    outside.write_text("untrusted")
    (identity_root / ".acquire.lock").symlink_to(outside)
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, content=b"trusted")

    with (
        BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client,
        pytest.raises(DomainError) as refused,
    ):
        acquire_verified_asset(asset, tmp_path, http_client=client)

    assert refused.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert called is False
    assert outside.read_text() == "untrusted"


def test_cancellation_interrupts_acquisition_lock_contention(tmp_path: Path) -> None:
    asset = _asset(b"trusted")
    identity_root = tmp_path / "managed-downloads" / asset.sha256
    identity_root.mkdir(parents=True)
    checks = 0
    called = False

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks > 1:
            raise DomainError(ErrorCode.PROCESS_CANCELLED, "cancelled")

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, content=b"trusted")

    with (
        portalocker.Lock(identity_root / ".acquire.lock", mode="a", timeout=1),
        BoundedHttpClient(sync_transport=httpx.MockTransport(handler)) as client,
        pytest.raises(DomainError) as cancelled,
    ):
        acquire_verified_asset(
            asset,
            tmp_path,
            http_client=client,
            cancel_check=cancel,
        )

    assert cancelled.value.code is ErrorCode.PROCESS_CANCELLED
    assert called is False


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
