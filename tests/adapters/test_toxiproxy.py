from __future__ import annotations

import hashlib
import io
import json
import platform
import tarfile
from pathlib import Path

import httpx
import pytest

from flameox.adapters.toxiproxy import ToxiproxyClient, ToxiproxyToolManager
from flameox.domain import DomainError, ErrorCode
from flameox.http_transport import BoundedHttpClient
from flameox.managed_tools import ManagedToolAsset

pytestmark = pytest.mark.unit


def test_toxiproxy_client_shapes_proxy_and_toxic_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def request(
        self: ToxiproxyClient,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        calls.append((method, path, payload))
        return {"name": "proxy"}

    monkeypatch.setattr(ToxiproxyClient, "_request", request)
    client = ToxiproxyClient()
    client.create_proxy(name="proxy", listen="127.0.0.1:10001", upstream="127.0.0.1:10002")
    client.update_proxy("proxy", enabled=False)
    client.add_toxic(
        proxy="proxy",
        name="latency",
        toxic_type="latency",
        stream="downstream",
        toxicity=1.0,
        attributes={"latency": 25, "jitter": 0},
    )

    assert calls == [
        (
            "POST",
            "/proxies",
            {
                "name": "proxy",
                "listen": "127.0.0.1:10001",
                "upstream": "127.0.0.1:10002",
                "enabled": True,
            },
        ),
        ("PUT", "/proxies/proxy", {"enabled": False}),
        (
            "POST",
            "/proxies/proxy/toxics",
            {
                "name": "latency",
                "type": "latency",
                "stream": "downstream",
                "toxicity": 1.0,
                "attributes": {"latency": 25, "jitter": 0},
            },
        ),
    ]


@pytest.mark.parametrize(
    ("toxic_type", "attributes"),
    [
        ("latency", {"latency": 1, "jitter": 0}),
        ("timeout", {"timeout": 1}),
        ("reset_peer", {}),
        ("bandwidth", {"rate": 1}),
        ("slicer", {"average_size": 1, "size_variation": 0, "delay": 0}),
        ("limit_data", {"bytes": 1}),
        ("slow_close", {"delay": 1}),
    ],
)
def test_toxiproxy_client_sends_all_declared_toxic_shapes(
    monkeypatch: pytest.MonkeyPatch,
    toxic_type: str,
    attributes: dict[str, int],
) -> None:
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    def request(
        self: ToxiproxyClient,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        requests.append((method, path, payload))
        return {"name": "toxic"}

    monkeypatch.setattr(ToxiproxyClient, "_request", request)

    result = ToxiproxyClient().add_toxic(
        proxy="proxy",
        name="toxic",
        toxic_type=toxic_type,
        attributes=attributes,
    )

    assert result == {"name": "toxic"}
    assert requests == [
        (
            "POST",
            "/proxies/proxy/toxics",
            {
                "name": "toxic",
                "type": toxic_type,
                "stream": "downstream",
                "toxicity": 1.0,
                "attributes": attributes,
            },
        )
    ]


@pytest.mark.parametrize("endpoint", ["0.0.0.0:1", "example.test:1", "127.0.0.1:0"])
def test_toxiproxy_client_rejects_non_loopback_or_invalid_endpoints(endpoint: str) -> None:
    with pytest.raises(ValueError):
        ToxiproxyClient().create_proxy(
            name="proxy",
            listen=endpoint,
            upstream="127.0.0.1:10002",
        )


def test_toxiproxy_client_rejects_unknown_toxic() -> None:
    with pytest.raises(ValueError):
        ToxiproxyClient().add_toxic(proxy="proxy", name="toxic", toxic_type="unknown")


@pytest.mark.anyio
async def test_toxiproxy_control_has_native_async_bounded_transport() -> None:
    calls: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b'"2.12.0"'),
        )

    async with BoundedHttpClient(async_transport=httpx.MockTransport(handler)) as http_client:
        client = ToxiproxyClient(http_client=http_client)
        version = await client.version_async()

    assert version == "2.12.0"
    assert calls == [("GET", "/version")]


@pytest.mark.parametrize(
    ("system", "machine", "asset", "executable"),
    [
        ("Linux", "x86_64", "toxiproxy_2.12.0_linux_amd64.tar.gz", "toxiproxy-server"),
        ("Linux", "aarch64", "toxiproxy_2.12.0_linux_arm64.tar.gz", "toxiproxy-server"),
        ("Darwin", "x86_64", "toxiproxy_2.12.0_darwin_amd64.tar.gz", "toxiproxy-server"),
        ("Darwin", "arm64", "toxiproxy_2.12.0_darwin_arm64.tar.gz", "toxiproxy-server"),
        (
            "Windows",
            "AMD64",
            "toxiproxy_2.12.0_windows_amd64.tar.gz",
            "toxiproxy-server.exe",
        ),
    ],
)
def test_toxiproxy_release_matrix_covers_supported_platform_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    asset: str,
    executable: str,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)

    selected = ToxiproxyToolManager.release_for_host()

    assert selected is not None
    assert selected[0] == asset
    assert selected[2] == executable


def test_toxiproxy_reports_unsupported_platform_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "FreeBSD")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    assert ToxiproxyToolManager.release_for_host() is None


def _archive(member_name: str, content: bytes = b"toxiproxy-server") -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        member = tarfile.TarInfo(member_name)
        member.mode = 0o755
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    return archive.getvalue()


def _stage_release(
    monkeypatch: pytest.MonkeyPatch,
    archive: bytes,
    *,
    expected_digest: str | None = None,
) -> BoundedHttpClient:
    digest = expected_digest or hashlib.sha256(archive).hexdigest()
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            member = next(
                item for item in bundle.getmembers() if item.name.endswith("toxiproxy-server")
            )
            stream = bundle.extractfile(member)
            executable = stream.read() if stream is not None else b""
    except (StopIteration, tarfile.TarError):
        executable = b""
    asset = ManagedToolAsset(
        manifest_revision="test-manifest",
        tool="toxiproxy",
        version="2.12.0",
        platform="linux",
        machine="x86_64",
        asset_name="toxiproxy-test.tar.gz",
        url="https://github.com/Shopify/toxiproxy/releases/download/v2.12.0/toxiproxy-test.tar.gz",
        allowed_origins=("https://github.com",),
        sha256=digest,
        byte_length=len(archive),
        max_bytes=max(1024 * 1024, len(archive)),
        executable_member="toxiproxy-server",
        executable_sha256=hashlib.sha256(executable).hexdigest(),
    )
    monkeypatch.setattr(
        ToxiproxyToolManager,
        "_asset_for_host",
        staticmethod(lambda: asset),
    )

    def download(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/toxiproxy-test.tar.gz")
        return httpx.Response(200, stream=httpx.ByteStream(archive))

    return BoundedHttpClient(sync_transport=httpx.MockTransport(download))


def test_toxiproxy_stage_verifies_archive_and_publishes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive("release/toxiproxy-server")
    http_client = _stage_release(monkeypatch, archive)

    with http_client:
        manager = ToxiproxyToolManager(tmp_path, http_client=http_client)
        receipt = manager.stage()
        persisted = json.loads(manager.receipt_path.read_text())

    assert receipt == manager.staged_receipt()
    assert receipt.executable.read_bytes() == b"toxiproxy-server"
    assert receipt.executable.stat().st_mode & 0o111
    assert persisted["asset_sha256"] == hashlib.sha256(archive).hexdigest()
    assert persisted["executable_sha256"] == hashlib.sha256(b"toxiproxy-server").hexdigest()


def test_toxiproxy_stage_rejects_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive("toxiproxy-server")
    http_client = _stage_release(monkeypatch, archive, expected_digest="0" * 64)

    with http_client, pytest.raises(DomainError) as error:
        ToxiproxyToolManager(tmp_path, http_client=http_client).stage()

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert error.value.details["expected_sha256"] == "0" * 64
    assert error.value.details["actual_sha256"] == hashlib.sha256(archive).hexdigest()


def test_toxiproxy_download_is_bounded_before_digest_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = b"x" * 800
    asset = ManagedToolAsset(
        manifest_revision="test-manifest",
        tool="toxiproxy",
        version="2.12.0",
        platform="linux",
        machine="x86_64",
        asset_name="release.tar.gz",
        url="https://github.com/Shopify/toxiproxy/releases/download/v2.12.0/release.tar.gz",
        allowed_origins=("https://github.com",),
        sha256=hashlib.sha256(expected).hexdigest(),
        byte_length=len(expected),
        max_bytes=1_024,
        executable_member="toxiproxy-server",
        executable_sha256="0" * 64,
    )
    monkeypatch.setattr(
        ToxiproxyToolManager,
        "_asset_for_host",
        staticmethod(lambda: asset),
    )
    http_client = BoundedHttpClient(
        sync_transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, stream=httpx.ByteStream(b"x" * 1_600))
        )
    )

    with http_client, pytest.raises(DomainError) as error:
        ToxiproxyToolManager(tmp_path, http_client=http_client).stage()

    assert error.value.code is ErrorCode.ARTIFACT_TOO_LARGE
    assert not any((tmp_path / "staging").iterdir())


@pytest.mark.parametrize("member_kind", ["traversal", "symlink"])
def test_toxiproxy_stage_rejects_unsafe_archive_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_kind: str,
) -> None:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        member = tarfile.TarInfo("../outside" if member_kind == "traversal" else "link")
        if member_kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
        else:
            member.size = 1
        bundle.addfile(member, io.BytesIO(b"x") if member.size else None)
    archive_bytes = archive.getvalue()
    http_client = _stage_release(monkeypatch, archive_bytes)

    with http_client, pytest.raises(DomainError) as error:
        ToxiproxyToolManager(tmp_path, http_client=http_client).stage()

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert not (tmp_path / "tools" / "toxiproxy-server").exists()
