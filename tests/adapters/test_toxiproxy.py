from __future__ import annotations

import hashlib
import io
import json
import platform
import tarfile
from pathlib import Path

import pytest

from flameox.adapters import toxiproxy as _toxiproxy
from flameox.adapters.toxiproxy import ToxiproxyClient, ToxiproxyToolManager
from flameox.domain import DomainError, ErrorCode


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
            "toxiproxy-server-windows-amd64.exe",
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
) -> None:
    digest = expected_digest or hashlib.sha256(archive).hexdigest()
    monkeypatch.setattr(
        ToxiproxyToolManager,
        "release_for_host",
        staticmethod(lambda: ("toxiproxy-test.tar.gz", digest, "toxiproxy-server")),
    )

    def open_archive(url: str, timeout: float) -> io.BytesIO:
        assert url.endswith("/toxiproxy-test.tar.gz")
        assert timeout == 120
        return io.BytesIO(archive)

    monkeypatch.setattr(_toxiproxy, "urlopen", open_archive)


def test_toxiproxy_stage_verifies_archive_and_publishes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive("release/toxiproxy-server")
    _stage_release(monkeypatch, archive)

    manager = ToxiproxyToolManager(tmp_path)
    receipt = manager.stage()
    persisted = json.loads(manager.receipt_path.read_text())

    assert receipt == manager.staged_receipt()
    assert receipt.executable.read_bytes() == b"toxiproxy-server"
    assert receipt.executable.stat().st_mode & 0o111
    assert persisted["sha256"] == hashlib.sha256(archive).hexdigest()
    assert persisted["executable_sha256"] == hashlib.sha256(b"toxiproxy-server").hexdigest()


def test_toxiproxy_stage_rejects_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive("toxiproxy-server")
    _stage_release(monkeypatch, archive, expected_digest="0" * 64)

    with pytest.raises(DomainError) as error:
        ToxiproxyToolManager(tmp_path).stage()

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert error.value.details["expected"] == "0" * 64
    assert error.value.details["actual"] == hashlib.sha256(archive).hexdigest()


def test_toxiproxy_download_is_bounded_before_digest_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedResponse:
        chunks = 0

        def __enter__(self) -> OversizedResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            self.chunks += 1
            return b"x" * 800 if self.chunks <= 2 else b""

    monkeypatch.setattr(_toxiproxy, "_MAX_RELEASE_BYTES", 1_024, raising=False)
    monkeypatch.setattr(
        ToxiproxyToolManager,
        "release_for_host",
        staticmethod(lambda: ("release.tar.gz", "0" * 64, "toxiproxy-server")),
    )
    monkeypatch.setattr(_toxiproxy, "urlopen", lambda *_args, **_kwargs: OversizedResponse())

    with pytest.raises(DomainError) as error:
        ToxiproxyToolManager(tmp_path).stage()

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
    _stage_release(monkeypatch, archive_bytes)

    with pytest.raises(DomainError) as error:
        ToxiproxyToolManager(tmp_path).stage()

    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
    assert not (tmp_path / "tools" / "toxiproxy-server").exists()
