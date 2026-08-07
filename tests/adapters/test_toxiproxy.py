from __future__ import annotations

import io
import platform
import re
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
def test_toxiproxy_client_accepts_all_declared_toxic_shapes(
    monkeypatch: pytest.MonkeyPatch,
    toxic_type: str,
    attributes: dict[str, int],
) -> None:
    monkeypatch.setattr(
        ToxiproxyClient,
        "_request",
        lambda self, method, path, payload=None: {},
    )
    ToxiproxyClient().add_toxic(
        proxy="proxy",
        name="toxic",
        toxic_type=toxic_type,
        attributes=attributes,
    )


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


def test_toxiproxy_release_fixtures_have_real_sha256_digests() -> None:
    releases = _toxiproxy._RELEASES

    assert releases
    for (system, machine), (asset, digest, executable) in releases.items():
        assert system in {"linux", "darwin", "windows"}
        assert machine
        assert asset.startswith("toxiproxy_2.12.0_")
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert executable.startswith("toxiproxy-server")


@pytest.mark.parametrize("member_kind", ["traversal", "symlink"])
def test_toxiproxy_archive_extraction_rejects_unsafe_members(
    tmp_path: Path,
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
    archive.seek(0)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(fileobj=archive, mode="r:gz") as bundle, pytest.raises(DomainError) as error:
        _toxiproxy._safe_extract(bundle, extracted)
    assert error.value.code is ErrorCode.ARTIFACT_INTEGRITY_FAILED
