from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import platform
import re
import tarfile
import tempfile
from dataclasses import dataclass
from http.client import HTTPResponse
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import portalocker

from flameox.atomic import atomic_write_json
from flameox.domain import DomainError, ErrorCode


class ToxiproxyApiError(RuntimeError):
    pass


TOXIPROXY_VERSION = "2.12.0"
_RELEASE_URL = f"https://github.com/Shopify/toxiproxy/releases/download/v{TOXIPROXY_VERSION}/"
_MAX_RELEASE_BYTES = 128 * 1024 * 1024
_RELEASES: dict[tuple[str, str], tuple[str, str, str]] = {
    ("linux", "x86_64"): (
        "toxiproxy_2.12.0_linux_amd64.tar.gz",
        "65e042d3fc2290c099527bc506446a6fd863a09f113b8944b636ece70221c2b3",
        "toxiproxy-server",
    ),
    ("linux", "aarch64"): (
        "toxiproxy_2.12.0_linux_arm64.tar.gz",
        "ab417dafc77ae8b54ec8461cc5da5180d0465a61dcbda86c72da72ee4ab843b7",
        "toxiproxy-server",
    ),
    ("darwin", "x86_64"): (
        "toxiproxy_2.12.0_darwin_amd64.tar.gz",
        "52ac1f99d7204b4b523f910b791bc86318b50ef94822c79a6157606704003ced",
        "toxiproxy-server",
    ),
    ("darwin", "arm64"): (
        "toxiproxy_2.12.0_darwin_arm64.tar.gz",
        "cece3905c06ad84d6b3bb4cd024c344e6466736b2a8cbaf9ef4fe45e62aba9b2",
        "toxiproxy-server",
    ),
    ("windows", "AMD64"): (
        "toxiproxy_2.12.0_windows_amd64.tar.gz",
        "fad5a5be0b5eedf4fd2dd7d36ff26d0508149eb1b14457e40c7a203a219304dd",
        "toxiproxy-server-windows-amd64.exe",
    ),
}


@dataclass(frozen=True, slots=True)
class ToxiproxyToolReceipt:
    version: str
    asset: str
    sha256: str
    executable: Path


class ToxiproxyToolManager:
    """Stage only the pinned, checksummed Toxiproxy server release asset."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.tools_root = workspace_root / "tools"
        self.receipt_path = self.tools_root / "toxiproxy-receipt.json"

    @staticmethod
    def release_for_host() -> tuple[str, str, str] | None:
        system = platform.system().lower()
        machine = platform.machine()
        return _RELEASES.get((system, machine))

    def staged_receipt(self) -> ToxiproxyToolReceipt | None:
        release = self.release_for_host()
        if release is None:
            return None
        asset, expected_digest, _ = release
        executable_name = "toxiproxy-server.exe" if os.name == "nt" else "toxiproxy-server"
        executable = self.tools_root / executable_name
        if not executable.is_file() or not self._receipt_matches(
            asset, expected_digest, executable
        ):
            return None
        return ToxiproxyToolReceipt(TOXIPROXY_VERSION, asset, expected_digest, executable)

    def stage(self) -> ToxiproxyToolReceipt:
        self.tools_root.mkdir(parents=True, exist_ok=True)
        try:
            with portalocker.Lock(self.tools_root / ".toxiproxy-stage.lock", mode="a", timeout=30):
                return self._stage_locked()
        except portalocker.exceptions.LockException as error:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "Toxiproxy staging is busy with another setup operation.",
                retryable=True,
                remediation=("Retry capability setup after the other setup operation finishes.",),
            ) from error

    def _stage_locked(self) -> ToxiproxyToolReceipt:
        release = self.release_for_host()
        if release is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "Toxiproxy has no managed release asset for this platform.",
                details={"version": TOXIPROXY_VERSION},
            )
        asset, expected_digest, member_name = release
        executable_name = "toxiproxy-server.exe" if os.name == "nt" else "toxiproxy-server"
        executable = self.tools_root / executable_name
        existing = self.staged_receipt()
        if existing is not None:
            return existing

        staging_root = self.workspace_root / "staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=staging_root) as temporary:
            archive = Path(temporary) / asset
            try:
                with (
                    urlopen(_RELEASE_URL + asset, timeout=120) as response,
                    archive.open("wb") as stream,
                ):
                    downloaded = 0
                    while chunk := response.read(1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > _MAX_RELEASE_BYTES:
                            raise DomainError(
                                ErrorCode.ARTIFACT_TOO_LARGE,
                                "The managed Toxiproxy download exceeded its size limit.",
                                details={"asset": asset, "limit_bytes": _MAX_RELEASE_BYTES},
                            )
                        stream.write(chunk)
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "The pinned Toxiproxy asset could not be downloaded.",
                    retryable=True,
                    details={"asset": asset, "url": _RELEASE_URL + asset},
                    remediation=("Retry capability setup when network access is available.",),
                ) from error
            actual_digest = _sha256(archive)
            if actual_digest != expected_digest:
                raise DomainError(
                    ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                    "The downloaded Toxiproxy asset failed SHA-256 verification.",
                    details={"asset": asset, "expected": expected_digest, "actual": actual_digest},
                )
            extracted = Path(temporary) / "extracted"
            extracted.mkdir()
            with tarfile.open(archive, "r:gz") as bundle:
                _safe_extract(bundle, extracted)
            source = next((path for path in extracted.rglob(member_name) if path.is_file()), None)
            if source is None:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "The verified Toxiproxy archive did not contain its server binary.",
                    details={"asset": asset, "member": member_name},
                )
            staged = self.tools_root / f".{executable.name}.staged"
            staged.write_bytes(source.read_bytes())
            staged.chmod(0o755)
            os.replace(staged, executable)
        atomic_write_json(
            self.receipt_path,
            {
                "schema_version": 1,
                "version": TOXIPROXY_VERSION,
                "asset": asset,
                "sha256": expected_digest,
                "executable_sha256": _sha256(executable),
                "executable": str(executable),
            },
        )
        return ToxiproxyToolReceipt(TOXIPROXY_VERSION, asset, expected_digest, executable)

    def _receipt_matches(self, asset: str, digest: str, executable: Path) -> bool:
        try:
            receipt = json.loads(self.receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return bool(
            receipt.get("version") == TOXIPROXY_VERSION
            and receipt.get("asset") == asset
            and receipt.get("sha256") == digest
            and receipt.get("executable_sha256") == _sha256(executable)
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(bundle: tarfile.TarFile, target: Path) -> None:
    root = target.resolve()
    for member in bundle.getmembers():
        if member.issym() or member.islnk():
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "The Toxiproxy archive contains a link entry.",
                details={"member": member.name},
            )
        destination = (target / member.name).resolve()
        try:
            destination.relative_to(root)
        except ValueError as error:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "The Toxiproxy archive contains an unsafe path.",
                details={"member": member.name},
            ) from error
    bundle.extractall(target)


@dataclass(frozen=True, slots=True)
class ToxiproxyClient:
    """Small standard-library client for the managed Toxiproxy control API."""

    base_url: str = "http://127.0.0.1:8474"
    timeout_seconds: float = 5.0

    def version(self) -> str:
        value = self._request("GET", "/version")
        if isinstance(value, str):
            return value
        return str(value.get("version", "")) if isinstance(value, dict) else ""

    def health(self) -> bool:
        try:
            self.version()
        except ToxiproxyApiError:
            return False
        return True

    def create_proxy(
        self, *, name: str, listen: str, upstream: str, enabled: bool = True
    ) -> dict[str, Any]:
        _require_name(name)
        _require_loopback_endpoint(listen)
        _require_loopback_endpoint(upstream)
        return self._object(
            self._request(
                "POST",
                "/proxies",
                {
                    "name": name,
                    "listen": listen,
                    "upstream": upstream,
                    "enabled": enabled,
                },
            )
        )

    def get_proxy(self, name: str) -> dict[str, Any]:
        _require_name(name)
        return self._object(self._request("GET", f"/proxies/{name}"))

    def update_proxy(self, name: str, *, enabled: bool) -> dict[str, Any]:
        _require_name(name)
        return self._object(self._request("PUT", f"/proxies/{name}", {"enabled": enabled}))

    def list_proxies(self) -> list[dict[str, Any]]:
        value = self._request("GET", "/proxies")
        if not isinstance(value, list):
            raise ToxiproxyApiError("Toxiproxy returned a non-list proxy collection")
        return [self._object(item) for item in value]

    def delete_proxy(self, name: str) -> None:
        _require_name(name)
        self._request("DELETE", f"/proxies/{name}")

    def add_toxic(
        self,
        *,
        proxy: str,
        name: str,
        toxic_type: str,
        stream: str = "downstream",
        toxicity: float = 1.0,
        attributes: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        _require_name(proxy)
        _require_name(name)
        if stream not in {"upstream", "downstream"}:
            raise ValueError("stream must be upstream or downstream")
        if not 0 <= toxicity <= 1:
            raise ValueError("toxicity must be between 0 and 1")
        if toxic_type not in {
            "latency",
            "timeout",
            "reset_peer",
            "bandwidth",
            "slicer",
            "limit_data",
            "slow_close",
        }:
            raise ValueError("unsupported Toxiproxy toxic type")
        if attributes is not None and len(attributes) > 8:
            raise ValueError("toxic attributes are bounded")
        if attributes is not None and any(
            not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", key)
            or not isinstance(value, int)
            or value < 0
            or value > 10_000_000_000
            for key, value in attributes.items()
        ):
            raise ValueError("toxic attributes must be bounded non-negative integers")
        return self._object(
            self._request(
                "POST",
                f"/proxies/{proxy}/toxics",
                {
                    "name": name,
                    "type": toxic_type,
                    "stream": stream,
                    "toxicity": toxicity,
                    "attributes": attributes or {},
                },
            )
        )

    def remove_toxic(self, *, proxy: str, name: str) -> None:
        _require_name(proxy)
        _require_name(name)
        self._request("DELETE", f"/proxies/{proxy}/toxics/{name}")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        request = Request(
            self.base_url.rstrip("/") + path,
            method=method,
            data=(
                json.dumps(payload, separators=(",", ":")).encode() if payload is not None else None
            ),
            headers={"Content-Type": "application/json"} if payload is not None else {},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return _decode(response)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise ToxiproxyApiError(f"Toxiproxy {method} {path} failed") from error

    @staticmethod
    def _object(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ToxiproxyApiError("Toxiproxy returned a non-object")
        return value


def _decode(response: HTTPResponse) -> Any:
    raw = response.read()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.decode("utf-8", errors="replace")


def _require_name(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value):
        raise ValueError("Toxiproxy names must be bounded simple identifiers")


def _require_loopback_endpoint(value: str) -> None:
    try:
        host, port_text = value.rsplit(":", 1)
        address = ipaddress.ip_address(host.strip("[]"))
        port = int(port_text)
    except (ValueError, TypeError) as error:
        raise ValueError("Toxiproxy endpoints must be loopback host:port values") from error
    if not address.is_loopback or not 1 <= port <= 65_535:
        raise ValueError("Toxiproxy endpoints must use a loopback address and valid port")
