from __future__ import annotations

import ipaddress
import os
import platform
import re
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import portalocker
from pydantic import JsonValue, TypeAdapter, ValidationError

from flameox.domain import DomainError, ErrorCode
from flameox.http_transport import (
    BoundedHttpClient,
    BoundedHttpError,
    HttpMethod,
    LoopbackHttpRequest,
    validate_loopback_base_url,
)
from flameox.managed_tools import (
    ManagedToolAsset,
    build_managed_tool_receipt,
    download_verified_asset,
    read_verified_tool_receipt,
    write_managed_tool_receipt,
)


class ToxiproxyApiError(RuntimeError):
    pass


TOXIPROXY_VERSION = "2.12.0"
_RELEASE_URL = f"https://github.com/Shopify/toxiproxy/releases/download/v{TOXIPROXY_VERSION}/"
_MAX_RELEASE_BYTES = 128 * 1024 * 1024
_RELEASE_ORIGINS = (
    "https://github.com",
    "https://release-assets.githubusercontent.com",
)
_OBJECT_RESPONSE = TypeAdapter(dict[str, JsonValue])
_OBJECT_LIST_RESPONSE = TypeAdapter(list[dict[str, JsonValue]])
_TOXIPROXY_MANIFEST_REVISION = "github-release:Shopify/toxiproxy@v2.12.0"


def _toxiproxy_asset(
    platform_name: str,
    machine: str,
    asset_name: str,
    asset_sha256: str,
    byte_length: int,
    executable_name: str,
    executable_sha256: str,
) -> ManagedToolAsset:
    return ManagedToolAsset(
        manifest_revision=_TOXIPROXY_MANIFEST_REVISION,
        tool="toxiproxy",
        version=TOXIPROXY_VERSION,
        platform=platform_name,
        machine=machine,
        asset_name=asset_name,
        url=_RELEASE_URL + asset_name,
        allowed_origins=_RELEASE_ORIGINS,
        sha256=asset_sha256,
        byte_length=byte_length,
        max_bytes=_MAX_RELEASE_BYTES,
        max_redirects=2,
        executable_member=executable_name,
        executable_sha256=executable_sha256,
    )


_RELEASES: dict[tuple[str, str], ManagedToolAsset] = {
    ("linux", "x86_64"): _toxiproxy_asset(
        "linux",
        "x86_64",
        "toxiproxy_2.12.0_linux_amd64.tar.gz",
        "65e042d3fc2290c099527bc506446a6fd863a09f113b8944b636ece70221c2b3",
        7_545_222,
        "toxiproxy-server",
        "556d891134a3c582dc1e1a3f7335fd55142e5965769855a00b944e13e48302fc",
    ),
    ("linux", "aarch64"): _toxiproxy_asset(
        "linux",
        "aarch64",
        "toxiproxy_2.12.0_linux_arm64.tar.gz",
        "ab417dafc77ae8b54ec8461cc5da5180d0465a61dcbda86c72da72ee4ab843b7",
        6_965_624,
        "toxiproxy-server",
        "53e770c1c3035b5a9f1bc629fce537db1f95f62b26f4ebe6e756afd701cf077c",
    ),
    ("darwin", "x86_64"): _toxiproxy_asset(
        "darwin",
        "x86_64",
        "toxiproxy_2.12.0_darwin_amd64.tar.gz",
        "52ac1f99d7204b4b523f910b791bc86318b50ef94822c79a6157606704003ced",
        7_633_338,
        "toxiproxy-server",
        "9625bba4bd96117eedae49f982aba4c2f462b268dd406c9ff18186f9b1ef8afe",
    ),
    ("darwin", "arm64"): _toxiproxy_asset(
        "darwin",
        "arm64",
        "toxiproxy_2.12.0_darwin_arm64.tar.gz",
        "cece3905c06ad84d6b3bb4cd024c344e6466736b2a8cbaf9ef4fe45e62aba9b2",
        7_193_129,
        "toxiproxy-server",
        "aa299966b52f16a8594f1cd0d1e9049dc2e8fe2c04a90c19860e2719b2b95d15",
    ),
    ("windows", "AMD64"): _toxiproxy_asset(
        "windows",
        "AMD64",
        "toxiproxy_2.12.0_windows_amd64.tar.gz",
        "fad5a5be0b5eedf4fd2dd7d36ff26d0508149eb1b14457e40c7a203a219304dd",
        7_691_600,
        "toxiproxy-server.exe",
        "e32f4cb58e62e844d4b5d47613525a5bdeea15c39fa22c9f413c756275ac0ba8",
    ),
}


@dataclass(frozen=True, slots=True)
class ToxiproxyToolReceipt:
    version: str
    asset: str
    sha256: str
    executable: Path
    executable_sha256: str
    manifest_revision: str


class ToxiproxyToolManager:
    """Stage only the pinned, checksummed Toxiproxy server release asset."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        http_client: BoundedHttpClient | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.tools_root = workspace_root / "tools"
        self.receipt_path = self.tools_root / "toxiproxy-receipt.json"
        self._http_client = http_client

    @staticmethod
    def release_for_host() -> tuple[str, str, str] | None:
        asset = ToxiproxyToolManager._asset_for_host()
        if asset is None or asset.executable_member is None:
            return None
        return asset.asset_name, asset.sha256, asset.executable_member

    @staticmethod
    def _asset_for_host() -> ManagedToolAsset | None:
        system = platform.system().lower()
        machine = platform.machine()
        return _RELEASES.get((system, machine))

    def staged_receipt(self) -> ToxiproxyToolReceipt | None:
        asset = self._asset_for_host()
        if asset is None:
            return None
        executable_name = (
            "toxiproxy-server.exe" if asset.platform == "windows" else "toxiproxy-server"
        )
        executable = self.tools_root / executable_name
        receipt = read_verified_tool_receipt(
            self.receipt_path,
            executable,
            asset,
            trusted_root=self.tools_root,
        )
        if receipt is None:
            return None
        return ToxiproxyToolReceipt(
            TOXIPROXY_VERSION,
            asset.asset_name,
            asset.sha256,
            executable,
            receipt.executable_sha256,
            asset.manifest_revision,
        )

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
        asset = self._asset_for_host()
        if asset is None or asset.executable_member is None:
            raise DomainError(
                ErrorCode.CAPABILITY_UNAVAILABLE,
                "Toxiproxy has no managed release asset for this platform.",
                details={"version": TOXIPROXY_VERSION},
            )
        member_name = asset.executable_member
        executable_name = (
            "toxiproxy-server.exe" if asset.platform == "windows" else "toxiproxy-server"
        )
        executable = self.tools_root / executable_name
        existing = self.staged_receipt()
        if existing is not None:
            return existing

        staging_root = self.workspace_root / "staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=staging_root) as temporary:
            archive = Path(temporary) / asset.asset_name
            client = self._http_client or BoundedHttpClient()
            try:
                with archive.open("wb") as stream:
                    download_verified_asset(
                        asset,
                        stream,
                        http_client=client,
                        deadline_monotonic=time.monotonic() + 120,
                    )
            except OSError as error:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "The pinned Toxiproxy asset could not be staged.",
                    retryable=True,
                    details={"asset": asset.asset_name},
                ) from error
            finally:
                if self._http_client is None:
                    client.close()
            extracted = Path(temporary) / "extracted"
            extracted.mkdir()
            with tarfile.open(archive, "r:gz") as bundle:
                _safe_extract(bundle, extracted)
            source = next((path for path in extracted.rglob(member_name) if path.is_file()), None)
            if source is None:
                raise DomainError(
                    ErrorCode.CAPABILITY_UNAVAILABLE,
                    "The verified Toxiproxy archive did not contain its server binary.",
                    details={"asset": asset.asset_name, "member": member_name},
                )
            staged = self.tools_root / f".{executable.name}.staged"
            try:
                shutil.copyfile(source, staged)
                staged.chmod(0o755)
                receipt = build_managed_tool_receipt(
                    asset,
                    staged,
                    trusted_root=self.tools_root,
                    installed_name=executable.name,
                )
                os.replace(staged, executable)
            finally:
                staged.unlink(missing_ok=True)
        write_managed_tool_receipt(self.receipt_path, receipt)
        return ToxiproxyToolReceipt(
            TOXIPROXY_VERSION,
            asset.asset_name,
            asset.sha256,
            executable,
            receipt.executable_sha256,
            asset.manifest_revision,
        )


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
    bundle.extractall(target, filter="data")


@dataclass(frozen=True, slots=True)
class ToxiproxyClient:
    """Typed provider client over the bounded loopback HTTP authority."""

    base_url: str = "http://127.0.0.1:8474"
    timeout_seconds: float = 5.0
    http_client: BoundedHttpClient | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", validate_loopback_base_url(self.base_url))
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def version(self) -> str:
        value = self._request("GET", "/version")
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            version = value.get("version")
            return version if isinstance(version, str) else ""
        return ""

    async def version_async(self) -> str:
        value = await self._request_async("GET", "/version")
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            version = value.get("version")
            return version if isinstance(version, str) else ""
        return ""

    def health(self) -> bool:
        try:
            self.version()
        except ToxiproxyApiError:
            return False
        return True

    async def health_async(self) -> bool:
        try:
            await self.version_async()
        except ToxiproxyApiError:
            return False
        return True

    def create_proxy(
        self, *, name: str, listen: str, upstream: str, enabled: bool = True
    ) -> dict[str, JsonValue]:
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

    async def create_proxy_async(
        self,
        *,
        name: str,
        listen: str,
        upstream: str,
        enabled: bool = True,
    ) -> dict[str, JsonValue]:
        _require_name(name)
        _require_loopback_endpoint(listen)
        _require_loopback_endpoint(upstream)
        return self._object(
            await self._request_async(
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

    def get_proxy(self, name: str) -> dict[str, JsonValue]:
        _require_name(name)
        return self._object(self._request("GET", f"/proxies/{name}"))

    async def get_proxy_async(self, name: str) -> dict[str, JsonValue]:
        _require_name(name)
        return self._object(await self._request_async("GET", f"/proxies/{name}"))

    def update_proxy(self, name: str, *, enabled: bool) -> dict[str, JsonValue]:
        _require_name(name)
        return self._object(self._request("PUT", f"/proxies/{name}", {"enabled": enabled}))

    async def update_proxy_async(self, name: str, *, enabled: bool) -> dict[str, JsonValue]:
        _require_name(name)
        return self._object(
            await self._request_async("PUT", f"/proxies/{name}", {"enabled": enabled})
        )

    def list_proxies(self) -> list[dict[str, JsonValue]]:
        value = self._request("GET", "/proxies")
        return self._object_list(value)

    async def list_proxies_async(self) -> list[dict[str, JsonValue]]:
        return self._object_list(await self._request_async("GET", "/proxies"))

    def delete_proxy(self, name: str) -> None:
        _require_name(name)
        self._request("DELETE", f"/proxies/{name}")

    async def delete_proxy_async(self, name: str) -> None:
        _require_name(name)
        await self._request_async("DELETE", f"/proxies/{name}")

    def add_toxic(
        self,
        *,
        proxy: str,
        name: str,
        toxic_type: str,
        stream: str = "downstream",
        toxicity: float = 1.0,
        attributes: dict[str, int] | None = None,
    ) -> dict[str, JsonValue]:
        path, payload = self._toxic_request(
            proxy=proxy,
            name=name,
            toxic_type=toxic_type,
            stream=stream,
            toxicity=toxicity,
            attributes=attributes,
        )
        return self._object(self._request("POST", path, payload))

    async def add_toxic_async(
        self,
        *,
        proxy: str,
        name: str,
        toxic_type: str,
        stream: str = "downstream",
        toxicity: float = 1.0,
        attributes: dict[str, int] | None = None,
    ) -> dict[str, JsonValue]:
        path, payload = self._toxic_request(
            proxy=proxy,
            name=name,
            toxic_type=toxic_type,
            stream=stream,
            toxicity=toxicity,
            attributes=attributes,
        )
        return self._object(await self._request_async("POST", path, payload))

    @staticmethod
    def _toxic_request(
        *,
        proxy: str,
        name: str,
        toxic_type: str,
        stream: str,
        toxicity: float,
        attributes: dict[str, int] | None,
    ) -> tuple[str, dict[str, JsonValue]]:
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
        attribute_payload: dict[str, JsonValue] = {
            key: value for key, value in (attributes or {}).items()
        }
        return (
            f"/proxies/{proxy}/toxics",
            {
                "name": name,
                "type": toxic_type,
                "stream": stream,
                "toxicity": toxicity,
                "attributes": attribute_payload,
            },
        )

    def remove_toxic(self, *, proxy: str, name: str) -> None:
        _require_name(proxy)
        _require_name(name)
        self._request("DELETE", f"/proxies/{proxy}/toxics/{name}")

    async def remove_toxic_async(self, *, proxy: str, name: str) -> None:
        _require_name(proxy)
        _require_name(name)
        await self._request_async("DELETE", f"/proxies/{proxy}/toxics/{name}")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, JsonValue] | None = None,
    ) -> JsonValue:
        client = self.http_client or BoundedHttpClient()
        try:
            response = client.request_loopback(self._control_request(method, path, payload))
            return response.json() if response.body else None
        except (BoundedHttpError, ValueError) as error:
            raise ToxiproxyApiError(f"Toxiproxy {method} {path} failed") from error
        finally:
            if self.http_client is None:
                client.close()

    async def _request_async(
        self,
        method: str,
        path: str,
        payload: dict[str, JsonValue] | None = None,
    ) -> JsonValue:
        client = self.http_client or BoundedHttpClient()
        try:
            response = await client.request_loopback_async(
                self._control_request(method, path, payload)
            )
            return response.json() if response.body else None
        except (BoundedHttpError, ValueError) as error:
            raise ToxiproxyApiError(f"Toxiproxy {method} {path} failed") from error
        finally:
            if self.http_client is None:
                await client.aclose()

    def _control_request(
        self,
        method: str,
        path: str,
        payload: dict[str, JsonValue] | None,
    ) -> LoopbackHttpRequest:
        return LoopbackHttpRequest(
            base_url=self.base_url,
            method=HttpMethod(method),
            path=path,
            deadline_monotonic=time.monotonic() + self.timeout_seconds,
            max_response_bytes=1024 * 1024,
            json_body=payload,
        )

    @staticmethod
    def _object(value: JsonValue) -> dict[str, JsonValue]:
        try:
            return _OBJECT_RESPONSE.validate_python(value)
        except ValidationError as error:
            raise ToxiproxyApiError("Toxiproxy returned a non-object") from error

    @staticmethod
    def _object_list(value: JsonValue) -> list[dict[str, JsonValue]]:
        try:
            return _OBJECT_LIST_RESPONSE.validate_python(value)
        except ValidationError as error:
            raise ToxiproxyApiError("Toxiproxy returned a non-list proxy collection") from error


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
