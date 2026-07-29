from __future__ import annotations

import os
import platform
import re
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, Protocol
from xml.etree import ElementTree

from pydantic import JsonValue

from flameox.application.workloads import AcceleratorIdentityRequirement
from flameox.domain import (
    AcceleratorDevice,
    AcceleratorIdentityFacet,
    AcceleratorLink,
    DomainError,
)
from flameox.domain.identity import digest_model
from flameox.domain.models import EnvironmentRecord, IdentityQuality, utc_now
from flameox.execution import ExecutionOutcome, ExecutionRequest, SubprocessBroker


class _ExecutionBroker(Protocol):
    async def run(self, request: ExecutionRequest) -> ExecutionOutcome: ...


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def collect_environment(
    accelerator: AcceleratorIdentityFacet | None = None,
) -> EnvironmentRecord:
    fields: dict[str, JsonValue] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "packages": {
            package: package_version
            for package in ("duckdb", "flameox", "mcp", "pyarrow", "pydantic")
            if (package_version := _package_version(package)) is not None
        },
        "executable": sys.executable,
    }
    missing_fields: tuple[str, ...] = ()
    quality = IdentityQuality.EXACT
    if accelerator is not None:
        fields["accelerator"] = accelerator.model_dump(mode="json")
        missing_fields = accelerator.missing_fields
        quality = accelerator.identity_quality
    identity = digest_model({"identity_quality": quality.value, "fields": fields})
    return EnvironmentRecord(
        environment_id=identity,
        observed_at=utc_now(),
        identity_quality=quality,
        fields=fields,
        missing_fields=missing_fields,
    )


class AcceleratorIdentityService:
    """Collect only explicitly declared CUDA identity through the vendor CLI."""

    def __init__(
        self,
        project_root: Path,
        *,
        broker: _ExecutionBroker | None = None,
    ) -> None:
        self.project_root = project_root
        self.broker = broker or SubprocessBroker()

    async def observe(
        self,
        required: tuple[AcceleratorIdentityRequirement, ...],
    ) -> AcceleratorIdentityFacet | None:
        if not required:
            return None
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return self._unavailable(required, "missing", "nvidia-smi was not found on PATH.")
        try:
            inventory = await self._run(executable, "-q", "-x")
        except PermissionError:
            return self._unavailable(
                required,
                "permission_denied",
                "Permission was denied while starting nvidia-smi.",
            )
        except DomainError as error:
            return self._unavailable(required, "unknown", error.message)
        if inventory.process.exit_code != 0:
            message = inventory.stderr.decode(errors="replace").strip()
            lowered = message.lower()
            status: Literal["permission_denied", "unsupported", "unknown"] = (
                "permission_denied"
                if "permission" in lowered
                else "unsupported"
                if "not supported" in lowered
                else "unknown"
            )
            return self._unavailable(required, status, message or "nvidia-smi failed.")
        try:
            driver, runtime, devices = _parse_nvidia_inventory(inventory.stdout)
        except (ElementTree.ParseError, ValueError) as error:
            return self._unavailable(required, "unknown", f"Invalid nvidia-smi XML: {error}")

        links: tuple[AcceleratorLink, ...] = ()
        topology_limitation: tuple[str, ...] = ()
        if "cuda.peer_topology" in required:
            try:
                topology = await self._run(executable, "topo", "-m")
                if topology.process.exit_code == 0:
                    links = _parse_nvidia_topology(topology.stdout.decode(errors="replace"))
                else:
                    topology_limitation = (
                        topology.stderr.decode(errors="replace").strip()
                        or "nvidia-smi topology query failed.",
                    )
            except (DomainError, PermissionError, ValueError) as error:
                topology_limitation = (f"Topology query failed: {error}",)

        missing = _missing_accelerator_fields(
            required,
            driver=driver,
            runtime=runtime,
            devices=devices,
            links=links,
            topology_failed=bool(topology_limitation),
        )
        return AcceleratorIdentityFacet(
            provider="cuda",
            status="available",
            identity_quality=IdentityQuality.PARTIAL if missing else IdentityQuality.EXACT,
            driver_version=driver,
            runtime_version=runtime,
            devices=devices,
            links=links,
            missing_fields=missing,
            limitations=topology_limitation,
        )

    async def _run(self, executable: str, *arguments: str) -> ExecutionOutcome:
        return await self.broker.run(
            ExecutionRequest(
                argv=(executable, *arguments),
                cwd=self.project_root,
                environment_allowlist=("PATH",),
                allowed_working_roots=(self.project_root,),
                timeout_seconds=10,
                max_output_bytes=1_048_576,
            )
        )

    @staticmethod
    def _unavailable(
        required: tuple[AcceleratorIdentityRequirement, ...],
        status: Literal["missing", "permission_denied", "unsupported", "unknown"],
        limitation: str,
    ) -> AcceleratorIdentityFacet:
        return AcceleratorIdentityFacet(
            provider="cuda",
            status=status,
            identity_quality=IdentityQuality.PARTIAL,
            missing_fields=required,
            limitations=(limitation,),
        )


def _parse_nvidia_inventory(
    payload: bytes,
) -> tuple[str | None, str | None, tuple[AcceleratorDevice, ...]]:
    root = ElementTree.fromstring(payload)
    driver = _xml_text(root, "driver_version")
    runtime = _xml_text(root, "cuda_version")
    devices: list[AcceleratorDevice] = []
    for index, gpu in enumerate(root.findall("gpu")):
        memory = _parse_mib(_xml_text(gpu, "fb_memory_usage/total"))
        mig_value = (_xml_text(gpu, "mig_mode/current_mig") or "").lower()
        devices.append(
            AcceleratorDevice(
                index=index,
                model=_xml_text(gpu, "product_name"),
                compute_capability=_xml_text(gpu, "compute_cap"),
                memory_mib=memory,
                mig_mode=(
                    "enabled"
                    if mig_value == "enabled"
                    else "disabled"
                    if mig_value == "disabled"
                    else "unknown"
                ),
            )
        )
    return driver, runtime, tuple(devices)


def _xml_text(element: ElementTree.Element, path: str) -> str | None:
    value = element.findtext(path)
    if value is None or value.strip() in {"", "N/A"}:
        return None
    return value.strip()


def _parse_mib(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)\s+MiB", value)
    if match is None:
        return None
    return int(match.group(1))


def _parse_nvidia_topology(value: str) -> tuple[AcceleratorLink, ...]:
    rows = [line.split() for line in value.splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty topology response")
    headers = [item for item in rows[0] if re.fullmatch(r"GPU\d+", item)]
    links: list[AcceleratorLink] = []
    for row in rows[1:]:
        if not row or not re.fullmatch(r"GPU\d+", row[0]):
            continue
        left = int(row[0][3:])
        for column, header in enumerate(headers, start=1):
            if column >= len(row):
                break
            right = int(header[3:])
            token = row[column]
            if right <= left or token == "X":
                continue
            links.append(_topology_link(left, right, token))
    return tuple(links)


def _topology_link(left: int, right: int, token: str) -> AcceleratorLink:
    nvlink = re.fullmatch(r"NV(\d+)", token)
    if nvlink is not None:
        return AcceleratorLink(
            left=left,
            right=right,
            kind="nvlink",
            width=int(nvlink.group(1)),
        )
    kinds: dict[str, Literal["pcie", "host_bridge", "numa", "system"]] = {
        "PIX": "pcie",
        "PXB": "host_bridge",
        "PHB": "host_bridge",
        "NODE": "numa",
        "SYS": "system",
    }
    return AcceleratorLink(left=left, right=right, kind=kinds.get(token, "unknown"))


def _missing_accelerator_fields(
    required: tuple[AcceleratorIdentityRequirement, ...],
    *,
    driver: str | None,
    runtime: str | None,
    devices: tuple[AcceleratorDevice, ...],
    links: tuple[AcceleratorLink, ...],
    topology_failed: bool,
) -> tuple[str, ...]:
    missing: list[str] = []
    if "cuda.driver" in required and driver is None:
        missing.append("cuda.driver")
    if "cuda.runtime" in required and runtime is None:
        missing.append("cuda.runtime")
    if "cuda.devices" in required and (
        not devices
        or any(
            device.model is None
            or device.compute_capability is None
            or device.memory_mib is None
            or device.mig_mode == "unknown"
            for device in devices
        )
    ):
        missing.append("cuda.devices")
    if "cuda.peer_topology" in required and (topology_failed or (len(devices) > 1 and not links)):
        missing.append("cuda.peer_topology")
    return tuple(missing)
