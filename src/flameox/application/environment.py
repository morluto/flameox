from __future__ import annotations

import json
import os
import platform
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Literal, Protocol, cast

from pydantic import Field, JsonValue

from flameox.adapters.artifact_workers import IsolatedWorkerHarness
from flameox.application.provider_catalog import NVIDIA_NVML_PROVIDER
from flameox.application.provider_runtime import ProviderRuntimeManager
from flameox.application.workloads import AcceleratorIdentityRequirement
from flameox.command_binding import ExecutableResolver
from flameox.domain import (
    AcceleratorDevice,
    AcceleratorIdentityFacet,
    AcceleratorIdentityStatus,
    AcceleratorLink,
    AcceleratorLinkKind,
    AcceleratorMigMode,
    DomainError,
)
from flameox.domain.identity import digest_model
from flameox.domain.models import CapabilityExtra, EnvironmentRecord, IdentityQuality, utc_now
from flameox.execution import ExecutionRequest, SubprocessBroker
from flameox.models import ContractModel
from flameox.storage import Workspace
from flameox.workers.nvml_contract import NVML_WORKER, NvmlSnapshot, NvmlWorkerRequest


class _NvmlObserver(Protocol):
    async def observe(self, *, include_topology: bool) -> NvmlSnapshot: ...


class MetalIdentitySnapshot(ContractModel):
    provider_version: str
    chip_model: str | None = None
    gpu_model: str | None = None
    gpu_core_count: int | None = Field(default=None, gt=0)
    device_count: int | None = Field(default=None, ge=0)
    metal_support: str | None = None
    unified_memory_bytes: int | None = Field(default=None, ge=0)
    macos_product_version: str | None = None
    macos_build: str | None = None
    limitations: tuple[str, ...] = ()


class _MetalObserver(Protocol):
    async def observe(self) -> MetalIdentitySnapshot: ...


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


class _WorkerNvmlObserver:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    async def observe(self, *, include_topology: bool) -> NvmlSnapshot:
        runtime = ProviderRuntimeManager(self.workspace.paths.records / "provider-runtimes").find(
            extra=CapabilityExtra.HARDWARE,
            requirement=NVIDIA_NVML_PROVIDER.requirement,
        )
        if runtime is None:
            raise FileNotFoundError("the managed NVIDIA NVML provider is not installed")
        return await IsolatedWorkerHarness(
            self.workspace,
            python=runtime.python,
        ).run_typed(NVML_WORKER, NvmlWorkerRequest(include_topology=include_topology))


class _SystemMetalObserver:
    def __init__(self, workspace: Workspace, broker: SubprocessBroker | None = None) -> None:
        self.workspace = workspace
        self.broker = broker or SubprocessBroker()

    async def observe(self) -> MetalIdentitySnapshot:
        if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
            return MetalIdentitySnapshot(
                provider_version="macos-system-tools/v1",
                limitations=("Apple Metal identity is supported only on arm64 macOS.",),
            )
        try:
            displays = await self._run_json(
                "system_profiler", "-json", "SPDisplaysDataType", "SPHardwareDataType"
            )
            product = await self._run_text("sw_vers", "-productVersion")
            build = await self._run_text("sw_vers", "-buildVersion")
            memory = await self._run_text("sysctl", "-n", "hw.memsize")
        except (DomainError, FileNotFoundError, PermissionError, ValueError) as error:
            return MetalIdentitySnapshot(
                provider_version="macos-system-tools/v1",
                limitations=(f"Metal identity collection failed: {type(error).__name__}.",),
            )
        return _parse_metal_identity(
            displays,
            product_version=product,
            build=build,
            memory=memory,
        )

    async def _run_text(self, name: str, *arguments: str) -> str:
        executable = ExecutableResolver().resolve_host_tool(name, cwd=self.workspace.project_root)
        if executable is None:
            raise FileNotFoundError(name)
        outcome = await self.broker.run(
            ExecutionRequest(
                argv=(str(executable.invocation_path), *arguments),
                cwd=self.workspace.project_root,
                environment_allowlist=("PATH",),
                allowed_working_roots=(self.workspace.project_root,),
                timeout_seconds=20,
                max_output_bytes=2 * 1024 * 1024,
                executable_binding=executable,
            )
        )
        if outcome.process.exit_code != 0:
            raise ValueError(f"{name} returned {outcome.process.exit_code}")
        return outcome.stdout.decode(errors="replace").strip()

    async def _run_json(self, name: str, *arguments: str) -> object:
        return json.loads(await self._run_text(name, *arguments))


class AcceleratorIdentityService:
    """Project explicitly declared CUDA identity from the isolated NVML provider."""

    def __init__(
        self,
        workspace: Workspace | None = None,
        *,
        observer: _NvmlObserver | None = None,
        metal_observer: _MetalObserver | None = None,
        broker: SubprocessBroker | None = None,
    ) -> None:
        self.observer = observer or (
            _WorkerNvmlObserver(workspace) if workspace is not None else None
        )
        self.metal_observer = metal_observer or (
            _SystemMetalObserver(workspace, broker) if workspace is not None else None
        )

    async def observe(
        self,
        required: tuple[AcceleratorIdentityRequirement, ...],
    ) -> AcceleratorIdentityFacet | None:
        if not required:
            return None
        if any(item.startswith("metal.") or item == "macos.build" for item in required):
            return await self._observe_metal(required)
        if self.observer is None:
            return self._unavailable(
                required, AcceleratorIdentityStatus.MISSING, "NVML provider unavailable."
            )
        try:
            snapshot = await self.observer.observe(
                include_topology="cuda.peer_topology" in required
            )
        except FileNotFoundError as error:
            return self._unavailable(required, AcceleratorIdentityStatus.MISSING, str(error))
        global_failures = snapshot.unavailable_fields
        if not snapshot.devices and global_failures:
            kinds = {failure.kind for failure in global_failures}
            status = (
                AcceleratorIdentityStatus.PERMISSION_DENIED
                if "permission_denied" in kinds
                else AcceleratorIdentityStatus.UNSUPPORTED
                if "not_supported" in kinds or "function_not_found" in kinds
                else AcceleratorIdentityStatus.MISSING
                if "library_not_found" in kinds or "driver_not_loaded" in kinds
                else AcceleratorIdentityStatus.UNKNOWN
            )
            return self._unavailable(
                required,
                cast(
                    Literal[
                        AcceleratorIdentityStatus.MISSING,
                        AcceleratorIdentityStatus.PERMISSION_DENIED,
                        AcceleratorIdentityStatus.UNSUPPORTED,
                        AcceleratorIdentityStatus.UNKNOWN,
                    ],
                    status,
                ),
                _failure_summary(snapshot),
            )
        devices = tuple(
            AcceleratorDevice(
                index=device.nvml_index,
                stable_id=device.uuid,
                pci_bus_id=device.pci_bus_id,
                model=device.name,
                compute_capability=(
                    f"{device.compute_capability[0]}.{device.compute_capability[1]}"
                    if device.compute_capability is not None
                    else None
                ),
                memory_bytes=device.total_memory_bytes,
                memory_mib=(
                    device.total_memory_bytes // (1024 * 1024)
                    if device.total_memory_bytes is not None
                    else None
                ),
                mig_mode=AcceleratorMigMode(device.mig_mode),
            )
            for device in snapshot.devices
        )
        index_by_uuid = {
            device.stable_id: device.index for device in devices if device.stable_id is not None
        }
        links = tuple(
            AcceleratorLink(
                left=index_by_uuid[link.left_uuid],
                right=index_by_uuid[link.right_uuid],
                kind=_topology_kind(link.common_ancestor),
            )
            for link in snapshot.peer_links
            if link.left_uuid in index_by_uuid and link.right_uuid in index_by_uuid
        )
        missing = _missing_accelerator_fields(
            required,
            driver=snapshot.driver_version,
            runtime=snapshot.cuda_driver_version,
            devices=devices,
            links=links,
            topology_failed=any(link.unavailable_fields for link in snapshot.peer_links),
        )
        return AcceleratorIdentityFacet(
            provider="cuda",
            status=AcceleratorIdentityStatus.AVAILABLE,
            identity_quality=IdentityQuality.PARTIAL if missing else IdentityQuality.EXACT,
            driver_version=snapshot.driver_version,
            runtime_version=snapshot.cuda_driver_version,
            provider_version=(
                f"nvidia-ml-py {snapshot.binding_version}; "
                f"NVML {snapshot.nvml_version or 'unknown'}"
            ),
            devices=devices,
            links=links,
            missing_fields=missing,
            limitations=_snapshot_limitations(snapshot),
        )

    async def _observe_metal(
        self,
        required: tuple[AcceleratorIdentityRequirement, ...],
    ) -> AcceleratorIdentityFacet:
        if self.metal_observer is None:
            return AcceleratorIdentityFacet(
                provider="metal",
                status=AcceleratorIdentityStatus.MISSING,
                identity_quality=IdentityQuality.PARTIAL,
                missing_fields=required,
                limitations=("Apple Metal identity provider unavailable.",),
            )
        snapshot = await self.metal_observer.observe()
        device = (
            AcceleratorDevice(
                index=0,
                stable_id=snapshot.chip_model,
                model=snapshot.gpu_model or snapshot.chip_model,
                memory_bytes=snapshot.unified_memory_bytes,
                memory_mib=(
                    snapshot.unified_memory_bytes // (1024 * 1024)
                    if snapshot.unified_memory_bytes is not None
                    else None
                ),
                gpu_core_count=snapshot.gpu_core_count,
            )
            if snapshot.device_count
            else None
        )
        devices = (device,) if device is not None else ()
        missing = tuple(
            field
            for field in required
            if (
                (field == "metal.devices" and (not devices or snapshot.device_count is None))
                or (field == "metal.support" and snapshot.metal_support is None)
                or (field == "metal.unified_memory" and snapshot.unified_memory_bytes is None)
                or (
                    field == "macos.build"
                    and (snapshot.macos_product_version is None or snapshot.macos_build is None)
                )
            )
        )
        return AcceleratorIdentityFacet(
            provider="metal",
            status=(
                AcceleratorIdentityStatus.AVAILABLE
                if devices or snapshot.metal_support is not None
                else AcceleratorIdentityStatus.UNSUPPORTED
            ),
            identity_quality=IdentityQuality.PARTIAL if missing else IdentityQuality.EXACT,
            provider_version=snapshot.provider_version,
            metal_support=snapshot.metal_support,
            unified_memory_bytes=snapshot.unified_memory_bytes,
            macos_product_version=snapshot.macos_product_version,
            macos_build=snapshot.macos_build,
            devices=devices,
            missing_fields=missing,
            limitations=snapshot.limitations,
        )

    @staticmethod
    def _unavailable(
        required: tuple[AcceleratorIdentityRequirement, ...],
        status: Literal[
            AcceleratorIdentityStatus.MISSING,
            AcceleratorIdentityStatus.PERMISSION_DENIED,
            AcceleratorIdentityStatus.UNSUPPORTED,
            AcceleratorIdentityStatus.UNKNOWN,
        ],
        limitation: str,
    ) -> AcceleratorIdentityFacet:
        return AcceleratorIdentityFacet(
            provider="cuda",
            status=status,
            identity_quality=IdentityQuality.PARTIAL,
            missing_fields=required,
            limitations=(limitation,),
        )


def _parse_metal_identity(
    payload: object,
    *,
    product_version: str,
    build: str,
    memory: str,
) -> MetalIdentitySnapshot:
    limitations: list[str] = []
    root = payload if isinstance(payload, dict) else {}
    displays_value = root.get("SPDisplaysDataType")
    hardware_value = root.get("SPHardwareDataType")
    displays = (
        [item for item in displays_value if isinstance(item, dict)]
        if isinstance(displays_value, list)
        else []
    )
    hardware = (
        [item for item in hardware_value if isinstance(item, dict)]
        if isinstance(hardware_value, list)
        else []
    )
    if not displays:
        limitations.append("system_profiler returned no display records.")
    display = displays[0] if displays else {}
    host = hardware[0] if hardware else {}
    chip = _bounded_string(host.get("chip_type"))
    model = _bounded_string(display.get("sppci_model")) or _bounded_string(display.get("_name"))
    raw_metal = _bounded_string(display.get("spdisplays_metal"))
    metal_match = re.search(r"\bMetal\s+[^,;]+", raw_metal or "")
    raw_cores = display.get("sppci_cores", display.get("spdisplays_gpu_core_count"))
    core_match = re.search(r"\d+", str(raw_cores)) if raw_cores is not None else None
    try:
        memory_bytes = int(memory)
        if memory_bytes < 0:
            raise ValueError
    except ValueError:
        memory_bytes = None
        limitations.append("sysctl returned an invalid unified-memory size.")
    return MetalIdentitySnapshot(
        provider_version=f"macos-system-tools/v1 ({build or 'unknown build'})",
        chip_model=chip,
        gpu_model=model,
        gpu_core_count=int(core_match.group()) if core_match is not None else None,
        device_count=len(displays) if isinstance(displays_value, list) else None,
        metal_support=metal_match.group().strip() if metal_match is not None else None,
        unified_memory_bytes=memory_bytes,
        macos_product_version=product_version or None,
        macos_build=build or None,
        limitations=tuple(limitations),
    )


def _bounded_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned[:300] or None


def _topology_kind(value: str | None) -> AcceleratorLinkKind:
    return {
        "internal": AcceleratorLinkKind.NVLINK,
        "single_switch": AcceleratorLinkKind.PCIE,
        "multiple_switches": AcceleratorLinkKind.HOST_BRIDGE,
        "host_bridge": AcceleratorLinkKind.HOST_BRIDGE,
        "numa": AcceleratorLinkKind.NUMA,
        "system": AcceleratorLinkKind.SYSTEM,
    }.get(value or "", AcceleratorLinkKind.UNKNOWN)


def _failure_summary(snapshot: NvmlSnapshot) -> str:
    return "; ".join(f"{failure.field}: {failure.kind}" for failure in snapshot.unavailable_fields)[
        :500
    ]


def _snapshot_limitations(snapshot: NvmlSnapshot) -> tuple[str, ...]:
    failures = [*snapshot.unavailable_fields]
    for device in snapshot.devices:
        failures.extend(device.unavailable_fields)
    for link in snapshot.peer_links:
        failures.extend(link.unavailable_fields)
    return tuple(dict.fromkeys(f"{failure.field}: {failure.kind}" for failure in failures))


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
            or device.mig_mode is AcceleratorMigMode.UNKNOWN
            for device in devices
        )
    ):
        missing.append("cuda.devices")
    if "cuda.peer_topology" in required and (topology_failed or (len(devices) > 1 and not links)):
        missing.append("cuda.peer_topology")
    return tuple(missing)
