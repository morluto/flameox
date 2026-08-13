from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable
from typing import Any, cast

from flameox.workers.nvml_contract import (
    NVML_WORKER,
    NvmlDeviceIdentity,
    NvmlFieldFailure,
    NvmlPeerLink,
    NvmlSnapshot,
    NvmlWorkerRequest,
)
from flameox.workers.protocol import (
    WorkerApplication,
    WorkerContext,
    WorkerFailureKind,
    run_typed_worker,
)

_provider: Any = None


def _nvml() -> Any:
    global _provider
    if _provider is None:
        _provider = importlib.import_module("pynvml")
    return _provider


def _text(value: object) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _failure(field: str, error: BaseException) -> NvmlFieldFailure:
    name = type(error).__name__
    kind = {
        "NVMLError_LibraryNotFound": "library_not_found",
        "NVMLError_DriverNotLoaded": "driver_not_loaded",
        "NVMLError_NoPermission": "permission_denied",
        "NVMLError_NotSupported": "not_supported",
        "NVMLError_GpuIsLost": "gpu_lost",
        "NVMLError_FunctionNotFound": "function_not_found",
        "NVMLError_InvalidArgument": "invalid_data",
    }.get(name, "provider_failure")
    return NvmlFieldFailure(field=field, kind=cast(Any, kind), message=f"{name}: {_text(error)}")


def _read[T](
    field: str,
    query: Callable[[], T],
    failures: list[NvmlFieldFailure],
) -> T | None:
    provider = _nvml()
    try:
        return query()
    except provider.NVMLError as error:
        failures.append(_failure(field, error))
        return None


def _cuda_version(value: int | None) -> str | None:
    if value is None:
        return None
    return f"{value // 1000}.{value % 1000 // 10}"


def _device(handle: object, index: int) -> NvmlDeviceIdentity:
    pynvml = _nvml()
    failures: list[NvmlFieldFailure] = []
    uuid = _read("uuid", lambda: _text(pynvml.nvmlDeviceGetUUID(handle)), failures)
    pci = _read("pci_bus_id", lambda: pynvml.nvmlDeviceGetPciInfo(handle), failures)
    name = _read("name", lambda: _text(pynvml.nvmlDeviceGetName(handle)), failures)
    memory = _read("total_memory_bytes", lambda: pynvml.nvmlDeviceGetMemoryInfo(handle), failures)
    capability = _read(
        "compute_capability",
        lambda: pynvml.nvmlDeviceGetCudaComputeCapability(handle),
        failures,
    )
    mig = _read("mig_mode", lambda: pynvml.nvmlDeviceGetMigMode(handle), failures)
    return NvmlDeviceIdentity(
        nvml_index=index,
        uuid=uuid,
        pci_bus_id=_text(pci.busId) if pci is not None else None,
        name=name,
        total_memory_bytes=int(memory.total) if memory is not None else None,
        compute_capability=(int(capability[0]), int(capability[1])) if capability else None,
        mig_mode=(
            "enabled"
            if mig is not None and int(mig[0]) == getattr(pynvml, "NVML_FEATURE_ENABLED", 1)
            else "disabled"
            if mig is not None
            else "unknown"
        ),
        unavailable_fields=tuple(failures),
    )


def _observe(request: NvmlWorkerRequest, _context: WorkerContext) -> NvmlSnapshot:
    pynvml = _nvml()
    topology_names = {
        getattr(pynvml, "NVML_TOPOLOGY_INTERNAL", -1): "internal",
        getattr(pynvml, "NVML_TOPOLOGY_SINGLE", -2): "single_switch",
        getattr(pynvml, "NVML_TOPOLOGY_MULTIPLE", -3): "multiple_switches",
        getattr(pynvml, "NVML_TOPOLOGY_HOSTBRIDGE", -4): "host_bridge",
        getattr(pynvml, "NVML_TOPOLOGY_NODE", -5): "numa",
        getattr(pynvml, "NVML_TOPOLOGY_SYSTEM", -6): "system",
    }
    failures: list[NvmlFieldFailure] = []
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as error:
        return NvmlSnapshot(
            binding_version=importlib.metadata.version("nvidia-ml-py"),
            unavailable_fields=(_failure("initialize", error),),
        )
    try:
        nvml_version = _read(
            "nvml_version", lambda: _text(pynvml.nvmlSystemGetNVMLVersion()), failures
        )
        driver = _read(
            "driver_version", lambda: _text(pynvml.nvmlSystemGetDriverVersion()), failures
        )
        encoded_cuda = _read(
            "cuda_driver_version",
            lambda: int(pynvml.nvmlSystemGetCudaDriverVersion_v2()),
            failures,
        )
        count = _read("device_count", lambda: int(pynvml.nvmlDeviceGetCount()), failures)
        handles: list[object] = []
        devices: list[NvmlDeviceIdentity] = []
        for index in range(count or 0):

            def device_handle(index: int = index) -> object:
                return cast(object, pynvml.nvmlDeviceGetHandleByIndex(index))

            handle = _read(
                f"devices[{index}].handle",
                device_handle,
                failures,
            )
            if handle is not None:
                handles.append(handle)
                devices.append(_device(handle, index))
        links: list[NvmlPeerLink] = []
        if request.include_topology:
            for left in range(len(handles)):
                for right in range(left + 1, len(handles)):
                    link_failures: list[NvmlFieldFailure] = []

                    def common_ancestor(left: int = left, right: int = right) -> object:
                        return cast(
                            object,
                            pynvml.nvmlDeviceGetTopologyCommonAncestor(
                                handles[left], handles[right]
                            ),
                        )

                    ancestor = _read(
                        "common_ancestor",
                        common_ancestor,
                        link_failures,
                    )
                    left_uuid = devices[left].uuid
                    right_uuid = devices[right].uuid
                    if left_uuid is None or right_uuid is None:
                        link_failures.append(
                            NvmlFieldFailure(
                                field="stable_endpoints",
                                kind="invalid_data",
                                message="Topology endpoints require both device UUIDs.",
                            )
                        )
                        continue
                    links.append(
                        NvmlPeerLink(
                            left_uuid=left_uuid,
                            right_uuid=right_uuid,
                            common_ancestor=cast(Any, topology_names.get(ancestor, "unknown"))
                            if ancestor is not None
                            else None,
                            unavailable_fields=tuple(link_failures),
                        )
                    )
        return NvmlSnapshot(
            binding_version=importlib.metadata.version("nvidia-ml-py"),
            nvml_version=nvml_version,
            driver_version=driver,
            cuda_driver_version=_cuda_version(encoded_cuda),
            devices=tuple(devices),
            peer_links=tuple(links),
            unavailable_fields=tuple(failures),
        )
    finally:
        pynvml.nvmlShutdown()


APPLICATION = WorkerApplication(
    definition=NVML_WORKER,
    handler=_observe,
    invalid_failure=WorkerFailureKind.PROVIDER_INCOMPATIBLE,
    invalid_message="NVML returned an invalid identity snapshot",
    caught=(ValueError, TypeError, AttributeError),
)


def main() -> int:
    return run_typed_worker(APPLICATION)


if __name__ == "__main__":
    raise SystemExit(main())
