from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId


class NvmlWorkerRequest(ContractModel):
    include_topology: bool = False


class NvmlFieldFailure(ContractModel):
    field: str = Field(min_length=1, max_length=100)
    kind: Literal[
        "library_not_found",
        "driver_not_loaded",
        "permission_denied",
        "not_supported",
        "gpu_lost",
        "function_not_found",
        "invalid_data",
        "provider_failure",
    ]
    message: str = Field(min_length=1, max_length=300)


class NvmlDeviceIdentity(ContractModel):
    nvml_index: Annotated[int, Field(ge=0)]
    uuid: str | None = Field(default=None, max_length=100)
    pci_bus_id: str | None = Field(default=None, max_length=100)
    name: str | None = Field(default=None, max_length=300)
    total_memory_bytes: Annotated[int, Field(ge=0)] | None = None
    compute_capability: tuple[Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]] | None = (
        None
    )
    mig_mode: Literal["enabled", "disabled", "unknown"] = "unknown"
    unavailable_fields: tuple[NvmlFieldFailure, ...] = ()


class NvmlPeerLink(ContractModel):
    left_uuid: str
    right_uuid: str
    common_ancestor: (
        Literal[
            "internal",
            "single_switch",
            "multiple_switches",
            "host_bridge",
            "numa",
            "system",
            "unknown",
        ]
        | None
    ) = None
    unavailable_fields: tuple[NvmlFieldFailure, ...] = ()


class NvmlSnapshot(ContractModel):
    binding_version: str = Field(min_length=1, max_length=100)
    nvml_version: str | None = Field(default=None, max_length=100)
    driver_version: str | None = Field(default=None, max_length=100)
    cuda_driver_version: str | None = Field(default=None, max_length=100)
    devices: tuple[NvmlDeviceIdentity, ...] = ()
    peer_links: tuple[NvmlPeerLink, ...] = ()
    unavailable_fields: tuple[NvmlFieldFailure, ...] = ()


NVML_WORKER = WorkerDefinition(
    operation=WorkerOperationId.NVML_OBSERVE,
    module="flameox.workers.nvml",
    request=TypeAdapter(NvmlWorkerRequest),
    response=TypeAdapter(NvmlSnapshot),
    name="NVIDIA NVML",
    implementation="flameox.workers.nvml/v1",
    timeout_seconds=30,
)
