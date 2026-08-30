from __future__ import annotations

import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from flameox.workers import nvml
from flameox.workers.nvml_contract import NvmlWorkerRequest

pytestmark = pytest.mark.unit


class NVMLError(Exception):
    pass


class NVMLError_NotSupported(NVMLError):
    pass


class _FakeNvml:
    NVMLError = NVMLError
    NVML_FEATURE_ENABLED = 1
    NVML_TOPOLOGY_INTERNAL = 0

    def __init__(self) -> None:
        self.shutdown = False

    def nvmlInit(self) -> None:
        return None

    def nvmlShutdown(self) -> None:
        self.shutdown = True

    def nvmlSystemGetNVMLVersion(self) -> bytes:
        return b"13.610.43"

    def nvmlSystemGetDriverVersion(self) -> bytes:
        return b"575.57.08"

    def nvmlSystemGetCudaDriverVersion_v2(self) -> int:
        return 12090

    def nvmlDeviceGetCount(self) -> int:
        return 2

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetUUID(self, handle: int) -> str:
        return f"GPU-{handle}"

    def nvmlDeviceGetPciInfo(self, handle: int) -> SimpleNamespace:
        return SimpleNamespace(busId=f"0000:0{handle + 1}:00.0")

    def nvmlDeviceGetName(self, _handle: int) -> str:
        return "NVIDIA H100"

    def nvmlDeviceGetMemoryInfo(self, _handle: int) -> SimpleNamespace:
        return SimpleNamespace(total=80 * 1024**3)

    def nvmlDeviceGetCudaComputeCapability(self, _handle: int) -> tuple[int, int]:
        return (9, 0)

    def nvmlDeviceGetMigMode(self, handle: int) -> tuple[int, int]:
        if handle == 1:
            raise NVMLError_NotSupported("MIG unavailable")
        return (0, 0)

    def nvmlDeviceGetTopologyCommonAncestor(self, _left: int, _right: int) -> int:
        return self.NVML_TOPOLOGY_INTERNAL


def test_nvml_worker_projects_stable_identity_and_field_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = _FakeNvml()
    monkeypatch.setattr(nvml, "_provider", provider)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "13.610.43")

    snapshot = nvml._observe(
        NvmlWorkerRequest(include_topology=True),
        tmp_path,
    )

    assert snapshot.cuda_driver_version == "12.9"
    assert snapshot.devices[0].uuid == "GPU-0"
    assert snapshot.devices[0].total_memory_bytes == 80 * 1024**3
    assert snapshot.devices[1].unavailable_fields[0].kind == "not_supported"
    assert snapshot.peer_links[0].common_ancestor == "internal"
    assert provider.shutdown is True
