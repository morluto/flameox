from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application.environment import (
    AcceleratorIdentityService,
    collect_environment,
)
from flameox.application.workloads import AcceleratorIdentityRequirement
from flameox.domain import (
    AcceleratorDevice,
    AcceleratorIdentityFacet,
    AcceleratorIdentityStatus,
    AcceleratorLink,
    AcceleratorLinkKind,
    AcceleratorMigMode,
    IdentityQuality,
)
from flameox.workers.nvml_contract import (
    NvmlDeviceIdentity,
    NvmlFieldFailure,
    NvmlPeerLink,
    NvmlSnapshot,
)

pytestmark = pytest.mark.integration


class _Observer:
    def __init__(self, snapshot: NvmlSnapshot) -> None:
        self.snapshot = snapshot
        self.include_topology: bool | None = None

    async def observe(self, *, include_topology: bool) -> NvmlSnapshot:
        self.include_topology = include_topology
        return self.snapshot


def _snapshot() -> NvmlSnapshot:
    return NvmlSnapshot(
        binding_version="13.610.43",
        nvml_version="13.610.43",
        driver_version="575.57.08",
        cuda_driver_version="12.9",
        devices=(
            NvmlDeviceIdentity(
                nvml_index=0,
                uuid="GPU-aaaa",
                pci_bus_id="0000:01:00.0",
                name="NVIDIA H100 80GB HBM3",
                total_memory_bytes=85_520_875_520,
                compute_capability=(9, 0),
                mig_mode="disabled",
            ),
            NvmlDeviceIdentity(
                nvml_index=1,
                uuid="GPU-bbbb",
                pci_bus_id="0000:02:00.0",
                name="NVIDIA H100 80GB HBM3",
                total_memory_bytes=85_520_875_520,
                compute_capability=(9, 0),
                mig_mode="enabled",
            ),
        ),
        peer_links=(
            NvmlPeerLink(
                left_uuid="GPU-aaaa",
                right_uuid="GPU-bbbb",
                common_ancestor="internal",
            ),
        ),
    )


@pytest.mark.anyio
async def test_declared_accelerator_identity_is_structured_and_bounded(
    tmp_path: Path,
) -> None:
    observer = _Observer(_snapshot())
    required: tuple[AcceleratorIdentityRequirement, ...] = (
        "cuda.driver",
        "cuda.runtime",
        "cuda.devices",
        "cuda.peer_topology",
    )

    facet = await AcceleratorIdentityService(observer=observer).observe(required)

    assert facet is not None
    assert facet.identity_quality is IdentityQuality.EXACT
    assert facet.driver_version == "575.57.08"
    assert facet.runtime_version == "12.9"
    assert [device.model for device in facet.devices] == [
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H100 80GB HBM3",
    ]
    assert [device.mig_mode for device in facet.devices] == ["disabled", "enabled"]
    assert facet.links[0].model_dump() == {
        "left": 0,
        "right": 1,
        "kind": "nvlink",
        "width": None,
    }
    assert facet.devices[0].stable_id == "GPU-aaaa"
    assert facet.devices[0].pci_bus_id == "0000:01:00.0"
    assert facet.devices[0].memory_bytes == 85_520_875_520
    assert observer.include_topology is True

    environment = collect_environment(facet)
    assert environment.identity_quality is IdentityQuality.EXACT
    assert environment.fields["accelerator"] == facet.model_dump(mode="json")


@pytest.mark.anyio
async def test_missing_required_accelerator_is_partial_not_exact(
    tmp_path: Path,
) -> None:
    required: tuple[AcceleratorIdentityRequirement, ...] = ("cuda.driver", "cuda.devices")

    facet = await AcceleratorIdentityService().observe(required)

    assert facet is not None
    assert facet.status == "missing"
    assert facet.identity_quality is IdentityQuality.PARTIAL
    assert facet.missing_fields == required
    environment = collect_environment(facet)
    assert environment.identity_quality is IdentityQuality.PARTIAL
    assert environment.missing_fields == required


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("permission_denied", "permission_denied"),
        ("not_supported", "unsupported"),
        ("provider_failure", "unknown"),
    ],
)
@pytest.mark.anyio
async def test_accelerator_probe_failure_states_remain_distinct(
    tmp_path: Path,
    kind: str,
    expected: str,
) -> None:
    snapshot = NvmlSnapshot(
        binding_version="13.610.43",
        unavailable_fields=(
            NvmlFieldFailure(field="initialize", kind=kind, message="provider failed"),  # type: ignore[arg-type]
        ),
    )
    facet = await AcceleratorIdentityService(observer=_Observer(snapshot)).observe(
        ("cuda.devices",)
    )

    assert facet is not None
    assert facet.status == expected
    assert facet.identity_quality is IdentityQuality.PARTIAL


@pytest.mark.anyio
async def test_unknown_inventory_fields_remain_distinct_and_partial(
    tmp_path: Path,
) -> None:
    snapshot = NvmlSnapshot(
        binding_version="13.610.43",
        devices=(NvmlDeviceIdentity(nvml_index=0, uuid="GPU-aaaa", name="H100"),),
    )

    facet = await AcceleratorIdentityService(observer=_Observer(snapshot)).observe(
        ("cuda.driver", "cuda.runtime", "cuda.devices")
    )

    assert facet is not None
    assert facet.status == "available"
    assert facet.identity_quality is IdentityQuality.PARTIAL
    assert facet.missing_fields == ("cuda.driver", "cuda.runtime", "cuda.devices")


@pytest.mark.parametrize(
    "change",
    [
        {"driver_version": "580.10"},
        {
            "devices": (
                AcceleratorDevice(
                    index=0,
                    model="NVIDIA H100 PCIe",
                    compute_capability="9.0",
                    memory_mib=81559,
                    mig_mode=AcceleratorMigMode.DISABLED,
                ),
            )
        },
        {
            "devices": (
                AcceleratorDevice(
                    index=0,
                    model="NVIDIA H100 SXM",
                    compute_capability="9.0",
                    memory_mib=81559,
                    mig_mode=AcceleratorMigMode.ENABLED,
                ),
            )
        },
        {"links": (AcceleratorLink(left=0, right=1, kind=AcceleratorLinkKind.HOST_BRIDGE),)},
    ],
)
def test_driver_model_mig_and_topology_each_change_environment_identity(
    change: dict[str, object],
) -> None:
    baseline = AcceleratorIdentityFacet(
        provider="cuda",
        status=AcceleratorIdentityStatus.AVAILABLE,
        identity_quality=IdentityQuality.EXACT,
        driver_version="575.57.08",
        runtime_version="12.9",
        devices=(
            AcceleratorDevice(
                index=0,
                model="NVIDIA H100 SXM",
                compute_capability="9.0",
                memory_mib=81559,
                mig_mode=AcceleratorMigMode.DISABLED,
            ),
        ),
        links=(AcceleratorLink(left=0, right=1, kind=AcceleratorLinkKind.NVLINK, width=18),),
    )
    changed = baseline.model_copy(update=change)

    assert (
        collect_environment(baseline).environment_id != collect_environment(changed).environment_id
    )
