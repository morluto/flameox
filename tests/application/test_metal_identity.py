from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application.environment import (
    AcceleratorIdentityService,
    MetalIdentitySnapshot,
    _parse_metal_identity,
    collect_environment,
)
from flameox.application.workloads import (
    AcceleratorIdentityRequirement,
    WorkloadEnvironmentIdentityConfig,
)
from flameox.domain import IdentityQuality

pytestmark = pytest.mark.unit


class _Observer:
    def __init__(self, snapshot: MetalIdentitySnapshot) -> None:
        self.snapshot = snapshot

    async def observe(self) -> MetalIdentitySnapshot:
        return self.snapshot


@pytest.mark.parametrize(
    ("chip", "cores", "metal", "build"),
    [
        ("Apple M2 Max", "38", "spdisplays_supported, Metal 3", "23G93"),
        ("Apple M5", "10", "spdisplays_supported, Metal 4", "25D125"),
    ],
)
def test_system_profiler_generations_parse_without_sensitive_identifiers(
    chip: str,
    cores: str,
    metal: str,
    build: str,
) -> None:
    snapshot = _parse_metal_identity(
        {
            "SPDisplaysDataType": [
                {
                    "_name": chip,
                    "sppci_model": chip,
                    "sppci_cores": cores,
                    "spdisplays_metal": metal,
                    "serial_number": "must-not-escape",
                }
            ],
            "SPHardwareDataType": [{"chip_type": chip, "serial_number": "must-not-escape"}],
        },
        product_version="26.3",
        build=build,
        memory="25769803776",
    )

    assert snapshot.chip_model == chip
    assert snapshot.gpu_core_count == int(cores)
    assert snapshot.metal_support == metal.split(", ")[1]
    assert snapshot.macos_build == build
    assert "serial" not in snapshot.model_dump_json().lower()


@pytest.mark.anyio
async def test_declared_metal_identity_is_exact_and_changes_environment_identity(
    tmp_path: Path,
) -> None:
    first = MetalIdentitySnapshot(
        provider_version="macos-system-tools/v1 (25D125)",
        chip_model="Apple M5",
        gpu_model="Apple M5",
        gpu_core_count=10,
        device_count=1,
        metal_support="Metal 4",
        unified_memory_bytes=25_769_803_776,
        macos_product_version="26.3",
        macos_build="25D125",
    )
    required: tuple[AcceleratorIdentityRequirement, ...] = (
        "metal.devices",
        "metal.support",
        "metal.unified_memory",
        "macos.build",
    )
    facet = await AcceleratorIdentityService(metal_observer=_Observer(first)).observe(required)

    assert facet is not None
    assert facet.provider == "metal"
    assert facet.identity_quality is IdentityQuality.EXACT
    assert facet.devices[0].gpu_core_count == 10
    changed = facet.model_copy(update={"macos_build": "25D126"})
    assert collect_environment(facet).environment_id != collect_environment(changed).environment_id


@pytest.mark.anyio
async def test_malformed_metal_output_is_partial_instead_of_exact(tmp_path: Path) -> None:
    snapshot = _parse_metal_identity(
        {"SPDisplaysDataType": "changed schema"},
        product_version="",
        build="",
        memory="not-a-number",
    )
    required: tuple[AcceleratorIdentityRequirement, ...] = (
        "metal.devices",
        "metal.support",
        "metal.unified_memory",
        "macos.build",
    )
    facet = await AcceleratorIdentityService(metal_observer=_Observer(snapshot)).observe(required)

    assert facet is not None
    assert facet.identity_quality is IdentityQuality.PARTIAL
    assert facet.missing_fields == required
    assert facet.status == "unsupported"


def test_cuda_and_metal_requirements_cannot_be_mixed() -> None:
    with pytest.raises(ValueError, match="cannot mix CUDA and Metal"):
        WorkloadEnvironmentIdentityConfig(required=("cuda.devices", "metal.devices"))
