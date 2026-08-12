from __future__ import annotations

import sys
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
    ProcessResult,
    process_termination_from_returncode,
)
from flameox.execution import ExecutionOutcome, ExecutionRequest, ProcessContainment

_INVENTORY = b"""\
<?xml version="1.0"?>
<nvidia_smi_log>
  <driver_version>575.57.08</driver_version>
  <cuda_version>12.9</cuda_version>
  <gpu>
    <product_name>NVIDIA H100 80GB HBM3</product_name>
    <compute_cap>9.0</compute_cap>
    <fb_memory_usage><total>81559 MiB</total></fb_memory_usage>
    <mig_mode><current_mig>Disabled</current_mig></mig_mode>
  </gpu>
  <gpu>
    <product_name>NVIDIA H100 80GB HBM3</product_name>
    <compute_cap>9.0</compute_cap>
    <fb_memory_usage><total>81559 MiB</total></fb_memory_usage>
    <mig_mode><current_mig>Enabled</current_mig></mig_mode>
  </gpu>
</nvidia_smi_log>
"""

_TOPOLOGY = """\
        GPU0    GPU1    CPU Affinity
GPU0     X      NV18    0-47
GPU1    NV18     X      0-47
"""


class _Broker:
    def __init__(self, inventory: bytes = _INVENTORY, topology: str = _TOPOLOGY) -> None:
        self.inventory = inventory
        self.topology = topology
        self.requests: list[ExecutionRequest] = []

    async def run(self, request: ExecutionRequest, **_: object) -> ExecutionOutcome:
        self.requests.append(request)
        stdout = self.topology.encode() if request.argv[-2:] == ("topo", "-m") else self.inventory
        return ExecutionOutcome(
            process=ProcessResult(termination=process_termination_from_returncode(0)),
            stdout=stdout,
            stderr=b"",
            resolved_executable=Path(request.argv[0]),
            containment=ProcessContainment.PROCESS_GROUP,
        )


class _FailedBroker:
    def __init__(self, *, stderr: bytes = b"", permission_denied: bool = False) -> None:
        self.stderr = stderr
        self.permission_denied = permission_denied

    async def run(self, request: ExecutionRequest, **_: object) -> ExecutionOutcome:
        if self.permission_denied:
            raise PermissionError(request.argv[0])
        return ExecutionOutcome(
            process=ProcessResult(termination=process_termination_from_returncode(1)),
            stdout=b"",
            stderr=self.stderr,
            resolved_executable=Path(request.argv[0]),
            containment=ProcessContainment.PROCESS_GROUP,
        )


@pytest.mark.anyio
async def test_declared_accelerator_identity_is_structured_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = _Broker()
    monkeypatch.setattr(
        "flameox.command_binding.shutil.which", lambda _name, path=None: sys.executable
    )
    required: tuple[AcceleratorIdentityRequirement, ...] = (
        "cuda.driver",
        "cuda.runtime",
        "cuda.devices",
        "cuda.peer_topology",
    )

    facet = await AcceleratorIdentityService(tmp_path, broker=broker).observe(required)

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
        "width": 18,
    }
    assert [request.argv[1:] for request in broker.requests] == [("-q", "-x"), ("topo", "-m")]
    assert all(request.allowed_working_roots == (tmp_path,) for request in broker.requests)

    environment = collect_environment(facet)
    assert environment.identity_quality is IdentityQuality.EXACT
    assert environment.fields["accelerator"] == facet.model_dump(mode="json")


@pytest.mark.anyio
async def test_missing_required_accelerator_is_partial_not_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flameox.command_binding.shutil.which", lambda _name, path=None: None)
    required: tuple[AcceleratorIdentityRequirement, ...] = ("cuda.driver", "cuda.devices")

    facet = await AcceleratorIdentityService(tmp_path).observe(required)

    assert facet is not None
    assert facet.status == "missing"
    assert facet.identity_quality is IdentityQuality.PARTIAL
    assert facet.missing_fields == required
    environment = collect_environment(facet)
    assert environment.identity_quality is IdentityQuality.PARTIAL
    assert environment.missing_fields == required


@pytest.mark.parametrize(
    ("broker", "expected"),
    [
        (_FailedBroker(permission_denied=True), "permission_denied"),
        (_FailedBroker(stderr=b"Feature not supported"), "unsupported"),
        (_FailedBroker(stderr=b"Driver communication failed"), "unknown"),
    ],
)
@pytest.mark.anyio
async def test_accelerator_probe_failure_states_remain_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broker: _FailedBroker,
    expected: str,
) -> None:
    monkeypatch.setattr(
        "flameox.command_binding.shutil.which", lambda _name, path=None: sys.executable
    )

    facet = await AcceleratorIdentityService(tmp_path, broker=broker).observe(("cuda.devices",))

    assert facet is not None
    assert facet.status == expected
    assert facet.identity_quality is IdentityQuality.PARTIAL


@pytest.mark.anyio
async def test_unknown_inventory_fields_remain_distinct_and_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flameox.command_binding.shutil.which", lambda _name, path=None: sys.executable
    )
    broker = _Broker(
        b"""<nvidia_smi_log><driver_version>N/A</driver_version><gpu>
        <product_name>H100</product_name><compute_cap>N/A</compute_cap>
        <fb_memory_usage><total>N/A</total></fb_memory_usage>
        <mig_mode><current_mig>N/A</current_mig></mig_mode>
        </gpu></nvidia_smi_log>"""
    )

    facet = await AcceleratorIdentityService(tmp_path, broker=broker).observe(
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
