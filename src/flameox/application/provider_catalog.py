from __future__ import annotations

from dataclasses import dataclass

from flameox.domain import CapabilityExtra


@dataclass(frozen=True, slots=True)
class ManagedProviderDefinition:
    name: str
    executable: str | None
    extra: CapabilityExtra
    requirement: str
    version_args: tuple[str, ...]
    supported_modes: tuple[str, ...]
    supported_formats: tuple[str, ...]
    features: tuple[str, ...]
    limitations: tuple[str, ...] = ()


SHRINKRAY_PROVIDER = ManagedProviderDefinition(
    name="shrinkray",
    executable="shrinkray",
    extra=CapabilityExtra.REDUCTION,
    requirement="shrinkray==26.7.8.0",
    version_args=("--version",),
    supported_modes=("reduction",),
    supported_formats=("generic-file",),
    features=("test_case_reduction", "bounded_predicate", "history"),
    limitations=(
        "The qualified profile disables LLM, Python, formatter, restart, and implicit "
        "language-download paths.",
    ),
)

NVIDIA_NVML_PROVIDER = ManagedProviderDefinition(
    name="nvidia-nvml",
    executable=None,
    extra=CapabilityExtra.HARDWARE,
    requirement="nvidia-ml-py==13.610.43",
    version_args=(),
    supported_modes=("identity",),
    supported_formats=("flameox.nvml-snapshot.v1",),
    features=("nvidia_identity", "stable_device_identity", "peer_topology"),
    limitations=("Read-only NVML queries run in an isolated provider process.",),
)

MANAGED_PROVIDERS = {
    NVIDIA_NVML_PROVIDER.name: NVIDIA_NVML_PROVIDER,
    SHRINKRAY_PROVIDER.name: SHRINKRAY_PROVIDER,
}


def managed_provider(name: str) -> ManagedProviderDefinition | None:
    return MANAGED_PROVIDERS.get(name)
