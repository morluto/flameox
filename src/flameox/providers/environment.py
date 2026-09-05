"""Provider environment contracts shared by CLI setup and live preparation."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Literal

from packaging.requirements import Requirement

from flameox import __version__
from flameox.providers.availability import (
    MANAGED_PROVIDER_EXTRAS,
    SYSTEM_PROVIDER_GUIDANCE,
    WORKLOAD_PROVIDER_GUIDANCE,
)

DEFAULT_PREPARATION_TIMEOUT_SECONDS = 1_800
MAX_PREPARATION_TIMEOUT_SECONDS = 3_600


class SetupFailure(RuntimeError):
    pass


class ProviderSelectionFailure(SetupFailure):
    pass


@dataclass(frozen=True, slots=True)
class ExternalRequirement:
    provider_id: str
    guidance: str


@dataclass(frozen=True, slots=True)
class ProviderPreparation:
    requested_providers: list[str]
    prepared_managed_providers: list[str]
    external_requirements: list[ExternalRequirement]
    preparation_command: list[str]
    launcher_command: str
    launcher_args: list[str]
    activation_status: Literal["ready", "restart_required", "unknown", "not_applicable"] = "unknown"

    @property
    def preparation_status(self) -> Literal["prepared", "not_applicable"]:
        return "prepared" if self.prepared_managed_providers else "not_applicable"

    @property
    def restart_required(self) -> bool | None:
        if not self.prepared_managed_providers or self.activation_status in {
            "ready",
            "not_applicable",
        }:
            return False
        return True if self.activation_status == "restart_required" else None

    @property
    def workload_requirements(self) -> list[ExternalRequirement]:
        return [
            ExternalRequirement(provider, WORKLOAD_PROVIDER_GUIDANCE[provider])
            for provider in self.requested_providers
            if provider in WORKLOAD_PROVIDER_GUIDANCE
        ]


def active_provider_status(providers: list[str]) -> Literal["ready", "restart_required", "unknown"]:
    """Compare the active distribution against the complete requested release contract."""
    try:
        distribution = importlib.metadata.distribution("flameox")
        if distribution.version != __version__:
            return "unknown"
        extras = {MANAGED_PROVIDER_EXTRAS[item] for item in providers}
        requirements = [Requirement(item) for item in distribution.requires or []]
        for requirement in requirements:
            if requirement.marker is not None and not any(
                requirement.marker.evaluate({"extra": extra}) for extra in extras | {""}
            ):
                continue
            try:
                version = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                return "restart_required"
            if not requirement.specifier.contains(version, prereleases=True):
                return "restart_required"
        return "ready" if requirements else "unknown"
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return "unknown"


def _validate_providers(providers: list[str]) -> None:
    unknown = sorted(
        set(providers).difference(MANAGED_PROVIDER_EXTRAS).difference(SYSTEM_PROVIDER_GUIDANCE)
    )
    if unknown:
        supported = ", ".join(sorted(MANAGED_PROVIDER_EXTRAS | SYSTEM_PROVIDER_GUIDANCE))
        raise ProviderSelectionFailure(
            f"Unknown provider {unknown[0]!r}; choose one of: {supported}"
        )


def mcp_launcher(providers: list[str]) -> tuple[str, list[str]]:
    """Return a version-bound MCP launcher for client configuration."""

    _validate_providers(providers)
    extras = sorted(
        {
            MANAGED_PROVIDER_EXTRAS[provider]
            for provider in providers
            if provider in MANAGED_PROVIDER_EXTRAS
        }
    )
    extras_suffix = f"[{','.join(extras)}]" if extras else ""
    requirement = f"flameox{extras_suffix}=={__version__}"
    return (
        "uvx",
        ["--python", "3.12", "--from", requirement, "flameox"],
    )


def external_provider_requirements(providers: list[str]) -> list[ExternalRequirement]:
    _validate_providers(providers)
    return [
        ExternalRequirement(provider, SYSTEM_PROVIDER_GUIDANCE[provider])
        for provider in dict.fromkeys(providers)
        if provider in SYSTEM_PROVIDER_GUIDANCE
    ]
