from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from flameox import __version__

PYTHON_PROVIDER_EXTRAS = {
    "aiperf": "inference",
    "memray": "memory",
    "otlp": "trace",
    "perfetto": "trace",
    "py-spy": "cpu",
    "torch": "torch",
}

SYSTEM_PROVIDER_GUIDANCE = {
    "compute-sanitizer": "Install NVIDIA Compute Sanitizer with the CUDA Toolkit.",
    "nsight-compute": "Install NVIDIA Nsight Compute with its extras/python interface.",
    "nsight-systems": "Install NVIDIA Nsight Systems and make nsys available on PATH.",
    "nvbench": "Build the target benchmark with NVBench and verify CUDA device access.",
    "perf": "Install Linux perf for the running kernel and grant profiling permission.",
    "perfetto": "Install Perfetto Trace Processor and make trace_processor_shell available.",
    "rocprofv3": "Install ROCProfiler SDK and make rocprofv3 available on PATH.",
    "triton": "Install Triton in the target Python environment and verify device access.",
}


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

    @property
    def preparation_status(self) -> Literal["prepared", "not_applicable"]:
        return "prepared" if self.prepared_managed_providers else "not_applicable"

    @property
    def restart_required(self) -> bool:
        return bool(self.prepared_managed_providers)


def _validate_providers(providers: list[str]) -> None:
    unknown = sorted(
        set(providers).difference(PYTHON_PROVIDER_EXTRAS).difference(SYSTEM_PROVIDER_GUIDANCE)
    )
    if unknown:
        supported = ", ".join(sorted(PYTHON_PROVIDER_EXTRAS | SYSTEM_PROVIDER_GUIDANCE))
        raise ProviderSelectionFailure(
            f"Unknown provider {unknown[0]!r}; choose one of: {supported}"
        )


def mcp_launcher(providers: list[str]) -> tuple[str, list[str]]:
    """Return a version-bound MCP launcher for client configuration."""

    extras = sorted(
        {
            PYTHON_PROVIDER_EXTRAS[provider]
            for provider in providers
            if provider in PYTHON_PROVIDER_EXTRAS
        }
    )
    extras_suffix = f"[{','.join(extras)}]" if extras else ""
    requirement = f"flameox{extras_suffix}=={__version__}"
    return (
        "uvx",
        ["--python", "3.12", "--from", requirement, "flameox"],
    )


def prepare_providers(providers: list[str], project_root: Path) -> ProviderPreparation:
    requested = list(dict.fromkeys(providers))
    _validate_providers(requested)
    managed = [item for item in requested if item in PYTHON_PROVIDER_EXTRAS]
    external = [
        ExternalRequirement(item, SYSTEM_PROVIDER_GUIDANCE[item])
        for item in requested
        if item in SYSTEM_PROVIDER_GUIDANCE
    ]
    launcher_command, launcher_args = mcp_launcher(managed)
    server_args = [*launcher_args, "mcp", "serve", "--project-root", str(project_root)]
    preparation_command: list[str] = []

    if managed:
        uvx = shutil.which(launcher_command)
        if uvx is None:
            raise SetupFailure("Provider preparation requires uvx on PATH.")
        preparation_command = [uvx, *launcher_args, "--version"]
        try:
            completed = subprocess.run(
                preparation_command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
            )
        except OSError as error:
            raise SetupFailure("uvx could not prepare the provider environment.") from error
        except subprocess.TimeoutExpired as error:
            raise SetupFailure("uvx provider preparation exceeded 300 seconds.") from error
        if completed.returncode != 0:
            raise SetupFailure(
                f"uvx provider preparation exited with status {completed.returncode}."
            )

    return ProviderPreparation(
        requested,
        managed,
        external,
        preparation_command,
        launcher_command,
        server_args,
    )
