from __future__ import annotations

import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import portalocker

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
    configured_managed_providers: list[str]
    external_requirements: list[ExternalRequirement]
    install_command: list[str]
    launcher_command: str
    launcher_args: list[str]

    @property
    def changed(self) -> bool:
        return bool(self.install_command)

    @property
    def installation_status(self) -> Literal["installed", "already_configured", "not_applicable"]:
        if not any(item in PYTHON_PROVIDER_EXTRAS for item in self.requested_providers):
            return "not_applicable"
        return "installed" if self.changed else "already_configured"


def managed_tool_extras(tool_directory: Path) -> set[str]:
    """Read the optional extras uv will replace when reinstalling the Flameox tool."""

    receipt = tool_directory / "flameox" / "uv-receipt.toml"
    if not receipt.is_file():
        return set()
    try:
        document: Any = tomllib.loads(receipt.read_text())
        requirements = document["tool"]["requirements"]
        flameox = next(
            item
            for item in requirements
            if isinstance(item, dict) and item.get("name") == "flameox"
        )
        extras = flameox.get("extras", [])
    except (
        OSError,
        KeyError,
        StopIteration,
        TypeError,
        tomllib.TOMLDecodeError,
    ) as exc:
        raise SetupFailure(f"Cannot read the existing uv tool receipt: {receipt}") from exc
    if not isinstance(extras, list) or not all(isinstance(item, str) for item in extras):
        raise SetupFailure(f"The existing uv tool receipt has invalid extras: {receipt}")
    known_extras = set(PYTHON_PROVIDER_EXTRAS.values())
    return {item for item in extras if item in known_extras}


def _uv_tool_directory(uv: Path) -> Path:
    completed = subprocess.run(
        [str(uv), "tool", "dir"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise SetupFailure("uv tool dir could not locate the managed tool environment.")
    return Path(completed.stdout.strip())


def _validate_providers(providers: list[str]) -> None:
    unknown = sorted(
        set(providers).difference(PYTHON_PROVIDER_EXTRAS).difference(SYSTEM_PROVIDER_GUIDANCE)
    )
    if unknown:
        supported = ", ".join(sorted(PYTHON_PROVIDER_EXTRAS | SYSTEM_PROVIDER_GUIDANCE))
        raise ProviderSelectionFailure(
            f"Unknown provider {unknown[0]!r}; choose one of: {supported}"
        )


def provider_install_command(
    providers: list[str], *, uv: Path, installed_extras: set[str] | None = None
) -> list[str]:
    _validate_providers(providers)
    requested_extras = {
        PYTHON_PROVIDER_EXTRAS[item] for item in providers if item in PYTHON_PROVIDER_EXTRAS
    }
    if not requested_extras or requested_extras.issubset(installed_extras or set()):
        return []
    extras = sorted((installed_extras or set()) | requested_extras)
    requirement = f"flameox[{','.join(extras)}]=={__version__}"
    return [
        str(uv),
        "tool",
        "install",
        "--force",
        "--python",
        "3.12",
        "--prerelease",
        "allow",
        requirement,
    ]


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


def prepare_providers(providers: list[str]) -> ProviderPreparation:
    requested = list(dict.fromkeys(providers))
    _validate_providers(requested)
    managed = [item for item in requested if item in PYTHON_PROVIDER_EXTRAS]
    external = [
        ExternalRequirement(item, SYSTEM_PROVIDER_GUIDANCE[item])
        for item in requested
        if item in SYSTEM_PROVIDER_GUIDANCE
    ]
    if not managed:
        launcher_command, launcher_args = mcp_launcher([])
        return ProviderPreparation(
            requested,
            [],
            external,
            [],
            launcher_command,
            launcher_args,
        )

    uv_text = shutil.which("uv")
    if uv_text is None:
        raise SetupFailure("Provider installation requires uv on PATH.")
    uv = Path(uv_text)
    tool_directory = _uv_tool_directory(uv)
    try:
        with portalocker.Lock(
            tool_directory / ".flameox-setup.lock",
            mode="a+",
            timeout=10,
            encoding="utf-8",
        ):
            installed_extras = managed_tool_extras(tool_directory)
            command = provider_install_command(
                managed,
                uv=uv,
                installed_extras=installed_extras,
            )
            if command:
                completed = subprocess.run(command, check=False)
                if completed.returncode != 0:
                    raise SetupFailure(
                        f"uv tool install exited with status {completed.returncode}."
                    )
            final_extras = managed_tool_extras(tool_directory) if command else installed_extras
            requested_extras = {PYTHON_PROVIDER_EXTRAS[item] for item in managed}
            missing_extras = requested_extras.difference(final_extras)
            if missing_extras:
                raise SetupFailure(
                    "uv reported success but the managed tool receipt is missing extras: "
                    + ", ".join(sorted(missing_extras))
                )
    except portalocker.exceptions.LockException as error:
        raise SetupFailure("Timed out waiting to update the managed Flameox tool.") from error

    configured = sorted(
        provider for provider, extra in PYTHON_PROVIDER_EXTRAS.items() if extra in final_extras
    )
    launcher_command, launcher_args = mcp_launcher(configured)
    return ProviderPreparation(
        requested,
        configured,
        external,
        command,
        launcher_command,
        launcher_args,
    )
