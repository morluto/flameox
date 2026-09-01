from __future__ import annotations

import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True, slots=True)
class ProviderInstallation:
    command: list[str]
    providers: list[str]


def managed_tool_extras(tool_directory: Path) -> set[str]:
    """Read the optional extras uv will replace when reinstalling the Flameox tool."""

    receipt = tool_directory / "flameox" / "uv-receipt.toml"
    if not receipt.is_file():
        return set()
    try:
        document: Any = tomllib.loads(receipt.read_text())
        requirements = document["tool"]["requirements"]
        flameox = next(item for item in requirements if item.get("name") == "flameox")
        extras = flameox.get("extras", [])
    except (OSError, KeyError, StopIteration, TypeError, tomllib.TOMLDecodeError) as exc:
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


def provider_install_command(
    providers: list[str], *, uv: Path, installed_extras: set[str] | None = None
) -> list[str]:
    unknown = sorted(
        set(providers).difference(PYTHON_PROVIDER_EXTRAS).difference(SYSTEM_PROVIDER_GUIDANCE)
    )
    if unknown:
        supported = ", ".join(sorted(PYTHON_PROVIDER_EXTRAS | SYSTEM_PROVIDER_GUIDANCE))
        raise SetupFailure(f"Unknown provider {unknown[0]!r}; choose one of: {supported}")
    requested_extras = {
        PYTHON_PROVIDER_EXTRAS[item] for item in providers if item in PYTHON_PROVIDER_EXTRAS
    }
    if not requested_extras:
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


def install_providers(providers: list[str]) -> ProviderInstallation:
    uv_text = shutil.which("uv")
    if uv_text is None:
        raise SetupFailure("Provider installation requires uv on PATH.")
    uv = Path(uv_text)
    installed_extras = managed_tool_extras(_uv_tool_directory(uv))
    command = provider_install_command(
        providers,
        uv=uv,
        installed_extras=installed_extras,
    )
    if command:
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SetupFailure(f"uv tool install exited with status {completed.returncode}.")
    final_extras = installed_extras | {
        PYTHON_PROVIDER_EXTRAS[item] for item in providers if item in PYTHON_PROVIDER_EXTRAS
    }
    managed_providers = {
        provider for provider, extra in PYTHON_PROVIDER_EXTRAS.items() if extra in final_extras
    }
    selected_system = set(providers).intersection(SYSTEM_PROVIDER_GUIDANCE)
    return ProviderInstallation(command, sorted(managed_providers | selected_system))


def provider_guidance(providers: list[str]) -> list[str]:
    return [
        SYSTEM_PROVIDER_GUIDANCE[item] for item in providers if item in SYSTEM_PROVIDER_GUIDANCE
    ]
