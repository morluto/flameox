from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

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


def provider_install_command(providers: list[str], *, uv: Path) -> list[str]:
    unknown = sorted(
        set(providers).difference(PYTHON_PROVIDER_EXTRAS).difference(SYSTEM_PROVIDER_GUIDANCE)
    )
    if unknown:
        supported = ", ".join(sorted(PYTHON_PROVIDER_EXTRAS | SYSTEM_PROVIDER_GUIDANCE))
        raise SetupFailure(f"Unknown provider {unknown[0]!r}; choose one of: {supported}")
    extras = sorted(
        {PYTHON_PROVIDER_EXTRAS[item] for item in providers if item in PYTHON_PROVIDER_EXTRAS}
    )
    if not extras:
        return []
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


def install_providers(providers: list[str]) -> list[str]:
    uv_text = shutil.which("uv")
    if uv_text is None:
        raise SetupFailure("Provider installation requires uv on PATH.")
    command = provider_install_command(providers, uv=Path(uv_text))
    if command:
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SetupFailure(f"uv tool install exited with status {completed.returncode}.")
    return command


def provider_guidance(providers: list[str]) -> list[str]:
    return [
        SYSTEM_PROVIDER_GUIDANCE[item] for item in providers if item in SYSTEM_PROVIDER_GUIDANCE
    ]
