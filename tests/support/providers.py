from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

PACKAGE_PROVIDERS = {
    "requires_coverage": "coverage",
    "requires_memray": "memray",
    "requires_torch": "torch",
    "requires_triton": "triton",
}
CONFIGURED_EXECUTABLE_PROVIDERS = {
    "requires_nvbench": "FLAMEOX_NVBENCH_EXECUTABLE",
}
EXECUTABLE_PROVIDERS = {
    "requires_bwrap": "bwrap",
    "requires_cargo": "cargo",
    "requires_perf": "perf",
    "requires_pyspy": "py-spy",
    "requires_systemd": "systemd-run",
    "requires_compute_sanitizer": "compute-sanitizer",
    "requires_ncu": "ncu",
}
PROVIDER_MARKERS = frozenset(
    {
        *PACKAGE_PROVIDERS,
        *CONFIGURED_EXECUTABLE_PROVIDERS,
        *EXECUTABLE_PROVIDERS,
        "requires_cute",
        "requires_perfetto",
        "requires_toxiproxy",
    }
)


def trace_processor_path() -> Path | None:
    configured = os.environ.get("FLAMEOX_TRACE_PROCESSOR")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate
        return None
    candidates = sorted(
        (Path.home() / ".local" / "share" / "perfetto" / "prebuilts").glob(
            "trace_processor_shell-*"
        )
    )
    return candidates[-1] if candidates else None


@cache
def systemd_user_scope_available() -> bool:
    systemd_run = shutil.which("systemd-run")
    true_executable = shutil.which("true")
    if systemd_run is None or true_executable is None:
        return False
    try:
        probe = subprocess.run(
            (
                systemd_run,
                "--user",
                "--scope",
                "--quiet",
                "--collect",
                "--expand-environment=no",
                "--",
                true_executable,
            ),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def provider_available(marker: str) -> bool:
    if marker == "requires_perfetto":
        return (
            importlib.util.find_spec("perfetto") is not None and trace_processor_path() is not None
        )
    if marker == "requires_toxiproxy":
        configured = os.environ.get("FLAMEOX_TOXIPROXY_SERVER")
        return bool(configured and Path(configured).is_file())
    if marker == "requires_systemd":
        return systemd_user_scope_available()
    if marker == "requires_triton":
        configured = os.environ.get("FLAMEOX_TRITON_PYTHON")
        if configured:
            return Path(configured).expanduser().is_file()
        return importlib.util.find_spec("triton") is not None
    if marker == "requires_cute":
        configured = os.environ.get("FLAMEOX_CUTE_WORKLOAD")
        return bool(configured and Path(configured).expanduser().is_file())
    if marker == "requires_compute_sanitizer":
        return shutil.which("compute-sanitizer") is not None and shutil.which("nvcc") is not None
    if marker == "requires_ncu":
        executable = shutil.which("ncu")
        if executable is None:
            return False
        from flameox.adapters.nsight_compute import find_ncu_report_interface

        return find_ncu_report_interface(executable=Path(executable)) is not None
    configured_variable = CONFIGURED_EXECUTABLE_PROVIDERS.get(marker)
    if configured_variable is not None:
        configured = os.environ.get(configured_variable)
        return bool(configured and Path(configured).expanduser().is_file())
    package = PACKAGE_PROVIDERS.get(marker)
    if package is not None:
        return importlib.util.find_spec(package) is not None
    executable = EXECUTABLE_PROVIDERS.get(marker)
    if executable is not None:
        return shutil.which(executable) is not None
    raise ValueError(f"Unknown provider marker: {marker}")


def provider_inventory() -> tuple[tuple[str, bool], ...]:
    return tuple(
        (marker, provider_available(marker))
        for marker in (
            *PACKAGE_PROVIDERS,
            *CONFIGURED_EXECUTABLE_PROVIDERS,
            *EXECUTABLE_PROVIDERS,
            "requires_cute",
            "requires_perfetto",
            "requires_toxiproxy",
        )
    )


def require_trace_processor() -> Path:
    path = trace_processor_path()
    if path is None:
        import pytest

        pytest.skip(
            "optional provider unavailable: install the Perfetto Python package "
            "and a local Trace Processor binary"
        )
    return path
