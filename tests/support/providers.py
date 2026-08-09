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
}
EXECUTABLE_PROVIDERS = {
    "requires_bwrap": "bwrap",
    "requires_cargo": "cargo",
    "requires_compute_sanitizer": "compute-sanitizer",
    "requires_perf": "perf",
    "requires_pyspy": "py-spy",
    "requires_systemd": "systemd-run",
}
PROVIDER_MARKERS = frozenset(
    {
        *PACKAGE_PROVIDERS,
        *EXECUTABLE_PROVIDERS,
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
            *EXECUTABLE_PROVIDERS,
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
