from __future__ import annotations

import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from pydantic import JsonValue

from flamo.domain.identity import digest_model
from flamo.domain.models import EnvironmentRecord, IdentityQuality, utc_now


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def collect_environment() -> EnvironmentRecord:
    fields: dict[str, JsonValue] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "packages": {
            package: package_version
            for package in ("duckdb", "flamo", "mcp", "pyarrow", "pydantic")
            if (package_version := _package_version(package)) is not None
        },
        "executable": sys.executable,
    }
    identity = digest_model({"identity_quality": "exact", "fields": fields})
    return EnvironmentRecord(
        environment_id=identity,
        observed_at=utc_now(),
        identity_quality=IdentityQuality.EXACT,
        fields=fields,
    )
