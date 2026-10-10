"""Explicit release discovery and isolated MCP environment verification."""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen

from packaging.version import InvalidVersion, Version

from flameox import __version__
from flameox.installation import verify_releases
from flameox.providers.environment import SetupFailure
from flameox.setup import ClientUpdatePlan

LATEST_RELEASE_URL = "https://pypi.org/pypi/flameox/json"
RELEASE_METADATA_MAX_BYTES = 1024 * 1024


def release_version(value: str) -> str:
    try:
        version = Version(value)
    except InvalidVersion as error:
        raise SetupFailure(f"Invalid Flameox release version: {value!r}") from error
    if version.local is not None:
        raise SetupFailure("Updates require a published release version, without a local suffix.")
    return str(version)


def latest_release_version() -> str:
    request = Request(
        LATEST_RELEASE_URL,
        headers={"Accept": "application/json", "User-Agent": f"flameox/{__version__}"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            data = response.read(RELEASE_METADATA_MAX_BYTES + 1)
        if len(data) > RELEASE_METADATA_MAX_BYTES:
            raise ValueError("release metadata exceeds the size limit")
        payload = json.loads(data)
        value = payload["info"]["version"]
        if payload["info"]["name"] != "flameox" or not isinstance(value, str):
            raise ValueError("unexpected package metadata")
        version = release_version(value)
        if Version(version).is_prerelease or Version(version).is_devrelease:
            raise ValueError("latest release is not stable")
        return version
    except (OSError, URLError, ValueError, KeyError, TypeError, RecursionError) as error:
        raise SetupFailure(
            f"Could not check the latest Flameox release on PyPI: {error}. "
            "Retry, or select a release with --version."
        ) from error


def prepare_updated_releases(plans: list[ClientUpdatePlan], timeout_seconds: int) -> None:
    """Verify every selected environment before publishing any client pin."""
    verify_releases(
        ((plan.requirement, plan.environment) for plan in plans), timeout_seconds=timeout_seconds
    )
