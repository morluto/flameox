"""Explicit release discovery and isolated MCP environment verification."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

from flameox import __version__
from flameox.command_binding import ExecutableResolver
from flameox.environment_policy import is_dangerous_environment_name
from flameox.execution import (
    INSTALLER_ENVIRONMENT_ALLOWLIST,
    ExecutionRequest,
    ProcessExecutionError,
    SubprocessBroker,
)
from flameox.providers.environment import SetupFailure
from flameox.runtime_errors import DomainError
from flameox.setup import ClientUpdatePlan

LATEST_RELEASE_URL = "https://pypi.org/pypi/flameox/json"
RELEASE_METADATA_MAX_BYTES = 1024 * 1024
RELEASE_METADATA_PROBE = (
    "import importlib.metadata,json; d=importlib.metadata.distribution('flameox'); "
    "print(json.dumps({'version':d.version,'extras':d.metadata.get_all('Provides-Extra') or []}))"
)


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
    deadline = time.monotonic() + timeout_seconds
    broker = SubprocessBroker()
    verified: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    with tempfile.TemporaryDirectory(prefix="flameox-update-") as directory:
        scratch = Path(directory).resolve()
        try:
            for plan in plans:
                requirement = plan.requirement
                environment_names = set(INSTALLER_ENVIRONMENT_ALLOWLIST) | {
                    "HOME",
                    "USERPROFILE",
                    "XDG_CACHE_HOME",
                    "XDG_CONFIG_HOME",
                    *(name for name in os.environ if name.startswith("UV_")),
                    *(name for name in plan.environment if name.startswith("UV_")),
                }
                overrides = {
                    name: value
                    for name, value in plan.environment.items()
                    if name in environment_names
                }
                unsupported = sorted(
                    name
                    for name in environment_names
                    if name.startswith("UV_")
                    and is_dangerous_environment_name(name)
                    and (name in os.environ or name in overrides)
                )
                if unsupported:
                    raise SetupFailure(
                        "Update preparation cannot forward uv credential environment variables "
                        f"({', '.join(unsupported)}). Update these launchers manually using their "
                        "authenticated package environment."
                    )
                environment_key = (requirement, tuple(sorted(overrides.items())))
                if environment_key in verified:
                    continue
                binding = ExecutableResolver().require_host_tool(
                    "uvx", cwd=scratch, environment={**os.environ, **overrides}
                )
                requested = Requirement(requirement)
                expected_version = next(iter(requested.specifier)).version
                launcher = (
                    "uvx",
                    "--no-config",
                    "--no-sources",
                    "--python",
                    "3.12",
                    "--from",
                    requirement,
                )
                for arguments in (
                    ("python", "-c", RELEASE_METADATA_PROBE),
                    ("flameox", "mcp", "inspect"),
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise SetupFailure(
                            "Flameox update preparation exceeded its overall timeout."
                        )
                    result = broker.run_sync(
                        ExecutionRequest(
                            argv=(*launcher, *arguments),
                            executable_binding=binding,
                            cwd=scratch,
                            allowed_working_roots=(scratch,),
                            environment_allowlist=tuple(sorted(environment_names)),
                            environment_overrides=overrides,
                            timeout_seconds=remaining,
                            max_output_bytes=1024 * 1024,
                        )
                    )
                    if getattr(result.process.termination, "exit_code", None) != 0:
                        raise SetupFailure(
                            f"Could not prepare {requirement}.\n"
                            f"uvx stderr:\n{result.stderr.decode(errors='replace')}"
                        )
                    if arguments[0] == "python":
                        metadata = json.loads(result.stdout)
                        if metadata["version"] != expected_version:
                            raise SetupFailure(f"Prepared release does not match {requirement}.")
                        if not requested.extras.issubset(set(metadata["extras"])):
                            raise SetupFailure(
                                f"Prepared release does not provide extras for {requirement}."
                            )
                    else:
                        catalog = json.loads(result.stdout)
                        if not isinstance(catalog["tools"], list) or not catalog["tools"]:
                            raise SetupFailure(
                                f"Prepared {requirement} did not expose a tool catalog."
                            )
                verified.add(environment_key)
        except ProcessExecutionError as error:
            diagnostic = (error.stderr or b"").decode(errors="replace")
            raise SetupFailure(
                f"Flameox update preparation failed: {error}\n{diagnostic}"
            ) from error
        except (DomainError, OSError, ValueError, KeyError, TypeError, RecursionError) as error:
            raise SetupFailure(
                f"Could not verify the Flameox update environment: {error}"
            ) from error
