"""Bounded verification of version-pinned MCP release environments."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path

from packaging.requirements import Requirement

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

RELEASE_METADATA_PROBE = (
    "import importlib.metadata,json; d=importlib.metadata.distribution('flameox'); "
    "print(json.dumps({'version':d.version,'extras':d.metadata.get_all('Provides-Extra') or []}))"
)


def verify_releases(
    releases: Iterable[tuple[str, dict[str, str]]], *, timeout_seconds: float
) -> None:
    """Verify distinct final launcher environments under one overall deadline."""
    deadline = time.monotonic() + timeout_seconds
    verified: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for requirement, environment in releases:
        key = (requirement, tuple(sorted(environment.items())))
        if key in verified:
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SetupFailure("Flameox release preparation exceeded its overall timeout.")
        _verify_release(requirement, environment=environment, timeout_seconds=remaining)
        verified.add(key)


def _verify_release(
    requirement: str, *, environment: dict[str, str], timeout_seconds: float
) -> None:
    """Verify the distribution and tool catalog before publishing a launcher."""
    deadline = time.monotonic() + timeout_seconds
    broker = SubprocessBroker()
    with tempfile.TemporaryDirectory(prefix="flameox-install-") as directory:
        scratch = Path(directory).resolve()
        try:
            environment_names = set(INSTALLER_ENVIRONMENT_ALLOWLIST) | {
                "HOME",
                "USERPROFILE",
                "XDG_CACHE_HOME",
                "XDG_CONFIG_HOME",
                *(name for name in os.environ if name.startswith("UV_")),
                *(name for name in environment if name.startswith("UV_")),
            }
            overrides = {
                name: value for name, value in environment.items() if name in environment_names
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
                    "Release preparation cannot forward uv credential environment variables "
                    f"({', '.join(unsupported)}). Prepare these launchers manually using their "
                    "authenticated package environment."
                )
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
                    raise SetupFailure("Flameox release preparation exceeded its overall timeout.")
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
                        raise SetupFailure(f"Prepared {requirement} did not expose a tool catalog.")
        except ProcessExecutionError as error:
            diagnostic = (error.stderr or b"").decode(errors="replace")
            raise SetupFailure(
                f"Flameox release preparation failed: {error}\n{diagnostic}"
            ) from error
        except (DomainError, OSError, ValueError, KeyError, TypeError, RecursionError) as error:
            raise SetupFailure(
                f"Could not verify the Flameox release environment: {error}"
            ) from error
