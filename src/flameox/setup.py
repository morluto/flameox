from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

from flameox import __version__
from flameox.providers.availability import (
    MANAGED_PROVIDER_EXTRAS,
    SYSTEM_PROVIDER_GUIDANCE,
)

DEFAULT_PREPARATION_TIMEOUT_SECONDS = 1_800
MAX_PREPARATION_TIMEOUT_SECONDS = 3_600


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
        set(providers).difference(MANAGED_PROVIDER_EXTRAS).difference(SYSTEM_PROVIDER_GUIDANCE)
    )
    if unknown:
        supported = ", ".join(sorted(MANAGED_PROVIDER_EXTRAS | SYSTEM_PROVIDER_GUIDANCE))
        raise ProviderSelectionFailure(
            f"Unknown provider {unknown[0]!r}; choose one of: {supported}"
        )


def mcp_launcher(providers: list[str]) -> tuple[str, list[str]]:
    """Return a version-bound MCP launcher for client configuration."""

    extras = sorted(
        {
            MANAGED_PROVIDER_EXTRAS[provider]
            for provider in providers
            if provider in MANAGED_PROVIDER_EXTRAS
        }
    )
    extras_suffix = f"[{','.join(extras)}]" if extras else ""
    requirement = f"flameox{extras_suffix}=={__version__}"
    return (
        "uvx",
        ["--python", "3.12", "--from", requirement, "flameox"],
    )


def _decode_stderr(stderr: bytes | str | None) -> str:
    if not stderr:
        return ""
    return stderr.strip() if isinstance(stderr, str) else stderr.decode(errors="replace").strip()


def _failure_message(message: str, stderr: bytes | str | None) -> str:
    diagnostic = _decode_stderr(stderr)
    return f"{message}\n\nuvx stderr:\n{diagnostic}" if diagnostic else message


def prepare_providers(
    providers: list[str],
    timeout_seconds: int = DEFAULT_PREPARATION_TIMEOUT_SECONDS,
) -> ProviderPreparation:
    if not 1 <= timeout_seconds <= MAX_PREPARATION_TIMEOUT_SECONDS:
        raise SetupFailure(
            f"timeout_seconds must be between 1 and {MAX_PREPARATION_TIMEOUT_SECONDS}"
        )
    requested = list(dict.fromkeys(providers))
    _validate_providers(requested)
    managed = [item for item in requested if item in MANAGED_PROVIDER_EXTRAS]
    external = [
        ExternalRequirement(item, SYSTEM_PROVIDER_GUIDANCE[item])
        for item in requested
        if item in SYSTEM_PROVIDER_GUIDANCE
    ]
    launcher_command, launcher_args = mcp_launcher(managed)
    server_args = [*launcher_args, "mcp", "serve"]
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
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
        except OSError as error:
            raise SetupFailure("uvx could not prepare the provider environment.") from error
        except subprocess.TimeoutExpired as error:
            raise SetupFailure(
                _failure_message(
                    f"uvx provider preparation exceeded {timeout_seconds} seconds.", error.stderr
                )
            ) from error
        if completed.returncode != 0:
            raise SetupFailure(
                _failure_message(
                    f"uvx provider preparation exited with status {completed.returncode}.",
                    completed.stderr,
                )
            )

    return ProviderPreparation(
        requested,
        managed,
        external,
        preparation_command,
        launcher_command,
        server_args,
    )
