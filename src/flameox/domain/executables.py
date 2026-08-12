from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator

from flameox.models import ContractModel


class ExecutableTrustPolicy(StrEnum):
    PROJECT_BOUND = "project_bound"
    MANAGED_TOOL = "managed_tool"
    TRUSTED_HOST_TOOL = "trusted_host_tool"
    EXACT_PATH = "exact_path"


class ExecutableResolutionOrigin(StrEnum):
    EXPLICIT_PATH = "explicit_path"
    PATH_SEARCH = "path_search"


class ExecutableResolutionRequest(ContractModel):
    token: str
    cwd: Path
    environment: dict[str, str] = Field(default_factory=dict)
    policy: ExecutableTrustPolicy
    allowed_roots: tuple[Path, ...] = ()

    @field_validator("token")
    @classmethod
    def validate_token(cls, value: str) -> str:
        if not value or "\x00" in value:
            raise ValueError("executable token must be non-empty and cannot contain NUL")
        return value


class ExecutableIdentity(ContractModel):
    sha256: str
    size: int = Field(ge=0)
    mode: int = Field(ge=0)
    device: int = Field(ge=0)
    inode: int = Field(ge=0)
    modified_ns: int = Field(ge=0)


class ExecutablePolicyDecision(ContractModel):
    policy: ExecutableTrustPolicy
    allowed: bool
    matched_root: Path | None = None


class ResolvedExecutable(ContractModel):
    requested_token: str
    invocation_path: Path
    canonical_target: Path
    origin: ExecutableResolutionOrigin
    matched_path_entry: Path | None = None
    identity: ExecutableIdentity
    policy_decision: ExecutablePolicyDecision
