from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import tomli_w
from pydantic import Field

from flameox.models import ContractModel


class ContainmentPolicy(StrEnum):
    REQUIRED_FOR_MCP = "required_for_mcp"
    PREFERRED = "preferred"
    DISABLED = "disabled"


class NetworkPolicy(StrEnum):
    DENY_WHEN_CONTAINED = "deny_when_contained"
    ALLOW = "allow"


class CaptureConfig(ContractModel):
    default_timeout_seconds: Annotated[float, Field(gt=0, le=86_400)] = 300
    max_artifact_bytes: Annotated[int, Field(gt=0)] = 4_294_967_296
    max_parallel_captures: Annotated[int, Field(gt=0, le=128)] = 2


class PrivacyConfig(ContractModel):
    record_environment_allowlist: tuple[str, ...] = ("CUDA_VISIBLE_DEVICES",)
    capture_git_diff: bool = False
    allow_core_content: bool = False


class ExecutionConfig(ContractModel):
    allow_privileged_collectors: bool = False
    allowed_working_roots: tuple[str, ...] = ("..",)
    child_environment_allowlist: tuple[str, ...] = ("PATH", "CUDA_VISIBLE_DEVICES")
    containment: ContainmentPolicy = ContainmentPolicy.REQUIRED_FOR_MCP
    network: NetworkPolicy = NetworkPolicy.DENY_WHEN_CONTAINED
    max_cpu_percent: Annotated[int, Field(gt=0, le=10_000)] = 100
    max_processes: Annotated[int, Field(gt=0)] = 256
    max_memory_bytes: Annotated[int, Field(gt=0)] = 17_179_869_184
    max_output_bytes: Annotated[int, Field(gt=0)] = 16_777_216
    resource_sampling_interval_ms: Annotated[int, Field(ge=25, le=10_000)] = 250
    max_resource_observed_files: Annotated[int, Field(gt=0, le=1_000_000)] = 10_000


class AnalysisConfig(ContractModel):
    default_row_limit: Annotated[int, Field(gt=0)] = 100
    max_row_limit: Annotated[int, Field(gt=0)] = 1_000
    trace_processor_path: str | None = None


class StorageConfig(ContractModel):
    max_workspace_bytes: Annotated[int, Field(gt=0)] = 107_374_182_400
    min_free_bytes: Annotated[int, Field(ge=0)] = 2_147_483_648
    max_staging_bytes: Annotated[int, Field(gt=0)] = 17_179_869_184
    max_files_per_import: Annotated[int, Field(gt=0)] = 1
    max_rows_per_generation: Annotated[int, Field(gt=0)] = 100_000_000


class WorkspaceConfig(ContractModel):
    schema_version: Literal[1] = 1
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    def to_toml(self) -> str:
        return tomli_w.dumps(self.model_dump(mode="python", exclude_none=True))

    @classmethod
    def from_path(cls, path: Path) -> WorkspaceConfig:
        with path.open("rb") as stream:
            return cls.model_validate(tomllib.load(stream))
