from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class CaptureConfig(ConfigModel):
    default_timeout_seconds: Annotated[float, Field(gt=0, le=86_400)] = 300
    max_artifact_bytes: Annotated[int, Field(gt=0)] = 4_294_967_296
    max_parallel_captures: Annotated[int, Field(gt=0, le=128)] = 2


class PrivacyConfig(ConfigModel):
    record_environment_allowlist: tuple[str, ...] = ("CUDA_VISIBLE_DEVICES",)
    capture_git_diff: bool = False
    allow_core_content: bool = False


class ExecutionConfig(ConfigModel):
    allow_privileged_collectors: bool = False
    allow_mcp_ad_hoc_commands: bool = False
    allowed_working_roots: tuple[str, ...] = ("..",)
    child_environment_allowlist: tuple[str, ...] = ("PATH", "CUDA_VISIBLE_DEVICES")
    containment: Literal["required_for_mcp", "preferred", "disabled"] = "required_for_mcp"
    network: Literal["deny_when_contained", "allow"] = "deny_when_contained"
    max_processes: Annotated[int, Field(gt=0)] = 256
    max_memory_bytes: Annotated[int, Field(gt=0)] = 17_179_869_184
    max_output_bytes: Annotated[int, Field(gt=0)] = 16_777_216


class AnalysisConfig(ConfigModel):
    default_row_limit: Annotated[int, Field(gt=0)] = 100
    max_row_limit: Annotated[int, Field(gt=0)] = 1_000
    trace_processor_path: str | None = None


class StorageConfig(ConfigModel):
    max_workspace_bytes: Annotated[int, Field(gt=0)] = 107_374_182_400
    min_free_bytes: Annotated[int, Field(ge=0)] = 2_147_483_648
    max_staging_bytes: Annotated[int, Field(gt=0)] = 17_179_869_184
    max_files_per_import: Annotated[int, Field(gt=0)] = 1
    max_rows_per_generation: Annotated[int, Field(gt=0)] = 100_000_000


class WorkspaceConfig(ConfigModel):
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
