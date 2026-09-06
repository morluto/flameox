"""Parsed immutable evidence contracts; filesystem verification belongs to the repository."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from flameox.runtime_contracts import LOWERCASE_SHA256_PATTERN, CaptureTarget, ExperimentDesign
from flameox.source_files import bundle_digest

type Digest = Annotated[str, Field(pattern=LOWERCASE_SHA256_PATTERN)]
type Nonempty = Annotated[str, Field(min_length=1)]
type Count = Annotated[int, Field(ge=0)]
type Argument = Annotated[str, Field(min_length=1, pattern=r"^[^\x00]+$")]
type Argv = Annotated[list[Argument], Field(min_length=1)]


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class VersionHeader(EvidenceModel):
    model_config = ConfigDict(extra="ignore")
    format_version: str


class Episode(EvidenceModel):
    created_at: str

    @field_validator("created_at")
    @classmethod
    def aware_timestamp(cls, value: str) -> str:
        if datetime.fromisoformat(value).tzinfo is None:
            raise ValueError("Evidence timestamps require a timezone")
        return value


class RepositoryMetadata(Episode):
    format_version: str


class ArtifactMetadata(EvidenceModel):
    format_version: str
    sha256: Digest
    size_bytes: Count


class ArtifactReference(ArtifactMetadata):
    role: Nonempty
    format: Nonempty
    producer: str | None = None


class InputIdentity(EvidenceModel):
    sha256: Digest
    size_bytes: Count
    format: Nonempty
    role: Nonempty


class ProviderIdentity(EvidenceModel):
    id: Nonempty
    version: Nonempty


class Coverage(EvidenceModel):
    rows_returned: Count
    rows_observed: Count
    complete: bool


class DataReference(EvidenceModel):
    path: Nonempty
    sha256: Digest
    size_bytes: Count
    media_type: Nonempty

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Evidence paths must be relative")
        return value


class ExecutionLimit(EvidenceModel):
    kind: str
    configured: int | float
    observed: int | float | None
    unit: str
    observation_available: bool
    recovery: str


class OracleOutcome(EvidenceModel):
    argv: Argv
    returncode: int | None
    status: Literal["passed", "failed"]
    failure_code: str | None


class CaptureExecution(EvidenceModel):
    case: str
    block: Annotated[int, Field(ge=1)]
    argv: Argv
    capture_argv: Argv
    cwd: str
    returncode: int | None
    status: Literal["pending", "succeeded", "failed"]
    failure_code: str | None
    missing_artifact_roles: list[Nonempty]
    semantic_oracle: OracleOutcome | None
    wall_time_ns: int | None
    containment: str
    limit: ExecutionLimit | None
    returncode_scope: Literal["workload", "collector"]
    workload_returncode: int | None
    executable_sha256: Digest

    @field_validator("cwd")
    @classmethod
    def absolute_cwd(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("Capture cwd must be absolute")
        return value

    @model_validator(mode="after")
    def attribution(self) -> CaptureExecution:
        if (self.returncode_scope == "collector" and self.workload_returncode is not None) or (
            self.returncode_scope == "workload" and self.workload_returncode != self.returncode
        ):
            raise ValueError("Invalid capture exit attribution")
        return self


class CaptureRequest(EvidenceModel):
    target: CaptureTarget
    mode: Literal["single", "experiment"]
    experiment: ExperimentDesign | None
    executions: list[CaptureExecution]


class AnalysisInput(EvidenceModel):
    sha256: Digest
    format: Nonempty
    path: Nonempty
    role: Nonempty
    producer: str | None


class AnalysisFailure(EvidenceModel):
    code: Nonempty
    message: str
    details: dict[str, JsonValue]


class AnalysisRequest(EvidenceModel):
    capability_id: Nonempty
    inputs: list[AnalysisInput] = Field(default_factory=list)
    offset: Count | None = None
    failure: AnalysisFailure | None = None
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    limits: dict[str, JsonValue] = Field(default_factory=dict)


class LogicalSource(InputIdentity):
    producer: str | None
    is_directory: bool
    artifact_indices: list[Count]
    relative_paths: list[Nonempty]

    def validate_members(self, members: list[ArtifactReference]) -> None:
        if self.is_directory:
            paths = self.relative_paths
            if len(paths) != len(members) or len({Path(path) for path in paths}) != len(paths):
                raise ValueError("Invalid bundle member paths")
            for relative in paths:
                DataReference.relative_path(relative)
            expected_digest = bundle_digest(
                (relative, member.sha256) for relative, member in zip(paths, members, strict=True)
            )
        else:
            if len(members) != 1 or members[0].role != self.role or self.relative_paths:
                raise ValueError("Invalid file membership")
            expected_digest = members[0].sha256
        if (self.sha256, self.size_bytes) != (
            expected_digest,
            sum(member.size_bytes for member in members),
        ):
            raise ValueError("Invalid source identity")


class SourceLayout(EvidenceModel):
    sources: list[LogicalSource]
    analysis_sources: list[Count]


class ManifestBody(EvidenceModel):
    evidence_kind: Nonempty
    capability_id: Nonempty
    provider: ProviderIdentity
    inputs: Annotated[list[InputIdentity], Field(min_length=1)]
    capture_request: CaptureRequest | None
    analysis_request: AnalysisRequest
    episode: Episode
    coverage: Coverage
    limitations: list[str]
    artifacts: list[ArtifactReference]
    data_files: Annotated[list[DataReference], Field(min_length=1)]
    source_layout: SourceLayout

    @model_validator(mode="after")
    def membership(self) -> ManifestBody:
        if len({item.role for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("Evidence artifact roles must be unique")
        layout = self.source_layout
        if len({item.role for item in layout.sources}) != len(layout.sources):
            raise ValueError("Evidence source roles must be unique")
        used: set[int] = set()
        for source in layout.sources:
            members = []
            for index in source.artifact_indices:
                if index >= len(self.artifacts) or index in used:
                    raise ValueError("Invalid artifact membership")
                used.add(index)
                member = self.artifacts[index]
                if (member.format, member.producer) != (source.format, source.producer):
                    raise ValueError("Inconsistent source metadata")
                members.append(member)
            source.validate_members(members)
        if used != set(range(len(self.artifacts))):
            raise ValueError("Incomplete artifact membership")
        inputs = self.analysis_request.inputs
        if len(layout.analysis_sources) != len(inputs):
            raise ValueError("Invalid analysis source mapping")
        for index, item in zip(layout.analysis_sources, inputs, strict=False):
            if index >= len(layout.sources):
                raise ValueError("Invalid analysis source index")
            source = layout.sources[index]
            if (source.sha256, source.format, source.producer) != (
                item.sha256,
                item.format,
                item.producer,
            ):
                raise ValueError("Invalid analysis source identity")
        return self


class EvidenceManifest(EvidenceModel):
    format_version: str
    evidence_id: Digest
    body: ManifestBody
