from __future__ import annotations

from typing import Annotated, Final, Literal, Protocol

from pydantic import Field, JsonValue, StringConstraints, model_validator

from flameox.domain.models import ArtifactKind, CommandSpec, Sensitivity
from flameox.models import ContractModel

ADAPTER_API_VERSION: Final[Literal[1]] = 1


class AdapterProbeContext(ContractModel):
    api_version: Literal[1] = ADAPTER_API_VERSION
    project_root: str


class AdapterProbeResult(ContractModel):
    api_version: Literal[1] = ADAPTER_API_VERSION
    status: Literal["available", "unavailable", "degraded"]
    adapter_version: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    limitations: Annotated[tuple[str, ...], Field(max_length=32)] = ()
    remediation: Annotated[tuple[str, ...], Field(max_length=32)] = ()


class AdapterPlanRequest(ContractModel):
    api_version: Literal[1] = ADAPTER_API_VERSION
    project_root: str
    output_root: str
    workload: CommandSpec


class AdapterArtifactDeclaration(ContractModel):
    relative_path: Annotated[
        str,
        StringConstraints(min_length=1, max_length=500, pattern=r"^[^/\\\x00][^\\\x00]*$"),
    ]
    kind: ArtifactKind
    role: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    media_type: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    sensitivity: Sensitivity = Sensitivity.INTERNAL

    @model_validator(mode="after")
    def path_stays_in_output_root(self) -> AdapterArtifactDeclaration:
        if self.relative_path.startswith("/") or ".." in self.relative_path.split("/"):
            raise ValueError("adapter artifact paths must stay below the output root")
        return self


class AdapterExecutionPlan(ContractModel):
    api_version: Literal[1] = ADAPTER_API_VERSION
    adapter: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    argv_prefix: Annotated[tuple[str, ...], Field(min_length=1, max_length=256)]
    artifacts: Annotated[
        tuple[AdapterArtifactDeclaration, ...],
        Field(min_length=1, max_length=16),
    ]
    permissions: Annotated[tuple[str, ...], Field(max_length=32)] = ()
    expected_overhead: Annotated[str, StringConstraints(min_length=1, max_length=1_000)]
    limitations: Annotated[tuple[str, ...], Field(max_length=32)] = ()
    extractor_version: Annotated[str, StringConstraints(min_length=1, max_length=100)]

    @model_validator(mode="after")
    def artifact_paths_are_unique(self) -> AdapterExecutionPlan:
        paths = [artifact.relative_path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("adapter artifact paths must be unique")
        return self


class AdapterValidationResult(ContractModel):
    api_version: Literal[1] = ADAPTER_API_VERSION
    valid: bool
    limitations: Annotated[tuple[str, ...], Field(max_length=32)] = ()


class AdapterExtractionResult(ContractModel):
    api_version: Literal[1] = ADAPTER_API_VERSION
    extractor_version: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    summary: dict[
        Annotated[str, StringConstraints(min_length=1, max_length=100)],
        JsonValue,
    ] = Field(default_factory=dict, max_length=128)
    limitations: Annotated[tuple[str, ...], Field(max_length=32)] = ()


class AdapterV1(Protocol):
    name: str
    api_version: Literal[1]

    async def probe(self, context: AdapterProbeContext) -> AdapterProbeResult: ...

    async def plan(self, request: AdapterPlanRequest) -> AdapterExecutionPlan: ...

    async def validate(
        self,
        _artifact_path: str,
        declaration: AdapterArtifactDeclaration,
    ) -> AdapterValidationResult: ...

    async def extract(
        self,
        _artifact_path: str,
        declaration: AdapterArtifactDeclaration,
    ) -> AdapterExtractionResult: ...
