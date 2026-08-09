from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal, cast

from pydantic import Field, model_validator

from flameox.application.pipelines import (
    PipelineStageDeclaration,
    RegisteredPipelineStageDeclaration,
    RegisterPipelineRequest,
    UnregisteredPipelineStageDeclaration,
)
from flameox.models import ContractModel

KERNEL_BUILD_SCHEMA_VERSION: Final[Literal["flameox.kernel-build.v1"]] = "flameox.kernel-build.v1"


class KernelBuildArtifact(ContractModel):
    path: Annotated[str, Field(min_length=1, max_length=500)]
    byte_length: Annotated[int, Field(ge=0)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    media_type: Annotated[str, Field(min_length=1, max_length=100)] = "application/octet-stream"
    role: Annotated[str, Field(min_length=1, max_length=100)] = "compiler_stage"

    @model_validator(mode="after")
    def relative_contained_path(self) -> KernelBuildArtifact:
        path = PurePosixPath(self.path)
        if path.is_absolute() or self.path in {".", ".."} or ".." in path.parts:
            raise ValueError("kernel-build artifact paths must be contained relative POSIX paths")
        if any(part in {"", "."} for part in path.parts):
            raise ValueError("kernel-build artifact paths must be normalized")
        return self


class KernelBuildStage(ContractModel):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    ordinal: Annotated[int, Field(ge=0, le=99)]
    predecessor: Annotated[str, Field(min_length=1, max_length=100)] | None = None
    status: Literal["available", "cached", "skipped", "unavailable", "failed"]
    format: Annotated[str, Field(min_length=1, max_length=100)]
    format_schema: Annotated[str, Field(min_length=1, max_length=100)]
    artifact: KernelBuildArtifact | None = None
    elapsed_ns: Annotated[int, Field(ge=0)] | None = None
    limitations: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=500)], ...],
        Field(max_length=20),
    ] = ()

    @model_validator(mode="after")
    def artifact_matches_status(self) -> KernelBuildStage:
        registered = self.status in {"available", "cached"}
        if registered != (self.artifact is not None):
            raise ValueError("available and cached stages require exactly one native artifact")
        return self


class KernelBuildManifestV1(ContractModel):
    schema_version: Literal["flameox.kernel-build.v1"] = KERNEL_BUILD_SCHEMA_VERSION
    producer: Literal["triton", "cute"]
    producer_version: Annotated[str, Field(min_length=1, max_length=100)]
    workload_identity: Annotated[str, Field(min_length=1, max_length=200)]
    device_identity: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    outcome: Literal["succeeded", "failed", "inconclusive"]
    cache_status: Literal["hit", "miss", "mixed", "unknown"] = "unknown"
    stages: Annotated[tuple[KernelBuildStage, ...], Field(min_length=1, max_length=100)]
    source_environment: dict[
        Annotated[str, Field(min_length=1, max_length=100)],
        Annotated[str, Field(max_length=500)],
    ] = Field(default_factory=dict)
    limitations: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=500)], ...],
        Field(max_length=20),
    ] = ()
    diagnostics: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=2_000)], ...],
        Field(max_length=50),
    ] = ()

    @model_validator(mode="after")
    def ordered_pipeline(self) -> KernelBuildManifestV1:
        if len(self.source_environment) > 20:
            raise ValueError("source_environment is limited to 20 entries")
        names = [stage.name for stage in self.stages]
        ordinals = [stage.ordinal for stage in self.stages]
        paths = [stage.artifact.path for stage in self.stages if stage.artifact is not None]
        if len(names) != len(set(names)) or len(ordinals) != len(set(ordinals)):
            raise ValueError("kernel-build stage names and ordinals must be unique")
        if ordinals != sorted(ordinals):
            raise ValueError("kernel-build stages must be declared in ordinal order")
        if len(paths) != len(set(paths)):
            raise ValueError("kernel-build artifact paths must be unique")
        seen: set[str] = set()
        for stage in self.stages:
            if stage.predecessor is not None and stage.predecessor not in seen:
                raise ValueError("stage predecessors must identify an earlier declared stage")
            seen.add(stage.name)
        if self.outcome == "succeeded" and any(
            stage.status in {"failed", "unavailable"} for stage in self.stages
        ):
            raise ValueError("a successful build cannot contain failed or unavailable stages")
        if self.outcome == "failed" and not any(stage.status == "failed" for stage in self.stages):
            raise ValueError("a failed build requires a failed stage")
        return self

    def bundle_paths(self, manifest_path: Path) -> tuple[Path, ...]:
        return tuple(
            manifest_path.parent / stage.artifact.path
            for stage in self.stages
            if stage.artifact is not None
        )

    def pipeline_request(
        self,
        *,
        run_id: str,
        registration_ids_by_path: dict[str, str],
    ) -> RegisterPipelineRequest:
        declarations: list[PipelineStageDeclaration] = []
        for stage in self.stages:
            if stage.artifact is not None:
                registration_id = registration_ids_by_path.get(stage.artifact.path)
                if registration_id is None:
                    raise ValueError(f"missing registration for {stage.artifact.path!r}")
                declarations.append(
                    RegisteredPipelineStageDeclaration(
                        name=stage.name,
                        ordinal=stage.ordinal,
                        predecessor=stage.predecessor,
                        format=stage.format,
                        format_schema=stage.format_schema,
                        elapsed_ns=stage.elapsed_ns,
                        limitations=stage.limitations,
                        status=cast(Literal["available", "cached"], stage.status),
                        registration_id=registration_id,
                    )
                )
            else:
                declarations.append(
                    UnregisteredPipelineStageDeclaration(
                        name=stage.name,
                        ordinal=stage.ordinal,
                        predecessor=stage.predecessor,
                        format=stage.format,
                        format_schema=stage.format_schema,
                        elapsed_ns=stage.elapsed_ns,
                        limitations=stage.limitations,
                        status=cast(Literal["skipped", "unavailable", "failed"], stage.status),
                    )
                )
        limitations = list(self.limitations)
        if self.diagnostics:
            limitations.append("compiler diagnostics are preserved in the kernel-build manifest")
        if self.device_identity is None:
            limitations.append("device identity was not supplied by the producer")
        return RegisterPipelineRequest(
            run_id=run_id,
            pipeline_name=f"{self.producer}.compiler",
            pipeline_schema=KERNEL_BUILD_SCHEMA_VERSION,
            producer=self.producer,
            producer_version=self.producer_version,
            stages=tuple(declarations),
            limitations=tuple(dict.fromkeys(limitations)),
        )


def kernel_build_json_schema() -> dict[str, object]:
    return KernelBuildManifestV1.model_json_schema()
