from __future__ import annotations

from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal

from pydantic import Field, model_validator

from flameox.models import ContractModel

KERNEL_BUILD_SCHEMA_VERSION: Final[Literal["flameox.kernel-build.v1"]] = "flameox.kernel-build.v1"


class KernelBuildProducer(StrEnum):
    TRITON = "triton"
    CUTE = "cute"


class KernelBuildOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class KernelBuildCacheStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class KernelBuildStageStatus(StrEnum):
    AVAILABLE = "available"
    CACHED = "cached"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


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


class _KernelBuildStage(ContractModel):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    ordinal: Annotated[int, Field(ge=0, le=99)]
    predecessor: Annotated[str, Field(min_length=1, max_length=100)] | None = None
    format: Annotated[str, Field(min_length=1, max_length=100)]
    format_schema: Annotated[str, Field(min_length=1, max_length=100)]
    elapsed_ns: Annotated[int, Field(ge=0)] | None = None
    limitations: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=500)], ...],
        Field(max_length=20),
    ] = ()


class ArtifactKernelBuildStage(_KernelBuildStage):
    status: Literal[KernelBuildStageStatus.AVAILABLE, KernelBuildStageStatus.CACHED]
    artifact: KernelBuildArtifact


class ArtifactlessKernelBuildStage(_KernelBuildStage):
    status: Literal[
        KernelBuildStageStatus.SKIPPED,
        KernelBuildStageStatus.UNAVAILABLE,
        KernelBuildStageStatus.FAILED,
    ]
    artifact: Literal[None] = None


type KernelBuildStage = Annotated[
    ArtifactKernelBuildStage | ArtifactlessKernelBuildStage,
    Field(discriminator="status"),
]


class KernelBuildManifestV1(ContractModel):
    schema_version: Literal["flameox.kernel-build.v1"] = KERNEL_BUILD_SCHEMA_VERSION
    producer: Literal[KernelBuildProducer.TRITON, KernelBuildProducer.CUTE]
    producer_version: Annotated[str, Field(min_length=1, max_length=100)]
    workload_identity: Annotated[str, Field(min_length=1, max_length=200)]
    device_identity: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    outcome: KernelBuildOutcome
    cache_status: KernelBuildCacheStatus = KernelBuildCacheStatus.UNKNOWN
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
        if self.outcome is KernelBuildOutcome.SUCCEEDED and any(
            stage.status in {KernelBuildStageStatus.FAILED, KernelBuildStageStatus.UNAVAILABLE}
            for stage in self.stages
        ):
            raise ValueError("a successful build cannot contain failed or unavailable stages")
        if self.outcome is KernelBuildOutcome.FAILED and not any(
            stage.status is KernelBuildStageStatus.FAILED for stage in self.stages
        ):
            raise ValueError("a failed build requires a failed stage")
        return self

    def bundle_paths(self, manifest_path: Path) -> tuple[Path, ...]:
        return tuple(
            manifest_path.parent / stage.artifact.path
            for stage in self.stages
            if stage.artifact is not None
        )


def kernel_build_json_schema() -> dict[str, object]:
    return KernelBuildManifestV1.model_json_schema()
