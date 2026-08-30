from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, model_validator

from flameox.models import ContractModel


def _relative_posix_path(value: str, *, subject: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or value in {".", ".."}
        or ".." in path.parts
    ):
        raise ValueError(f"kernel-build {subject} paths must be contained relative POSIX paths")
    if value != path.as_posix():
        raise ValueError(f"kernel-build {subject} paths must be normalized")
    return path


class KernelBuildProducer(StrEnum):
    TRITON = "triton"
    CUTE = "cute"


class KernelBuildArtifact(ContractModel):
    """One provider-native file, identified without assigning it pipeline state."""

    path: Annotated[str, Field(min_length=1, max_length=500)]
    byte_length: Annotated[int, Field(ge=0)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    media_type: Annotated[str, Field(min_length=1, max_length=100)] = "application/octet-stream"

    @model_validator(mode="after")
    def relative_contained_path(self) -> KernelBuildArtifact:
        _relative_posix_path(self.path, subject="artifact")
        return self


class KernelBuildArtifactGroup(ContractModel):
    """One native compiler dump directory, without a cross-group ordering claim."""

    path: Annotated[str, Field(min_length=1, max_length=500)]
    artifacts: Annotated[tuple[KernelBuildArtifact, ...], Field(min_length=1, max_length=99)]

    @model_validator(mode="after")
    def contained_artifacts(self) -> KernelBuildArtifactGroup:
        _relative_posix_path(self.path, subject="group")
        prefix = f"{self.path}/"
        artifact_paths = [artifact.path for artifact in self.artifacts]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("kernel-build group artifact paths must be unique")
        if any(not artifact_path.startswith(prefix) for artifact_path in artifact_paths):
            raise ValueError(
                "kernel-build group artifacts must remain below their native directory"
            )
        return self


class KernelBuildManifest(ContractModel):
    """Provider-native compiler evidence grouped by its original dump directories.

    The manifest is deliberately not a pipeline. It preserves raw group membership,
    bytes, and provider provenance; ``ArtifactPipeline`` owns normalized stage
    ordering and predecessor lineage. The authoritative run owns execution
    semantics.
    """

    producer: Literal[KernelBuildProducer.TRITON, KernelBuildProducer.CUTE]
    native_groups: Annotated[tuple[KernelBuildArtifactGroup, ...], Field(max_length=99)] = ()
    attachments: Annotated[tuple[KernelBuildArtifact, ...], Field(max_length=20)] = ()

    @model_validator(mode="after")
    def grouped_provenance(self) -> KernelBuildManifest:
        group_paths = [group.path for group in self.native_groups]
        if len(group_paths) != len(set(group_paths)):
            raise ValueError("kernel-build native group paths must be unique")
        artifact_paths = [artifact.path for artifact in self.artifacts]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError(
                "kernel-build artifact paths must be unique across groups and attachments"
            )
        if len(artifact_paths) > 99:
            raise ValueError("kernel-build manifest has at most 99 native artifacts")
        return self

    @property
    def artifacts(self) -> tuple[KernelBuildArtifact, ...]:
        return (
            *(artifact for group in self.native_groups for artifact in group.artifacts),
            *self.attachments,
        )


def kernel_build_json_schema() -> dict[str, object]:
    return KernelBuildManifest.model_json_schema()
