"""Semantic identity helpers for compiler lineage.

Compiler and target qualification is derived from authoritative run semantics,
not copied into provider-native evidence.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from flameox.domain.identity import digest_model
from flameox.models import ContractModel


class CompilerTarget(ContractModel):
    """Explicit cross-compilation intent selected by the managed plan."""

    backend: Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")]
    architecture: Annotated[
        str,
        Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.+-]+$"),
    ]
    warp_size: Annotated[int, Field(gt=0, le=1_024)] | None = None


class CompilerIdentity(ContractModel):
    """The exact Triton distribution executed by one declared interpreter."""

    adapter: Literal["triton.compiler"]
    distribution: Annotated[
        str,
        Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    ]
    version: Annotated[str, Field(min_length=1, max_length=200)]
    content_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    interpreter_digest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class CompilerTargetIdentity(ContractModel):
    """One CUDA compilation target bound to the authoritative environment."""

    backend: Literal["cuda"]
    architecture: Annotated[str, Field(pattern=r"^sm_[0-9]{2,3}a?$")]
    warp_size: Literal[32]
    ptx_version: Annotated[str, Field(pattern=r"^[0-9]+\.[0-9]+$")] | None = None
    environment_id: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class CompilerQualification(ContractModel):
    """Run-scoped compiler and target authority for managed native outputs."""

    compiler: CompilerIdentity
    target: CompilerTargetIdentity | None = None


def compiler_identity_id(value: object) -> str | None:
    """Return the identity of one exact managed compiler distribution."""

    if value is None:
        return None
    qualification = CompilerQualification.model_validate(value)
    return digest_model(qualification.compiler.model_dump(mode="json"))


def compiler_target_identity_id(value: object) -> str | None:
    """Return the target identity only when it was observed and qualified."""

    if value is None:
        return None
    qualification = CompilerQualification.model_validate(value)
    if qualification.target is None:
        return None
    return digest_model(qualification.target.model_dump(mode="json"))
