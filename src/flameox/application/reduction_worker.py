from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from flameox.application.native_reducer import NativePartitioning, NativeReductionLimits
from flameox.domain.models import CommandSpec
from flameox.models import ContractModel


class NativeReductionWorkerRequest(ContractModel):
    """Parsed, self-consistent input for the isolated native reducer process."""

    artifact_path: Path
    partitioning: NativePartitioning
    predicate_command: CommandSpec
    limits: NativeReductionLimits
    predicate_timeout_seconds: Annotated[float, Field(gt=0, le=3_600)]
    project_root: Path
    workspace_root: Path
    staging_root: Path
    max_output_bytes: Annotated[int, Field(gt=0)]
    minimum_free_bytes: Annotated[int, Field(ge=0)]
    maximum_rss_bytes: Annotated[int, Field(gt=0)]
    sampling_interval_ms: Annotated[int, Field(ge=25, le=10_000)]
    max_observed_files: Annotated[int, Field(ge=1, le=1_000_000)]
