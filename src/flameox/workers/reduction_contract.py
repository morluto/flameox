from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from flameox.application.reduction_contracts import (
    PredicateClassification,
    ReductionDisposition,
    ReductionFormat,
    ReductionMinimality,
)
from flameox.domain.executables import ResolvedExecutable
from flameox.domain.models import CommandSpec
from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId, WorkerOutputFile

SHRINKRAY_VERSION: Literal["26.7.8.0"] = "26.7.8.0"
SHRINKRAY_REQUIREMENT: Literal["shrinkray==26.7.8.0"] = "shrinkray==26.7.8.0"
SHRINKRAY_PROFILE: Literal["flameox.shrinkray.offline-v1"] = "flameox.shrinkray.offline-v1"
UNRESOLVED_EXIT_CODE: Literal[101] = 101


class ReductionPredicateConfig(ContractModel):
    schema_version: Literal[1] = 1
    operation_root: str = Field(min_length=1, max_length=4_096)
    receipt_root: str = Field(min_length=1, max_length=4_096)
    counter_path: str = Field(min_length=1, max_length=4_096)
    deadline_monotonic: float = Field(gt=0)
    predicate_command: CommandSpec
    predicate_executable_binding: ResolvedExecutable
    predicate_repetitions: Annotated[int, Field(ge=1, le=20)]
    predicate_timeout_seconds: Annotated[float, Field(gt=0, le=3_600)]
    max_attempts: Annotated[int, Field(ge=1, le=100_000)]
    max_candidate_bytes: Annotated[int, Field(gt=0)]
    max_output_bytes: Annotated[int, Field(gt=0)]
    project_root: str = Field(min_length=1, max_length=4_096)
    workspace_root: str = Field(min_length=1, max_length=4_096)
    staging_root: str = Field(min_length=1, max_length=4_096)
    minimum_free_bytes: Annotated[int, Field(ge=0)]
    maximum_rss_bytes: Annotated[int, Field(gt=0)]
    sampling_interval_ms: Annotated[int, Field(ge=25, le=10_000)]
    max_observed_files: Annotated[int, Field(ge=1, le=1_000_000)]


class ShrinkRayWorkerRequest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_path: str = Field(min_length=1, max_length=4_096)
    shrinkray_executable: str = Field(min_length=1, max_length=4_096)
    shrinkray_executable_binding: ResolvedExecutable
    predicate_bridge_executable: str = Field(min_length=1, max_length=4_096)
    predicate_bridge_binding: ResolvedExecutable
    predicate_config: ReductionPredicateConfig
    input_format: ReductionFormat
    seed: int = 0
    parallelism: Literal[1] = 1
    wall_time_seconds: Annotated[float, Field(gt=0, le=86_400)]
    max_staging_bytes: Annotated[int, Field(gt=0)]
    max_staging_files: Annotated[int, Field(ge=8, le=100_000)]


class ShrinkRayWorkerResult(ContractModel):
    schema_version: Literal[1] = 1
    disposition: ReductionDisposition
    tool_completed: bool
    final_classification: PredicateClassification
    final_candidate: WorkerOutputFile
    attempt_receipts: WorkerOutputFile
    attempted: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    unresolved: int = Field(ge=0)
    contradictory: int = Field(ge=0)
    timed_out: int = Field(ge=0)
    history: WorkerOutputFile | None = None
    stdout: WorkerOutputFile
    stderr: WorkerOutputFile
    original_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    original_size_bytes: int = Field(ge=0)
    final_size_bytes: int = Field(ge=0)
    minimality: Literal[ReductionMinimality.NOT_CLAIMED] = ReductionMinimality.NOT_CLAIMED
    budget_exhausted: bool
    shrinkray_version: Literal["26.7.8.0"] = SHRINKRAY_VERSION
    profile: Literal["flameox.shrinkray.offline-v1"] = SHRINKRAY_PROFILE
    limitations: tuple[str, ...] = ()


SHRINKRAY_WORKER = WorkerDefinition(
    operation=WorkerOperationId.REDUCTION_EXECUTE,
    module="flameox.workers.reduction",
    request=TypeAdapter(ShrinkRayWorkerRequest),
    response=TypeAdapter(ShrinkRayWorkerResult),
    name="ShrinkRay reduction",
    implementation="flameox.workers.reduction/shrinkray-26.7.8.0-offline-v1",
)
