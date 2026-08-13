from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId


class NsightComputeWorkerRequest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_path: str = Field(min_length=1, max_length=4_096)
    interface_path: str = Field(min_length=1, max_length=4_096)
    max_ranges: Annotated[int, Field(gt=0, le=1_000)]
    max_actions: Annotated[int, Field(gt=0, le=10_000)]
    max_metrics: Annotated[int, Field(gt=0, le=100_000_000)]
    max_observations: Annotated[int, Field(gt=0, le=100_000_000)]


class NsightComputeWorkerResult(ContractModel):
    schema_version: Literal[1] = 1
    report_version: str = Field(min_length=1, max_length=200)
    measurements: tuple[dict[str, JsonValue], ...]
    observations: tuple[dict[str, JsonValue], ...]
    metric_ids: tuple[str, ...]
    section_ids: tuple[str, ...]
    range_count: Annotated[int, Field(ge=0)]
    action_count: Annotated[int, Field(ge=0)]
    roofline_present: bool
    limitations: tuple[str, ...] = Field(default=(), max_length=1_024)


NSIGHT_COMPUTE_WORKER = WorkerDefinition(
    operation=WorkerOperationId.NSIGHT_COMPUTE_PARSE,
    module="flameox.workers.nsight_compute",
    request=TypeAdapter(NsightComputeWorkerRequest),
    response=TypeAdapter(NsightComputeWorkerResult),
    name="Nsight Compute",
    implementation="flameox.workers.nsight_compute/v1",
)
