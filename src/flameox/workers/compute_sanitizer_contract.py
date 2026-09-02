from __future__ import annotations

from typing import Annotated

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId


class ComputeSanitizerWorkerRequest(ContractModel):
    artifact_path: str = Field(min_length=1, max_length=4_096)
    max_records: Annotated[int, Field(gt=0, le=100_000_000)]
    max_frames: Annotated[int, Field(gt=0, le=4_096)]


class ComputeSanitizerWorkerResult(ContractModel):
    records: tuple[dict[str, JsonValue], ...]
    classifications: dict[str, Annotated[int, Field(ge=0)]] = Field(max_length=128)
    limitations: tuple[str, ...] = Field(default=(), max_length=1_024)
    truncated: bool

    @model_validator(mode="after")
    def classifications_count_records(self) -> ComputeSanitizerWorkerResult:
        if sum(self.classifications.values()) != len(self.records):
            raise ValueError("classification counts must partition records")
        return self


COMPUTE_SANITIZER_WORKER = WorkerDefinition(
    operation=WorkerOperationId.COMPUTE_SANITIZER_PARSE,
    module="flameox.workers.compute_sanitizer",
    request=TypeAdapter(ComputeSanitizerWorkerRequest),
    response=TypeAdapter(ComputeSanitizerWorkerResult),
    name="Compute Sanitizer",
    implementation="flameox.workers.compute_sanitizer/v1",
)
