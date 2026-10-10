from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId, WorkerOutputFile


class MemrayExtractionLimits(ContractModel):
    max_input_bytes: Annotated[int, Field(gt=0, le=1 << 40)]
    max_provider_records: Annotated[int, Field(gt=0, le=100_000_000)]
    max_frames: Annotated[int, Field(gt=0, le=10_000_000)]
    max_stack_depth: Annotated[int, Field(gt=0, le=4_096)]
    max_aggregate_rows: Annotated[int, Field(gt=0, le=20_000_000)]
    max_output_bytes: Annotated[int, Field(gt=0, le=1 << 40)]


class MemrayMetricCoverage(ContractModel):
    status: Literal["available"] = "available"
    records_seen: int = Field(ge=0)
    records_selected: int = Field(ge=0)
    dropped_stack_frames: int = Field(ge=0)

    @model_validator(mode="after")
    def normalized_work_is_observed(self) -> MemrayMetricCoverage:
        if self.records_selected > self.records_seen:
            raise ValueError("selected Memray coverage exceeds observed work")
        return self

    @property
    def complete(self) -> bool:
        return self.records_seen == self.records_selected and self.dropped_stack_frames == 0


class MemrayExtractionCoverage(ContractModel):
    metric: MemrayMetricCoverage
    frame_contributions_dropped: int = Field(ge=0)
    aggregate_rows_dropped: int = Field(ge=0)


class MemrayWorkerRequest(ContractModel):
    artifact_path: str = Field(min_length=1, max_length=4_096)
    metric: Literal["memory.high_watermark", "memory.retained_end"]
    limits: MemrayExtractionLimits


class MemrayWorkerResult(ContractModel):
    reader_version: str = Field(min_length=1, max_length=100)
    peak_memory_bytes: int = Field(ge=0)
    retained_end_bytes: int = Field(ge=0)
    allocation_operations: int | None = Field(default=None, ge=0)
    total_allocated_bytes: int | None = Field(default=None, ge=0)
    capture_records: int = Field(ge=0)
    has_native_traces: bool
    coverage: MemrayExtractionCoverage
    files: tuple[WorkerOutputFile, ...] = Field(min_length=2, max_length=2)


MEMRAY_WORKER = WorkerDefinition(
    operation=WorkerOperationId.MEMRAY_PARSE,
    module="flameox.workers.memray",
    request=TypeAdapter(MemrayWorkerRequest),
    response=TypeAdapter(MemrayWorkerResult),
    name="Memray",
    implementation="flameox.workers.memray",
    timeout_seconds=300,
)
