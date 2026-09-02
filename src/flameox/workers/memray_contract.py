from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId, WorkerOutputFile

MEMRAY_EXTRACTOR_NAME = "memray"
MEMRAY_EXTRACTOR_VERSION = "7"


class MemrayExtractionLimits(ContractModel):
    max_input_bytes: Annotated[int, Field(gt=0, le=1 << 40)]
    max_provider_records: Annotated[int, Field(gt=0, le=100_000_000)]
    max_frames: Annotated[int, Field(gt=0, le=10_000_000)]
    max_stack_depth: Annotated[int, Field(gt=0, le=4_096)]
    max_aggregate_rows: Annotated[int, Field(gt=0, le=20_000_000)]
    max_unique_edges: Annotated[int, Field(gt=0, le=20_000_000)]
    max_representative_stacks: Annotated[int, Field(gt=0, le=10_000_000)]
    max_output_bytes: Annotated[int, Field(gt=0, le=1 << 40)]
    wall_time_seconds: Annotated[float, Field(gt=0, le=86_400)]
    max_worker_memory_bytes: Annotated[int, Field(gt=0, le=1 << 50)]
    temporary_allocation_threshold: Annotated[int, Field(ge=0, le=1_000)] = 1


class MemrayMetricCoverage(ContractModel):
    status: Literal["available"] = "available"
    records_seen: int = Field(ge=0)
    records_selected: int = Field(ge=0)
    record_bytes_seen: int = Field(ge=0)
    record_bytes_selected: int = Field(ge=0)
    dropped_stack_frames: int = Field(ge=0)
    dropped_stack_frame_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def normalized_work_is_observed(self) -> MemrayMetricCoverage:
        if (
            self.records_selected > self.records_seen
            or self.record_bytes_selected > self.record_bytes_seen
        ):
            raise ValueError("selected Memray coverage exceeds observed work")
        return self

    @property
    def complete(self) -> bool:
        return self.records_seen == self.records_selected and self.dropped_stack_frames == 0


class MemrayMetricUnavailable(ContractModel):
    status: Literal["unavailable"] = "unavailable"
    reason: Literal["unsupported_capture_format"] = "unsupported_capture_format"

    @property
    def complete(self) -> bool:
        return True


type MemrayMetricCoverageState = Annotated[
    MemrayMetricCoverage | MemrayMetricUnavailable,
    Field(discriminator="status"),
]


class MemrayExtractionCoverage(ContractModel):
    high_watermark: MemrayMetricCoverage
    retained_end: MemrayMetricCoverage
    allocation_volume: MemrayMetricCoverageState
    temporary: MemrayMetricCoverageState
    frames_published: int = Field(ge=0)
    aggregate_rows_published: int = Field(ge=0)
    frame_contributions_dropped: int = Field(ge=0)
    frame_contribution_bytes_dropped: int = Field(ge=0)
    aggregate_rows_dropped: int = Field(ge=0)
    aggregate_inclusive_bytes_dropped: int = Field(ge=0)
    edge_rows_published: int = Field(ge=0)
    edge_rows_dropped: int = Field(ge=0)
    edge_weight_bytes_dropped: int = Field(ge=0)
    representative_stacks_published: int = Field(ge=0)
    representative_stacks_dropped: int = Field(ge=0)
    representative_stack_weight_bytes_dropped: int = Field(ge=0)
    output_bytes: int = Field(ge=0)

    @property
    def complete(self) -> bool:
        return (
            self.high_watermark.complete
            and self.retained_end.complete
            and self.allocation_volume.complete
            and self.temporary.complete
            and self.frame_contributions_dropped == 0
            and self.aggregate_rows_dropped == 0
            and self.edge_rows_dropped == 0
            and self.representative_stacks_dropped == 0
        )


class MemrayWorkerProgress(ContractModel):
    phase: Literal[
        "computing_statistics",
        "normalizing_high_watermark",
        "normalizing_retained_end",
        "normalizing_allocation_volume",
        "normalizing_temporary",
        "writing_evidence",
    ]
    records_seen: int = Field(ge=0)
    records_selected: int = Field(ge=0)
    record_bytes_seen: int = Field(ge=0)


class MemrayWorkerRequest(ContractModel):
    artifact_path: str = Field(min_length=1, max_length=4_096)
    run_id: str = Field(min_length=1, max_length=200)
    artifact_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    limits: MemrayExtractionLimits


class MemrayWorkerResult(ContractModel):
    reader_version: str = Field(min_length=1, max_length=100)
    peak_memory_bytes: int = Field(ge=0)
    retained_end_bytes: int = Field(ge=0)
    temporary_allocated_bytes: int | None = Field(default=None, ge=0)
    allocation_operations: int | None = Field(default=None, ge=0)
    total_allocated_bytes: int | None = Field(default=None, ge=0)
    capture_records: int = Field(ge=0)
    has_native_traces: bool
    coverage: MemrayExtractionCoverage
    files: tuple[WorkerOutputFile, ...] = Field(min_length=5, max_length=5)


MEMRAY_WORKER = WorkerDefinition(
    operation=WorkerOperationId.MEMRAY_PARSE,
    module="flameox.workers.memray",
    request=TypeAdapter(MemrayWorkerRequest),
    response=TypeAdapter(MemrayWorkerResult),
    name="Memray",
    implementation=f"flameox.workers.memray/v{MEMRAY_EXTRACTOR_VERSION}",
    timeout_seconds=300,
)
