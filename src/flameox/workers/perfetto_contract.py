from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId


class PerfettoExtractRequest(ContractModel):
    operation: Literal["extract"]
    artifact_path: str = Field(min_length=1, max_length=4_096)
    binary_path: str = Field(min_length=1, max_length=4_096)
    max_rows: Annotated[int, Field(gt=0, le=100_000_000)]


class PerfettoWindowRequest(ContractModel):
    operation: Literal["window"]
    artifact_path: str = Field(min_length=1, max_length=4_096)
    binary_path: str = Field(min_length=1, max_length=4_096)
    start_ns: int
    end_ns: int
    limit: Annotated[int, Field(gt=0, le=10_000)]
    after_ts: int | None = None
    after_id: int | None = None


type PerfettoWorkerRequest = Annotated[
    PerfettoExtractRequest | PerfettoWindowRequest,
    Field(discriminator="operation"),
]


class PerfettoSliceRow(ContractModel):
    id: int
    parent_id: int | None
    name: str
    ts: int
    dur: int
    track_id: int
    category: str | None
    thread_name: str | None
    process_name: str | None
    filename: str | None
    line: int | None
    input_shapes: str | None
    allocation_bytes: int | None
    phase: str | None
    correlation_id: str | None
    device: str | None
    stream: str | None


class PerfettoWindowRow(ContractModel):
    id: int
    parent_id: int | None
    name: str
    category: str | None
    ts: int
    dur: int
    track_id: int


class PerfettoExtractResult(ContractModel):
    operation: Literal["extract"] = "extract"
    truncated: bool
    rows: tuple[PerfettoSliceRow, ...]


class PerfettoWindowResult(ContractModel):
    operation: Literal["window"] = "window"
    total: Annotated[int, Field(ge=0)]
    rows: tuple[PerfettoWindowRow, ...]


type PerfettoWorkerResult = Annotated[
    PerfettoExtractResult | PerfettoWindowResult,
    Field(discriminator="operation"),
]


PERFETTO_WORKER: WorkerDefinition[PerfettoWorkerRequest, PerfettoWorkerResult] = WorkerDefinition(
    operation=WorkerOperationId.PERFETTO_QUERY,
    module="flameox.workers.perfetto",
    request=TypeAdapter(PerfettoWorkerRequest),
    response=TypeAdapter(PerfettoWorkerResult),
    name="Perfetto",
    implementation="flameox.workers.perfetto/v1",
)
