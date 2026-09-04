from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId


class PyperfWorkerRequest(ContractModel):
    artifact_path: str = Field(min_length=1, max_length=4_096)
    max_rows: Annotated[int, Field(gt=0, le=100_000)]
    projection: Literal["samples", "series"] = "samples"
    metric: str | None = Field(default=None, min_length=1, max_length=200)


class PyperfWorkerResult(ContractModel):
    reader_version: str = Field(min_length=1, max_length=100)
    benchmark_names: tuple[str, ...]
    measurement_count: Annotated[int, Field(ge=0)]
    warmup_count: Annotated[int, Field(ge=0)]
    rows: tuple[dict[str, JsonValue], ...]
    truncated: bool
    limitations: tuple[str, ...] = Field(default=(), max_length=1_024)


PYPERF_WORKER = WorkerDefinition(
    operation=WorkerOperationId.PYPERF_PARSE,
    module="flameox.workers.pyperf",
    request=TypeAdapter(PyperfWorkerRequest),
    response=TypeAdapter(PyperfWorkerResult),
    name="pyperf",
    implementation="flameox.workers.pyperf/v1",
)
