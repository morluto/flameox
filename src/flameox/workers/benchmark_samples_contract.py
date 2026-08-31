from __future__ import annotations

from typing import Annotated

from pydantic import Field, JsonValue, TypeAdapter

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId


class BenchmarkSamplesWorkerRequest(ContractModel):
    artifact_path: str = Field(min_length=1, max_length=4_096)
    max_rows: Annotated[int, Field(gt=0, le=1_001)]


class BenchmarkSamplesWorkerResult(ContractModel):
    producer: str = Field(min_length=1, max_length=200)
    producer_version: str | None = Field(default=None, max_length=200)
    benchmark_names: tuple[str, ...]
    measurement_count: Annotated[int, Field(ge=0)]
    warmup_count: Annotated[int, Field(ge=0)]
    rows: tuple[dict[str, JsonValue], ...]
    truncated: bool
    limitations: tuple[str, ...] = Field(default=(), max_length=1_024)


BENCHMARK_SAMPLES_WORKER = WorkerDefinition(
    operation=WorkerOperationId.BENCHMARK_SAMPLES_PARSE,
    module="flameox.workers.benchmark_samples",
    request=TypeAdapter(BenchmarkSamplesWorkerRequest),
    response=TypeAdapter(BenchmarkSamplesWorkerResult),
    name="benchmark samples",
    implementation="flameox.workers.benchmark_samples/v1",
)
