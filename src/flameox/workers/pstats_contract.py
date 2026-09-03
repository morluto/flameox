from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter

from flameox.models import ContractModel
from flameox.runtime_contracts import PstatsMetric
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId


class PstatsWorkerRequest(ContractModel):
    artifact_path: str = Field(min_length=1, max_length=4_096)
    projection: Literal["hotspots", "call_graph"] = "hotspots"
    metric: PstatsMetric = "self_time_seconds"
    function: str | None = Field(default=None, min_length=1, max_length=200)
    direction: Literal["callers", "callees", "both"] = "both"
    max_rows: Annotated[int, Field(gt=0, le=1_001)]


class PstatsWorkerResult(ContractModel):
    reader_version: str = Field(min_length=1, max_length=100)
    metric: PstatsMetric
    function_count: Annotated[int, Field(ge=0)]
    edge_count: Annotated[int, Field(ge=0)] = 0
    rows: tuple[dict[str, JsonValue], ...]
    truncated: bool
    limitations: tuple[str, ...] = Field(default=(), max_length=16)


PSTATS_WORKER = WorkerDefinition(
    operation=WorkerOperationId.PSTATS_PARSE,
    module="flameox.workers.pstats",
    request=TypeAdapter(PstatsWorkerRequest),
    response=TypeAdapter(PstatsWorkerResult),
    name="pstats",
    implementation="flameox.workers.pstats/v1",
)
