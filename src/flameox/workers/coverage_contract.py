from __future__ import annotations

from typing import Annotated

from pydantic import Field, JsonValue, TypeAdapter

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId


class CoverageWorkerRequest(ContractModel):
    artifact_path: str = Field(min_length=1, max_length=4_096)
    project_root: str = Field(min_length=1, max_length=4_096)
    max_rows: Annotated[int, Field(gt=0, le=1_001)]


class CoverageWorkerResult(ContractModel):
    reader_version: str = Field(min_length=1, max_length=100)
    file_count: Annotated[int, Field(ge=0)]
    line_count: Annotated[int, Field(ge=0)]
    arc_count: Annotated[int, Field(ge=0)]
    rows: tuple[dict[str, JsonValue], ...]
    truncated: bool
    limitations: tuple[str, ...] = Field(default=(), max_length=1_024)


COVERAGE_WORKER = WorkerDefinition(
    operation=WorkerOperationId.COVERAGE_PARSE,
    module="flameox.workers.coverage",
    request=TypeAdapter(CoverageWorkerRequest),
    response=TypeAdapter(CoverageWorkerResult),
    name="coverage.py",
    implementation="flameox.workers.coverage/v1",
)
