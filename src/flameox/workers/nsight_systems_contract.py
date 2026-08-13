from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId


class NsightSystemsWorkerRequest(ContractModel):
    schema_version: Literal[1] = 1
    artifact_path: str = Field(min_length=1, max_length=4_096)
    max_rows_per_table: Annotated[int, Field(gt=0, le=100_000_000)]


class NsightSystemsWorkerResult(ContractModel):
    schema_version: Literal[1] = 1
    schema_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    tables: tuple[str, ...] = Field(max_length=1_024)
    events: tuple[dict[str, JsonValue], ...]
    coverage: dict[str, bool] = Field(max_length=64)
    truncated_tables: tuple[str, ...] = Field(max_length=1_024)


NSIGHT_SYSTEMS_WORKER = WorkerDefinition(
    operation=WorkerOperationId.NSIGHT_SYSTEMS_PARSE,
    module="flameox.workers.nsight_systems",
    request=TypeAdapter(NsightSystemsWorkerRequest),
    response=TypeAdapter(NsightSystemsWorkerResult),
    name="Nsight Systems",
    implementation="flameox.workers.nsight_systems/v1",
)
