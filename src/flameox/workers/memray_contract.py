from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, TypeAdapter

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId, WorkerOutputFile


class MemrayWorkerRequest(ContractModel):
    artifact_path: str = Field(min_length=1, max_length=4_096)
    run_id: str = Field(min_length=1, max_length=200)
    artifact_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    workload_cwd: str | None = Field(default=None, min_length=1, max_length=4_096)
    project_root: str = Field(min_length=1, max_length=4_096)
    source_state_id: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    generation_id: str = Field(min_length=1, max_length=200)
    published_at: datetime
    extractor_name: Literal["memray"] = "memray"
    extractor_version: Literal["3"] = "3"


class MemrayWorkerResult(ContractModel):
    reader_version: str = Field(min_length=1, max_length=100)
    peak_memory_bytes: int = Field(ge=0)
    retained_end_bytes: int = Field(ge=0)
    total_allocations: int = Field(ge=0)
    frame_count: int = Field(ge=0)
    has_native_traces: bool
    files: tuple[WorkerOutputFile, ...] = Field(min_length=3, max_length=3)


MEMRAY_WORKER = WorkerDefinition(
    operation=WorkerOperationId.MEMRAY_PARSE,
    module="flameox.workers.memray",
    request=TypeAdapter(MemrayWorkerRequest),
    response=TypeAdapter(MemrayWorkerResult),
    name="Memray",
    implementation="flameox.workers.memray/v3",
    timeout_seconds=300,
)
