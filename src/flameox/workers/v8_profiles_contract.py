from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId


class V8ProfileRequest(ContractModel):
    profile_kind: Literal["cpu", "heap"]
    artifact_path: str = Field(min_length=1, max_length=4_096)
    artifact_id: str = Field(min_length=1, max_length=200)
    project_root: str = Field(min_length=1, max_length=4_096)
    max_nodes: Annotated[int, Field(gt=0, le=100_000)]
    max_samples: Annotated[int, Field(gt=0, le=1_000_000)]
    max_rows: Annotated[int, Field(gt=0, le=100_000)]


class V8ProfileResult(ContractModel):
    profile_kind: Literal["cpu", "heap"]
    node_count: Annotated[int, Field(ge=0)]
    sample_count: Annotated[int, Field(ge=0)]
    total_sampled_bytes: Annotated[int, Field(ge=0)] = 0
    frames: tuple[dict[str, JsonValue], ...]
    frame_measurements: tuple[dict[str, JsonValue], ...]
    limitations: tuple[str, ...] = ()


V8_PROFILE_WORKER: WorkerDefinition[V8ProfileRequest, V8ProfileResult] = WorkerDefinition(
    operation=WorkerOperationId.V8_PROFILE_PARSE,
    module="flameox.workers.v8_profiles",
    request=TypeAdapter(V8ProfileRequest),
    response=TypeAdapter(V8ProfileResult),
    name="V8 profile",
    implementation="flameox.workers.v8_profiles/v1",
)
