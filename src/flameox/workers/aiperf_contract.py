from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId, WorkerOutputFile

AIPERF_NATIVE_MODEL: Literal["aiperf.common.models.MetricRecordInfo"] = (
    "aiperf.common.models.MetricRecordInfo"
)
AIPERF_PROJECTION_PROFILE: Literal["flameox.aiperf.request.v1"] = "flameox.aiperf.request.v1"
type AIPerfOutcome = Literal["succeeded", "failed", "cancelled"]
type AIPerfErrorCategory = Literal[
    "authentication",
    "cancelled",
    "connection",
    "invalid_request",
    "not_found",
    "permission_denied",
    "provider_error",
    "rate_limited",
    "server_error",
    "timeout",
    "unavailable",
]


class AIPerfWorkerRequest(ContractModel):
    artifact_path: str = Field(min_length=1, max_length=4_096)
    max_rows: Annotated[int, Field(gt=0, le=1_000_000)]
    max_line_bytes: Annotated[int, Field(gt=0, le=4 * 1024 * 1024)]


class AIPerfProjectionRow(ContractModel):
    """The complete prompt-free information allowed to leave the provider worker."""

    line_index: Annotated[int, Field(ge=0)]
    source_request_id: str = Field(min_length=1, max_length=500)
    provider_request_id: str | None = Field(default=None, max_length=500)
    conversation_id: str | None = Field(default=None, max_length=500)
    turn_index: Annotated[int, Field(ge=0)] | None = None
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    scheduled_ns: Annotated[int, Field(ge=0)] | None = None
    observed_started_ns: Annotated[int, Field(ge=0)]
    ttft_ns: Annotated[int, Field(ge=0)] | None = None
    latency_ns: Annotated[int, Field(ge=0)] | None = None
    tpot_ns: Annotated[int, Field(ge=0)] | None = None
    mean_itl_ns: Annotated[int, Field(ge=0)] | None = None
    outcome: AIPerfOutcome
    error_type: AIPerfErrorCategory | None = None
    error_code: str | None = Field(default=None, pattern=r"^[0-9]{1,3}$")


class AIPerfWorkerResult(ContractModel):
    output: WorkerOutputFile
    row_count: Annotated[int, Field(ge=0)]
    truncated: bool
    aiperf_version: str = Field(min_length=1, max_length=100)
    native_model: Literal["aiperf.common.models.MetricRecordInfo"] = AIPERF_NATIVE_MODEL
    projection_profile: Literal["flameox.aiperf.request.v1"] = AIPERF_PROJECTION_PROFILE


AIPERF_WORKER = WorkerDefinition(
    operation=WorkerOperationId.AIPERF_PARSE,
    module="flameox.workers.aiperf",
    request=TypeAdapter(AIPerfWorkerRequest),
    response=TypeAdapter(AIPerfWorkerResult),
    name="AIPerf",
    implementation="flameox.workers.aiperf/v1",
)
