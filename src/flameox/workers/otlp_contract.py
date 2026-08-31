from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from flameox.models import ContractModel
from flameox.workers.protocol import WorkerDefinition, WorkerOperationId

type OtlpRow = dict[str, JsonValue]


class OtlpWorkerRequest(ContractModel):
    artifact_path: str = Field(min_length=1, max_length=4_096)
    media_type: str = Field(min_length=1, max_length=200)
    row_limit: Annotated[int, Field(gt=0, le=100_000_000)]
    start_ns: Annotated[int, Field(ge=0)] | None = None
    end_ns: Annotated[int, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def window_is_complete_and_ordered(self) -> OtlpWorkerRequest:
        if (self.start_ns is None) != (self.end_ns is None):
            raise ValueError("OTLP window bounds must be supplied together")
        if self.start_ns is not None and self.end_ns is not None and self.end_ns <= self.start_ns:
            raise ValueError("OTLP end_ns must be greater than start_ns")
        return self


class OtlpWorkerResult(ContractModel):
    row_limit_exceeded: bool = False
    resources: tuple[OtlpRow, ...] = ()
    scopes: tuple[OtlpRow, ...] = ()
    spans: tuple[OtlpRow, ...] = ()
    events: tuple[OtlpRow, ...] = ()
    links: tuple[OtlpRow, ...] = ()
    counts: dict[Literal["resources", "scopes", "spans", "events", "links"], int] = Field(
        default_factory=dict, max_length=5
    )
    limitations: tuple[str, ...] = Field(default=(), max_length=1_024)

    @model_validator(mode="after")
    def result_has_one_shape(self) -> OtlpWorkerResult:
        if self.row_limit_exceeded:
            if set(self.counts) != {"resources", "scopes", "spans", "events", "links"}:
                raise ValueError("row-limit result requires observed counts")
        elif self.counts:
            raise ValueError("successful row result cannot carry limit counts")
        return self


OTLP_WORKER = WorkerDefinition(
    operation=WorkerOperationId.OTLP_PARSE,
    module="flameox.workers.otlp",
    request=TypeAdapter(OtlpWorkerRequest),
    response=TypeAdapter(OtlpWorkerResult),
    name="OTLP",
    implementation="flameox.workers.otlp/v1",
)
