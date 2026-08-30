from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from flameox.domain import EvidenceLevel
from flameox.models import ContractModel


class InferenceRequestOutcomeKind(StrEnum):
    UNREPORTED = "unreported"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UnreportedInferenceRequestOutcome(ContractModel):
    kind: Literal[InferenceRequestOutcomeKind.UNREPORTED] = InferenceRequestOutcomeKind.UNREPORTED
    error_type: Literal[None] = Field(default=None, exclude_if=lambda value: value is None)
    error_code: Literal[None] = Field(default=None, exclude_if=lambda value: value is None)


class SucceededInferenceRequestOutcome(ContractModel):
    kind: Literal[InferenceRequestOutcomeKind.SUCCEEDED] = InferenceRequestOutcomeKind.SUCCEEDED
    error_type: Literal[None] = Field(default=None, exclude_if=lambda value: value is None)
    error_code: Literal[None] = Field(default=None, exclude_if=lambda value: value is None)


class FailedInferenceRequestOutcome(ContractModel):
    kind: Literal[InferenceRequestOutcomeKind.FAILED] = InferenceRequestOutcomeKind.FAILED
    error_type: str | None = None
    error_code: str | None = None


class CancelledInferenceRequestOutcome(ContractModel):
    kind: Literal[InferenceRequestOutcomeKind.CANCELLED] = InferenceRequestOutcomeKind.CANCELLED
    error_type: str | None = None
    error_code: str | None = None


type ReportedInferenceRequestOutcome = Annotated[
    SucceededInferenceRequestOutcome
    | FailedInferenceRequestOutcome
    | CancelledInferenceRequestOutcome,
    Field(discriminator="kind"),
]

type InferenceRequestOutcome = Annotated[
    UnreportedInferenceRequestOutcome
    | SucceededInferenceRequestOutcome
    | FailedInferenceRequestOutcome
    | CancelledInferenceRequestOutcome,
    Field(discriminator="kind"),
]


class InferenceRequestItem(ContractModel):
    """Canonical request evidence with one typed provider outcome."""

    request_id: str
    run_id: str
    artifact_id: str
    source_request_id: str
    provider_request_id: str | None
    input_tokens: int
    output_tokens: int
    scheduled_ns: int | None
    observed_started_ns: int | None
    ttft_ns: int | None
    latency_ns: int | None
    tpot_ns: int | None
    mean_itl_ns: int | None
    outcome: InferenceRequestOutcome
    queue_ns: int | None
    prefill_ns: int | None
    decode_ns: int | None
    cache_hit: bool | None
    prefix_hash_count: int | None
    evidence_level: EvidenceLevel
