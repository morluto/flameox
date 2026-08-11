from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import ConfigDict, Field, computed_field, model_validator

from flameox.domain import EvidenceLevel
from flameox.models import ContractModel


class InferenceRequestOutcomeKind(StrEnum):
    UNREPORTED = "unreported"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UnreportedInferenceRequestOutcome(ContractModel):
    kind: Literal[InferenceRequestOutcomeKind.UNREPORTED] = InferenceRequestOutcomeKind.UNREPORTED


class SucceededInferenceRequestOutcome(ContractModel):
    kind: Literal[InferenceRequestOutcomeKind.SUCCEEDED] = InferenceRequestOutcomeKind.SUCCEEDED


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


class InferenceRequestOutcomeColumns(NamedTuple):
    success: bool | None
    cancelled: bool | None
    error_type: str | None
    error_code: str | None


def inference_request_outcome_columns(
    outcome: InferenceRequestOutcome,
) -> InferenceRequestOutcomeColumns:
    if outcome.kind is InferenceRequestOutcomeKind.UNREPORTED:
        return InferenceRequestOutcomeColumns(None, None, None, None)
    error_type: str | None = None
    error_code: str | None = None
    if isinstance(outcome, FailedInferenceRequestOutcome | CancelledInferenceRequestOutcome):
        error_type = outcome.error_type
        error_code = outcome.error_code
    return InferenceRequestOutcomeColumns(
        success=outcome.kind is InferenceRequestOutcomeKind.SUCCEEDED,
        cancelled=outcome.kind is InferenceRequestOutcomeKind.CANCELLED,
        error_type=error_type,
        error_code=error_code,
    )


_LEGACY_OUTCOME_FIELDS = frozenset({"success", "cancelled", "error_type", "error_code"})


def _advertise_outcome_projections(schema: dict[str, Any]) -> None:
    properties = schema.setdefault("properties", {})
    assert isinstance(properties, dict)
    properties.pop("outcome", None)
    nullable_boolean = {"anyOf": [{"type": "boolean"}, {"type": "null"}], "readOnly": True}
    nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}], "readOnly": True}
    properties.update(
        {
            "success": nullable_boolean,
            "cancelled": nullable_boolean,
            "error_type": nullable_string,
            "error_code": nullable_string,
        }
    )
    required = schema.setdefault("required", [])
    assert isinstance(required, list)
    if "outcome" in required:
        required.remove("outcome")
    for field_name in _LEGACY_OUTCOME_FIELDS:
        if field_name not in required:
            required.append(field_name)


def _parse_flat_outcome(value: Mapping[str, object]) -> dict[str, object]:
    parsed = dict(value)
    supplied_fields = _LEGACY_OUTCOME_FIELDS.intersection(parsed)
    if not supplied_fields:
        return parsed
    if supplied_fields != _LEGACY_OUTCOME_FIELDS:
        missing = sorted(_LEGACY_OUTCOME_FIELDS - supplied_fields)
        raise ValueError(f"inference outcome is missing fields: {missing}")
    if "outcome" in parsed:
        raise ValueError("use either outcome or flattened outcome fields, not both")

    success = parsed.pop("success")
    cancelled = parsed.pop("cancelled")
    error_type = parsed.pop("error_type")
    error_code = parsed.pop("error_code")
    if success is None and cancelled is None and error_type is None and error_code is None:
        outcome: dict[str, object] = {"kind": InferenceRequestOutcomeKind.UNREPORTED}
    elif success is True and cancelled is False and error_type is None and error_code is None:
        outcome = {"kind": InferenceRequestOutcomeKind.SUCCEEDED}
    elif success is False and cancelled is True:
        outcome = {
            "kind": InferenceRequestOutcomeKind.CANCELLED,
            "error_type": error_type,
            "error_code": error_code,
        }
    elif success is False and cancelled is False:
        outcome = {
            "kind": InferenceRequestOutcomeKind.FAILED,
            "error_type": error_type,
            "error_code": error_code,
        }
    else:
        raise ValueError("inference outcome fields do not describe a supported outcome")
    parsed["outcome"] = outcome
    return parsed


class InferenceRequestItem(ContractModel):
    """Canonical request evidence with legacy storage columns as projections."""

    model_config = ConfigDict(json_schema_extra=_advertise_outcome_projections)

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
    outcome: InferenceRequestOutcome = Field(exclude=True)
    queue_ns: int | None
    prefill_ns: int | None
    decode_ns: int | None
    cache_hit: bool | None
    prefix_hash_count: int | None
    evidence_level: EvidenceLevel

    @model_validator(mode="before")
    @classmethod
    def parse_flat_storage_outcome(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        return _parse_flat_outcome(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success(self) -> bool | None:
        return inference_request_outcome_columns(self.outcome).success

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cancelled(self) -> bool | None:
        return inference_request_outcome_columns(self.outcome).cancelled

    @computed_field  # type: ignore[prop-decorator]
    @property
    def error_type(self) -> str | None:
        return inference_request_outcome_columns(self.outcome).error_type

    @computed_field  # type: ignore[prop-decorator]
    @property
    def error_code(self) -> str | None:
        return inference_request_outcome_columns(self.outcome).error_code
