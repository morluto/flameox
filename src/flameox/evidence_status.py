from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, TypeAdapter

from flameox.action_graph import ToolAction
from flameox.models import ContractModel


class EvidenceStatus(StrEnum):
    AVAILABLE = "available"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class _EvidenceAvailability(ContractModel):
    """Fields shared by every machine-readable evidence state."""

    reason: str = Field(min_length=1, max_length=200)


class _EvidenceWithoutRecovery(_EvidenceAvailability):
    next_action: Literal[None] = None


class AvailableEvidence(_EvidenceWithoutRecovery):
    status: Literal[EvidenceStatus.AVAILABLE] = EvidenceStatus.AVAILABLE


class EmptyEvidence(_EvidenceWithoutRecovery):
    status: Literal[EvidenceStatus.EMPTY] = EvidenceStatus.EMPTY


class RecoverableEmptyEvidence(_EvidenceAvailability):
    status: Literal[EvidenceStatus.EMPTY] = EvidenceStatus.EMPTY
    next_action: ToolAction


class PartialEvidence(_EvidenceWithoutRecovery):
    status: Literal[EvidenceStatus.PARTIAL] = EvidenceStatus.PARTIAL


class UnknownEvidence(_EvidenceWithoutRecovery):
    status: Literal[EvidenceStatus.UNKNOWN] = EvidenceStatus.UNKNOWN


class UnavailableEvidence(_EvidenceWithoutRecovery):
    status: Literal[EvidenceStatus.UNAVAILABLE] = EvidenceStatus.UNAVAILABLE


class RecoverableUnavailableEvidence(_EvidenceAvailability):
    status: Literal[EvidenceStatus.UNAVAILABLE] = EvidenceStatus.UNAVAILABLE
    next_action: ToolAction


type EvidenceAvailability = (
    AvailableEvidence
    | EmptyEvidence
    | RecoverableEmptyEvidence
    | PartialEvidence
    | UnknownEvidence
    | RecoverableUnavailableEvidence
    | UnavailableEvidence
)

_EVIDENCE_AVAILABILITY_ADAPTER: TypeAdapter[EvidenceAvailability] = TypeAdapter(
    EvidenceAvailability
)


def parse_evidence_availability(value: object) -> EvidenceAvailability:
    return _EVIDENCE_AVAILABILITY_ADAPTER.validate_python(value)


def available_availability(reason: str = "evidence_present") -> EvidenceAvailability:
    return AvailableEvidence(reason=reason)


def empty_availability(reason: str = "no_matching_evidence") -> EvidenceAvailability:
    return EmptyEvidence(reason=reason)


def recoverable_empty_evidence(
    reason: str,
    *,
    next_action: ToolAction,
) -> EvidenceAvailability:
    return RecoverableEmptyEvidence(reason=reason, next_action=next_action)


def partial_availability(reason: str) -> EvidenceAvailability:
    return PartialEvidence(reason=reason)


def unavailable_availability(reason: str) -> EvidenceAvailability:
    return UnavailableEvidence(reason=reason)


def recoverable_unavailable_evidence(
    reason: str,
    *,
    next_action: ToolAction,
) -> EvidenceAvailability:
    return RecoverableUnavailableEvidence(
        reason=reason,
        next_action=next_action,
    )
