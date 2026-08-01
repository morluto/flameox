from __future__ import annotations

from typing import Literal

from pydantic import Field

from flameox.models import ContractModel

EvidenceStatus = Literal["available", "empty", "unavailable", "partial", "unknown"]


class EvidenceAvailability(ContractModel):
    """Machine-readable evidence presence shared by analysis and query results."""

    status: EvidenceStatus
    reason: str = Field(min_length=1, max_length=200)
    next_tool: str | None = None
    next_arguments: dict[str, object] | None = None


def available_availability() -> EvidenceAvailability:
    return EvidenceAvailability(status="available", reason="evidence_present")


def empty_availability(reason: str = "no_matching_evidence") -> EvidenceAvailability:
    return EvidenceAvailability(status="empty", reason=reason)
