from __future__ import annotations

import pytest
from pydantic import ValidationError

from flameox.evidence_status import (
    parse_evidence_availability,
    recoverable_unavailable_evidence,
)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "status": "unavailable",
            "reason": "not_extracted",
            "next_tool": "extract_memray",
        },
        {
            "status": "unavailable",
            "reason": "not_extracted",
            "next_arguments": {"run_id": "run-1"},
        },
        {
            "status": "available",
            "reason": "evidence_present",
            "next_tool": "extract_memray",
            "next_arguments": {"run_id": "run-1"},
        },
    ],
)
def test_evidence_availability_rejects_incoherent_recovery(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_evidence_availability(payload)


def test_recoverable_unavailable_evidence_round_trips_with_one_complete_action() -> None:
    evidence = recoverable_unavailable_evidence(
        "not_extracted",
        next_tool="extract_memray",
        next_arguments={"run_id": "run-1"},
    )

    assert parse_evidence_availability(evidence.model_dump(mode="python")) == evidence
    assert evidence.model_dump(mode="json") == {
        "status": "unavailable",
        "reason": "not_extracted",
        "next_tool": "extract_memray",
        "next_arguments": {"run_id": "run-1"},
    }
