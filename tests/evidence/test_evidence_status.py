from __future__ import annotations

import pytest

from flameox.action_graph import ActionId, tool_action
from flameox.evidence_status import recoverable_unavailable_evidence

pytestmark = pytest.mark.unit


def test_recoverable_unavailable_evidence_serializes_one_complete_action() -> None:
    evidence = recoverable_unavailable_evidence(
        "not_extracted",
        next_action=tool_action(
            ActionId.EXTRACT_MEMRAY,
            run_id="run-1",
            idempotency_key="extract-run-1",
        ),
    )

    assert evidence.model_dump(mode="json") == {
        "status": "unavailable",
        "reason": "not_extracted",
        "next_action": {
            "kind": "tool",
            "action": "artifact.extract.memray",
            "arguments": {
                "run_id": "run-1",
                "idempotency_key": "extract-run-1",
                "temporary_allocation_threshold": 1,
            },
        },
    }
