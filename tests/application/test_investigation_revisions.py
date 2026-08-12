from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application import (
    CreateInvestigationRequest,
    InvestigationService,
    RecordHypothesisRequest,
)
from flameox.catalog import Catalog
from flameox.domain import (
    DomainError,
    ErrorCode,
)
from flameox.storage import Workspace
from flameox.storage.control_plane import ControlPlane, ControlRelationship


def test_investigation_hypothesis_revision_uses_compare_and_swap(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    Catalog(workspace).rebuild()
    service = InvestigationService(workspace)
    investigation = service.create(
        CreateInvestigationRequest(
            question="Why does the reverse scan scale linearly?",
        )
    )
    first = service.record_hypothesis(
        RecordHypothesisRequest(
            investigation_id=investigation.investigation_id,
            claim="Python loop overhead dominates.",
            prediction="Runtime doubles with length.",
            discriminating_condition="A vectorized implementation removes the slope.",
        )
    )
    second = service.record_hypothesis(
        RecordHypothesisRequest(
            investigation_id=investigation.investigation_id,
            hypothesis_id=first.hypothesis_id,
            expected_revision=1,
            claim=first.claim,
            prediction="Runtime approximately doubles with length.",
            discriminating_condition=first.discriminating_condition,
        )
    )

    assert second.revision == 2
    assert ControlPlane(workspace).list_relationships(
        source_kind="hypotheses",
        source_id=second.hypothesis_id,
    ) == (
        ControlRelationship(
            relationship="belongs_to",
            target_kind="investigations",
            target_id=investigation.investigation_id,
        ),
    )
    with pytest.raises(DomainError) as stale:
        service.record_hypothesis(
            RecordHypothesisRequest(
                investigation_id=investigation.investigation_id,
                hypothesis_id=first.hypothesis_id,
                expected_revision=1,
                claim=first.claim,
                prediction=first.prediction,
                discriminating_condition=first.discriminating_condition,
            )
        )
    assert stale.value.code is ErrorCode.REVISION_CONFLICT
