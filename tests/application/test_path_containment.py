from __future__ import annotations

from pathlib import Path

import pytest

from flameox.application.evidence_lookup import EvidenceLookupService
from flameox.application.records import (
    EvidenceInput,
    FindingService,
    RecordFindingRequest,
)
from flameox.application.recoverable_move import validate_manifest_id
from flameox.domain import (
    DomainError,
    ErrorCode,
    EvidenceLevel,
    EvidenceReferenceType,
    EvidenceRelation,
    FindingAssessment,
    FindingConfidence,
)
from flameox.storage import Workspace

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [".", "..", "D:outside"])
def test_manifest_ids_reject_current_and_parent_directory(value: str) -> None:
    with pytest.raises(DomainError) as error:
        validate_manifest_id(value, kind="manifest")

    assert error.value.code is ErrorCode.EXECUTION_REFUSED


@pytest.mark.parametrize("ref_id", [".", "..", "../outside", r"..\outside", "D:outside"])
def test_generation_lookup_rejects_malformed_digest(tmp_path: Path, ref_id: str) -> None:
    workspace = Workspace.initialize(tmp_path)

    with pytest.raises(DomainError) as error:
        EvidenceLookupService(workspace).get(EvidenceReferenceType.GENERATION, ref_id)

    assert error.value.code is ErrorCode.INVALID_ARGUMENTS


def test_finding_rejects_malformed_generation_digest(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    request = RecordFindingRequest(
        kind="performance",
        title="Candidate is faster",
        claim="The candidate is faster.",
        evidence_level=EvidenceLevel.DERIVED,
        confidence=FindingConfidence.HIGH,
        assessment=FindingAssessment.SUPPORTED,
        evidence=(
            EvidenceInput(
                ref_type=EvidenceReferenceType.GENERATION,
                ref_id="../outside",
                relation=EvidenceRelation.SUPPORTS,
            ),
        ),
    )

    with pytest.raises(DomainError) as error:
        FindingService(workspace).record(request)

    assert error.value.code is ErrorCode.INVALID_ARGUMENTS
