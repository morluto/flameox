from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from flameox.application import (
    EvidenceInput,
    EvidenceSummaryRequest,
    EvidenceSummaryService,
    FindingService,
    RecordFindingRequest,
)
from flameox.application.evidence_lookup import EvidenceLookupService
from flameox.domain import (
    DomainError,
    ErrorCode,
    EvidenceLevel,
    EvidenceReferenceType,
    EvidenceRelation,
    FindingAssessment,
    FindingConfidence,
    digest_model,
)
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace
from flameox.storage.control_plane import ControlPlane

pytestmark = [pytest.mark.integration, pytest.mark.serial]


def _publish_analyses(workspace: Workspace, *analysis_ids: str) -> None:
    observed_at = datetime.now(UTC)
    parameters: dict[str, object] = {}
    GenerationPublisher(workspace).publish_rows(
        {
            "analyses": [
                {
                    "analysis_id": analysis_id,
                    "recipe": "relation-test",
                    "recipe_version": "1",
                    "parameters_json": "{}",
                    "parameters_digest": digest_model(parameters),
                    "corpus_commit_id": workspace.corpus.read_head().commit_id,
                    "input_generation_ids": [],
                    "input_run_ids": [],
                    "input_artifact_ids": [],
                    "result_digest": digest_model({"analysis_id": analysis_id}),
                    "result_artifact_id": None,
                    "coverage_json": "{}",
                    "limitations": [],
                    "started_at": observed_at,
                    "completed_at": observed_at,
                }
                for analysis_id in analysis_ids
            ]
        },
        publisher="finding-relation-test",
        publisher_version="1",
    )


def _request(
    *,
    assessment: FindingAssessment,
    edges: tuple[EvidenceInput, ...],
    finding_id: str | None = None,
    expected_revision: int | None = None,
) -> RecordFindingRequest:
    return RecordFindingRequest(
        kind="correctness",
        title="Relation-aware finding",
        claim="The selected evidence establishes the claim.",
        evidence_level=EvidenceLevel.DERIVED,
        confidence=FindingConfidence.HIGH,
        assessment=assessment,
        evidence=edges,
        finding_id=finding_id,
        expected_revision=expected_revision,
    )


@pytest.mark.parametrize(
    "relation",
    (EvidenceRelation.CONTEXT, EvidenceRelation.CONTRADICTS),
)
def test_supported_finding_rejects_edges_without_supporting_force(
    tmp_path: Path,
    relation: EvidenceRelation,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _publish_analyses(workspace, "analysis-a")

    with pytest.raises(DomainError) as error:
        FindingService(workspace).record(
            _request(
                assessment=FindingAssessment.SUPPORTED,
                edges=(
                    EvidenceInput(
                        ref_type=EvidenceReferenceType.ANALYSIS,
                        ref_id="analysis-a",
                        relation=relation,
                    ),
                ),
            )
        )

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
    assert "supports or validates" in error.value.message


def test_refuted_finding_requires_contradicting_evidence(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _publish_analyses(workspace, "analysis-a")

    with pytest.raises(DomainError, match="requires a contradicts"):
        FindingService(workspace).record(
            _request(
                assessment=FindingAssessment.REFUTED,
                edges=(
                    EvidenceInput(
                        ref_type=EvidenceReferenceType.ANALYSIS,
                        ref_id="analysis-a",
                        relation=EvidenceRelation.SUPPORTS,
                    ),
                ),
            )
        )


def test_decisive_finding_rejects_unresolved_mixed_relations(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _publish_analyses(workspace, "analysis-a", "analysis-b")
    mixed = (
        EvidenceInput(
            ref_type=EvidenceReferenceType.ANALYSIS,
            ref_id="analysis-a",
            relation=EvidenceRelation.SUPPORTS,
        ),
        EvidenceInput(
            ref_type=EvidenceReferenceType.ANALYSIS,
            ref_id="analysis-b",
            relation=EvidenceRelation.CONTRADICTS,
        ),
    )

    with pytest.raises(DomainError, match="must remain inconclusive"):
        FindingService(workspace).record(
            _request(assessment=FindingAssessment.SUPPORTED, edges=mixed)
        )

    recorded = FindingService(workspace).record(
        _request(assessment=FindingAssessment.INCONCLUSIVE, edges=mixed)
    )
    summary = EvidenceSummaryService(workspace).summarize(
        EvidenceSummaryRequest(finding_ids=(recorded.finding.finding_id,))
    )
    assert summary.summary.claims[0].support_status == "mixed_unresolved"


def test_validates_relation_requires_semantic_validation_source(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _publish_analyses(workspace, "analysis-a")

    with pytest.raises(DomainError) as error:
        FindingService(workspace).record(
            _request(
                assessment=FindingAssessment.SUPPORTED,
                edges=(
                    EvidenceInput(
                        ref_type=EvidenceReferenceType.ANALYSIS,
                        ref_id="analysis-a",
                        relation=EvidenceRelation.VALIDATES,
                    ),
                ),
            )
        )

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
    assert "no semantic-validation contract" in error.value.message


def test_finding_revisions_own_disjoint_evidence_edges(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _publish_analyses(workspace, "analysis-a", "analysis-b")
    service = FindingService(workspace)
    first = service.record(
        _request(
            assessment=FindingAssessment.SUPPORTED,
            edges=(
                EvidenceInput(
                    ref_type=EvidenceReferenceType.ANALYSIS,
                    ref_id="analysis-a",
                    relation=EvidenceRelation.SUPPORTS,
                ),
            ),
        )
    )
    second = service.record(
        _request(
            assessment=FindingAssessment.SUPPORTED,
            finding_id=first.finding.finding_id,
            expected_revision=1,
            edges=(
                EvidenceInput(
                    ref_type=EvidenceReferenceType.ANALYSIS,
                    ref_id="analysis-b",
                    relation=EvidenceRelation.SUPPORTS,
                ),
            ),
        )
    )

    with EvidenceLookupService(workspace).session() as evidence:
        revision_one = evidence.references(
            owner_type="finding",
            owner_id=first.finding.finding_id,
            owner_revision=1,
        )
        revision_two = evidence.references(
            owner_type="finding",
            owner_id=first.finding.finding_id,
            owner_revision=2,
        )
    assert [item.ref_id for item in revision_one] == ["analysis-a"]
    assert [item.ref_id for item in revision_two] == ["analysis-b"]
    assert second.finding.revision == 2
    control = ControlPlane(workspace)
    assert [
        item.target_id
        for item in control.list_revision_relationships(
            source_kind="findings",
            source_id=first.finding.finding_id,
            source_revision=1,
        )
    ] == ["analysis-a"]
    assert [
        item.target_id
        for item in control.list_revision_relationships(
            source_kind="findings",
            source_id=first.finding.finding_id,
            source_revision=2,
        )
    ] == ["analysis-b"]

    summary = EvidenceSummaryService(workspace).summarize(
        EvidenceSummaryRequest(finding_ids=(first.finding.finding_id,))
    )
    assert summary.summary.claims[0].evidence[0]["ref_id"] == "analysis-b"
    assert summary.summary.claims[0].support_status == "supported_by_evidence"
