from __future__ import annotations

import json
from typing import Annotated, cast

from pydantic import Field, JsonValue, model_validator

from flameox.application.evidence_lookup import EvidenceLookupService
from flameox.application.evidence_relations import (
    assessment_relation_error,
    qualify_evidence_relation,
)
from flameox.domain import (
    CursorNamespace,
    DomainError,
    ErrorCode,
    EvidenceLevel,
    EvidenceReference,
    EvidenceReferenceType,
    EvidenceRelation,
    Finding,
    FindingAssessment,
    FindingConfidence,
    FindingLifecycle,
    Hypothesis,
    Investigation,
    InvestigationStatus,
    new_id,
)
from flameox.evidence import GenerationPublisher
from flameox.models import ContractModel
from flameox.pagination import CursorPageContract
from flameox.storage import ControlRecordStore, CursorStore, Workspace
from flameox.storage.control_plane import ControlRelationship


class CreateInvestigationRequest(ContractModel):
    question: str = Field(min_length=1, max_length=500)
    symptom: str | None = None
    parent_investigation_id: str | None = None


class RecordHypothesisRequest(ContractModel):
    investigation_id: str
    claim: str = Field(min_length=1, max_length=500)
    prediction: str = Field(min_length=1, max_length=500)
    discriminating_condition: str = Field(min_length=1, max_length=500)
    hypothesis_id: str | None = None
    expected_revision: int | None = Field(default=None, ge=1)
    assessment: FindingAssessment = FindingAssessment.UNASSESSED
    lifecycle: FindingLifecycle = FindingLifecycle.ACTIVE


class EvidenceInput(ContractModel):
    ref_type: EvidenceReferenceType
    ref_id: str
    relation: EvidenceRelation


class RecordFindingRequest(ContractModel):
    kind: str
    title: str = Field(min_length=1, max_length=500)
    claim: str = Field(min_length=1, max_length=500)
    evidence_level: EvidenceLevel
    confidence: FindingConfidence
    assessment: FindingAssessment
    lifecycle: FindingLifecycle = FindingLifecycle.ACTIVE
    limitations: tuple[str, ...] = ()
    next_experiments: tuple[dict[str, JsonValue], ...] = ()
    evidence: Annotated[tuple[EvidenceInput, ...], Field(max_length=50)] = ()
    finding_id: str | None = None
    expected_revision: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def evidence_edges_are_unique(self) -> RecordFindingRequest:
        edges = tuple((item.ref_type, item.ref_id, item.relation) for item in self.evidence)
        if len(edges) != len(set(edges)):
            raise ValueError("finding evidence relations must be unique")
        return self


class FindingResult(ContractModel):
    finding: Finding
    evidence: tuple[EvidenceReference, ...]
    corpus_commit_id: str


class InvestigationListResult(CursorPageContract):
    page_items_field = "investigations"

    corpus_commit_id: str
    investigations: tuple[Investigation, ...]
    total: int


class FindingListResult(CursorPageContract):
    page_items_field = "findings"

    corpus_commit_id: str
    findings: tuple[Finding, ...]
    total: int


class InvestigationService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)
        self.evidence = EvidenceLookupService(workspace)
        self.investigations = ControlRecordStore(
            workspace,
            kind="investigations",
            model=Investigation,
            id_field="investigation_id",
        )
        self.hypotheses = ControlRecordStore(
            workspace,
            kind="hypotheses",
            model=Hypothesis,
            id_field="hypothesis_id",
            revision_field="revision",
        )

    def create(self, request: CreateInvestigationRequest) -> Investigation:
        if request.parent_investigation_id is not None:
            self.investigations.read(request.parent_investigation_id)
        investigation = Investigation(
            investigation_id=new_id(),
            question=request.question,
            symptom=request.symptom,
            project_root=str(self.workspace.project_root),
            status=InvestigationStatus.OPEN,
            parent_investigation_id=request.parent_investigation_id,
        )
        relationships = (
            (
                ControlRelationship(
                    relationship="parent",
                    target_kind="investigations",
                    target_id=request.parent_investigation_id,
                ),
            )
            if request.parent_investigation_id is not None
            else ()
        )
        self.investigations.create(investigation, relationships=relationships)
        self.publisher.publish_rows(
            {"investigations": [self._investigation_row(investigation)]},
            publisher="flameox.investigations",
            publisher_version="1",
        )
        return investigation

    def list(self, *, limit: int, cursor: str | None = None) -> InvestigationListResult:
        with self.evidence.session() as evidence:
            offset = _decode_offset(
                cursor,
                cursors=self.workspace.cursors,
                namespace=CursorNamespace.INVESTIGATIONS,
                snapshot_id=evidence.commit_id,
            )
            selected, total = evidence.list_investigations(offset=offset, limit=limit)
            corpus_commit_id = evidence.commit_id
        next_offset = offset + len(selected)
        return InvestigationListResult(
            corpus_commit_id=corpus_commit_id,
            investigations=selected,
            total=total,
            next_cursor=_encode_offset(
                cursors=self.workspace.cursors,
                namespace=CursorNamespace.INVESTIGATIONS,
                snapshot_id=corpus_commit_id,
                offset=next_offset,
                total=total,
            ),
        )

    def record_hypothesis(self, request: RecordHypothesisRequest) -> Hypothesis:
        self.investigations.read(request.investigation_id)
        if request.hypothesis_id is None:
            if request.expected_revision is not None:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "A new hypothesis cannot specify expected_revision.",
                )
            hypothesis = Hypothesis(
                hypothesis_id=new_id(),
                investigation_id=request.investigation_id,
                claim=request.claim,
                prediction=request.prediction,
                discriminating_condition=request.discriminating_condition,
                assessment=request.assessment,
                lifecycle=request.lifecycle,
            )
            self.hypotheses.create(
                hypothesis,
                relationships=(
                    ControlRelationship(
                        relationship="belongs_to",
                        target_kind="investigations",
                        target_id=request.investigation_id,
                    ),
                ),
            )
        else:
            if request.expected_revision is None:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "A hypothesis revision requires expected_revision.",
                )
            current = self.hypotheses.read(request.hypothesis_id)
            if current.investigation_id != request.investigation_id:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "A hypothesis cannot move between investigations.",
                )
            hypothesis = current.validated_copy(
                update={
                    "revision": request.expected_revision + 1,
                    "claim": request.claim,
                    "prediction": request.prediction,
                    "discriminating_condition": request.discriminating_condition,
                    "assessment": request.assessment,
                    "lifecycle": request.lifecycle,
                }
            )
            self.hypotheses.append(
                hypothesis,
                expected_revision=request.expected_revision,
                relationships=(
                    ControlRelationship(
                        relationship="belongs_to",
                        target_kind="investigations",
                        target_id=request.investigation_id,
                    ),
                ),
            )
        self.publisher.publish_rows(
            {"hypotheses": [self._hypothesis_row(hypothesis)]},
            publisher="flameox.hypotheses",
            publisher_version="1",
        )
        return hypothesis

    def _investigation_row(self, value: Investigation) -> dict[str, object]:
        return {
            **value.model_dump(mode="python"),
            "status": value.status.value,
        }

    def _hypothesis_row(self, value: Hypothesis) -> dict[str, object]:
        return {
            **value.model_dump(mode="python"),
            "assessment": value.assessment.value,
            "lifecycle": value.lifecycle.value,
        }


class FindingService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)
        self.evidence = EvidenceLookupService(workspace)
        self.findings = ControlRecordStore(
            workspace,
            kind="findings",
            model=Finding,
            id_field="finding_id",
            revision_field="revision",
        )

    def record(self, request: RecordFindingRequest) -> FindingResult:
        if any(
            item.ref_type is EvidenceReferenceType.STATIC_CANDIDATE
            and item.relation is not EvidenceRelation.CONTEXT
            for item in request.evidence
        ):
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Static-analysis candidates can only provide context; measured evidence must "
                "carry the supporting or contradicting relation.",
            )
        relation_error = assessment_relation_error(
            request.assessment,
            tuple(item.relation for item in request.evidence),
        )
        if relation_error is not None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                relation_error,
            )
        if request.evidence_level is EvidenceLevel.OBSERVED and any(
            item.ref_type
            in {
                EvidenceReferenceType.ANALYSIS,
                EvidenceReferenceType.COMPARISON,
            }
            for item in request.evidence
        ):
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Derived analyses cannot be labeled as directly observed evidence.",
            )
        with self.evidence.session() as evidence:
            for item in request.evidence:
                qualification = qualify_evidence_relation(
                    evidence,
                    ref_type=item.ref_type,
                    ref_id=item.ref_id,
                    relation=item.relation,
                )
                if not qualification.qualified:
                    raise DomainError(
                        (
                            ErrorCode.COMPARISON_INVALID
                            if item.ref_type is EvidenceReferenceType.COMPARISON
                            else ErrorCode.WORKSPACE_INVALID
                        ),
                        f"Evidence cannot carry its declared relation: {qualification.reason}.",
                        details={
                            "ref_type": item.ref_type.value,
                            "ref_id": item.ref_id,
                            "relation": item.relation.value,
                        },
                    )
        relationships = tuple(
            ControlRelationship(
                relationship=item.relation.value,
                target_kind=item.ref_type.value,
                target_id=item.ref_id,
            )
            for item in request.evidence
        )
        if request.finding_id is None:
            if request.expected_revision is not None:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "A new finding cannot specify expected_revision.",
                )
            finding = Finding(
                finding_id=new_id(),
                revision=1,
                kind=request.kind,
                title=request.title,
                claim=request.claim,
                evidence_level=request.evidence_level,
                confidence=request.confidence,
                assessment=request.assessment,
                lifecycle=request.lifecycle,
                limitations=request.limitations,
                next_experiments=request.next_experiments,
            )
            self.findings.create(finding, relationships=relationships)
        else:
            if request.expected_revision is None:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "A finding revision requires expected_revision.",
                )
            current = self.findings.read(request.finding_id)
            finding = current.validated_copy(
                update={
                    "revision": request.expected_revision + 1,
                    "kind": request.kind,
                    "title": request.title,
                    "claim": request.claim,
                    "evidence_level": request.evidence_level,
                    "confidence": request.confidence,
                    "assessment": request.assessment,
                    "lifecycle": request.lifecycle,
                    "limitations": request.limitations,
                    "next_experiments": request.next_experiments,
                }
            )
            self.findings.append(
                finding,
                expected_revision=request.expected_revision,
                relationships=relationships,
            )
        references = tuple(
            EvidenceReference(
                owner_type="finding",
                owner_id=finding.finding_id,
                owner_revision=finding.revision,
                ref_type=item.ref_type,
                ref_id=item.ref_id,
                relation=item.relation,
            )
            for item in request.evidence
        )
        published = self.publisher.publish_rows(
            {
                "findings": [self._finding_row(finding)],
                "evidence_refs": [reference.model_dump(mode="python") for reference in references],
            },
            publisher="flameox.findings",
            publisher_version="1",
        )
        return FindingResult(
            finding=finding,
            evidence=references,
            corpus_commit_id=published.commit.commit_id,
        )

    def get(self, finding_id: str) -> FindingResult:
        finding = self.findings.read(finding_id)
        with self.evidence.session() as evidence:
            references = evidence.references(
                owner_type="finding",
                owner_id=finding.finding_id,
                owner_revision=finding.revision,
            )
            corpus_commit_id = evidence.commit_id
        return FindingResult(
            finding=finding,
            evidence=references,
            corpus_commit_id=corpus_commit_id,
        )

    def list(self, *, limit: int, cursor: str | None = None) -> FindingListResult:
        with self.evidence.session() as evidence:
            offset = _decode_offset(
                cursor,
                cursors=self.workspace.cursors,
                namespace=CursorNamespace.FINDINGS,
                snapshot_id=evidence.commit_id,
            )
            selected, total = evidence.list_findings(offset=offset, limit=limit)
            corpus_commit_id = evidence.commit_id
        next_offset = offset + len(selected)
        return FindingListResult(
            corpus_commit_id=corpus_commit_id,
            findings=selected,
            total=total,
            next_cursor=_encode_offset(
                cursors=self.workspace.cursors,
                namespace=CursorNamespace.FINDINGS,
                snapshot_id=corpus_commit_id,
                offset=next_offset,
                total=total,
            ),
        )

    def _finding_row(self, value: Finding) -> dict[str, object]:
        return {
            "finding_id": value.finding_id,
            "revision": value.revision,
            "created_at": value.created_at,
            "kind": value.kind,
            "title": value.title,
            "claim": value.claim,
            "evidence_level": value.evidence_level.value,
            "confidence": value.confidence,
            "assessment": value.assessment.value,
            "lifecycle": value.lifecycle.value,
            "limitations": list(value.limitations),
            "next_experiments_json": json.dumps(
                value.next_experiments,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }


def _decode_offset(
    cursor: str | None,
    *,
    cursors: CursorStore,
    namespace: CursorNamespace,
    snapshot_id: str,
) -> int:
    if cursor is None:
        return 0
    position = cast(
        tuple[int],
        cursors.resolve(
            cursor,
            namespace=namespace,
            snapshot_id=snapshot_id,
            scope_digest="all",
        ),
    )
    return position[0]


def _encode_offset(
    *,
    cursors: CursorStore,
    namespace: CursorNamespace,
    snapshot_id: str,
    offset: int,
    total: int,
) -> str | None:
    if offset >= total:
        return None
    return cursors.issue(
        namespace=namespace,
        snapshot_id=snapshot_id,
        scope_digest="all",
        position=(offset,),
    )
