from __future__ import annotations

import json
from typing import ClassVar, Literal

from pydantic import Field, JsonValue

from flamo.catalog import Catalog
from flamo.domain import (
    DomainError,
    ErrorCode,
    EvidenceLevel,
    EvidenceReference,
    Finding,
    FindingAssessment,
    FindingLifecycle,
    Hypothesis,
    Investigation,
    InvestigationStatus,
    new_id,
)
from flamo.evidence import GenerationPublisher
from flamo.models import ContractModel
from flamo.storage import ArtifactStore, JsonRecordStore, RunStore, Workspace


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
    ref_type: Literal[
        "analysis",
        "artifact",
        "comparison",
        "generation",
        "observation",
        "run",
        "run_set",
        "trial",
    ]
    ref_id: str
    relation: Literal["supports", "contradicts", "context", "validates"]


class RecordFindingRequest(ContractModel):
    kind: str
    title: str = Field(min_length=1, max_length=500)
    claim: str = Field(min_length=1, max_length=500)
    evidence_level: EvidenceLevel
    confidence: Literal["high", "medium", "low", "unknown"]
    assessment: FindingAssessment
    lifecycle: FindingLifecycle = FindingLifecycle.ACTIVE
    limitations: tuple[str, ...] = ()
    next_experiments: tuple[dict[str, JsonValue], ...] = ()
    evidence: tuple[EvidenceInput, ...] = ()
    finding_id: str | None = None
    expected_revision: int | None = Field(default=None, ge=1)


class FindingResult(ContractModel):
    finding: Finding
    evidence: tuple[EvidenceReference, ...]
    corpus_commit_id: str


class InvestigationService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)
        self.investigations = JsonRecordStore(
            workspace,
            kind="investigations",
            model=Investigation,
            id_field="investigation_id",
        )
        self.hypotheses = JsonRecordStore(
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
        self.investigations.create(investigation)
        self.publisher.publish_rows(
            {"investigations": [self._investigation_row(investigation)]},
            publisher="flamo.investigations",
            publisher_version="1",
        )
        return investigation

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
            self.hypotheses.create(hypothesis)
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
            hypothesis = current.model_copy(
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
            )
        self.publisher.publish_rows(
            {"hypotheses": [self._hypothesis_row(hypothesis)]},
            publisher="flamo.hypotheses",
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
    _TABLES: ClassVar[dict[str, tuple[str, str]]] = {
        "analysis": ("analyses", "analysis_id"),
        "comparison": ("comparisons", "comparison_id"),
        "observation": ("observations", "observation_id"),
        "run_set": ("run_sets", "run_set_id"),
        "trial": ("trials", "trial_id"),
    }

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.publisher = GenerationPublisher(workspace)
        self.findings = JsonRecordStore(
            workspace,
            kind="findings",
            model=Finding,
            id_field="finding_id",
            revision_field="revision",
        )

    def record(self, request: RecordFindingRequest) -> FindingResult:
        if request.assessment is FindingAssessment.SUPPORTED and not request.evidence:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "A supported finding requires at least one evidence reference.",
            )
        if request.evidence_level is EvidenceLevel.OBSERVED and any(
            item.ref_type in {"analysis", "comparison"} for item in request.evidence
        ):
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "Derived analyses cannot be labeled as directly observed evidence.",
            )
        for item in request.evidence:
            self._require_reference(item)
            if (
                request.assessment is FindingAssessment.SUPPORTED
                and item.relation == "supports"
                and item.ref_type == "comparison"
                and not self._comparison_can_support(item.ref_id)
            ):
                raise DomainError(
                    ErrorCode.COMPARISON_INVALID,
                    "An invalid or inconclusive comparison cannot support a finding.",
                    details={"comparison_id": item.ref_id},
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
            self.findings.create(finding)
        else:
            if request.expected_revision is None:
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    "A finding revision requires expected_revision.",
                )
            current = self.findings.read(request.finding_id)
            finding = current.model_copy(
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
            self.findings.append(finding, expected_revision=request.expected_revision)
        references = tuple(
            EvidenceReference(
                owner_type="finding",
                owner_id=finding.finding_id,
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
            publisher="flamo.findings",
            publisher_version="1",
        )
        return FindingResult(
            finding=finding,
            evidence=references,
            corpus_commit_id=published.commit.commit_id,
        )

    def _require_reference(self, item: EvidenceInput) -> None:
        if item.ref_type == "run":
            RunStore(self.workspace).read(item.ref_id)
            return
        if item.ref_type == "artifact":
            ArtifactStore(self.workspace).get(item.ref_id)
            return
        if item.ref_type == "generation":
            path = self.workspace.paths.generations / item.ref_id / "manifest.json"
            if path.is_file():
                return
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Generation {item.ref_id!r} does not exist.",
            )
        table = self._TABLES[item.ref_type]
        with Catalog(self.workspace).open_snapshot() as snapshot:
            found = snapshot.execute(
                f'SELECT 1 FROM "{table[0]}" WHERE "{table[1]}" = ? LIMIT 1',
                (item.ref_id,),
            ).fetchone()
        if found is None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"{item.ref_type} evidence {item.ref_id!r} does not exist.",
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

    def _comparison_can_support(self, comparison_id: str) -> bool:
        with Catalog(self.workspace).open_snapshot() as snapshot:
            row = snapshot.execute(
                "SELECT validity, decision FROM comparisons WHERE comparison_id = ? "
                "ORDER BY published_at DESC LIMIT 1",
                (comparison_id,),
            ).fetchone()
        return (
            row is not None
            and row[0] == "valid"
            and row[1] not in {"inconclusive", "descriptive_only"}
        )
