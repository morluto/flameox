from __future__ import annotations

from dataclasses import dataclass

from flameox.application.evidence_lookup import EvidenceSession
from flameox.domain import (
    EvidenceReferenceType,
    EvidenceRelation,
    FindingAssessment,
)

_DECISIVE_COMPARISON_RELATIONS = {
    EvidenceRelation.SUPPORTS,
    EvidenceRelation.CONTRADICTS,
    EvidenceRelation.VALIDATES,
}
_POSITIVE_RELATIONS = {
    EvidenceRelation.SUPPORTS,
    EvidenceRelation.VALIDATES,
}


@dataclass(frozen=True, slots=True)
class EvidenceRelationQualification:
    """The semantic force an evidence edge is allowed to carry.

    The relation is a claim made by the finding author.  Qualification answers
    the narrower question that Flameox can establish mechanically: whether the
    referenced record has enough recorded state to carry that relation at all.
    It deliberately does not infer that arbitrary evidence proves arbitrary
    prose.
    """

    qualified: bool
    reason: str | None = None


def qualify_evidence_relation(
    evidence: EvidenceSession,
    *,
    ref_type: EvidenceReferenceType,
    ref_id: str,
    relation: EvidenceRelation,
) -> EvidenceRelationQualification:
    record = evidence.get(ref_type, ref_id)
    data = record.data

    if relation is EvidenceRelation.CONTEXT:
        return EvidenceRelationQualification(qualified=True)

    if ref_type is EvidenceReferenceType.COMPARISON:
        if relation not in _DECISIVE_COMPARISON_RELATIONS:
            return EvidenceRelationQualification(qualified=True)
        if data.get("validity") != "valid":
            return EvidenceRelationQualification(
                qualified=False,
                reason="comparison is not valid",
            )
        if data.get("decision") in {None, "inconclusive", "descriptive_only"}:
            return EvidenceRelationQualification(
                qualified=False,
                reason="comparison has no decision-bearing result",
            )
        return EvidenceRelationQualification(qualified=True)

    if relation is not EvidenceRelation.VALIDATES:
        return EvidenceRelationQualification(qualified=True)

    if ref_type is EvidenceReferenceType.RUN:
        if data.get("execution_status") != "succeeded":
            return EvidenceRelationQualification(
                qualified=False,
                reason="run execution did not succeed",
            )
        if data.get("validation_status") != "passed":
            return EvidenceRelationQualification(
                qualified=False,
                reason="run validation did not pass",
            )
        return EvidenceRelationQualification(qualified=True)

    if ref_type is EvidenceReferenceType.TRIAL:
        if data.get("outcome") != "succeeded":
            return EvidenceRelationQualification(
                qualified=False,
                reason="trial outcome did not succeed",
            )
        if data.get("validation_status") != "passed":
            return EvidenceRelationQualification(
                qualified=False,
                reason="trial validation did not pass",
            )
        return EvidenceRelationQualification(qualified=True)

    return EvidenceRelationQualification(
        qualified=False,
        reason=f"{ref_type.value} evidence has no semantic-validation contract",
    )


def assessment_relation_error(
    assessment: FindingAssessment,
    relations: tuple[EvidenceRelation, ...],
) -> str | None:
    positive = any(relation in _POSITIVE_RELATIONS for relation in relations)
    contradictory = EvidenceRelation.CONTRADICTS in relations

    if assessment is FindingAssessment.SUPPORTED and not positive:
        return "A supported finding requires a supports or validates evidence relation."
    if assessment is FindingAssessment.REFUTED and not contradictory:
        return "A refuted finding requires a contradicts evidence relation."
    if (
        assessment in {FindingAssessment.SUPPORTED, FindingAssessment.REFUTED}
        and positive
        and contradictory
    ):
        return (
            "Mixed supporting and contradicting evidence must remain inconclusive "
            "until a new finding revision resolves the contradiction."
        )
    return None


def is_positive_relation(relation: EvidenceRelation) -> bool:
    return relation in _POSITIVE_RELATIONS
