from __future__ import annotations

import json
from datetime import datetime

from pydantic import Field, JsonValue

from flameox.domain import (
    AnalysisRecord,
    EvidenceReference,
    EvidenceReferenceType,
    EvidenceRelation,
    digest_model,
)
from flameox.models import ContractModel


class AnalysisReferenceTarget(ContractModel):
    ref_type: EvidenceReferenceType
    ref_id: str
    relation: EvidenceRelation = EvidenceRelation.CONTEXT


class AnalysisProvenanceInput(ContractModel):
    recipe: str
    recipe_version: str = "1"
    parameters: dict[str, JsonValue]
    corpus_commit_id: str
    input_generation_ids: tuple[str, ...] = ()
    input_run_ids: tuple[str, ...] = ()
    input_artifact_ids: tuple[str, ...] = ()
    result_digest: str
    coverage: dict[str, JsonValue] = Field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    started_at: datetime
    completed_at: datetime
    references: tuple[AnalysisReferenceTarget, ...] = ()


class AnalysisProvenance(ContractModel):
    analysis: AnalysisRecord
    evidence: tuple[EvidenceReference, ...]

    def rows(self) -> dict[str, list[dict[str, object]]]:
        return {
            "analyses": [_analysis_row(self.analysis)],
            "evidence_refs": [reference.model_dump(mode="python") for reference in self.evidence],
        }


def _analysis_row(value: AnalysisRecord) -> dict[str, object]:
    return {
        "analysis_id": value.analysis_id,
        "recipe": value.recipe,
        "recipe_version": value.recipe_version,
        "parameters_json": json.dumps(
            value.parameters,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "parameters_digest": value.parameters_digest,
        "corpus_commit_id": value.corpus_commit_id,
        "input_generation_ids": list(value.input_generation_ids),
        "input_run_ids": list(value.input_run_ids),
        "input_artifact_ids": list(value.input_artifact_ids),
        "result_digest": value.result_digest,
        "result_artifact_id": value.result_artifact_id,
        "coverage_json": json.dumps(
            value.coverage,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "limitations": list(value.limitations),
        "started_at": value.started_at,
        "completed_at": value.completed_at,
    }


def build_analysis_provenance(
    value: AnalysisProvenanceInput,
) -> AnalysisProvenance:
    parameters_digest = digest_model(value.parameters)
    analysis_id = digest_model(
        {
            "identity_profile": "flameox.analysis.v1",
            "recipe": value.recipe,
            "recipe_version": value.recipe_version,
            "parameters_digest": parameters_digest,
            "corpus_commit_id": value.corpus_commit_id,
            "input_generation_ids": value.input_generation_ids,
            "input_run_ids": value.input_run_ids,
            "input_artifact_ids": value.input_artifact_ids,
            "result_digest": value.result_digest,
            "coverage": value.coverage,
            "limitations": value.limitations,
        }
    )
    analysis = AnalysisRecord(
        analysis_id=analysis_id,
        recipe=value.recipe,
        recipe_version=value.recipe_version,
        parameters=value.parameters,
        parameters_digest=parameters_digest,
        corpus_commit_id=value.corpus_commit_id,
        input_generation_ids=value.input_generation_ids,
        input_run_ids=value.input_run_ids,
        input_artifact_ids=value.input_artifact_ids,
        result_digest=value.result_digest,
        coverage=value.coverage,
        limitations=value.limitations,
        started_at=value.started_at,
        completed_at=value.completed_at,
    )
    return AnalysisProvenance(
        analysis=analysis,
        evidence=tuple(
            sorted(
                (
                    EvidenceReference(
                        owner_type="analysis",
                        owner_id=analysis.analysis_id,
                        ref_type=target.ref_type,
                        ref_id=target.ref_id,
                        relation=target.relation,
                    )
                    for target in value.references
                ),
                key=lambda reference: (
                    reference.ref_type.value,
                    reference.ref_id,
                    reference.relation.value,
                ),
            )
        ),
    )


def context_references(
    *,
    run_ids: tuple[str, ...] = (),
    artifact_ids: tuple[str, ...] = (),
    generation_ids: tuple[str, ...] = (),
    run_set_ids: tuple[str, ...] = (),
) -> tuple[AnalysisReferenceTarget, ...]:
    return tuple(
        AnalysisReferenceTarget(
            ref_type=ref_type,
            ref_id=ref_id,
        )
        for ref_type, values in (
            (EvidenceReferenceType.RUN, run_ids),
            (EvidenceReferenceType.ARTIFACT, artifact_ids),
            (EvidenceReferenceType.GENERATION, generation_ids),
            (EvidenceReferenceType.RUN_SET, run_set_ids),
        )
        for ref_id in values
    )
