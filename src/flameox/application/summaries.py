from __future__ import annotations

import html
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, TextIO, cast

from pydantic import Field, JsonValue, model_validator

from flameox.application.evidence_lookup import EvidenceLookupService, EvidenceSession
from flameox.application.evidence_relations import (
    is_positive_relation,
    qualify_evidence_relation,
)
from flameox.domain import (
    ArtifactKind,
    EvidenceLevel,
    EvidenceReferenceType,
    EvidenceRelation,
    ExecutionStatus,
    FindingAssessment,
    LimitationDetail,
    LimitationSource,
    Sensitivity,
    TrialFailureClass,
    TrialOutcome,
    ValidationStatus,
    digest_model,
)
from flameox.models import ContractModel
from flameox.storage import Workspace

_DISCARD_CHUNK_SIZE = 4096


def _discard_rest_of_line(stream: TextIO) -> None:
    """Discard the remainder of an overlong line without materializing it.

    Instead of calling unbounded ``stream.readline()`` which loads the
    entire remaining line into memory, read fixed-size chunks until a
    newline or EOF is encountered.  Peak memory is bounded by
    ``_DISCARD_CHUNK_SIZE``, not by the physical line length.
    """
    while True:
        chunk = stream.readline(_DISCARD_CHUNK_SIZE)
        if not chunk or chunk.endswith("\n"):
            break


_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class SummaryExcerptPolicy(StrEnum):
    NONE = "none"
    INTERNAL = "internal"


class SummarySensitiveContextPolicy(StrEnum):
    REDACT = "redact"
    INCLUDE = "include"


class SummaryRunRole(StrEnum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"
    CONTEXT = "context"


class SummaryReferenceKind(StrEnum):
    COMPARISON = "comparison"
    ANALYSIS = "analysis"


class SummarySupportStatus(StrEnum):
    SUPPORTED_BY_EVIDENCE = "supported_by_evidence"
    RECORDED_SUPPORT_MISSING = "recorded_support_missing"
    CONTRADICTED = "contradicted"
    MIXED_UNRESOLVED = "mixed_unresolved"
    NOT_SUPPORTING = "not_supporting"


class SummaryProofShape(StrEnum):
    VALIDATED_PAIR = "validated_pair"
    CANDIDATE_VALIDATED = "candidate_validated"
    BASELINE_OBSERVATION = "baseline_observation"
    INCOMPLETE = "incomplete"
    SELECTED_EVIDENCE = "selected_evidence"


class SummarySelectionShape(StrEnum):
    SINGLE_RUN = "single_run"
    TWO_CLAIMED_RUNS = "two_claimed_runs"
    MIXED_EVIDENCE = "mixed_evidence"


class SummaryClaimedRole(ContractModel):
    run_id: str
    role: SummaryRunRole


class SummaryProofAssessment(ContractModel):
    shape: SummaryProofShape
    binding_type: Literal["comparison", "run_validation"] | None = None
    binding_id: str | None = None
    verified_baseline_run_id: str | None = None
    verified_candidate_run_id: str | None = None
    satisfied_conditions: tuple[str, ...] = ()
    missing_conditions: tuple[str, ...] = ()


def _unique_limitation_details(
    details: tuple[LimitationDetail, ...] | list[LimitationDetail],
) -> tuple[LimitationDetail, ...]:
    result: list[LimitationDetail] = []
    seen: set[tuple[str, str, str]] = set()
    for item in details:
        key = (item.source, item.code, item.message)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return tuple(result)


def _summary_limitation_details(
    runs: tuple[SummaryRun, ...],
) -> tuple[LimitationDetail, ...]:
    return _unique_limitation_details([item for run in runs for item in run.limitation_details])


class EvidenceSummaryRequest(ContractModel):
    baseline_run_id: str | None = None
    candidate_run_id: str | None = None
    run_ids: Annotated[tuple[str, ...], Field(max_length=20)] = ()
    comparison_ids: Annotated[tuple[str, ...], Field(max_length=10)] = ()
    analysis_ids: Annotated[tuple[str, ...], Field(max_length=20)] = ()
    finding_ids: Annotated[tuple[str, ...], Field(max_length=20)] = ()
    output_excerpts: SummaryExcerptPolicy = SummaryExcerptPolicy.NONE
    sensitive_context: SummarySensitiveContextPolicy = SummarySensitiveContextPolicy.REDACT

    @model_validator(mode="after")
    def bounded_unique_selection(self) -> EvidenceSummaryRequest:
        runs = tuple(
            value
            for value in (self.baseline_run_id, self.candidate_run_id, *self.run_ids)
            if value is not None
        )
        references = (*runs, *self.comparison_ids, *self.analysis_ids, *self.finding_ids)
        if not references:
            raise ValueError("select at least one evidence reference")
        if len(references) > 50:
            raise ValueError("evidence summaries are limited to 50 references")
        if len(references) != len(set(references)):
            raise ValueError("evidence summary references must be unique")
        return self


class SummaryArtifact(ContractModel):
    artifact_id: str
    kind: ArtifactKind
    role: str
    sensitivity: Sensitivity
    producer: str | None = None
    producer_version: str | None = None
    excerpt: tuple[str, ...] = ()
    excerpt_truncated: bool = False


class SummaryAttempt(ContractModel):
    trial_id: str
    outcome: TrialOutcome
    failure_class: TrialFailureClass
    exclusion_reason: str | None = None
    combination_id: str
    factors: dict[str, JsonValue] = Field(default_factory=dict)


class SummaryRun(ContractModel):
    run_id: str
    claimed_role: SummaryRunRole
    execution_status: ExecutionStatus
    validation_status: ValidationStatus
    workload_definition_id: str | None
    workload_instance_id: str | None
    measurement_protocol_id: str | None
    inference_protocol_identity_id: str | None
    argv: tuple[str, ...]
    source: dict[str, JsonValue]
    environment: dict[str, JsonValue]
    execution_identity: dict[str, JsonValue] | None = None
    external_context: dict[str, JsonValue] | None = None
    artifacts: tuple[SummaryArtifact, ...]
    attempts: tuple[SummaryAttempt, ...] = ()
    limitations: tuple[str, ...] = ()
    limitation_details: tuple[LimitationDetail, ...] = ()


class SummaryReference(ContractModel):
    ref_type: SummaryReferenceKind
    ref_id: str
    data: dict[str, JsonValue]


class SummaryClaim(ContractModel):
    finding_id: str
    title: str
    claim: str
    evidence_level: EvidenceLevel
    assessment: FindingAssessment
    support_status: SummarySupportStatus
    evidence: tuple[dict[str, JsonValue], ...]
    limitations: tuple[str, ...] = ()


class EvidenceSummary(ContractModel):
    summary_digest: str
    corpus_commit_id: str
    selection_shape: SummarySelectionShape
    claimed_roles: tuple[SummaryClaimedRole, ...]
    proof_shape: SummaryProofShape
    proof_assessment: SummaryProofAssessment
    runs: tuple[SummaryRun, ...]
    references: tuple[SummaryReference, ...]
    claims: tuple[SummaryClaim, ...]
    limitations: tuple[str, ...]
    limitation_details: tuple[LimitationDetail, ...] = ()
    truncation: tuple[str, ...] = ()


class EvidenceSummaryBundle(ContractModel):
    summary: EvidenceSummary
    markdown: str


class EvidenceSummaryService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.evidence = EvidenceLookupService(workspace)

    def summarize(self, request: EvidenceSummaryRequest) -> EvidenceSummaryBundle:
        selected_runs = self._selected_runs(request)
        truncation: list[str] = []
        with self.evidence.session() as evidence:
            runs = tuple(
                self._run_summary(
                    evidence,
                    run_id,
                    role,
                    request=request,
                    truncation=truncation,
                )
                for run_id, role in selected_runs
            )
            references = tuple(
                [
                    SummaryReference(
                        ref_type=SummaryReferenceKind.COMPARISON,
                        ref_id=ref_id,
                        data=self._compact_reference(
                            SummaryReferenceKind.COMPARISON,
                            evidence.get(EvidenceReferenceType.COMPARISON, ref_id).data,
                        ),
                    )
                    for ref_id in request.comparison_ids
                ]
                + [
                    SummaryReference(
                        ref_type=SummaryReferenceKind.ANALYSIS,
                        ref_id=ref_id,
                        data=self._compact_reference(
                            SummaryReferenceKind.ANALYSIS,
                            evidence.get(EvidenceReferenceType.ANALYSIS, ref_id).data,
                        ),
                    )
                    for ref_id in request.analysis_ids
                ]
            )
            claims = tuple(self._claim(evidence, finding_id) for finding_id in request.finding_ids)
            proof_assessment = self._proof_assessment(
                evidence,
                runs=runs,
                references=references,
            )
            corpus_commit_id = evidence.commit_id
        selection_shape = self._selection_shape(request)
        claimed_roles = tuple(
            SummaryClaimedRole(run_id=run_id, role=role) for run_id, role in selected_runs
        )
        limitations = self._limitations(
            request,
            runs,
            references,
            claims,
            proof_assessment,
        )
        limitation_details = _summary_limitation_details(runs)
        payload = {
            "corpus_commit_id": corpus_commit_id,
            "selection_shape": selection_shape,
            "claimed_roles": [item.model_dump(mode="json") for item in claimed_roles],
            "proof_shape": proof_assessment.shape,
            "proof_assessment": proof_assessment.model_dump(mode="json"),
            "runs": [run.model_dump(mode="json") for run in runs],
            "references": [item.model_dump(mode="json") for item in references],
            "claims": [claim.model_dump(mode="json") for claim in claims],
            "limitations": limitations,
            "limitation_details": [item.model_dump(mode="json") for item in limitation_details],
            "truncation": truncation,
        }
        summary = EvidenceSummary(
            summary_digest=digest_model(payload),
            corpus_commit_id=corpus_commit_id,
            selection_shape=selection_shape,
            claimed_roles=claimed_roles,
            proof_shape=proof_assessment.shape,
            proof_assessment=proof_assessment,
            runs=runs,
            references=references,
            claims=claims,
            limitations=limitations,
            limitation_details=limitation_details,
            truncation=tuple(truncation),
        )
        return EvidenceSummaryBundle(
            summary=summary,
            markdown=render_evidence_summary_markdown(summary),
        )

    @staticmethod
    def _selected_runs(
        request: EvidenceSummaryRequest,
    ) -> tuple[tuple[str, SummaryRunRole], ...]:
        values: list[tuple[str, SummaryRunRole]] = []
        if request.baseline_run_id is not None:
            values.append((request.baseline_run_id, SummaryRunRole.BASELINE))
        if request.candidate_run_id is not None:
            values.append((request.candidate_run_id, SummaryRunRole.CANDIDATE))
        values.extend((run_id, SummaryRunRole.CONTEXT) for run_id in request.run_ids)
        return tuple(values)

    def _run_summary(
        self,
        evidence: EvidenceSession,
        run_id: str,
        role: SummaryRunRole,
        *,
        request: EvidenceSummaryRequest,
        truncation: list[str],
    ) -> SummaryRun:
        run = evidence.run(run_id)
        source = self._identity_row(
            evidence,
            "source_states",
            "source_state_id",
            run.source_state_id,
        )
        environment = self._identity_row(
            evidence,
            "environments",
            "environment_id",
            run.environment_id,
        )
        artifacts: list[SummaryArtifact] = []
        excerpted = 0
        for registration in run.artifacts[:50]:
            excerpt: tuple[str, ...] = ()
            excerpt_truncated = False
            effective_sensitivity = registration.sensitivity
            if (
                request.output_excerpts is SummaryExcerptPolicy.INTERNAL
                and registration.kind
                in {ArtifactKind.PROCESS_OUTPUT, ArtifactKind.VALIDATION_OUTPUT}
                and registration.sensitivity is not Sensitivity.SENSITIVE
            ):
                artifact = evidence.artifact(registration.artifact_id)
                effective_sensitivity = artifact.metadata.effective_sensitivity
                if excerpted < 4 and effective_sensitivity is not Sensitivity.SENSITIVE:
                    excerpt, excerpt_truncated = self._excerpt(artifact.payload_path)
                    excerpted += 1
                elif excerpted >= 4:
                    truncation.append(f"run:{run_id}:output_artifacts")
            artifacts.append(
                SummaryArtifact(
                    artifact_id=registration.artifact_id,
                    kind=registration.kind,
                    role=registration.role,
                    sensitivity=effective_sensitivity,
                    producer=registration.producer,
                    producer_version=registration.producer_version,
                    excerpt=excerpt,
                    excerpt_truncated=excerpt_truncated,
                )
            )
        if len(run.artifacts) > 50:
            truncation.append(f"run:{run_id}:artifacts")
        attempts, attempts_truncated = self._attempts(evidence, run_id)
        if attempts_truncated:
            truncation.append(f"run:{run_id}:attempts")
        external_context = (
            run.external_context.model_dump(mode="json")
            if run.external_context is not None
            else None
        )
        if (
            external_context is not None
            and run.external_context is not None
            and run.external_context.sensitivity is Sensitivity.SENSITIVE
            and request.sensitive_context is SummarySensitiveContextPolicy.REDACT
        ):
            for field in ("lease_id", "worker_id", "orchestration_run_id"):
                external_context[field] = "[redacted]"
        argv = run.command.argv if run.command is not None else ()
        run_limitations = list(run.limitations)
        run_limitation_details = list(run.limitation_details)
        if len(argv) > 1 and request.sensitive_context is SummarySensitiveContextPolicy.REDACT:
            argv = (
                argv[0],
                f"[{len(argv) - 1} arguments redacted; digest={digest_model(argv[1:])}]",
            )
            run_limitations.append(
                "Command arguments are redacted by default; request sensitive context "
                "explicitly to include them."
            )
            run_limitation_details.append(
                LimitationDetail(
                    source=LimitationSource.VALIDATION,
                    code="sensitive_context_redacted",
                    message=run_limitations[-1],
                )
            )
        return SummaryRun(
            run_id=run_id,
            claimed_role=role,
            execution_status=run.execution_status,
            validation_status=run.validation_status,
            workload_definition_id=run.workload_definition_id,
            workload_instance_id=run.workload_instance_id,
            measurement_protocol_id=run.measurement_protocol_id,
            inference_protocol_identity_id=run.inference_protocol_identity_id,
            argv=argv,
            source=source,
            environment=environment,
            execution_identity=(
                run.execution_identity.model_dump(mode="json")
                if run.execution_identity is not None
                else None
            ),
            external_context=external_context,
            artifacts=tuple(artifacts),
            attempts=attempts,
            limitations=tuple(run_limitations),
            limitation_details=_unique_limitation_details(run_limitation_details),
        )

    @staticmethod
    def _identity_row(
        evidence: EvidenceSession,
        table: str,
        identifier: str,
        value: str | None,
    ) -> dict[str, JsonValue]:
        if value is None:
            return {"identity": None, "identity_quality": "unknown"}
        connection = evidence.execute(
            f'SELECT * FROM "{table}" WHERE "{identifier}" = ? ORDER BY published_at DESC LIMIT 1',
            (value,),
        )
        row = connection.fetchone()
        if row is None:
            return {"identity": value, "identity_quality": "unknown"}
        columns = [item[0] for item in connection.description]
        result: dict[str, JsonValue] = {}
        for name, item in zip(columns, row, strict=True):
            if name.endswith("_json") and isinstance(item, str):
                result[name.removesuffix("_json")] = cast(JsonValue, json.loads(item))
            elif item is None or isinstance(item, str | int | float | bool):
                result[name] = item
            elif isinstance(item, list | tuple):
                result[name] = cast(JsonValue, list(item))
            else:
                result[name] = str(item)
        return result

    @staticmethod
    def _attempts(
        evidence: EvidenceSession,
        run_id: str,
    ) -> tuple[tuple[SummaryAttempt, ...], bool]:
        rows = evidence.execute(
            "SELECT trial_id, outcome, failure_class, exclusion_reason, combination_id, "
            "factors_json FROM trials WHERE run_id = ? "
            "ORDER BY published_at DESC LIMIT 21",
            (run_id,),
        ).fetchall()
        return (
            tuple(
                SummaryAttempt(
                    trial_id=str(row[0]),
                    outcome=TrialOutcome(str(row[1])),
                    failure_class=TrialFailureClass(
                        str(row[2]) if row[2] is not None else TrialFailureClass.NONE
                    ),
                    exclusion_reason=str(row[3]) if row[3] is not None else None,
                    combination_id=str(row[4] or row[0]),
                    factors=(
                        cast(dict[str, JsonValue], json.loads(str(row[5])))
                        if row[5] is not None
                        else {}
                    ),
                )
                for row in rows[:20]
            ),
            len(rows) > 20,
        )

    def _claim(
        self,
        evidence_session: EvidenceSession,
        finding_id: str,
    ) -> SummaryClaim:
        finding = evidence_session.finding(finding_id)
        references = evidence_session.references(
            owner_type="finding",
            owner_id=finding_id,
            owner_revision=finding.revision,
        )
        assessed: list[tuple[EvidenceRelation, bool, str | None]] = []
        evidence_rows: list[dict[str, JsonValue]] = []
        for reference in references:
            qualification = qualify_evidence_relation(
                evidence_session,
                ref_type=reference.ref_type,
                ref_id=reference.ref_id,
                relation=reference.relation,
            )
            assessed.append((reference.relation, qualification.qualified, qualification.reason))
            evidence_rows.append(
                {
                    "ref_type": cast(JsonValue, reference.ref_type.value),
                    "ref_id": cast(JsonValue, reference.ref_id),
                    "relation": cast(JsonValue, reference.relation.value),
                    "qualified": qualification.qualified,
                    "qualification_reason": cast(JsonValue, qualification.reason),
                }
            )
        evidence = tuple(evidence_rows)
        qualified_positive = any(
            qualified and is_positive_relation(relation)
            for relation, qualified, _reason in assessed
        )
        qualified_contradiction = any(
            qualified and relation is EvidenceRelation.CONTRADICTS
            for relation, qualified, _reason in assessed
        )
        status: SummarySupportStatus
        limitations = list(finding.limitations)
        limitations.extend(
            f"Evidence relation {relation.value} is unqualified: {reason}."
            for relation, qualified, reason in assessed
            if not qualified and reason is not None
        )
        if qualified_positive and qualified_contradiction:
            status = SummarySupportStatus.MIXED_UNRESOLVED
        elif qualified_contradiction:
            status = SummarySupportStatus.CONTRADICTED
        elif finding.assessment is FindingAssessment.SUPPORTED and qualified_positive:
            status = SummarySupportStatus.SUPPORTED_BY_EVIDENCE
        elif finding.assessment is FindingAssessment.SUPPORTED:
            status = SummarySupportStatus.RECORDED_SUPPORT_MISSING
            limitations.append(
                "The recorded supported assessment has no qualified supports or validates edge."
            )
        else:
            status = SummarySupportStatus.NOT_SUPPORTING
        return SummaryClaim(
            finding_id=finding.finding_id,
            title=finding.title,
            claim=finding.claim,
            evidence_level=finding.evidence_level,
            assessment=finding.assessment,
            support_status=status,
            evidence=evidence,
            limitations=tuple(limitations),
        )

    @staticmethod
    def _selection_shape(request: EvidenceSummaryRequest) -> SummarySelectionShape:
        selected_run_count = sum(
            value is not None for value in (request.baseline_run_id, request.candidate_run_id)
        ) + len(request.run_ids)
        non_run_count = (
            len(request.comparison_ids) + len(request.analysis_ids) + len(request.finding_ids)
        )
        if selected_run_count == 1 and non_run_count == 0:
            return SummarySelectionShape.SINGLE_RUN
        if (
            request.baseline_run_id is not None
            and request.candidate_run_id is not None
            and not request.run_ids
            and non_run_count == 0
        ):
            return SummarySelectionShape.TWO_CLAIMED_RUNS
        return SummarySelectionShape.MIXED_EVIDENCE

    def _proof_assessment(
        self,
        evidence: EvidenceSession,
        *,
        runs: tuple[SummaryRun, ...],
        references: tuple[SummaryReference, ...],
    ) -> SummaryProofAssessment:
        baseline = next(
            (run for run in runs if run.claimed_role is SummaryRunRole.BASELINE),
            None,
        )
        candidate = next(
            (run for run in runs if run.claimed_role is SummaryRunRole.CANDIDATE),
            None,
        )
        if baseline is None and candidate is None:
            return SummaryProofAssessment(shape=SummaryProofShape.SELECTED_EVIDENCE)
        if baseline is not None and candidate is None:
            return SummaryProofAssessment(
                shape=SummaryProofShape.BASELINE_OBSERVATION,
                verified_baseline_run_id=baseline.run_id,
                satisfied_conditions=("baseline_run_resolved",),
            )
        if baseline is None:
            assert candidate is not None
            satisfied, missing = self._semantic_validation_conditions(evidence, candidate)
            if not missing:
                return SummaryProofAssessment(
                    shape=SummaryProofShape.CANDIDATE_VALIDATED,
                    binding_type="run_validation",
                    binding_id=candidate.run_id,
                    verified_candidate_run_id=candidate.run_id,
                    satisfied_conditions=satisfied,
                )
            return SummaryProofAssessment(
                shape=SummaryProofShape.INCOMPLETE,
                satisfied_conditions=satisfied,
                missing_conditions=missing,
            )

        assert candidate is not None
        baseline_satisfied, baseline_missing = self._semantic_validation_conditions(
            evidence,
            baseline,
            prefix="baseline_",
        )
        candidate_satisfied, candidate_missing = self._semantic_validation_conditions(
            evidence,
            candidate,
            prefix="candidate_",
        )
        pair_satisfied = [*baseline_satisfied, *candidate_satisfied]
        pair_missing = [*baseline_missing, *candidate_missing]
        for code, compatible in (
            (
                "workload_definition_compatible",
                baseline.workload_definition_id is not None
                and baseline.workload_definition_id == candidate.workload_definition_id,
            ),
            (
                "measurement_protocol_compatible",
                baseline.measurement_protocol_id is not None
                and baseline.measurement_protocol_id == candidate.measurement_protocol_id,
            ),
            (
                "environment_compatible",
                self._environment_identity(baseline) is not None
                and self._environment_identity(baseline) == self._environment_identity(candidate),
            ),
        ):
            (pair_satisfied if compatible else pair_missing).append(code)

        bindings = [
            reference
            for reference in references
            if reference.ref_type is SummaryReferenceKind.COMPARISON
            and self._comparison_binds_runs(
                evidence,
                reference,
                baseline_run_id=baseline.run_id,
                candidate_run_id=candidate.run_id,
            )
        ]
        if len(bindings) == 1:
            pair_satisfied.append("exact_valid_comparison_binding")
        elif not bindings:
            pair_missing.append("exact_valid_comparison_binding")
        else:
            pair_missing.append("unambiguous_comparison_binding")
        if not pair_missing and len(bindings) == 1:
            binding = bindings[0]
            return SummaryProofAssessment(
                shape=SummaryProofShape.VALIDATED_PAIR,
                binding_type="comparison",
                binding_id=binding.ref_id,
                verified_baseline_run_id=baseline.run_id,
                verified_candidate_run_id=candidate.run_id,
                satisfied_conditions=tuple(pair_satisfied),
            )
        return SummaryProofAssessment(
            shape=SummaryProofShape.INCOMPLETE,
            satisfied_conditions=tuple(pair_satisfied),
            missing_conditions=tuple(dict.fromkeys(pair_missing)),
        )

    @staticmethod
    def _semantic_validation_conditions(
        evidence: EvidenceSession,
        run: SummaryRun,
        *,
        prefix: str = "",
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        manifest = evidence.run(run.run_id)
        satisfied: list[str] = []
        missing: list[str] = []
        execution_eligible = manifest.execution_status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.NOT_APPLICABLE,
        }
        (satisfied if execution_eligible else missing).append(f"{prefix}execution_eligible")
        validation_passed = manifest.validation_status is ValidationStatus.PASSED
        (satisfied if validation_passed else missing).append(f"{prefix}validation_passed")
        receipt_passed = (
            manifest.oracle_receipt is not None and manifest.oracle_receipt.receipt.status == "pass"
        )
        if not receipt_passed and manifest.inference_protocol_identity_json is not None:
            try:
                protocol = json.loads(manifest.inference_protocol_identity_json)
                oracle_result = (
                    protocol.get("oracle_result") if isinstance(protocol, dict) else None
                )
                receipt_passed = (
                    isinstance(oracle_result, dict) and oracle_result.get("status") == "pass"
                )
            except (TypeError, ValueError):
                receipt_passed = False
        (satisfied if receipt_passed else missing).append(f"{prefix}semantic_validation_receipt")
        return tuple(satisfied), tuple(missing)

    @staticmethod
    def _environment_identity(run: SummaryRun) -> str | None:
        value = run.environment.get("environment_id") or run.environment.get("identity")
        return value if isinstance(value, str) else None

    @staticmethod
    def _comparison_binds_runs(
        evidence: EvidenceSession,
        reference: SummaryReference,
        *,
        baseline_run_id: str,
        candidate_run_id: str,
    ) -> bool:
        data = evidence.get(EvidenceReferenceType.COMPARISON, reference.ref_id).data
        if data.get("validity") != "valid":
            return False
        if data.get("decision") in {None, "inconclusive", "descriptive_only"}:
            return False
        baseline_set_id = data.get("baseline_run_set_id")
        candidate_set_id = data.get("candidate_run_set_id")
        if not isinstance(baseline_set_id, str) or not isinstance(candidate_set_id, str):
            return False
        return EvidenceSummaryService._run_set_exactly_contains(
            evidence,
            baseline_set_id,
            baseline_run_id,
        ) and EvidenceSummaryService._run_set_exactly_contains(
            evidence,
            candidate_set_id,
            candidate_run_id,
        )

    @staticmethod
    def _run_set_exactly_contains(
        evidence: EvidenceSession,
        run_set_id: str,
        run_id: str,
    ) -> bool:
        data = evidence.get(EvidenceReferenceType.RUN_SET, run_set_id).data
        raw_members = data.get("members_json")
        raw_selection = data.get("selection_json")
        membership_digest = data.get("membership_digest")
        corpus_commit_id = data.get("corpus_commit_id")
        if (
            not isinstance(raw_members, str)
            or not isinstance(raw_selection, str)
            or not isinstance(membership_digest, str)
            or not isinstance(corpus_commit_id, str)
        ):
            return False
        try:
            members = json.loads(raw_members)
            selection = json.loads(raw_selection)
        except (TypeError, ValueError):
            return False
        if (
            not isinstance(members, list)
            or not isinstance(selection, dict)
            or digest_model(members) != membership_digest
            or digest_model(
                {
                    "corpus_commit_id": corpus_commit_id,
                    "selection": selection,
                    "members": members,
                }
            )
            != run_set_id
        ):
            return False
        included_run_ids = [
            member.get("run_id")
            for member in members
            if isinstance(member, dict) and member.get("included") is True
        ]
        return included_run_ids == [run_id]

    @staticmethod
    def _limitations(
        request: EvidenceSummaryRequest,
        runs: tuple[SummaryRun, ...],
        references: tuple[SummaryReference, ...],
        claims: tuple[SummaryClaim, ...],
        proof_assessment: SummaryProofAssessment,
    ) -> tuple[str, ...]:
        limitations = [limitation for run in runs for limitation in run.limitations]
        limitations.extend(limitation for claim in claims for limitation in claim.limitations)
        if (
            request.candidate_run_id is not None
            and request.baseline_run_id is None
            and proof_assessment.shape is SummaryProofShape.CANDIDATE_VALIDATED
        ):
            limitations.append(
                "Candidate validation alone does not establish behavior relative to a baseline."
            )
        if (
            request.baseline_run_id is not None
            and request.candidate_run_id is not None
            and len(
                {
                    run.environment.get("environment_id") or run.environment.get("identity")
                    for run in runs[:2]
                }
            )
            > 1
        ):
            limitations.append("Baseline and candidate environment identities differ.")
        for reference in references:
            if (
                reference.ref_type is SummaryReferenceKind.COMPARISON
                and reference.data.get("validity") != "valid"
            ):
                limitations.append(
                    f"Comparison {reference.ref_id} is "
                    f"{reference.data.get('validity', 'unknown')} and is not supporting proof."
                )
        if proof_assessment.missing_conditions:
            limitations.append(
                "Proof remains incomplete; missing conditions: "
                + ", ".join(proof_assessment.missing_conditions)
                + "."
            )
        return tuple(dict.fromkeys(limitations))

    def _excerpt(self, path: Path) -> tuple[tuple[str, ...], bool]:
        selected: list[str] = []
        truncated = False
        with path.open(encoding="utf-8", errors="replace") as stream:
            for _ in range(20):
                line = stream.readline(201)
                if not line:
                    break
                if len(line) == 201 and not line.endswith("\n"):
                    truncated = True
                    _discard_rest_of_line(stream)
                selected.append(_CONTROL.sub("", _ANSI_ESCAPE.sub("", line.rstrip("\r\n"))))
            if stream.readline(1):
                truncated = True
        return tuple(selected), truncated

    @staticmethod
    def _compact_reference(
        ref_type: SummaryReferenceKind,
        data: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        fields = (
            (
                "comparison_id",
                "baseline_run_set_id",
                "candidate_run_set_id",
                "metric",
                "unit",
                "estimand",
                "decision",
                "validity",
                "mismatches",
                "complete_pair_n",
                "corpus_commit_id",
            )
            if ref_type is SummaryReferenceKind.COMPARISON
            else (
                "analysis_id",
                "recipe",
                "recipe_version",
                "corpus_commit_id",
                "input_run_ids",
                "input_artifact_ids",
                "result_digest",
                "coverage_json",
                "limitations",
            )
        )
        return {field: data[field] for field in fields if field in data}


def render_evidence_summary_markdown(summary: EvidenceSummary) -> str:
    lines = [
        "# Flameox evidence summary",
        "",
        f"- Summary digest: `{summary.summary_digest}`",
        f"- Corpus commit: `{summary.corpus_commit_id}`",
        f"- Selection shape: `{summary.selection_shape}`",
        f"- Proof shape: `{summary.proof_shape}`",
    ]
    if summary.proof_assessment.binding_id is not None:
        lines.append(
            f"- Proof binding: `{summary.proof_assessment.binding_type}:"
            f"{summary.proof_assessment.binding_id}`"
        )
    if summary.proof_assessment.missing_conditions:
        lines.append(
            "- Missing proof conditions: `"
            + ", ".join(summary.proof_assessment.missing_conditions)
            + "`"
        )
    for run in summary.runs:
        lines.extend(
            (
                "",
                f"## Claimed {_inline(run.claimed_role.title())} run",
                "",
                f"- Run: `{run.run_id}`",
                f"- Execution: `{run.execution_status}`",
                f"- Validation: `{run.validation_status}`",
                "- Resolved argv:",
                "",
                _fenced(json.dumps(run.argv, ensure_ascii=False)),
                "",
                "- Source identity:",
                "",
                _fenced(json.dumps(run.source, ensure_ascii=False, sort_keys=True)),
                "",
                "- Environment identity:",
                "",
                _fenced(json.dumps(run.environment, ensure_ascii=False, sort_keys=True)),
            )
        )
        for artifact in run.artifacts:
            lines.append(
                f"- Artifact `{artifact.artifact_id}`: `{artifact.kind}` / "
                f"`{_inline(artifact.role)}` ({artifact.sensitivity})"
            )
            if artifact.excerpt:
                lines.extend(("", _fenced("\n".join(artifact.excerpt))))
    for claim in summary.claims:
        lines.extend(
            (
                "",
                f"## Finding: {_inline(claim.title)}",
                "",
                f"- Evidence level: `{claim.evidence_level}`",
                f"- Assessment: `{claim.assessment}`",
                f"- Support in this selection: `{claim.support_status}`",
                "",
                _fenced(claim.claim),
            )
        )
    if summary.limitations:
        lines.extend(("", "## Limitations", ""))
        lines.extend(f"- {_inline(value)}" for value in summary.limitations)
    if summary.truncation:
        lines.extend(("", "## Truncation", ""))
        lines.extend(f"- `{_inline(value)}`" for value in summary.truncation)
    return "\n".join(lines) + "\n"


def _inline(value: str) -> str:
    cleaned = html.escape(_CONTROL.sub("", _ANSI_ESCAPE.sub("", value)))
    return re.sub(r"([\\`*_{}\[\]()#+.!|>-])", r"\\\1", cleaned)


def _fenced(value: str) -> str:
    cleaned = _CONTROL.sub("", _ANSI_ESCAPE.sub("", value))
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", cleaned)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{cleaned}\n{fence}"
