from __future__ import annotations

import html
import json
import re
from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, model_validator

from flameox.application.evidence_lookup import EvidenceLookupService
from flameox.catalog import Catalog, Snapshot
from flameox.domain import (
    ArtifactKind,
    Finding,
    FindingAssessment,
    LimitationDetail,
    Sensitivity,
    digest_model,
)
from flameox.models import ContractModel
from flameox.storage import ArtifactStore, JsonRecordStore, RunStore, Workspace

_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


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
    return _unique_limitation_details(
        [item for run in runs for item in run.limitation_details]
    )


class EvidenceSummaryRequest(ContractModel):
    baseline_run_id: str | None = None
    candidate_run_id: str | None = None
    run_ids: Annotated[tuple[str, ...], Field(max_length=20)] = ()
    comparison_ids: Annotated[tuple[str, ...], Field(max_length=10)] = ()
    analysis_ids: Annotated[tuple[str, ...], Field(max_length=20)] = ()
    finding_ids: Annotated[tuple[str, ...], Field(max_length=20)] = ()
    output_excerpts: Literal["none", "internal"] = "none"
    sensitive_context: Literal["redact", "include"] = "redact"

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
    kind: str
    role: str
    sensitivity: str
    producer: str | None = None
    producer_version: str | None = None
    excerpt: tuple[str, ...] = ()
    excerpt_truncated: bool = False


class SummaryAttempt(ContractModel):
    trial_id: str
    outcome: str
    failure_class: str
    exclusion_reason: str | None = None
    combination_id: str
    factors: dict[str, JsonValue] = Field(default_factory=dict)


class SummaryRun(ContractModel):
    run_id: str
    proof_role: Literal["baseline", "candidate", "context"]
    execution_status: str
    validation_status: str
    workload_definition_id: str | None
    workload_instance_id: str | None
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
    ref_type: Literal["comparison", "analysis"]
    ref_id: str
    data: dict[str, JsonValue]


class SummaryClaim(ContractModel):
    finding_id: str
    title: str
    claim: str
    evidence_level: str
    assessment: str
    support_status: Literal["as_recorded", "not_supporting", "candidate_only"]
    evidence: tuple[dict[str, JsonValue], ...]
    limitations: tuple[str, ...] = ()


class EvidenceSummary(ContractModel):
    schema_version: Literal[1] = 1
    summary_digest: str
    corpus_commit_id: str
    proof_shape: Literal[
        "paired_validation",
        "candidate_only_validation",
        "baseline_only_observation",
        "selected_evidence",
    ]
    runs: tuple[SummaryRun, ...]
    references: tuple[SummaryReference, ...]
    claims: tuple[SummaryClaim, ...]
    limitations: tuple[str, ...]
    limitation_details: tuple[LimitationDetail, ...] = ()
    truncation: tuple[str, ...] = ()


class EvidenceSummaryBundle(ContractModel):
    schema_version: Literal[1] = 1
    summary: EvidenceSummary
    markdown: str


class EvidenceSummaryService:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs = RunStore(workspace)
        self.artifacts = ArtifactStore(workspace)
        self.lookup = EvidenceLookupService(workspace)
        self.findings = JsonRecordStore(
            workspace,
            kind="findings",
            model=Finding,
            id_field="finding_id",
            revision_field="revision",
        )

    def summarize(self, request: EvidenceSummaryRequest) -> EvidenceSummaryBundle:
        head = self.workspace.corpus.read_head()
        selected_runs = self._selected_runs(request)
        truncation: list[str] = []
        with Catalog(self.workspace).open_snapshot(head.commit_id) as snapshot:
            runs = tuple(
                self._run_summary(
                    snapshot,
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
                        ref_type="comparison",
                        ref_id=ref_id,
                        data=self._compact_reference(
                            "comparison",
                            self.lookup.get("comparison", ref_id).data,
                        ),
                    )
                    for ref_id in request.comparison_ids
                ]
                + [
                    SummaryReference(
                        ref_type="analysis",
                        ref_id=ref_id,
                        data=self._compact_reference(
                            "analysis",
                            self.lookup.get("analysis", ref_id).data,
                        ),
                    )
                    for ref_id in request.analysis_ids
                ]
            )
            claims = tuple(
                self._claim(snapshot, finding_id, request) for finding_id in request.finding_ids
            )
        limitations = self._limitations(request, runs, references, claims)
        limitation_details = _summary_limitation_details(runs)
        payload = {
            "corpus_commit_id": head.commit_id,
            "proof_shape": self._proof_shape(request),
            "runs": [run.model_dump(mode="json") for run in runs],
            "references": [item.model_dump(mode="json") for item in references],
            "claims": [claim.model_dump(mode="json") for claim in claims],
            "limitations": limitations,
            "limitation_details": [item.model_dump(mode="json") for item in limitation_details],
            "truncation": truncation,
        }
        summary = EvidenceSummary(
            summary_digest=digest_model(payload),
            corpus_commit_id=head.commit_id,
            proof_shape=self._proof_shape(request),
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
    ) -> tuple[tuple[str, Literal["baseline", "candidate", "context"]], ...]:
        values: list[tuple[str, Literal["baseline", "candidate", "context"]]] = []
        if request.baseline_run_id is not None:
            values.append((request.baseline_run_id, "baseline"))
        if request.candidate_run_id is not None:
            values.append((request.candidate_run_id, "candidate"))
        values.extend((run_id, "context") for run_id in request.run_ids)
        return tuple(values)

    def _run_summary(
        self,
        snapshot: Snapshot,
        run_id: str,
        role: Literal["baseline", "candidate", "context"],
        *,
        request: EvidenceSummaryRequest,
        truncation: list[str],
    ) -> SummaryRun:
        run = self.runs.read(run_id)
        source = self._identity_row(
            snapshot,
            "source_states",
            "source_state_id",
            run.source_state_id,
        )
        environment = self._identity_row(
            snapshot,
            "environments",
            "environment_id",
            run.environment_id,
        )
        artifacts: list[SummaryArtifact] = []
        excerpted = 0
        for registration in run.artifacts[:50]:
            excerpt: tuple[str, ...] = ()
            excerpt_truncated = False
            if (
                request.output_excerpts == "internal"
                and registration.kind
                in {ArtifactKind.PROCESS_OUTPUT, ArtifactKind.VALIDATION_OUTPUT}
                and registration.sensitivity is not Sensitivity.SENSITIVE
            ):
                if excerpted < 4:
                    excerpt, excerpt_truncated = self._excerpt(registration.artifact_id)
                    excerpted += 1
                else:
                    truncation.append(f"run:{run_id}:output_artifacts")
            artifacts.append(
                SummaryArtifact(
                    artifact_id=registration.artifact_id,
                    kind=registration.kind.value,
                    role=registration.role,
                    sensitivity=registration.sensitivity.value,
                    producer=registration.producer,
                    producer_version=registration.producer_version,
                    excerpt=excerpt,
                    excerpt_truncated=excerpt_truncated,
                )
            )
        if len(run.artifacts) > 50:
            truncation.append(f"run:{run_id}:artifacts")
        attempts, attempts_truncated = self._attempts(snapshot, run_id)
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
            and run.external_context.sensitivity == "sensitive"
            and request.sensitive_context == "redact"
        ):
            for field in ("lease_id", "worker_id", "orchestration_run_id"):
                external_context[field] = "[redacted]"
        argv = run.command.argv if run.command is not None else ()
        run_limitations = list(run.limitations)
        run_limitation_details = list(run.limitation_details)
        if len(argv) > 1 and request.sensitive_context == "redact":
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
                    source="validation",
                    code="sensitive_context_redacted",
                    message=run_limitations[-1],
                )
            )
        return SummaryRun(
            run_id=run_id,
            proof_role=role,
            execution_status=run.execution_status.value,
            validation_status=run.validation_status.value,
            workload_definition_id=run.workload_definition_id,
            workload_instance_id=run.workload_instance_id,
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
        snapshot: Snapshot,
        table: str,
        identifier: str,
        value: str | None,
    ) -> dict[str, JsonValue]:
        if value is None:
            return {"identity": None, "identity_quality": "unknown"}
        connection = snapshot.execute(
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
        snapshot: Snapshot,
        run_id: str,
    ) -> tuple[tuple[SummaryAttempt, ...], bool]:
        rows = snapshot.execute(
            "SELECT trial_id, outcome, failure_class, exclusion_reason, combination_id, "
            "factors_json FROM trials WHERE run_id = ? "
            "ORDER BY published_at DESC LIMIT 21",
            (run_id,),
        ).fetchall()
        return (
            tuple(
                SummaryAttempt(
                    trial_id=str(row[0]),
                    outcome=str(row[1]),
                    failure_class=str(row[2] or "unknown"),
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
        snapshot: Snapshot,
        finding_id: str,
        request: EvidenceSummaryRequest,
    ) -> SummaryClaim:
        finding = self.findings.read(finding_id)
        rows = snapshot.execute(
            "SELECT ref_type, ref_id, relation FROM evidence_references "
            "WHERE owner_type = 'finding' AND owner_id = ? "
            "ORDER BY ref_type, ref_id LIMIT 51",
            (finding_id,),
        ).fetchall()
        evidence = tuple(
            {
                "ref_type": cast(JsonValue, str(row[0])),
                "ref_id": cast(JsonValue, str(row[1])),
                "relation": cast(JsonValue, str(row[2])),
            }
            for row in rows[:50]
        )
        invalid_comparison = False
        for item in evidence:
            if item["ref_type"] == "comparison":
                comparison = self.lookup.get(
                    "comparison",
                    cast(str, item["ref_id"]),
                ).data
                invalid_comparison = invalid_comparison or comparison.get("validity") != "valid"
        candidate_only = request.candidate_run_id is not None and request.baseline_run_id is None
        status: Literal["as_recorded", "not_supporting", "candidate_only"]
        limitations = list(finding.limitations)
        if len(rows) > 50:
            limitations.append("Finding evidence references were truncated at 50 items.")
        if invalid_comparison or finding.assessment is not FindingAssessment.SUPPORTED:
            status = "not_supporting"
        elif candidate_only:
            status = "candidate_only"
            limitations.append(
                "Candidate-only validation does not establish that the behavior regressed "
                "on the base revision."
            )
        else:
            status = "as_recorded"
        return SummaryClaim(
            finding_id=finding.finding_id,
            title=finding.title,
            claim=finding.claim,
            evidence_level=finding.evidence_level.value,
            assessment=finding.assessment.value,
            support_status=status,
            evidence=evidence,
            limitations=tuple(limitations),
        )

    @staticmethod
    def _proof_shape(
        request: EvidenceSummaryRequest,
    ) -> Literal[
        "paired_validation",
        "candidate_only_validation",
        "baseline_only_observation",
        "selected_evidence",
    ]:
        if request.baseline_run_id is not None and request.candidate_run_id is not None:
            return "paired_validation"
        if request.candidate_run_id is not None:
            return "candidate_only_validation"
        if request.baseline_run_id is not None:
            return "baseline_only_observation"
        return "selected_evidence"

    @staticmethod
    def _limitations(
        request: EvidenceSummaryRequest,
        runs: tuple[SummaryRun, ...],
        references: tuple[SummaryReference, ...],
        claims: tuple[SummaryClaim, ...],
    ) -> tuple[str, ...]:
        limitations = [limitation for run in runs for limitation in run.limitations]
        limitations.extend(limitation for claim in claims for limitation in claim.limitations)
        if request.candidate_run_id is not None and request.baseline_run_id is None:
            limitations.append(
                "This selection contains candidate validation without a base observation."
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
            if reference.ref_type == "comparison" and reference.data.get("validity") != "valid":
                limitations.append(
                    f"Comparison {reference.ref_id} is "
                    f"{reference.data.get('validity', 'unknown')} and is not supporting proof."
                )
        return tuple(dict.fromkeys(limitations))

    def _excerpt(self, artifact_id: str) -> tuple[tuple[str, ...], bool]:
        path = self.artifacts.get(artifact_id).payload_path
        selected: list[str] = []
        truncated = False
        with path.open(encoding="utf-8", errors="replace") as stream:
            for _ in range(20):
                line = stream.readline(201)
                if not line:
                    break
                if len(line) == 201 and not line.endswith("\n"):
                    truncated = True
                    stream.readline()
                selected.append(_CONTROL.sub("", _ANSI_ESCAPE.sub("", line.rstrip("\r\n"))))
            if stream.readline(1):
                truncated = True
        return tuple(selected), truncated

    @staticmethod
    def _compact_reference(
        ref_type: Literal["comparison", "analysis"],
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
            if ref_type == "comparison"
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
        f"- Proof shape: `{summary.proof_shape}`",
    ]
    for run in summary.runs:
        lines.extend(
            (
                "",
                f"## {_inline(run.proof_role.title())} run",
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
