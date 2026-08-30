from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import JsonValue

from flameox.application.capture import CaptureService
from flameox.application.evidence_lookup import EvidenceLookupService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.run_projection import RunProjectionService
from flameox.application.run_rows import run_row
from flameox.application.summaries import (
    EvidenceSummary,
    EvidenceSummaryRequest,
    EvidenceSummaryService,
    SummaryClaim,
    SummaryExcerptPolicy,
    SummaryProofAssessment,
    SummaryProofShape,
    SummarySelectionShape,
    SummarySensitiveContextPolicy,
    SummarySupportStatus,
    render_evidence_summary_markdown,
)
from flameox.catalog import Catalog
from flameox.domain import (
    CaptureStatus,
    EvidenceLevel,
    EvidenceReferenceType,
    ExecutionStatus,
    ExternalExecutionContext,
    FindingAssessment,
    OracleReceiptRecord,
    OracleReceiptV1,
    OracleStatus,
    RunSemantics,
    Sensitivity,
    ValidationStatus,
    digest_model,
)
from flameox.domain.models import ExecutionRunManifest, IncludedRunSetMember, RunSet
from flameox.evidence import GenerationPublisher
from flameox.storage import RunStore, Workspace
from tests.support.capture import disable_containment

pytestmark = pytest.mark.unit

_ENVIRONMENT_ID = "sha256:" + "1" * 64
_WORKLOAD_DEFINITION_ID = "sha256:" + "2" * 64
_WORKLOAD_INSTANCE_ID = "sha256:" + "3" * 64
_MEASUREMENT_PROTOCOL_ID = "sha256:" + "4" * 64


def _write_workload(project: Path) -> None:
    (project / "flameox.toml").write_text(
        """
[workloads.proof]
argv = ["python", "-c", "print('{message}')"]
cwd = "."
timeout_seconds = 5
[workloads.proof.parameters]
message = ["candidate", "token-secret"]
"""
    )


def _analysis_row(workspace: Workspace, *, result_digest: str) -> dict[str, object]:
    observed_at = datetime.now(UTC)
    return {
        "analysis_id": "analysis-1",
        "recipe": "snapshot-proof",
        "recipe_version": "1",
        "parameters_json": "{}",
        "parameters_digest": "0" * 64,
        "corpus_commit_id": workspace.corpus.read_head().commit_id,
        "input_generation_ids": [],
        "input_run_ids": [],
        "input_artifact_ids": [],
        "result_digest": result_digest,
        "result_artifact_id": None,
        "coverage_json": "{}",
        "limitations": [],
        "started_at": observed_at,
        "completed_at": observed_at,
    }


def _publish_validated_run(
    workspace: Workspace,
    run_id: str,
    *,
    include_receipt: bool = True,
) -> ExecutionRunManifest:
    receipt = (
        OracleReceiptRecord(
            receipt=OracleReceiptV1(
                schema_version="flameox.oracle-receipt.v1",
                status=OracleStatus.PASS,
                reason="semantic_match",
            ),
            receipt_artifact_id="sha256:" + "5" * 64,
        )
        if include_receipt
        else None
    )
    manifest = ExecutionRunManifest(
        run_id=run_id,
        execution_status=ExecutionStatus.SUCCEEDED,
        capture_status=CaptureStatus.REGISTERED,
        validation_status=ValidationStatus.PASSED,
        finished_at=datetime.now(UTC),
        workload_definition_id=_WORKLOAD_DEFINITION_ID,
        workload_instance_id=_WORKLOAD_INSTANCE_ID,
        measurement_protocol_id=_MEASUREMENT_PROTOCOL_ID,
        environment_id=_ENVIRONMENT_ID,
        semantics=RunSemantics.unavailable(origin="internal", adapter=None),
        oracle_receipt=receipt,
    )
    RunStore(workspace).create(manifest)
    GenerationPublisher(workspace).publish_rows(
        {"runs": [run_row(manifest)]},
        publisher="summary-proof-run-test",
        publisher_version="1",
        input_run_ids=(run_id,),
    )
    return manifest


def _publish_comparison_binding(
    workspace: Workspace,
    *,
    baseline_run_id: str,
    candidate_run_id: str,
    comparison_id: str = "comparison-proof",
) -> str:
    corpus_commit_id = workspace.corpus.read_head().commit_id
    created_at = datetime.now(UTC)

    def run_set(run_id: str) -> RunSet:
        member = IncludedRunSetMember(run_id=run_id, order=0)
        members = (member,)
        selection: dict[str, JsonValue] = {"mode": "proof-test"}
        member_payload = [item.model_dump(mode="json") for item in members]
        return RunSet(
            run_set_id=digest_model(
                {
                    "corpus_commit_id": corpus_commit_id,
                    "selection": selection,
                    "members": member_payload,
                }
            ),
            corpus_commit_id=corpus_commit_id,
            created_at=created_at,
            selection=selection,
            members=members,
            membership_digest=digest_model(member_payload),
        )

    baseline = run_set(baseline_run_id)
    candidate = run_set(candidate_run_id)

    def run_set_row(value: RunSet) -> dict[str, object]:
        return {
            "run_set_id": value.run_set_id,
            "corpus_commit_id": value.corpus_commit_id,
            "created_at": value.created_at,
            "selection_json": json.dumps(value.selection, sort_keys=True),
            "members_json": json.dumps(
                [item.model_dump(mode="json") for item in value.members],
                sort_keys=True,
            ),
            "membership_digest": value.membership_digest,
        }

    GenerationPublisher(workspace).publish_rows(
        {
            "run_sets": [run_set_row(baseline), run_set_row(candidate)],
            "comparisons": [
                {
                    "comparison_id": comparison_id,
                    "experiment_id": None,
                    "baseline_run_set_id": baseline.run_set_id,
                    "candidate_run_set_id": candidate.run_set_id,
                    "metric": "latency",
                    "unit": "ns",
                    "metric_source": "measurement",
                    "metric_contract_id": "sha256:" + "6" * 64,
                    "measurement_series_id": None,
                    "protocol_identity_id": "sha256:" + "7" * 64,
                    "value_domain": "strictly_positive",
                    "zero_policy": "reject",
                    "polarity": "lower_is_better",
                    "estimand": "difference_in_median_logs",
                    "practical_threshold": 0.05,
                    "baseline_value_int": 10,
                    "baseline_value_float": None,
                    "candidate_value_int": 5,
                    "candidate_value_float": None,
                    "absolute_change_int": -5,
                    "absolute_change_float": None,
                    "relative_change": -0.5,
                    "effect_size": -0.5,
                    "confidence_low": -0.6,
                    "confidence_high": -0.4,
                    "confidence_level": 0.95,
                    "method": "bootstrap",
                    "random_seed": 1,
                    "independent_unit": "run",
                    "paired": False,
                    "baseline_attempted_n": 1,
                    "baseline_eligible_n": 1,
                    "baseline_failed_n": 0,
                    "baseline_excluded_n": 0,
                    "baseline_missing_n": 0,
                    "baseline_out_of_domain_n": 0,
                    "candidate_attempted_n": 1,
                    "candidate_eligible_n": 1,
                    "candidate_failed_n": 0,
                    "candidate_excluded_n": 0,
                    "candidate_missing_n": 0,
                    "candidate_out_of_domain_n": 0,
                    "complete_pair_n": None,
                    "multiplicity_json": None,
                    "decision": "meaningful_improvement",
                    "validity": "valid",
                    "mismatches": [],
                }
            ],
        },
        publisher="summary-proof-binding-test",
        publisher_version="1",
        input_run_ids=(baseline_run_id, candidate_run_id),
    )
    return comparison_id


def test_summary_resolves_references_inside_its_pinned_snapshot(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    published = GenerationPublisher(workspace).publish_rows(
        {"analyses": [_analysis_row(workspace, result_digest="old-result")]},
        publisher="summary-snapshot-test",
        publisher_version="1",
    )

    result = EvidenceSummaryService(workspace).summarize(
        EvidenceSummaryRequest(analysis_ids=("analysis-1",))
    )

    assert result.summary.corpus_commit_id == published.commit.commit_id
    assert result.summary.references[0].data["result_digest"] == "old-result"


def test_evidence_session_does_not_advance_when_head_changes(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    publisher = GenerationPublisher(workspace)
    first = publisher.publish_rows(
        {"analyses": [_analysis_row(workspace, result_digest="old-result")]},
        publisher="session-snapshot-test",
        publisher_version="1",
    )
    lookup = EvidenceLookupService(workspace)

    with lookup.session() as session:
        second = publisher.publish_rows(
            {"analyses": [_analysis_row(workspace, result_digest="new-result")]},
            publisher="concurrent-session-test",
            publisher_version="1",
        )
        pinned = session.get(EvidenceReferenceType.ANALYSIS, "analysis-1")

    current = lookup.get(EvidenceReferenceType.ANALYSIS, "analysis-1")
    assert pinned.corpus_commit_id == first.commit.commit_id
    assert pinned.data["result_digest"] == "old-result"
    assert current.corpus_commit_id == second.commit.commit_id
    assert current.data["result_digest"] == "new-result"


def test_candidate_validation_requires_a_passing_semantic_receipt(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    _publish_validated_run(workspace, "candidate-spoofed", include_receipt=False)

    spoofed = EvidenceSummaryService(workspace).summarize(
        EvidenceSummaryRequest(candidate_run_id="candidate-spoofed")
    )

    assert spoofed.summary.proof_shape == "incomplete"
    assert "semantic_validation_receipt" in spoofed.summary.proof_assessment.missing_conditions
    assert spoofed.summary.proof_assessment.verified_candidate_run_id is None

    _publish_validated_run(workspace, "candidate-validated")
    validated = EvidenceSummaryService(workspace).summarize(
        EvidenceSummaryRequest(candidate_run_id="candidate-validated")
    )

    assert validated.summary.proof_shape == "candidate_validated"
    assert validated.summary.proof_assessment.binding_type == "run_validation"
    assert validated.summary.proof_assessment.verified_candidate_run_id == "candidate-validated"


def test_two_claimed_runs_are_not_a_validated_pair_without_an_exact_binding(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _publish_validated_run(workspace, "baseline")
    _publish_validated_run(workspace, "candidate")

    result = EvidenceSummaryService(workspace).summarize(
        EvidenceSummaryRequest(
            baseline_run_id="baseline",
            candidate_run_id="candidate",
        )
    )

    assert result.summary.selection_shape == "two_claimed_runs"
    assert result.summary.proof_shape == "incomplete"
    assert "exact_valid_comparison_binding" in result.summary.proof_assessment.missing_conditions
    assert result.summary.proof_assessment.verified_baseline_run_id is None
    assert result.summary.proof_assessment.verified_candidate_run_id is None


def test_validated_pair_requires_comparison_membership_in_the_claimed_roles(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _publish_validated_run(workspace, "baseline")
    _publish_validated_run(workspace, "candidate")
    comparison_id = _publish_comparison_binding(
        workspace,
        baseline_run_id="baseline",
        candidate_run_id="candidate",
    )

    result = EvidenceSummaryService(workspace).summarize(
        EvidenceSummaryRequest(
            baseline_run_id="baseline",
            candidate_run_id="candidate",
            comparison_ids=(comparison_id,),
        )
    )

    assert result.summary.selection_shape == "mixed_evidence"
    assert result.summary.proof_shape == "validated_pair"
    assert result.summary.proof_assessment.binding_id == comparison_id
    assert result.summary.proof_assessment.verified_baseline_run_id == "baseline"
    assert result.summary.proof_assessment.verified_candidate_run_id == "candidate"

    reversed_roles = EvidenceSummaryService(workspace).summarize(
        EvidenceSummaryRequest(
            baseline_run_id="candidate",
            candidate_run_id="baseline",
            comparison_ids=(comparison_id,),
        )
    )
    assert reversed_roles.summary.proof_shape == "incomplete"
    assert (
        "exact_valid_comparison_binding"
        in reversed_roles.summary.proof_assessment.missing_conditions
    )


@pytest.mark.anyio
async def test_summary_is_stable_and_redacts_sensitive_execution_context(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_workload(tmp_path)
    disable_containment(workspace)
    context = ExternalExecutionContext(
        orchestrator="crabbox",
        provider="runpod",
        lease_id="lease-secret",
        worker_id="worker-secret",
        orchestration_run_id="remote-secret",
        sensitivity=Sensitivity.SENSITIVE,
    )
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="proof",
        adapter="command",
        parameters={"message": "candidate"},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        external_context=context,
    )
    captured = await service.execute(plan.plan_token)
    request = EvidenceSummaryRequest(candidate_run_id=captured.run.run_id)

    first = EvidenceSummaryService(workspace).summarize(request)
    second = EvidenceSummaryService(workspace).summarize(request)

    assert first == second
    assert first.summary.selection_shape == "single_run"
    assert first.summary.proof_shape == "incomplete"
    assert "validation_passed" in first.summary.proof_assessment.missing_conditions
    summarized_context = first.summary.runs[0].external_context
    assert summarized_context is not None
    assert summarized_context["provider"] == "runpod"
    assert summarized_context["lease_id"] == "[redacted]"
    assert "lease-secret" not in first.markdown
    assert "candidate" not in first.summary.runs[0].argv
    assert "arguments redacted" in first.summary.runs[0].argv[-1]
    assert first.summary.summary_digest in first.markdown
    assert json.loads(first.summary.model_dump_json())["summary_digest"] in first.markdown
    assert any("Proof remains incomplete" in item for item in first.summary.limitations)

    included = EvidenceSummaryService(workspace).summarize(
        request.validated_copy(update={"sensitive_context": SummarySensitiveContextPolicy.INCLUDE})
    )
    assert included.summary.runs[0].external_context is not None
    assert included.summary.runs[0].external_context["lease_id"] == "lease-secret"
    assert included.summary.runs[0].argv[-1] == "print('candidate')"
    assert included.summary.summary_digest != first.summary.summary_digest

    projection = RunProjectionService(workspace).get(captured.run.run_id)
    projection_json = projection.model_dump_json()
    assert "candidate" not in projection_json
    assert "lease-secret" not in projection_json
    assert "worker-secret" not in projection_json
    assert "remote-secret" not in projection_json
    assert projection.command is not None
    assert projection.command.argument_count == 2
    assert projection.external_context is not None
    assert projection.external_context.lease_id == "[redacted]"


@pytest.mark.anyio
async def test_summary_never_excerpts_sensitive_process_output(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    _write_workload(tmp_path)
    disable_containment(workspace)
    service = CaptureService(workspace)
    plan = await service.plan(
        workload_name="proof",
        adapter="command",
        parameters={"message": "candidate"},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    captured = await service.execute(plan.plan_token)
    store = RunStore(workspace)
    run = store.read(captured.run.run_id)
    registrations = tuple(
        item.validated_copy(update={"sensitivity": Sensitivity.SENSITIVE}) for item in run.artifacts
    )
    sensitive_run = store.append(
        run.validated_copy(update={"revision": run.revision + 1, "artifacts": registrations}),
        expected_revision=run.revision,
    )
    GenerationPublisher(workspace).publish_rows(
        {"runs": [run_row(sensitive_run)]},
        publisher="test.sensitive-run",
        publisher_version="1",
        input_run_ids=(run.run_id,),
    )
    Catalog(workspace).rebuild()

    result = EvidenceSummaryService(workspace).summarize(
        EvidenceSummaryRequest(
            candidate_run_id=run.run_id,
            output_excerpts=SummaryExcerptPolicy.INTERNAL,
        )
    )

    assert all(not artifact.excerpt for artifact in result.summary.runs[0].artifacts)


def test_markdown_renderer_contains_untrusted_text_without_structure_injection() -> None:
    claim = SummaryClaim(
        finding_id="finding",
        title="heading\n# injected <img src=x onerror=alert(1)>",
        claim="`````\n# not a heading\n\x1b[31mred",
        evidence_level=EvidenceLevel.INFERRED,
        assessment=FindingAssessment.INCONCLUSIVE,
        support_status=SummarySupportStatus.NOT_SUPPORTING,
        evidence=(),
        limitations=("unsafe `inline`",),
    )
    summary = EvidenceSummary(
        summary_digest="0" * 64,
        corpus_commit_id="1" * 64,
        selection_shape=SummarySelectionShape.MIXED_EVIDENCE,
        claimed_roles=(),
        proof_shape=SummaryProofShape.SELECTED_EVIDENCE,
        proof_assessment=SummaryProofAssessment(shape=SummaryProofShape.SELECTED_EVIDENCE),
        runs=(),
        references=(),
        claims=(claim,),
        limitations=claim.limitations,
    )

    rendered = render_evidence_summary_markdown(summary)

    assert "\x1b" not in rendered
    assert "\n# injected" not in rendered
    assert "<img" not in rendered
    assert "&lt;img" in rendered
    assert "``````text" in rendered
    assert "unsafe \\`inline\\`" in rendered


def test_summary_request_has_structural_reference_limits() -> None:
    with pytest.raises(ValueError, match="limited to 50 references"):
        EvidenceSummaryRequest(
            baseline_run_id="baseline",
            candidate_run_id="candidate",
            run_ids=tuple(f"run-{index}" for index in range(20)),
            comparison_ids=tuple(f"comparison-{index}" for index in range(10)),
            analysis_ids=tuple(f"analysis-{index}" for index in range(20)),
        )


def test_trial_failure_class_fallback_for_null_column_value() -> None:
    """A NULL failure_class from externally written rows must degrade to NONE.

    Regression: the strict ``TrialFailureClass(str(row[2]))`` conversion raised
    ``ValueError`` when ``row[2]`` was ``None`` (``str(None)`` is ``"None"``,
    which is not a valid enum member). The fix falls back to
    ``TrialFailureClass.NONE``.
    """
    from flameox.domain import TrialFailureClass

    with pytest.raises(ValueError):
        TrialFailureClass(str(None))

    fallback = TrialFailureClass(str(None) if None is not None else TrialFailureClass.NONE)
    assert fallback is TrialFailureClass.NONE


def test_excerpt_discards_overlong_line_in_bounded_chunks() -> None:
    """Overlong lines must not be materialized during excerpt extraction.

    Regression for #291: ``_excerpt()`` called unbounded ``stream.readline()``
    to discard the remainder of a line longer than 200 characters, materializing
    a multi-gigabyte string in memory. The fix uses bounded ``_DISCARD_CHUNK_SIZE``
    chunks instead.
    """
    import io

    from flameox.application.summaries import _discard_rest_of_line

    long_line = "X" * 10_000
    content = f"{long_line}\nshort line\n"
    stream = io.StringIO(content)
    line = stream.readline(201)
    assert len(line) == 201
    assert not line.endswith("\n")
    _discard_rest_of_line(stream)
    next_line = stream.readline()
    assert next_line == "short line\n"
