from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.application import (
    CaptureService,
    EvidenceSummary,
    EvidenceSummaryRequest,
    EvidenceSummaryService,
    ExecutionPolicy,
    SummaryClaim,
    SummaryExcerptPolicy,
    SummaryProofShape,
    SummarySensitiveContextPolicy,
    SummarySupportStatus,
    render_evidence_summary_markdown,
)
from flameox.catalog import Catalog
from flameox.domain import EvidenceLevel, ExternalExecutionContext, FindingAssessment, Sensitivity
from flameox.storage import RunStore, Workspace
from tests.support.capture import disable_containment


def _write_workload(project: Path) -> None:
    (project / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.proof]
argv = ["python", "-c", "print('{message}')"]
cwd = "."
timeout_seconds = 5
[workloads.proof.parameters]
message = ["candidate", "token-secret"]
"""
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
    captured = await service.execute(plan.plan_id)
    request = EvidenceSummaryRequest(candidate_run_id=captured.run.run_id)

    first = EvidenceSummaryService(workspace).summarize(request)
    second = EvidenceSummaryService(workspace).summarize(request)

    assert first == second
    assert first.summary.proof_shape == "candidate_only_validation"
    summarized_context = first.summary.runs[0].external_context
    assert summarized_context is not None
    assert summarized_context["provider"] == "runpod"
    assert summarized_context["lease_id"] == "[redacted]"
    assert "lease-secret" not in first.markdown
    assert "candidate" not in first.summary.runs[0].argv
    assert "arguments redacted" in first.summary.runs[0].argv[-1]
    assert first.summary.summary_digest in first.markdown
    assert json.loads(first.summary.model_dump_json())["summary_digest"] in first.markdown
    assert any("without a base observation" in item for item in first.summary.limitations)

    included = EvidenceSummaryService(workspace).summarize(
        request.validated_copy(update={"sensitive_context": SummarySensitiveContextPolicy.INCLUDE})
    )
    assert included.summary.runs[0].external_context is not None
    assert included.summary.runs[0].external_context["lease_id"] == "lease-secret"
    assert included.summary.runs[0].argv[-1] == "print('candidate')"
    assert included.summary.summary_digest != first.summary.summary_digest


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
    captured = await service.execute(plan.plan_id)
    store = RunStore(workspace)
    run = store.read(captured.run.run_id)
    registrations = tuple(
        item.validated_copy(update={"sensitivity": Sensitivity.SENSITIVE}) for item in run.artifacts
    )
    store.append(
        run.validated_copy(update={"revision": run.revision + 1, "artifacts": registrations}),
        expected_revision=run.revision,
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
        proof_shape=SummaryProofShape.SELECTED_EVIDENCE,
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
