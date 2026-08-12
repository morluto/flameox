from __future__ import annotations

from pathlib import Path

import pytest

from flameox.adapters import ObservationExtractor
from flameox.analysis import RecipeService
from flameox.application import (
    AnalysisMaterializationService,
    CaptureService,
    ComparisonService,
    CreateInvestigationRequest,
    EvidenceInput,
    ExecutionAnalysisRequest,
    ExecutionPolicy,
    FindingService,
    FreezeRunIdsRequest,
    InvestigationService,
    MeasurementCompareRunSetsRequest,
    RecordFindingRequest,
    RecordHypothesisRequest,
    RunSetService,
)
from flameox.domain import (
    EvidenceLevel,
    EvidenceReferenceType,
    EvidenceRelation,
    FindingAssessment,
    FindingConfidence,
    MetricPolarity,
)
from flameox.storage import Workspace


@pytest.mark.anyio
async def test_configuration_interaction_is_visible_as_semantic_evidence(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    (tmp_path / "policy.py").write_text(
        "import sys\n"
        "from flameox.sdk import observe, phase\n"
        "mode = sys.argv[1]\n"
        "with phase('ppo_epoch'):\n"
        "    observe('policy.old_log_prob_source', "
        "source='current_policy' if mode == 'bad' else 'frozen_policy')\n"
        "    observe('policy.clipping_enabled', value=mode != 'bad')\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
schema_version = 1
[workloads.policy]
argv = ["python", "policy.py", "{mode}"]
cwd = "."
timeout_seconds = 10

[workloads.policy.parameters]
mode = ["bad", "fixed"]
"""
    )
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    investigations = InvestigationService(workspace)
    investigation = investigations.create(
        CreateInvestigationRequest(
            question="Does the configuration disable the clipping safeguard?"
        )
    )
    hypothesis = investigations.record_hypothesis(
        RecordHypothesisRequest(
            investigation_id=investigation.investigation_id,
            claim="Recomputing old log probabilities disables PPO clipping.",
            prediction="The bad mode reports current-policy probabilities and clipping disabled.",
            discriminating_condition=(
                "The fixed mode reports frozen probabilities and clipping enabled."
            ),
        )
    )
    capture = CaptureService(workspace)
    runs = {}
    analyses = {}
    for mode in ("bad", "fixed"):
        plan = await capture.plan(
            workload_name="policy",
            adapter="command",
            parameters={"mode": mode},
            execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
        )
        result = await capture.execute(plan.plan_token)
        extracted = ObservationExtractor(workspace).extract(result.run.run_id)
        assert extracted.observation_count == 4
        runs[mode] = result.run
        analyses[mode] = RecipeService(workspace).execution(result.run.run_id)

    by_name = {item.name: item for item in analyses["bad"].observations}
    assert by_name["policy.old_log_prob_source"].value_json == ('{"source":"current_policy"}')
    assert by_name["policy.clipping_enabled"].value_json == '{"value":false}'
    assert by_name["policy.clipping_enabled"].context == "ppo_epoch"
    fixed_by_name = {item.name: item for item in analyses["fixed"].observations}
    assert fixed_by_name["policy.clipping_enabled"].value_json == '{"value":true}'
    recorded_execution = AnalysisMaterializationService(workspace).record(
        ExecutionAnalysisRequest(
            recipe="execution",
            input_id=runs["bad"].run_id,
        )
    )

    cohorts = RunSetService(workspace)
    baseline = cohorts.freeze(FreezeRunIdsRequest(run_ids=(runs["bad"].run_id,)))
    candidate = cohorts.freeze(FreezeRunIdsRequest(run_ids=(runs["fixed"].run_id,)))
    comparison = ComparisonService(workspace).record(
        MeasurementCompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate.run_set_id,
            metric="process.wall_time",
            unit="ns",
            polarity=MetricPolarity.NEUTRAL,
            practical_threshold=0,
        )
    )
    finding = FindingService(workspace).record(
        RecordFindingRequest(
            kind="correctness",
            title="Configuration selects current-policy probabilities",
            claim="The bad mode explicitly reports clipping disabled.",
            evidence_level=EvidenceLevel.OBSERVED,
            confidence=FindingConfidence.HIGH,
            assessment=FindingAssessment.SUPPORTED,
            evidence=(
                EvidenceInput(
                    ref_type=EvidenceReferenceType.OBSERVATION,
                    ref_id=by_name["policy.clipping_enabled"].observation_id,
                    relation=EvidenceRelation.SUPPORTS,
                ),
            ),
        )
    )

    assert hypothesis.hypothesis_id
    assert recorded_execution.evidence
    assert comparison.analysis is not None
    assert finding.finding.assessment is FindingAssessment.SUPPORTED
