from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from flameox.analysis import RecipeService
from flameox.application.analysis_records import (
    AnalysisMaterializationService,
    ScalingAnalysisRequest,
)
from flameox.application.capture import CaptureService
from flameox.application.execution_policy import ExecutionPolicy
from flameox.application.experiments import ExperimentService
from flameox.application.records import (
    CreateInvestigationRequest,
    EvidenceInput,
    FindingService,
    InvestigationService,
    RecordFindingRequest,
    RecordHypothesisRequest,
)
from flameox.domain import (
    ComparisonDecision,
    ComparisonValidity,
    EvidenceLevel,
    EvidenceReferenceType,
    EvidenceRelation,
    ExecutionStatus,
    FindingAssessment,
    FindingConfidence,
    ValidationStatus,
)
from flameox.storage import Workspace

pytestmark = [pytest.mark.integration, pytest.mark.serial]


def git(project: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=project,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.mark.anyio
async def test_reverse_scan_investigation_proves_candidate_with_oracle(
    tmp_path: Path,
) -> None:
    (tmp_path / "scan.py").write_text(
        "import sys\n"
        "mode, length = sys.argv[1], int(sys.argv[2])\n"
        "values = list(range(length))\n"
        "if mode == 'baseline':\n"
        "    for _ in range(50):\n"
        "        total = 0\n"
        "        for value in reversed(values):\n"
        "            total += value\n"
        "elif mode == 'candidate':\n"
        "    for _ in range(50):\n"
        "        total = sum(values)\n"
        "else:\n"
        "    total = -1\n"
        "print(total)\n"
    )
    (tmp_path / "validate.py").write_text(
        "import hashlib, json, os, subprocess, sys\n"
        "mode, length = sys.argv[1], int(sys.argv[2])\n"
        "actual = int(subprocess.check_output("
        "[sys.executable, 'scan.py', mode, str(length)], text=True))\n"
        "expected = length * (length - 1) // 2\n"
        "if actual != expected:\n"
        "    raise SystemExit(f'{actual} != {expected}')\n"
        "outputs = {\n"
        "    treatment: int(subprocess.check_output(\n"
        "        [sys.executable, 'scan.py', treatment, str(length)], text=True))\n"
        "    for treatment in ('baseline', 'candidate')\n"
        "}\n"
        "def digest(value):\n"
        "    return 'sha256:' + hashlib.sha256(str(value).encode()).hexdigest()\n"
        "payload = {\n"
        "    'schema_version': 'flameox.oracle-receipt.v1',\n"
        "    'status': 'pass', 'reason': 'pair_match',\n"
        "    'binding': {\n"
        "        'pair_id': digest(f'pair:{length}'),\n"
        "        'treatment': mode,\n"
        "        'input_identity': digest(f'input:{length}'),\n"
        "        'workload_identity': os.environ['FLAMEOX_WORKLOAD_INSTANCE_ID'],\n"
        "        'output_identity': digest(outputs.get(mode, actual)),\n"
        "        'compared_property': 'forward',\n"
        "        'oracle_identity': digest('reverse-scan-pair-oracle'),\n"
        "        'tolerance': {'absolute': 0, 'relative': 0, 'equal_nan': False},\n"
        "    },\n"
        "}\n"
        "with open(os.environ['FLAMEOX_ORACLE_RECEIPT'], 'w') as stream:\n"
        "    json.dump(payload, stream)\n"
    )
    (tmp_path / "flameox.toml").write_text(
        """
[workloads.reverse_scan]
argv = ["python", "scan.py", "{mode}", "{length}"]
cwd = "."
timeout_seconds = 30

[workloads.reverse_scan.parameters]
mode = ["baseline", "candidate", "broken"]
length = [32768, 65536, 131072]

[workloads.reverse_scan.oracle]
strength = "cross_treatment_equivalence"
argv = ["python", "validate.py", "{mode}", "{length}"]
receipt_schema = "flameox.oracle-receipt.v1"

[experiments.reverse_scan_scaling]
workload = "reverse_scan"
treatment_factor = "mode"
design = "randomized_complete_blocks"
blocks = 1
primary_metric = "pyperf.workload"
polarity = "lower_is_better"
estimand = "median_paired_log_ratio"
practical_threshold = 0.05
confidence_level = 0.95
random_seed = 1984
[experiments.reverse_scan_scaling.factors]
mode = ["baseline", "candidate"]
length = [32768, 65536, 131072]
"""
    )
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "add", "scan.py", "validate.py", "flameox.toml")
    git(tmp_path, "commit", "-m", "golden fixture")
    workspace = Workspace.initialize(tmp_path)
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
            question="Does the Python reverse scan dominate long sequences?",
            symptom="Runtime grows with sequence length.",
        )
    )
    hypothesis = investigations.record_hypothesis(
        RecordHypothesisRequest(
            investigation_id=investigation.investigation_id,
            claim="The Python reverse loop is avoidable interpreter overhead.",
            prediction="Replacing it with a native reduction materially lowers time.",
            discriminating_condition=(
                "Outputs remain equal while fresh paired measurements improve."
            ),
        )
    )
    experiment_service = ExperimentService(workspace)
    experiment_plan = await experiment_service.plan(
        experiment_name="reverse_scan_scaling",
        investigation_id=investigation.investigation_id,
        hypothesis_id=hypothesis.hypothesis_id,
        adapter="pyperf",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    experiment = await experiment_service.run(experiment_plan.plan_token)
    assert len(experiment.trials) == 6
    assert all(trial.validation_status is ValidationStatus.PASSED for trial in experiment.trials)
    scaling = RecipeService(workspace).scaling(experiment.experiment.experiment_id)
    scaling_record = AnalysisMaterializationService(workspace).record(
        ScalingAnalysisRequest(
            recipe="scaling",
            experiment_id=experiment.experiment.experiment_id,
        )
    )
    assert experiment.comparison is not None
    comparison = experiment.comparison
    assert comparison.comparison.validity is ComparisonValidity.VALID, (
        comparison.comparison.mismatches
    )
    assert comparison.comparison.decision is ComparisonDecision.MEANINGFUL_IMPROVEMENT, (
        comparison.comparison.model_dump(mode="json")
    )

    capture = CaptureService(workspace)
    broken_plan = await capture.plan(
        workload_name="reverse_scan",
        adapter="pyperf",
        parameters={"mode": "broken", "length": 32768},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    broken = await capture.execute(broken_plan.plan_token)
    assert broken.run.execution_status is ExecutionStatus.SUCCEEDED
    assert broken.run.validation_status is ValidationStatus.FAILED
    finding = FindingService(workspace).record(
        RecordFindingRequest(
            kind="performance",
            title="Native reduction removes reverse-scan interpreter overhead",
            claim="The candidate is materially faster with equivalent output.",
            evidence_level=EvidenceLevel.DERIVED,
            confidence=FindingConfidence.HIGH,
            assessment=FindingAssessment.SUPPORTED,
            evidence=(
                EvidenceInput(
                    ref_type=EvidenceReferenceType.COMPARISON,
                    ref_id=comparison.comparison.comparison_id,
                    relation=EvidenceRelation.SUPPORTS,
                ),
                EvidenceInput(
                    ref_type=EvidenceReferenceType.ANALYSIS,
                    ref_id=scaling_record.analysis.analysis_id,
                    relation=EvidenceRelation.CONTEXT,
                ),
            ),
        )
    )

    assert scaling.attempted_trials == 6
    assert {point.input_value for point in scaling.points} == {
        32768.0,
        65536.0,
        131072.0,
    }
    assert {fit.variant for fit in scaling.fits} == {"baseline", "candidate"}
    assert scaling_record.result == scaling
    assert finding.finding.assessment is FindingAssessment.SUPPORTED
