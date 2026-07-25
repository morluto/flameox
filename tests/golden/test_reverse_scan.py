from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from flamo.adapters import PyPerfExtractor
from flamo.application import (
    CaptureService,
    CompareRunSetsRequest,
    ComparisonService,
    CreateInvestigationRequest,
    EvidenceInput,
    ExecutionPolicy,
    FindingService,
    FreezeRunSetRequest,
    InvestigationService,
    RecordFindingRequest,
    RecordHypothesisRequest,
    RunSetService,
)
from flamo.domain import (
    ComparisonDecision,
    ComparisonValidity,
    EvidenceLevel,
    FindingAssessment,
    ValidationStatus,
)
from flamo.storage import Workspace


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
        "assert total == length * (length - 1) // 2\n"
    )
    (tmp_path / "validate.py").write_text(
        "import sys\n"
        "_mode, length = sys.argv[1], int(sys.argv[2])\n"
        "print(length * (length - 1) // 2)\n"
    )
    (tmp_path / "flamo.toml").write_text(
        """
schema_version = 1
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
"""
    )
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "add", "scan.py", "validate.py", "flamo.toml")
    git(tmp_path, "commit", "-m", "golden fixture")
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.model_copy(
        update={
            "execution": workspace.config.execution.model_copy(update={"containment": "disabled"})
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
    investigations.record_hypothesis(
        RecordHypothesisRequest(
            investigation_id=investigation.investigation_id,
            claim="The Python reverse loop is avoidable interpreter overhead.",
            prediction="Replacing it with a native reduction materially lowers time.",
            discriminating_condition=(
                "Outputs remain equal while fresh paired measurements improve."
            ),
        )
    )
    capture = CaptureService(workspace)
    run_ids: dict[tuple[str, int], str] = {}
    for length in (32768, 65536, 131072):
        for mode in ("baseline", "candidate"):
            plan = await capture.plan(
                workload_name="reverse_scan",
                adapter="pyperf",
                parameters={"mode": mode, "length": length},
                execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
            )
            result = await capture.execute(plan.plan_id)
            assert result.run.validation_status is ValidationStatus.PASSED
            PyPerfExtractor(workspace).extract(result.run.run_id)
            run_ids[(mode, length)] = result.run.run_id
    broken_plan = await capture.plan(
        workload_name="reverse_scan",
        adapter="pyperf",
        parameters={"mode": "broken", "length": 32768},
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )
    broken = await capture.execute(broken_plan.plan_id)
    assert broken.run.validation_status is not ValidationStatus.PASSED
    cohorts = RunSetService(workspace)
    baseline = cohorts.freeze(
        FreezeRunSetRequest(run_ids=(run_ids[("baseline", 32768)],))
    )
    candidate = cohorts.freeze(
        FreezeRunSetRequest(run_ids=(run_ids[("candidate", 32768)],))
    )
    comparison = ComparisonService(workspace).record(
        CompareRunSetsRequest(
            baseline_run_set_id=baseline.run_set_id,
            candidate_run_set_id=candidate.run_set_id,
            metric="pyperf.workload",
            unit="ns",
            polarity="lower_is_better",
            practical_threshold=0.05,
            random_seed=1984,
        )
    )
    finding = FindingService(workspace).record(
        RecordFindingRequest(
            kind="performance",
            title="Native reduction removes reverse-scan interpreter overhead",
            claim="The candidate is materially faster with equivalent output.",
            evidence_level=EvidenceLevel.DERIVED,
            confidence="high",
            assessment=FindingAssessment.SUPPORTED,
            evidence=(
                EvidenceInput(
                    ref_type="comparison",
                    ref_id=comparison.comparison.comparison_id,
                    relation="supports",
                ),
            ),
        )
    )

    assert comparison.comparison.validity is ComparisonValidity.VALID
    assert comparison.comparison.decision is ComparisonDecision.MEANINGFUL_IMPROVEMENT
    assert finding.finding.assessment is FindingAssessment.SUPPORTED
