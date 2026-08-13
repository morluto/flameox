from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from flameox.application import (
    CreateInvestigationRequest,
    ExecutionPolicy,
    ExperimentService,
    InvestigationService,
)
from flameox.domain import TrialOutcome, ValidationStatus
from flameox.storage import ArtifactStore, RunStore, Workspace

pytestmark = [pytest.mark.integration, pytest.mark.process, pytest.mark.serial]


@pytest.mark.anyio
async def test_semantic_matrix_preserves_typed_categorical_evidence(tmp_path: Path) -> None:
    example = Path(__file__).parents[2] / "examples" / "semantic-matrix"
    for name in ("flameox.toml", "semantic_workload.py", "semantic_oracle.py"):
        shutil.copyfile(example / name, tmp_path / name)
    workspace = Workspace.initialize(tmp_path)
    config = workspace.config.validated_copy(
        update={
            "execution": workspace.config.execution.validated_copy(
                update={"containment": "disabled"}
            )
        }
    )
    workspace.paths.config.write_text(config.to_toml())
    investigation = InvestigationService(workspace).create(
        CreateInvestigationRequest(question="Does the candidate preserve semantic behavior?")
    )
    service = ExperimentService(workspace)
    plan = await service.plan(
        experiment_name="semantic_matrix",
        investigation_id=investigation.investigation_id,
        adapter="command",
        execution_policy=ExecutionPolicy.TRUSTED_LOCAL,
    )

    result = await service.run(plan.plan_token)

    assert len(result.trials) == 8
    assert {trial.factors["treatment"] for trial in result.trials} == {
        "reference",
        "candidate",
    }
    mismatch = next(
        trial
        for trial in result.trials
        if trial.factors["treatment"] == "candidate" and trial.factors["case"] == "mismatch"
    )
    assert mismatch.outcome is TrialOutcome.ORACLE_FAILED
    assert mismatch.oracle_receipt is not None
    assert mismatch.oracle_receipt.output_field == "forward"
    assert mismatch.oracle_receipt.absolute_error == 0.25
    assert mismatch.oracle_receipt_artifact_id is not None
    raw = ArtifactStore(workspace).get(mismatch.oracle_receipt_artifact_id)
    assert b'"cross_treatment_mismatch"' in raw.payload_path.read_bytes()
    expected_rejections = [
        trial for trial in result.trials if trial.factors["case"] == "expected_rejection"
    ]
    assert len(expected_rejections) == 2
    assert all(trial.outcome is TrialOutcome.SUCCEEDED for trial in expected_rejections)
    assert all(
        trial.oracle_receipt is not None and trial.oracle_receipt.reason == "expected_rejection"
        for trial in expected_rejections
    )
    unsupported = [trial for trial in result.trials if trial.factors["backend"] == "unavailable"]
    assert len(unsupported) == 2
    assert all(trial.validation_status is ValidationStatus.UNSUPPORTED for trial in unsupported)
    assert all(trial.outcome is TrialOutcome.UNSUPPORTED for trial in unsupported)
    assert result.outcome is not None
    assert result.outcome.disposition != "all_clean"
    assert result.outcome.first_failure_trial_id == mismatch.trial_id
    persisted = service.get_trial(mismatch.trial_id, experiment_id=result.experiment.experiment_id)
    assert persisted.factors == mismatch.factors
    assert persisted.oracle_receipt == mismatch.oracle_receipt
    run = RunStore(workspace).read(mismatch.run_id or "")
    assert run.source_state_id is not None
    assert run.environment_id
    assert run.workload_definition_id == result.experiment.workload_definition_id
