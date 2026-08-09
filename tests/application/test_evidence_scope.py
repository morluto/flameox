from pathlib import Path

import pytest

from flameox.application import DrilldownService
from flameox.domain import (
    CaptureStatus,
    DomainError,
    ErrorCode,
    ExecutionStatus,
    ValidationStatus,
)
from flameox.domain.models import ExecutionRunManifest
from flameox.evidence_scope import EvidenceScope
from flameox.storage import RunStore, Workspace


def test_drilldown_rejects_run_absent_from_pinned_corpus(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    RunStore(workspace).create(
        ExecutionRunManifest(
            run_id="projection-only",
            execution_status=ExecutionStatus.PLANNED,
            capture_status=CaptureStatus.PENDING,
            validation_status=ValidationStatus.NOT_REQUESTED,
            environment_id="sha256:" + "0" * 64,
        )
    )

    with pytest.raises(DomainError) as error:
        DrilldownService(workspace).callers("projection-only", "frame")

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
    assert "pinned corpus" in error.value.message


def test_mixed_evidence_scope_preserves_run_and_artifact_inputs() -> None:
    scope = EvidenceScope(
        input_ids=("run-1", "sha256:" + "a" * 64),
        run_ids=("run-1",),
        artifact_ids=("sha256:" + "a" * 64,),
    )

    predicate, parameters = scope.predicate(
        run_column="run_id",
        artifact_column="artifact_id",
    )

    assert predicate == "(run_id IN (?)) OR (artifact_id IN (?))"
    assert parameters == ("run-1", "sha256:" + "a" * 64)
