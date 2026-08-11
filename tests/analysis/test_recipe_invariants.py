from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest
from pydantic import ValidationError

from flameox.analysis import RecipeService
from flameox.analysis.recipe_models import FailureAnalysisResult
from flameox.domain import DomainError, ErrorCode
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace
from tests.support.analysis import run_row


def test_read_only_analysis_pins_snapshot_without_workspace_write_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    head = workspace.corpus.read_head().commit_id

    def fail_write_lock(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("read-only analysis acquired the workspace write lock")

    monkeypatch.setattr(workspace, "write_locked", fail_write_lock)

    result = RecipeService(workspace).failures(limit=10)

    assert result.corpus_commit_id == head
    assert result.eligible_runs == 0
    assert result.failed_runs == 0
    assert result.population_status == "empty"
    assert result.empty_reason == "no_runs"
    assert workspace.corpus.read_head().commit_id == head


def test_failure_analysis_distinguishes_filtered_and_failure_empty_populations(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    publisher = GenerationPublisher(workspace)
    successful = run_row("successful")
    publisher.publish_rows(
        {"runs": [successful]},
        publisher="test",
        publisher_version="1",
    )

    observed = RecipeService(workspace).failures()
    filtered = RecipeService(workspace).failures(environment_id="different")

    assert (observed.eligible_runs, observed.failed_runs) == (1, 0)
    assert observed.population_status == "observed"
    assert observed.empty_reason == "no_failures"
    assert observed.coverage == {
        "source_identity": 0.0,
        "artifact": 0.0,
        "symbolized_frames": 0.0,
    }
    assert (filtered.eligible_runs, filtered.failed_runs) == (0, 0)
    assert filtered.population_status == "filtered_empty"
    assert filtered.empty_reason == "no_matching_runs"
    assert observed.validated_copy() == observed

    contradictory = observed.model_dump(mode="json")
    contradictory["empty_reason"] = "no_runs"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FailureAnalysisResult.model_validate(contradictory)


def test_analysis_rejects_input_absent_from_pinned_corpus(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    with pytest.raises(DomainError) as error:
        RecipeService(workspace).hotspots("missing-run")

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
