from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.analysis import RecipeService
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace
from tests.support.analysis import run_row


def test_execution_compares_path_and_semantic_observation_changes(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)

    def observation(
        run_id: str,
        observation_id: str,
        name: str,
        value: str,
        line: int,
    ) -> dict[str, object]:
        return {
            "observation_id": observation_id,
            "run_id": run_id,
            "artifact_id": None,
            "kind": "configuration",
            "name": name,
            "value_json": value,
            "file": "policy.py",
            "line_from": line,
            "line_to": line,
            "context": "update",
            "evidence_level": "observed",
        }

    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row("baseline"), run_row("candidate")],
            "observations": [
                observation("baseline", "old-source", "old_log_prob_source", '"rollout"', 10),
                observation("baseline", "removed", "legacy_branch", "true", 20),
                observation("candidate", "new-source", "old_log_prob_source", '"epoch"', 10),
                observation("candidate", "added", "clip_fraction", "0.21", 30),
            ],
        },
        publisher="execution-fixture",
        publisher_version="1",
    )

    result = RecipeService(workspace).execution(
        "baseline",
        comparison_input_id="candidate",
    )

    assert [item.name for item in result.added] == ["clip_fraction"]
    assert [item.name for item in result.removed] == ["legacy_branch"]
    assert [item.name for item in result.changed] == ["old_log_prob_source"]
    assert result.changed[0].baseline_value_json == '"rollout"'
    assert result.changed[0].candidate_value_json == '"epoch"'


def test_execution_comparison_diffs_complete_inputs_before_limiting(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)

    def observation(
        run_id: str,
        observation_id: str,
        name: str,
        value: str,
        line: int,
    ) -> dict[str, object]:
        return {
            "observation_id": observation_id,
            "run_id": run_id,
            "artifact_id": None,
            "kind": "configuration",
            "name": name,
            "value_json": value,
            "file": "policy.py",
            "line_from": line,
            "line_to": line,
            "context": "update",
            "evidence_level": "observed",
        }

    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row("baseline"), run_row("candidate")],
            "observations": [
                observation("baseline", "shared-old", "shared", '"old"', 10),
                observation("baseline", "baseline-only", "baseline_only", "true", 20),
                observation("candidate", "candidate-early", "candidate_early", "true", 1),
                observation("candidate", "shared-new", "shared", '"new"', 10),
            ],
        },
        publisher="bounded-execution-fixture",
        publisher_version="1",
    )

    result = RecipeService(workspace).execution(
        "baseline",
        comparison_input_id="candidate",
        limit=1,
    )

    assert [item.name for item in result.added] == ["candidate_early"]
    assert [item.name for item in result.removed] == ["baseline_only"]
    assert [item.name for item in result.changed] == ["shared"]
    assert result.returned == 1
    assert result.truncated
    assert result.model_dump(mode="json")["returned"] == 1

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(result).model_validate({**result.model_dump(mode="python"), "returned": 0})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(result).model_validate({**result.model_dump(mode="python"), "truncated": False})
