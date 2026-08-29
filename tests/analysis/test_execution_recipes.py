from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.analysis import (
    ExecutionObservation,
    ExecutionObservationChange,
    ExecutionObservationFilter,
    RecipeService,
)
from flameox.domain import DomainError, ErrorCode
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace
from tests.support.analysis import run_row

pytestmark = pytest.mark.integration


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

    recipes = RecipeService(workspace)
    added = recipes.execution(
        "baseline",
        comparison_input_id="candidate",
        collection="added",
    )
    removed = recipes.execution(
        "baseline",
        comparison_input_id="candidate",
        collection="removed",
    )
    changed = recipes.execution(
        "baseline",
        comparison_input_id="candidate",
        collection="changed",
    )

    assert [item.name for item in added.items] == ["clip_fraction"]
    assert [item.name for item in removed.items] == ["legacy_branch"]
    assert [item.name for item in changed.items] == ["old_log_prob_source"]
    changed_item = changed.items[0]
    assert isinstance(changed_item, ExecutionObservationChange)
    assert changed_item.baseline_value_json == '"rollout"'
    assert changed_item.candidate_value_json == '"epoch"'
    assert added.totals == removed.totals == changed.totals


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

    recipes = RecipeService(workspace)
    result = recipes.execution(
        "baseline",
        comparison_input_id="candidate",
        limit=1,
    )
    added = recipes.execution(
        "baseline",
        comparison_input_id="candidate",
        collection="added",
        limit=1,
    )
    removed = recipes.execution(
        "baseline",
        comparison_input_id="candidate",
        collection="removed",
        limit=1,
    )
    changed = recipes.execution(
        "baseline",
        comparison_input_id="candidate",
        collection="changed",
        limit=1,
    )

    assert [item.name for item in added.items] == ["candidate_early"]
    assert [item.name for item in removed.items] == ["baseline_only"]
    assert [item.name for item in changed.items] == ["shared"]
    assert result.totals.model_dump() == {
        "observations": 2,
        "added": 1,
        "removed": 1,
        "changed": 1,
    }
    assert result.returned == 1
    assert result.truncated
    assert result.model_dump(mode="json")["returned"] == 1

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(result).model_validate({**result.model_dump(mode="python"), "returned": 0})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(result).model_validate({**result.model_dump(mode="python"), "truncated": False})


def test_execution_cursor_pins_snapshot_and_filters_before_paging(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    def observation(
        observation_id: str,
        *,
        file: str,
        name: str,
        line: int,
        kind: str = "line",
    ) -> dict[str, object]:
        return {
            "observation_id": observation_id,
            "run_id": "run",
            "artifact_id": None,
            "kind": kind,
            "name": name,
            "value_json": "true",
            "file": file,
            "line_from": line,
            "line_to": line,
            "context": None,
            "evidence_level": "observed",
        }

    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row("run")],
            "observations": [
                observation("first", file="src/agent.py", name="step", line=10),
                observation("second", file="src/agent.py", name="step", line=20),
                observation("excluded", file="tests/test_agent.py", name="step", line=15),
            ],
        },
        publisher="execution-page-fixture",
        publisher_version="1",
    )
    filters = ExecutionObservationFilter(
        file_prefix="src/",
        kind="line",
        name="step",
        line_from=5,
        line_to=25,
    )
    recipes = RecipeService(workspace)
    first = recipes.execution("run", filters=filters, limit=1)
    assert first.next_cursor is not None
    first_items = tuple(item for item in first.items if isinstance(item, ExecutionObservation))
    assert first_items == first.items
    assert [item.observation_id for item in first_items] == ["first"]

    GenerationPublisher(workspace).publish_rows(
        {
            "observations": [
                observation("later", file="src/agent.py", name="step", line=1),
            ]
        },
        publisher="later-execution-fixture",
        publisher_version="1",
    )
    second = recipes.execution("run", filters=filters, limit=1, cursor=first.next_cursor)

    assert second.corpus_commit_id == first.corpus_commit_id
    second_items = tuple(item for item in second.items if isinstance(item, ExecutionObservation))
    assert second_items == second.items
    assert [item.observation_id for item in second_items] == ["second"]
    assert second.total == 2
    assert second.next_cursor is None


def test_execution_comparison_rejects_ambiguous_semantic_observations(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)

    def observation(observation_id: str, run_id: str, value: str) -> dict[str, object]:
        return {
            "observation_id": observation_id,
            "run_id": run_id,
            "artifact_id": None,
            "kind": "line",
            "name": "step",
            "value_json": value,
            "file": "agent.py",
            "line_from": 1,
            "line_to": 1,
            "context": None,
            "evidence_level": "observed",
        }

    GenerationPublisher(workspace).publish_rows(
        {
            "runs": [run_row("baseline"), run_row("candidate")],
            "observations": [
                observation("baseline-a", "baseline", "true"),
                observation("baseline-b", "baseline", "false"),
                observation("candidate", "candidate", "true"),
            ],
        },
        publisher="ambiguous-execution-fixture",
        publisher_version="1",
    )

    with pytest.raises(DomainError) as raised:
        RecipeService(workspace).execution(
            "baseline",
            comparison_input_id="candidate",
            collection="changed",
        )

    assert raised.value.code is ErrorCode.EVIDENCE_SCHEMA_MISMATCH
    assert raised.value.details["input_id"] == "baseline"
