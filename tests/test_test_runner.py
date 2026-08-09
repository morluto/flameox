from __future__ import annotations

import json
import runpy
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.support.ownership import load_ownership

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "test.py"
RECORDS = load_ownership(ROOT / "tests" / "ownership.toml")
runner: Any = SimpleNamespace(**runpy.run_path(str(RUNNER)))
RUNNER_GLOBALS: dict[str, Any] = runner.affected_plan.__globals__
PlanMutation = Callable[[dict[str, Any]], None]


def _plan(
    monkeypatch: pytest.MonkeyPatch, paths: set[str], *, event: str = "local"
) -> dict[str, Any]:
    monkeypatch.setitem(RUNNER_GLOBALS, "changed_paths", lambda _base, _head: (paths, None))
    return cast(dict[str, Any], runner.affected_plan(RECORDS, "HEAD", event=event, head="HEAD"))


def test_list_reports_lanes_and_metadata_commands() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "list"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "  golden" in result.stdout
    assert "  optional-nvbench" in result.stdout
    assert "Metadata commands:" in result.stdout
    assert "  capabilities validate managed setup metadata against extras" in result.stdout


def test_affected_plan_no_changes_is_valid_and_explicitly_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(monkeypatch, set())

    assert runner.validate_affected_plan(plan, RECORDS) == plan
    assert plan["schema_version"] == 3
    assert plan["fallback_reason"] == "none"
    assert plan["full"] is False
    assert plan["lanes"] == []
    assert plan["optional_lanes"] == []
    assert plan["selected_lanes"] == []
    assert plan["run_performance"] is False
    assert len(plan["unselected_lanes"]) == len(runner.ALL_PLANNED_LANES)


@pytest.mark.parametrize(
    ("path", "full", "fallback"),
    [
        ("src/flameox/domain/models.py", True, "production_change"),
        ("tests/domain/test_identity.py", False, "none"),
        ("docs/testing.md", False, "none"),
        ("vendor/new-file.txt", True, "unknown_path"),
    ],
)
def test_affected_plan_classifies_source_test_docs_and_unknown_changes(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    full: bool,
    fallback: str,
) -> None:
    plan = _plan(monkeypatch, {path})

    assert plan["full"] is full
    assert plan["fallback_reason"] == fallback
    assert path in plan["changed_paths"]
    if path == "tests/domain/test_identity.py":
        selected = {item["lane"] for item in plan["selected_lanes"]}
        assert "core" in selected
        assert "core" not in {item["lane"] for item in plan["unselected_lanes"]}


def test_direct_ownership_does_not_pass_through_unknown_path_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(monkeypatch, {"tests/domain/test_identity.py"})

    assert plan["full"] is False
    assert plan["fallback_reason"] == "none"
    assert "unowned_test_path" not in plan["fallback_reason"]
    assert any(
        item["lane"] == "core" and "owned test path" in item["reason"]
        for item in plan["selected_lanes"]
    )


def test_performance_marked_test_selects_performance_lane_on_pull_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(monkeypatch, {"tests/performance/test_catalog_scale.py"}, event="pull_request")

    assert plan["full"] is False
    assert plan["lanes"] == []
    assert plan["run_performance"] is True
    assert any(item["lane"] == "performance" for item in plan["selected_lanes"])
    assert all(item["lane"] != "performance" for item in plan["unselected_lanes"])


@pytest.mark.parametrize(
    "event", ["pull_request", "merge_group", "push", "schedule", "workflow_dispatch"]
)
def test_gpu_provider_lanes_are_local_only_and_never_emitted_to_hosted_plans(
    monkeypatch: pytest.MonkeyPatch,
    event: str,
) -> None:
    plan = _plan(monkeypatch, {"tests/adapters/test_nvbench_live.py"}, event=event)

    emitted = {
        *plan["optional_lanes"],
        *(item["lane"] for item in plan["selected_lanes"]),
        *(item["lane"] for item in plan["unselected_lanes"]),
        *(item["lane"] for item in plan["matrix"]["optional_lanes"]),
    }
    assert emitted.isdisjoint(runner.GPU_PROVIDER_LANES)
    assert runner.validate_affected_plan(plan, RECORDS) == plan


def test_hosted_optional_provider_lane_remains_schedulable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(monkeypatch, {"tests/adapters/test_coverage.py"}, event="pull_request")

    assert "optional-coverage" in plan["optional_lanes"]


@pytest.mark.parametrize(
    "event",
    ["local", "pull_request", "merge_group", "push", "schedule", "workflow_dispatch"],
)
def test_affected_plan_records_event_policy(
    monkeypatch: pytest.MonkeyPatch,
    event: str,
) -> None:
    plan = _plan(monkeypatch, set(), event=event)

    assert plan["event"] == event
    assert plan["provenance"]["event"] == event
    if event in runner.FULL_RUN_EVENTS:
        assert plan["full"] is True
        assert plan["run_performance"] is True
        assert "event_requires_full_validation" in plan["fallback_reason"]


def test_missing_base_selects_full_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_BASE_SHA", raising=False)
    monkeypatch.setitem(
        RUNNER_GLOBALS,
        "changed_paths",
        lambda _base, _head: (set(), "missing_base_revision"),
    )

    plan = runner.affected_plan(RECORDS, None, event="pull_request", head="HEAD")

    assert plan["base_available"] is False
    assert plan["full"] is True
    assert plan["fallback_reason"] == "missing_base_revision"
    assert plan["lanes"] == list(runner.TEST_LANES)
    assert plan["optional_lanes"] == list(runner.CI_PROVIDER_LANES)
    assert set(plan["optional_lanes"]).isdisjoint(runner.GPU_PROVIDER_LANES)


def _remove_schema_version(plan: dict[str, Any]) -> None:
    del plan["schema_version"]


def _add_unknown_field(plan: dict[str, Any]) -> None:
    plan["unexpected"] = True


def _add_unknown_lane(plan: dict[str, Any]) -> None:
    plan["lanes"] = ["unknown-lane"]


def _oversize_matrix(plan: dict[str, Any]) -> None:
    plan["matrix"]["lanes"] = [{"lane": "core"}] * (runner.PLAN_MAX_MATRIX_ITEMS + 1)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_remove_schema_version, id="missing-field"),
        pytest.param(_add_unknown_field, id="extra-field"),
        pytest.param(_add_unknown_lane, id="unknown-lane"),
        pytest.param(_oversize_matrix, id="oversized-matrix"),
    ],
)
def test_affected_plan_validator_rejects_malformed_output(
    monkeypatch: pytest.MonkeyPatch,
    mutate: PlanMutation,
) -> None:
    plan = _plan(monkeypatch, set())
    malformed = deepcopy(plan)
    mutate(malformed)

    with pytest.raises((runner.PlanValidationError, ValueError)):
        runner.validate_affected_plan(malformed, RECORDS)


def test_affected_plan_validator_rejects_inconsistent_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(monkeypatch, set())
    malformed = deepcopy(plan)
    malformed["base_available"] = False

    with pytest.raises((runner.PlanValidationError, ValueError), match="provenance"):
        runner.validate_affected_plan(malformed, RECORDS)


def test_affected_cli_emits_json_contract_for_merge_group() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "affected",
            "--base",
            "HEAD",
            "--head",
            "HEAD",
            "--event",
            "merge_group",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["schema_version"] == 3
    assert plan["event"] == "merge_group"
    runner.validate_affected_plan(plan, RECORDS)
