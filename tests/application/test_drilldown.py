from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from flameox.action_graph import ActionId
from flameox.application import DrilldownService
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace


def _frame(frame_id: str) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "language": "Python",
        "function": frame_id,
        "module": None,
        "file": "work.py",
        "line": 1,
        "column": None,
        "address": None,
        "build_id": None,
        "module_relative_address": None,
        "inline_chain_id": None,
        "source_state_id": None,
        "artifact_id": "sha256:" + "a" * 64,
        "inlined": False,
        "symbolization": "complete",
    }


def test_call_edge_cursor_preserves_rows_that_differ_only_by_unit(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    artifact_id = "sha256:" + "a" * 64
    now = datetime(2026, 8, 30, tzinfo=UTC)
    GenerationPublisher(workspace).publish_rows(
        {
            "artifact_registrations": [
                {
                    "registration_id": "registration",
                    "run_id": "run",
                    "artifact_id": artifact_id,
                    "display_name": "profile",
                    "kind": "memory_profile",
                    "media_type": "application/octet-stream",
                    "byte_length": 1,
                    "sensitivity": "normal",
                    "role": "primary",
                    "producer": "memray",
                    "producer_version": "1",
                    "registered_at": now,
                }
            ],
            "frames": [_frame("parent"), _frame("child")],
            "call_edges": [
                {
                    "run_id": "run",
                    "artifact_id": artifact_id,
                    "parent_frame_id": "parent",
                    "child_frame_id": "child",
                    "metric": "same.metric",
                    "weight_value": 10,
                    "unit": unit,
                    "sample_count": 1,
                }
                for unit in ("bytes", "ns")
            ],
        },
        publisher="test",
        publisher_version="1",
        input_artifact_ids=(artifact_id,),
    )

    first = DrilldownService(workspace).callees(artifact_id, "parent", limit=1)
    assert first.next_cursor is not None
    second = DrilldownService(workspace).callees(
        artifact_id,
        "parent",
        limit=1,
        cursor=first.next_cursor,
    )

    assert {first.frames[0].unit, second.frames[0].unit} == {"bytes", "ns"}
    assert second.next_cursor is None


def test_empty_memory_stack_drilldown_routes_to_project_frames(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    artifact_id = "sha256:" + "a" * 64
    GenerationPublisher(workspace).publish_rows(
        {
            "artifact_registrations": [
                {
                    "registration_id": "registration",
                    "run_id": "run",
                    "artifact_id": artifact_id,
                    "display_name": "profile",
                    "kind": "memory_profile",
                    "media_type": "application/octet-stream",
                    "byte_length": 1,
                    "sensitivity": "normal",
                    "role": "primary",
                    "producer": "memray",
                    "producer_version": "1",
                    "registered_at": datetime(2026, 8, 30, tzinfo=UTC),
                }
            ],
            "frames": [_frame("framework")],
        },
        publisher="test",
        publisher_version="1",
        input_artifact_ids=(artifact_id,),
    )

    result = DrilldownService(workspace).examples(
        artifact_id,
        "framework",
        metric="memory.retained_end",
    )

    assert result.examples == ()
    assert result.recovery is not None
    assert result.recovery.action is ActionId.ANALYZE_MEMORY
    assert result.recovery.arguments["query"]["project_only"] is True
