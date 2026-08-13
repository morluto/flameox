from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from flameox.analysis import RecipeService
from flameox.analysis.recipe_models import (
    HotspotResult,
    MemoryAnalysisResult,
    parse_writable_root_observation,
)
from flameox.evidence import GenerationPublisher
from flameox.storage import Workspace
from tests.support.analysis import run_row

pytestmark = pytest.mark.integration


def test_memory_reports_phase_correlated_growth(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    rows: list[dict[str, object]] = []
    for index, (phase, value) in enumerate(
        (("warmup", 100), ("steady_state", 240), ("shutdown", 260))
    ):
        rows.append(
            {
                "measurement_id": f"memory-{index}",
                "run_id": "memory-run",
                "artifact_id": None,
                "name": "memory.retained_end",
                "value_int": value,
                "value_float": None,
                "unit": "bytes",
                "aggregation": "single",
                "scope": "process",
                "trial_id": None,
                "worker_id": None,
                "worker_run_index": 0,
                "value_index": index,
                "loop_count": None,
                "is_warmup": phase == "warmup",
                "block_id": None,
                "variant_id": None,
                "order_in_block": None,
                "phase": phase,
                "dimensions": {},
                "evidence_level": "observed",
            }
        )
    GenerationPublisher(workspace).publish_rows(
        {"runs": [run_row("memory-run")], "measurements": rows},
        publisher="memory-fixture",
        publisher_version="1",
    )

    result = RecipeService(workspace).memory("memory-run")

    assert result.schema_version == 2
    assert [
        item.value.value if item.value is not None else None for item in result.measurements
    ] == [
        100,
        240,
        260,
    ]
    assert [point.phase for point in result.phase_growth] == [
        "warmup",
        "steady_state",
        "shutdown",
    ]
    assert [point.delta for point in result.phase_growth] == [None, 140.0, 20.0]
    assert result.truncated is result.runtime_resources_truncated
    assert result.validated_copy() == result

    contradictory = result.model_dump()
    contradictory["truncated"] = not result.runtime_resources_truncated
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MemoryAnalysisResult.model_validate(contradictory)

    assert result.runtime_resource_totals is not None
    with pytest.raises(ValidationError, match="runtime-resource total"):
        result.validated_copy(
            update={
                "runtime_resource_totals": result.runtime_resource_totals.validated_copy(
                    update={"run_count": len(result.runtime_resources) - 1}
                )
            }
        )

    contradictory = result.model_dump(mode="json")
    contradictory["unavailable_metrics"] = ["peak_rss"]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MemoryAnalysisResult.model_validate(contradictory)

    hotspots = RecipeService(workspace).hotspots("memory-run")
    assert hotspots.validated_copy() == hotspots
    contradictory = hotspots.model_dump(mode="json")
    contradictory["evidence_status"] = "available"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        HotspotResult.model_validate(contradictory)


def test_hotspots_join_frames_within_the_measurement_artifact(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    baseline_artifact = "sha256:" + "a" * 64
    candidate_artifact = "sha256:" + "b" * 64
    rows: dict[str, list[dict[str, object]]] = {
        "runs": [run_row("baseline-run"), run_row("candidate-run")],
        "artifact_registrations": [
            {
                "registration_id": f"registration-{name}",
                "run_id": f"{name}-run",
                "artifact_id": artifact_id,
                "display_name": f"{name}.bin",
                "kind": "memory_profile",
                "media_type": "application/octet-stream",
                "byte_length": 1,
                "sensitivity": "internal",
                "role": "primary",
                "producer": "fixture",
                "producer_version": "1",
                "registered_at": run_row(f"{name}-run")["created_at"],
            }
            for name, artifact_id in (
                ("baseline", baseline_artifact),
                ("candidate", candidate_artifact),
            )
        ],
        "frames": [
            {
                "frame_id": "shared-logical-frame",
                "language": "Python",
                "function": f"{name}_function",
                "module": "fixture",
                "file": "fixture.py",
                "line": line,
                "column": None,
                "address": None,
                "build_id": None,
                "module_relative_address": None,
                "inline_chain_id": None,
                "source_state_id": "source",
                "artifact_id": artifact_id,
                "inlined": False,
                "symbolization": "complete",
            }
            for name, artifact_id, line in (
                ("baseline", baseline_artifact, 10),
                ("candidate", candidate_artifact, 20),
            )
        ],
        "frame_measurements": [
            {
                "run_id": "candidate-run",
                "artifact_id": candidate_artifact,
                "frame_id": "shared-logical-frame",
                "metric": "memory.retained_end",
                "self_value": 8,
                "inclusive_value": 13,
                "unit": "bytes",
                "sample_count": 1,
                "thread_name": "main",
                "process_name": "python",
                "phase": "shutdown",
            }
        ],
    }
    GenerationPublisher(workspace).publish_rows(
        rows,
        publisher="artifact-scoped-frame-fixture",
        publisher_version="1",
    )

    result = RecipeService(workspace).hotspots("candidate-run")

    assert result.total == 1
    assert result.coverage["completely_symbolized"] == 1
    assert [(item.function, item.line) for item in result.hotspots] == [("candidate_function", 20)]


def test_writable_root_observation_derives_availability_from_growth() -> None:
    available = parse_writable_root_observation(
        {
            "run_id": "run",
            "writable_root_identity": "root",
            "target_path": "build",
            "growth_bytes": 0,
            "available": True,
        }
    )
    unavailable = parse_writable_root_observation(
        {
            "run_id": "run",
            "writable_root_identity": "root",
            "target_path": "build",
            "growth_bytes": None,
            "available": False,
            "unavailable_reason": "resource_summary_unavailable",
        }
    )

    assert available.available is True
    assert unavailable.available is False
    assert parse_writable_root_observation(unavailable.model_dump()) == unavailable

    contradictory = unavailable.model_dump()
    contradictory["available"] = True
    with pytest.raises(ValidationError):
        parse_writable_root_observation(contradictory)
