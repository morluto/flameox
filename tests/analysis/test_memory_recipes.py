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
    assert MemoryAnalysisResult.model_validate(result.model_dump()) == result

    contradictory = result.model_dump()
    contradictory["truncated"] = not result.runtime_resources_truncated
    with pytest.raises(ValidationError, match="truncation fields must agree"):
        MemoryAnalysisResult.model_validate(contradictory)

    contradictory = result.model_dump()
    contradictory["truncated"] = True
    contradictory["runtime_resources_truncated"] = True
    with pytest.raises(ValidationError, match="runtime-resource total"):
        MemoryAnalysisResult.model_validate(contradictory)

    contradictory = result.model_dump(mode="json")
    contradictory["unavailable_metrics"] = ["peak_rss"]
    with pytest.raises(ValidationError, match="derive from runtime resources"):
        MemoryAnalysisResult.model_validate(contradictory)

    hotspots = RecipeService(workspace).hotspots("memory-run")
    assert HotspotResult.model_validate(hotspots.model_dump(mode="json")) == hotspots
    contradictory = hotspots.model_dump(mode="json")
    contradictory["evidence_status"] = "available"
    with pytest.raises(ValidationError, match="evidence status fields must agree"):
        HotspotResult.model_validate(contradictory)


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
