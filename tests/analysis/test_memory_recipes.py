from __future__ import annotations

from pathlib import Path

from flameox.analysis import RecipeService
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

    assert [point.phase for point in result.phase_growth] == [
        "warmup",
        "steady_state",
        "shutdown",
    ]
    assert [point.delta for point in result.phase_growth] == [None, 140.0, 20.0]
