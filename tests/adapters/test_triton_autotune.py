from __future__ import annotations

import json
from pathlib import Path

import pytest

from flameox.adapters.triton_autotune import load_triton_autotune_selections

pytestmark = pytest.mark.unit


def _event(*, cache_hit: bool = False, duration_ms: float | None = 32.0) -> dict[str, object]:
    return {
        "function_name": "workload.kernel",
        "key_digest": "sha256:" + "1" * 64,
        "cache_hit": cache_hit,
        "duration_ms": duration_ms,
        "winner": {
            "kwargs": {"BLOCK": 256},
            "num_warps": 8,
            "num_stages": 1,
            "num_ctas": 1,
            "maxnreg": None,
            "ir_override": None,
        },
        "candidate_count": 2,
        "candidates_truncated": False,
        "timings_truncated": False,
        "candidates": [
            {
                "config": {
                    "kwargs": {"BLOCK": 128},
                    "num_warps": 4,
                    "num_stages": 1,
                    "num_ctas": 1,
                    "maxnreg": None,
                    "ir_override": None,
                },
                "timings_ms": [2.0, 1.8, 2.2],
            },
            {
                "config": {
                    "kwargs": {"BLOCK": 256},
                    "num_warps": 8,
                    "num_stages": 1,
                    "num_ctas": 1,
                    "maxnreg": None,
                    "ir_override": None,
                },
                "timings_ms": [1.0, 0.9, 1.1],
            },
        ],
    }


def test_listener_output_preserves_provider_selection_without_artifact_inference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "triton-autotune.jsonl"
    path.write_text(json.dumps(_event()) + "\n")

    selections, limitations = load_triton_autotune_selections(path, run_id="run-1")

    assert limitations == ()
    assert len(selections) == 1
    selection = selections[0]
    assert selection.function_name == "workload.kernel"
    assert selection.cache_hit is False
    assert selection.duration_ms == 32.0
    assert selection.candidate_count == 2
    assert selection.candidates_truncated is False
    assert selection.winner_config_id in {item.config_id for item in selection.candidates}
    assert "artifact_id" not in selection.row()
    assert "pipeline_id" not in selection.row()


def test_listener_output_treats_cache_hit_duration_as_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "triton-autotune.jsonl"
    path.write_text(json.dumps(_event(cache_hit=True, duration_ms=None)) + "\n")

    selections, limitations = load_triton_autotune_selections(path, run_id="run-1")

    assert limitations == ()
    assert selections[0].cache_hit is True
    assert selections[0].duration_ms is None


def test_listener_output_rejects_inconsistent_cache_hit_duration(tmp_path: Path) -> None:
    path = tmp_path / "triton-autotune.jsonl"
    path.write_text(json.dumps(_event(cache_hit=True, duration_ms=1.0)) + "\n")

    selections, limitations = load_triton_autotune_selections(path, run_id="run-1")

    assert selections == ()
    assert limitations == ("1 invalid Triton autotune listener event(s) were omitted.",)


def test_listener_output_reports_an_unavailable_hook_without_cache_fallback(tmp_path: Path) -> None:
    path = tmp_path / "triton-autotune.jsonl"
    path.write_text(json.dumps({"listener_unavailable": "Triton listener is unavailable."}) + "\n")

    selections, limitations = load_triton_autotune_selections(path, run_id="run-1")

    assert selections == ()
    assert limitations == ("Triton listener is unavailable.",)
