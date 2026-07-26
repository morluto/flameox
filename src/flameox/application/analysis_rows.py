from __future__ import annotations

import json

from flameox.domain import AnalysisRecord


def analysis_row(value: AnalysisRecord) -> dict[str, object]:
    return {
        "analysis_id": value.analysis_id,
        "recipe": value.recipe,
        "recipe_version": value.recipe_version,
        "parameters_json": json.dumps(
            value.parameters,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "parameters_digest": value.parameters_digest,
        "corpus_commit_id": value.corpus_commit_id,
        "input_generation_ids": list(value.input_generation_ids),
        "input_run_ids": list(value.input_run_ids),
        "input_artifact_ids": list(value.input_artifact_ids),
        "result_digest": value.result_digest,
        "result_artifact_id": value.result_artifact_id,
        "coverage_json": json.dumps(
            value.coverage,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "limitations": list(value.limitations),
        "started_at": value.started_at,
        "completed_at": value.completed_at,
    }
