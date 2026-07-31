from __future__ import annotations

from datetime import UTC, datetime


def run_row(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC),
        "run_type": "execution",
        "execution_status": "succeeded",
        "capture_status": "complete",
        "validation_status": "passed",
        "workload_definition_id": "workload",
        "workload_instance_id": "workload-instance",
        "measurement_protocol_id": "protocol",
        "environment_id": "environment",
        "source_state_id": "source",
        "collector": "fixture",
        "collector_version": "1",
        "exit_code": 0,
        "wall_time_ns": 1,
        "manifest_path": f"runs/{run_id}/manifest.json",
    }
