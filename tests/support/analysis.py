from __future__ import annotations

from datetime import UTC, datetime

FIXTURE_CREATED_AT = datetime(2025, 1, 2, 3, 4, tzinfo=UTC)


def run_row(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "created_at": FIXTURE_CREATED_AT,
        "run_type": "execution",
        "execution_status": "succeeded",
        "capture_status": "complete",
        "validation_status": "passed",
        "workload_definition_id": "workload",
        "workload_instance_id": "workload-instance",
        "measurement_protocol_id": "protocol",
        "environment_id": "environment",
        "source_state_id": "source",
        "adapter": "fixture",
        "adapter_version": "1",
        "run_semantic_id": "sha256:" + "f" * 64,
        "exit_code": 0,
        "wall_time_ns": 1,
    }
