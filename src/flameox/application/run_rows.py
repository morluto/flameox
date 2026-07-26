from __future__ import annotations

from flameox.domain import RunManifest


def run_row(manifest: RunManifest) -> dict[str, object]:
    return {
        "run_id": manifest.run_id,
        "created_at": manifest.created_at,
        "run_type": manifest.run_type.value,
        "execution_status": manifest.execution_status.value,
        "capture_status": manifest.capture_status.value,
        "validation_status": manifest.validation_status.value,
        "workload_definition_id": manifest.workload_definition_id,
        "workload_instance_id": manifest.workload_instance_id,
        "measurement_protocol_id": manifest.measurement_protocol_id,
        "environment_id": manifest.environment_id,
        "source_state_id": manifest.source_state_id,
        "collector": manifest.collector,
        "collector_version": manifest.collector_version,
        "exit_code": manifest.process.exit_code if manifest.process else None,
        "wall_time_ns": (manifest.process.wall_time_ns if manifest.process else None),
        "manifest_path": f"runs/{manifest.run_id}/manifest.json",
    }
