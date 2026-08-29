from __future__ import annotations

import json

from flameox.domain import ResourceAvailability, RunManifest, digest_model
from flameox.storage.control_plane import canonical_json


def run_row(manifest: RunManifest) -> dict[str, object]:
    return {
        "run_id": manifest.run_id,
        "run_revision": manifest.revision,
        "run_manifest_digest": digest_model(manifest.model_dump(mode="json")),
        "created_at": manifest.created_at,
        "run_type": manifest.run_type.value,
        "execution_status": manifest.execution_status.value,
        "capture_status": manifest.capture_status.value,
        "validation_status": manifest.validation_status.value,
        "workload_definition_id": manifest.workload_definition_id,
        "workload_instance_id": manifest.workload_instance_id,
        "measurement_protocol_id": manifest.measurement_protocol_id,
        "source_measurement_run_id": manifest.source_measurement_run_id,
        "environment_id": manifest.environment_id,
        "source_state_id": manifest.source_state_id,
        "adapter": manifest.semantics.adapter,
        "adapter_version": manifest.semantics.adapter_version,
        "run_semantic_id": manifest.semantics.semantic_id,
        "exit_code": manifest.process.exit_code if manifest.process else None,
        "wall_time_ns": (manifest.process.wall_time_ns if manifest.process else None),
        "orchestrator": (
            manifest.external_context.orchestrator if manifest.external_context else None
        ),
        "provider": manifest.external_context.provider if manifest.external_context else None,
        "lease_id": manifest.external_context.lease_id if manifest.external_context else None,
        "worker_id": manifest.external_context.worker_id if manifest.external_context else None,
        "orchestration_run_id": (
            manifest.external_context.orchestration_run_id if manifest.external_context else None
        ),
        "execution_identity_id": (
            manifest.execution_identity.identity_id if manifest.execution_identity else None
        ),
        "execution_identity_quality": (
            manifest.execution_identity.quality if manifest.execution_identity else None
        ),
        "execution_identity_json": (
            json.dumps(
                manifest.execution_identity.model_dump(mode="json"),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if manifest.execution_identity
            else None
        ),
        "inference_protocol_identity_id": manifest.inference_protocol_identity_id,
        "inference_protocol_identity_json": manifest.inference_protocol_identity_json,
        "limitations": list(manifest.limitations),
        "limitation_details_json": json.dumps(
            [item.model_dump(mode="json") for item in manifest.limitation_details],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "resource_availability": (
            ResourceAvailability.UNAVAILABLE
            if manifest.process is None or manifest.process.resources is None
            else (
                ResourceAvailability.AVAILABLE
                if (
                    not manifest.process.resources.unavailable_metrics
                    and manifest.process.resources.policy_termination is None
                )
                else ResourceAvailability.PARTIAL
            )
        ),
        "manifest_path": f"control-plane:run/{manifest.run_id}@{manifest.revision}",
        "manifest_json": canonical_json(
            manifest.model_dump(mode="json", exclude={"process": {"timed_out"}})
        ),
    }
