from __future__ import annotations

import json
from pathlib import Path

from flameox.domain import (
    ArtifactRegistration,
    EnvironmentRecord,
    RunManifest,
    SourceState,
)
from flameox.domain.models import utc_now
from flameox.execution import ProcessObservation


def artifact_registration_row(
    registration: ArtifactRegistration,
    *,
    byte_length: int,
) -> dict[str, object]:
    return {
        **registration.model_dump(mode="python"),
        "kind": registration.kind.value,
        "sensitivity": registration.sensitivity.value,
        "byte_length": byte_length,
    }


def environment_row(environment: EnvironmentRecord) -> dict[str, object]:
    return {
        "environment_id": environment.environment_id,
        "observed_at": environment.observed_at,
        "identity_quality": environment.identity_quality.value,
        "fields_json": _json(environment.fields),
        "missing_fields": list(environment.missing_fields),
    }


def source_state_row(source_state: SourceState) -> dict[str, object]:
    return {
        "source_state_id": source_state.source_state_id,
        "identity_quality": source_state.identity_quality.value,
        "repository_root": source_state.repository_root,
        "head_commit": source_state.head_commit,
        "diff_digest": source_state.diff_digest,
        "executable_digest": source_state.executable_digest,
        "build_id": source_state.build_id,
        "fields_json": _json(source_state.fields),
        "missing_fields": list(source_state.missing_fields),
    }


def process_observation_rows(
    run_id: str,
    observations: tuple[ProcessObservation, ...],
    *,
    artifact_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Project broker observations into the two bounded process evidence tables."""
    phases = sorted({item.snapshot_phase for item in observations}) or ["post_root_exit"]
    summaries: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    for phase in phases:
        phase_items = tuple(item for item in observations if item.snapshot_phase == phase)
        snapshot_id = f"{artifact_id}:{run_id}:{phase}"
        observed_at = max(
            (item.observed_at for item in phase_items),
            default=utc_now(),
        )
        summaries.append(
            {
                "run_id": run_id,
                "artifact_id": artifact_id,
                "snapshot_id": snapshot_id,
                "phase": phase,
                "observed_at": observed_at,
                "entry_count": len(phase_items),
                "sources_json": _json(sorted({item.discovery_source for item in phase_items})),
                "limitations": sorted(
                    {
                        f"process:{item.pid}:{failure}"
                        for item in phase_items
                        for failure in item.failures
                    }
                ),
            }
        )
        entries.extend(
            {
                "snapshot_id": snapshot_id,
                "pid": item.pid,
                "create_time": item.create_time,
                "parent_pid": item.parent_pid,
                "parent_create_time": item.parent_create_time,
                "discovery_source": item.discovery_source,
                "name": item.name,
                "status": item.status,
                "rss_bytes": item.rss_bytes,
                "cpu_user_seconds": item.cpu_user_seconds,
                "cpu_system_seconds": item.cpu_system_seconds,
                "thread_count": item.thread_count,
                "fd_count": item.fd_count,
                "observed_at": item.observed_at,
                "alive_before_cleanup": item.alive_before_cleanup,
                "cleanup_action": item.cleanup_action,
                "cleanup_outcome": item.cleanup_outcome,
                "failures_json": _json(list(item.failures)),
            }
            for item in phase_items
        )
    return summaries, entries


def runtime_resource_summary_row(
    manifest: RunManifest,
    *,
    sampling_interval_ms: int,
) -> dict[str, object]:
    resources = manifest.process.resources if manifest.process is not None else None
    unavailable = set(resources.unavailable_metrics if resources is not None else ())
    if resources is None:
        unavailable.update({"minimum_free_bytes", "staging_growth_bytes", "peak_rss_bytes"})
    return {
        "run_id": manifest.run_id,
        "sampling_interval_ms": (
            resources.sampling_interval_ms if resources is not None else sampling_interval_ms
        ),
        "minimum_free_bytes": resources.minimum_free_bytes if resources is not None else None,
        "staging_growth_bytes": resources.staging_growth_bytes if resources is not None else None,
        "peak_rss_bytes": resources.peak_rss_bytes if resources is not None else None,
        "peak_rss_backend": resources.peak_rss_backend if resources is not None else None,
        "policy_termination": (resources.policy_termination if resources is not None else None),
        "unavailable_metrics": _normalized_unavailable_metrics(unavailable),
    }


def runtime_writable_root_rows(
    manifest: RunManifest,
    *,
    project_root: Path,
) -> list[dict[str, object]]:
    resources = manifest.process.resources if manifest.process is not None else None
    unavailable = set(resources.unavailable_metrics if resources is not None else ())
    rows: list[dict[str, object]] = []
    for binding in manifest.writable_roots:
        growth = (
            resources.writable_root_growth_bytes.get(binding.storage_path) if resources else None
        )
        marker = f"writable_root_growth:{binding.storage_path}"
        target = Path(binding.target_path)
        try:
            normalized = target.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            normalized = "."
        rows.append(
            {
                "run_id": manifest.run_id,
                "writable_root_identity": binding.target_identity,
                "target_path": normalized,
                "growth_bytes": growth,
                "available": growth is not None,
                "unavailable_reason": (
                    _unavailable_reason(unavailable, marker) if growth is None else None
                ),
            }
        )
    return rows


def _normalized_unavailable_metrics(metrics: set[str]) -> list[str]:
    normalized: set[str] = set()
    for metric in metrics:
        if metric.startswith("writable_root_growth:"):
            normalized.add("writable_root_growth")
        elif metric in {
            "minimum_free_bytes",
            "staging_growth_bytes",
            "peak_rss_bytes",
        }:
            normalized.add(metric.removesuffix("_bytes"))
        else:
            normalized.add(metric)
    return sorted(normalized)


def _unavailable_reason(metrics: set[str], marker: str) -> str:
    if marker in metrics:
        return "writable_root_growth_unavailable"
    if "writable_root_growth" in metrics:
        return "writable_root_growth_unavailable"
    return "resource_summary_unavailable"


def _json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
