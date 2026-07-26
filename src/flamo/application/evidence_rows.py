from __future__ import annotations

import json

from flamo.domain import ArtifactRegistration, EnvironmentRecord, SourceState


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


def _json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
