from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

FIXTURE = (
    Path(__file__).parents[1] / "docs" / "fixtures" / "gitcontribute-external-validation-v1.json"
)
FIELDS = {
    "schema_version",
    "producer",
    "receipt_sha256",
    "validation_id",
    "investigation_id",
    "kind",
    "repository",
    "revision",
    "artifact_sha256",
    "provider",
    "external_run_id",
    "argv",
    "working_dir",
    "environment",
    "artifacts",
    "started_at",
    "completed_at",
    "exit_code",
    "classification",
    "truncated",
    "limitations",
    "incomplete",
}


def _validate(value: dict[str, Any]) -> None:
    if set(value) != FIELDS:
        raise ValueError("external receipt fields changed")
    if value["schema_version"] != "gitcontribute.external-validation.v1":
        raise ValueError("unsupported schema")
    for digest in (
        value["receipt_sha256"],
        value["artifact_sha256"],
        *value["artifacts"].values(),
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid digest")
        bytes.fromhex(digest)
    started = datetime.fromisoformat(value["started_at"].replace("Z", "+00:00"))
    completed = datetime.fromisoformat(value["completed_at"].replace("Z", "+00:00"))
    if completed < started:
        raise ValueError("invalid timestamps")
    if not value["incomplete"] and (
        not value["repository"] or not value["revision"] or not value["artifact_sha256"]
    ):
        raise ValueError("complete receipts require source and artifact identity")


def _downstream_digest(value: dict[str, Any]) -> str:
    projected = dict(value)
    projected["receipt_sha256"] = ""
    if not projected["truncated"]:
        projected.pop("truncated")
    payload = json.dumps(projected, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def test_external_receipt_fixture_matches_downstream_contract() -> None:
    value = json.loads(FIXTURE.read_text())

    _validate(value)

    assert value["producer"] == "flameox"
    assert value["environment"]["flameox.version"] == "0.1.2"
    assert value["receipt_sha256"] == _downstream_digest(value)
    assert value["incomplete"] is True
    assert value["limitations"]
    serialized = json.dumps(value, sort_keys=True)
    assert "/home/" not in serialized
    assert "\\Users\\" not in serialized


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra="unknown"),
        lambda value: value.update(schema_version="unsupported"),
        lambda value: value.update(artifact_sha256="bad"),
        lambda value: value.update(completed_at="2020-01-01T00:00:00Z"),
        lambda value: value.update(incomplete=False, repository=""),
    ],
)
def test_external_receipt_validator_rejects_invalid_payloads(mutation: Any) -> None:
    value = json.loads(FIXTURE.read_text())
    mutation(value)

    with pytest.raises((ValueError, TypeError)):
        _validate(value)
