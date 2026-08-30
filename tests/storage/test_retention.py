from __future__ import annotations

import json
from pathlib import Path

from flameox.storage import RetentionIntentStore, Workspace
from flameox.storage.control_plane import ControlPlane


def test_retention_payload_does_not_duplicate_control_plane_revision(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    store = RetentionIntentStore(workspace)
    intent = store.acquire(
        corpus_commit_id=workspace.corpus.read_head().commit_id,
        owner_kind="test",
        owner_id="owner",
        operation_digest="sha256:" + "1" * 64,
    )

    control = ControlPlane(workspace)
    assert "revision" not in json.loads(
        control.read_record(kind="retention_intents", record_id=intent.intent_id)
    )

    completed = store.complete(intent, materialized_commit_id="sha256:" + "2" * 64)
    assert "revision" not in json.loads(
        control.read_record(kind="retention_intents", record_id=completed.intent_id)
    )
