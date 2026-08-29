from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from flameox.domain import DomainError, ErrorCode
from flameox.domain.models import utc_now
from flameox.models import ContractModel
from flameox.storage import (
    AuthorizedPlanStore,
    CompletedRetentionIntent,
    RetentionIntentStore,
    Workspace,
)
from flameox.storage.control_plane import ControlPlane, ControlRelationship

pytestmark = [pytest.mark.integration, pytest.mark.serial]


class ExampleIntent(ContractModel):
    command: tuple[str, ...]


class CapabilityIntent(ContractModel):
    plan_token: str
    command: tuple[str, ...]


def test_authorized_plan_is_durable_and_atomically_single_use(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    first = AuthorizedPlanStore(workspace, family="example", model=ExampleIntent)
    second = AuthorizedPlanStore(workspace, family="example", model=ExampleIntent)
    intent = ExampleIntent(command=("tool", "--bounded"))
    expires_at = utc_now() + timedelta(minutes=5)

    first.issue("opaque-token", "content-digest", intent, expires_at=expires_at)

    with sqlite3.connect(workspace.paths.control_plane) as connection:
        stored_payload = connection.execute(
            "SELECT payload_json FROM authorized_plans WHERE token = ?",
            ("opaque-token",),
        ).fetchone()
    assert stored_payload is not None
    assert "opaque-token" not in stored_payload[0]

    assert second.inspect("opaque-token") == intent
    assert second.consume("opaque-token", expected_digest="content-digest") == intent
    with pytest.raises(DomainError) as reused:
        first.consume("opaque-token", expected_digest="content-digest")
    assert reused.value.code is ErrorCode.PLAN_TOKEN_CONSUMED


def test_authorized_plan_reports_unknown_and_expired_states(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    plans = AuthorizedPlanStore(workspace, family="example", model=ExampleIntent)

    with pytest.raises(DomainError) as unknown:
        plans.consume("unknown-token")
    assert unknown.value.code is ErrorCode.PLAN_TOKEN_UNKNOWN

    plans.issue(
        "expired-token",
        "intent-digest",
        ExampleIntent(command=("tool",)),
        expires_at=utc_now() - timedelta(seconds=1),
    )
    with pytest.raises(DomainError) as expired:
        plans.consume("expired-token")
    assert expired.value.code is ErrorCode.PLAN_TOKEN_EXPIRED


def test_authorized_plan_rejects_same_token_with_different_intent(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    plans = AuthorizedPlanStore(workspace, family="example", model=ExampleIntent)
    expires_at = utc_now() + timedelta(minutes=5)
    plans.issue(
        "opaque-token",
        "first-digest",
        ExampleIntent(command=("first",)),
        expires_at=expires_at,
    )

    with pytest.raises(DomainError) as conflict:
        plans.issue(
            "opaque-token",
            "second-digest",
            ExampleIntent(command=("second",)),
            expires_at=expires_at,
        )
    assert conflict.value.code is ErrorCode.REVISION_CONFLICT


def test_plan_capability_is_not_stored_inside_authorized_intent(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    plans = AuthorizedPlanStore(workspace, family="capability", model=CapabilityIntent)
    intent = CapabilityIntent(plan_token="opaque-capability", command=("tool",))

    plans.issue(
        intent.plan_token,
        "intent-digest",
        intent,
        expires_at=utc_now() + timedelta(minutes=5),
    )

    with sqlite3.connect(workspace.paths.control_plane) as connection:
        payload = connection.execute(
            "SELECT payload_json FROM authorized_plans WHERE token = ?",
            (intent.plan_token,),
        ).fetchone()
    assert payload is not None
    assert "plan_token" not in payload[0]
    assert "opaque-capability" not in payload[0]
    assert plans.inspect(intent.plan_token) == intent


def test_control_plane_refuses_unknown_future_schema(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    with sqlite3.connect(workspace.paths.control_plane) as connection:
        connection.execute(
            "UPDATE control_plane_metadata SET value = '999' WHERE key = 'schema_version'"
        )

    with pytest.raises(DomainError) as error:
        ControlPlane(workspace).initialize()

    assert error.value.code is ErrorCode.WORKSPACE_INVALID


def test_control_plane_refuses_older_schema_instead_of_migrating_it(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    with sqlite3.connect(workspace.paths.control_plane) as connection:
        connection.execute(
            "UPDATE control_plane_metadata SET value = '1' WHERE key = 'schema_version'"
        )

    with pytest.raises(DomainError) as error:
        ControlPlane(workspace).initialize()

    assert error.value.code is ErrorCode.WORKSPACE_INVALID


def test_control_plane_migrates_v2_relationships_without_guessing_history(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    control = ControlPlane(workspace)
    control.create_record(
        kind="findings",
        record_id="finding-1",
        revision=1,
        payload_json='{"revision":1}',
        relationships=(
            ControlRelationship(
                relationship="supports",
                target_kind="analysis",
                target_id="analysis-1",
            ),
        ),
    )
    with sqlite3.connect(workspace.paths.control_plane) as connection:
        connection.executescript(
            """
            DROP TABLE record_revision_relationships;
            DROP TABLE record_revision_relationship_sets;
            ALTER TABLE relationships RENAME TO relationships_v3;
            CREATE TABLE relationships (
                source_kind TEXT NOT NULL,
                source_id TEXT NOT NULL,
                relationship TEXT NOT NULL,
                target_kind TEXT NOT NULL,
                target_id TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_kind, source_id, relationship, target_kind, target_id)
            ) STRICT;
            INSERT INTO relationships(
                source_kind, source_id, relationship, target_kind, target_id,
                payload_json, created_at
            )
            SELECT source_kind, source_id, relationship, target_kind, target_id,
                   payload_json, created_at
            FROM relationships_v3;
            DROP TABLE relationships_v3;
            UPDATE control_plane_metadata SET value = '2' WHERE key = 'schema_version';
            """
        )

    ControlPlane(workspace).initialize()

    current = control.list_relationships(
        source_kind="findings",
        source_id="finding-1",
    )
    assert current[0].ownership_quality == "legacy_current_only"
    with pytest.raises(DomainError) as ambiguous:
        control.list_revision_relationships(
            source_kind="findings",
            source_id="finding-1",
            source_revision=1,
        )
    assert ambiguous.value.details["ownership_quality"] == "legacy_current_only"


def test_retention_intent_durably_bridges_snapshot_to_materialization(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    commit_id = workspace.corpus.read_head().commit_id
    store = RetentionIntentStore(workspace)

    first = store.acquire(
        corpus_commit_id=commit_id,
        owner_kind="analysis",
        owner_id="analysis-1",
        operation_digest="operation-1",
    )
    repeated = store.acquire(
        corpus_commit_id=commit_id,
        owner_kind="analysis",
        owner_id="analysis-1",
        operation_digest="operation-1",
    )

    assert repeated == first
    assert store.pending_commit_ids() == (commit_id,)
    completed = store.complete(first, materialized_commit_id="materialized-1")
    assert isinstance(completed, CompletedRetentionIntent)
    assert store.pending_commit_ids() == ()
    assert store.complete(completed, materialized_commit_id="materialized-1") == completed
