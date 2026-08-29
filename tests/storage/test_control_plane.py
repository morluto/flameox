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
from flameox.storage.control_plane import ControlPlane

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


def test_control_plane_uses_one_durable_format_sentinel(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    with sqlite3.connect(workspace.paths.control_plane) as connection:
        assert connection.execute("SELECT format FROM control_plane_format").fetchall() == [
            (ControlPlane.FORMAT,)
        ]


def test_control_plane_refuses_an_incompatible_durable_format(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    with sqlite3.connect(workspace.paths.control_plane) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("UPDATE control_plane_format SET format = 'other.control-plane'")

    with pytest.raises(DomainError) as error:
        ControlPlane(workspace).initialize()

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
    assert error.value.details == {
        "required_format": ControlPlane.FORMAT,
        "stored_format": "other.control-plane",
    }


def test_control_plane_rejects_preexisting_tables_without_the_current_sentinel(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    with sqlite3.connect(workspace.paths.control_plane) as connection:
        connection.executescript(
            """
            DROP TABLE control_plane_format;
            CREATE TABLE control_plane_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;
            INSERT INTO control_plane_metadata(key, value) VALUES('schema_version', '5');
            """
        )

    with pytest.raises(DomainError) as error:
        ControlPlane(workspace).initialize()

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
    assert error.value.details == {"required_format": ControlPlane.FORMAT}


def test_control_plane_has_no_table_local_version_or_legacy_ownership_columns(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    with sqlite3.connect(workspace.paths.control_plane) as connection:
        table_columns = {
            table: {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            for table in (
                "cursors",
                "projection_intents",
                "relationships",
                "record_revision_relationships",
                "record_revision_relationship_sets",
            )
        }

    assert "schema_version" not in table_columns["cursors"]
    assert "projection_schema_version" not in table_columns["projection_intents"]
    for columns in (
        table_columns["relationships"],
        table_columns["record_revision_relationships"],
        table_columns["record_revision_relationship_sets"],
    ):
        assert "ownership_quality" not in columns


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
