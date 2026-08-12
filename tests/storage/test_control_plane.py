from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from flameox.domain import DomainError, ErrorCode
from flameox.domain.models import utc_now
from flameox.models import ContractModel
from flameox.storage import AuthorizedPlanStore, Workspace
from flameox.storage.control_plane import ControlPlane


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
    assert reused.value.code is ErrorCode.INVALID_CAPTURE_PLAN


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
