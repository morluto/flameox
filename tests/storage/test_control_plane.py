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


def test_authorized_plan_is_durable_and_atomically_single_use(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    first = AuthorizedPlanStore(workspace, family="example", model=ExampleIntent)
    second = AuthorizedPlanStore(workspace, family="example", model=ExampleIntent)
    intent = ExampleIntent(command=("tool", "--bounded"))
    expires_at = utc_now() + timedelta(minutes=5)

    first.issue("opaque-token", "content-digest", intent, expires_at=expires_at)

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


def test_control_plane_refuses_unknown_future_schema(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    with sqlite3.connect(workspace.paths.control_plane) as connection:
        connection.execute(
            "UPDATE control_plane_metadata SET value = '999' WHERE key = 'schema_version'"
        )

    with pytest.raises(DomainError) as error:
        ControlPlane(workspace).initialize()

    assert error.value.code is ErrorCode.WORKSPACE_INVALID
