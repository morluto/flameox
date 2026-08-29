from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from flameox.domain import (
    CURSOR_POSITION_SPECS,
    CursorNamespace,
    DomainError,
    ErrorCode,
)
from flameox.storage import CursorStore, Workspace

pytestmark = [pytest.mark.integration, pytest.mark.serial]

_TOKEN = "A" * 43
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _store(workspace: Workspace, *, now: datetime = _NOW, token: str = _TOKEN) -> CursorStore:
    return CursorStore(workspace, clock=lambda: now, token_factory=lambda: token)


def _issue(store: CursorStore) -> str:
    return store.issue(
        namespace=CursorNamespace.RUNS,
        snapshot_id="commit-1",
        scope_digest="scope-1",
        position=("2026-01-01T00:00:00+00:00", "run-1"),
    )


@pytest.fixture(scope="module")
def cursor_workspace(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    # The property test below performs only rejected reads, so sharing this
    # initialized workspace cannot leak generated state between examples.
    return Workspace.initialize(tmp_path_factory.mktemp("cursor-fuzz"))


def test_cursor_is_an_opaque_workspace_scoped_handle(tmp_path: Path) -> None:
    first_workspace = Workspace.initialize(tmp_path / "first")
    second_workspace = Workspace.initialize(tmp_path / "second")
    first = _store(first_workspace)
    token = _issue(first)

    assert token == _TOKEN
    assert "run-1" not in token
    with sqlite3.connect(first_workspace.paths.control_plane) as connection:
        row = connection.execute("SELECT cursor_digest, position_json FROM cursors").fetchone()
        columns = {item[1] for item in connection.execute("PRAGMA table_info(cursors)")}
    assert row is not None
    assert "schema_version" not in columns
    assert row[0] != token
    assert json.loads(row[1]) == ["2026-01-01T00:00:00+00:00", "run-1"]
    assert first.resolve(
        token,
        namespace=CursorNamespace.RUNS,
        snapshot_id="commit-1",
        scope_digest="scope-1",
    ) == ("2026-01-01T00:00:00+00:00", "run-1")

    with pytest.raises(DomainError) as cross_workspace:
        _store(second_workspace).resolve(
            token,
            namespace=CursorNamespace.RUNS,
            snapshot_id="commit-1",
            scope_digest="scope-1",
        )
    assert cross_workspace.value.code is ErrorCode.STALE_CURSOR


@pytest.mark.parametrize(
    ("namespace", "snapshot_id", "scope_digest"),
    [
        (CursorNamespace.ARTIFACTS, "commit-1", "scope-1"),
        (CursorNamespace.RUNS, "commit-2", "scope-1"),
        (CursorNamespace.RUNS, "commit-1", "scope-2"),
    ],
)
def test_cursor_replay_fails_across_boundaries(
    tmp_path: Path,
    namespace: CursorNamespace,
    snapshot_id: str,
    scope_digest: str,
) -> None:
    store = _store(Workspace.initialize(tmp_path))
    token = _issue(store)

    with pytest.raises(DomainError) as error:
        store.resolve(
            token,
            namespace=namespace,
            snapshot_id=snapshot_id,
            scope_digest=scope_digest,
        )

    assert error.value.code is ErrorCode.STALE_CURSOR


def test_modified_expired_and_revoked_handles_fail_closed(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    store = _store(workspace)
    modified_token = _TOKEN[:-1] + "B"
    token = _issue(store)

    with pytest.raises(DomainError):
        store.resolve(
            modified_token,
            namespace=CursorNamespace.RUNS,
            snapshot_id="commit-1",
            scope_digest="scope-1",
        )
    with pytest.raises(DomainError) as expired:
        _store(workspace, now=_NOW + timedelta(minutes=16)).resolve(
            token,
            namespace=CursorNamespace.RUNS,
            snapshot_id="commit-1",
            scope_digest="scope-1",
        )
    assert expired.value.code is ErrorCode.STALE_CURSOR

    assert store.revoke(token)
    with pytest.raises(DomainError) as revoked:
        store.resolve(
            token,
            namespace=CursorNamespace.RUNS,
            snapshot_id="commit-1",
            scope_digest="scope-1",
        )
    assert revoked.value.code is ErrorCode.STALE_CURSOR


@given(
    st.text(max_size=256).filter(
        lambda value: value != _TOKEN and not (len(value) == 43 and value.isascii())
    )
)
def test_malformed_public_handles_never_reach_a_decoder(
    cursor_workspace: Workspace,
    value: str,
) -> None:
    with pytest.raises(DomainError) as error:
        _store(cursor_workspace).resolve(
            value,
            namespace=CursorNamespace.RUNS,
            snapshot_id="commit-1",
            scope_digest="scope-1",
        )

    assert error.value.code is ErrorCode.STALE_CURSOR


def test_cursor_registry_is_closed_and_validates_bool_as_int(tmp_path: Path) -> None:
    assert set(CURSOR_POSITION_SPECS) == set(CursorNamespace)
    store = _store(Workspace.initialize(tmp_path))

    with pytest.raises(DomainError) as error:
        store.issue(
            namespace=CursorNamespace.INVESTIGATIONS,
            snapshot_id="commit-1",
            scope_digest="all",
            position=(True,),
        )

    assert error.value.code is ErrorCode.STALE_CURSOR


def test_corrupt_persisted_position_is_rejected_before_query_use(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    store = _store(workspace)
    token = _issue(store)
    with sqlite3.connect(workspace.paths.control_plane) as connection:
        connection.execute(
            "UPDATE cursors SET position_json = ?",
            ('{"unexpected":"shape"}',),
        )

    with pytest.raises(DomainError) as error:
        store.resolve(
            token,
            namespace=CursorNamespace.RUNS,
            snapshot_id="commit-1",
            scope_digest="scope-1",
        )

    assert error.value.code is ErrorCode.STALE_CURSOR
