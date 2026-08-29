from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, cast

from flameox.domain import DomainError, ErrorCode
from flameox.domain.models import utc_now
from flameox.domain.projections import ProjectionIntent, ProjectionIntentSpec, ProjectionState

if TYPE_CHECKING:
    from flameox.storage.workspace import Workspace


@dataclass(frozen=True, slots=True)
class ControlRelationship:
    relationship: str
    target_kind: str
    target_id: str
    payload_json: str = "{}"


@dataclass(frozen=True, slots=True)
class CursorControlRecord:
    workspace_id: str
    namespace: str
    snapshot_id: str
    scope_digest: str
    position_json: str
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None


_SCHEMA = (
    """
    CREATE TABLE control_plane_format (
        format TEXT PRIMARY KEY CHECK (format = 'flameox.control-plane')
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS authorized_plans (
        token TEXT PRIMARY KEY,
        family TEXT NOT NULL,
        intent_digest TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        consumed_at TEXT
    ) STRICT
    """,
    """
    CREATE INDEX IF NOT EXISTS authorized_plans_expiry
    ON authorized_plans(family, expires_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS cursors (
        cursor_digest TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        namespace TEXT NOT NULL,
        snapshot_id TEXT NOT NULL,
        scope_digest TEXT NOT NULL,
        position_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        revoked_at TEXT
    ) STRICT
    """,
    """
    CREATE INDEX IF NOT EXISTS cursors_expiry
    ON cursors(expires_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS operations (
        operation_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        state TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision >= 0),
        plan_token TEXT REFERENCES authorized_plans(token),
        run_id TEXT,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS operations_kind_run_id
    ON operations(kind, run_id) WHERE run_id IS NOT NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS capture_admissions (
        run_id TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        process_id INTEGER NOT NULL,
        process_start_identity TEXT NOT NULL,
        boot_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        acquired_at TEXT NOT NULL,
        heartbeat_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE INDEX IF NOT EXISTS capture_admissions_expiry
    ON capture_admissions(expires_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS staging_owners (
        path TEXT PRIMARY KEY,
        owner_kind TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('active', 'released')),
        process_id INTEGER NOT NULL,
        process_start_identity TEXT NOT NULL,
        boot_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE INDEX IF NOT EXISTS staging_owners_state
    ON staging_owners(state, updated_at)
    """,
    """
    CREATE TABLE IF NOT EXISTS operation_revisions (
        operation_id TEXT NOT NULL REFERENCES operations(operation_id) ON DELETE CASCADE,
        revision INTEGER NOT NULL CHECK (revision >= 0),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (operation_id, revision)
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS idempotency_keys (
        scope TEXT NOT NULL,
        key_digest TEXT NOT NULL,
        intent_digest TEXT NOT NULL,
        target_kind TEXT NOT NULL,
        target_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (scope, key_digest)
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        run_type TEXT NOT NULL,
        current_revision INTEGER NOT NULL CHECK (current_revision >= 0),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS run_revisions (
        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        revision INTEGER NOT NULL CHECK (revision >= 0),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (run_id, revision)
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS projection_intents (
        intent_id TEXT PRIMARY KEY,
        domain_kind TEXT NOT NULL,
        domain_id TEXT NOT NULL,
        domain_revision INTEGER NOT NULL CHECK (domain_revision >= 0),
        domain_digest TEXT NOT NULL,
        projection_kind TEXT NOT NULL,
        operation_digest TEXT NOT NULL UNIQUE,
        spec_json TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('pending', 'published', 'failed')),
        generation_id TEXT,
        corpus_commit_id TEXT,
        failure_code TEXT,
        failure_message TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (
            domain_kind, domain_id, domain_revision,
            projection_kind
        ),
        CHECK (
            (state = 'pending' AND generation_id IS NULL AND corpus_commit_id IS NULL
                AND failure_code IS NULL AND failure_message IS NULL)
            OR
            (state = 'published' AND generation_id IS NOT NULL AND corpus_commit_id IS NOT NULL
                AND failure_code IS NULL AND failure_message IS NULL)
            OR
            (state = 'failed' AND generation_id IS NULL AND corpus_commit_id IS NULL
                AND failure_code IS NOT NULL AND failure_message IS NOT NULL)
        )
    ) STRICT
    """,
    """
    CREATE INDEX IF NOT EXISTS projection_intents_state
    ON projection_intents(state, updated_at, intent_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS projection_intents_domain
    ON projection_intents(domain_kind, domain_id, domain_revision DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS records (
        kind TEXT NOT NULL,
        record_id TEXT NOT NULL,
        current_revision INTEGER,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (kind, record_id)
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS record_revisions (
        kind TEXT NOT NULL,
        record_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision >= 0),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (kind, record_id, revision),
        FOREIGN KEY (kind, record_id) REFERENCES records(kind, record_id) ON DELETE CASCADE
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS relationships (
        source_kind TEXT NOT NULL,
        source_id TEXT NOT NULL,
        relationship TEXT NOT NULL,
        target_kind TEXT NOT NULL,
        target_id TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        PRIMARY KEY (source_kind, source_id, relationship, target_kind, target_id)
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS record_revision_relationships (
        source_kind TEXT NOT NULL,
        source_id TEXT NOT NULL,
        source_revision INTEGER NOT NULL CHECK (source_revision >= 0),
        relationship TEXT NOT NULL,
        target_kind TEXT NOT NULL,
        target_id TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        PRIMARY KEY (
            source_kind, source_id, source_revision, relationship, target_kind, target_id
        ),
        FOREIGN KEY (source_kind, source_id, source_revision)
            REFERENCES record_revisions(kind, record_id, revision) ON DELETE CASCADE
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS record_revision_relationship_sets (
        source_kind TEXT NOT NULL,
        source_id TEXT NOT NULL,
        source_revision INTEGER NOT NULL CHECK (source_revision >= 0),
        created_at TEXT NOT NULL,
        PRIMARY KEY (source_kind, source_id, source_revision),
        FOREIGN KEY (source_kind, source_id, source_revision)
            REFERENCES record_revisions(kind, record_id, revision) ON DELETE CASCADE
    ) STRICT
    """,
)


class ControlPlane:
    """The transactional authority for mutable workspace control state."""

    FORMAT = "flameox.control-plane"
    MAX_OPERATION_REVISIONS = 64
    MAX_PROJECTION_SPEC_BYTES = 256 * 1024

    def __init__(self, workspace: Workspace) -> None:
        self.path = workspace.paths.control_plane

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                stored_format = self._stored_format(connection)
                if stored_format is None:
                    if self._has_user_tables(connection):
                        self._raise_incompatible_format(stored_format)
                    for statement in _SCHEMA:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO control_plane_format(format) VALUES (?)",
                        (self.FORMAT,),
                    )
                elif stored_format != self.FORMAT:
                    self._raise_incompatible_format(stored_format)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        with suppress(OSError):
            os.chmod(self.path, 0o600)

    @staticmethod
    def _has_user_tables(connection: sqlite3.Connection) -> bool:
        return (
            connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                LIMIT 1
                """
            ).fetchone()
            is not None
        )

    @classmethod
    def _stored_format(cls, connection: sqlite3.Connection) -> str | None:
        table_exists = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'control_plane_format'
            """
        ).fetchone()
        if table_exists is None:
            return None
        rows = connection.execute("SELECT format FROM control_plane_format").fetchall()
        if len(rows) != 1:
            cls._raise_incompatible_format(None)
        return str(rows[0][0])

    @classmethod
    def _raise_incompatible_format(cls, stored_format: str | None) -> None:
        details = {"required_format": cls.FORMAT}
        if stored_format is not None:
            details["stored_format"] = stored_format
        raise DomainError(
            ErrorCode.WORKSPACE_INVALID,
            "The SQLite control plane has an incompatible durable format. Create a new "
            "workspace; Flameox does not migrate control-plane files.",
            details=details,
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def issue_plan(
        self,
        *,
        token: str,
        family: str,
        intent_digest: str,
        payload_json: str,
        expires_at: datetime,
    ) -> None:
        created_at = utc_now().isoformat()
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO authorized_plans(
                        token, family, intent_digest, payload_json, created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        token,
                        family,
                        intent_digest,
                        payload_json,
                        created_at,
                        expires_at.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "An authorized plan token already exists.",
            ) from exc

    def inspect_plan(self, *, token: str, family: str) -> tuple[str, str]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT intent_digest, payload_json, expires_at, consumed_at
                FROM authorized_plans WHERE token = ? AND family = ?
                """,
                (token, family),
            ).fetchone()
        return self._available_plan(row)

    def consume_plan(
        self,
        *,
        token: str,
        family: str,
        expected_digest: str | None,
    ) -> tuple[str, str]:
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT intent_digest, payload_json, expires_at, consumed_at
                FROM authorized_plans WHERE token = ? AND family = ?
                """,
                (token, family),
            ).fetchone()
            intent_digest, payload_json = self._available_plan(row)
            if expected_digest is not None and intent_digest != expected_digest:
                raise DomainError(
                    ErrorCode.PLAN_ID_MISMATCH,
                    "The authorized plan digest does not match the expected intent.",
                    details={
                        "expected_plan_id": expected_digest,
                        "actual_plan_id": intent_digest,
                    },
                    remediation=(
                        "Review the stored plan identity and retry with that expected plan ID.",
                    ),
                )
            changed = connection.execute(
                """
                UPDATE authorized_plans SET consumed_at = ?
                WHERE token = ? AND family = ? AND consumed_at IS NULL
                """,
                (utc_now().isoformat(), token, family),
            ).rowcount
            if changed != 1:
                raise self._unavailable_plan()
            return intent_digest, payload_json

    def issue_cursor(
        self,
        *,
        cursor_digest: str,
        workspace_id: str,
        namespace: str,
        snapshot_id: str,
        scope_digest: str,
        position_json: str,
        created_at: datetime,
        expires_at: datetime,
    ) -> None:
        try:
            with self.transaction() as connection:
                self._purge_cursors(connection, observed_at=created_at, limit=128)
                connection.execute(
                    """
                    INSERT INTO cursors(
                        cursor_digest, workspace_id, namespace,
                        snapshot_id, scope_digest, position_json, created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cursor_digest,
                        workspace_id,
                        namespace,
                        snapshot_id,
                        scope_digest,
                        position_json,
                        created_at.isoformat(),
                        expires_at.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "A cursor handle collision occurred.",
                retryable=True,
            ) from exc

    def read_cursor(
        self,
        *,
        cursor_digest: str,
        workspace_id: str,
    ) -> CursorControlRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT workspace_id, namespace, snapshot_id, scope_digest,
                       position_json, created_at, expires_at, revoked_at
                FROM cursors
                WHERE cursor_digest = ? AND workspace_id = ?
                """,
                (cursor_digest, workspace_id),
            ).fetchone()
        if row is None:
            return None
        return CursorControlRecord(
            workspace_id=str(row["workspace_id"]),
            namespace=str(row["namespace"]),
            snapshot_id=str(row["snapshot_id"]),
            scope_digest=str(row["scope_digest"]),
            position_json=str(row["position_json"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            expires_at=datetime.fromisoformat(str(row["expires_at"])),
            revoked_at=(
                datetime.fromisoformat(str(row["revoked_at"]))
                if row["revoked_at"] is not None
                else None
            ),
        )

    def active_cursor_snapshot_ids(
        self,
        *,
        workspace_id: str,
        namespace: str,
        observed_at: datetime,
    ) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT snapshot_id
                FROM cursors
                WHERE workspace_id = ? AND namespace = ?
                  AND revoked_at IS NULL AND expires_at > ?
                ORDER BY snapshot_id
                """,
                (workspace_id, namespace, observed_at.isoformat()),
            ).fetchall()
        return tuple(str(row["snapshot_id"]) for row in rows)

    def revoke_cursor(
        self,
        *,
        cursor_digest: str,
        workspace_id: str,
        revoked_at: datetime,
    ) -> bool:
        with self.transaction() as connection:
            changed = connection.execute(
                """
                UPDATE cursors SET revoked_at = ?
                WHERE cursor_digest = ? AND workspace_id = ? AND revoked_at IS NULL
                """,
                (revoked_at.isoformat(), cursor_digest, workspace_id),
            ).rowcount
        return changed == 1

    def purge_cursors(self, *, observed_at: datetime, limit: int = 128) -> int:
        with self.transaction() as connection:
            return self._purge_cursors(connection, observed_at=observed_at, limit=limit)

    def create_record(
        self,
        *,
        kind: str,
        record_id: str,
        revision: int | None,
        payload_json: str,
        relationships: tuple[ControlRelationship, ...] = (),
    ) -> None:
        observed_at = utc_now().isoformat()
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO records(
                        kind, record_id, current_revision, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (kind, record_id, revision, payload_json, observed_at, observed_at),
                )
                if revision is not None:
                    connection.execute(
                        """
                        INSERT INTO record_revisions(
                            kind, record_id, revision, payload_json, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (kind, record_id, revision, payload_json, observed_at),
                    )
                self._replace_relationships(
                    connection,
                    source_kind=kind,
                    source_id=record_id,
                    source_revision=revision,
                    relationships=relationships,
                    observed_at=observed_at,
                )
        except sqlite3.IntegrityError as exc:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                f"{kind} {record_id!r} already exists.",
            ) from exc

    def read_record(self, *, kind: str, record_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM records WHERE kind = ? AND record_id = ?",
                (kind, record_id),
            ).fetchone()
        if row is None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"{kind} {record_id!r} does not exist or is invalid.",
            )
        return str(row["payload_json"])

    def list_records(self, *, kind: str) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM records WHERE kind = ? ORDER BY record_id",
                (kind,),
            ).fetchall()
        return tuple(str(row["payload_json"]) for row in rows)

    def list_relationships(
        self,
        *,
        source_kind: str,
        source_id: str,
    ) -> tuple[ControlRelationship, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT relationship, target_kind, target_id, payload_json
                FROM relationships
                WHERE source_kind = ? AND source_id = ?
                ORDER BY relationship, target_kind, target_id
                """,
                (source_kind, source_id),
            ).fetchall()
        return tuple(
            ControlRelationship(
                relationship=str(row["relationship"]),
                target_kind=str(row["target_kind"]),
                target_id=str(row["target_id"]),
                payload_json=str(row["payload_json"]),
            )
            for row in rows
        )

    def list_revision_relationships(
        self,
        *,
        source_kind: str,
        source_id: str,
        source_revision: int,
    ) -> tuple[ControlRelationship, ...]:
        with self._connect() as connection:
            relationship_set = connection.execute(
                """
                SELECT 1 FROM record_revision_relationship_sets
                WHERE source_kind = ? AND source_id = ? AND source_revision = ?
                """,
                (source_kind, source_id, source_revision),
            ).fetchone()
            if relationship_set is None:
                raise DomainError(
                    ErrorCode.WORKSPACE_INVALID,
                    "The requested record revision has no exact relationship-set authority.",
                    details={
                        "source_kind": source_kind,
                        "source_id": source_id,
                        "source_revision": source_revision,
                    },
                )
            rows = connection.execute(
                """
                SELECT relationship, target_kind, target_id, payload_json
                FROM record_revision_relationships
                WHERE source_kind = ? AND source_id = ? AND source_revision = ?
                ORDER BY relationship, target_kind, target_id
                """,
                (source_kind, source_id, source_revision),
            ).fetchall()
        return tuple(
            ControlRelationship(
                relationship=str(row["relationship"]),
                target_kind=str(row["target_kind"]),
                target_id=str(row["target_id"]),
                payload_json=str(row["payload_json"]),
            )
            for row in rows
        )

    def append_record(
        self,
        *,
        kind: str,
        record_id: str,
        expected_revision: int,
        next_revision: int,
        payload_json: str,
        relationships: tuple[ControlRelationship, ...] | None = None,
    ) -> None:
        if next_revision != expected_revision + 1:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                f"{kind} {record_id!r} has a stale expected revision.",
                details={
                    "expected_revision": expected_revision,
                    "supplied_next_revision": next_revision,
                },
            )
        observed_at = utc_now().isoformat()
        try:
            with self.transaction() as connection:
                changed = connection.execute(
                    """
                    UPDATE records
                    SET current_revision = ?, payload_json = ?, updated_at = ?
                    WHERE kind = ? AND record_id = ? AND current_revision = ?
                    """,
                    (
                        next_revision,
                        payload_json,
                        observed_at,
                        kind,
                        record_id,
                        expected_revision,
                    ),
                ).rowcount
                if changed != 1:
                    actual_row = connection.execute(
                        "SELECT current_revision FROM records WHERE kind = ? AND record_id = ?",
                        (kind, record_id),
                    ).fetchone()
                    actual = actual_row["current_revision"] if actual_row is not None else None
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        f"{kind} {record_id!r} changed before the update.",
                        retryable=True,
                        details={
                            "expected_revision": expected_revision,
                            "actual_revision": actual,
                        },
                    )
                connection.execute(
                    """
                    INSERT INTO record_revisions(
                        kind, record_id, revision, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (kind, record_id, next_revision, payload_json, observed_at),
                )
                if relationships is not None:
                    self._replace_relationships(
                        connection,
                        source_kind=kind,
                        source_id=record_id,
                        source_revision=next_revision,
                        relationships=relationships,
                        observed_at=observed_at,
                    )
        except sqlite3.IntegrityError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "An immutable record revision already contains different data.",
            ) from exc

    def create_run(
        self,
        *,
        run_id: str,
        run_type: str,
        revision: int,
        payload_json: str,
        projection_intent: ProjectionIntentSpec | None = None,
    ) -> None:
        observed_at = utc_now().isoformat()
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO runs(
                        run_id, run_type, current_revision, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, run_type, revision, payload_json, observed_at, observed_at),
                )
                connection.execute(
                    """
                    INSERT INTO run_revisions(run_id, revision, payload_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, revision, payload_json, observed_at),
                )
                if projection_intent is not None:
                    self._require_run_projection_binding(
                        projection_intent,
                        run_id=run_id,
                        revision=revision,
                    )
                    self._insert_projection_intent(
                        connection,
                        projection_intent,
                        observed_at=observed_at,
                    )
        except sqlite3.IntegrityError as exc:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                f"Run {run_id!r} already exists.",
            ) from exc

    def read_run(self, run_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise DomainError(
                ErrorCode.RUN_NOT_FOUND,
                f"Run {run_id!r} does not exist.",
                remediation=("Call list_runs to choose an existing run.",),
                details={"missing_entity": "run"},
            )
        return str(row["payload_json"])

    def read_run_revision(self, run_id: str, revision: int) -> str:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM run_revisions
                WHERE run_id = ? AND revision = ?
                """,
                (run_id, revision),
            ).fetchone()
        if row is None:
            raise DomainError(
                ErrorCode.RUN_NOT_FOUND,
                f"Run {run_id!r} has no revision {revision}.",
                details={"missing_entity": "run_revision", "revision": revision},
            )
        return str(row["payload_json"])

    def list_runs(self) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM runs ORDER BY created_at, run_id"
            ).fetchall()
        return tuple(str(row["payload_json"]) for row in rows)

    def append_run(
        self,
        *,
        run_id: str,
        run_type: str,
        expected_revision: int,
        next_revision: int,
        payload_json: str,
        projection_intent: ProjectionIntentSpec | None = None,
    ) -> None:
        if next_revision != expected_revision + 1:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "The next run revision is not consecutive.",
            )
        observed_at = utc_now().isoformat()
        try:
            with self.transaction() as connection:
                changed = connection.execute(
                    """
                    UPDATE runs
                    SET current_revision = ?, payload_json = ?, updated_at = ?
                    WHERE run_id = ? AND run_type = ? AND current_revision = ?
                    """,
                    (
                        next_revision,
                        payload_json,
                        observed_at,
                        run_id,
                        run_type,
                        expected_revision,
                    ),
                ).rowcount
                if changed != 1:
                    actual_row = connection.execute(
                        "SELECT current_revision, run_type FROM runs WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()
                    if actual_row is not None and actual_row["run_type"] != run_type:
                        raise DomainError(
                            ErrorCode.WORKSPACE_INVALID,
                            "A run revision cannot change its run type.",
                        )
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        f"Run {run_id!r} changed before the update.",
                        retryable=True,
                        details={
                            "expected_revision": expected_revision,
                            "actual_revision": (
                                actual_row["current_revision"] if actual_row is not None else None
                            ),
                        },
                    )
                connection.execute(
                    """
                    INSERT INTO run_revisions(run_id, revision, payload_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, next_revision, payload_json, observed_at),
                )
                if projection_intent is not None:
                    self._require_run_projection_binding(
                        projection_intent,
                        run_id=run_id,
                        revision=next_revision,
                    )
                    self._insert_projection_intent(
                        connection,
                        projection_intent,
                        observed_at=observed_at,
                    )
        except sqlite3.IntegrityError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "An immutable run revision already contains different data.",
            ) from exc

    def create_projection_intent(self, spec: ProjectionIntentSpec) -> ProjectionIntent:
        observed_at = utc_now().isoformat()
        with self.transaction() as connection:
            self._insert_projection_intent(connection, spec, observed_at=observed_at)
            row = self._projection_intent_row(connection, spec.intent_id)
        return self._projection_intent(row)

    def read_projection_intent(self, intent_id: str) -> ProjectionIntent:
        with self._connect() as connection:
            row = self._projection_intent_row(connection, intent_id)
        return self._projection_intent(row)

    def list_projection_intents(
        self,
        *,
        state: ProjectionState | None = None,
    ) -> tuple[ProjectionIntent, ...]:
        with self._connect() as connection:
            if state is None:
                rows = connection.execute(
                    "SELECT * FROM projection_intents ORDER BY created_at, intent_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM projection_intents
                    WHERE state = ? ORDER BY created_at, intent_id
                    """,
                    (state.value,),
                ).fetchall()
        return tuple(self._projection_intent(row) for row in rows)

    def latest_projection_intent(
        self,
        *,
        domain_kind: str,
        domain_id: str,
        projection_kind: str,
    ) -> ProjectionIntent | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM projection_intents
                WHERE domain_kind = ? AND domain_id = ? AND projection_kind = ?
                ORDER BY domain_revision DESC, created_at DESC LIMIT 1
                """,
                (domain_kind, domain_id, projection_kind),
            ).fetchone()
        return self._projection_intent(row) if row is not None else None

    def mark_projection_published(
        self,
        *,
        intent_id: str,
        generation_id: str,
        corpus_commit_id: str,
    ) -> ProjectionIntent:
        observed_at = utc_now().isoformat()
        with self.transaction() as connection:
            row = self._projection_intent_row(connection, intent_id)
            current = self._projection_intent(row)
            if current.state is ProjectionState.PUBLISHED:
                if (
                    current.generation_id != generation_id
                    or current.corpus_commit_id != corpus_commit_id
                ):
                    raise DomainError(
                        ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                        "A projection intent is already linked to a different publication.",
                    )
                return current
            connection.execute(
                """
                UPDATE projection_intents
                SET state = 'published', generation_id = ?, corpus_commit_id = ?,
                    failure_code = NULL, failure_message = NULL, updated_at = ?
                WHERE intent_id = ?
                """,
                (generation_id, corpus_commit_id, observed_at, intent_id),
            )
            updated = self._projection_intent_row(connection, intent_id)
        return self._projection_intent(updated)

    def mark_projection_failed(
        self,
        *,
        intent_id: str,
        failure_code: str,
        failure_message: str,
    ) -> ProjectionIntent:
        observed_at = utc_now().isoformat()
        with self.transaction() as connection:
            row = self._projection_intent_row(connection, intent_id)
            current = self._projection_intent(row)
            if current.state is ProjectionState.PUBLISHED:
                return current
            connection.execute(
                """
                UPDATE projection_intents
                SET state = 'failed', generation_id = NULL, corpus_commit_id = NULL,
                    failure_code = ?, failure_message = ?, updated_at = ?
                WHERE intent_id = ?
                """,
                (failure_code, failure_message, observed_at, intent_id),
            )
            updated = self._projection_intent_row(connection, intent_id)
        return self._projection_intent(updated)

    def retry_projection(self, intent_id: str) -> ProjectionIntent:
        observed_at = utc_now().isoformat()
        with self.transaction() as connection:
            row = self._projection_intent_row(connection, intent_id)
            current = self._projection_intent(row)
            if current.state is not ProjectionState.FAILED:
                return current
            connection.execute(
                """
                UPDATE projection_intents
                SET state = 'pending', failure_code = NULL, failure_message = NULL,
                    updated_at = ?
                WHERE intent_id = ? AND state = 'failed'
                """,
                (observed_at, intent_id),
            )
            updated = self._projection_intent_row(connection, intent_id)
        return self._projection_intent(updated)

    def create_operation(
        self,
        *,
        operation_id: str,
        kind: str,
        state: str,
        revision: int,
        idempotency_digest: str,
        intent_digest: str,
        payload_json: str,
        run_id: str | None = None,
    ) -> None:
        observed_at = utc_now().isoformat()
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO operations(
                        operation_id, kind, state, revision, run_id, payload_json, created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        operation_id,
                        kind,
                        state,
                        revision,
                        run_id,
                        payload_json,
                        observed_at,
                        observed_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO operation_revisions(
                        operation_id, revision, payload_json, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (operation_id, revision, payload_json, observed_at),
                )
                connection.execute(
                    """
                    INSERT INTO idempotency_keys(
                        scope, key_digest, intent_digest, target_kind, target_id, created_at
                    ) VALUES (?, ?, ?, 'operation', ?, ?)
                    """,
                    (kind, idempotency_digest, intent_digest, operation_id, observed_at),
                )
        except sqlite3.IntegrityError as exc:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "The idempotency key is already bound to an operation.",
            ) from exc

    def read_operation(self, operation_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        if row is None:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                f"Operation {operation_id!r} does not exist or is invalid.",
            )
        return str(row["payload_json"])

    def find_operation(self, *, kind: str, idempotency_digest: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT operations.payload_json
                FROM idempotency_keys
                JOIN operations ON operations.operation_id = idempotency_keys.target_id
                WHERE idempotency_keys.scope = ? AND idempotency_keys.key_digest = ?
                  AND idempotency_keys.target_kind = 'operation'
                """,
                (kind, idempotency_digest),
            ).fetchone()
        return str(row["payload_json"]) if row is not None else None

    def find_operation_by_run_id(self, *, kind: str, run_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM operations WHERE kind = ? AND run_id = ?",
                (kind, run_id),
            ).fetchone()
        return str(row["payload_json"]) if row is not None else None

    def create_idempotent_record(
        self,
        *,
        kind: str,
        record_id: str,
        revision: int | None,
        payload_json: str,
        idempotency_digest: str,
        intent_digest: str,
    ) -> tuple[str, str, bool]:
        """Atomically bind an idempotency key or return its existing record."""
        observed_at = utc_now().isoformat()
        with self.transaction() as connection:
            existing = connection.execute(
                """
                SELECT intent_digest, target_kind, target_id
                FROM idempotency_keys WHERE scope = ? AND key_digest = ?
                """,
                (kind, idempotency_digest),
            ).fetchone()
            if existing is not None:
                if existing["target_kind"] != "record":
                    raise DomainError(
                        ErrorCode.REVISION_CONFLICT,
                        "The idempotency key is bound to another control-plane target.",
                    )
                row = connection.execute(
                    "SELECT payload_json FROM records WHERE kind = ? AND record_id = ?",
                    (kind, str(existing["target_id"])),
                ).fetchone()
                if row is None:
                    raise DomainError(
                        ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                        "The idempotency binding references a missing record.",
                    )
                return str(existing["intent_digest"]), str(row["payload_json"]), False
            connection.execute(
                """
                INSERT INTO records(
                    kind, record_id, current_revision, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (kind, record_id, revision, payload_json, observed_at, observed_at),
            )
            if revision is not None:
                connection.execute(
                    """
                    INSERT INTO record_revisions(
                        kind, record_id, revision, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (kind, record_id, revision, payload_json, observed_at),
                )
            connection.execute(
                """
                INSERT INTO idempotency_keys(
                    scope, key_digest, intent_digest, target_kind, target_id, created_at
                ) VALUES (?, ?, ?, 'record', ?, ?)
                """,
                (kind, idempotency_digest, intent_digest, record_id, observed_at),
            )
        return intent_digest, payload_json, True

    def read_idempotent_record(
        self,
        *,
        kind: str,
        idempotency_digest: str,
    ) -> tuple[str, str] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT idempotency_keys.intent_digest, records.payload_json
                FROM idempotency_keys
                JOIN records
                  ON records.kind = idempotency_keys.scope
                 AND records.record_id = idempotency_keys.target_id
                WHERE idempotency_keys.scope = ?
                  AND idempotency_keys.key_digest = ?
                  AND idempotency_keys.target_kind = 'record'
                """,
                (kind, idempotency_digest),
            ).fetchone()
        if row is None:
            return None
        return str(row["intent_digest"]), str(row["payload_json"])

    def list_operations(self) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM operations ORDER BY created_at, operation_id"
            ).fetchall()
        return tuple(str(row["payload_json"]) for row in rows)

    def append_operation(
        self,
        *,
        operation_id: str,
        state: str,
        expected_revision: int,
        next_revision: int,
        payload_json: str,
    ) -> None:
        if next_revision != expected_revision + 1:
            raise DomainError(
                ErrorCode.REVISION_CONFLICT,
                "The next operation revision is not consecutive.",
            )
        observed_at = utc_now().isoformat()
        with self.transaction() as connection:
            changed = connection.execute(
                """
                UPDATE operations
                SET state = ?, revision = ?, payload_json = ?, updated_at = ?
                WHERE operation_id = ? AND revision = ?
                """,
                (
                    state,
                    next_revision,
                    payload_json,
                    observed_at,
                    operation_id,
                    expected_revision,
                ),
            ).rowcount
            if changed != 1:
                row = connection.execute(
                    "SELECT revision FROM operations WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                raise DomainError(
                    ErrorCode.REVISION_CONFLICT,
                    f"Operation {operation_id!r} changed before the update.",
                    retryable=True,
                    details={
                        "expected_revision": expected_revision,
                        "actual_revision": row["revision"] if row is not None else None,
                    },
                )
            connection.execute(
                """
                INSERT INTO operation_revisions(
                    operation_id, revision, payload_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (operation_id, next_revision, payload_json, observed_at),
            )
            # The current operation row and idempotency binding are retained for
            # reconnectability. Historical heartbeat/progress revisions are diagnostic
            # context, so retain the creation revision plus a bounded recent tail.
            connection.execute(
                """
                DELETE FROM operation_revisions
                WHERE operation_id = ? AND revision != 0 AND revision NOT IN (
                    SELECT revision FROM operation_revisions
                    WHERE operation_id = ?
                    ORDER BY revision DESC
                    LIMIT ?
                )
                """,
                (
                    operation_id,
                    operation_id,
                    self.MAX_OPERATION_REVISIONS - 1,
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @staticmethod
    def _purge_cursors(
        connection: sqlite3.Connection,
        *,
        observed_at: datetime,
        limit: int,
    ) -> int:
        if limit < 1:
            return 0
        return connection.execute(
            """
            DELETE FROM cursors
            WHERE cursor_digest IN (
                SELECT cursor_digest FROM cursors
                WHERE expires_at <= ? OR revoked_at IS NOT NULL
                ORDER BY expires_at, cursor_digest
                LIMIT ?
            )
            """,
            (observed_at.isoformat(), limit),
        ).rowcount

    @staticmethod
    def _replace_relationships(
        connection: sqlite3.Connection,
        *,
        source_kind: str,
        source_id: str,
        source_revision: int | None,
        relationships: tuple[ControlRelationship, ...],
        observed_at: str,
    ) -> None:
        connection.execute(
            "DELETE FROM relationships WHERE source_kind = ? AND source_id = ?",
            (source_kind, source_id),
        )
        connection.executemany(
            """
            INSERT INTO relationships(
                source_kind, source_id, relationship, target_kind, target_id,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    source_kind,
                    source_id,
                    item.relationship,
                    item.target_kind,
                    item.target_id,
                    item.payload_json,
                    observed_at,
                )
                for item in relationships
            ),
        )
        if source_revision is not None:
            connection.execute(
                """
                INSERT INTO record_revision_relationship_sets(
                    source_kind, source_id, source_revision, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (source_kind, source_id, source_revision, observed_at),
            )
            connection.executemany(
                """
                INSERT INTO record_revision_relationships(
                    source_kind, source_id, source_revision, relationship, target_kind,
                    target_id, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        source_kind,
                        source_id,
                        source_revision,
                        item.relationship,
                        item.target_kind,
                        item.target_id,
                        item.payload_json,
                        observed_at,
                    )
                    for item in relationships
                ),
            )

    @staticmethod
    def _require_run_projection_binding(
        spec: ProjectionIntentSpec,
        *,
        run_id: str,
        revision: int,
    ) -> None:
        if (
            spec.domain_kind != "run"
            or spec.domain_id != run_id
            or spec.domain_revision != revision
        ):
            raise DomainError(
                ErrorCode.INVALID_ARGUMENTS,
                "A run revision can only commit its own exact projection intent.",
            )

    def _insert_projection_intent(
        self,
        connection: sqlite3.Connection,
        spec: ProjectionIntentSpec,
        *,
        observed_at: str,
    ) -> bool:
        spec_json = canonical_json(spec.model_dump(mode="json"))
        if len(spec_json.encode("utf-8")) > self.MAX_PROJECTION_SPEC_BYTES:
            raise DomainError(
                ErrorCode.STORAGE_QUOTA_EXCEEDED,
                "Projection replay context exceeds the bounded control-plane limit.",
                details={"max_projection_spec_bytes": self.MAX_PROJECTION_SPEC_BYTES},
            )
        try:
            connection.execute(
                """
                INSERT INTO projection_intents(
                    intent_id, domain_kind, domain_id, domain_revision, domain_digest,
                    projection_kind, operation_digest, spec_json,
                    state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    spec.intent_id,
                    spec.domain_kind,
                    spec.domain_id,
                    spec.domain_revision,
                    spec.domain_digest,
                    spec.projection_kind,
                    spec.operation_digest,
                    spec_json,
                    observed_at,
                    observed_at,
                ),
            )
        except sqlite3.IntegrityError as exc:
            row = connection.execute(
                """
                SELECT spec_json FROM projection_intents
                WHERE intent_id = ? OR operation_digest = ? OR (
                    domain_kind = ? AND domain_id = ? AND domain_revision = ?
                    AND projection_kind = ?
                )
                LIMIT 1
                """,
                (
                    spec.intent_id,
                    spec.operation_digest,
                    spec.domain_kind,
                    spec.domain_id,
                    spec.domain_revision,
                    spec.projection_kind,
                ),
            ).fetchone()
            if row is not None and str(row["spec_json"]) == spec_json:
                return False
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "A projection identity is already bound to different immutable intent.",
            ) from exc
        return True

    @staticmethod
    def _projection_intent_row(
        connection: sqlite3.Connection,
        intent_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM projection_intents WHERE intent_id = ?",
            (intent_id,),
        ).fetchone()
        if row is None:
            raise DomainError(
                ErrorCode.RUN_NOT_FOUND,
                f"Projection intent {intent_id!r} does not exist.",
                details={"missing_entity": "projection_intent"},
            )
        return cast(sqlite3.Row, row)

    @staticmethod
    def _projection_intent(row: sqlite3.Row) -> ProjectionIntent:
        try:
            spec = ProjectionIntentSpec.model_validate_json(str(row["spec_json"]))
            return ProjectionIntent.model_validate(
                {
                    **spec.model_dump(mode="python"),
                    "state": row["state"],
                    "generation_id": row["generation_id"],
                    "corpus_commit_id": row["corpus_commit_id"],
                    "failure_code": row["failure_code"],
                    "failure_message": row["failure_message"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "The control plane contains an invalid projection intent.",
            ) from exc

    @staticmethod
    def _available_plan(row: sqlite3.Row | None) -> tuple[str, str]:
        if row is None:
            raise ControlPlane._unavailable_plan("unknown")
        if row["consumed_at"] is not None:
            raise ControlPlane._unavailable_plan("consumed")
        if datetime.fromisoformat(row["expires_at"]) <= utc_now():
            raise ControlPlane._unavailable_plan("expired")
        return str(row["intent_digest"]), str(row["payload_json"])

    @staticmethod
    def _unavailable_plan(
        state: Literal["unknown", "expired", "consumed"] = "consumed",
    ) -> DomainError:
        messages = {
            "unknown": "Authorized plan token is unknown.",
            "expired": "Authorized plan has expired.",
            "consumed": "Authorized plan has already been consumed.",
        }
        codes = {
            "unknown": ErrorCode.PLAN_TOKEN_UNKNOWN,
            "expired": ErrorCode.PLAN_TOKEN_EXPIRED,
            "consumed": ErrorCode.PLAN_TOKEN_CONSUMED,
        }
        return DomainError(
            codes[state],
            messages[state],
            remediation=("Create and review a new plan before executing.",),
        )


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
