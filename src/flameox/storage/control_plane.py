from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from flameox.domain import DomainError, ErrorCode
from flameox.domain.models import utc_now

if TYPE_CHECKING:
    from flameox.storage.workspace import Workspace


@dataclass(frozen=True, slots=True)
class ControlRelationship:
    relationship: str
    target_kind: str
    target_id: str
    payload_json: str = "{}"


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS control_plane_metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
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
)


class ControlPlane:
    """The transactional authority for mutable workspace control state."""

    SCHEMA_VERSION = 2

    def __init__(self, workspace: Workspace) -> None:
        self.path = workspace.paths.control_plane

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                current_version = self._current_schema_version(connection)
                if current_version not in {0, self.SCHEMA_VERSION}:
                    raise DomainError(
                        ErrorCode.WORKSPACE_INVALID,
                        "The SQLite control plane uses an incompatible schema. Create a new "
                        "workspace for this redesigned control plane.",
                        details={
                            "stored_schema_version": current_version,
                            "supported_schema_version": self.SCHEMA_VERSION,
                        },
                    )
                for statement in _SCHEMA:
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT INTO control_plane_metadata(key, value) VALUES('schema_version', ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (str(self.SCHEMA_VERSION),),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        with suppress(OSError):
            os.chmod(self.path, 0o600)

    @staticmethod
    def _current_schema_version(connection: sqlite3.Connection) -> int:
        metadata_exists = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'control_plane_metadata'
            """
        ).fetchone()
        if metadata_exists is None:
            return 0
        row = connection.execute(
            "SELECT value FROM control_plane_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            return 0
        try:
            version = int(row[0])
        except (TypeError, ValueError) as exc:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "The SQLite control-plane schema version is invalid.",
            ) from exc
        if version < 1:
            raise DomainError(
                ErrorCode.WORKSPACE_INVALID,
                "The SQLite control-plane schema version is invalid.",
            )
        return version

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
                    ErrorCode.INVALID_CAPTURE_PLAN,
                    "The authorized plan digest does not match the expected intent.",
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
        except sqlite3.IntegrityError as exc:
            raise DomainError(
                ErrorCode.ARTIFACT_INTEGRITY_FAILED,
                "An immutable run revision already contains different data.",
            ) from exc

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
    ) -> None:
        observed_at = utc_now().isoformat()
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO operations(
                        operation_id, kind, state, revision, payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (operation_id, kind, state, revision, payload_json, observed_at, observed_at),
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
    def _replace_relationships(
        connection: sqlite3.Connection,
        *,
        source_kind: str,
        source_id: str,
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

    @staticmethod
    def _available_plan(row: sqlite3.Row | None) -> tuple[str, str]:
        if row is None or row["consumed_at"] is not None:
            raise ControlPlane._unavailable_plan()
        if datetime.fromisoformat(row["expires_at"]) <= utc_now():
            raise ControlPlane._unavailable_plan()
        return str(row["intent_digest"]), str(row["payload_json"])

    @staticmethod
    def _unavailable_plan() -> DomainError:
        return DomainError(
            ErrorCode.INVALID_CAPTURE_PLAN,
            "Authorized plan is missing, expired, or already consumed.",
        )


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
