from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime, timedelta

from flameox.domain.cursors import (
    CURSOR_POSITION_SPECS,
    CursorNamespace,
    CursorPosition,
    validate_cursor_position,
)
from flameox.domain.errors import DomainError, ErrorCode
from flameox.domain.models import utc_now
from flameox.storage.control_plane import ControlPlane, CursorControlRecord
from flameox.storage.locks import RETENTION_SHARED, WorkspaceLockResource
from flameox.storage.workspace import Workspace

_CURSOR_HANDLE = re.compile(r"[A-Za-z0-9_-]{43}", re.ASCII)


class CursorStore:
    """Issue and resolve opaque, disposable workspace-scoped continuation handles."""

    MAX_ENCODED_LENGTH = 64
    MAX_POSITION_BYTES = 2_048
    SCHEMA_VERSION = 1

    def __init__(
        self,
        workspace: Workspace,
        *,
        clock: Callable[[], datetime] = utc_now,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self.workspace = workspace
        self.control = ControlPlane(workspace)
        self._clock = clock
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(32))

    def issue(
        self,
        *,
        namespace: CursorNamespace,
        snapshot_id: str,
        scope_digest: str,
        position: object,
    ) -> str:
        self._validate_binding(snapshot_id=snapshot_id, scope_digest=scope_digest)
        validated = validate_cursor_position(namespace, position)
        position_json = json.dumps(
            validated,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(position_json.encode()) > self.MAX_POSITION_BYTES:
            raise DomainError(ErrorCode.STALE_CURSOR, "Cursor position is invalid.")
        created_at = self._clock()
        spec = CURSOR_POSITION_SPECS[namespace]
        expires_at = created_at + timedelta(seconds=spec.max_age_seconds)
        workspace_id = self.workspace.identity.workspace_id
        for _ in range(3):
            token = self._token_factory()
            if not self._valid_handle(token):
                raise DomainError(
                    ErrorCode.INTERNAL_ERROR,
                    "The cursor handle generator returned an invalid token.",
                )
            try:
                retention = (
                    nullcontext()
                    if self.workspace.lock_manager.holds(WorkspaceLockResource.RETENTION)
                    else self.workspace.locked(
                        RETENTION_SHARED,
                        phase="cursor snapshot retention",
                    )
                )
                with retention:
                    self.control.issue_cursor(
                        cursor_digest=self._digest(token),
                        workspace_id=workspace_id,
                        namespace=namespace.value,
                        snapshot_id=snapshot_id,
                        scope_digest=scope_digest,
                        position_json=position_json,
                        created_at=created_at,
                        expires_at=expires_at,
                    )
            except DomainError as error:
                if error.code is ErrorCode.REVISION_CONFLICT:
                    continue
                raise
            return token
        raise DomainError(
            ErrorCode.INTERNAL_ERROR,
            "Unable to allocate a unique cursor handle.",
        )

    def retained_corpus_commit_ids(self) -> tuple[str, ...]:
        return self.control.active_cursor_snapshot_ids(
            workspace_id=self.workspace.identity.workspace_id,
            namespace=CursorNamespace.EXECUTION_ANALYSIS.value,
            observed_at=self._clock(),
        )

    def resolve(
        self,
        cursor: str,
        *,
        namespace: CursorNamespace,
        snapshot_id: str,
        scope_digest: str,
    ) -> CursorPosition:
        self._validate_binding(snapshot_id=snapshot_id, scope_digest=scope_digest)
        record, position = self._resolve_binding(
            cursor, namespace=namespace, scope_digest=scope_digest
        )
        if record.snapshot_id != snapshot_id:
            raise DomainError(
                ErrorCode.STALE_CURSOR,
                "Cursor belongs to a different query or immutable snapshot.",
            )
        return position

    def resolve_bound(
        self,
        cursor: str,
        *,
        namespace: CursorNamespace,
        scope_digest: str,
    ) -> tuple[str, CursorPosition]:
        """Resolve an opaque cursor together with its immutable snapshot binding."""

        record, position = self._resolve_binding(
            cursor, namespace=namespace, scope_digest=scope_digest
        )
        return record.snapshot_id, position

    def _resolve_binding(
        self,
        cursor: str,
        *,
        namespace: CursorNamespace,
        scope_digest: str,
    ) -> tuple[CursorControlRecord, CursorPosition]:
        record = self._active_record(cursor)
        if record.namespace != namespace.value or record.scope_digest != scope_digest:
            raise DomainError(
                ErrorCode.STALE_CURSOR,
                "Cursor belongs to a different query or immutable snapshot.",
            )
        try:
            payload = json.loads(record.position_json)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise self._unavailable() from error
        return record, validate_cursor_position(namespace, payload)

    def namespace(self, cursor: str) -> CursorNamespace:
        """Return routing identity for one active opaque cursor."""
        record = self._active_record(cursor)
        try:
            return CursorNamespace(record.namespace)
        except ValueError as error:
            raise self._unavailable() from error

    def _active_record(self, cursor: str) -> CursorControlRecord:
        self._require_handle(cursor)
        record = self.control.read_cursor(
            cursor_digest=self._digest(cursor),
            workspace_id=self.workspace.identity.workspace_id,
        )
        observed_at = self._clock()
        if (
            record is None
            or record.schema_version != self.SCHEMA_VERSION
            or record.revoked_at is not None
            or record.created_at >= record.expires_at
            or record.expires_at <= observed_at
        ):
            raise self._unavailable()
        return record

    def revoke(self, cursor: str) -> bool:
        self._require_handle(cursor)
        return self.control.revoke_cursor(
            cursor_digest=self._digest(cursor),
            workspace_id=self.workspace.identity.workspace_id,
            revoked_at=self._clock(),
        )

    def purge_expired(self, *, limit: int = 128) -> int:
        return self.control.purge_cursors(observed_at=self._clock(), limit=limit)

    @classmethod
    def _require_handle(cls, cursor: str) -> None:
        if not cls._valid_handle(cursor):
            raise cls._unavailable()

    @classmethod
    def _valid_handle(cls, cursor: object) -> bool:
        return (
            isinstance(cursor, str)
            and len(cursor) <= cls.MAX_ENCODED_LENGTH
            and _CURSOR_HANDLE.fullmatch(cursor) is not None
        )

    @staticmethod
    def _validate_binding(*, snapshot_id: str, scope_digest: str) -> None:
        if not snapshot_id or len(snapshot_id) > 512 or not scope_digest or len(scope_digest) > 512:
            raise DomainError(ErrorCode.STALE_CURSOR, "Cursor query binding is invalid.")

    @staticmethod
    def _digest(cursor: str) -> str:
        # This is an index for a high-entropy bearer handle, not a signature or MAC.
        return hashlib.sha256(cursor.encode()).hexdigest()

    @staticmethod
    def _unavailable() -> DomainError:
        return DomainError(
            ErrorCode.STALE_CURSOR,
            "Cursor is missing, expired, revoked, or invalid.",
        )
