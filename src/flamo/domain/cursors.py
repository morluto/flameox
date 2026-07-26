from __future__ import annotations

import base64
import binascii
import json

from flamo.domain.errors import DomainError, ErrorCode
from flamo.domain.identity import digest_model

type CursorValue = str | int


class CursorCodec:
    """Integrity-check and bind keyset positions to one immutable query scope."""

    @staticmethod
    def encode(
        *,
        namespace: str,
        snapshot_id: str,
        scope_digest: str,
        position: tuple[CursorValue, ...],
    ) -> str:
        content = {
            "schema_version": 1,
            "namespace": namespace,
            "snapshot_id": snapshot_id,
            "scope_digest": scope_digest,
            "position": list(position),
        }
        envelope = {**content, "digest": digest_model(content)}
        return base64.urlsafe_b64encode(
            json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode()
        ).decode()

    @staticmethod
    def decode(
        cursor: str,
        *,
        namespace: str,
        snapshot_id: str,
        scope_digest: str,
    ) -> tuple[CursorValue, ...]:
        try:
            envelope = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
            position = envelope["position"]
            if not isinstance(position, list) or any(
                not isinstance(value, (str, int)) or isinstance(value, bool) for value in position
            ):
                raise ValueError("cursor position is invalid")
            content = {
                "schema_version": envelope["schema_version"],
                "namespace": envelope["namespace"],
                "snapshot_id": envelope["snapshot_id"],
                "scope_digest": envelope["scope_digest"],
                "position": position,
            }
            if envelope["digest"] != digest_model(content):
                raise ValueError("cursor digest mismatch")
        except (
            ValueError,
            KeyError,
            TypeError,
            UnicodeDecodeError,
            binascii.Error,
            json.JSONDecodeError,
        ) as exc:
            raise DomainError(ErrorCode.STALE_CURSOR, "Cursor is invalid.") from exc
        if (
            content["namespace"] != namespace
            or content["snapshot_id"] != snapshot_id
            or content["scope_digest"] != scope_digest
        ):
            raise DomainError(
                ErrorCode.STALE_CURSOR,
                "Cursor belongs to a different query or immutable snapshot.",
            )
        return tuple(position)
